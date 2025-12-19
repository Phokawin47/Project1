from __future__ import annotations
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..registry import TRAINERS
from ..utils.metrics import psnr, ssim


@TRAINERS.register("gan_denoise")
class GANDenoiseTrainer:
    def __init__(
        self,
        device: str = "cuda",
        lambda_rec: float = 100.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lambda_rec = lambda_rec
        self.adv_criterion = nn.BCELoss()

    def _run_epoch(
        self,
        G: nn.Module,
        D: nn.Module,
        loader: DataLoader,
        opt_G=None,
        opt_D=None,
        rec_criterion=None,
        use_amp: bool = False,
        scaler=None,
    ):
        is_train = opt_G is not None and opt_D is not None
        G.train(is_train)
        D.train(is_train)

        total_g = total_d = total_psnr = total_ssim = 0.0
        n = 0

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        desc = "Train" if is_train else "Val"

        with grad_ctx:
            for batch in tqdm(loader, desc=desc, leave=False):
                if len(batch) == 3:
                    noisy, clean, _ = batch
                else:
                    noisy, clean = batch

                noisy = noisy.to(self.device, non_blocking=True)
                clean = clean.to(self.device, non_blocking=True)
                bs = noisy.size(0)

                # ============================
                #  Train Discriminator
                # ============================
                if is_train:
                    opt_D.zero_grad(set_to_none=True)

                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=use_amp,
                    ):
                        fake_img = G(noisy).detach()

                        pred_real = D(noisy, clean)
                        pred_fake = D(noisy, fake_img)

                        valid = torch.ones_like(pred_real)
                        fake = torch.zeros_like(pred_fake)

                        loss_real = self.adv_criterion(pred_real, valid)
                        loss_fake = self.adv_criterion(pred_fake, fake)
                        loss_D = 0.5 * (loss_real + loss_fake)

                    if use_amp:
                        scaler.scale(loss_D).backward()
                        scaler.step(opt_D)
                    else:
                        loss_D.backward()
                        opt_D.step()
                else:
                    loss_D = torch.tensor(0.0, device=self.device)

                # ============================
                #  Train Generator
                # ============================
                if is_train:
                    opt_G.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=use_amp,
                ):
                    fake_img = G(noisy)
                    pred_fake = D(noisy, fake_img)

                    valid = torch.ones_like(pred_fake)

                    loss_adv = self.adv_criterion(pred_fake, valid)
                    loss_rec = rec_criterion(fake_img, clean)
                    loss_G = loss_adv + self.lambda_rec * loss_rec

                if is_train:
                    if use_amp:
                        scaler.scale(loss_G).backward()
                        scaler.step(opt_G)
                        scaler.update()
                    else:
                        loss_G.backward()
                        opt_G.step()

                # ============================
                #  Metrics
                # ============================
                total_g += float(loss_G.item()) * bs
                total_d += float(loss_D.item()) * bs

                fake_c = fake_img.clamp(0.0, 1.0)
                clean_c = clean.clamp(0.0, 1.0)

                total_psnr += psnr(fake_c, clean_c) * bs
                total_ssim += ssim(fake_c, clean_c) * bs
                n += bs

        return {
            "g_loss": total_g / max(1, n),
            "d_loss": total_d / max(1, n),
            "psnr": total_psnr / max(1, n),
            "ssim": total_ssim / max(1, n),
        }

    def fit(
        self,
        G: nn.Module,
        D: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        opt_G,
        opt_D,
        rec_criterion,
        epochs: int,
        run_dir: Path,
        logger,
        scheduler_G=None,
        scheduler_D=None,
        use_amp: bool = False,
        save_best: str = "val_psnr",
        mode: str = "max",
        early_stopping_patience: int | None = None,
    ):
        G.to(self.device)
        D.to(self.device)

        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best = -float("inf") if mode == "max" else float("inf")
        bad_epochs = 0
        best_path = ckpt_dir / "best.pt"

        for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
            t0 = time.time()

            tr = self._run_epoch(
                G, D, train_loader,
                opt_G=opt_G,
                opt_D=opt_D,
                rec_criterion=rec_criterion,
                use_amp=use_amp,
                scaler=scaler,
            )

            va = self._run_epoch(
                G, D, val_loader,
                opt_G=None,
                opt_D=None,
                rec_criterion=rec_criterion,
                use_amp=False,
                scaler=None,
            )

            if scheduler_G is not None:
                scheduler_G.step()
            if scheduler_D is not None:
                scheduler_D.step()

            dt = time.time() - t0
            cur_lr = float(opt_G.param_groups[0]["lr"])

            metrics = {
                "epoch": epoch,
                "lr": cur_lr,
                "time_sec": round(dt, 3),
                "train_g_loss": tr["g_loss"],
                "train_d_loss": tr["d_loss"],
                "train_psnr": tr["psnr"],
                "train_ssim": tr["ssim"],
                "val_g_loss": va["g_loss"],
                "val_psnr": va["psnr"],
                "val_ssim": va["ssim"],
            }

            logger.log(
                f"Epoch {epoch:03d}/{epochs} | lr={cur_lr:.6g} | "
                f"G={metrics['train_g_loss']:.4f} "
                f"D={metrics['train_d_loss']:.4f} | "
                f"PSNR={metrics['val_psnr']:.3f} "
                f"SSIM={metrics['val_ssim']:.4f} | "
                f"{dt:.1f}s"
            )
            logger.log_metrics(metrics)

            cur = metrics[save_best]
            improved = (cur > best) if mode == "max" else (cur < best)

            if improved:
                best = cur
                bad_epochs = 0
                torch.save(
                    {
                        "G": G.state_dict(),
                        "D": D.state_dict(),
                        "metrics": metrics,
                    },
                    best_path,
                )
                logger.log(f"✓ Saved best GAN checkpoint ({save_best}={best:.4f})")
            else:
                bad_epochs += 1

            torch.save(
                {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "metrics": metrics,
                },
                ckpt_dir / "last.pt",
            )

            if early_stopping_patience and bad_epochs >= early_stopping_patience:
                logger.log("Early stopping triggered.")
                break

        return {"best_metric": best, "best_path": str(best_path)}
