from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..registry import TRAINERS
from ..utils.metrics import psnr, ssim
from ..utils.val_images import save_val_from_loader


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


@TRAINERS.register("gan_denoise")
class GANDenoiseTrainer:
    def __init__(self, device: str = "cuda", lambda_rec: float = 100.0):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lambda_rec = float(lambda_rec)
        self.adv_criterion = nn.BCELoss()

    def _save_ckpt(
        self,
        path: Path,
        *,
        G: nn.Module,
        D: nn.Module,
        opt_G: torch.optim.Optimizer,
        opt_D: torch.optim.Optimizer,
        scheduler_G: Any | None,
        scheduler_D: Any | None,
        scaler: torch.amp.GradScaler | None,
        metrics: Dict[str, Any],
        best_metric: float,
        bad_epochs: int,
    ) -> None:
        ckpt = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "optimizer_G": opt_G.state_dict(),
            "optimizer_D": opt_D.state_dict(),
            "scheduler_G": scheduler_G.state_dict() if scheduler_G is not None else None,
            "scheduler_D": scheduler_D.state_dict() if scheduler_D is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
            "best_metric": float(best_metric),
            "bad_epochs": int(bad_epochs),
            "epoch": _safe_int(metrics.get("epoch", 0), 0),
        }
        torch.save(ckpt, path)

    def _maybe_resume(
        self,
        resume_from: str | Path | None,
        *,
        G: nn.Module,
        D: nn.Module,
        opt_G: torch.optim.Optimizer,
        opt_D: torch.optim.Optimizer,
        scheduler_G: Any | None,
        scheduler_D: Any | None,
        scaler: torch.amp.GradScaler | None,
        mode: str,
        logger,
    ) -> Tuple[int, float, int]:
        if resume_from is None:
            best_metric = -float("inf") if mode == "max" else float("inf")
            return 0, best_metric, 0

        resume_path = Path(resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoints" / "last.pt"

        if not resume_path.exists():
            logger.log(f"[WARN] resume_from not found: {resume_path} (start from scratch)")
            best_metric = -float("inf") if mode == "max" else float("inf")
            return 0, best_metric, 0

        obj = torch.load(resume_path, map_location="cpu")

        if "G" in obj:
            G.load_state_dict(obj["G"])
        if "D" in obj:
            D.load_state_dict(obj["D"])

        if "optimizer_G" in obj and obj["optimizer_G"] is not None:
            try:
                opt_G.load_state_dict(obj["optimizer_G"])
            except Exception as e:
                logger.log(f"[WARN] optimizer_G resume failed: {e}")
        if "optimizer_D" in obj and obj["optimizer_D"] is not None:
            try:
                opt_D.load_state_dict(obj["optimizer_D"])
            except Exception as e:
                logger.log(f"[WARN] optimizer_D resume failed: {e}")

        if scheduler_G is not None and "scheduler_G" in obj and obj["scheduler_G"] is not None:
            try:
                scheduler_G.load_state_dict(obj["scheduler_G"])
            except Exception as e:
                logger.log(f"[WARN] scheduler_G resume failed: {e}")
        if scheduler_D is not None and "scheduler_D" in obj and obj["scheduler_D"] is not None:
            try:
                scheduler_D.load_state_dict(obj["scheduler_D"])
            except Exception as e:
                logger.log(f"[WARN] scheduler_D resume failed: {e}")

        if scaler is not None and "scaler" in obj and obj["scaler"] is not None:
            try:
                scaler.load_state_dict(obj["scaler"])
            except Exception as e:
                logger.log(f"[WARN] scaler resume failed: {e}")

        start_epoch = _safe_int(obj.get("epoch", obj.get("metrics", {}).get("epoch", 0)), 0)
        best_metric = float(obj.get("best_metric", -float("inf") if mode == "max" else float("inf")))
        bad_epochs = _safe_int(obj.get("bad_epochs", 0), 0)

        logger.log(f"Resumed from: {resume_path} (epoch={start_epoch}, best_metric={best_metric}, bad_epochs={bad_epochs})")
        return start_epoch, best_metric, bad_epochs

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

                if is_train:
                    opt_D.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self.device.type, enabled=use_amp):
                        fake_img = G(noisy)
                        if isinstance(fake_img, (tuple, list)):
                            fake_img = fake_img[0]
                        fake_img = fake_img.detach()

                        pred_real = D(noisy, clean)
                        pred_fake = D(noisy, fake_img)

                        valid = torch.ones_like(pred_real)
                        fake = torch.zeros_like(pred_fake)

                        loss_real = self.adv_criterion(pred_real, valid)
                        loss_fake = self.adv_criterion(pred_fake, fake)
                        loss_D = 0.5 * (loss_real + loss_fake)

                    if use_amp and scaler is not None:
                        scaler.scale(loss_D).backward()
                        scaler.step(opt_D)
                    else:
                        loss_D.backward()
                        opt_D.step()
                else:
                    loss_D = torch.tensor(0.0, device=self.device)

                if is_train:
                    opt_G.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self.device.type, enabled=use_amp):
                    fake_img = G(noisy)
                    if isinstance(fake_img, (tuple, list)):
                        fake_img = fake_img[0]

                    pred_fake = D(noisy, fake_img)
                    valid = torch.ones_like(pred_fake)

                    loss_adv = self.adv_criterion(pred_fake, valid)
                    loss_rec = rec_criterion(fake_img, clean)
                    loss_G = loss_adv + self.lambda_rec * loss_rec

                if is_train:
                    if use_amp and scaler is not None:
                        scaler.scale(loss_G).backward()
                        scaler.step(opt_G)
                        scaler.update()
                    else:
                        loss_G.backward()
                        opt_G.step()

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
        save_val_images: bool = True,
        val_image_every: int = 1,
        val_image_max: int = 8,
        resume_from: str | Path | None = None,
    ):
        G.to(self.device)
        D.to(self.device)

        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if self.device.type == "cuda" else None

        start_epoch, best, bad_epochs = self._maybe_resume(
            resume_from,
            G=G,
            D=D,
            opt_G=opt_G,
            opt_D=opt_D,
            scheduler_G=scheduler_G,
            scheduler_D=scheduler_D,
            scaler=scaler,
            mode=mode,
            logger=logger,
        )

        best_path = ckpt_dir / "best.pt"
        last_path = ckpt_dir / "last.pt"

        for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc="Epochs"):
            t0 = time.time()

            tr = self._run_epoch(
                G,
                D,
                train_loader,
                opt_G=opt_G,
                opt_D=opt_D,
                rec_criterion=rec_criterion,
                use_amp=use_amp,
                scaler=scaler,
            )

            va = self._run_epoch(
                G,
                D,
                val_loader,
                opt_G=None,
                opt_D=None,
                rec_criterion=rec_criterion,
                use_amp=False,
                scaler=None,
            )

            if save_val_images and (epoch % int(val_image_every) == 0):
                try:
                    out_dir = save_val_from_loader(
                        forward_fn=lambda x: G(x),
                        val_loader=val_loader,
                        device=self.device,
                        run_dir=run_dir,
                        epoch=epoch,
                        max_items=int(val_image_max),
                        prefix="val",
                    )
                    logger.log(f"Saved val images -> {out_dir}")
                except Exception as e:
                    logger.log(f"[WARN] Save val images failed: {e}")

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

            cur = metrics.get(save_best)
            if cur is None:
                cur = metrics.get("val_psnr")
            improved = (cur > best) if mode == "max" else (cur < best)

            if improved:
                best = float(cur)
                bad_epochs = 0
                self._save_ckpt(
                    best_path,
                    G=G,
                    D=D,
                    opt_G=opt_G,
                    opt_D=opt_D,
                    scheduler_G=scheduler_G,
                    scheduler_D=scheduler_D,
                    scaler=scaler,
                    metrics=metrics,
                    best_metric=best,
                    bad_epochs=bad_epochs,
                )
                logger.log(f"✓ Saved best GAN checkpoint ({save_best}={best:.4f})")
            else:
                bad_epochs += 1

            self._save_ckpt(
                last_path,
                G=G,
                D=D,
                opt_G=opt_G,
                opt_D=opt_D,
                scheduler_G=scheduler_G,
                scheduler_D=scheduler_D,
                scaler=scaler,
                metrics=metrics,
                best_metric=best,
                bad_epochs=bad_epochs,
            )

            if early_stopping_patience and bad_epochs >= int(early_stopping_patience):
                logger.log("Early stopping triggered.")
                break

        return {"best_metric": best, "best_path": str(best_path), "last_path": str(last_path)}
