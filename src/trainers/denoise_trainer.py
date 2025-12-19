from __future__ import annotations
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..registry import TRAINERS
from ..utils.metrics import psnr, ssim

@TRAINERS.register("denoise")
class DenoiseTrainer:
    def __init__(
        self,
        device: str = "cuda",
        lambda_rec: float | None = None,  # <<< เพิ่มบรรทัดนี้
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")


    def _run_epoch(self, model, loader, optimizer=None, criterion=None, use_amp: bool = False, scaler=None):
        is_train = optimizer is not None
        model.train(is_train)

        total_loss = total_psnr = total_ssim = 0.0
        n = 0

        # ✅ สำคัญ: ปิด grad ตอน val
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

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                # (จะใช้ amp กับ val ด้วยก็ได้)
                with torch.autocast(device_type=str(self.device).split(":")[0], enabled=use_amp):
                    pred = model(noisy)
                    loss = criterion(pred, clean)

                if is_train:
                    if use_amp and scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                # metrics
                total_loss += float(loss.item()) * noisy.size(0)
                pred_c = pred.clamp(0.0, 1.0)
                clean_c = clean.clamp(0.0, 1.0)
                total_psnr += psnr(pred_c, clean_c) * noisy.size(0)
                total_ssim += ssim(pred_c, clean_c) * noisy.size(0)
                n += noisy.size(0)

        return {"loss": total_loss/max(1,n), "psnr": total_psnr/max(1,n), "ssim": total_ssim/max(1,n)}


    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        criterion,
        epochs: int,
        run_dir: Path,
        logger,
        scheduler=None,
        use_amp: bool = False,
        save_best: str = "val_loss",
        mode: str = "min",
        early_stopping_patience: int | None = None,
    ):
        model = model.to(self.device)
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best = float("inf") if mode == "min" else -float("inf")
        bad_epochs = 0
        best_path = ckpt_dir / "best.pt"

        for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
            t0 = time.time()
            tr = self._run_epoch(model, train_loader, optimizer=optimizer, criterion=criterion, use_amp=use_amp, scaler=scaler)
            va = self._run_epoch(model, val_loader, optimizer=None, criterion=criterion, use_amp=False, scaler=None)
            dt = time.time() - t0

            # scheduler step at end of epoch (common practice)
            if scheduler is not None:
                scheduler.step()

            # current lr (first param group)
            cur_lr = float(optimizer.param_groups[0]["lr"])

            metrics = {
                "epoch": epoch,
                "lr": cur_lr,
                "time_sec": round(dt, 3),
                "train_loss": tr["loss"],
                "train_psnr": tr["psnr"],
                "train_ssim": tr["ssim"],
                "val_loss": va["loss"],
                "val_psnr": va["psnr"],
                "val_ssim": va["ssim"],
            }

            logger.log(
                f"Epoch {epoch:03d}/{epochs} | lr={cur_lr:.6g} | "
                f"train_loss={metrics['train_loss']:.6f} psnr={metrics['train_psnr']:.3f} ssim={metrics['train_ssim']:.4f} | "
                f"val_loss={metrics['val_loss']:.6f} psnr={metrics['val_psnr']:.3f} ssim={metrics['val_ssim']:.4f} | "
                f"{dt:.1f}s"
            )
            logger.log_metrics(metrics)

            key = save_best
            cur = metrics.get(key, metrics["val_loss"])
            improved = (cur < best) if mode == "min" else (cur > best)

            if improved:
                best = cur
                bad_epochs = 0
                torch.save({"model": model.state_dict(), "metrics": metrics}, best_path)
                logger.log(f"✓ Saved best checkpoint: {best_path.name} ({key}={best:.6f})")
            else:
                bad_epochs += 1

            torch.save({"model": model.state_dict(), "metrics": metrics}, ckpt_dir / "last.pt")

            if early_stopping_patience is not None and bad_epochs >= early_stopping_patience:
                logger.log(f"Early stopping triggered (patience={early_stopping_patience}).")
                break

        return {"best_metric": best, "best_path": str(best_path)}
