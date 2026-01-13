from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..registry import TRAINERS
from ..utils.metrics import psnr, ssim
from ..utils.val_images import save_val_from_loader


@TRAINERS.register("armnet_denoise")
class ARMNetDenoiseTrainer:
    def __init__(self, device="cuda", lambda_rec: float | None = None):
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, fallback to CPU")
            device = "cpu"

        self.lambda_rec = lambda_rec
        self.device = torch.device(device)

    # -------------------------------------------------
    # One epoch (train or validation)
    # -------------------------------------------------
    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        use_amp: bool = False,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> Dict[str, float]:

        is_train = optimizer is not None
        model.train(is_train)

        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        desc = "Train" if is_train else "Val"

        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        with grad_ctx:
            for batch in tqdm(loader, desc=desc, leave=False):
                # -------------------------------------------------
                # Unpack batch
                # -------------------------------------------------
                if len(batch) == 3:
                    noisy, clean, _ = batch
                else:
                    noisy, clean = batch

                noisy = noisy.to(self.device, non_blocking=True)
                clean = clean.to(self.device, non_blocking=True)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                # -------------------------------------------------
                # Forward
                # -------------------------------------------------
                with torch.autocast(device_type=device_type, enabled=use_amp and is_train):
                    pred = model(noisy)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    loss = criterion(pred, clean)

                # -------------------------------------------------
                # Backward
                # -------------------------------------------------
                if is_train:
                    if use_amp and scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                # -------------------------------------------------
                # Metrics (image domain)
                # -------------------------------------------------
                bs = noisy.size(0)
                total_loss += loss.item() * bs

                pred_c = pred.clamp(0.0, 1.0)
                clean_c = clean.clamp(0.0, 1.0)

                total_psnr += psnr(pred_c, clean_c) * bs
                total_ssim += ssim(pred_c, clean_c) * bs

                num_samples += bs

        return {
            "loss": total_loss / max(1, num_samples),
            "psnr": total_psnr / max(1, num_samples),
            "ssim": total_ssim / max(1, num_samples),
        }

    # -------------------------------------------------
    # Full training loop
    # -------------------------------------------------
    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        run_dir: Path,
        logger,
        scheduler: Any | None = None,
        use_amp: bool = False,
        save_best: str = "val_loss",
        mode: str = "min",
        early_stopping_patience: int | None = None,
        # --- NEW: save val triplet images ---
        save_val_images: bool = True,
        val_image_every: int = 1,
        val_image_max: int = 8,
    ) -> Dict[str, Any]:

        model = model.to(self.device)

        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


        best_metric = float("inf") if mode == "min" else -float("inf")
        bad_epochs = 0
        best_ckpt = ckpt_dir / "best.pt"

        for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
            start_time = time.time()

            train_metrics = self._run_epoch(
                model,
                train_loader,
                criterion,
                optimizer=optimizer,
                use_amp=use_amp,
                scaler=scaler,
            )

            val_metrics = self._run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                use_amp=False,
                scaler=None,
            )

            # --- save val images: Original | Addnoise | Denoise ---
            if save_val_images and (epoch % int(val_image_every) == 0):
                try:
                    out_dir = save_val_from_loader(
                        forward_fn=lambda x: model(x),
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

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]

            metrics = {
                "epoch": epoch,
                "lr": lr,
                "time_sec": round(elapsed, 2),
                "train_loss": train_metrics["loss"],
                "train_psnr": train_metrics["psnr"],
                "train_ssim": train_metrics["ssim"],
                "val_loss": val_metrics["loss"],
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
            }

            logger.log(
                f"Epoch {epoch:03d}/{epochs} | lr={lr:.6g} | "
                f"train_loss={metrics['train_loss']:.6f} "
                f"psnr={metrics['train_psnr']:.3f} "
                f"ssim={metrics['train_ssim']:.4f} | "
                f"val_loss={metrics['val_loss']:.6f} "
                f"psnr={metrics['val_psnr']:.3f} "
                f"ssim={metrics['val_ssim']:.4f} | "
                f"{elapsed:.1f}s"
            )

            logger.log_metrics(metrics)

            key_val = metrics.get(save_best, metrics["val_loss"])
            improved = (key_val < best_metric) if mode == "min" else (key_val > best_metric)

            if improved:
                best_metric = key_val
                bad_epochs = 0
                torch.save(
                    {"model": model.state_dict(), "metrics": metrics},
                    best_ckpt,
                )
                logger.log(f"✓ Best checkpoint saved ({save_best}={best_metric:.6f})")
            else:
                bad_epochs += 1

            torch.save(
                {"model": model.state_dict(), "metrics": metrics},
                ckpt_dir / "last.pt",
            )

            if early_stopping_patience is not None and bad_epochs >= early_stopping_patience:
                logger.log("Early stopping triggered.")
                break

        return {
            "best_metric": best_metric,
            "best_checkpoint": str(best_ckpt),
        }
