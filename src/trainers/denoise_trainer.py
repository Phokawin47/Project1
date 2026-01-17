from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..registry import TRAINERS
from ..utils.metrics import psnr, ssim
from ..utils.val_images import save_val_from_loader


def _device_type(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


@TRAINERS.register("denoise")
class DenoiseTrainer:
    """
    Trainer that supports:
      - AMP (torch.amp.GradScaler)
      - Save/Resume (model + optimizer + scheduler + scaler + best + bad_epochs)
      - Optional val triplet image saving (Original | Addnoise | Denoise)

    Note: Resume is opt-in via resume_from=...
    """
    def __init__(self, device: str = "cuda", lambda_rec: float | None = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # IO: checkpoint save/load
    # ----------------------------
    def _save_ckpt(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler: torch.amp.GradScaler | None,
        metrics: Dict[str, Any],
        best_metric: float,
        bad_epochs: int,
    ) -> None:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
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
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler: torch.amp.GradScaler | None,
        mode: str,
        logger,
    ) -> Tuple[int, float, int]:
        """
        Returns: (start_epoch, best_metric, bad_epochs)
        start_epoch = last completed epoch (so training continues at start_epoch+1)
        """
        if resume_from is None:
            best_metric = float("inf") if mode == "min" else -float("inf")
            return 0, best_metric, 0

        resume_path = Path(resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoints" / "last.pt"

        if not resume_path.exists():
            logger.log(f"[WARN] resume_from not found: {resume_path} (start from scratch)")
            best_metric = float("inf") if mode == "min" else -float("inf")
            return 0, best_metric, 0

        obj = torch.load(resume_path, map_location="cpu")

        if "model" in obj:
            model.load_state_dict(obj["model"])

        if "optimizer" in obj and obj["optimizer"] is not None:
            try:
                optimizer.load_state_dict(obj["optimizer"])
            except Exception as e:
                logger.log(f"[WARN] optimizer resume failed: {e}")

        if scheduler is not None and "scheduler" in obj and obj["scheduler"] is not None:
            try:
                scheduler.load_state_dict(obj["scheduler"])
            except Exception as e:
                logger.log(f"[WARN] scheduler resume failed: {e}")

        if scaler is not None and "scaler" in obj and obj["scaler"] is not None:
            try:
                scaler.load_state_dict(obj["scaler"])
            except Exception as e:
                logger.log(f"[WARN] scaler resume failed: {e}")

        start_epoch = _safe_int(obj.get("epoch", obj.get("metrics", {}).get("epoch", 0)), 0)
        best_metric = float(obj.get("best_metric", float("inf") if mode == "min" else -float("inf")))
        bad_epochs = _safe_int(obj.get("bad_epochs", 0), 0)

        logger.log(f"Resumed from: {resume_path} (epoch={start_epoch}, best_metric={best_metric}, bad_epochs={bad_epochs})")
        return start_epoch, best_metric, bad_epochs

    # ----------------------------
    # Train/Val epoch
    # ----------------------------
    def _run_epoch(self, model, loader, optimizer=None, criterion=None, use_amp: bool = False, scaler=None):
        is_train = optimizer is not None
        model.train(is_train)

        total_loss = total_psnr = total_ssim = 0.0
        n = 0

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        desc = "Train" if is_train else "Val"

        dev_type = _device_type(self.device)

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

                with torch.autocast(device_type=dev_type, enabled=use_amp and is_train):
                    pred = model(noisy)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    loss = criterion(pred, clean)

                if is_train:
                    if use_amp and scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                bs = noisy.size(0)
                total_loss += float(loss.item()) * bs
                pred_c = pred.clamp(0.0, 1.0)
                clean_c = clean.clamp(0.0, 1.0)
                total_psnr += psnr(pred_c, clean_c) * bs
                total_ssim += ssim(pred_c, clean_c) * bs
                n += bs

        return {"loss": total_loss / max(1, n), "psnr": total_psnr / max(1, n), "ssim": total_ssim / max(1, n)}

    # ----------------------------
    # Full training loop
    # ----------------------------
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
        # val images
        save_val_images: bool = True,
        val_image_every: int = 1,
        val_image_max: int = 8,
        # resume
        resume_from: str | Path | None = None,
    ):
        model = model.to(self.device)
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if self.device.type == "cuda" else None

        start_epoch, best, bad_epochs = self._maybe_resume(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            mode=mode,
            logger=logger,
        )

        best_path = ckpt_dir / "best.pt"
        last_path = ckpt_dir / "last.pt"

        for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc="Epochs"):
            t0 = time.time()

            tr = self._run_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                criterion=criterion,
                use_amp=use_amp,
                scaler=scaler,
            )
            va = self._run_epoch(
                model,
                val_loader,
                optimizer=None,
                criterion=criterion,
                use_amp=False,
                scaler=None,
            )

            # save val images
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

            dt = time.time() - t0

            if scheduler is not None:
                scheduler.step()

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
                best = float(cur)
                bad_epochs = 0
                self._save_ckpt(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metrics=metrics,
                    best_metric=best,
                    bad_epochs=bad_epochs,
                )
                logger.log(f"✓ Saved best checkpoint: {best_path.name} ({key}={best:.6f})")
            else:
                bad_epochs += 1

            # always save last
            self._save_ckpt(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                metrics=metrics,
                best_metric=best,
                bad_epochs=bad_epochs,
            )

            if early_stopping_patience is not None and bad_epochs >= int(early_stopping_patience):
                logger.log(f"Early stopping triggered (patience={early_stopping_patience}).")
                break

        return {"best_metric": best, "best_path": str(best_path), "last_path": str(last_path)}
