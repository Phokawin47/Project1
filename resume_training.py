from __future__ import annotations

"""
resume_training.py (stateful)
-----------------------------
Resume training from an existing run directory (runs/<model>/<tag>/) using the SAME config.final.json,
loading weights from checkpoints/best.pt or checkpoints/last.pt (or any checkpoint path).

This version supports **exact resume** when your checkpoint contains:
- model weights
- optimizer state
- scheduler state
- AMP GradScaler state

It remains backward compatible with older checkpoints that only have model weights:
- It will rebuild optimizer/scheduler from config and "fast-forward" the scheduler to approximate continuity.

Outputs:
- Continues saving checkpoints into the SAME run_dir/checkpoints (best.pt, last.pt)
- Writes resume logs to:
    run_dir/train.resume.log.txt
    run_dir/metrics.resume.jsonl
- Also writes final test artifacts (if test_args exists) to:
    run_dir/test_metrics.json
    run_dir/metrics.test.jsonl

Usage examples:
  python resume_training.py --run_dir runs/unet/20260113_195930_Unet_schedule_66b4934c --ckpt last --max_minutes 175
  python resume_training.py --run_dir runs/dncnn/20260117_002549_dncnn_schedule_7e17edcc --ckpt best
"""

import argparse
import json
import time
import inspect
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader

from src.config import load_config, deep_update, parse_overrides
from src.registry import MODELS, DATASETS, TRAINERS
from src.utils.seed import set_seed
from src.utils.logger import TextLogger
from src.optim_schedulers import build_optimizer, build_scheduler

# register modules
import src.model.dncnn  # noqa: F401
import src.model.gan    # noqa: F401
import src.model.Unet   # noqa: F401
import src.model.armnet # noqa: F401

try:
    import src.data.brain_tumor_user  # noqa: F401
except Exception:
    pass

import src.trainers.denoise_trainer      # noqa: F401
import src.trainers.gan_denoise_trainer  # noqa: F401
import src.trainers.armnet_trainer       # noqa: F401

try:
    from src.utils.val_images import save_val_from_loader
except Exception:
    save_val_from_loader = None


def _supports_arg(cls, name: str) -> bool:
    try:
        return name in inspect.signature(cls.__init__).parameters
    except Exception:
        return False


class FixedNoiseWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds, fixed_seed: int = 999):
        self.base_ds = base_ds
        self.fixed_seed = int(fixed_seed)

        self._gen_attr = None
        for name in ("_gen", "_g", "gen", "generator"):
            if hasattr(base_ds, name) and isinstance(getattr(base_ds, name), torch.Generator):
                self._gen_attr = name
                break

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        seed = self.fixed_seed + int(idx)

        saved_state = None
        if self._gen_attr is not None:
            g = getattr(self.base_ds, self._gen_attr)
            try:
                saved_state = g.get_state()
                g.manual_seed(seed)
            except Exception:
                saved_state = None

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        out = self.base_ds[idx]

        if saved_state is not None and self._gen_attr is not None:
            try:
                getattr(self.base_ds, self._gen_attr).set_state(saved_state)
            except Exception:
                pass

        return out


def _build_datasets(cfg: dict) -> Tuple[Any, Any, Any]:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    ds_cls = DATASETS.get(data_cfg["name"])

    args = dict(data_cfg.get("args", {}))
    val_root_dir = args.pop("val_root_dir", None)

    val_fixed_noise = bool(args.pop("val_fixed_noise", True))
    val_fixed_seed = int(args.pop("val_fixed_seed", 999))

    val_args_cfg = data_cfg.get("val_args", None)
    test_args_cfg = data_cfg.get("test_args", None)

    if _supports_arg(ds_cls, "split"):
        train_args = dict(args)
        train_args["split"] = "train"
        train_args.setdefault("fixed_noise", False)
        train_args.setdefault("fixed_seed", val_fixed_seed)

        v_args = dict(args)
        if isinstance(val_args_cfg, dict):
            v_args.update(val_args_cfg)
        v_args["split"] = "val"
        v_args.setdefault("fixed_noise", True if val_fixed_noise else False)
        v_args.setdefault("fixed_seed", val_fixed_seed)
        if val_root_dir:
            v_args["root_dir"] = val_root_dir

        t_args = None
        if isinstance(test_args_cfg, dict):
            t_args = dict(args)
            t_args.update(test_args_cfg)
            t_args["split"] = "test"
            t_args.setdefault("fixed_noise", True)
            t_args.setdefault("fixed_seed", val_fixed_seed)

        train_ds = ds_cls(**train_args)

        need_wrap_val = False
        try:
            val_ds = ds_cls(**v_args)
        except TypeError:
            v2 = dict(v_args); v2.pop("fixed_noise", None); v2.pop("fixed_seed", None)
            val_ds = ds_cls(**v2)
            need_wrap_val = True
        if need_wrap_val and bool(v_args.get("fixed_noise", False)):
            val_ds = FixedNoiseWrapper(val_ds, fixed_seed=int(v_args.get("fixed_seed", val_fixed_seed)))

        test_ds = None
        if t_args is not None:
            need_wrap_test = False
            try:
                test_ds = ds_cls(**t_args)
            except TypeError:
                t2 = dict(t_args); t2.pop("fixed_noise", None); t2.pop("fixed_seed", None)
                test_ds = ds_cls(**t2)
                need_wrap_test = True
            if need_wrap_test and bool(t_args.get("fixed_noise", False)):
                test_ds = FixedNoiseWrapper(test_ds, fixed_seed=int(t_args.get("fixed_seed", val_fixed_seed)))

        return train_ds, val_ds, test_ds

    raise RuntimeError("This resume script expects a dataset that supports split='train/val/test'.")


def _load_ckpt(run_dir: Path, which: str):
    ckpt_dir = run_dir / "checkpoints"
    if which in ("best", "last"):
        path = ckpt_dir / f"{which}.pt"
    else:
        path = Path(which)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location="cpu")
    return path, obj


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _get_epoch_from_ckpt(ckpt_obj: dict) -> int:
    if "epoch" in ckpt_obj:
        return _safe_int(ckpt_obj.get("epoch", 0), 0)
    m = ckpt_obj.get("metrics", {})
    return _safe_int(m.get("epoch", 0), 0)


def _save_test_artifacts(run_dir: Path, test_metrics: dict) -> None:
    run_dir = Path(run_dir)
    try:
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        with (run_dir / "metrics.test.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(test_metrics) + "\n")
    except Exception:
        pass


def _fast_forward_scheduler(scheduler, steps: int):
    if scheduler is None:
        return
    for _ in range(max(0, int(steps))):
        try:
            scheduler.step()
        except Exception:
            break


def _optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _scaler_to_device(scaler: torch.amp.GradScaler, device: torch.device) -> None:
    try:
        st = scaler.state_dict()
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)
        scaler.load_state_dict(st)
    except Exception:
        pass


def _make_logger(run_dir: Path) -> TextLogger:
    return TextLogger(
        log_path=run_dir / "train.resume.log.txt",
        jsonl_path=run_dir / "metrics.resume.jsonl",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str, help="Existing run directory created previously.")
    ap.add_argument("--ckpt", default="last", help="Which checkpoint to load: 'last', 'best', or a path to .pt")
    ap.add_argument("--override", nargs="*", default=[], help="Optional config overrides, same format as run.py")
    ap.add_argument("--max_minutes", type=float, default=0.0, help="Stop after N minutes (0 = no limit).")
    ap.add_argument("--save_val_images", type=int, default=1, help="1/0 to enable val image saving (if available).")
    ap.add_argument("--val_image_every", type=int, default=1)
    ap.add_argument("--val_image_max", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.final.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.final.json not found in {run_dir}. Resume from an existing run dir.")

    cfg = load_config(str(cfg_path))
    cfg = deep_update(cfg, parse_overrides(args.override))

    set_seed(cfg.get("seed", None))
    train_cfg = cfg.get("training", {})
    trainer_name = train_cfg.get("trainer")
    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    logger = _make_logger(run_dir)
    logger.log(f"Resume run_dir: {run_dir}")
    logger.log(f"Loaded config: {cfg_path}")
    logger.log(f"Overrides: {args.override}")
    logger.log(f"Checkpoint: {args.ckpt}")

    train_ds, val_ds, test_ds = _build_datasets(cfg)
    logger.log(f"Datasets: train={len(train_ds)} val={len(val_ds)}" + (f" test={len(test_ds)}" if test_ds is not None else ""))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(train_cfg.get("test_batch_size", train_cfg["batch_size"])),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
        )

    ckpt_path, ckpt_obj = _load_ckpt(run_dir, args.ckpt)
    start_epoch = _get_epoch_from_ckpt(ckpt_obj)
    logger.log(f"Loaded checkpoint: {ckpt_path.name} (epoch={start_epoch})")

    loss_name = str(train_cfg.get("loss", "l1")).lower()
    criterion = torch.nn.MSELoss() if loss_name == "mse" else torch.nn.L1Loss()

    rec_loss_name = str(train_cfg.get("rec_loss", None) or train_cfg.get("rec_criterion", {}).get("name", "l1")).lower()
    rec_criterion = torch.nn.MSELoss() if rec_loss_name == "mse" else torch.nn.L1Loss()

    # trainer instance for _run_epoch
    if trainer_name == "armnet_denoise":
        trainer = TRAINERS.get(trainer_name)(device=train_cfg.get("device", "cuda"))
    else:
        trainer = TRAINERS.get(trainer_name)(device=train_cfg.get("device", "cuda"), lambda_rec=train_cfg.get("lambda_rec", 100.0))

    use_amp = bool(train_cfg.get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if (device.type == "cuda") else None

    # best/bad_epochs continuity (prefer checkpoint)
    mode = train_cfg.get("save_best_mode", "min")
    best_metric = float(ckpt_obj.get("best_metric", float("inf") if mode == "min" else -float("inf")))
    bad_epochs = _safe_int(ckpt_obj.get("bad_epochs", 0), 0)

    if trainer_name == "gan_denoise":
        G_cfg = cfg["model"]["generator"]
        D_cfg = cfg["model"]["discriminator"]
        G = MODELS.get(G_cfg["name"])(**G_cfg["args"]).to(device)
        D = MODELS.get(D_cfg["name"])(**D_cfg["args"]).to(device)

        if "G" in ckpt_obj:
            G.load_state_dict(ckpt_obj["G"])
        if "D" in ckpt_obj:
            D.load_state_dict(ckpt_obj["D"])

        optimizer_G = build_optimizer(G, train_cfg["optimizer_G_cfg"])
        optimizer_D = build_optimizer(D, train_cfg["optimizer_D_cfg"])
        scheduler_G = build_scheduler(optimizer_G, train_cfg.get("scheduler_G_cfg"))
        scheduler_D = build_scheduler(optimizer_D, train_cfg.get("scheduler_D_cfg"))

        loaded_sched_state = False
        if "optimizer_G" in ckpt_obj and ckpt_obj["optimizer_G"] is not None:
            try:
                optimizer_G.load_state_dict(ckpt_obj["optimizer_G"])
                _optimizer_to_device(optimizer_G, device)
                logger.log("Restored optimizer_G state from checkpoint.")
            except Exception as e:
                logger.log(f"[WARN] Failed to restore optimizer_G: {e}")
        if "optimizer_D" in ckpt_obj and ckpt_obj["optimizer_D"] is not None:
            try:
                optimizer_D.load_state_dict(ckpt_obj["optimizer_D"])
                _optimizer_to_device(optimizer_D, device)
                logger.log("Restored optimizer_D state from checkpoint.")
            except Exception as e:
                logger.log(f"[WARN] Failed to restore optimizer_D: {e}")

        if scheduler_G is not None and "scheduler_G" in ckpt_obj and ckpt_obj["scheduler_G"] is not None:
            try:
                scheduler_G.load_state_dict(ckpt_obj["scheduler_G"])
                loaded_sched_state = True
                logger.log("Restored scheduler_G state from checkpoint.")
            except Exception as e:
                logger.log(f"[WARN] Failed to restore scheduler_G: {e}")
        if scheduler_D is not None and "scheduler_D" in ckpt_obj and ckpt_obj["scheduler_D"] is not None:
            try:
                scheduler_D.load_state_dict(ckpt_obj["scheduler_D"])
                loaded_sched_state = True
                logger.log("Restored scheduler_D state from checkpoint.")
            except Exception as e:
                logger.log(f"[WARN] Failed to restore scheduler_D: {e}")

        if scaler is not None and "scaler" in ckpt_obj and ckpt_obj["scaler"] is not None:
            try:
                scaler.load_state_dict(ckpt_obj["scaler"])
                _scaler_to_device(scaler, device)
                logger.log("Restored AMP scaler state from checkpoint.")
            except Exception as e:
                logger.log(f"[WARN] Failed to restore scaler: {e}")

        if not loaded_sched_state:
            _fast_forward_scheduler(scheduler_G, start_epoch)
            _fast_forward_scheduler(scheduler_D, start_epoch)

    else:
        model_cfg = cfg["model"]
        model = MODELS.get(model_cfg["name"])(**model_cfg.get("args", {})).to(device)

        if "model" in ckpt_obj:
            model.load_state_dict(ckpt_obj["model"])

        optimizer = build_optimizer(model, train_cfg["optimizer_cfg"])
        scheduler = build_scheduler(optimizer, train_cfg.get("scheduler_cfg"))

        loaded_sched_state = False
        if "optimizer" in ckpt_obj and ckpt_obj["optimizer"] is not None:
            try:
                optimizer.load_state_dict(ckpt_obj["optimizer"])
                _optimizer_to_device(optimizer, device)
                logger.log("Restored optimizer state from checkpoint.")
            except Exception as e:
                logger.log(f"[WARN] Failed to restore optimizer: {e}")

        if scheduler is not None and "scheduler" in ckpt_obj and ckpt_obj["scheduler"] is not None:
            try:
                scheduler.load_state_dict(ckpt_obj["scheduler"])
                loaded_sched_state = True
                logger.log("Restored scheduler state from checkpoint.")
            except Exception as e:
                logger.log(f"[WARN] Failed to restore scheduler: {e}")

        if scaler is not None and "scaler" in ckpt_obj and ckpt_obj["scaler"] is not None:
            try:
                scaler.load_state_dict(ckpt_obj["scaler"])
                _scaler_to_device(scaler, device)
                logger.log("Restored AMP scaler state from checkpoint.")
            except Exception as e:
                logger.log(f"[WARN] Failed to restore scaler: {e}")

        if not loaded_sched_state:
            _fast_forward_scheduler(scheduler, start_epoch)

    total_epochs = int(train_cfg.get("epochs", 1))
    remaining = max(1, total_epochs - int(start_epoch))
    logger.log(f"Will continue for up to {remaining} epoch(s) to reach total_epochs={total_epochs}.")

    t_deadline = None
    if args.max_minutes and args.max_minutes > 0:
        t_deadline = time.time() + float(args.max_minutes) * 60.0
        logger.log(f"Time limit: {args.max_minutes:.1f} minutes (will stop gracefully).")

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    patience = train_cfg.get("early_stopping_patience", None)

    for step in range(1, remaining + 1):
        epoch = start_epoch + step

        if t_deadline is not None and time.time() >= t_deadline:
            logger.log("Reached time limit. Stopping (you can resume again).")
            break

        t0 = time.time()

        if trainer_name == "gan_denoise":
            tr = trainer._run_epoch(G, D, train_loader, opt_G=optimizer_G, opt_D=optimizer_D, rec_criterion=rec_criterion, use_amp=use_amp, scaler=scaler)
            va = trainer._run_epoch(G, D, val_loader, opt_G=None, opt_D=None, rec_criterion=rec_criterion, use_amp=False, scaler=None)

            if scheduler_G is not None:
                scheduler_G.step()
            if scheduler_D is not None:
                scheduler_D.step()

            cur_lr = float(optimizer_G.param_groups[0]["lr"])
            metrics = {
                "epoch": epoch,
                "lr": cur_lr,
                "time_sec": round(time.time() - t0, 3),
                "train_g_loss": tr["g_loss"],
                "train_d_loss": tr["d_loss"],
                "train_psnr": tr["psnr"],
                "train_ssim": tr["ssim"],
                "val_g_loss": va["g_loss"],
                "val_psnr": va["psnr"],
                "val_ssim": va["ssim"],
            }
            logger.log(
                f"Epoch {epoch:03d}/{total_epochs} | lr={cur_lr:.6g} | "
                f"G={metrics['train_g_loss']:.4f} D={metrics['train_d_loss']:.4f} | "
                f"val_PSNR={metrics['val_psnr']:.3f} val_SSIM={metrics['val_ssim']:.4f} | "
                f"{metrics['time_sec']:.1f}s"
            )
            logger.log_metrics(metrics)

            if save_val_from_loader is not None and int(args.save_val_images) == 1 and (epoch % int(args.val_image_every) == 0):
                try:
                    out_dir = save_val_from_loader(
                        forward_fn=lambda x: G(x),
                        val_loader=val_loader,
                        device=device,
                        run_dir=run_dir,
                        epoch=epoch,
                        max_items=int(args.val_image_max),
                        prefix="val",
                    )
                    logger.log(f"Saved val images -> {out_dir}")
                except Exception as e:
                    logger.log(f"[WARN] Save val images failed: {e}")

            key = train_cfg.get("save_best", "val_psnr")
            mode2 = train_cfg.get("save_best_mode", "max")
            cur = metrics.get(key, metrics.get("val_psnr"))
            improved = (cur > best_metric) if mode2 == "max" else (cur < best_metric)

            if improved:
                best_metric = float(cur)
                bad_epochs = 0
                torch.save(
                    {
                        "G": G.state_dict(),
                        "D": D.state_dict(),
                        "optimizer_G": optimizer_G.state_dict(),
                        "optimizer_D": optimizer_D.state_dict(),
                        "scheduler_G": scheduler_G.state_dict() if scheduler_G is not None else None,
                        "scheduler_D": scheduler_D.state_dict() if scheduler_D is not None else None,
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "metrics": metrics,
                        "best_metric": best_metric,
                        "bad_epochs": bad_epochs,
                        "epoch": epoch,
                    },
                    ckpt_dir / "best.pt",
                )
                logger.log(f"✓ Saved best checkpoint: best.pt ({key}={best_metric:.6f})")
            else:
                bad_epochs += 1

            torch.save(
                {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D": optimizer_D.state_dict(),
                    "scheduler_G": scheduler_G.state_dict() if scheduler_G is not None else None,
                    "scheduler_D": scheduler_D.state_dict() if scheduler_D is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "metrics": metrics,
                    "best_metric": best_metric,
                    "bad_epochs": bad_epochs,
                    "epoch": epoch,
                },
                ckpt_dir / "last.pt",
            )

        else:
            tr = trainer._run_epoch(model, train_loader, optimizer=optimizer, criterion=criterion, use_amp=use_amp, scaler=scaler)
            va = trainer._run_epoch(model, val_loader, optimizer=None, criterion=criterion, use_amp=False, scaler=None)

            if scheduler is not None:
                scheduler.step()

            cur_lr = float(optimizer.param_groups[0]["lr"])
            metrics = {
                "epoch": epoch,
                "lr": cur_lr,
                "time_sec": round(time.time() - t0, 3),
                "train_loss": tr["loss"],
                "train_psnr": tr["psnr"],
                "train_ssim": tr["ssim"],
                "val_loss": va["loss"],
                "val_psnr": va["psnr"],
                "val_ssim": va["ssim"],
            }
            logger.log(
                f"Epoch {epoch:03d}/{total_epochs} | lr={cur_lr:.6g} | "
                f"train_loss={metrics['train_loss']:.6f} psnr={metrics['train_psnr']:.3f} ssim={metrics['train_ssim']:.4f} | "
                f"val_loss={metrics['val_loss']:.6f} psnr={metrics['val_psnr']:.3f} ssim={metrics['val_ssim']:.4f} | "
                f"{metrics['time_sec']:.1f}s"
            )
            logger.log_metrics(metrics)

            if save_val_from_loader is not None and int(args.save_val_images) == 1 and (epoch % int(args.val_image_every) == 0):
                try:
                    out_dir = save_val_from_loader(
                        forward_fn=lambda x: model(x),
                        val_loader=val_loader,
                        device=device,
                        run_dir=run_dir,
                        epoch=epoch,
                        max_items=int(args.val_image_max),
                        prefix="val",
                    )
                    logger.log(f"Saved val images -> {out_dir}")
                except Exception as e:
                    logger.log(f"[WARN] Save val images failed: {e}")

            key = train_cfg.get("save_best", "val_loss")
            mode2 = train_cfg.get("save_best_mode", "min")
            cur = metrics.get(key, metrics["val_loss"])
            improved = (cur < best_metric) if mode2 == "min" else (cur > best_metric)

            if improved:
                best_metric = float(cur)
                bad_epochs = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "metrics": metrics,
                        "best_metric": best_metric,
                        "bad_epochs": bad_epochs,
                        "epoch": epoch,
                    },
                    ckpt_dir / "best.pt",
                )
                logger.log(f"✓ Saved best checkpoint: best.pt ({key}={best_metric:.6f})")
            else:
                bad_epochs += 1

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "metrics": metrics,
                    "best_metric": best_metric,
                    "bad_epochs": bad_epochs,
                    "epoch": epoch,
                },
                ckpt_dir / "last.pt",
            )

        if patience is not None and bad_epochs >= int(patience):
            logger.log(f"Early stopping triggered (patience={patience}).")
            break

    if test_loader is not None:
        logger.log("Running final evaluation on TEST split...")
        if trainer_name == "gan_denoise":
            te = trainer._run_epoch(G, D, test_loader, opt_G=None, opt_D=None, rec_criterion=rec_criterion, use_amp=False, scaler=None)
            test_metrics = {
                "epoch": -1,
                "split": "test",
                "test_g_loss": te.get("g_loss"),
                "test_d_loss": te.get("d_loss"),
                "test_psnr": te.get("psnr"),
                "test_ssim": te.get("ssim"),
            }
            logger.log(
                f"TEST | PSNR={test_metrics['test_psnr']:.3f} SSIM={test_metrics['test_ssim']:.4f} "
                f"G={test_metrics['test_g_loss']:.4f} D={test_metrics['test_d_loss']:.4f}"
            )
            logger.log_metrics(test_metrics)
            _save_test_artifacts(run_dir, test_metrics)
        else:
            te = trainer._run_epoch(model, test_loader, criterion=criterion, optimizer=None, use_amp=False, scaler=None)
            test_metrics = {
                "epoch": -1,
                "split": "test",
                "test_loss": te.get("loss"),
                "test_psnr": te.get("psnr"),
                "test_ssim": te.get("ssim"),
            }
            logger.log(
                f"TEST | loss={test_metrics['test_loss']:.6f} "
                f"psnr={test_metrics['test_psnr']:.3f} ssim={test_metrics['test_ssim']:.4f}"
            )
            logger.log_metrics(test_metrics)
            _save_test_artifacts(run_dir, test_metrics)

    logger.log("Resume run finished.")


if __name__ == "__main__":
    main()
