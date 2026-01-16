from __future__ import annotations

import argparse
import hashlib
import json
import inspect
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import load_config, save_config, deep_update, parse_overrides
from src.registry import MODELS, DATASETS, TRAINERS
from src.utils.seed import set_seed
from src.utils.logger import TextLogger
from src.optim_schedulers import build_optimizer, build_scheduler

import torch.multiprocessing as mp

# --- import modules for registration (keep these!) ---
import src.model.dncnn               # noqa: F401
import src.model.gan                 # noqa: F401
import src.model.Unet                # noqa: F401
import src.model.armnet              # noqa: F401

# IMPORTANT: import the dataset module you actually use in your project.
try:
    import src.data.brain_tumor_user  # noqa: F401
except Exception:
    pass

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

import src.trainers.denoise_trainer        # noqa: F401
import src.trainers.gan_denoise_trainer    # noqa: F401
import src.trainers.armnet_trainer         # noqa: F401


def make_run_dir(root: Path, model_name: str, exp_name: str | None, cfg: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_bytes = json.dumps(cfg, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(cfg_bytes).hexdigest()[:8]
    tag = f"{ts}_{h}" if not exp_name else f"{ts}_{exp_name}_{h}"
    run_dir = root / model_name / tag
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _normalize_training_cfg(cfg: dict) -> dict:
    # Back-compat: training.lr/training.optimizer/training.weight_decay -> optimizer_cfg
    tr = dict(cfg.get("training", {}))
    if "optimizer_cfg" not in tr:
        opt_name = tr.get("optimizer", "adamw")
        lr = tr.get("lr", None)
        wd = tr.get("weight_decay", None)
        opt_args = {}
        if lr is not None:
            opt_args["lr"] = lr
        if wd is not None:
            opt_args["weight_decay"] = wd
        tr["optimizer_cfg"] = {"name": opt_name, "args": opt_args}
    if "scheduler_cfg" not in tr:
        tr["scheduler_cfg"] = {"name": "none"}
    return tr


class FixedNoiseWrapper(torch.utils.data.Dataset):
    """
    Make noise deterministic per-sample (idx) without rewriting your dataset.

    This wrapper is only needed if the dataset does NOT accept fixed_noise/fixed_seed args.
    """
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


def _supports_arg(cls, name: str) -> bool:
    try:
        return name in inspect.signature(cls.__init__).parameters
    except Exception:
        return False


def _build_datasets(cfg: dict, data_cfg: dict, train_cfg: dict, logger: TextLogger):
    """
    Build train/val/test datasets.

    Priority:
    1) If dataset supports split=... -> build by split using data.args, data.val_args, data.test_args
    2) Else fallback to manual splitting (train/val only) like before (test=None)
    """
    ds_cls = DATASETS.get(data_cfg["name"])

    args = dict(data_cfg.get("args", {}))
    # legacy val_root_dir path (optional)
    val_root_dir = args.pop("val_root_dir", None)

    # global fixed noise settings (used if val_args/test_args don't override)
    val_fixed_noise = bool(args.pop("val_fixed_noise", True))
    val_fixed_seed = int(args.pop("val_fixed_seed", 999))

    val_args_cfg = data_cfg.get("val_args", None)
    test_args_cfg = data_cfg.get("test_args", None)

    supports_split = _supports_arg(ds_cls, "split")

    # -------------------------
    # Native split path
    # -------------------------
    if supports_split:
        # TRAIN
        train_args = dict(args)
        train_args["split"] = "train"
        # default: train noise should be random (unless user explicitly forces fixed_noise in args)
        train_args.setdefault("fixed_noise", False)
        train_args.setdefault("fixed_seed", val_fixed_seed)

        # VAL (merge args + val_args override)
        v_args = dict(args)
        if isinstance(val_args_cfg, dict):
            v_args.update(val_args_cfg)
        v_args["split"] = "val"
        # default val noise deterministic
        if "fixed_noise" not in v_args:
            v_args["fixed_noise"] = True if val_fixed_noise else False
        if "fixed_seed" not in v_args:
            v_args["fixed_seed"] = val_fixed_seed
        if val_root_dir:
            v_args["root_dir"] = val_root_dir

        # TEST (optional)
        t_args = None
        if isinstance(test_args_cfg, dict):
            t_args = dict(args)
            t_args.update(test_args_cfg)
            t_args["split"] = "test"
            # default test noise deterministic
            t_args.setdefault("fixed_noise", True)
            t_args.setdefault("fixed_seed", val_fixed_seed)

        # build datasets (wrap if dataset doesn't accept fixed_noise/fixed_seed)
        train_ds = ds_cls(**train_args)

        # val dataset
        need_wrap_val = False
        try:
            val_ds = ds_cls(**v_args)
        except TypeError:
            v2 = dict(v_args)
            v2.pop("fixed_noise", None)
            v2.pop("fixed_seed", None)
            val_ds = ds_cls(**v2)
            need_wrap_val = True

        if need_wrap_val and bool(v_args.get("fixed_noise", False)):
            val_ds = FixedNoiseWrapper(val_ds, fixed_seed=int(v_args.get("fixed_seed", val_fixed_seed)))

        # test dataset (optional)
        test_ds = None
        if t_args is not None:
            need_wrap_test = False
            try:
                test_ds = ds_cls(**t_args)
            except TypeError:
                t2 = dict(t_args)
                t2.pop("fixed_noise", None)
                t2.pop("fixed_seed", None)
                test_ds = ds_cls(**t2)
                need_wrap_test = True

            if need_wrap_test and bool(t_args.get("fixed_noise", False)):
                test_ds = FixedNoiseWrapper(test_ds, fixed_seed=int(t_args.get("fixed_seed", val_fixed_seed)))

        logger.log(
            f"Dataset: native split | "
            f"train={len(train_ds)} val={len(val_ds)}" + (f" test={len(test_ds)}" if test_ds is not None else "")
        )
        return train_ds, val_ds, test_ds

    # -------------------------
    # Fallback path (manual split): train/val only
    # -------------------------
    def make_ds(root_dir_value: str | None, fixed: bool):
        a = dict(args)
        if root_dir_value is not None:
            a["root_dir"] = root_dir_value

        if fixed and val_fixed_noise:
            try:
                ds = ds_cls(**a, fixed_noise=True, fixed_seed=val_fixed_seed)
                return ds, False
            except TypeError:
                ds = ds_cls(**a)
                return ds, True
        else:
            try:
                ds = ds_cls(**a, fixed_noise=False, fixed_seed=val_fixed_seed)
                return ds, False
            except TypeError:
                ds = ds_cls(**a)
                return ds, False

    # Build full dataset
    train_full, _ = make_ds(root_dir_value=args.get("root_dir"), fixed=False)

    # Case A: separate val_root_dir
    if val_root_dir:
        val_full, need_wrap = make_ds(root_dir_value=val_root_dir, fixed=True)
        if need_wrap and val_fixed_noise:
            val_full = FixedNoiseWrapper(val_full, fixed_seed=val_fixed_seed)
        logger.log(f"Dataset: val_root_dir={val_root_dir} | fixed_val={val_fixed_noise} seed={val_fixed_seed}")
        return train_full, val_full, None

    # Case B: split from same root (train/val only)
    N = len(train_full)
    val_ratio = float(train_cfg.get("val_ratio", 0.2))
    n_val = max(1, int(N * val_ratio))

    g = torch.Generator().manual_seed(cfg.get("split_seed", 1234))
    perm = torch.randperm(N, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = Subset(train_full, train_idx)

    val_full, need_wrap = make_ds(root_dir_value=None, fixed=True)
    if need_wrap and val_fixed_noise:
        val_full = FixedNoiseWrapper(val_full, fixed_seed=val_fixed_seed)
    val_ds = Subset(val_full, val_idx)

    logger.log(f"Dataset: manual split | train={len(train_ds)} val={len(val_ds)} (no test)")
    return train_ds, val_ds, None


@torch.no_grad()
def _eval_denoise_model(model, loader, criterion, device, use_amp: bool):
    model.eval()
    total = 0.0
    n = 0
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            noisy, clean = batch[0], batch[1]
        elif isinstance(batch, dict):
            noisy, clean = batch["noisy"], batch["clean"]
        else:
            raise ValueError("Unknown batch format")

        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.autocast(device_type=device_type, enabled=use_amp):
            pred = model(noisy)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            loss = criterion(pred, clean)

        total += float(loss.item()) * noisy.size(0)
        n += int(noisy.size(0))

    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = deep_update(cfg, parse_overrides(args.override))

    set_seed(cfg.get("seed", None))
    cfg["training"] = _normalize_training_cfg(cfg)
    train_cfg = cfg["training"]
    trainer_name = train_cfg.get("trainer")

    model_cfg = cfg["model"]
    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    out_cfg = cfg.get("output", {})

    # name for run directory
    if "name" in model_cfg:
        model_name = model_cfg["name"]
    elif "generator" in model_cfg and "discriminator" in model_cfg:
        model_name = f"GAN_{model_cfg['generator']['name']}_{model_cfg['discriminator']['name']}"
    else:
        model_name = "unknown_model"

    run_dir = make_run_dir(Path(out_cfg.get("runs_root", "runs")), model_name, out_cfg.get("exp_name", None), cfg)

    logger = TextLogger(log_path=run_dir / "train.log.txt", jsonl_path=run_dir / "metrics.jsonl")
    logger.log(f"Run dir: {run_dir}")
    logger.log(f"Config: {args.config}")
    logger.log(f"Overrides: {args.override}")
    save_config(cfg, run_dir / "config.final.json")

    # Build datasets: (train, val, test)
    train_ds, val_ds, test_ds = _build_datasets(cfg, data_cfg, train_cfg, logger)

    msg = f"Dataset size: train={len(train_ds)} val={len(val_ds)}"
    if test_ds is not None:
        msg += f" test={len(test_ds)}"
    logger.log(msg)

    # loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=True,  # important for BatchNorm stability
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

    # Build model(s)
    if trainer_name == "gan_denoise":
        G_cfg = cfg["model"]["generator"]
        D_cfg = cfg["model"]["discriminator"]
        G = MODELS.get(G_cfg["name"])(**G_cfg["args"]).to(device)
        D = MODELS.get(D_cfg["name"])(**D_cfg["args"]).to(device)
        logger.log(f"Model: GAN | G={G_cfg['name']} D={D_cfg['name']}")
    else:
        model_cls = MODELS.get(model_cfg["name"])
        model = model_cls(**model_cfg.get("args", {})).to(device)
        logger.log(f"Model: {model_cfg['name']} | args={model_cfg.get('args', {})}")

    # Loss
    loss_name = str(train_cfg.get("loss", "l1")).lower()
    criterion = torch.nn.MSELoss() if loss_name == "mse" else torch.nn.L1Loss()

    # Optimizers / schedulers
    if trainer_name == "gan_denoise":
        optimizer_G = build_optimizer(G, train_cfg["optimizer_G_cfg"])
        optimizer_D = build_optimizer(D, train_cfg["optimizer_D_cfg"])

        scheduler_G = build_scheduler(optimizer_G, train_cfg.get("scheduler_G_cfg"))
        scheduler_D = build_scheduler(optimizer_D, train_cfg.get("scheduler_D_cfg"))
    else:
        optimizer = build_optimizer(model, train_cfg["optimizer_cfg"])
        scheduler = build_scheduler(optimizer, train_cfg.get("scheduler_cfg"))

    # Trainer instance
    if trainer_name == "armnet_denoise":
        trainer = TRAINERS.get(trainer_name)(device=train_cfg.get("device", "cuda"))
    else:
        trainer = TRAINERS.get(trainer_name)(
            device=train_cfg.get("device", "cuda"),
            lambda_rec=train_cfg.get("lambda_rec", 100.0),
        )

    use_amp = bool(train_cfg.get("amp", False))
    logger.log(f"Training: epochs={train_cfg['epochs']} bs={train_cfg['batch_size']} loss={loss_name} amp={use_amp}")
    if "optimizer_cfg" in train_cfg:
        logger.log(f"Optimizer: {train_cfg['optimizer_cfg']}")
    logger.log(f"Scheduler: {train_cfg.get('scheduler_cfg', {'name':'none'})}")

    # Optional: val image saving params (if your trainers support them)
    save_val_images = bool(train_cfg.get("save_val_images", True))
    val_image_every = int(train_cfg.get("val_image_every", 1))
    val_image_max = int(train_cfg.get("val_image_max", 8))

    # Rec criterion (GAN)
    rec_loss_name = str(train_cfg.get("rec_loss", "l1")).lower()
    rec_criterion = torch.nn.MSELoss() if rec_loss_name == "mse" else torch.nn.L1Loss()

    # Train
    if trainer_name == "gan_denoise":
        res = trainer.fit(
            G=G,
            D=D,
            train_loader=train_loader,
            val_loader=val_loader,
            opt_G=optimizer_G,
            opt_D=optimizer_D,
            rec_criterion=rec_criterion,
            epochs=int(train_cfg["epochs"]),
            run_dir=run_dir,
            logger=logger,
            scheduler_G=scheduler_G,
            scheduler_D=scheduler_D,
            use_amp=use_amp,
            early_stopping_patience=train_cfg.get("early_stopping_patience"),
            save_val_images=save_val_images,
            val_image_every=val_image_every,
            val_image_max=val_image_max,
        )
    else:
        res = trainer.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=int(train_cfg["epochs"]),
            run_dir=run_dir,
            logger=logger,
            scheduler=scheduler,
            use_amp=use_amp,
            save_best=train_cfg.get("save_best", "val_loss"),
            mode=train_cfg.get("save_best_mode", "min"),
            early_stopping_patience=train_cfg.get("early_stopping_patience", None),
            save_val_images=save_val_images,
            val_image_every=val_image_every,
            val_image_max=val_image_max,
        )
    logger.log(f"Done. Best: {res}")

    # Evaluate on TEST (optional)
    if test_loader is not None:
        logger.log("Running final evaluation on TEST split...")
        if trainer_name == "gan_denoise":
            # reuse GAN trainer's _run_epoch in eval mode (opt_G/opt_D None)
            te = trainer._run_epoch(
                G,
                D,
                test_loader,
                opt_G=None,
                opt_D=None,
                rec_criterion=rec_criterion,
                use_amp=False,
                scaler=None,
            )
            # te contains psnr/ssim (and g_loss/d_loss)
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
        else:
            # For denoise/armnet trainers, easiest is to call their _run_epoch in val mode
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


if __name__ == "__main__":
    main()
