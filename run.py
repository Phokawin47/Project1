from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import load_config, save_config, deep_update, parse_overrides
from src.registry import MODELS, DATASETS, TRAINERS
from src.utils.seed import set_seed
from src.utils.logger import TextLogger
from src.optim_schedulers import build_optimizer, build_scheduler

# --- import modules for registration (keep these!) ---
import src.model.dncnn               # noqa: F401

# IMPORTANT: import the dataset module you actually use in your project.
# If you use your own dataset file under src/data/, import it here.
# Example (template): import src.data.brain_tumor_user  # noqa: F401
try:
    import src.data.brain_tumor_user  # noqa: F401
except Exception:
    # If your project registers dataset from another module, it's OK.
    pass

import src.trainers.denoise_trainer   # noqa: F401


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
    Make val noise deterministic per-sample (idx) without rewriting your dataset.

    How it works:
    - If the wrapped dataset supports fixed_noise/fixed_seed args, use those instead (preferred).
    - Otherwise, for each __getitem__(idx), set the dataset's internal generator (if any) to seed+idx
      temporarily, and also set torch global RNG seed. Then restore generator state.

    Designed to work with datasets that use:
    - dataset._gen or dataset._g as torch.Generator in torch.rand/torch.randn_like(..., generator=...)
    - global RNG (torch.rand/torch.randn_like without generator)
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


def _build_datasets(cfg: dict, data_cfg: dict, train_cfg: dict, logger: TextLogger):
    """
    Build train/val datasets.

    Supports:
    - Separate val root_dir via data.args.val_root_dir (optional)
    - Fixed val noise via data.args.val_fixed_noise (default True) and data.args.val_fixed_seed (default 999)

    Behavior:
    - If val_root_dir exists -> train uses root_dir, val uses val_root_dir
    - Else -> split indices from train root_dir (reproducible)
    - For fixed val noise, we try passing fixed_noise/fixed_seed to dataset.
      If not supported, wrap val dataset with FixedNoiseWrapper.
    """
    ds_cls = DATASETS.get(data_cfg["name"])

    args = dict(data_cfg.get("args", {}))
    val_root_dir = args.pop("val_root_dir", None)

    val_fixed_noise = bool(args.pop("val_fixed_noise", True))
    val_fixed_seed = int(args.pop("val_fixed_seed", 999))

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

    # Build train dataset
    train_full, _ = make_ds(root_dir_value=None, fixed=False)

    # Case A: separate val_root_dir
    if val_root_dir:
        val_full, need_wrap = make_ds(root_dir_value=val_root_dir, fixed=True)
        if need_wrap and val_fixed_noise:
            val_full = FixedNoiseWrapper(val_full, fixed_seed=val_fixed_seed)
        logger.log(
            f"Dataset: val_root_dir={val_root_dir} | fixed_val={val_fixed_noise} seed={val_fixed_seed}"
        )
        return train_full, val_full

    # Case B: split from same root
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

    logger.log(f"Dataset: split | fixed_val={val_fixed_noise} seed={val_fixed_seed}")
    return train_ds, val_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = deep_update(cfg, parse_overrides(args.override))

    set_seed(cfg.get("seed", None))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = _normalize_training_cfg(cfg)
    out_cfg = cfg.get("output", {})

    model_name = model_cfg["name"]
    run_dir = make_run_dir(Path(out_cfg.get("runs_root", "runs")), model_name, out_cfg.get("exp_name", None), cfg)

    logger = TextLogger(log_path=run_dir / "train.log.txt", jsonl_path=run_dir / "metrics.jsonl")
    logger.log(f"Run dir: {run_dir}")
    logger.log(f"Config: {args.config}")
    logger.log(f"Overrides: {args.override}")
    save_config(cfg, run_dir / "config.final.json")

    train_ds, val_ds = _build_datasets(cfg, data_cfg, train_cfg, logger)
    logger.log(f"Dataset size: {len(train_ds) + len(val_ds)} | train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )

    model_cls = MODELS.get(model_name)
    model = model_cls(**model_cfg.get("args", {}))
    logger.log(f"Model: {model_name} | args={model_cfg.get('args', {})}")

    loss_name = str(train_cfg.get("loss", "l1")).lower()
    criterion = torch.nn.MSELoss() if loss_name == "mse" else torch.nn.L1Loss()

    optimizer = build_optimizer(model, train_cfg["optimizer_cfg"])
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler_cfg", None))

    trainer = TRAINERS.get(train_cfg.get("trainer", "denoise"))(device=train_cfg.get("device", "cuda"))
    use_amp = bool(train_cfg.get("amp", False))

    logger.log(f"Training: epochs={train_cfg['epochs']} bs={train_cfg['batch_size']} loss={loss_name} amp={use_amp}")
    logger.log(f"Optimizer: {train_cfg['optimizer_cfg']}")
    logger.log(f"Scheduler: {train_cfg.get('scheduler_cfg', {'name':'none'})}")

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
    )
    logger.log(f"Done. Best: {res}")


if __name__ == "__main__":
    main()
