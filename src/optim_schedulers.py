from __future__ import annotations
import torch

def build_optimizer(model, cfg: dict):
    # New style:
    # "optimizer": {"name":"adamw","args":{"lr":1e-4,"weight_decay":1e-4}}
    # Back-compat style:
    # "training": {"lr":1e-4,"weight_decay":1e-4,"optimizer":"adamw"}
    name = str(cfg.get("name", "adamw")).lower()
    args = dict(cfg.get("args", {}))
    if name == "sgd":
        # sensible defaults if missing
        args.setdefault("lr", 1e-3)
        args.setdefault("momentum", 0.9)
        args.setdefault("weight_decay", 1e-4)
        return torch.optim.SGD(model.parameters(), **args)
    if name == "adam":
        args.setdefault("lr", 1e-4)
        args.setdefault("weight_decay", 0.0)
        return torch.optim.Adam(model.parameters(), **args)
    # default: adamw
    args.setdefault("lr", 1e-4)
    args.setdefault("weight_decay", 0.0)
    return torch.optim.AdamW(model.parameters(), **args)

def build_scheduler(optimizer, cfg: dict | None):
    # Examples:
    # {"name":"multistep","args":{"milestones":[10,20],"gamma":0.1}}
    # {"name":"none"}
    if cfg is None:
        return None
    name = str(cfg.get("name", "none")).lower()
    args = dict(cfg.get("args", {}))
    if name in ("none", "null", ""):
        return None
    if name == "multistep":
        args.setdefault("milestones", [10, 20])
        args.setdefault("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **args)
    if name == "step":
        args.setdefault("step_size", 10)
        args.setdefault("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, **args)
    if name == "cosine":
        # expects T_max
        args.setdefault("T_max", 50)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **args)
    raise ValueError(f"Unknown scheduler: {name}")
