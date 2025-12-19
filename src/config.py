from __future__ import annotations
from pathlib import Path
import json

def deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d

def set_by_dotted_path(cfg: dict, dotted: str, value):
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def parse_overrides(overrides: list[str]) -> dict:
    # overrides: ["training.epochs=50", 'optimizer.name="sgd"']
    out = {}
    for item in overrides:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            v_parsed = json.loads(v)
        except Exception:
            v_parsed = v
        set_by_dotted_path(out, k, v_parsed)
    return out

def load_config(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))

def save_config(cfg: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
