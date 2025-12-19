from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json

@dataclass
class TextLogger:
    # Prints to console and appends to a text file.
    # Optionally logs dict metrics to JSONL (one dict per line).
    log_path: Path
    jsonl_path: Path | None = None

    def __post_init__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.write_text("", encoding="utf-8")

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_metrics(self, metrics: dict):
        if self.jsonl_path is None:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
