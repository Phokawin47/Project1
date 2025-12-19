from __future__ import annotations

class Registry:
    def __init__(self):
        self._items = {}

    def register(self, name: str):
        def deco(obj):
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"Unknown key: {name}. Available: {sorted(self._items.keys())}")
        return self._items[name]

MODELS = Registry()
DATASETS = Registry()
TRAINERS = Registry()
