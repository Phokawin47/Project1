from __future__ import annotations

import os
from typing import List, Optional, Tuple, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from ..registry import DATASETS

# Be tolerant to truncated/corrupted images (common in large exports)
ImageFile.LOAD_TRUNCATED_IMAGES = True


@DATASETS.register("brain_tumor_dataset")
class BrainTumorDataset(Dataset):
    """
    Split-aware dataset (train/val/test/all) + optional extra-train dirs.

    Mode (root_dir as list/tuple):
      - First folder = base_dir that will be split deterministically into train/val/test
        according to (val_ratio, test_ratio, split_seed)
      - Remaining folders = extra_train_dirs (ONLY added to split=train)

    Mode (root_dir as str):
      - base_dir = root_dir, split deterministically into train/val/test

    Optional val_root_dir:
      - If provided, it becomes base_dir (split from this directory)
      - root_dir (if list) that differs from val_root_dir becomes extra_train_dirs

    Notes:
      - split="all" returns base + extra (everything)
      - split="val" and split="test" never include extra_train_dirs
      - Supports reading 16-bit PNG (mode I;16) and normal 8-bit images.
    """

    def __init__(
        self,
        root_dir: Union[str, Sequence[str]],
        image_size: Tuple[int, int] = (64, 64),
        transform=None,
        noise_factor: float = 0.2,
        noise_range=None,
        noise_sampling: str = "uniform",
        use_grayscale: bool = False,
        return_noise_level: bool = False,
        seed: Optional[int] = None,
        val_root_dir: Optional[str] = None,
        fixed_noise: bool = False,
        fixed_seed: int = 999,
        split: str = "train",  # "train" | "val" | "test" | "all"
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        split_seed: int = 1234,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.image_size = tuple(image_size)
        self.transform = transform

        self.noise_factor = float(noise_factor)
        self.noise_range = noise_range
        self.noise_sampling = str(noise_sampling)
        self.use_grayscale = bool(use_grayscale)
        self.return_noise_level = bool(return_noise_level)

        self.fixed_noise = bool(fixed_noise)
        self.fixed_seed = int(fixed_seed)

        self.split = str(split).lower()
        if self.split not in ("train", "val", "test", "all"):
            raise ValueError("split ต้องเป็น 'train' หรือ 'val' หรือ 'test' หรือ 'all'")

        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)

        if self.split != "all":
            if not (0.0 < self.val_ratio < 1.0):
                raise ValueError(f"val_ratio ต้องอยู่ใน (0,1) แต่ได้ {val_ratio}")
            if not (0.0 <= self.test_ratio < 1.0):
                raise ValueError(f"test_ratio ต้องอยู่ใน [0,1) แต่ได้ {test_ratio}")
            if self.val_ratio + self.test_ratio >= 1.0:
                raise ValueError(
                    f"val_ratio + test_ratio ต้องน้อยกว่า 1.0 แต่ได้ {self.val_ratio + self.test_ratio}"
                )

        self.split_seed = int(split_seed)

        # Deterministic noise sampling (optional)
        if seed is not None:
            self._gen = torch.Generator().manual_seed(int(seed))
        else:
            self._gen = None

        # Default transform: only used for 8-bit images
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])

        # validate noise_range
        if self.noise_range is not None:
            a, b = self.noise_range
            if not (0 <= a < b):
                raise ValueError(
                    f"noise_range ต้องเป็น (min,max) ที่ 0 <= min < max, แต่ได้ {self.noise_range}"
                )

        # -------- build image_paths --------
        base_dirs, extra_train_dirs = self._resolve_base_and_extra_dirs(root_dir, val_root_dir)
        # Keep for debugging
        self.base_dirs = list(base_dirs)
        self.extra_train_dirs = list(extra_train_dirs)

        if self.split == "all":
            all_paths = self._gather_images_multi(base_dirs) + self._gather_images_multi(extra_train_dirs)
            self.image_paths = self._unique_sorted(all_paths)
        else:
            base_paths = self._gather_images_multi(base_dirs)
            train_base, val_base, test_base = self._deterministic_split_3way(
                base_paths, self.val_ratio, self.test_ratio, self.split_seed
            )

            if self.split == "val":
                self.image_paths = val_base
            elif self.split == "test":
                self.image_paths = test_base
            else:  # train
                extra_paths = self._gather_images_multi(extra_train_dirs)
                self.image_paths = self._unique_sorted(train_base + extra_paths)

    # ---------------- helpers ----------------
    def _normalize_dirs(self, d: Union[str, Sequence[str], None]) -> List[str]:
        if d is None:
            return []
        if isinstance(d, (list, tuple)):
            return [str(x) for x in d]
        return [str(d)]

    def _resolve_base_and_extra_dirs(
        self,
        root_dir: Union[str, Sequence[str]],
        val_root_dir: Optional[str],
    ) -> Tuple[List[str], List[str]]:
        roots = self._normalize_dirs(root_dir)

        if val_root_dir is not None:
            base_dirs = [str(val_root_dir)]
            extra_dirs = [d for d in roots if os.path.abspath(d) != os.path.abspath(val_root_dir)]
            return base_dirs, extra_dirs

        if len(roots) >= 2:
            return [roots[0]], roots[1:]

        return roots, []

    def _gather_images_multi(self, root_dirs: Union[str, Sequence[str]]) -> List[str]:
        dirs = self._normalize_dirs(root_dirs)
        paths: List[str] = []
        for d in dirs:
            for subdir, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".jpg", ".png", ".jpeg")):
                        paths.append(os.path.join(subdir, f))
        paths.sort()  # important for deterministic split
        return paths

    def _unique_sorted(self, paths: List[str]) -> List[str]:
        uniq = {}
        for p in paths:
            uniq[p] = True
        out = list(uniq.keys())
        out.sort()
        return out

    def _deterministic_split_3way(
        self,
        paths: List[str],
        val_ratio: float,
        test_ratio: float,
        split_seed: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split base paths deterministically into train/val/test."""
        n = len(paths)
        if n == 0:
            return [], [], []

        g = torch.Generator().manual_seed(int(split_seed))
        perm = torch.randperm(n, generator=g).tolist()

        n_test = int(round(n * float(test_ratio)))
        n_val = int(round(n * float(val_ratio)))

        # Make small datasets behave reasonably
        if test_ratio > 0 and n >= 3:
            n_test = max(1, n_test)
        if val_ratio > 0 and n >= 3:
            n_val = max(1, n_val)

        # Ensure at least 1 train sample when possible
        if n_test + n_val >= n:
            overflow = (n_test + n_val) - (n - 1)
            if overflow > 0:
                reduce_val = min(n_val, overflow)
                n_val -= reduce_val
                overflow -= reduce_val
            if overflow > 0:
                n_test = max(0, n_test - overflow)

        test_idx = set(perm[:n_test])
        val_idx = set(perm[n_test : n_test + n_val])

        train_paths = [p for i, p in enumerate(paths) if (i not in test_idx) and (i not in val_idx)]
        val_paths = [p for i, p in enumerate(paths) if i in val_idx]
        test_paths = [p for i, p in enumerate(paths) if i in test_idx]
        return train_paths, val_paths, test_paths

    # ---------------- dataset interface ----------------
    def __len__(self):
        return len(self.image_paths)

    def _sample_sigma(self, gen=None):
        if self.noise_range is None:
            return self.noise_factor

        a, b = self.noise_range
        if self.noise_sampling == "uniform":
            r = torch.rand((), generator=gen)
            return a + (b - a) * r

        if self.noise_sampling == "log_uniform":
            r = torch.rand((), generator=gen)
            a_t = torch.tensor(a, dtype=torch.float32)
            b_t = torch.tensor(b, dtype=torch.float32)
            return torch.exp(torch.log(a_t) + (torch.log(b_t) - torch.log(a_t)) * r)

        raise ValueError("noise_sampling ต้องเป็น 'uniform' หรือ 'log_uniform'")

    def _read_image_tensor(self, img_path: str) -> torch.Tensor:
        """Return CHW float tensor in [0,1], resized to self.image_size. Supports 16-bit PNG."""
        try:
            with Image.open(img_path) as im:
                # ---- 16-bit PNG ----
                if im.mode in ("I;16", "I;16B", "I;16L"):
                    im.load()  # force decode before closing file
                    a = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
                    t = torch.from_numpy(a)[None, ...]  # 1,H,W
                    t = F.interpolate(
                        t.unsqueeze(0),
                        size=tuple(self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    return t.clamp(0.0, 1.0)

                # ---- 8-bit fallback ----
                if self.use_grayscale:
                    im = im.convert("L")
                else:
                    im = im.convert("RGB")

                # detach from decoder/file
                im = im.copy()

            t = self.transform(im)  # C,H,W in [0,1]
            return t.clamp(0.0, 1.0)

        except Exception as e:
            raise RuntimeError(f"Failed to read image: {img_path} ({type(e).__name__}: {e})")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            clean_image = self._read_image_tensor(img_path)
        except Exception:
            # fallback to a neighboring sample (avoid worker crash)
            if len(self.image_paths) == 0:
                raise
            new_idx = (int(idx) + 1) % len(self.image_paths)
            img_path = self.image_paths[new_idx]
            clean_image = self._read_image_tensor(img_path)

        # fixed noise per-sample (by local idx within this split)
        if self.fixed_noise:
            gen = torch.Generator().manual_seed(int(self.fixed_seed) + int(idx))
        else:
            gen = self._gen

        sigma = self._sample_sigma(gen)

        # Rician noise: sqrt( (x + n1)^2 + (n2)^2 )
        noise_real = torch.randn(clean_image.shape, generator=gen) * sigma
        noise_imag = torch.randn(clean_image.shape, generator=gen) * sigma
        noisy_image = torch.sqrt((clean_image + noise_real) ** 2 + noise_imag ** 2)
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

        if self.return_noise_level:
            return noisy_image, clean_image, torch.tensor(float(sigma), dtype=torch.float32)

        return noisy_image, clean_image
