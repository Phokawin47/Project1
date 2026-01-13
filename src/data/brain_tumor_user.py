from __future__ import annotations

import os
from typing import List, Optional, Tuple, Sequence, Union

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

from ..registry import DATASETS


@DATASETS.register("brain_tumor_dataset")
class BrainTumorDataset(Dataset):
    """
    Mode แบบที่ 2:
    - ถ้า root_dir เป็น list/tuple:
        folder แรก = base_dir ที่จะ split เป็น train/val (ตาม val_ratio + split_seed)
        folder ที่เหลือ = extra train (ไม่เข้า val)
      => val ถูกสุ่มจาก folder แรกเท่านั้น และไม่ซ้ำกับ train
    - ถ้า root_dir เป็น str เดี่ยว:
        จะ split จาก root_dir เดียวแบบเดิม
    - ถ้ากำหนด val_root_dir มา:
        จะถือว่า val_root_dir คือ base_dir ที่จะ split (แทน folder แรก)
        และ root_dir (ถ้าเป็น list) ส่วนที่เหลือจะเป็น extra train
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
        seed: Optional[int] = None,          # seed สำหรับ sampling noise (ทั้ง dataset)
        val_root_dir: Optional[str] = None,  # (optional) ให้บังคับ base_dir ที่จะ split เป็น train/val
        fixed_noise: bool = False,
        fixed_seed: int = 999,

        split: str = "train",                # "train" | "val" | "all"
        val_ratio: float = 0.2,
        split_seed: int = 1234,
        **kwargs
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform

        self.noise_factor = float(noise_factor)
        self.noise_range = noise_range
        self.noise_sampling = noise_sampling
        self.use_grayscale = bool(use_grayscale)
        self.return_noise_level = bool(return_noise_level)

        self.fixed_noise = bool(fixed_noise)
        self.fixed_seed = int(fixed_seed)

        self.split = str(split).lower()
        if self.split not in ("train", "val", "all"):
            raise ValueError("split ต้องเป็น 'train' หรือ 'val' หรือ 'all'")

        self.val_ratio = float(val_ratio)
        if not (0.0 < self.val_ratio < 1.0):
            raise ValueError(f"val_ratio ต้องอยู่ใน (0,1) แต่ได้ {val_ratio}")

        self.split_seed = int(split_seed)

        # ทำให้การสุ่ม noise reproducible (ถ้ากำหนด seed)
        if seed is not None:
            self._gen = torch.Generator().manual_seed(int(seed))
        else:
            self._gen = None

        # transform default
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])

        # validate noise_range
        if self.noise_range is not None:
            a, b = self.noise_range
            if not (0 <= a < b):
                raise ValueError(f"noise_range ต้องเป็น (min,max) ที่ 0 <= min < max, แต่ได้ {self.noise_range}")

        # -------- build image_paths (แบบที่ 2) --------
        base_dirs, extra_train_dirs = self._resolve_base_and_extra_dirs(root_dir, val_root_dir)

        if self.split == "all":
            # all = รวมทุกอย่าง (base + extra)
            all_paths = self._gather_images_multi(base_dirs) + self._gather_images_multi(extra_train_dirs)
            self.image_paths = self._unique_sorted(all_paths)
        else:
            # split จาก base เท่านั้น
            base_paths = self._gather_images_multi(base_dirs)
            train_base, val_base = self._deterministic_split(base_paths, self.val_ratio, self.split_seed)

            if self.split == "val":
                self.image_paths = val_base
            else:
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

        # ถ้าผู้ใช้กำหนด val_root_dir -> ให้เป็น base_dir (split จากตรงนี้)
        if val_root_dir is not None:
            base_dirs = [str(val_root_dir)]
            # extra train = root_dir ทั้งหมดที่ "ไม่ใช่ val_root_dir"
            extra_dirs = [d for d in roots if os.path.abspath(d) != os.path.abspath(val_root_dir)]
            return base_dirs, extra_dirs

        # ไม่กำหนด val_root_dir:
        # - ถ้า root_dir เป็น list: folderแรกเป็น base, ที่เหลือเป็น extra train
        if len(roots) >= 2:
            return [roots[0]], roots[1:]

        # - ถ้า root_dir เดี่ยว: base = root_dir, extra = []
        return roots, []

    def _gather_images_multi(self, root_dirs: Union[str, Sequence[str]]) -> List[str]:
        dirs = self._normalize_dirs(root_dirs)
        paths: List[str] = []
        for d in dirs:
            for subdir, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".jpg", ".png", ".jpeg")):
                        paths.append(os.path.join(subdir, f))
        paths.sort()  # สำคัญมาก: ทำให้ split deterministic
        return paths

    def _unique_sorted(self, paths: List[str]) -> List[str]:
        # ลบซ้ำแบบ deterministic
        uniq = {}
        for p in paths:
            uniq[p] = True
        out = list(uniq.keys())
        out.sort()
        return out

    def _deterministic_split(self, paths: List[str], val_ratio: float, split_seed: int):
        n = len(paths)
        if n == 0:
            return [], []
        n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0

        g = torch.Generator().manual_seed(int(split_seed))
        perm = torch.randperm(n, generator=g).tolist()

        val_idx = set(perm[:n_val])
        train_paths = [p for i, p in enumerate(paths) if i not in val_idx]
        val_paths   = [p for i, p in enumerate(paths) if i in val_idx]
        return train_paths, val_paths

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
        """
        Return CHW float tensor in [0,1] resized to self.image_size.
        Supports PNG 16-bit (I;16) and normal 8-bit images.
        """
        im = Image.open(img_path)

        # ---- 16-bit PNG path ----
        if im.mode in ("I;16", "I;16B", "I;16L"):
            a = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0  # H,W in [0,1]
            t = torch.from_numpy(a)[None, ...]  # 1,H,W
            # resize with torch (preserve range)
            t = F.interpolate(
                t.unsqueeze(0),
                size=tuple(self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            return t.clamp(0.0, 1.0)

        # ---- 8-bit fallback (เดิม) ----
        if self.use_grayscale:
            im = im.convert("L")
        else:
            im = im.convert("RGB")

        # ใช้ transform เดิมได้
        t = self.transform(im)  # C,H,W in [0,1]
        return t.clamp(0.0, 1.0)


    def __getitem__(self, idx):
        
        img_path = self.image_paths[idx]
        clean_image = self._read_image_tensor(img_path)


        # fixed_noise ต่อ sample (หมายเหตุ: idx จะเปลี่ยนตาม split)
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
