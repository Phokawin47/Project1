from __future__ import annotations
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from ..registry import DATASETS

@DATASETS.register("brain_tumor_rician")
class BrainTumorRicianDataset(Dataset):
    # Image-domain Rician noise dataset. Returns (noisy, clean) or (noisy, clean, sigma).
    def __init__(
        self,
        root_dir: str,
        image_size=(64, 64),
        transform=None,
        noise_factor: float = 0.2,
        noise_range=None,
        noise_sampling: str = "uniform",
        use_grayscale: bool = False,
        return_noise_level: bool = False,
        seed: int | None = None,
    ):
        self.root_dir = root_dir
        self.image_size = tuple(image_size)
        self.transform = transform
        self.noise_factor = float(noise_factor)
        self.noise_range = noise_range
        self.noise_sampling = noise_sampling
        self.use_grayscale = use_grayscale
        self.return_noise_level = return_noise_level
        self._g = None
        if seed is not None:
            self._g = torch.Generator()
            self._g.manual_seed(int(seed))

        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    self.image_paths.append(os.path.join(subdir, f))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under: {root_dir}")

        if self.transform is None:
            self.transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def _sample_sigma(self) -> float:
        if self.noise_range is None:
            return float(self.noise_factor)
        lo, hi = float(self.noise_range[0]), float(self.noise_range[1])
        if self.noise_sampling == "log_uniform":
            lo2, hi2 = torch.log(torch.tensor(lo)), torch.log(torch.tensor(hi))
            u = torch.rand((), generator=self._g) if self._g is not None else torch.rand(())
            return float(torch.exp(lo2 + (hi2 - lo2) * u).item())
        u = torch.rand((), generator=self._g) if self._g is not None else torch.rand(())
        return float((lo + (hi - lo) * u).item())

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L" if self.use_grayscale else "RGB")
        clean = self.transform(img)
        sigma = self._sample_sigma()

        if self._g is not None:
            noise_real = torch.randn_like(clean, generator=self._g) * sigma
            noise_imag = torch.randn_like(clean, generator=self._g) * sigma
        else:
            noise_real = torch.randn_like(clean) * sigma
            noise_imag = torch.randn_like(clean) * sigma

        noisy = torch.sqrt((clean + noise_real) ** 2 + noise_imag ** 2).clamp(0.0, 1.0)
        if self.return_noise_level:
            return noisy, clean, torch.tensor(sigma, dtype=torch.float32)
        return noisy, clean
