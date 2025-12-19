import torch
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

import torch
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_size=(64, 64),
        transform=None,
        noise_factor=0.2,          # ใช้กรณี fix ค่าเดียว
        noise_range=None,          # เช่น (0.2, 0.5) เพื่อสุ่มหลายช่วง
        noise_sampling="uniform",  # "uniform" หรือ "log_uniform"
        use_grayscale=False,
        return_noise_level=False,  # ถ้าต้องการให้ return sigma ออกมาด้วย
        seed=None                  # ถ้าต้องการ reproducible (ทางเลือก)
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.noise_factor = float(noise_factor)
        self.noise_range = noise_range
        self.noise_sampling = noise_sampling
        self.use_grayscale = use_grayscale
        self.return_noise_level = return_noise_level

        if seed is not None:
            # ทำให้การสุ่มใน dataset reproducible ได้ (ต่อ worker อาจต้องปรับเพิ่ม)
            self._gen = torch.Generator().manual_seed(seed)
        else:
            self._gen = None

        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])

        if self.noise_range is not None:
            a, b = self.noise_range
            if not (0 <= a < b):
                raise ValueError(f"noise_range ต้องเป็น (min,max) ที่ 0 <= min < max, แต่ได้ {self.noise_range}")

    def __len__(self):
        return len(self.image_paths)

    def _sample_sigma(self):
        """สุ่มค่า noise sigma ต่อ sample"""
        if self.noise_range is None:
            return self.noise_factor

        a, b = self.noise_range

        if self.noise_sampling == "uniform":
            r = torch.rand((), generator=self._gen)
            return a + (b - a) * r

        if self.noise_sampling == "log_uniform":
            # เหมาะเมื่ออยากให้ค่าต่ำ/สูงไม่กระจุก
            r = torch.rand((), generator=self._gen)
            return torch.exp(torch.log(torch.tensor(a)) + (torch.log(torch.tensor(b)) - torch.log(torch.tensor(a))) * r)

        raise ValueError("noise_sampling ต้องเป็น 'uniform' หรือ 'log_uniform'")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.use_grayscale:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        clean_image = self.transform(image)

        sigma = self._sample_sigma()

        # Rician noise: sqrt( (x + n1)^2 + (n2)^2 )
        noise_real = torch.randn_like(clean_image) * sigma
        noise_imag = torch.randn_like(clean_image) * sigma
        noisy_image = torch.sqrt((clean_image + noise_real) ** 2 + noise_imag ** 2)
        noisy_image = torch.clamp(noisy_image, 0., 1.)

        if self.return_noise_level:
            return noisy_image, clean_image, torch.tensor(sigma, dtype=torch.float32)
        return noisy_image, clean_image
