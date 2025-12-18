import torch
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, image_size=(64, 64), transform=None, noise_factor=0.2, use_grayscale=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.noise_factor = noise_factor
        self.use_grayscale = use_grayscale
        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))

        # default transform: Resize -> ToTensor
        if self.transform is None:
            tfs = [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
            self.transform = transforms.Compose(tfs)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # (ทางเลือก) ทำให้เป็น Grayscale ถ้าต้องการ 1 ช่องสัญญาณ
        if self.use_grayscale:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        clean_image = self.transform(image)

        # Rician noise
        noise_real = torch.randn_like(clean_image) * self.noise_factor
        noise_imag = torch.randn_like(clean_image) * self.noise_factor
        noisy_image = torch.sqrt((clean_image + noise_real) ** 2 + noise_imag ** 2)
        noisy_image = torch.clamp(noisy_image, 0., 1.)

        return noisy_image, clean_image