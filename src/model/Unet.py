import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data import BrainTumorDataset
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def main():
    # ====== Config ======
    root_dir = "./data"  # <- แก้เป็นโฟลเดอร์รูปของคุณ
    image_size = (288, 288)  # กำหนดขนาดรูป (height, width)
    use_grayscale = True  # ถ้า True จะใช้ 1 ช่องสัญญาณ (อย่าลืมปรับโมเดลข้างล่าง)
    noise_factor = 0.2
    noise_range=(0.2, 0.5)
    batch_size = 16
    val_ratio = 0.2
    num_workers= 4
    # ====== Dataset / Split ======
    dataset = BrainTumorDataset(root_dir, image_size=image_size, transform=None, noise_factor=noise_factor, noise_range=noise_range, return_noise_level=True, use_grayscale=use_grayscale)
    n_total = len(dataset)
    n_test = int(n_total * 0.1)  # 10% for test
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ====== แสดง Shape ของข้อมูล ======
    sample_noisy, sample_clean, noise = next(iter(train_loader))
    channels = 1 if use_grayscale else 3
    print(f"Dataset Info:")
    print(f"  - Total images: {n_total}")
    print(f"  - Train images: {n_train}")
    print(f"  - Val images: {n_val}")
    print(f"  - Test images: {n_test}")
    print(f"  - Image size: {image_size}")
    print(f"  - Noise factor: {noise}")
    print(f"  - Channels: {channels} ({'Grayscale' if use_grayscale else 'RGB'})")
    print(f"  - Batch shape: {sample_noisy.shape} (batch_size, channels, height, width)")
    print(f"  - Device: {device}")
    print("-" * 50)

if __name__ == "__main__":
    main()