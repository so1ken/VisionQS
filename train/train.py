# train.py
# Обучение простого сверточного автоэнкодера (PyTorch) для anomaly detection
# Тренируется только на хороших изображениях (data/train_good) и вычисляет порог на val_good.
# Сохраняет model.pth и threshold.txt в папку server/model/

import argparse
import os
from pathlib import Path
from PIL import Image
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ------------------------------
# Простая Dataset, загружает все изображения из папки (flat folder)
# ------------------------------
class ImageFolderFlat(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = Path(folder)
        self.paths = [p for p in sorted(self.folder.iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, p.name


# ------------------------------
# Conv Autoencoder (тот же, что на сервере)
# ------------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out


# ------------------------------
# Утилиты
# ------------------------------

def get_transforms(img_size, augment=False):
    t = []
    if augment:
        # простые аугментации для robustness
        t += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    t += [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    return transforms.Compose(t)


def compute_per_image_scores(model, loader, device):
    model.eval()
    losses = []
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for x, names in loader:
            x = x.to(device)
            x_hat = model(x)
            per_pixel = criterion(x_hat, x)
            per_img = per_pixel.view(per_pixel.size(0), -1).mean(dim=1)
            losses.extend(per_img.cpu().numpy().tolist())
    return np.array(losses)


# ------------------------------
# Training routine
# ------------------------------

def train(args):
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu')
    print('Device:', device)

    # prepare datasets
    train_dir = Path(args.data_dir) / 'train_good'
    val_dir = Path(args.data_dir) / 'val_good'
    assert train_dir.exists(), f"Train folder not found: {train_dir}"
    assert val_dir.exists(), f"Val folder not found: {val_dir}"

    train_tf = get_transforms(args.img_size, augment=args.augment)
    val_tf = get_transforms(args.img_size, augment=False)

    train_ds = ImageFolderFlat(train_dir, transform=train_tf)
    val_ds = ImageFolderFlat(val_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    out_dir = Path(args.out_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, _ in pbar:
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({'train_loss': f"{np.mean(train_losses):.6f}"})

        # compute per-image scores on val
        val_scores = compute_per_image_scores(model, val_loader, device)
        val_mean = float(np.mean(val_scores))
        print(f"Epoch {epoch}: train_loss={np.mean(train_losses):.6f} val_mean_score={val_mean:.6f}")

        # save best model (by mean val score)
        if val_mean < best_val:
            best_val = val_mean
            model_path = out_dir / 'model.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")

    # final: load best model and compute threshold
    best_model_path = out_dir / 'model.pth'
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_val_scores = compute_per_image_scores(model, val_loader, device)

    # threshold as percentile
    thresh = float(np.percentile(final_val_scores, args.threshold_percentile))
    thresh_path = out_dir / 'threshold.txt'
    with open(thresh_path, 'w') as f:
        f.write(str(thresh))

    print(f"Final threshold (percentile={args.threshold_percentile}) = {thresh:.6f}")
    print(f"Saved threshold to {thresh_path}")


# ------------------------------
# CLI
# ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='data folder with train_good/ and val_good/')
    parser.add_argument('--out-model-dir', type=str, default='server/model', help='where to save model and threshold')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--threshold-percentile', type=float, default=95.0)
    parser.add_argument('--augment', action='store_true', help='use simple augmentations during training')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', type=str, choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args()

    train(args)
