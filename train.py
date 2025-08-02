# train.py
# This is my main training loop. I updated this to be extremely careful with satellite data.
# Satellite images can be 8-bit or 16-bit. If I normalize them wrong, the AI learns nothing.
# I also added an automatic save feature that only saves the model if it actually gets smarter.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import tifffile as tiff
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xarray as xr

from model import FieldBoundaryUNet

class SatelliteDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(list(Path(img_dir).glob("*.nc")))
        self.mask_paths = sorted(list(Path(mask_dir).glob("*.tif")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Open the 4D NetCDF file (Bands x Time x Height x Width)
        with xr.open_dataset(self.img_paths[idx]) as ds:
            arr = ds.to_array().values
            
        # 2. FLATTEN THE TIME SERIES (Median Composite)
        img = np.median(arr, axis=1)
        img = np.nan_to_num(img)
        img = np.clip(img, 0, 10000) / 10000.0
        img = img.astype(np.float32)

        # 3. Load the corresponding .tif Mask
        mask = tiff.imread(str(self.mask_paths[idx]))
        
        # CRITICAL FIX: If the Kaggle mask loads as RGBA (4 channels) or RGB (3 channels), 
        # we force it down to a single channel (Height x Width)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
            
        mask = (mask > 0).astype(np.float32)
        
        # Ensure PyTorch format (1, Height, Width)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(img), torch.from_numpy(mask)

def calculate_iou(preds, targets):
    preds = torch.sigmoid(preds) > 0.5
    intersection = (preds & (targets > 0.5)).float().sum()
    union = (preds | (targets > 0.5)).float().sum()
    if union == 0: return 0
    return intersection / union

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Starting training on: {device}")
    
    batch_size = 8
    epochs = 25 
    
    train_data = SatelliteDataset("datasets/curated/train/images", "datasets/curated/train/masks")
    valid_data = SatelliteDataset("datasets/curated/valid/images", "datasets/curated/valid/masks")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    
    model = FieldBoundaryUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler('cuda')
    
    Path("weights").mkdir(exist_ok=True)
    best_iou = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for imgs, masks in bar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # Validation Loop
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_iou += calculate_iou(preds, masks).item()
                
        val_iou /= len(valid_loader)
        print(f"Epoch {epoch} Summary | Train Loss: {train_loss/len(train_loader):.4f} | Valid IoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "weights/best.pt")
            print(f"[+] AI improved! Saved new best model with IoU: {best_iou:.4f}")

if __name__ == "__main__":
    train_model()