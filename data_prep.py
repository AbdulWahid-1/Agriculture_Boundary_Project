# data_prep.py
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def prepare_multispectral_subset():
    source_dir = Path("datasets/ai4boundaries")
    
    # We are extracting exactly 1,500 images to keep training fast and safe for 6GB VRAM
    TRAIN_LIMIT = 1500
    VALID_LIMIT = 300
    
    target_dir = Path("datasets/curated")
    for split in ["train", "valid"]:
        (target_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    # Kaggle pre-split folders for Sentinel-2 data
    folder_mappings = [
        ("train", "train", "masks/train", TRAIN_LIMIT),
        ("valid", "val", "masks/val", VALID_LIMIT)
    ]

    for split_name, img_sub, mask_sub, limit in folder_mappings:
        img_dir = source_dir / img_sub
        mask_dir = source_dir / mask_sub
        
        if not img_dir.exists() or not mask_dir.exists():
            print(f"[-] Missing Kaggle folder: {img_dir} or {mask_dir}")
            continue

        print(f"[+] Scanning {split_name} Sentinel-2 data...")
        all_imgs = list(img_dir.rglob("*.nc"))
        
        paired_data = []
        for img_path in all_imgs:
            # The Kaggle mask is named exactly the same but ends in .tif
            mask_name = img_path.stem + ".tif"
            mask_path = mask_dir / mask_name
            
            if mask_path.exists():
                paired_data.append((img_path, mask_path))
                
        # Shuffle for a diverse agricultural sample
        random.seed(42)
        random.shuffle(paired_data)
        subset = paired_data[:limit]
        
        print(f"[+] Copying {len(subset)} Multi-spectral {split_name} pairs...")
        for img, mask in tqdm(subset):
            shutil.copy(img, target_dir / split_name / "images" / img.name)
            shutil.copy(mask, target_dir / split_name / "masks" / mask.name)

    print("[+] Dataset preparation complete. Ready for Multi-spectral training.")

if __name__ == "__main__":
    prepare_multispectral_subset()