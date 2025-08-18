# inference.py
# I use this script to test my AI on a new satellite image.
# It converts the math logic into physical green polygon lines to map the boundaries.

import torch
import numpy as np
import tifffile as tiff
import cv2
import xarray as xr
from pathlib import Path
from model import FieldBoundaryUNet

def extract_boundaries(model_path, image_path, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FieldBoundaryUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Open 4D NetCDF and flatten it using the Median Composite technique
    with xr.open_dataset(image_path) as ds:
        arr = ds.to_array().values
        
    img_6band = np.median(arr, axis=1)
    img_6band = np.nan_to_num(img_6band)
    img_6band = np.clip(img_6band, 0, 10000) / 10000.0
    img_6band = img_6band.astype(np.float32)

    # Feed all 6 bands into the AI
    tensor = torch.from_numpy(img_6band).unsqueeze(0).to(device)
    
    print("[+] Extracting field boundaries...")
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (prediction > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # We only use the first 3 bands (RGB) to draw the visual map
    display_img = (img_6band[:3, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    
    cv2.drawContours(display_img, contours, -1, (0, 255, 0), 2)
    
    Path(save_path).parent.mkdir(exist_ok=True)
    cv2.imwrite(str(save_path), display_img)
    print(f"[+] Done! Saved map with polygon boundaries to: {save_path}")

if __name__ == "__main__":
    input_file = list(Path("datasets/curated/valid/images").glob("*.nc"))[0]
    output_file = "output/extracted_map.png"
    weights = "weights/best.pt"
    
    extract_boundaries(weights, input_file, output_file)