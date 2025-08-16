# evaluate.py
# This generates my final performance graph comparing my AI to the ground truth.

import torch
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from model import FieldBoundaryUNet

def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0: return 0
    return intersection / union

def evaluate_model(model_path, image_path, mask_path):
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

    ground_truth = tiff.imread(str(mask_path))
    
    # CRITICAL FIX: Strip the extra junk color channels from the Kaggle mask 
    # so it matches the AI's 1-channel prediction.
    if len(ground_truth.shape) == 3:
        ground_truth = ground_truth[:, :, 0]
        
    ground_truth = (ground_truth > 0).astype(np.uint8)
    
    tensor = torch.from_numpy(img_6band).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (prediction > 0.5).astype(np.uint8)

    iou_score = calculate_iou(binary_mask, ground_truth)
    print(f"=== MY AI'S REPORT CARD ===")
    print(f"Intersection over Union (IoU): {iou_score:.4f}")

    # Generate RGB visual image and boost the brightness by 3.5x so we can see it
    vis_img = img_6band[:3, :, :].transpose(1, 2, 0)
    vis_img = np.clip(vis_img * 255 * 3.5, 0, 255).astype(np.uint

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(vis_img)
    axes[0].set_title("Raw Satellite Image")
    axes[0].axis('off')
    
    axes[1].imshow(ground_truth, cmap='gray')
    axes[1].set_title("The Perfect Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title(f"My AI Prediction\nIoU Score: {iou_score:.4f}")
    axes[2].axis('off')
    
    Path("output").mkdir(exist_ok=True)
    out_file = "output/performance_graph.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"[+] Saved the performance graph to {out_file}")

if __name__ == "__main__":
    valid_images = list(Path("datasets/curated/valid/images").glob("*.nc"))
    if not valid_images:
        print("[-] Run data_prep.py first.")
    else:
        test_img = valid_images[0]
        # Match the .nc file to its .tif mask
        test_mask = Path("datasets/curated/valid/masks") / (test_img.stem + ".tif")
        evaluate_model("weights/best.pt", test_img, test_mask)