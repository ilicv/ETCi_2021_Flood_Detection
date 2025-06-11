#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path

# === Configuration ===
# Root directory containing scene subfolders with 'tiles/...'
# Script should be run from the dataset root (e.g. .../_reduced_dataset_test)
gt_root = Path.cwd()
# Output folder for comparison composites
diff_root = gt_root / 'mask_comparison'
# Resize images for display
img_size = 256

# === Gather tile-image & mask pairs ===
def gather_label_pairs(root):
    """
    For each scene under root, find matching VH, VV, flood_label, and water_body_label images.
    Returns list of tuples: (vh_path, vv_path, flood_path, water_path).
    """
    pairs = []
    for flood_dir in root.rglob('tiles/flood_label'):
        tiles_dir = flood_dir.parent
        water_dir = tiles_dir / 'water_body_label'
        vh_dir = tiles_dir / 'vh'
        vv_dir = tiles_dir / 'vv'
        if not (water_dir.exists() and vh_dir.exists() and vv_dir.exists()):
            continue
        for flood_path in flood_dir.glob('*.png'):
            base = flood_path.stem
            water_path = water_dir / f"{base}.png"
            vh_path    = vh_dir    / f"{base}_vh.png"
            vv_path    = vv_dir    / f"{base}_vv.png"
            if water_path.exists() and vh_path.exists() and vv_path.exists():
                pairs.append((vh_path, vv_path, flood_path, water_path))
    return pairs

# === Save comparison composites ===
def save_comparisons(pairs, root_out):
    """
    For each tile, build a composite of six images side-by-side:
    [VH grayscale] [VV grayscale] [flood mask in red] [water mask in blue] [mask overlay] [overlay on VH]
    Differences: red for flood-only, blue for water-only, yellow for both.
    """
    for vh_path, vv_path, flood_path, water_path in pairs:
        # Load and resize images
        vh_img    = cv2.resize(cv2.imread(str(vh_path), cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        vv_img    = cv2.resize(cv2.imread(str(vv_path), cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        flood_img = cv2.resize(cv2.imread(str(flood_path), cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        water_img = cv2.resize(cv2.imread(str(water_path), cv2.IMREAD_GRAYSCALE), (img_size, img_size))

        # Convert VH and VV to BGR
        vh_color = cv2.cvtColor(vh_img, cv2.COLOR_GRAY2BGR)
        vv_color = cv2.cvtColor(vv_img, cv2.COLOR_GRAY2BGR)

        # Red flood mask, Blue water mask
        flood_color = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        flood_color[..., 2] = flood_img  # red channel
        water_color = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        water_color[..., 0] = water_img  # blue channel

        # Compute overlay mask
        mask_f = flood_img > 0
        mask_w = water_img > 0
        overlay = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        overlay[mask_f & ~mask_w] = [0, 0, 255]      # flood-only: red
        overlay[mask_w & ~mask_f] = [255, 0, 0]      # water-only: blue
        overlay[mask_f & mask_w] = [0, 255, 255]     # both: yellow

        # Overlay on VH: start with VH and apply same overlay colors
        overlay_on_vh = vh_color.copy()
        overlay_on_vh[mask_f & ~mask_w] = [0, 0, 255]
        overlay_on_vh[mask_w & ~mask_f] = [255, 0, 0]
        overlay_on_vh[mask_f & mask_w] = [0, 255, 255]

        # Concatenate six panels horizontally
        composite = np.hstack([vh_color, vv_color, flood_color,
                               water_color, overlay, overlay_on_vh])

        # Determine output path, preserving scene structure
        rel = flood_path.relative_to(gt_root)
        scene = rel.parts[0]
        tile_name = flood_path.stem
        out_dir = root_out / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{tile_name}_comparison.png"
        cv2.imwrite(str(out_file), composite)

# === Main ===
def main():
    print(f"Dataset root: {gt_root}")
    print(f"Saving comparison composites to: {diff_root}\n")
    pairs = gather_label_pairs(gt_root)
    # Filter only those with any mask differences
    diff_pairs = []
    for vh_p, vv_p, f_p, w_p in pairs:
        f_img = cv2.imread(str(f_p), cv2.IMREAD_GRAYSCALE)
        w_img = cv2.imread(str(w_p), cv2.IMREAD_GRAYSCALE)
        if not np.array_equal(f_img, w_img):
            diff_pairs.append((vh_p, vv_p, f_p, w_p))
    print(f"Found {len(diff_pairs)} tiles with flood vs water differences.")
    save_comparisons(diff_pairs, diff_root)
    print("Done. Composite images saved for each differing tile.")

if __name__ == '__main__':
    main()
