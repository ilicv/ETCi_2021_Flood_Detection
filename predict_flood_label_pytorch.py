#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# === Configuration ===
IMG_SIZE = 256                   # Resize input tiles
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_WEIGHTS = 'best_flood_unet.pth'  # PyTorch model weights file
DST_FOLDER = 'predicted_results'      # Where to save predicted masks
THRESHOLD = 0.5                      # Probability threshold

# === Model Definition ===
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1):
        super().__init__()
        self.d1 = DoubleConv(in_ch, 16); self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(16, 32);   self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(32, 64);   self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(64,128);   self.p4 = nn.MaxPool2d(2)
        self.bn = DoubleConv(128,256)
        self.u4 = nn.ConvTranspose2d(256,128,2,2); self.c4 = DoubleConv(256,128)
        self.u3 = nn.ConvTranspose2d(128,64,2,2);  self.c3 = DoubleConv(128,64)
        self.u2 = nn.ConvTranspose2d(64,32,2,2);   self.c2 = DoubleConv(64,32)
        self.u1 = nn.ConvTranspose2d(32,16,2,2);   self.c1 = DoubleConv(32,16)
        self.out = nn.Conv2d(16, out_ch, kernel_size=1)
    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3); p4 = self.p4(d4)
        bn = self.bn(p4)
        u4 = self.u4(bn); c4 = self.c4(torch.cat([u4, d4], dim=1))
        u3 = self.u3(c4); c3 = self.c3(torch.cat([u3, d3], dim=1))
        u2 = self.u2(c3); c2 = self.c2(torch.cat([u2, d2], dim=1))
        u1 = self.u1(c2); c1 = self.c1(torch.cat([u1, d1], dim=1))
        return torch.sigmoid(self.out(c1))

# === Utility Functions ===
def replicate_dirs(src_root, dst_root):
    for root, dirs, _ in os.walk(src_root):
        dirs[:] = [d for d in dirs if d != DST_FOLDER]
        rel = os.path.relpath(root, src_root)
        os.makedirs(os.path.join(dst_root, rel), exist_ok=True)

def predict_and_save(model, vh_path, vv_path, out_path):
    # load grayscale
    vh = cv2.imread(vh_path, cv2.IMREAD_GRAYSCALE)
    vv = cv2.imread(vv_path, cv2.IMREAD_GRAYSCALE)
    # resize and normalize
    vh_t = cv2.resize(vh, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    vv_t = cv2.resize(vv, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    inp = np.stack([vh_t, vv_t], axis=0)[None, ...]  # [1,2,H,W]
    tensor = torch.from_numpy(inp).to(DEVICE)
    # predict
    model.eval()
    with torch.no_grad():
        pred = model(tensor)[0,0].cpu().numpy()
    mask = (pred > THRESHOLD).astype(np.uint8) * 255
    # create composite
    vh_color = cv2.cvtColor((vh_t*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vv_color = cv2.cvtColor((vv_t*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = vh_color.copy()
    overlay[mask==255] = [0,0,255]
    comp = np.hstack([vh_color, vv_color, mask_color, overlay])
    cv2.imwrite(out_path, comp)

# === Main ===
def main():
    src_root = os.getcwd()
    dst_root = os.path.join(src_root, DST_FOLDER)
    print(f"Loading model weights '{MODEL_WEIGHTS}' on {DEVICE}...")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    print("Model loaded. Preparing output directories...")
    replicate_dirs(src_root, dst_root)
    # traverse VH folders
    for root, dirs, files in os.walk(src_root):
        dirs[:] = [d for d in dirs if d != DST_FOLDER]
        if os.path.basename(root).lower() == 'vh':
            rel_dir = os.path.relpath(root, src_root)
            base_dir = os.path.dirname(rel_dir)
            vv_dir = os.path.join(src_root, base_dir, 'vv')
            out_dir = os.path.join(dst_root, base_dir, 'flood_label')
            os.makedirs(out_dir, exist_ok=True)
            for fn in files:
                if not fn.lower().endswith('_vh.png'):
                    continue
                base = fn[:-7]
                vh_path = os.path.join(root, fn)
                vv_path = os.path.join(vv_dir, f"{base}_vv.png")
                out_path = os.path.join(out_dir, f"{base}.png")
                if os.path.exists(vv_path):
                    predict_and_save(model, vh_path, vv_path, out_path)
                else:
                    print(f"⚠️ Missing VV for {fn}, skipping.")
    print(f"✅ Prediction complete. Results in '{DST_FOLDER}'")

if __name__ == '__main__':
    main()
