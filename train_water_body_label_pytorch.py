#!/usr/bin/env python3
import os
import random
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# === Configuration ===
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 10
VAL_SPLIT = 0.2
SEED = 42
LR = 1e-4
MODEL_SAVE = 'best_water_body_unet.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_COUNT = 6  # Number of samples to display

# === Model Definition ===
class DoubleConv(nn.Module):
    """Two consecutive conv-ReLU layers."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

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
        self.out = nn.Conv2d(16,out_ch,1)
    def forward(self,x):
        d1=self.d1(x); p1=self.p1(d1)
        d2=self.d2(p1); p2=self.p2(d2)
        d3=self.d3(p2); p3=self.p3(d3)
        d4=self.d4(p3); p4=self.p4(d4)
        bn=self.bn(p4)
        u4=self.u4(bn); c4=self.c4(torch.cat([u4,d4],1))
        u3=self.u3(c4); c3=self.c3(torch.cat([u3,d3],1))
        u2=self.u2(c3); c2=self.c2(torch.cat([u2,d2],1))
        u1=self.u1(c2); c1=self.c1(torch.cat([u1,d1],1))
        return torch.sigmoid(self.out(c1))

# === Dataset Definition ===
class FloodDataset(Dataset):
    def __init__(self, triplets): self.triplets = triplets
    def __len__(self): return len(self.triplets)
    def __getitem__(self, idx):
        vh_path, vv_path, mask_path = self.triplets[idx]
        vh = cv2.resize(cv2.imread(vh_path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)) / 255.0
        vv = cv2.resize(cv2.imread(vv_path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)) / 255.0
        m  = cv2.resize(cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)) / 255.0
        X = np.stack([vh, vv], axis=0).astype(np.float32)
        y = m[np.newaxis,...].astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)

# === Utilities ===
def iou_score(preds, targets, thresh=0.5):
    preds = (preds > thresh).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    union = ((preds + targets) > 0).sum(dim=(1,2,3))
    return (inter / (union + 1e-6)).mean().item()


def pixel_accuracy(preds, targets, thresh=0.5):
    preds = (preds > thresh).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return (correct / total).item()


def gather_triplets(root):
    triplets=[]
    for scene in root.iterdir():
        tiles=scene/'tiles'
        vh_d, vv_d, m_d = tiles/'vh', tiles/'vv', tiles/'water_body_label'
        if not (vh_d.exists() and vv_d.exists() and m_d.exists()): continue
        for vh_f in vh_d.glob('*_vh.png'):
            base = vh_f.stem[:-3]
            vv_f = vv_d/f"{base}_vv.png"; m_f = m_d/f"{base}.png"
            if vv_f.exists() and m_f.exists(): triplets.append((str(vh_f),str(vv_f),str(m_f)))
    return triplets

# === Plotting Functions ===
def plot_metrics(train_vals, val_vals, ylabel, title, fname):
    epochs = range(1, len(train_vals)+1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_vals, label=f'train {ylabel}')
    plt.plot(epochs, val_vals, label=f'val {ylabel}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def show_sample_preds(model, dataset, fname, n=SAMPLE_COUNT):
    idxs = random.sample(range(len(dataset)), n)
    plt.figure(figsize=(n*3, 9))
    for i, idx in enumerate(idxs):
        X, _ = dataset[idx]
        vh, vv = X[0].numpy(), X[1].numpy()
        inp = X.unsqueeze(0).to(DEVICE)
        with torch.no_grad(): pred = model(inp)[0,0].cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8)
        # Plot VH
        plt.subplot(3, n, i+1); plt.imshow(vh, cmap='gray'); plt.axis('off')
        # Plot VV
        plt.subplot(3, n, i+1+n); plt.imshow(vv, cmap='gray'); plt.axis('off')
        # Plot Predicted Mask
        plt.subplot(3, n, i+1+2*n); plt.imshow(mask, cmap='Blues'); plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# === Training Loop ===
def main():
    multiprocessing.freeze_support()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if DEVICE.type=='cuda': torch.cuda.manual_seed(SEED)
    root = Path(__file__).parent
    all_t = gather_triplets(root)
    print(f"Found {len(all_t)} samples total.")
    random.shuffle(all_t)
    split = int(len(all_t)*(1-VAL_SPLIT))
    train_l, val_l = all_t[:split], all_t[split:]
    print(f"Training: {len(train_l)}, Validation: {len(val_l)}")
    train_ds = FloodDataset(train_l)
    val_ds = FloodDataset(val_l)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # Histories
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    train_iou_hist, val_iou_hist = [], []
    best_val = float('inf')

    for epoch in range(1, EPOCHS+1):
        start = time.time()
        model.train()
        t_loss=t_iou=t_acc=0
        for X,y in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            X,y = X.to(DEVICE), y.to(DEVICE)
            p = model(X)
            loss = criterion(p, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            bs = X.size(0)
            t_loss += loss.item()*bs
            t_iou  += iou_score(p, y)*bs
            t_acc  += pixel_accuracy(p, y)*bs
        t_loss /= len(train_dl.dataset)
        t_iou  /= len(train_dl.dataset)
        t_acc  /= len(train_dl.dataset)

        model.eval()
        v_loss=v_iou=v_acc=0
        for X,y in tqdm(val_dl, desc=f"Epoch {epoch}/{EPOCHS} [Val]  "):
            X,y = X.to(DEVICE), y.to(DEVICE)
            with torch.no_grad(): p = model(X)
            bs = X.size(0)
            v_loss += criterion(p, y).item()*bs
            v_iou  += iou_score(p, y)*bs
            v_acc  += pixel_accuracy(p, y)*bs
        v_loss /= len(val_dl.dataset)
        v_iou  /= len(val_dl.dataset)
        v_acc  /= len(val_dl.dataset)
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']

        # Save history
        train_loss_hist.append(t_loss); val_loss_hist.append(v_loss)
        train_acc_hist.append(t_acc);   val_acc_hist.append(v_acc)
        train_iou_hist.append(t_iou);   val_iou_hist.append(v_iou)

        if v_loss < best_val:
            print(f"Epoch {epoch}: val_loss improved from {best_val:.5f} to {v_loss:.5f}, saving model to {MODEL_SAVE}")
            torch.save(model.state_dict(), MODEL_SAVE)
            best_val = v_loss
        print(f"{epoch}/{EPOCHS} - {elapsed:.1f}s - loss: {t_loss:.4f} - acc: {t_acc:.4f} - iou: {t_iou:.4f} - \
              val_loss: {v_loss:.4f} - val_acc: {v_acc:.4f} - val_iou: {v_iou:.4f} - lr: {lr:.1e}")

    # Plot and save training metrics
    plot_metrics(train_loss_hist, val_loss_hist, 'loss', 'Training & Validation Loss', 'water_body_label_training_metrics_pytorch.png')
    plot_metrics(train_acc_hist,  val_acc_hist,  'accuracy', 'Training & Validation Accuracy', 'water_body_label_training_metrics_pytorch.png')
    plot_metrics(train_iou_hist,  val_iou_hist,  'IoU', 'Training & Validation IoU', 'water_body_label_training_metrics_pytorch.png')

    # Save sample predictions
    show_sample_preds(model, val_ds, 'water_body_label_sample_predictions_pytorch.png', n=SAMPLE_COUNT)

    print(f"✅ Training finished. Best model saved to {MODEL_SAVE}")

if __name__=='__main__': main()
