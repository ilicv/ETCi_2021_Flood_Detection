#!/usr/bin/env python3
import os
from pathlib import Path
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# === Configuration ===
IMG_SIZE = 256             # Resize tiles to 256x256
BATCH_SIZE = 16            # Training batch size
EPOCHS = 10                # Max epochs
VAL_SPLIT = 0.2            # Fraction for validation
SEED = 42                  # Random seed for reproducibility
MODEL_NAME = 'best_water_body_unet.h5'

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
# Assume script in dataset root; scenes are subfolders containing 'tiles'
dataset_root = Path(__file__).parent
vh_folder = 'vh'
vv_folder = 'vv'
mask_folder = 'water_body_label'

# === Data gathering ===

def gather_tile_paths(root):
    """
    Walk through dataset_root, find matching VH, VV and water-body label tiles.
    Returns list of (vh_path, vv_path, mask_path).
    """
    triplets = []
    for scene_dir in root.iterdir():
        tiles_dir = scene_dir / 'tiles'
        vh_dir = tiles_dir / vh_folder
        vv_dir = tiles_dir / vv_folder
        mask_dir = tiles_dir / mask_folder
        if not (vh_dir.is_dir() and vv_dir.is_dir() and mask_dir.is_dir()):
            continue
        for vh_file in vh_dir.glob('*_vh.png'):
            base = vh_file.stem.replace('_vh', '')
            vv_file = vv_dir / f"{base}_vv.png"
            mask_file = mask_dir / f"{base}.png"
            if vv_file.exists() and mask_file.exists():
                triplets.append((vh_file, vv_file, mask_file))
    return triplets

print("Gathering tile paths for water-body labels...")
all_tiles = gather_tile_paths(dataset_root)
print(f"Total tiles found: {len(all_tiles)}")

# Shuffle and split into train/validation
random.shuffle(all_tiles)
split = int(len(all_tiles) * (1 - VAL_SPLIT))
train_tiles = all_tiles[:split]
val_tiles = all_tiles[split:]
print(f"Training tiles: {len(train_tiles)}, Validation tiles: {len(val_tiles)}")

# === Data loaders ===

def load_batch(triplet_list):
    X = np.zeros((len(triplet_list), IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)
    y = np.zeros((len(triplet_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    for i, (vh_path, vv_path, mask_path) in enumerate(triplet_list):
        vh = cv2.imread(str(vh_path), cv2.IMREAD_GRAYSCALE)
        vv = cv2.imread(str(vv_path), cv2.IMREAD_GRAYSCALE)
        vh = cv2.resize(vh, (IMG_SIZE, IMG_SIZE)) / 255.0
        vv = cv2.resize(vv, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0
        X[i, ..., 0] = vh
        X[i, ..., 1] = vv
        y[i, ..., 0] = mask
    return X, y

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, triplets, batch_size):
        self.triplets = triplets
        self.batch_size = batch_size
    def __len__(self):
        return int(np.ceil(len(self.triplets) / self.batch_size))
    def __getitem__(self, idx):
        batch = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        return load_batch(batch)

# === U-Net model definition ===

def unet(input_shape=(IMG_SIZE, IMG_SIZE, 2)):
    inputs = Input(input_shape)
    x = inputs
    skips = []
    f = 16
    # Downsampling
    for _ in range(4):
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = MaxPooling2D()(x)
        f *= 2
    # Bottleneck
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    f //= 2
    # Upsampling
    for skip in reversed(skips):
        x = Conv2DTranspose(f, 2, strides=(2,2), padding='same')(x)
        x = Concatenate()([x, skip])
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        f //= 2
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# === Training setup ===
print("Building U-Net model for water-body segmentation...")
model = unet()
model.compile(
    optimizer=Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', BinaryIoU(threshold=0.5, name='iou')]
)
model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint(MODEL_NAME, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

# Generators
dg_train = DataGenerator(train_tiles, BATCH_SIZE)
dg_val = DataGenerator(val_tiles, BATCH_SIZE)

# === Train ===
history = model.fit(
    dg_train,
    validation_data=dg_val,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# === Plot training metrics ===
def plot_history(hist):
    loss = hist.history['loss']; val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']; val_acc = hist.history['val_accuracy']
    iou = hist.history['iou']; val_iou = hist.history['val_iou']
    epochs_range = range(1, len(loss)+1)
    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1); plt.plot(epochs_range, loss, label='train loss'); plt.plot(epochs_range, val_loss, label='val loss'); plt.legend(); plt.title('Loss')
    plt.subplot(1,3,2); plt.plot(epochs_range, acc, label='train acc'); plt.plot(epochs_range, val_acc, label='val acc'); plt.legend(); plt.title('Accuracy')
    plt.subplot(1,3,3); plt.plot(epochs_range, iou, label='train IoU'); plt.plot(epochs_range, val_iou, label='val IoU'); plt.legend(); plt.title('Mean IoU')
    plt.tight_layout()
    plt.savefig('water_body_label_training_metrics.png')
    plt.show()

plot_history(history)

# === Sample predictions ===
def show_preds(model, triplets, n=6):
    sample = random.sample(triplets, n)
    Xs, ys = load_batch(sample)
    preds = model.predict(Xs)
    plt.figure(figsize=(12,6))
    for i in range(n):
        plt.subplot(3, n, i+1); plt.imshow(Xs[i,...,0], cmap='gray'); plt.axis('off')
        plt.subplot(3, n, i+1+n); plt.imshow(Xs[i,...,1], cmap='gray'); plt.axis('off')
        plt.subplot(3, n, i+1+2*n); plt.imshow((preds[i,...,0]>0.5).astype(np.uint8), cmap='Blues'); plt.axis('off')
    plt.tight_layout()
    plt.savefig('water_body_label_sample_predictions.png')
    plt.show()

show_preds(model, val_tiles)

print("✅ Training complete. Best model saved as", MODEL_NAME)
