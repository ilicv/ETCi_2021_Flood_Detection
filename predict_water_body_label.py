#!/usr/bin/env python3
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model

# === Configuration ===
IMG_SIZE = 256                      # Resize input tiles
INPUT_CHANNELS = 2                  # VH + VV
MODEL_WEIGHTS = 'best_water_body_unet.h5'  # Water body-model weights file
DST_FOLDER = 'predicted_results'    # Where to save composite images
THRESHOLD = 0.5                     # Probability threshold for mask


def unet(input_shape=(IMG_SIZE, IMG_SIZE, INPUT_CHANNELS)):
    inputs = Input(input_shape)
    x = inputs
    skips = []
    filters = 16
    # Encoder
    for _ in range(4):
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = MaxPooling2D()(x)
        filters *= 2
    # Bottleneck
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    # Decoder
    for skip in reversed(skips):
        filters //= 2
        x = Conv2DTranspose(filters, 2, strides=(2,2), padding='same')(x)
        x = Concatenate()([x, skip])
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, outputs)


def replicate_dirs(src_root, dst_root):
    # Mirror directory tree from src_root under dst_root, skipping DST_FOLDER
    for root, dirs, _ in os.walk(src_root):
        dirs[:] = [d for d in dirs if d != DST_FOLDER]
        rel = os.path.relpath(root, src_root)
        os.makedirs(os.path.join(dst_root, rel), exist_ok=True)


def predict_and_save_composite(model, vh_path, vv_path, out_path):
    # Load and preprocess
    vh = cv2.imread(vh_path, cv2.IMREAD_GRAYSCALE)
    vv = cv2.imread(vv_path, cv2.IMREAD_GRAYSCALE)
    vh_resized = cv2.resize(vh, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    vv_resized = cv2.resize(vv, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    inp = np.stack([vh_resized, vv_resized], axis=-1)[None, ...]
    # Predict mask
    pred = model.predict(inp)[0, ..., 0]
    mask = (pred > THRESHOLD).astype(np.uint8) * 255
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Prepare input images for display
    vh_color = cv2.cvtColor((vh_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vv_color = cv2.cvtColor((vv_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Overlay mask in blue on VH image
    overlay = vh_color.copy()
    overlay[mask == 255] = [255, 0, 0]
    # Concatenate images horizontally: VH, VV, Mask, Overlay
    composite = np.hstack([vh_color, vv_color, mask_color, overlay])
    # Save composite
    cv2.imwrite(out_path, composite)


def main():
    src_root = os.getcwd()
    dst_root = os.path.join(src_root, DST_FOLDER)
    print(f"Loading water body UNet model from '{MODEL_WEIGHTS}'...")
    model = unet()
    model.load_weights(MODEL_WEIGHTS)
    print("Model loaded. Preparing output folders...")
    replicate_dirs(src_root, dst_root)

    # Process each VH tile
    for root, dirs, files in os.walk(src_root):
        dirs[:] = [d for d in dirs if d != DST_FOLDER]
        if os.path.basename(root) == 'vh':
            rel_dir = os.path.relpath(root, src_root)
            base_dir = os.path.dirname(rel_dir)
            vv_dir = os.path.join(src_root, base_dir, 'vv')
            out_dir = os.path.join(dst_root, base_dir, 'water_body_label')
            os.makedirs(out_dir, exist_ok=True)
            for vh_fn in files:
                if not vh_fn.lower().endswith('_vh.png'):
                    continue
                base = vh_fn[:-7]  # strip '_vh.png'
                vv_fn = base + '_vv.png'
                vh_path = os.path.join(root, vh_fn)
                vv_path = os.path.join(vv_dir, vv_fn)
                out_path = os.path.join(out_dir, base + '.png')
                if os.path.exists(vv_path):
                    predict_and_save_composite(model, vh_path, vv_path, out_path)
                else:
                    print(f"⚠️ Missing VV tile for {vh_fn}, skipping.")

    print(f"✅ Composite prediction complete. Results in '{DST_FOLDER}'.")

if __name__ == '__main__':
    main()
