"""Decode UTKFace once into a 64x64 grayscale uint8 cache.
Grayscale conversion matches the paper's code (0.299R + 0.587G + 0.114B).
Usage: python3 prep.py <path/to/UTKFace dir> data/utk64.npz
"""
import glob, os, sys
import numpy as np
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(src, "*.jpg")))
imgs, ages, names = [], [], []
for f in files:
    try:
        age = int(os.path.basename(f).split("_")[0])
    except ValueError:
        continue
    rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    g = Image.fromarray((gray * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR)
    imgs.append(np.asarray(g, dtype=np.uint8)); ages.append(age); names.append(os.path.basename(f))
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
np.savez_compressed(out, x=np.stack(imgs), age=np.array(ages, dtype=np.int16), name=np.array(names))
print(len(imgs), "images; age mean %.1f median %.0f" % (np.mean(ages), np.median(ages)))
