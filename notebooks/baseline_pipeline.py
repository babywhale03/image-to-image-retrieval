import numpy as np
from PIL import Image

# Color histogram feature extraction
def compute_color_histogram(img, bins=(8, 8, 8), size=(224, 224)):
    # Resize image
    img = img.resize(size)
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H, W, 3)

    # Compute 3D color histogram
    pixels = arr.reshape(-1, 3)  # (N,3)
    hist, _ = np.histogramdd(
        pixels,
        bins=bins,
        range=[(0, 1), (0, 1), (0, 1)]
    )
    hist = hist.flatten().astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist

# Extract baseline features for a list of items
def extract_baseline_features(items, viz_step=500):
    print("Extracting baseline color histogram features")
    feats = []
    # Iterate over items and compute features
    for i, it in enumerate(items):
        if viz_step and i % viz_step == 0:
            print(f"Processing {i} / {len(items)}")
        img = Image.open(it["path"]).convert("RGB")
        feat = compute_color_histogram(img)
        feats.append(feat)
    return np.stack(feats, axis=0)