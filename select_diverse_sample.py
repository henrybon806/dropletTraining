"""
Pick a diverse subset of images to hand-label, using k-means clustering on
simple image statistics (downsized brightness profile). This spreads your
labeling budget across the actual variation in the dataset (dark vs bright
frames, different flask sizes/positions, etc) instead of hoping random
sampling happens to cover it.

Usage:
    python select_diverse_sample.py --img_dir ./merged_images --out_dir ./to_label --n 300
"""

import argparse
import os
import shutil

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def extract_features(img_dir, files, feat_size=(32, 16)):
    """
    Downsize each image to a tiny grayscale thumbnail and flatten it.
    This is a cheap stand-in for "what does this image roughly look like"
    (overall brightness, gradient shape, dark blob position/size) without
    needing anything fancier than PIL + numpy.
    """
    feats = []
    for fname in files:
        img = Image.open(os.path.join(img_dir, fname)).convert("L")
        img = img.resize(feat_size)
        arr = np.array(img, dtype=np.float32).flatten() / 255.0
        feats.append(arr)
    return np.stack(feats)


def select_diverse(img_dir, out_dir, n, seed=0):
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if n >= len(files):
        print(f"Requested {n} but only {len(files)} images exist, selecting all of them")
        selected = files
    else:
        print(f"Extracting features for {len(files)} images...")
        feats = extract_features(img_dir, files)

        print(f"Clustering into {n} groups...")
        km = KMeans(n_clusters=n, random_state=seed, n_init=10)
        labels = km.fit_predict(feats)

        # pick the image closest to each cluster center, so each selected
        # image is a genuine representative of that region of variation
        selected = []
        for c in range(n):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                continue
            center = km.cluster_centers_[c]
            dists = np.linalg.norm(feats[idxs] - center, axis=1)
            best = idxs[np.argmin(dists)]
            selected.append(files[best])

    os.makedirs(out_dir, exist_ok=True)
    for fname in selected:
        shutil.copy(os.path.join(img_dir, fname), os.path.join(out_dir, fname))

    print(f"Copied {len(selected)} diverse images to {out_dir}")
    print(f"Now run: python manual_label.py --img_dir {out_dir} --out labels_manual.csv --n {len(selected)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_dir", default="./to_label")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    select_diverse(args.img_dir, args.out_dir, args.n, args.seed)