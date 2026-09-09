"""
Horizon detection pipeline for flask images.

Workflow:
1. Generate pseudo-labels for all images using intensity thresholding
2. Manually review/fix labels in labels.csv (spot check ~10-20%)
3. Train HorizonNet to regress horizon row from image
4. Run inference on new images

Usage:
    python horizon_detector.py label --img_dir ./images --out labels.csv
    python horizon_detector.py train --img_dir ./images --labels labels.csv --epochs 50
    python horizon_detector.py infer --img_dir ./images --checkpoint model.pt
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


IMG_H, IMG_W = 256, 512  # resize target, adjust to your aspect ratio


# ---------------------------------------------------------------------------
# 1. Pseudo-label generation
# ---------------------------------------------------------------------------

def candidate_horizon_row(img_arr, thresh_frac=0.92, top_rows=10, safety_margin=0.9):
    """
    img_arr: 2D numpy array, grayscale, values 0-255 or 0-1.

    Two-pass detection:
      Pass 1: naive row-mean threshold, ignoring occlusion. Any dark
      object in frame drags the row-mean down early, so this lands at
      or above the true horizon (top of the object, not the horizon).
      Treat this as a safe upper bound, never below the real horizon.

      Pass 2: use the rows strictly above that rough estimate (still
      guaranteed to be background, not yet at the horizon) to figure
      out which columns are occluded by the object. Recompute the row
      profile using only unoccluded columns, over the full image.
    """
    h, w = img_arr.shape

    row_profile_all = img_arr.mean(axis=1)
    top_val = np.median(row_profile_all[:top_rows])
    thresh = top_val * thresh_frac
    below = np.where(row_profile_all < thresh)[0]
    rough_row = int(below[0]) if len(below) else h - 1

    band_end = max(top_rows, int(rough_row * safety_margin))
    band = img_arr[:band_end, :]
    col_brightness = band.mean(axis=0)
    clear_thresh = np.percentile(col_brightness, 75) * 0.85
    clear_cols = col_brightness > clear_thresh

    if clear_cols.sum() < w * 0.15:
        # nothing looked occluded in the band, fall back to using everything
        clear_cols = np.ones(w, dtype=bool)

    row_profile = img_arr[:, clear_cols].mean(axis=1)
    top_val2 = np.median(row_profile[:top_rows])
    thresh2 = top_val2 * thresh_frac
    below2 = np.where(row_profile < thresh2)[0]
    return int(below2[0]) if len(below2) else rough_row


def generate_labels(img_dir, out_csv, thresh_frac=0.92):
    """
    Walk img_dir, compute pseudo-label for each image, write to CSV.
    You should open out_csv afterward and hand-correct any row where
    the model's guess looks wrong (heavy blur, low contrast, weird lighting).
    """
    rows = []
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for fname in files:
        path = os.path.join(img_dir, fname)
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
        h = arr.shape[0]
        row = candidate_horizon_row(arr, thresh_frac=thresh_frac)
        norm_row = row / h if row >= 0 else -1
        rows.append((fname, row, norm_row, h))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "row_px", "row_norm", "img_height"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} pseudo-labels to {out_csv}")
    print("Review this file and fix any obviously wrong rows before training.")


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

class HorizonDataset(Dataset):
    def __init__(self, img_dir, labels_csv, augment=False):
        self.img_dir = img_dir
        self.augment = augment
        self.samples = []
        missing = 0
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if float(r["row_norm"]) < 0:
                    continue  # skip failed labels
                if not os.path.exists(os.path.join(img_dir, r["filename"])):
                    missing += 1
                    continue
                self.samples.append((r["filename"], float(r["row_norm"])))
        if missing:
            print(f"Warning: {missing} labeled rows point to files not found in {img_dir}, skipped them")

    def __len__(self):
        return len(self.samples)

    def _load(self, fname):
        img = Image.open(os.path.join(self.img_dir, fname)).convert("L")
        img = img.resize((IMG_W, IMG_H))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def _augment(self, arr):
        # brightness/contrast jitter
        if random.random() < 0.7:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            arr = np.clip(arr * gain + bias, 0, 1)

        # gaussian noise
        if random.random() < 0.5:
            arr = np.clip(arr + np.random.normal(0, 0.02, arr.shape), 0, 1)

        # random occluder blob, simulates the flask at random position/size
        if random.random() < 0.5:
            h, w = arr.shape
            cx = random.randint(w // 4, 3 * w // 4)
            cy = random.randint(h // 4, 3 * h // 4)
            r = random.randint(h // 8, h // 3)
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            arr = arr.copy()
            arr[mask] = random.uniform(0, 0.3)

        # horizontal flip (doesn't change horizon row)
        if random.random() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1])

        return arr

    def __getitem__(self, idx):
        fname, row_norm = self.samples[idx]
        arr = self._load(fname)
        if self.augment:
            arr = self._augment(arr)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()  # (1, H, W)
        target = torch.tensor([row_norm], dtype=torch.float32)
        return tensor, target


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------

class HorizonNet(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()

        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64),
            block(64, 128), block(128, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.Sigmoid(),  # output in [0, 1], multiply by img height for pixels
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# 4. Train / eval loop
# ---------------------------------------------------------------------------

def train(img_dir, labels_csv, epochs=50, batch_size=16, lr=1e-3,
          val_frac=0.15, checkpoint_out="model.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = HorizonDataset(img_dir, labels_csv, augment=True)
    n_val = max(1, int(len(full_ds) * val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False  # disable augmentation for validation split

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = HorizonNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_out)

        print(f"epoch {epoch+1:3d}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    print(f"Best val loss: {best_val:.5f}. Saved to {checkpoint_out}")


# ---------------------------------------------------------------------------
# 5. Inference
# ---------------------------------------------------------------------------

def infer(img_dir, checkpoint, out_csv="predictions.csv", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HorizonNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rows = []
    with torch.no_grad():
        for fname in files:
            img = Image.open(os.path.join(img_dir, fname)).convert("L")
            orig_h = img.height
            img_resized = img.resize((IMG_W, IMG_H))
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_norm = model(tensor).item()
            pred_px = pred_norm * orig_h
            rows.append((fname, pred_px, pred_norm))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "row_px", "row_norm"])
        writer.writerows(rows)

    print(f"Wrote predictions for {len(rows)} images to {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_label = sub.add_parser("label")
    p_label.add_argument("--img_dir", required=True)
    p_label.add_argument("--out", default="labels.csv")
    p_label.add_argument("--thresh_frac", type=float, default=0.92)

    p_train = sub.add_parser("train")
    p_train.add_argument("--img_dir", required=True)
    p_train.add_argument("--labels", required=True)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--checkpoint_out", default="model.pt")

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--img_dir", required=True)
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--out", default="predictions.csv")

    args = parser.parse_args()

    if args.cmd == "label":
        generate_labels(args.img_dir, args.out, args.thresh_frac)
    elif args.cmd == "train":
        train(args.img_dir, args.labels, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr,
              checkpoint_out=args.checkpoint_out)
    elif args.cmd == "infer":
        infer(args.img_dir, args.checkpoint, args.out)