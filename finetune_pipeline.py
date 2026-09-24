"""
Combine the new `dataset/` (2500 images + contact-point CSV) with the old
labeled data that's still actually present on disk, augment all of it
offline (vertical flip, horizontal flip, blur, +/-10deg rotation), and
fine-tune the current model (model_rig_tilt_v4.pt) on the result instead
of training from scratch.

Old data note: labels.csv (4050 rows) and labels_manual.csv (300 rows)
reference images that no longer exist anywhere in this repo (they lived in
a gitignored images/ folder that's gone), so they're excluded. Only
labels_new.csv and labels_rig.csv have their images present, scattered
across testimages/, testingimages/, to_label_round2/, and tomove/.

Usage:
    python finetune_pipeline.py full \\
        --dataset_dir dataset --dataset_csv dataset/labels.csv \\
        --old_labels labels_new.csv labels_rig.csv \\
        --old_search_dirs testimages testingimages to_label_round2 tomove \\
        --init_checkpoint model_rig_tilt_v4.pt \\
        --checkpoint_out model_rig_tilt_v5.pt \\
        --n_augments 10
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
from PIL import Image

from testTrain import IMG_H, IMG_W, train as train_model

ROW_FIELDS = ["filename", "row_left_px", "row_right_px",
              "row_left_norm", "row_right_norm", "img_height"]


# ---------------------------------------------------------------------------
# 1. Convert dataset/labels.csv (contact points) -> row_left/right format
# ---------------------------------------------------------------------------

def convert_dataset_labels(dataset_dir, dataset_csv, out_csv):
    """
    dataset/labels.csv gives contactLeftX/Y and contactRightX/Y: real (x, y)
    points that aren't at the image edges. Every other label file in this
    repo (and HorizonNet's output) assumes the line is read at x=0 and
    x=width-1, so fit a line through the two contact points and extrapolate
    it out to both edges.
    """
    rows = []
    skipped = 0
    with open(dataset_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fname = r["filename"]
            path = os.path.join(dataset_dir, fname)
            if not os.path.exists(path):
                skipped += 1
                continue
            with Image.open(path) as im:
                w, h = im.size
            x0, y0 = float(r["contactLeftX"]), float(r["contactLeftY"])
            x1, y1 = float(r["contactRightX"]), float(r["contactRightY"])
            if abs(x1 - x0) < 1e-6:
                left_px = right_px = (y0 + y1) / 2.0
            else:
                slope = (y1 - y0) / (x1 - x0)
                left_px = y0 + slope * (0 - x0)
                right_px = y0 + slope * ((w - 1) - x0)
            rows.append((fname, left_px, right_px, left_px / h, right_px / h, h))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ROW_FIELDS)
        writer.writerows(rows)
    print(f"Converted {len(rows)} rows from {dataset_csv} -> {out_csv} ({skipped} skipped, image not found)")


# ---------------------------------------------------------------------------
# 2. Resolve old label CSVs to wherever their images actually live, and
#    symlink everything (new + old) into one merged image directory
# ---------------------------------------------------------------------------

def merge_into(dataset_dir, dataset_row_csv, old_label_csvs, old_search_dirs,
                merged_dir, merged_csv):
    os.makedirs(merged_dir, exist_ok=True)
    written = set()
    out_rows = []

    def add_source(row_csv, search_dirs):
        found, missing = 0, 0
        with open(row_csv, newline="") as f:
            for r in csv.DictReader(f):
                fname = r["filename"]
                if fname in written:
                    continue
                src = None
                for d in search_dirs:
                    cand = os.path.join(d, fname)
                    if os.path.exists(cand):
                        src = cand
                        break
                if src is None:
                    missing += 1
                    continue
                dst = os.path.join(merged_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)
                written.add(fname)
                out_rows.append({k: r[k] for k in ROW_FIELDS})
                found += 1
        print(f"  {row_csv}: {found} images linked, {missing} missing (skipped)")

    add_source(dataset_row_csv, [dataset_dir])
    for csv_path in old_label_csvs:
        add_source(csv_path, old_search_dirs)

    with open(merged_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Merged {len(out_rows)} labeled images -> {merged_dir} / {merged_csv}")


# ---------------------------------------------------------------------------
# 3. Offline augmentation: persist ~n copies per image using exactly the
#    transform types requested (vertical flip, horizontal flip, blur,
#    slight rotation), not the full on-the-fly augmentation set testTrain.py
#    already applies during training.
# ---------------------------------------------------------------------------

def _load_gray_resized(path):
    img = Image.open(path).convert("L").resize((IMG_W, IMG_H))
    return np.array(img, dtype=np.float32) / 255.0


def _rotate(arr, left, right, max_angle_deg=10):
    h, w = arr.shape
    angle = random.uniform(-max_angle_deg, max_angle_deg)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT101)

    left_px, right_px = left * h, right * h
    p0 = M @ np.array([0.0, left_px, 1.0])
    p1 = M @ np.array([w - 1.0, right_px, 1.0])
    dx = p1[0] - p0[0]
    if abs(dx) < 1e-6:
        new_left_px, new_right_px = p0[1], p1[1]
    else:
        slope = (p1[1] - p0[1]) / dx
        new_left_px = p0[1] + slope * (0 - p0[0])
        new_right_px = p0[1] + slope * ((w - 1) - p0[0])
    return rotated, float(np.clip(new_left_px / h, 0.0, 1.0)), float(np.clip(new_right_px / h, 0.0, 1.0))


def _vflip(arr, left, right):
    return arr[::-1, :].copy(), 1.0 - left, 1.0 - right


def _hflip(arr, left, right):
    return arr[:, ::-1].copy(), right, left


def _blur(arr):
    sigma = random.uniform(0.8, 2.5)
    return cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)


def _augment_once(arr, left, right, max_angle_deg):
    # pick 1-3 of the requested transform types, apply geometry ops first,
    # blur last so it reflects the final framing
    choices = random.sample(["rotate", "vflip", "hflip", "blur"], k=random.randint(1, 3))
    if "rotate" in choices:
        arr, left, right = _rotate(arr, left, right, max_angle_deg)
    if "vflip" in choices:
        arr, left, right = _vflip(arr, left, right)
    if "hflip" in choices:
        arr, left, right = _hflip(arr, left, right)
    if "blur" in choices:
        arr = _blur(arr)
    return arr, left, right


def augment_offline(img_dir, labels_csv, out_dir, out_csv, n_augments=10, max_angle_deg=10, seed=0):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    out_rows = []

    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        fname = r["filename"]
        src = os.path.join(img_dir, fname)
        if not os.path.exists(src):
            continue
        left = float(r["row_left_norm"])
        right = float(r["row_right_norm"])
        base_arr = _load_gray_resized(src)
        h = IMG_H

        # keep the original too
        orig_out = os.path.join(out_dir, fname)
        if not os.path.exists(orig_out):
            os.symlink(os.path.abspath(src), orig_out)
        out_rows.append({"filename": fname, "row_left_px": left * h, "row_right_px": right * h,
                          "row_left_norm": left, "row_right_norm": right, "img_height": h})

        stem, _ = os.path.splitext(fname)
        for i in range(n_augments):
            arr, l, r_ = _augment_once(base_arr, left, right, max_angle_deg)
            aug_name = f"aug{i:02d}__{stem}.png"
            Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(os.path.join(out_dir, aug_name))
            out_rows.append({"filename": aug_name, "row_left_px": l * h, "row_right_px": r_ * h,
                              "row_left_norm": l, "row_right_norm": r_, "img_height": h})

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} images ({len(rows)} originals x ~{n_augments} augmentations) -> {out_dir} / {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def full_pipeline(args):
    work = args.work_dir
    os.makedirs(work, exist_ok=True)

    dataset_row_csv = os.path.join(work, "dataset_labels_converted.csv")
    convert_dataset_labels(args.dataset_dir, args.dataset_csv, dataset_row_csv)

    merged_dir = os.path.join(work, "merged_raw")
    merged_csv = os.path.join(work, "merged_labels.csv")
    print("Merging new + old labeled data...")
    merge_into(args.dataset_dir, dataset_row_csv, args.old_labels, args.old_search_dirs,
               merged_dir, merged_csv)

    final_dir = os.path.join(work, "final_images")
    final_csv = os.path.join(work, "final_labels.csv")
    print(f"Generating ~{args.n_augments} offline augmentations per image...")
    augment_offline(merged_dir, merged_csv, final_dir, final_csv,
                     n_augments=args.n_augments, max_angle_deg=args.max_angle_deg)

    print(f"Fine-tuning from {args.init_checkpoint} -> {args.checkpoint_out}")
    train_model(final_dir, final_csv, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, checkpoint_out=args.checkpoint_out, init_checkpoint=args.init_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_full = sub.add_parser("full", help="run the whole pipeline: convert, merge, augment, fine-tune")
    p_full.add_argument("--dataset_dir", default="dataset")
    p_full.add_argument("--dataset_csv", default="dataset/labels.csv")
    p_full.add_argument("--old_labels", nargs="*", default=["labels_new.csv", "labels_rig.csv"])
    p_full.add_argument("--old_search_dirs", nargs="*",
                         default=["testimages", "testingimages", "to_label_round2", "tomove"])
    p_full.add_argument("--work_dir", default="finetune_work")
    p_full.add_argument("--n_augments", type=int, default=10)
    p_full.add_argument("--max_angle_deg", type=float, default=10)
    p_full.add_argument("--init_checkpoint", default="model_rig_tilt_v4.pt")
    p_full.add_argument("--checkpoint_out", default="model_rig_tilt_v5.pt")
    p_full.add_argument("--epochs", type=int, default=50)
    p_full.add_argument("--batch_size", type=int, default=16)
    p_full.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    if args.cmd == "full":
        full_pipeline(args)
