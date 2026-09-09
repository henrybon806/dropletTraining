"""
Manually label the horizon row by clicking on images.

Click once on the horizon in each image window. Close the window without
clicking to skip that image. Progress saves after every image, so you can
stop and resume later, just re-run the same command.

Usage:
    python manual_label.py --img_dir ./images --out labels_manual.csv --n 100
"""

import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
from PIL import Image


def load_existing(out_csv):
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                done.add(r["filename"])
    return done


def main(img_dir, out_csv, n, seed):
    random.seed(seed)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.shuffle(files)

    done = load_existing(out_csv)
    write_header = not os.path.exists(out_csv)

    remaining = [f for f in files if f not in done][:max(0, n - len(done))]
    if not remaining:
        print(f"Already have {len(done)} labels, nothing left to do for n={n}")
        return

    print(f"{len(done)} already labeled, labeling {len(remaining)} more")
    print("Click on the horizon in each image. Close the window without clicking to skip.")

    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filename", "row_px", "row_norm", "img_height"])

        for fname in remaining:
            path = os.path.join(img_dir, fname)
            img = Image.open(path)
            h = img.height

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img, cmap="gray")
            ax.set_title(fname)
            ax.axis("off")

            pts = plt.ginput(1, timeout=0)
            plt.close(fig)

            if not pts:
                print(f"skipped {fname}")
                continue

            row_px = pts[0][1]
            row_norm = row_px / h
            writer.writerow([fname, row_px, row_norm, h])
            f.flush()
            print(f"{fname}: row={row_px:.1f} ({row_norm:.3f})")

    print("Done for this batch. Re-run the same command to label more if you raise --n.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out", default="labels_manual.csv")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.img_dir, args.out, args.n, args.seed)