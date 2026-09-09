"""
Manually label a (possibly tilted) line by clicking twice per image.

Click the LEFT end of the line first, then the RIGHT end. Close the window
without clicking (or with only one click) to skip that image. Progress
saves after every image, so you can stop and resume later, just re-run
the same command.

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
    print("Click the LEFT end of the line, then the RIGHT end.")
    print("Close the window without two clicks to skip that image.")

    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filename", "row_left_px", "row_right_px",
                              "row_left_norm", "row_right_norm", "img_height"])

        for fname in remaining:
            path = os.path.join(img_dir, fname)
            img = Image.open(path)
            h = img.height

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{fname}  (click LEFT end, then RIGHT end)")
            ax.axis("off")

            pts = plt.ginput(2, timeout=0)
            plt.close(fig)

            if len(pts) < 2:
                print(f"skipped {fname}")
                continue

            (x_left, y_left), (x_right, y_right) = pts
            # sort by x, in case the two clicks were made right-to-left
            if x_left > x_right:
                (x_left, y_left), (x_right, y_right) = (x_right, y_right), (x_left, y_left)

            left_norm = y_left / h
            right_norm = y_right / h
            writer.writerow([fname, y_left, y_right, left_norm, right_norm, h])
            f.flush()
            print(f"{fname}: left={y_left:.1f} ({left_norm:.3f})  right={y_right:.1f} ({right_norm:.3f})")

    print("Done for this batch. Re-run the same command to label more if you raise --n.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out", default="labels_manual.csv")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.img_dir, args.out, args.n, args.seed)