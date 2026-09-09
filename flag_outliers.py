"""
Draw the predicted horizon row on each image so you can quickly scan
for obviously wrong labels before training.

Usage:
    python review_labels.py --img_dir ./images --labels labels.csv --out ./review
"""

import argparse
import csv
import os

from PIL import Image, ImageDraw


def draw_line(img_path, row_px, out_path, color=(255, 0, 0), thickness=2):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    row_px = int(row_px)
    draw.line([(0, row_px), (img.width, row_px)], fill=color, width=thickness)
    img.save(out_path)


def main(img_dir, labels_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    skipped = 0
    for r in rows:
        row_px = float(r["row_px"])
        if row_px < 0:
            skipped += 1
            continue
        src = os.path.join(img_dir, r["filename"])
        dst = os.path.join(out_dir, r["filename"])
        draw_line(src, row_px, dst)

    print(f"Wrote {len(rows) - skipped} annotated images to {out_dir}")
    if skipped:
        print(f"Skipped {skipped} images with no valid label (row_px < 0)")
    print("Open the folder in your OS image viewer or a contact sheet tool and scroll through.")
    print("For any that look wrong, go back to labels.csv, find that filename,")
    print("and either fix row_px/row_norm by hand or delete the row to exclude it from training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="./review")
    args = parser.parse_args()
    main(args.img_dir, args.labels, args.out)