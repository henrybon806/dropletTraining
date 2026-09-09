"""
Build grid contact sheets from a flagged-images CSV, with the predicted
horizon line drawn on each thumbnail, for fast visual scanning.

Usage:
    python make_contact_sheets.py --img_dir ./images --flagged flagged.csv --out ./sheets
"""

import argparse
import csv
import math
import os

from PIL import Image, ImageDraw


def draw_line(img, row_px, color=(255, 0, 0), thickness=2):
    draw = ImageDraw.Draw(img)
    draw.line([(0, int(row_px)), (img.width, int(row_px))], fill=color, width=thickness)
    return img


def make_sheets(img_dir, flagged_csv, out_dir, thumb_w=220, grid_rows=5, grid_cols=5):
    os.makedirs(out_dir, exist_ok=True)
    with open(flagged_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    per_sheet = grid_rows * grid_cols
    n_sheets = math.ceil(len(rows) / per_sheet) if rows else 0
    label_h = 20

    for s in range(n_sheets):
        chunk = rows[s * per_sheet:(s + 1) * per_sheet]

        thumbs = []
        max_thumb_h = 0
        for r in chunk:
            img = Image.open(os.path.join(img_dir, r["filename"])).convert("RGB")
            scale = thumb_w / img.width
            row_px_scaled = float(r["row_px"]) * scale
            img = img.resize((thumb_w, int(img.height * scale)))
            img = draw_line(img, row_px_scaled)
            thumbs.append((img, r["filename"]))
            max_thumb_h = max(max_thumb_h, img.height)

        sheet_w = thumb_w * grid_cols
        sheet_h = (max_thumb_h + label_h) * grid_rows
        sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)

        for i, (img, fname) in enumerate(thumbs):
            row_i, col_i = divmod(i, grid_cols)
            x = col_i * thumb_w
            y = row_i * (max_thumb_h + label_h)
            sheet.paste(img, (x, y))
            draw.text((x + 2, y + max_thumb_h + 2), fname[:30], fill=(255, 255, 255))

        sheet.save(os.path.join(out_dir, f"sheet_{s:03d}.png"))

    print(f"Wrote {n_sheets} contact sheets to {out_dir} ({per_sheet} images each)")
    print("Open the folder and flip through the sheets. For any bad line,")
    print("note the filename printed under it, then fix or drop that row in labels.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--flagged", required=True)
    parser.add_argument("--out", default="./sheets")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    make_sheets(args.img_dir, args.flagged, args.out, grid_rows=args.rows, grid_cols=args.cols)