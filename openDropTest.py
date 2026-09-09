"""
Run OpenDrop's Contact Angle analysis, using the trained horizon model to
set the baseline (surface line) automatically instead of dragging it by
hand each time.

Verified against opendrop's actual source (features/conan.py, fit/conan.py):
extract_contact_angle_features() takes a Line2 baseline directly, and
contact_angle_fit() takes the extracted drop_points plus the same baseline.

IMPORTANT: this needs the real jdber1/opendrop package, NOT `pip install
opendrop` (that installs an unrelated AirDrop implementation from PyPI with
the same name). Install with:

    pip install git+https://github.com/jdber1/opendrop.git

That build also needs SUNDIALS/ARKODE and Boost.Math as native
dependencies (used by the pendant-drop Young-Laplace solver), see
opendrop's install docs. Contact Angle mode doesn't use young_laplace_fit
at all, so if you only care about contact angle, you may be able to strip
those build deps down, worth checking the install script if the ARKODE
build step fails on your machine.

The drop region still needs definition since the model only predicts the
baseline, not a bounding box around the drop. Defaults to the full image
width, extended from a bit above the baseline (BASELINE_MARGIN_PX) up to
the top of the image. Adjust with --drop_x0/--drop_x1 or pass
--manual_drop_region to select it by hand as OpenDrop normally would.

Usage:
    python opendrop_contact_angle.py \\
        --image ./images/frame_1.jpeg \\
        --checkpoint model_v3.pt
"""

import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from testTrain import HorizonNet, IMG_H, IMG_W
from opendrop.geometry import Line2, Rect2
from opendrop.features.conan import extract_contact_angle_features
from opendrop.fit.conan import contact_angle_fit


def predict_horizon_row(model, image_path, device):
    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    img_resized = img.resize((IMG_W, IMG_H))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        row_norm = model(tensor).item()
    return row_norm * orig_h, orig_w, orig_h


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HorizonNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    horizon_row, orig_w, orig_h = predict_horizon_row(model, args.image, device)
    horizon_row = int(round(horizon_row))
    print(f"Predicted baseline at row {horizon_row}px (image {orig_w}x{orig_h})")

    # Baseline as a Line2: two points on the horizontal line the model found.
    # If your model already predicts y_left/y_right separately for tilt,
    # swap these two y-values accordingly instead of using the same row twice.
    baseline = Line2(pt0=(0, horizon_row), pt1=(orig_w, horizon_row))

    image = cv2.imread(args.image)

    if args.manual_drop_region:
        x, y, w, h = cv2.selectROI("Select drop region", image, fromCenter=False)
        roi = Rect2(x=x, y=y, w=w, h=h)
        cv2.destroyAllWindows()
    else:
        drop_x0 = args.drop_x0 if args.drop_x0 is not None else 0
        drop_x1 = args.drop_x1 if args.drop_x1 is not None else orig_w
        margin = args.baseline_margin_px
        roi = Rect2(x0=drop_x0, y0=0, x1=drop_x1, y1=min(orig_h, horizon_row + margin))
    print("Drop ROI:", roi)

    features = extract_contact_angle_features(
        image,
        baseline=baseline,
        inverted=args.inverted,
        roi=roi,
        labels=args.show_features,
    )

    if args.show_features and features.labels is not None:
        labelled_image = image.copy()
        labelled_image[features.labels == 1] = [255, 0, 0]  # generic edges
        labelled_image[features.labels == 2] = [0, 255, 0]  # drop profile
        cv2.imshow("Extracted features", labelled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    fit = contact_angle_fit(features.drop_points, baseline=baseline)

    print(
        "\nResults\n=======\n",
        "Left contact angle [deg]: ", np.degrees(fit.left_angle) if fit.left_angle is not None else None, "\n",
        "Right contact angle [deg]: ", np.degrees(fit.right_angle) if fit.right_angle is not None else None, "\n",
        "Left contact point: ", fit.left_contact, "\n",
        "Right contact point: ", fit.right_contact,
    )

    return fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--drop_x0", type=int, default=None)
    parser.add_argument("--drop_x1", type=int, default=None)
    parser.add_argument("--baseline_margin_px", type=int, default=0,
                         help="extend the drop ROI this many px below the baseline, in case the drop bulges past it")
    parser.add_argument("--inverted", action="store_true",
                         help="pass this if your drop hangs from above the baseline rather than sitting on top of it")
    parser.add_argument("--manual_drop_region", action="store_true",
                         help="select the drop region by hand instead of using full width")
    parser.add_argument("--show_features", action="store_true")
    args = parser.parse_args()
    main(args)