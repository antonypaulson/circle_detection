"""Visual viewer: OpenCV windows, or headless overlay export."""

from __future__ import annotations

import argparse
import os
import sys

from circle_detection.constants import DEFAULT_INFER_SAMPLES, DEFAULT_MODEL_PATH, DEFAULT_NOISE, IMAGE_SIZE
from circle_detection.infer import run_inference
from circle_detection.shapes import iou
from circle_detection.visualize import (
    draw_prediction_overlay,
    image_to_uint8,
    require_cv2,
    upscale,
)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--samples", type=int, default=DEFAULT_INFER_SAMPLES)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip OpenCV windows; print stats and optionally save overlays.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Write overlay PNGs (also used automatically in headless mode if unset).",
    )
    parser.add_argument("--max-save", type=int, default=16)
    return parser


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def show_interactive(images, labels, predictions, image_size: int = IMAGE_SIZE) -> None:
    cv = require_cv2()
    print("Press f/j or right-arrow for the next image; q or Esc to quit.")
    for i, prediction in enumerate(predictions):
        pred_px = image_size * prediction
        true_px = image_size * labels[i]
        prediction_int = [
            int(round(pred_px[0])),
            int(round(pred_px[1])),
            max(int(round(pred_px[2])), 1),
        ]
        score = iou((true_px[0], true_px[1], true_px[2]), (pred_px[0], pred_px[1], pred_px[2]))
        print(f"True x, y, r: {int(true_px[0])} {int(true_px[1])} {int(true_px[2])}")
        print(f"Pred x, y, r: {prediction_int[0]} {prediction_int[1]} {prediction_int[2]}")
        print(f"IOU: {score:2.2f}\n")

        raw = upscale(image_to_uint8(images[i]))
        overlay = upscale(draw_prediction_overlay(images[i], pred_px[0], pred_px[1], pred_px[2]))
        cv.imshow("Raw Image", raw)
        cv.imshow("Overlay Image", overlay)

        while True:
            key = cv.waitKey(0)
            # q / Esc
            if key in (ord("q"), ord("Q"), 27, 1048603, 1048689):
                cv.destroyAllWindows()
                sys.exit(0)
            # f / j / right arrow
            if key in (ord("f"), ord("j"), 102, 106, 65363):
                break

    cv.destroyAllWindows()


def run(args: argparse.Namespace):
    headless = args.headless or not _has_display()
    save_dir = args.save_dir
    if headless and save_dir is None:
        save_dir = "inference_output"

    images, labels, predictions, _scores = run_inference(
        model_path=args.model_path,
        samples=args.samples,
        noise=args.noise,
        image_size=args.image_size,
        save_dir=save_dir if headless or args.save_dir else None,
        seed=args.seed,
        max_save=args.max_save,
    )

    if headless:
        if not args.headless and not _has_display():
            print("No display detected; ran headless and wrote overlays instead of OpenCV windows.")
        return images, labels, predictions

    show_interactive(images, labels, predictions, image_size=args.image_size)
    return images, labels, predictions


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="View circle detections. Uses OpenCV windows when a display is available."
    )
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
