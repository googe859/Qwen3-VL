#!/usr/bin/env python
"""
Utility to stack per-layer visualization images vertically.

Given a directory containing files like `block_00_l2.png`, this script sorts
them alphanumerically and concatenates them top-to-bottom into one tall image.
It works for outputs from `run_infer2.py`, `run_infer3.py`, or `run_infer4.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stack visualization images vertically.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing per-layer PNG files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern to select images (default: *.png).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stitched_layers.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=10,
        help="Vertical spacing (pixels) between stacked images.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="#ffffff",
        help="Background color (hex or PIL color name).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} under {input_dir}")

    images = [Image.open(fp).convert("RGB") for fp in files]
    widths = [img.width for img in images]
    heights = [img.height for img in images]

    canvas_width = max(widths)
    canvas_height = sum(heights) + args.spacing * (len(images) - 1)
    stitched = Image.new("RGB", (canvas_width, canvas_height), color=args.background)

    y = 0
    for img, fp in zip(images, files):
        if img.width < canvas_width:
            pad = Image.new("RGB", (canvas_width, img.height), color=args.background)
            pad.paste(img, (0, 0))
            img = pad
        stitched.paste(img, (0, y))
        y += img.height + args.spacing
        img.close()

    stitched.save(args.output)
    print(f"Saved stitched image with {len(files)} layers to {args.output}")


if __name__ == "__main__":
    main()
