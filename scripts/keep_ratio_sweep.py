#!/usr/bin/env python
"""
Sweep KEEP_RATIO values for run_infer2.2.py and collect baseline/sparse answers.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable


def default_ratio_schedule() -> list[float]:
    coarse = [round(x / 10, 2) for x in range(1, 10)]
    fine = [0.01, 0.03, 0.05, 0.07, 0.09]
    return sorted({*coarse, *fine})


def extract_block(text: str, start: str, end_markers: Iterable[str]) -> str | None:
    start_idx = text.find(start)
    if start_idx == -1:
        return None
    start_idx += len(start)
    end_idx = len(text)
    for marker in end_markers:
        marker_idx = text.find(marker, start_idx)
        if marker_idx != -1:
            end_idx = min(end_idx, marker_idx)
    block = text[start_idx:end_idx].strip()
    return block or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KEEP_RATIO sweep for run_infer2.2.py")
    parser.add_argument("--checkpoint", required=True, help="Path or HF repo id for the checkpoint.")
    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument("--question", required=True, help="Question to ask the model.")
    parser.add_argument("--output", default="keep_ratio_results.txt", help="File to store collected answers.")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="*",
        help="Custom KEEP_RATIO values. Defaults to 0.01-0.09 (step 0.02) and 0.1-0.9 (step 0.1).",
    )
    parser.add_argument("--run-script", default="scripts/run_infer2.2.py", help="Path to run_infer2.2.py.")
    parser.add_argument("--max-new-tokens", type=int, help="Forwarded to run_infer2.2.py.")
    parser.add_argument("--cpu-only", action="store_true", help="Forwarded to run_infer2.2.py.")
    parser.add_argument("--fixed-image-size", type=int, help="Forwarded to run_infer2.2.py.")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended verbatim when calling run_infer2.2.py.",
    )
    args = parser.parse_args()

    ratios = args.ratios or default_ratio_schedule()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    rows.append(f"Source script: {args.run_script}")
    rows.append(f"Image: {args.image}")
    rows.append(f"Question: {args.question}")
    rows.append("")

    for ratio in ratios:
        cmd = [
            "python",
            args.run_script,
            "--checkpoint",
            args.checkpoint,
            "--image",
            args.image,
            "--question",
            args.question,
            "--keep-ratio",
            f"{ratio:.4f}",
        ]
        if args.max_new_tokens is not None:
            cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
        if args.cpu_only:
            cmd.append("--cpu-only")
        if args.fixed_image_size is not None:
            cmd.extend(["--fixed-image-size", str(args.fixed_image_size)])
        if args.extra_args:
            cmd.extend(args.extra_args)

        print(f"[Sweep] KEEP_RATIO={ratio:.4f} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        stdout = result.stdout
        stderr = result.stderr
        baseline = extract_block(
            stdout,
            "Baseline Answer:",
            ["\nSparse Answer:", "\nKeeping top", "\n[Sparse", "\nSparse pass", "\nSparse visualization"],
        )
        sparse = extract_block(
            stdout,
            "Sparse Answer:",
            ["\nSparse pass", "\nSparse visualization", "\nQuestion:", "\nRunning representation"],
        )

        rows.append(f"=== KEEP_RATIO={ratio:.4f} ===")
        rows.append(f"Command: {' '.join(cmd)}")
        rows.append(f"Exit code: {result.returncode}")
        rows.append("Baseline Answer:")
        rows.append(baseline or "<missing>")
        rows.append("")
        rows.append("Sparse Answer:")
        rows.append(sparse or "<missing>")
        rows.append("")
        if result.returncode != 0 or stderr.strip():
            rows.append("stderr:")
            rows.append(stderr.strip() or "<empty>")
            rows.append("")

    output_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"Sweep finished. Results saved to {output_path}")


if __name__ == "__main__":
    main()
