#!/usr/bin/env python
"""
Minimal VQA inference entry point for Qwen3-VL.

Usage example:
    python scripts/run_infer.py \
        --checkpoint ./checkpoints/Qwen3-VL-4B-Instruct \
        --image ./cookbooks/assets/demo.jpeg \
        --question "Describe the image."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VQA inference with Qwen3-VL.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/Qwen3-VL-4B-Instruct",
        help="Model directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image used for VQA.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation length for the answer.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force running inference on CPU.",
    )
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def main() -> None:
    args = parse_args()

    device_map = "auto"
    torch_dtype = "auto"
    if args.cpu_only:
        device_map = {"": "cpu"}
        torch_dtype = torch.float32

    print(f"Loading model from {args.checkpoint} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    image = load_image(args.image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    print("Running generation ...")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print("\nQuestion:", args.question)
    print("Answer:", answer)
    import inspect
    print(inspect.signature(model.forward))


if __name__ == "__main__":
    main()
