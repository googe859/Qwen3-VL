#!/usr/bin/env python
"""
Attention-like visualization for Qwen3-VL vision blocks.

This script runs a single-turn VQA inference while capturing the vision
blocks' outputs. For a user-specified image token (row, col), it computes
the cosine similarity between that token and all other visual tokens per
block, providing an attention-style heatmap plus an overlay on the image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Buffer to collect per-block outputs
vision_block_feats: list[torch.Tensor] = []


def register_vision_hooks(model):
    """Attach hooks that save each vision block's output."""

    handles = []

    def capture_output(_, __, output):
        hidden = output[0] if isinstance(output, tuple) else output
        vision_block_feats.append(hidden.detach().cpu())

    for block in model.model.visual.blocks:
        handles.append(block.register_forward_hook(capture_output))
    return handles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize per-block similarities for Qwen3-VL.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/Qwen3-VL-4B-Instruct",
        help="Model directory or HF repo id.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--token-row",
        type=int,
        default=0,
        help="Row index (after spatial merge) for the target visual token.",
    )
    parser.add_argument(
        "--token-col",
        type=int,
        default=0,
        help="Column index (after spatial merge) for the target visual token.",
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
        help="Run everything on CPU (slower).",
    )
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def main() -> None:
    args = parse_args()
    out_dir = Path("visualizations_attention")
    out_dir.mkdir(exist_ok=True)
    vision_block_feats.clear()

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

    hook_handles = register_vision_hooks(model)

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

    print("Running representation pass ...")
    with torch.inference_mode():
        _ = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    for handle in hook_handles:
        handle.remove()

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

    merge = model.model.visual.spatial_merge_size
    t, h, w = inputs["image_grid_thw"][0].tolist()
    h_m, w_m = h // merge, w // merge
    vision_len = t * h_m * w_m

    row = max(0, min(args.token_row, h_m - 1))
    col = max(0, min(args.token_col, w_m - 1))
    target_idx = row * w_m + col

    img_width, img_height = image.size
    patch_w = img_width / w_m
    patch_h = img_height / h_m
    rect_x = col * patch_w
    rect_y = row * patch_h

    print(f"Target token -> row {row}, col {col}, index {target_idx}")

    for idx, feat in enumerate(vision_block_feats[: len(model.model.visual.blocks)]):
        block_feat = feat[:vision_len]  # (vision_len, hidden)

        target_vec = block_feat[target_idx : target_idx + 1]
        sim = F.cosine_similarity(block_feat, target_vec, dim=-1)
        sim_grid = sim.view(t, h_m, w_m).squeeze(0)

        sim_np = sim_grid.to(torch.float32).cpu().numpy()
        sim_norm = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-6)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(sim_np, cmap="inferno")
        axes[0].set_title(f"Block {idx} similarity")
        axes[0].axis("off")

        heat_resized = Image.fromarray((sim_norm * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        axes[1].imshow(image)
        axes[1].imshow(heat_resized, cmap="inferno", alpha=0.4)
        axes[1].add_patch(
            plt.Rectangle(
                (rect_x, rect_y),
                patch_w,
                patch_h,
                linewidth=1.0,
                edgecolor="red",
                facecolor="none",
            )
        )
        axes[1].set_title("Overlay")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"block_{idx:02d}_token_{row}_{col}.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
