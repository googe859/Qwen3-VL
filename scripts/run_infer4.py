#!/usr/bin/env python
"""
Attention map visualization for Qwen3-VL vision blocks.

This script captures the TRUE attention weights from each vision block
by registering forward hooks on their attention modules. For a chosen
visual token (row, col) after spatial merge, it extracts the attention
scores directed from that token to all others, and produces a heatmap +
overlay similar to run_infer3, but based on actual attention instead of
cosine similarity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_mod

# Storage for per-block attention maps (after forward hooks fire)
vision_block_attn: list[tuple[int, torch.Tensor]] = []
ORIGINAL_EAGER_FN = qwen3_mod.eager_attention_forward


def capture_eager_attention_forward(self, query_states, key_states, value_states, *args, **kwargs):
    attn_output, attn_weights = ORIGINAL_EAGER_FN(self, query_states, key_states, value_states, *args, **kwargs)
    if attn_weights is not None:
        # attn_weights shape: [batch, num_heads, seq_len, seq_len]
        block_idx = getattr(self, "_block_idx", None)
        if block_idx is not None:
            vision_block_attn.append((block_idx, attn_weights.detach().cpu()))
    return attn_output, attn_weights


def install_attention_hooks(model):
    """Switch attention implementation to eager and monkey patch the eager forward."""

    model.set_attn_implementation("eager")
    qwen3_mod.eager_attention_forward = capture_eager_attention_forward

    for idx, block in enumerate(model.model.visual.blocks):
        block.attn.config._attn_implementation = "eager"
        block.attn._block_idx = idx


def restore_attention_hooks(model):
    """Restore original attention implementation."""

    qwen3_mod.eager_attention_forward = ORIGINAL_EAGER_FN
    for block in model.model.visual.blocks:
        if hasattr(block.attn, "_block_idx"):
            del block.attn._block_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize true attention maps for Qwen3-VL vision blocks.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--token-row", type=int, default=0)
    parser.add_argument("--token-col", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def main():
    args = parse_args()
    out_dir = Path("visualizations_true_attention")
    out_dir.mkdir(exist_ok=True)
    vision_block_attn.clear()

    device_map = "auto"
    torch_dtype = "auto"
    if args.cpu_only:
        device_map = {"": "cpu"}
        torch_dtype = torch.float32

    print(f"Loading model from {args.checkpoint} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint, device_map=device_map, torch_dtype=torch_dtype
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    install_attention_hooks(model)

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

    print("Running representation pass to capture attention ...")
    with torch.inference_mode():
        _ = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    restore_attention_hooks(model)

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

    print("Total attention blocks captured:", len(vision_block_attn))

    for idx, attn_probs in vision_block_attn:
        attn_flat = attn_probs.squeeze(0).mean(dim=0)  # average heads
        attn_flat = attn_flat[target_idx]
        attn_grid = attn_flat[:vision_len].view(t, h_m, w_m).squeeze(0)

        attn_np = attn_grid.to(torch.float32).cpu().numpy()
        attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-6)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(attn_np, cmap="inferno")
        axes[0].set_title(f"Block {idx} true attention")
        axes[0].axis("off")

        heat_resized = Image.fromarray((attn_norm * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        axes[1].imshow(image)
        axes[1].imshow(heat_resized, cmap="inferno", alpha=0.4)
        axes[1].add_patch(
            plt.Rectangle((rect_x, rect_y), patch_w, patch_h, linewidth=1.0, edgecolor="red", facecolor="none")
        )
        axes[1].set_title("Overlay")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"block_{idx:02d}_token_{row}_{col}_true_attn.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
