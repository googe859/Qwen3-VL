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
import matplotlib.pyplot as plt
import numpy as np


merged_feats: list[tuple[str, torch.Tensor]] = []


def _normalize_hidden(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim == 2:  # (seq, hidden)
        return hidden.unsqueeze(0)
    if hidden.ndim == 3:  # (batch, seq, hidden)
        return hidden
    if hidden.ndim == 4:  # (batch, channels, h, w)
        b, c, h, w = hidden.shape
        return hidden.view(b, c, -1).transpose(1, 2)
    raise ValueError(f"Unsupported hidden shape: {hidden.shape}")



def register_merger_hooks(model, max_blocks=None):
    handles = []

    def capture(module_name):
        def hook(_, __, output):
            merged_feats.append((module_name, output.detach().cpu()))

        return hook

    handles.append(model.model.visual.merger.register_forward_hook(capture("merger_final")))

    deep_list = model.model.visual.deepstack_merger_list
    indexes = model.model.visual.deepstack_visual_indexes
    limit = len(indexes) if max_blocks is None else min(len(indexes), max_blocks)
    for idx in range(limit):
        block_idx = indexes[idx]
        handles.append(
            deep_list[idx].register_forward_hook(capture(f"deepstack_block_{block_idx:02d}"))
        )

    return handles



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
    parser.add_argument(
        "--max-early-blocks",
        type=int,
        default=4,
        help="Number of early vision blocks to capture (-1 for all).",
    )
    parser.add_argument(
        "--fixed-image-size",
        type=int,
        default=1024,
        help="Resize input image to (size, size) before feeding the model (-1 to disable).",
    )

    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def main() -> None:
    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)
    merged_feats.clear()
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
    max_blocks = None if args.max_early_blocks < 0 else args.max_early_blocks
    merger_handles = register_merger_hooks(model, max_blocks)

    orig_image = load_image(args.image)
    model_image = orig_image
    if args.fixed_image_size > 0:
        model_image = orig_image.resize(
            (args.fixed_image_size, args.fixed_image_size),
            Image.BICUBIC,
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": model_image},
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
        model(**inputs, output_hidden_states=True, return_dict=True)
    for handle in merger_handles:
        handle.remove()

     

       
       
        # layer_idx = 1  # 先挑第1层
        # hs = outputs.hidden_states[layer_idx][0, :vision_token_len, :]  # (864, 2560)# 形状: [batch, seq_len, hidden_dim]# 前N个位置: [vision_token_len, hidden_dim]
        # norms = hs.norm(dim=-1)  # (237?,)
        # grid = norms.view(t, h, w).squeeze(0)  # (24, 36)
        #print(grid.min().item(), grid.max().item())
        #print(outputs.keys()) #odict_keys(['logits', 'past_key_values', 'rope_deltas', 'hidden_states'])
        #print(len(outputs.hidden_states), outputs.hidden_states[0].shape)#37 torch.Size([1, 237, 2560])
        # print(inputs["image_grid_thw"]) # tensor([[ 1, 24, 36]], device='cuda:0')
        # print("pixel_values:", inputs["pixel_values"].shape) #pixel_values: torch.Size([864, 1536])
        # print("image_grid_thw:", inputs["image_grid_thw"]) #image_grid_thw: tensor([[ 1, 24, 36]], device='cuda:0')
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


    t, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
    merge_size = model.model.visual.spatial_merge_size
    merged_h = max(1, grid_h // merge_size)
    merged_w = max(1, grid_w // merge_size)
    model_arr = np.array(model_image, dtype=np.float32)

    for idx, (label, feat) in enumerate(merged_feats):
        feat = feat.to(torch.float32)
        vision_len = t * merged_h * merged_w
        pure_patches = feat[:vision_len]
        grid = pure_patches.norm(dim=-1).view(t, merged_h, merged_w)
        grid = grid.mean(0) if t > 1 else grid.squeeze(0)
        grid_np = grid.cpu().numpy()
        min_val, max_val = np.percentile(grid_np, [1, 99])
        grid_clipped = np.clip(grid_np, min_val, max_val)
        grid_norm = (grid_clipped - min_val) / (max_val - min_val + 1e-6)
        grid_display = np.power(grid_norm, 0.8)

        cmap = plt.get_cmap("inferno")
        heat_color = (cmap(grid_display)[..., :3] * 255).astype("uint8")
        heat = Image.fromarray(heat_color).resize(model_image.size, Image.BILINEAR)
        heat_arr = np.array(heat, dtype=np.float32)
        overlay_arr = np.clip(0.6 * model_arr + 0.4 * heat_arr, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(grid_display, cmap="inferno")
        axes[0].set_title(f"{label}")
        axes[0].axis("off")

        axes[1].imshow(overlay_arr.astype(np.uint8))
        axes[1].set_title("Overlayed Visualization")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"{idx:02d}_{label}.png", bbox_inches="tight")
        plt.close(fig)

    






if __name__ == "__main__":
    main()
