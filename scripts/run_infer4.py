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



   ####qwen3vl中有四块ptach拼成一个ptach的步骤，主要过程如下(我们需要逆运算回去)
#      pos_embed = pos_embed.view(
#       t,
#       h // merge_size, merge_size,
#       w // merge_size, merge_size,
#       -1,
#   ).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
#对于注意力图没有最后一个维度
def unshuffle_attention(vec, t, grid_h, grid_w, merge):
    # vec: [seq_len]
    seq_len = t * grid_h * grid_w
    vec = vec[:seq_len]
    vec = vec.unsqueeze(-1)                  # -> [seq_len, 1]
    vec = vec.view(
        t,
        grid_h // merge,grid_w // merge,
        merge, merge,
        1,
    )
    vec = vec.permute(0, 1, 3, 2, 4, 5).contiguous().squeeze(-1)
    return vec.view(t, grid_h, grid_w)       # 最后一维被 squeeze 掉



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
    t, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
    h_m, w_m = grid_h // merge, grid_w // merge
    
    vision_len =  grid_h * grid_h

    row = max(0, min(args.token_row, h_m - 1))
    col = max(0, min(args.token_col, w_m - 1))
    i= row % merge
    j= col % merge
    x_blk= row // merge
    y_blk= col // merge
    
   # ============================================================
# 【Qwen3-VL / InternVL 空间合并 (Spatial Merge) 坐标映射说明】
# ============================================================
#
# 模型视觉塔输出的初始网格尺寸：
#   (t, grid_h, grid_w, hidden)
#   - t: 时间维（图像时通常 = 1）
#   - grid_h, grid_w: 图像被划分为的 patch 网格行列数
#     例如 24×36，即每张图像被拆成 24×36 个小 patch
#   - hidden: 每个 patch 的嵌入维度
#
# 模型的空间合并参数为：
#   merge_size = ms  （常见取值 2）
#   表示每 ms×ms 个 patch 合并为一个“粗粒度 patch”。
#
# ------------------------------------------------------------
# 一、view 重排
# ------------------------------------------------------------
#   pos_embed = pos_embed.view(
#       t,
#       grid_h // ms, ms,       # → (块行, 块内行)
#       grid_w // ms, ms,       # → (块列, 块内列)
#       hidden
#   )
#
#   原始 patch 坐标 (x, y) 映射为：
#       x_blk = x // ms    # 外层块行坐标
#       i     = x %  ms    # 块内行偏移
#       y_blk = y // ms    # 外层块列坐标
#       j     = y %  ms    # 块内列偏移
#
#   重排后坐标为：
#       (t_idx, x_blk, i, y_blk, j, hidden)
#
# ------------------------------------------------------------
# 二、permute 调换维度
# ------------------------------------------------------------
#   pos_embed = pos_embed.permute(0, 1, 3, 2, 4, 5)
#   维度顺序变为：
#       (t_idx, x_blk, y_blk, i, j, hidden)
#
#   这样同一 block 内的 ms×ms patch 会在内存中连续排列，
#   便于后续 flatten 或线性聚合。
#
# ------------------------------------------------------------
# 三、flatten 展平后的序列顺序（row-major）
# ------------------------------------------------------------
#   展平顺序: (t, x_blk, y_blk, i, j)
#
#   因此，原始坐标 (x, y) 对应 flatten 后序列的线性下标：
#
#       flat_idx =
#           ((((t_idx * (grid_h//ms) + x_blk)
#                 * (grid_w//ms) + y_blk)
#                 * ms + i)
#                 * ms + j)
    target_idx = (((((t-1)* (grid_h//merge) + x_blk)* (grid_w//merge) + y_blk)* merge + i)* merge + j)
    img_width, img_height = image.size
    patch_w = img_width / grid_w
    patch_h = img_height / grid_h
    rect_x = col * patch_w
    rect_y = row * patch_h


    print("Total attention blocks captured:", len(vision_block_attn))

    for idx, attn_probs in vision_block_attn:
        attn_flat = attn_probs.squeeze(0).mean(dim=0)  # average heads

        attn_flat = attn_flat[target_idx]
        
        attn_grid = unshuffle_attention(attn_flat, t, grid_h, grid_w, merge)

        attn_np = attn_grid.to(torch.float32).cpu().numpy().squeeze(0)
        min_val, max_val = np.percentile(attn_np, [1, 99])
        attn_clipped = np.clip(attn_np, min_val, max_val)
        attn_norm = (attn_clipped - min_val) / (max_val - min_val + 1e-6)
        attn_display = np.power(attn_norm, 0.8)

        cmap = plt.get_cmap("inferno")
        heat_rgb = (cmap(attn_display)[..., :3] * 255).astype("uint8")
        heat_img = Image.fromarray(heat_rgb).resize(image.size, Image.BILINEAR)
        heat_arr = np.array(heat_img).astype(np.float32)
        overlay_arr = np.clip(
            0.7 * np.array(image, dtype=np.float32) + 0.3 * heat_arr,
            0,
            255,
        ).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(attn_display, cmap="inferno")
        axes[0].set_title(f"Block {idx} true attention")
        axes[0].axis("off")

        axes[1].imshow(overlay_arr)
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
