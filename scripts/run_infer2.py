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
#hook get vision block features
vision_block_feats = []
def register_vision_hooks(model):
    """Capture each vision block's output."""
    handles = []

    def capture_output(_, __, output):
        hidden = output[0] if isinstance(output, tuple) else output
        vision_block_feats.append(hidden.detach().cpu())

    for block in model.model.visual.blocks:
        handles.append(block.register_forward_hook(capture_output))
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
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def main() -> None:
    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)
    vision_block_feats.clear()
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

    print("Running generation ...")
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    for handle in hook_handles:
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
    # print("\nQuestion:", args.question)
    # print("Answer:", answer)
    # import inspect
    # print(inspect.signature(model.forward))
    # Remove hooks
    
    print("Captured blocks:", len(vision_block_feats))
    print("Vision block count:", len(model.model.visual.blocks))

    merge = model.model.visual.spatial_merge_size  # 通常=2
    t, h, w = inputs["image_grid_thw"][0].tolist()
    h_m, w_m = h // merge, w // merge
    vision_len = t * h_m * w_m

    for idx, feat in enumerate(vision_block_feats):
        grid = feat[:vision_len].norm(dim=-1).view(t, h_m, w_m).squeeze(0)
        grid_np = grid.to(torch.float32).cpu().numpy()
        grid_norm = (grid_np - grid_np.min()) / (grid_np.max() - grid_np.min() + 1e-6)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(grid_np, cmap="inferno")
        axes[0].set_title(f"Vision Block {idx} - L2 Heatmap")
        axes[0].axis("off")

        heat_resized = Image.fromarray((grid_norm * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        axes[1].imshow(image)
        axes[1].imshow(heat_resized, cmap="inferno", alpha=0.4)
        axes[1].set_title("Overlayed Visualization")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"block_{idx:02d}_l2.png", bbox_inches="tight")
        plt.close(fig)






if __name__ == "__main__":
    main()
