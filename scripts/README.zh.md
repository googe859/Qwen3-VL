# 推理与可视化脚本

本目录收录了多种 Qwen3-VL 的本地运行脚本，覆盖 L2 热力图、注意力可视化以及不同的视觉 token 裁剪策略。使用前请先按照根目录 README 安装依赖，并将模型 checkpoint 放在 `./checkpoints/`。

## `run_infer.py`
最小化 VQA 演示：输入一张图片和一个问题，输出模型回答。
```
python scripts/run_infer.py \
  --checkpoint ./checkpoints/Qwen3-VL-4B-Instruct \
  --image ./cookbooks/assets/demo.jpeg \
  --question "Describe the scene."
```

## `run_infer2.py` – L2 热力图
捕获各个视觉 block 的输出，对视觉 token 做 L2 范数，并生成热图/叠加图，保存到 `visualizations/`。

## `run_infer2_2.py` – 早期视觉塔 L2
在 patch embedding 和 early blocks 上挂钩，得到未合并前的更清晰轮廓，结果写入 `visualizations_vtower/`。

## `run_infer4.py` – 真正注意力
切换到 eager attention 并抓取 softmax 权重，绘制真实注意力图，输出目录 `visualizations_true_attention/`（显存开销较大）。

## `run_infer2.2.py` – 基线 vs 稀疏（保留最高 L2）
先运行一次完整视觉 token 作基线，再按 `--keep-ratio` 只保留 L2 最大的 token，并在语言模型前删除其余 token，用于对比答案差异。

## `run_infer2.3.py` – 保留最低 L2 token
基于 2.2 改写，改成保留 L2 最小的一批 token，适用于研究“去掉显著区域”后的影响。

## `run_infer2.4.py` – 随机抽取 token
稀疏阶段随机采样视觉 token（固定随机种子），用来比较结构化裁剪与随机丢弃的效果。

## `keep_ratio_sweep.py` – KEEP_RATIO 扫描
批量调用 `run_infer2.2.py`，依次遍历多个 `--keep-ratio` 值，收集 Baseline/Sparse 答案到一个文本文件。

## `stitch_layers.py`
将同一脚本生成的多张 PNG 按顺序拼接成长图，方便整体浏览。

