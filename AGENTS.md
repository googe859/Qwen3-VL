# Repository Guidelines

## Project Structure & Module Organization
- `scripts/` hosts runnable entry points for vision-language inference (`run_infer.py`), L2 heatmaps (`run_infer2*.py`), attention probes, and stitching helpers. Generated PNGs are saved under `visualizations*/`.
- `cookbooks/` contains tutorial notebooks plus demo media in `cookbooks/assets/` for quick experiments.
- `evaluation/mmmu/` provides the MMMU benchmark harness (`run_mmmu.py`) with helper utilities and its own `requirements.txt`.
- `qwen-vl-utils/` and `qwen-vl-finetune/` are editable packages for preprocessing, patch ops, and fine-tuning. Install them with `pip install -e ...` when you need to modify shared logic.
- `checkpoints/` stores downloaded weights, while `docker/` holds container specs for reproducible runs.

## Build, Test, and Development Commands
- `pip install -r requirements_web_demo.txt` sets up the minimal runtime for local scripts and demos.
- `pip install -e qwen-vl-utils` (and optionally `qwen-vl-finetune`) enables live editing of the utility layers.
- `python scripts/run_infer.py --checkpoint ./checkpoints/Qwen3-VL-4B-Instruct --image cookbooks/assets/eg.jpg --question "..."` performs a single VQA smoke test.
- `python scripts/run_infer2.py --checkpoint ... --image ... --question ...` emits per-layer L2 heatmaps; append `--cpu-only` on low-VRAM hosts.
- `python evaluation/mmmu/run_mmmu.py infer MMMU_DEV_VAL ...` then `python evaluation/mmmu/run_mmmu.py eval ...` runs the benchmark and judge scoring (requires API keys in env).

## Coding Style & Naming Conventions
- Target Python 3.10+, four-space indentation, snake_case for functions/modules, CamelCase for classes, and kebab-case CLI flags.
- `qwen-vl-utils/pyproject.toml` configures Ruff (line length 119, double quotes). Run `ruff check qwen-vl-utils/src` and `ruff check --select I ...` to auto-order imports before sending patches.
- Prefer explicit type hints, short helpers, and docstrings that explain tensor shapes or device placement; add lightweight inline comments when hook order matters in visualization code.

## Testing Guidelines
- No unified unit suite exists; run the specific script you touched plus the base VQA smoke test. Keep a lightweight sample image (e.g., `cookbooks/assets/eg.jpg`) handy.
- For MMMU contributions, rerun `run_mmmu.py infer` on a small split, archive JSONL outputs, and re-evaluate with the same judge to confirm deltas.
- When updating reusable utilities, embed doctest-style snippets in docstrings so downstream notebooks exercise the new behavior.

## Commit & Pull Request Guidelines
- Follow the repo’s short, imperative commit subjects (e.g., `Add visualization hooks`). Keep messages under ~72 chars and focus on the primary change.
- PRs should include: concise summary + rationale, exact test commands with results (attach heatmap screenshots or paths for visualization diffs), and links to any issues, discussions, or external checkpoints.
- Do not check in large binaries; store artifacts outside Git and reference their download path or command inside the PR description.

## Environment & Security Notes
- Match CUDA drivers with `nvidia-smi` output; WSL users still need CUDA Toolkit installed inside the distro (`/usr/local/cuda-*`). Update `PATH` and `LD_LIBRARY_PATH` accordingly before running torch builds.
- Keep secrets (API keys for MMMU judges or HF tokens) in environment variables rather than committing them to the repo or notebooks.
