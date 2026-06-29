# SBE H200 Video Codecase

This repository runs a real Wan2.1 text-to-video acceleration experiment for an H200 GPU. It is designed to produce a table with:

`Method | FLOPs proxy | Speedup | Latency | VBench | LPIPS-L1 | SSIM | PSNR`

The default H200 config uses `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `832x480`, `129` frames, `12` steps, and up to `80` heldout prompts under a 3 hour budget.

## What It Tests

- `Wan no-cache 12-step`: baseline.
- `TeaCache t=0.2` and `TeaCache t=0.45`: public cache baselines.
- `FasterCache`: Diffusers hook baseline.
- `Uniform TeaCache t=0.3`: simple non-semantic cache policy.
- `SBE-RiskGate v5`: the proposed semantic-risk pre-dispatch policy.

SBE-RiskGate v5 does not do expensive online VLM checking. It dispatches before generation:

- low-risk blocks such as attribute / containment / presence use TeaCache with type-specific thresholds;
- high-risk blocks such as spatial relation / state change / counting use no-cache 12-step.

## Setup

```bash
git clone <your-repo-url>
cd sbe-h200-video-codecase
bash scripts/setup.sh
```

`scripts/setup.sh` installs Python dependencies, installs this package with `pip install -e .`, checks CUDA, checks `imageio-ffmpeg`, and tries to install official VBench. If the official VBench install fails, generation still works and the runner will clearly mark the VBench column as `proxy`.

If the H200 server already has the Wan model downloaded:

```bash
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export LOCAL_FILES_ONLY=1
```

If it does not, Diffusers will download `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` from Hugging Face, assuming network access and Hugging Face permission are available.

## Quick Smoke Test

```bash
bash scripts/run_smoke.sh
```

This generates a tiny `320x192`, `33` frame run to verify the environment.

## Main H200 Run

```bash
bash scripts/run_h200_3h.sh
```

Outputs are written to `outputs/<run_name>/`:

- `main_table.csv`
- `main_table.md`
- `per_video_rows.csv`
- `generation_rows.csv`
- `eval_status.json`
- `contact_sheet_middle.jpg`
- `report.tex`
- `report.pdf` if `pdflatex` is installed

## VBench Modes

The code distinguishes official VBench from a proxy score.

Default mode:

```bash
bash scripts/run_h200_3h.sh
```

This allows proxy VBench if official VBench results are not supplied. `eval_status.json` will say:

```json
{"vbench_source": "proxy"}
```

Strict mode:

```bash
export REQUIRE_OFFICIAL_VBENCH=1
export OFFICIAL_VBENCH_JSON=/path/to/official_vbench_result.json
bash scripts/run_h200_3h.sh
```

Strict mode refuses to run the final table without an official VBench JSON. This avoids accidentally reporting a proxy as official VBench.

After a normal generation run, you can also try official VBench custom-input evaluation:

```bash
bash scripts/run_official_vbench_custom.sh outputs/<run_name>
```

This script organizes generated videos by method and calls the official `vbench evaluate --mode=custom_input` command for the custom-input dimensions supported by VBench: subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, and imaging quality.

If VBench produces a JSON summary, rebuild the final table with:

```bash
python -m sbe_h200_video.rebuild_report \
  --config configs/h200_3h.yaml \
  --out-dir outputs/<run_name> \
  --official-vbench-json /path/to/vbench_result.json
```

## Why This Uses H200 Better Than a Small Local GPU

The main config intentionally uses longer and larger videos: `129` frames at `832x480`, multiple methods, and a balanced heldout prompt set. This makes memory bandwidth and VRAM matter, while keeping the wall-clock budget around three hours.

On a small local GPU, use `configs/smoke.yaml` or reduce `max_prompts`, `num_frames`, `width`, and `height`.
