# SBE H200 Video Codecase

This repository runs a 5-6 hour H200-scale Wan2.1 text-to-video acceleration experiment for:

**SBE-Online Continuous RiskGate**

The goal is an AAAI-style pilot package: real video generation, fixed-cache baselines, official VBench-4D custom-input evaluation, and ablations that test whether the continuous online formula matters.

The code assumes the H200 node is almost empty. It checks and installs the video generation stack, spaCy, ffmpeg support, and official VBench before running the experiment.

The runner also tries multiple runtime paths automatically. If the current checkout directory has little disk space or cannot be written to, it tries common H200/cloud paths such as `/root/autodl-tmp`, `/mnt/data`, `/data`, `$HOME`, and `/tmp`. The selected cache/output path is printed at startup.

## One-command H200 run

```bash
git clone https://github.com/tangzhexuan378-code/sbe-h200-video-codecase.git
cd sbe-h200-video-codecase

bash scripts/run_all_h200.sh
```

Equivalent explicit commands:

```bash
bash scripts/bootstrap_env.sh
bash scripts/run_h200_6h.sh
```

If the Wan model is already available locally:

```bash
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export LOCAL_FILES_ONLY=1
bash scripts/run_h200_6h.sh
```

Shorter debugging variants:

```bash
bash scripts/run_smoke.sh       # tiny environment check
bash scripts/run_h200_4h.sh     # shorter 4h-style run
```

## Main method

The main method does online prompt parsing. It does not use pre-labeled block types.

```text
prompt
-> spaCy online parser
-> semantic feature vector phi(B)
-> uncertainty U(B)
-> risk R(B)
-> continuous risk intensity q(B)
-> per-denoising-step cache threshold tau_t(B)
-> video generation
```

Formulas:

```math
R(B)=w^\top \phi(B)+\lambda U(B)
```

```math
q(B)=\mathrm{clip}\left(\frac{R(B)-r_0}{s}+\eta U(B),0,1\right)
```

```math
\tau_t(B)=b_t(1-\alpha q(B))-\beta q(B)
```

Meaning:

- `phi(B)` is extracted online from the prompt using spaCy tokens plus general rules.
- `U(B)` is parsing uncertainty. Vague prompts, many entities, and multiple actions increase uncertainty.
- `R(B)` is semantic risk.
- `q(B)` maps risk to a continuous value in `[0, 1]`.
- `tau_t(B)` is the cache threshold for denoising step `t`.
- `None` in a threshold schedule means no-cache / full compute for that step.

## H200 5-6 hour setting

Default config: `configs/h200_6h_online_continuous.yaml`.

| Item | Setting |
|---|---|
| Model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Resolution | `832 x 480` |
| Frames | `129` |
| Steps | `12` |
| Prompt split | heldout |
| Prompt count | `60`, balanced round-robin across semantic types |
| Main budget | `5.5` generation hours plus official VBench-4D evaluation |
| Output videos | around 600+ generated videos, depending on time budget and PAB cap |

This is intentionally larger than the local 5090 pilot: more prompts, longer videos, more methods, and official VBench evaluation over the generated videos.

## Compared methods

The default 6h config runs:

| Method | Role |
|---|---|
| `baseline_12step` | Wan no-cache 12-step baseline |
| `teacache_12step_t0.2` | mild fixed TeaCache baseline |
| `teacache_12step_t0.3` | medium fixed TeaCache baseline |
| `teacache_12step_t0.45` | aggressive fixed TeaCache baseline |
| `fastercache_s2` | Diffusers FasterCache baseline |
| `pab_s2_c3` | PAB baseline, capped to preserve budget |
| `sbe_online_discrete` | online parsing, but discrete low/medium/high window ablation |
| `sbe_online_continuous_no_u` | continuous threshold without uncertainty `U(B)` |
| `sbe_online_continuous_rule_only` | continuous threshold without spaCy model; rule-only parser |
| `sbe_online_continuous_quality` | quality-first continuous variant |
| `sbe_online_continuous_fast` | speed-first continuous variant |
| `sbe_online_continuous_full` | main method: continuous threshold with uncertainty |

Key ablation logic:

| Comparison | Reviewer question answered |
|---|---|
| full vs fixed TeaCache | Is semantic risk better than a fixed cache threshold? |
| full vs discrete | Is the continuous formula better than a low/medium/high table? |
| full vs no-U | Does uncertainty `U(B)` help? |
| full vs rule-only | Does spaCy online parsing help beyond keyword rules? |
| fast / quality / full | What speed-quality frontier does the method expose? |

## Outputs

After `bash scripts/run_h200_6h.sh`, outputs are written to:

```text
outputs/<run_name>/
```

Important files:

| File | Meaning |
|---|---|
| `main_table.csv` | final paper-style table |
| `main_table.md` | same table in Markdown |
| `report.tex` | LaTeX report |
| `report.pdf` | compiled PDF if `pdflatex` exists |
| `generation_rows.csv` | raw generation rows, latency, memory, online risk, q, U, features, threshold schedule |
| `per_video_rows.csv` | per-video sampled-frame metrics |
| `official_vbench4_scores.json` | official VBench custom-input 4D scores |
| `official_vbench4_scores.csv` | official VBench custom-input 4D scores as CSV |
| `env_status.json` | CUDA / torch / spaCy / VBench / ffmpeg environment |
| `videos/` | generated videos |
| `frames/` | sampled frames for SSIM/PSNR/LPIPS-L1 proxy |

## Main table columns

| Column | Meaning | Direction | Source |
|---|---|---:|---|
| `FLOPs_proxy_down` | approximate compute cost | lower is better | proxy from TeaCache compute/skip events; not Nsight |
| `Speedup_up` | speedup over Wan-12 | higher is better | measured wall-clock latency ratio |
| `Latency_down` | latency | lower is better | real wall-clock timing |
| `VBench_up` | official VBench-4D average | higher is better | official `vbench evaluate`, custom-input |
| `VBench4_imaging_up` | imaging quality | higher is better | official VBench |
| `VBench4_temporal_up` | temporal stability | higher is better | official VBench temporal flickering |
| `VBench4_motion_up` | motion smoothness | higher is better | official VBench |
| `VBench4_dynamic_up` | dynamic degree | higher is better | official VBench |
| `LPIPS_L1_down` | sampled-frame L1 proxy | lower is better | proxy |
| `SSIM_up` | structural similarity | higher is better | computed from sampled frames |
| `PSNR_up` | peak signal-to-noise ratio | higher is better | computed from sampled frames |

Honesty notes:

- Official VBench-4D is real official VBench custom-input evaluation.
- FLOPs is a proxy unless you additionally run Nsight / torch profiler.
- Full 16D VBench is not claimed by this code path.
- The scripts fail rather than invent official VBench scores.

## Empty-node setup policy

The bootstrap script handles an empty H200 node:

| Component | Behavior |
|---|---|
| Python env | creates `.venv` |
| torch / diffusers / transformers | installs from `requirements.txt` |
| spaCy | installs `spacy`, tries `en_core_web_sm`, falls back to `spacy.blank("en") + rules` |
| VBench | installs `vbench`; if official VBench fails, the strict H200 run stops |
| ffmpeg | uses `imageio-ffmpeg` |
| Wan model | downloads via Diffusers unless `WAN_MODEL_PATH` / `LOCAL_FILES_ONLY=1` is set |
| CUDA/GPU | logged in `env_status.json`; warns if GPU is not H200 |

## Path fallback and retry policy

The scripts try hard to recover from common cluster issues, but they do not fake success.

| Failure type | Automatic recovery |
|---|---|
| repo disk too small | choose a writable large path from `/root/autodl-tmp`, `/mnt/data`, `/data`, `$HOME`, `/tmp` |
| Hugging Face / model cache path unwritable | move `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `DIFFUSERS_CACHE` to the selected runtime path |
| pip cache path unwritable | move `PIP_CACHE_DIR` to the selected runtime path |
| VBench / torch cache path unwritable | move `XDG_CACHE_HOME`, `TORCH_HOME`, `VBENCH_CACHE` to the selected runtime path |
| default PyPI slow/fails | retry with Tsinghua PyPI mirror |
| VBench PyPI fails | retry with GitHub source install |
| spaCy model download fails | fallback to `spacy.blank("en") + rules`, recorded in output |
| no GPU / no driver / too little disk | fail early with `preflight_status.json`; no fake table |

To force a specific large disk:

```bash
export SBE_WORK_ROOT=/path/to/large/writable/disk
bash scripts/run_all_h200.sh
```

## Expected paper-facing interpretation

The desired outcome is not simply "fastest method wins." The main claim is:

> SBE-Online Continuous RiskGate parses each prompt online, computes a continuous semantic risk score, and maps the score to denoising-step cache thresholds. This gives a better speed-quality tradeoff than fixed cache policies while avoiding VLM-in-the-loop runtime overhead and avoiding pre-labeled block types.

The run is designed to support or falsify that claim with real generated videos and official VBench-4D scores.
