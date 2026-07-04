# SBE H200 Video Codecase

This repository runs an H200-scale Wan2.1 text-to-video acceleration experiment for:

**SBE-Online Continuous RiskGate**
中文：**在线连续语义风险门控缓存调度**。

The code assumes the H200 node is almost empty. It checks and installs the video generation stack, spaCy, ffmpeg support, and official VBench before running the experiment.

## One-Command H200 Run

```bash
git clone https://github.com/tangzhexuan378-code/sbe-h200-video-codecase.git
cd sbe-h200-video-codecase

bash scripts/bootstrap_env.sh
bash scripts/run_h200_4h.sh
```

If the Wan model is already available locally:

```bash
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export LOCAL_FILES_ONLY=1
bash scripts/run_h200_4h.sh
```

## What The Main Method Does

The main method does **online** prompt parsing. It does not use pre-labeled block types.

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

## Compared Methods

The default H200 config runs:

| Method | Purpose |
|---|---|
| `baseline_12step` | Wan no-cache 12-step baseline |
| `teacache_12step_t0.2` | mild fixed TeaCache baseline |
| `teacache_12step_t0.3` | medium fixed TeaCache baseline |
| `teacache_12step_t0.45` | aggressive fixed TeaCache baseline |
| `fastercache_s2` | Diffusers FasterCache baseline |
| `pab_s2_c3` | PAB baseline, optional and capped by prompt count |
| `sbe_online_discrete` | online parsing, but discrete low/medium/high ablation |
| `sbe_online_continuous_no_u` | continuous threshold without uncertainty `U(B)` |
| `sbe_online_continuous_full` | main method: continuous threshold with uncertainty |

## Outputs

After `bash scripts/run_h200_4h.sh`, outputs are written to:

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
| `generation_rows.csv` | raw generation rows, latency, memory, scheduler details |
| `per_video_rows.csv` | per-video proxy metrics and sampled-frame metrics |
| `official_vbench4_scores.json` | official VBench custom-input 4D scores |
| `official_vbench4_scores.csv` | official VBench custom-input 4D scores as CSV |
| `env_status.json` | CUDA / torch / spaCy / VBench / ffmpeg environment |
| `videos/` | generated videos |
| `frames/` | sampled frames for SSIM/PSNR/LPIPS-L1 proxy |

## Metrics

| Column | Chinese | Direction | Source |
|---|---|---:|---|
| `FLOPs_proxy_down` | FLOPs 近似计算量 | lower is better | proxy from TeaCache compute/skip events; not Nsight |
| `Speedup_up` | 加速比 | higher is better | measured wall-clock latency ratio |
| `Latency_down` | 延迟 | lower is better | real wall-clock timing |
| `VBench_up` | 官方 VBench 4维均分 | higher is better | official `vbench evaluate`, custom-input |
| `VBench4_imaging_up` | 成像质量 | higher is better | official VBench |
| `VBench4_temporal_up` | 时序稳定 | higher is better | official VBench temporal flickering |
| `VBench4_motion_up` | 运动平滑 | higher is better | official VBench |
| `VBench4_dynamic_up` | 动态程度 | higher is better | official VBench |
| `LPIPS_L1_down` | 感知差异近似 | lower is better | sampled-frame L1 proxy |
| `SSIM_up` | 结构相似度 | higher is better | computed from sampled frames |
| `PSNR_up` | 峰值信噪比 | higher is better | computed from sampled frames |

Important honesty note:

- Official VBench 4D is real official VBench custom-input evaluation.
- FLOPs is a proxy unless you additionally run Nsight / torch profiler.
- Full 16D VBench is not claimed by this code path.

## Smoke Test

For a quick sanity check:

```bash
bash scripts/bootstrap_env.sh
bash scripts/run_smoke.sh
```

The smoke test uses smaller videos and fewer prompts. It is only for environment validation.

## If Something Fails

The scripts intentionally fail rather than invent scores.

- If VBench cannot be installed, `run_h200_4h.sh` stops by default.
- If spaCy model download fails, the parser falls back to `spacy.blank("en") + rules`, and this is recorded in `env_status.json` and `generation_rows.csv`.
- If the Wan model cannot be downloaded, set `WAN_MODEL_PATH` to a local model directory or configure the server network / Hugging Face mirror.
