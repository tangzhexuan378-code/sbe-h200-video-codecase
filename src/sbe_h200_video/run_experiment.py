from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import time
import types
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw

try:
    from diffusers.hooks import (
        FasterCacheConfig,
        PyramidAttentionBroadcastConfig,
        apply_faster_cache,
        apply_pyramid_attention_broadcast,
    )
except Exception:
    FasterCacheConfig = None
    PyramidAttentionBroadcastConfig = None
    apply_faster_cache = None
    apply_pyramid_attention_broadcast = None

from .cases import select_cases
from .spacy_online import continuous_threshold_schedule, discrete_threshold_schedule
from .teacache import (
    disable_teacache,
    install_teacache_forward,
    reset_teacache_state,
    summarize_teacache,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def stable_seed(case_id: str, seed_idx: int) -> int:
    return 41000 + seed_idx * 1000 + (zlib.crc32(case_id.encode("utf-8")) % 997)


def to_pil(frame: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype.kind == "f" and arr.max(initial=0) <= 1.5:
        arr = arr * 255.0
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def save_sample_frames(frames: list[Any], out_dir: Path, stem: str, count: int) -> dict[str, str]:
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(frames) - 1, max(3, count), dtype=int).tolist()
    paths: dict[str, str] = {}
    for pos, index in enumerate(indices):
        label = f"sample_{pos:02d}"
        path = frame_dir / f"{stem}_{label}.jpg"
        to_pil(frames[index]).save(path, quality=92)
        paths[label] = str(path)
    paths["frame_first"] = paths["sample_00"]
    paths["frame_middle"] = paths[f"sample_{len(indices) // 2:02d}"]
    paths["frame_last"] = paths[f"sample_{len(indices) - 1:02d}"]
    return paths


def load_pipe(cfg: dict[str, Any]) -> WanPipeline:
    model_id = os.environ.get("WAN_MODEL_PATH", cfg["model"]["model_id"])
    local_only = bool(cfg["model"].get("local_files_only", False)) or os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype_from_name(cfg["model"].get("torch_dtype", "bfloat16")),
        local_files_only=local_only,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def sbe_variant_cfg(cfg: dict[str, Any], variant_name: str) -> dict[str, Any]:
    merged = dict(cfg["sbe_online_continuous"])
    merged.update(cfg.get("sbe_online_variants", {}).get(variant_name, {}))
    return merged


def action_for_method(method: str, case: dict[str, str], cfg: dict[str, Any]) -> tuple[int, float | list[float | None] | None, dict[str, Any]]:
    steps = int(cfg["generation"]["steps"])
    if method == "baseline_12step":
        return steps, None, {"scheduler": "baseline"}
    if method.startswith("teacache_12step_t"):
        return steps, float(method.split("_t", 1)[1]), {"scheduler": "fixed_teacache"}
    if method == "uniform_teacache_t0.3":
        return steps, 0.30, {"scheduler": "uniform_teacache"}
    if method == "sbe_online_continuous_full":
        details = continuous_threshold_schedule(case["prompt"], steps, sbe_variant_cfg(cfg, "full"), use_uncertainty=True)
        return steps, details["threshold_schedule"], details
    if method == "sbe_online_continuous_no_u":
        details = continuous_threshold_schedule(case["prompt"], steps, sbe_variant_cfg(cfg, "no_u"), use_uncertainty=False)
        return steps, details["threshold_schedule"], details
    if method == "sbe_online_continuous_rule_only":
        details = continuous_threshold_schedule(
            case["prompt"],
            steps,
            sbe_variant_cfg(cfg, "rule_only"),
            use_uncertainty=True,
            force_rule_only=True,
        )
        return steps, details["threshold_schedule"], details
    if method == "sbe_online_continuous_fast":
        details = continuous_threshold_schedule(case["prompt"], steps, sbe_variant_cfg(cfg, "fast"), use_uncertainty=True)
        return steps, details["threshold_schedule"], details
    if method == "sbe_online_continuous_quality":
        details = continuous_threshold_schedule(case["prompt"], steps, sbe_variant_cfg(cfg, "quality"), use_uncertainty=True)
        return steps, details["threshold_schedule"], details
    if method == "sbe_online_discrete":
        details = discrete_threshold_schedule(case["prompt"], steps, sbe_variant_cfg(cfg, "discrete"))
        return steps, details["threshold_schedule"], details
    if method == "sbe_riskgate_v5":
        policy = cfg["sbe_riskgate_v5"]["policy"][case["block_type"]]
        if policy["action"] == "nocache":
            return int(policy.get("steps", steps)), None, {"scheduler": "legacy_type_policy", "legacy_block_type": case["block_type"]}
        return int(policy.get("steps", steps)), float(policy["threshold"]), {
            "scheduler": "legacy_type_policy",
            "legacy_block_type": case["block_type"],
        }
    raise ValueError(f"Unsupported TeaCache/SBE method: {method}")


def run_pipe(
    pipe: WanPipeline,
    original_forward: Any,
    out_dir: Path,
    case: dict[str, str],
    seed_idx: int,
    variant: str,
    steps: int,
    threshold: float | list[float | None] | None,
    scheduler_details: dict[str, Any] | None,
    height: int,
    width: int,
    num_frames: int,
    guidance_scale: float,
    fps: int,
    sample_count: int,
    save_videos: bool,
) -> dict[str, Any]:
    if threshold is None:
        pipe.transformer.forward = original_forward
        disable_teacache(pipe.transformer)
    else:
        install_teacache_forward(pipe.transformer)
        reset_teacache_state(pipe.transformer, steps=steps, threshold=threshold)

    video_id = f"{case['case_id']}_seed{seed_idx}_{variant}"
    generator = torch.Generator(device="cuda").manual_seed(stable_seed(case["case_id"], seed_idx))
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = pipe(
        prompt=case["prompt"],
        num_inference_steps=steps,
        height=height,
        width=width,
        num_frames=num_frames,
        generator=generator,
        guidance_scale=guidance_scale,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    frames = result.frames[0]

    video_path = ""
    if save_videos:
        video_dir = out_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(video_dir / f"{video_id}.mp4")
        export_to_video(frames, video_path, fps=fps)

    row = {
        "case_id": case["case_id"],
        "split": case["split"],
        "block_type": case["block_type"],
        "seed_idx": seed_idx,
        "variant": variant,
        "steps": steps,
        "threshold": "" if threshold is None else threshold,
        "threshold_schedule": "" if not isinstance(threshold, list) else json.dumps(threshold, ensure_ascii=False),
        "online_risk": "" if not scheduler_details else scheduler_details.get("risk", ""),
        "online_q": "" if not scheduler_details else scheduler_details.get("q", ""),
        "online_uncertainty": ""
        if not scheduler_details
        else scheduler_details.get("parsed", {}).get("uncertainty", ""),
        "online_features": "" if not scheduler_details else scheduler_details.get("features_json", ""),
        "online_parser_source": ""
        if not scheduler_details
        else scheduler_details.get("parsed", {}).get("parser_source", ""),
        "online_risk_level": "" if not scheduler_details else scheduler_details.get("risk_level", ""),
        "elapsed_seconds": round(elapsed, 4),
        "peak_memory_gb": round(peak_gb, 3),
        "prompt": case["prompt"],
        "target": case["target"],
        "negative": case["negative"],
        "video_path": video_path,
    }
    row.update(save_sample_frames(frames, out_dir, video_id, sample_count))
    row.update(summarize_teacache(pipe.transformer))
    return row


def install_timestep_tracker(pipe: WanPipeline) -> None:
    original_forward = pipe.transformer.forward

    def tracked_forward(self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) >= 2:
            timestep = args[1]
        if timestep is not None:
            try:
                self._sbe_current_timestep = int(timestep.flatten()[0].detach().cpu().item())
            except Exception:
                self._sbe_current_timestep = -1
        return original_forward(*args, **kwargs)

    pipe.transformer._sbe_current_timestep = -1
    pipe.transformer.forward = types.MethodType(tracked_forward, pipe.transformer)


def count_diffusers_hooks(module: torch.nn.Module) -> int:
    total = 0
    for submodule in module.modules():
        registry = getattr(submodule, "_diffusers_hook", None)
        hooks = getattr(registry, "hooks", None)
        if hooks:
            total += len(hooks)
    return total


def configure_builtin(pipe: WanPipeline, variant: str) -> dict[str, Any]:
    if variant == "fastercache_s2":
        if FasterCacheConfig is None or apply_faster_cache is None:
            raise RuntimeError("diffusers FasterCache hooks are unavailable in this environment.")
        install_timestep_tracker(pipe)
        callback = lambda: int(getattr(pipe.transformer, "_sbe_current_timestep", -1))
        config = FasterCacheConfig(
            spatial_attention_block_skip_range=2,
            spatial_attention_timestep_skip_range=(-1, 681),
            attention_weight_callback=lambda _module: 0.5,
            tensor_format="BCFHW",
            is_guidance_distilled=True,
            current_timestep_callback=callback,
        )
        apply_faster_cache(pipe.transformer, config)
        return {"hook_count": count_diffusers_hooks(pipe.transformer), "hook_config": repr(config)}

    if variant == "pab_s2_c3":
        if PyramidAttentionBroadcastConfig is None or apply_pyramid_attention_broadcast is None:
            raise RuntimeError("diffusers PAB hooks are unavailable in this environment.")
        install_timestep_tracker(pipe)
        callback = lambda: int(getattr(pipe.transformer, "_sbe_current_timestep", -1))
        config = PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=2,
            cross_attention_block_skip_range=3,
            spatial_attention_timestep_skip_range=(0, 1000),
            cross_attention_timestep_skip_range=(0, 1000),
            current_timestep_callback=callback,
        )
        apply_pyramid_attention_broadcast(pipe.transformer, config)
        return {"hook_count": count_diffusers_hooks(pipe.transformer), "hook_config": repr(config)}

    raise ValueError(f"Unknown built-in method: {variant}")


def img_array(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((256, 144))
    return np.asarray(image).astype(np.float32) / 255.0


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    mux, muy = a.mean(), b.mean()
    vx, vy = a.var(), b.var()
    cov = ((a - mux) * (b - muy)).mean()
    numerator = (2 * mux * muy + c1) * (2 * cov + c2)
    denominator = (mux * mux + muy * muy + c1) * (vx + vy + c2) + 1e-8
    return float(numerator / denominator)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def try_lpips_score(paths_a: list[str], paths_b: list[str]) -> tuple[float | None, str]:
    if importlib.util.find_spec("lpips") is None:
        return None, "lpips_not_installed"
    try:
        import lpips
        import torchvision.transforms.functional as TF

        loss_fn = lpips.LPIPS(net="alex").to("cuda")
        values = []
        with torch.no_grad():
            for path_a, path_b in zip(paths_a, paths_b):
                image_a = TF.to_tensor(Image.open(path_a).convert("RGB")).unsqueeze(0).to("cuda") * 2 - 1
                image_b = TF.to_tensor(Image.open(path_b).convert("RGB")).unsqueeze(0).to("cuda") * 2 - 1
                values.append(float(loss_fn(image_a, image_b).detach().cpu().item()))
        return float(np.mean(values)), "lpips_alex"
    except Exception as exc:
        return None, f"lpips_error:{exc!r}"


def compute_metrics(rows: list[dict[str, Any]], cfg: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    baseline = df[df["variant"] == "baseline_12step"].set_index(["case_id", "seed_idx"])
    sample_cols = sorted([column for column in df.columns if column.startswith("sample_")])
    metric_rows = []

    for row in df.to_dict("records"):
        key = (row["case_id"], row["seed_idx"])
        if key not in baseline.index:
            continue
        base_row = baseline.loc[key]
        ssims, l1s, psnrs = [], [], []
        row_paths, base_paths = [], []
        for col in sample_cols:
            current = img_array(row[col])
            reference = img_array(base_row[col])
            ssims.append(ssim_simple(current, reference))
            l1s.append(float(np.abs(current - reference).mean()))
            psnrs.append(psnr(current, reference))
            row_paths.append(row[col])
            base_paths.append(base_row[col])

        lpips_value, lpips_source = None, "disabled"
        if cfg["evaluation"].get("try_lpips", True) and row["variant"] != "baseline_12step":
            lpips_value, lpips_source = try_lpips_score(row_paths, base_paths)

        ssim_value = float(np.mean(ssims))
        l1_value = float(np.mean(l1s))
        row["SSIM"] = round(ssim_value, 4)
        row["PSNR"] = round(float(np.mean(psnrs)), 2)
        row["LPIPS"] = "" if lpips_value is None else round(float(lpips_value), 4)
        row["LPIPS_source"] = lpips_source
        row["LPIPS_L1_proxy"] = round(l1_value, 4)
        proxy = max(0.0, min(100.0, 100.0 * (0.65 * ssim_value + 0.35 * (1.0 - l1_value))))
        row["VBench_proxy"] = round(float(proxy), 2)
        metric_rows.append(row)

    result = pd.DataFrame(metric_rows)
    result.to_csv(out_dir / "per_video_rows.csv", index=False)
    return result


def official_vbench_available() -> bool:
    return importlib.util.find_spec("vbench") is not None


def load_official_vbench_scores(path_text: str) -> dict[str, float]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores: dict[str, float] = {}

    if isinstance(payload, dict) and isinstance(payload.get("scores"), dict):
        for method, method_scores in payload["scores"].items():
            if not isinstance(method_scores, dict):
                continue
            for key, value in method_scores.items():
                if isinstance(value, (int, float)):
                    scores[f"{method}/{key}"] = float(value)
            if isinstance(method_scores.get("vbench4_avg"), (int, float)):
                scores[method] = float(method_scores["vbench4_avg"])

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_text = str(key)
                joined = f"{prefix}/{key_text}" if prefix else key_text
                if isinstance(value, (int, float)):
                    scores[key_text] = float(value)
                    scores[joined] = float(value)
                else:
                    walk(value, joined)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, prefix)

    walk(payload)
    return scores


def method_label(method: str) -> str:
    labels = {
        "baseline_12step": "Wan no-cache 12-step",
        "teacache_12step_t0.2": "TeaCache t=0.2",
        "teacache_12step_t0.3": "TeaCache t=0.3",
        "teacache_12step_t0.45": "TeaCache t=0.45",
        "fastercache_s2": "FasterCache",
        "pab_s2_c3": "PAB",
        "sbe_riskgate_v5": "SBE-RiskGate v5",
        "sbe_online_continuous_full": "SBE-online continuous full",
        "sbe_online_continuous_no_u": "SBE-online continuous no-U",
        "sbe_online_continuous_rule_only": "SBE-online rule-only",
        "sbe_online_continuous_fast": "SBE-online continuous fast",
        "sbe_online_continuous_quality": "SBE-online continuous quality",
        "sbe_online_discrete": "SBE-online discrete",
        "uniform_teacache_t0.3": "Uniform TeaCache t=0.3",
    }
    return labels.get(method, method)


def method_vbench_score(method: str, group: pd.DataFrame, official_scores: dict[str, float]) -> float:
    label = method_label(method)
    keys = (
        method,
        label,
        f"{method}/total_score",
        f"{label}/total_score",
        f"{method}/vbench4_avg",
        f"{label}/vbench4_avg",
        f"{method}/VBench",
        f"{label}/VBench",
    )
    for key in keys:
        if key in official_scores:
            return round(float(official_scores[key]), 2)
    return round(float(group["VBench_proxy"].mean()), 2)


def method_official_dim(method: str, label: str, official_scores: dict[str, float], key: str) -> str:
    for candidate in (f"{method}/{key}", f"{label}/{key}"):
        if candidate in official_scores:
            return f"{float(official_scores[candidate]):.2f}"
    return ""


def method_flops_proxy(method: str, group: pd.DataFrame, latency: float, base_latency: float) -> float:
    if method == "baseline_12step":
        return 1.0
    computed = pd.to_numeric(group.get("teacache_computed", pd.Series(dtype=float)), errors="coerce").fillna(0)
    skipped = pd.to_numeric(group.get("teacache_skipped", pd.Series(dtype=float)), errors="coerce").fillna(0)
    total = computed + skipped
    if float(total.sum()) > 0:
        # TeaCache skips transformer block passes. This estimates forward-count
        # cost, not hardware profiler FLOPs.
        return round(float((computed / total.clip(lower=1)).mean()), 4)
    return round(float(latency / base_latency), 4)


def build_summary(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    out_dir: Path,
    vbench_source: str,
    official_scores: dict[str, float],
) -> pd.DataFrame:
    base_latency = float(df[df["variant"] == "baseline_12step"]["elapsed_seconds"].mean())
    rows = []
    for method in cfg["report"]["table_methods"]:
        group = df[df["variant"] == method]
        if group.empty:
            continue
        latency = float(group["elapsed_seconds"].mean())
        label = method_label(method)
        lpips_values = pd.to_numeric(group["LPIPS"], errors="coerce")
        lpips_mean = float(lpips_values.mean()) if lpips_values.notna().any() else float(group["LPIPS_L1_proxy"].mean())
        rows.append(
            {
                "Method": label,
                "variant": method,
                "n": int(len(group)),
                "FLOPs_proxy_down": method_flops_proxy(method, group, latency, base_latency),
                "Speedup_up": round(base_latency / latency, 4),
                "Latency_down": f"{latency:.4f}s",
                "VBench_up": method_vbench_score(method, group, official_scores),
                "VBench_source": vbench_source,
                "VBench4_imaging_up": method_official_dim(method, label, official_scores, "imaging_quality"),
                "VBench4_temporal_up": method_official_dim(method, label, official_scores, "temporal_flickering"),
                "VBench4_motion_up": method_official_dim(method, label, official_scores, "motion_smoothness"),
                "VBench4_dynamic_up": method_official_dim(method, label, official_scores, "dynamic_degree"),
                "LPIPS_L1_down": round(float(group["LPIPS_L1_proxy"].mean()), 4),
                "LPIPS_down": "" if math.isnan(lpips_mean) else round(lpips_mean, 4),
                "SSIM_up": round(float(group["SSIM"].mean()), 4),
                "PSNR_up": round(float(group["PSNR"].mean()), 2),
                "PeakMemGB": round(float(group["peak_memory_gb"].mean()), 2),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "main_table.csv", index=False)
    with (out_dir / "main_table.md").open("w", encoding="utf-8") as handle:
        handle.write(summary.to_markdown(index=False))
        handle.write("\n")
    return summary


def make_contact_sheet(df: pd.DataFrame, out_dir: Path, max_cases: int = 40) -> None:
    methods = list(df["variant"].drop_duplicates())
    cases = list(df[df["variant"] == "baseline_12step"]["case_id"].drop_duplicates())[:max_cases]
    width, height, label_h = 180, 104, 34
    sheet = Image.new("RGB", (len(methods) * width, len(cases) * (height + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for method_idx, method in enumerate(methods):
        draw.text((method_idx * width + 4, 2), method_label(method)[:24], fill=(0, 0, 0))
    for case_idx, case_id in enumerate(cases):
        for method_idx, method in enumerate(methods):
            subset = df[(df["case_id"] == case_id) & (df["variant"] == method)]
            if subset.empty:
                continue
            row = subset.iloc[0]
            image = Image.open(row["frame_middle"]).convert("RGB").resize((width, height))
            x = method_idx * width
            y = case_idx * (height + label_h)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + height + 3), f"{case_idx:02d} {case_id}"[:30], fill=(0, 0, 0))
    sheet.save(out_dir / "contact_sheet_middle.jpg", quality=92)


def write_report(summary: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> None:
    table_rows = "\n".join(
        " & ".join(
            [
                str(row["Method"]),
                f"{float(row['FLOPs_proxy_down']):.4f}",
                f"{float(row['Speedup_up']):.4f}",
                str(row["Latency_down"]),
                f"{float(row['VBench_up']):.2f}",
                str(row.get("VBench4_imaging_up", "")),
                str(row.get("VBench4_temporal_up", "")),
                str(row.get("VBench4_motion_up", "")),
                str(row.get("VBench4_dynamic_up", "")),
                f"{float(row['LPIPS_L1_down']):.4f}",
                f"{float(row['SSIM_up']):.4f}",
                f"{float(row['PSNR_up']):.2f}",
            ]
        )
        + r" \\"
        for _, row in summary.iterrows()
    )
    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage{{geometry,booktabs,adjustbox,graphicx,array}}
\geometry{{margin=1.2cm,landscape}}
\title{{{cfg['report'].get('title','H200 SBE Main Table')}}}
\date{{\today}}
\begin{{document}}
\maketitle
\section{{Main Table}}
The VBench column uses an official VBench JSON only when it is supplied through
evaluation.official\_vbench\_json or OFFICIAL\_VBENCH\_JSON. Otherwise it is a
clearly marked VBench proxy. Check main\_table.csv and eval\_status.json.
\begin{{center}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{lrrrrrrrrrrr}}
\toprule
Method & FLOPs $\downarrow$ & Speedup $\uparrow$ & Latency $\downarrow$ & VBench-4D $\uparrow$ & Imaging $\uparrow$ & Temporal $\uparrow$ & Motion $\uparrow$ & Dynamic $\uparrow$ & LPIPS-L1 $\downarrow$ & SSIM $\uparrow$ & PSNR $\uparrow$ \\
\midrule
{table_rows}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{center}}
\section{{Contact Sheet}}
\includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{{contact_sheet_middle.jpg}}
\end{{document}}
"""
    (out_dir / "report.tex").write_text(tex, encoding="utf-8")
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=out_dir, check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=out_dir, check=True)
    except Exception as exc:
        (out_dir / "report_compile_error.txt").write_text(repr(exc), encoding="utf-8")


def run_teacache_and_sbe_methods(
    cfg: dict[str, Any],
    cases: list[dict[str, str]],
    deadline: float,
    out_dir: Path,
) -> list[dict[str, Any]]:
    methods = [
        method
        for method in cfg["methods"]
        if method.startswith("teacache_") or method in {"baseline_12step", "sbe_riskgate_v5", "uniform_teacache_t0.3"}
        or method.startswith("sbe_online_")
    ]
    if not methods:
        return []
    log(f"Loading Wan pipeline for methods: {methods}")
    pipe = load_pipe(cfg)
    original_forward = pipe.transformer.forward
    gen = cfg["generation"]
    seed_idx = int(cfg["experiment"].get("seed_idx", 0))
    rows: list[dict[str, Any]] = []
    for case in cases:
        for method in methods:
            if time.time() > deadline:
                log("Time budget reached during TeaCache/SBE phase.")
                break
            steps, threshold, scheduler_details = action_for_method(method, case, cfg)
            detail_text = ""
            if scheduler_details and "risk" in scheduler_details:
                detail_text = f" risk={scheduler_details.get('risk')} q={scheduler_details.get('q')}"
            log(f"Generate {case['case_id']} {method} steps={steps} threshold={threshold}{detail_text}")
            rows.append(
                run_pipe(
                    pipe,
                    original_forward,
                    out_dir,
                    case,
                    seed_idx,
                    method,
                    steps,
                    threshold,
                    scheduler_details,
                    int(gen["height"]),
                    int(gen["width"]),
                    int(gen["num_frames"]),
                    float(gen["guidance_scale"]),
                    int(gen["fps"]),
                    int(cfg["experiment"]["sample_frame_count"]),
                    bool(cfg["experiment"]["save_videos"]),
                )
            )
            pd.DataFrame(rows).to_csv(out_dir / "generation_rows_partial.csv", index=False)
        if time.time() > deadline:
            break
    del pipe
    torch.cuda.empty_cache()
    return rows


def run_builtin_methods(
    cfg: dict[str, Any],
    cases: list[dict[str, str]],
    deadline: float,
    out_dir: Path,
) -> list[dict[str, Any]]:
    methods = [method for method in cfg["methods"] if method in {"fastercache_s2", "pab_s2_c3"}]
    if cfg.get("optional_methods", {}).get("include_pab", False) and "pab_s2_c3" not in methods:
        methods.append("pab_s2_c3")
    rows: list[dict[str, Any]] = []
    gen = cfg["generation"]
    seed_idx = int(cfg["experiment"].get("seed_idx", 0))
    for method in methods:
        if time.time() > deadline:
            break
        log(f"Loading Wan pipeline for {method}")
        pipe = load_pipe(cfg)
        hook_info = configure_builtin(pipe, method)
        original_forward = pipe.transformer.forward
        max_cases = int(cfg.get("optional_methods", {}).get("pab_max_prompts", len(cases))) if method == "pab_s2_c3" else len(cases)
        for case in cases[:max_cases]:
            if time.time() > deadline:
                log(f"Time budget reached during {method}.")
                break
            log(f"Generate {case['case_id']} {method}")
            row = run_pipe(
                pipe,
                original_forward,
                out_dir,
                case,
                seed_idx,
                method,
                int(gen["steps"]),
                None,
                {"scheduler": "diffusers_builtin_hook"},
                int(gen["height"]),
                int(gen["width"]),
                int(gen["num_frames"]),
                float(gen["guidance_scale"]),
                int(gen["fps"]),
                int(cfg["experiment"]["sample_frame_count"]),
                bool(cfg["experiment"]["save_videos"]),
            )
            row.update(hook_info)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / f"generation_rows_partial_{method}.csv", index=False)
        del pipe
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-official-vbench", action="store_true")
    parser.add_argument("--allow-vbench-proxy", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    official_json = os.environ.get("OFFICIAL_VBENCH_JSON", "") or str(cfg["evaluation"].get("official_vbench_json", ""))
    official_scores = load_official_vbench_scores(official_json)
    vbench_source = "official_vbench_json" if official_scores else "proxy"
    if args.require_official_vbench and not official_scores:
        raise SystemExit(
            "Official VBench was required, but no official score JSON was provided. "
            "Set OFFICIAL_VBENCH_JSON=/path/to/vbench_result.json or rerun with --allow-vbench-proxy."
        )
    if vbench_source == "proxy" and not args.allow_vbench_proxy:
        raise SystemExit("No official VBench JSON found. Use --allow-vbench-proxy to write clearly marked proxy scores.")

    cases = select_cases(cfg["experiment"]["split"], int(cfg["experiment"]["max_prompts"]))
    (args.out_dir / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    deadline = time.time() + float(cfg["experiment"]["time_budget_hours"]) * 3600

    rows = []
    rows.extend(run_teacache_and_sbe_methods(cfg, cases, deadline, args.out_dir))
    rows.extend(run_builtin_methods(cfg, cases, deadline, args.out_dir))
    if not rows:
        raise SystemExit("No videos were generated.")

    pd.DataFrame(rows).to_csv(args.out_dir / "generation_rows.csv", index=False)
    metric_df = compute_metrics(rows, cfg, args.out_dir)
    summary = build_summary(metric_df, cfg, args.out_dir, vbench_source, official_scores)
    make_contact_sheet(metric_df, args.out_dir)
    write_report(summary, args.out_dir, cfg)

    status = {
        "vbench_source": vbench_source,
        "official_vbench_module_installed": official_vbench_available(),
        "official_vbench_json": official_json,
        "num_rows": int(len(metric_df)),
        "num_cases": int(metric_df["case_id"].nunique()),
        "methods_completed": sorted(metric_df["variant"].unique().tolist()),
        "time_budget_hours": cfg["experiment"]["time_budget_hours"],
        "note": "VBench_up is proxy unless vbench_source == official_vbench_json.",
    }
    (args.out_dir / "eval_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
