from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DIMS = ["imaging_quality", "temporal_flickering", "motion_smoothness", "dynamic_degree"]


def numeric_from_obj(obj: Any) -> float | None:
    if isinstance(obj, (int, float)):
        value = float(obj)
        # VBench raw scores are often in [0,1]. Convert to percent-like scores.
        return value * 100.0 if 0.0 <= value <= 1.5 else value
    if isinstance(obj, list):
        for item in obj:
            value = numeric_from_obj(item)
            if value is not None:
                return value
    if isinstance(obj, dict):
        for key in ("score", "mean", "value", "video_results"):
            if key in obj:
                value = numeric_from_obj(obj[key])
                if value is not None:
                    return value
    return None


def find_dim_score(obj: Any, dim: str) -> float | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) == dim:
                found = numeric_from_obj(value)
                if found is not None:
                    return found
            found = find_dim_score(value, dim)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_dim_score(item, dim)
            if found is not None:
                return found
    return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def method_scores(work_dir: Path, method: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    json_paths = sorted(work_dir.glob(f"{method}_*/results_*_eval_results.json"))
    for path in json_paths:
        payload = load_json(path)
        for dim in DIMS:
            if dim in scores:
                continue
            value = find_dim_score(payload, dim)
            if value is not None:
                scores[dim] = round(float(value), 2)
    if scores:
        values = [scores[dim] for dim in DIMS if dim in scores]
        if values:
            scores["vbench4_avg"] = round(sum(values) / len(values), 2)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    scores = {}
    for method_dir in sorted((args.work_dir / "by_method").glob("*")):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        method_score = method_scores(args.work_dir, method)
        if method_score:
            scores[method] = method_score

    payload = {
        "source": "official_vbench4_custom_input",
        "note": "Scores are official VBench custom-input dimensions, multiplied by 100 when raw output is in [0,1].",
        "dimensions": DIMS,
        "scores": scores,
    }
    out_json = args.run_dir / "official_vbench4_scores.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_lines = ["method,vbench4_avg,imaging_quality,temporal_flickering,motion_smoothness,dynamic_degree"]
    for method, values in scores.items():
        csv_lines.append(
            ",".join(
                [
                    method,
                    str(values.get("vbench4_avg", "")),
                    str(values.get("imaging_quality", "")),
                    str(values.get("temporal_flickering", "")),
                    str(values.get("motion_smoothness", "")),
                    str(values.get("dynamic_degree", "")),
                ]
            )
        )
    (args.run_dir / "official_vbench4_scores.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
