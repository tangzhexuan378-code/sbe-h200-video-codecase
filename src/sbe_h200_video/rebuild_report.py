from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .run_experiment import (
    build_summary,
    load_config,
    load_official_vbench_scores,
    make_contact_sheet,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--official-vbench-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows_path = args.out_dir / "per_video_rows.csv"
    if not rows_path.exists():
        raise SystemExit(f"Missing {rows_path}. Run the generation experiment first.")

    scores = load_official_vbench_scores(str(args.official_vbench_json or ""))
    source = "official_vbench_json" if scores else "proxy"
    df = pd.read_csv(rows_path)
    summary = build_summary(df, cfg, args.out_dir, source, scores)
    make_contact_sheet(df, args.out_dir)
    write_report(summary, args.out_dir, cfg)

    status_path = args.out_dir / "eval_status.json"
    status = {}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "vbench_source": source,
            "official_vbench_json": "" if args.official_vbench_json is None else str(args.official_vbench_json),
            "note": "Report rebuilt after generation.",
        }
    )
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
