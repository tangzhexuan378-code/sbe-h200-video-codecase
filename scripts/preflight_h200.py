from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path


def command_output(cmd: list[str], timeout: int = 20) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def url_ok(url: str, timeout: int = 12) -> dict[str, object]:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "sbe-h200-preflight"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": 200 <= int(resp.status) < 500, "status": int(resp.status)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def tcp_ok(host: str, port: int = 443, timeout: int = 8) -> dict[str, object]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-gb", type=float, default=180.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    disk = shutil.disk_usage(Path.cwd())
    free_gb = disk.free / (1024**3)
    status: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "free_disk_gb": round(free_gb, 2),
        "min_free_disk_gb": args.min_free_gb,
        "disk_ok": free_gb >= args.min_free_gb,
        "git": command_output(["git", "--version"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "network": {
            "github_tcp": tcp_ok("github.com"),
            "huggingface_tcp": tcp_ok("huggingface.co"),
            "pypi_tcp": tcp_ok("pypi.org"),
            "github_head": url_ok("https://github.com"),
            "huggingface_head": url_ok("https://huggingface.co"),
            "pypi_head": url_ok("https://pypi.org"),
        },
        "env": {
            "WAN_MODEL_PATH": os.environ.get("WAN_MODEL_PATH", ""),
            "LOCAL_FILES_ONLY": os.environ.get("LOCAL_FILES_ONLY", ""),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
            "HF_HOME": os.environ.get("HF_HOME", ""),
        },
    }

    hard_fail = []
    if not status["disk_ok"]:
        hard_fail.append(f"free disk {free_gb:.1f} GB < required {args.min_free_gb:.1f} GB")
    if not status["git"].get("ok"):
        hard_fail.append("git is not available")
    if not status["nvidia_smi"].get("ok"):
        hard_fail.append("nvidia-smi is not available; GPU/driver may be missing")

    status["hard_failures"] = hard_fail
    text = json.dumps(status, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if hard_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
