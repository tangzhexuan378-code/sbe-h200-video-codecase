from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess

status = {
    "torch": importlib.util.find_spec("torch") is not None,
    "diffusers": importlib.util.find_spec("diffusers") is not None,
    "transformers": importlib.util.find_spec("transformers") is not None,
    "lpips": importlib.util.find_spec("lpips") is not None,
    "spacy": importlib.util.find_spec("spacy") is not None,
    "vbench": importlib.util.find_spec("vbench") is not None,
    "vbench_command": shutil.which("vbench"),
    "ffmpeg_binary": shutil.which("ffmpeg"),
}

try:
    import spacy

    try:
        spacy.load("en_core_web_sm")
        status["spacy_model"] = "en_core_web_sm"
    except Exception:
        status["spacy_model"] = "fallback:spacy.blank(en)+rules"
except Exception as exc:
    status["spacy_error"] = repr(exc)

try:
    import imageio_ffmpeg

    status["imageio_ffmpeg_binary"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception as exc:
    status["imageio_ffmpeg_error"] = repr(exc)

try:
    import torch

    status["cuda_available"] = torch.cuda.is_available()
    status["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    status["bf16_supported"] = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None
    status["torch_version"] = torch.__version__
    status["torch_cuda"] = torch.version.cuda
except Exception as exc:
    status["torch_error"] = repr(exc)

try:
    output = subprocess.check_output(["vbench", "--help"], text=True, stderr=subprocess.STDOUT, timeout=20)
    status["vbench_help_ok"] = "evaluate" in output
except Exception as exc:
    status["vbench_help_error"] = repr(exc)

print(json.dumps(status, indent=2, ensure_ascii=False))
