from __future__ import annotations

import importlib.util
import json
import shutil

status = {
    "torch": importlib.util.find_spec("torch") is not None,
    "diffusers": importlib.util.find_spec("diffusers") is not None,
    "transformers": importlib.util.find_spec("transformers") is not None,
    "lpips": importlib.util.find_spec("lpips") is not None,
    "vbench": importlib.util.find_spec("vbench") is not None,
    "ffmpeg_binary": shutil.which("ffmpeg"),
}

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
except Exception as exc:
    status["torch_error"] = repr(exc)

print(json.dumps(status, indent=2, ensure_ascii=False))
