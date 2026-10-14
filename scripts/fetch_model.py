#!/usr/bin/env python3
"""Download the sentence encoder into the image at build time.

The application refuses to reach the network at runtime (`HF_HUB_OFFLINE`), so
the weights have to be present before the first request. Fetching them during
the build also means a deploy either succeeds with a working model or fails
loudly at build time, rather than starting and then erroring on the first
transaction a user clicks.

Run:
    HF_HOME=/opt/hf python scripts/fetch_model.py
"""
from __future__ import annotations

import os
import sys

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    # the download itself needs the network, so the offline flags this project
    # sets elsewhere must not apply here
    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(flag, None)
    os.environ.setdefault("HF_HOME", "/opt/hf")

    from huggingface_hub import snapshot_download
    path = snapshot_download(
        MODEL_ID,
        # the PyTorch weights and the tokeniser, not the ONNX, OpenVINO,
        # TensorFlow or Rust copies the repository also carries
        allow_patterns=["*.json", "*.txt", "pytorch_model.bin", "*.safetensors",
                        "1_Pooling/*", "2_Normalize/*"],
        ignore_patterns=["onnx/*", "openvino/*", "*.h5", "*.ot", "*.msgpack"],
    )
    print(f"[PASS] {MODEL_ID} cached at {path}")
    total = sum(f.stat().st_size for f in __import__("pathlib").Path(path)
                .rglob("*") if f.is_file())
    print(f"       {total / 1024 / 1024:.0f} MB on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
