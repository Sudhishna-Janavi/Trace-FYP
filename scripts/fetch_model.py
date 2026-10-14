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
        # The ONNX graph and the tokeniser, and nothing else. The deployed
        # application runs this graph under onnxruntime rather than PyTorch:
        # the weights are the same 86 MB either way, but torch brings a runtime
        # that does not fit in a free instance alongside them.
        allow_patterns=["*.json", "*.txt", "onnx/model.onnx",
                        "1_Pooling/*", "2_Normalize/*"],
        ignore_patterns=["onnx/model_O*", "onnx/*qint8*", "onnx/*quint8*",
                         "openvino/*", "*.h5", "*.ot", "*.msgpack",
                         "*.bin", "*.safetensors"],
    )
    print(f"[PASS] {MODEL_ID} cached at {path}")
    total = sum(f.stat().st_size for f in __import__("pathlib").Path(path)
                .rglob("*") if f.is_file())
    print(f"       {total / 1024 / 1024:.0f} MB on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
