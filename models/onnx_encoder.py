"""The sentence encoder, run under ONNX Runtime instead of PyTorch.

Why
    The weights are the same 86 MB either way. PyTorch is what does not fit:
    with torch loaded the application measured 401 MB resident, against a
    512 MB limit on every free instance that does not demand a payment card.
    Inference here is a fixed sequence of matrix multiplications with no
    training, no autograd and no dynamic graph, so the framework that supports
    those things is paying rent it does not earn.

What it must reproduce exactly
    `sentence-transformers` applies three steps, named in the model's own
    `modules.json`: the transformer, mean pooling weighted by the attention
    mask, then L2 normalisation. `1_Pooling/config.json` confirms mean pooling
    rather than CLS, and `sentence_bert_config.json` gives a 256-token limit
    with no lower-casing. All three are reimplemented below.

    Reimplementing a model's inference is only safe if it is checked rather
    than assumed, so `scripts/verify_onnx.py` compares this against the
    PyTorch pipeline on the evaluation data and fails on any disagreement.
"""
from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

import numpy as np

MAX_SEQ_LENGTH = 256      # sentence_bert_config.json
EMBEDDING_DIM = 384


def _snapshot() -> Path:
    from models.minilm_embeddings import _snapshot_path
    return Path(_snapshot_path())


class OnnxEncoder:
    """A drop-in replacement for the SentenceTransformer this project used.

    Only `encode` is reproduced, with the keyword arguments the rest of the
    codebase passes, so no caller needs to know which engine is underneath.
    """

    def __init__(self, snapshot: Path | None = None) -> None:
        self.snapshot = snapshot or _snapshot()
        graph = self.snapshot / "onnx" / "model.onnx"
        if not graph.exists():
            raise RuntimeError(
                f"No ONNX graph at {graph}. Run scripts/fetch_model.py, which "
                f"downloads it alongside the tokeniser.")
        self._graph = graph

    @cached_property
    def _session(self):
        import onnxruntime as ort
        options = ort.SessionOptions()
        # One thread. A free instance has a fraction of a core, and the thread
        # pools cost memory that is better spent on the graph.
        options.intra_op_num_threads = int(os.getenv("ONNX_THREADS", "1"))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The arena allocator keeps every block it has ever needed, so a single
        # wide batch permanently raises the floor: measured 149 MB after load
        # and 312 MB after one inference. Disabling it trades a little speed
        # for memory that stays flat, which is the scarce resource here.
        options.enable_cpu_mem_arena = os.getenv("ONNX_ARENA", "0") == "1"
        return ort.InferenceSession(str(self._graph), options,
                                    providers=["CPUExecutionProvider"])

    @cached_property
    def _tokeniser(self):
        from tokenizers import Tokenizer
        tokeniser = Tokenizer.from_file(str(self.snapshot / "tokenizer.json"))
        tokeniser.enable_truncation(max_length=MAX_SEQ_LENGTH)
        tokeniser.enable_padding(pad_id=0, pad_token="[PAD]")
        return tokeniser

    def encode(self, sentences, batch_size: int = 32,
               convert_to_numpy: bool = True, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, **_ignored) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = list(sentences)
        if not sentences:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        out = [self._encode_batch(sentences[start:start + batch_size],
                                  normalize_embeddings)
               for start in range(0, len(sentences), batch_size)]
        return np.vstack(out)

    def _encode_batch(self, batch: list[str], normalise: bool) -> np.ndarray:
        encoded = self._tokeniser.encode_batch(batch)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        types = np.array([e.type_ids for e in encoded], dtype=np.int64)

        hidden, = self._session.run(
            ["last_hidden_state"],
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": types})

        # mean over real tokens only: padding must not drag the average towards
        # whatever the pad embedding happens to be
        weights = mask.astype(np.float32)[..., None]
        summed = (hidden * weights).sum(axis=1)
        counts = np.clip(weights.sum(axis=1), 1e-9, None)
        pooled = summed / counts

        if normalise:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, 1e-12, None)
        return pooled.astype(np.float32)


def load_encoder() -> OnnxEncoder:
    return OnnxEncoder()
