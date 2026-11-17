"""nanoGPT backend (karpathy/nanoGPT): checkpoint + tiktoken gpt2."""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, List, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _load_gpt_module(nanogpt_dir: str):
    path = os.path.join(nanogpt_dir, "model.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"nanoGPT model.py not found at {path}")
    spec = importlib.util.spec_from_file_location("nanogpt_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nanogpt_model"] = mod
    spec.loader.exec_module(mod)
    return mod


class NanoGPTBackend:
    def __init__(self, ckpt: str, nanogpt_dir: str = "."):
        import tiktoken
        mod = _load_gpt_module(nanogpt_dir)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = mod.GPT(mod.GPTConfig(**ck["model_args"]))
        sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
        self.model.load_state_dict(sd)
        self.model.eval()
        self.enc = tiktoken.get_encoding("gpt2")
        self.layers = self.model.transformer.h
        self.name = f"nanogpt:{os.path.basename(ckpt)}"

    def encode(self, text: str) -> List[int]:
        return [int(t) for t in self.enc.encode(text, allowed_special="all")]

    def first_token(self, text: str) -> int:
        ids = self.encode(text)
        if not ids:
            raise ValueError(f"first_token: {text!r} is empty.")
        return ids[0]

    def capture(self, ids: Sequence[int]) -> np.ndarray:
        store: dict = {}

        def mk(i):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                store[i] = h[0, -1].detach().float().numpy()
            return hook

        handles = [self.layers[i].register_forward_hook(mk(i))
                   for i in range(len(self.layers))]
        try:
            with torch.no_grad():
                self.model(torch.tensor([list(ids)], dtype=torch.long))
        finally:
            for h in handles:
                h.remove()
        return np.stack([store[i] for i in range(len(self.layers))])

    def generate(self, prompt: str, n_tokens: int) -> Tuple[List[int], str]:
        ids = self.encode(prompt)
        out = list(ids)
        with torch.no_grad():
            for _ in range(n_tokens):
                logits, _ = self.model(torch.tensor([out], dtype=torch.long))
                out.append(int(torch.argmax(logits[0, -1])))
        gen = out[len(ids):]
        return gen, self.enc.decode(gen)

    def top_prob(self, ids: Sequence[int]) -> float:
        with torch.no_grad():
            logits, _ = self.model(torch.tensor([list(ids)], dtype=torch.long))
        return float(torch.softmax(logits[0, -1], -1).max())
