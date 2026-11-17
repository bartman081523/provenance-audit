"""HuggingFace backend: works for any causal LM with a `layers` list."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is only needed at runtime
    import torch
except Exception:  # pragma: no cover
    torch = None


def find_text_layers(model: Any):
    """Locate the decoder layer list (Gemma 3/4, Llama, GPT-2, ...)."""
    best = None
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) < 4:
            continue
        child = mod[0]
        if not (hasattr(child, "self_attn") or hasattr(child, "attn")
                or hasattr(child, "attention")):
            continue
        score = len(mod) + (100 if ("language_model" in name
                                    or "text_model" in name) else 0)
        if best is None or score > best[0]:
            best = (score, mod)
    if best is None:
        raise RuntimeError("find_text_layers: no decoder layer list found.")
    return best[1]


class HFBackend:
    """Minimal read-only adapter: encode, capture, generate, top_prob."""

    def __init__(self, model_id: str, device: str = "cpu", local: bool = False,
                 dtype: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = {"local_files_only": True} if local else {}
        dt = getattr(torch, dtype) if dtype else (
            torch.bfloat16 if device == "cuda" else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dt, **kw).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_id, **kw)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = device
        self.layers = find_text_layers(self.model)
        self.name = model_id

    # -- encode -----------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        ids = [int(i) for i in self.tok(text, add_special_tokens=False)["input_ids"]]
        if self.tok.bos_token_id is not None:
            ids = [int(self.tok.bos_token_id)] + ids
        return ids

    def first_token(self, text: str) -> int:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"first_token: {text!r} is empty.")
        return int(ids[0])

    # -- capture ----------------------------------------------------------
    def capture(self, ids: Sequence[int]) -> np.ndarray:
        store: dict = {}

        def mk(i):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                store[i] = h[0, -1].detach().to("cpu", torch.float32).numpy()
            return hook

        handles = [self.layers[i].register_forward_hook(mk(i))
                   for i in range(len(self.layers))]
        try:
            t = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
            with torch.no_grad():
                self.model(t)
        finally:
            for h in handles:
                h.remove()
        if len(store) != len(self.layers):
            raise RuntimeError("capture: missing layer hooks.")
        return np.stack([store[i] for i in range(len(self.layers))])

    # -- generate ---------------------------------------------------------
    def generate(self, prompt: str, n_tokens: int) -> Tuple[List[int], str]:
        ids = self.encode(prompt)
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                t, max_new_tokens=n_tokens, do_sample=False, use_cache=True,
                pad_token_id=self.tok.eos_token_id)
        gen = [int(x) for x in out[0, len(ids):].tolist()]
        return gen, self.tok.decode(gen)

    def top_prob(self, ids: Sequence[int]) -> float:
        t = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(t).logits[0, -1].float()
        return float(torch.softmax(logits, -1).max())
