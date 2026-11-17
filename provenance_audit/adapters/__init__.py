"""Backend adapters. A backend needs only:

    encode(text) -> list[int]
    first_token(text) -> int
    capture(ids) -> ndarray (n_layers, hidden)   # last token
    generate(prompt, n) -> (ids, text)
    top_prob(ids) -> float
    name: str

Anything with that shape works, including the Tiny 2.1 adapter.
"""
from .hf import HFBackend, find_text_layers  # noqa: F401
from .nanogpt import NanoGPTBackend  # noqa: F401

__all__ = ["HFBackend", "NanoGPTBackend", "find_text_layers", "registry"]


def registry() -> dict:
    return {"hf": HFBackend, "nanogpt": NanoGPTBackend}
