"""Command line: calibrate, flag, audit."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import audit as audit_mod
from . import axes, indicator
from .cells import CODES, FACTS, FAMILIAR, NOVEL

SEEDS = ("The harbour lights fade after midnight.",
         "A copper wire carries the signal faster than glass.",
         "The old clock in the hallway loses two minutes a day.",
         "Rain has been falling on the roof since noon.",
         "The garden gate stays open in summer.",
         "A single candle burns at the far end of the table.")


def make_backend(kind: str, args: argparse.Namespace):
    if kind == "hf":
        from .adapters.hf import HFBackend
        return HFBackend(args.model, args.device, args.local, args.dtype)
    if kind == "nanogpt":
        from .adapters.nanogpt import NanoGPTBackend
        return NanoGPTBackend(args.ckpt, args.nanogpt_dir)
    raise SystemExit(f"unknown backend {kind!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="provenance-audit")
    ap.add_argument("command", choices=["calibrate", "flag", "audit"])
    ap.add_argument("--backend", default="hf", choices=["hf", "nanogpt"])
    ap.add_argument("--model", default=None, help="HF model id or local path")
    ap.add_argument("--ckpt", default=None, help="nanoGPT ckpt.pt")
    ap.add_argument("--nanogpt-dir", default=".")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--calib", default="calibration.json")
    ap.add_argument("--text", default=None, help="text for flag")
    ap.add_argument("--entities", default=None,
                    help="comma-separated entities for audit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.command == "calibrate":
        b = make_backend(a.backend, a)
        calib = axes.calibrate_all(b, SEEDS)
        payload = axes.export(calib)
        out = a.out or a.calib
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(json.dumps({"backend": payload["backend"],
                          "verdict": payload["verdict"],
                          "all_pass": payload["all_pass"],
                          "axes": {k: {"layer": v["layer"],
                                       "acc": round(v["acc"], 3),
                                       "p": round(v["p"], 4)}
                                   for k, v in payload["axes"].items()}},
                         indent=1))
        print(f"calibration: {out}")
        return 0 if payload["all_pass"] else 1

    if a.command == "flag":
        if not a.text:
            raise SystemExit("flag needs --text")
        b = make_backend(a.backend, a)
        print(json.dumps(indicator.flag(b, indicator.load(a.calib), a.text),
                         indent=1))
        return 0

    # audit
    if not a.entities:
        raise SystemExit("audit needs --entities")
    b = make_backend(a.backend, a)
    vocab = [c for _, c in CODES]
    ents = [e.strip() for e in a.entities.split(",") if e.strip()]
    res = audit_mod.audit(b, ents, vocab)
    print(json.dumps({"n": res["n"], "n_asserted": res["n_asserted"],
                      "n_flagged": res["n_flagged"],
                      "flag_rate": round(res["flag_rate_on_assertions"], 3),
                      "mean_stability": round(res["mean_stability"], 3),
                      "rows": [{"entity": r["entity"], "claim": r["claim"],
                                "stability": round(r["stability"], 2),
                                "H": r["H"]} for r in res["rows"]]}, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print(f"audit: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
