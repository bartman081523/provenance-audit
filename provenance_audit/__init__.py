"""provenance-audit: provenance axes and a hallucination indicator.

Three directions in the residual stream -- context, familiarity, own text --
fitted on minimal pairs and combined into a one-pass justification flag.
"""
from . import audit, axes, cells, core, indicator  # noqa: F401
from .axes import calibrate_all, export  # noqa: F401
from .indicator import flag  # noqa: F401

__version__ = "0.1.0"
__all__ = ["core", "cells", "axes", "audit", "indicator",
           "calibrate_all", "export", "flag"]
