"""Stage 1: background estimation, illumination flattening and foreground mask.

Public API: BgConfig, remove_background, BgResult. CLI: ``python -m cellcount.bg_removal --help``.
"""
from .config import BgConfig
from .pipeline import BgResult, remove_background

__all__ = ["BgConfig", "BgResult", "remove_background"]
