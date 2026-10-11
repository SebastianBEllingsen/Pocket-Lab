"""cellcount: blood cell counting and classification.

The package is organised as one sub-package per pipeline stage plus shared code:

    cellcount.common      dataset loading, image selection, metrics (shared by all stages)
    cellcount.viz         generic figure/overlay rendering (knows nothing about any stage)
    cellcount.bg_removal  stage 1: background estimation, flattening, foreground mask

Each stage is runnable as ``python -m cellcount.<stage> --help``.
See docs/CONVENTIONS.md for the design rules.
"""
