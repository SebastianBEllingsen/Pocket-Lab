"""Command line for the synthetic data generator.

Run:  python -m cellcount.synth [flags]        (the cut-out bank comes from  python -m cellcount.synth.bank)

Typical use
    python -m cellcount.synth --n 6 --show                 look at 6 scenes with their boxes (keys r/w/p/s toggle classes)
    python -m cellcount.synth --n 6 --save previews        write PNG pages instead of opening a window
    python -m cellcount.synth --n 2000 --out Data_Sets/synth_out --split train    write a YOLO dataset

Flags
    --bank DIR           cut-out bank (default <data root>/synth_bank)
    --data-root DIR      folder with Data_Set_1, ... (default $CELLCOUNT_DATA_ROOT or ./Data_Sets)
    --n N                number of scenes (default 4)
    --seed N             random seed (default 0); scene k uses its own seed so scenes are reproducible
    --out DIR            write a YOLO dataset here (images, labels, data.yaml, wbc_types.csv, manifest.csv)
    --split NAME         train/val/test folder to write into (default train; synthetic data belongs in train)
    --start N            first image number (continue an existing dataset)
    --show / --save DIR  look at the scenes in a window / save PNG pages
    --no-boxes           start with the boxes hidden
    one flag per scene parameter (see SceneConfig in scene.py), for example
    --tile --rbc-radius --coverage LO,HI --min-dist LO,HI --wbc-mean --platelet-mean --smudge-mean
    --wbc-types a,b,c --rim-prob --own-bg-weight --stain-strength LO,HI --noise-sigma LO,HI ...
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields

import numpy as np

from ..common.datasets import resolve_data_root
from .export import box_layers, write_dataset
from .scene import Bank, SceneConfig, generate_scene


def _add_config_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("scene parameters (defaults from SceneConfig)")
    d = SceneConfig()
    for f in fields(SceneConfig):
        v = getattr(d, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(v, tuple):
            g.add_argument(flag, default=",".join(str(x) for x in v), metavar="A,B", help=f"default {','.join(str(x) for x in v)}")
        else:
            g.add_argument(flag, type=type(v), default=v, help=f"default {v}")


def config_from_args(a: argparse.Namespace) -> SceneConfig:
    d, kw = SceneConfig(), {}
    for f in fields(SceneConfig):
        v, dv = getattr(a, f.name), getattr(d, f.name)
        if isinstance(dv, tuple):
            parts = [s for s in str(v).split(",") if s]
            kw[f.name] = tuple(type(dv[0])(s) for s in parts) if not isinstance(dv[0], str) else tuple(parts)
        else:
            kw[f.name] = v
    return SceneConfig(**kw)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cellcount.synth", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--bank", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", metavar="DIR", default=None)
    p.add_argument("--no-boxes", action="store_true")
    _add_config_flags(p)
    a = p.parse_args(argv)
    try:
        cfg = config_from_args(a)
        bank = Bank.from_dir(a.bank or resolve_data_root(a.data_root) / "synth_bank")
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    scenes = [generate_scene(bank, cfg, np.random.default_rng([a.seed, a.start + k])) for k in range(a.n)]
    if a.out:
        stems = write_dataset(scenes, a.out, a.split, a.start)
        print(f"wrote {len(stems)} scenes to {a.out} ({a.split}); {sum(len(s.boxes) for s in scenes)} boxes")
    if a.show or a.save or not a.out:
        from ..viz import Page, Panel, save_pages, show_pages
        pages = []
        for k, sc in enumerate(scenes):
            counts = {n: sum(1 for b in sc.boxes if b[0] == i) for i, n in enumerate(("rbc", "wbc", "platelet", "smudge"))}
            types = sorted({t for t in sc.subtypes if t})
            info = [f"scene {a.start + k} (seed {a.seed})  {sc.image.shape[1]}x{sc.image.shape[0]}px",
                    "  ".join(f"{n}={c}" for n, c in counts.items()) + (f"   wbc types: {', '.join(types)}" if types else ""),
                    "  ".join(f"{k2}={v}" for k2, v in sc.params.items())]
            pages.append(Page("Synthetic scene", info, [Panel("Scene", "overlay", sc.image, "Hotkeys r/w/p/s toggle the box colours.",
                                                              box_layers(sc, 1, not a.no_boxes))], 1, f"scene_{a.start + k:04d}"))
        if a.save:
            for path in save_pages(pages, a.save):
                print(f"saved {path}")
        if a.show or not a.save and not a.out:
            show_pages(pages)
    return 0
