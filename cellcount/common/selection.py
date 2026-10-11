"""Shared image-selection flags so every stage CLI picks images the same way.

Public API
    add_selection_args(parser)                 adds the flags below to an argparse parser
    select_samples(samples, args, scorer)      returns the chosen Samples (in order)
    print_listing(samples)                     prints "index  name" lines (for --list)

Flags (all optional, combinable; results are concatenated in this order)
    --image NAME_OR_PATH   image by dataset name/stem, or by a file path (repeatable)
    --index N              image by index in the sorted dataset (repeatable; see --list)
    --random N             N random images (reproducible with --seed)
    --worst N              the N images with the lowest score (needs a reference mask);
                           searched within --pool random images to bound the run time
    --pool N               candidate pool size for --worst (default 100, 0 = whole dataset)
    --seed S               random seed (default 0)
    --list                 print all image indices and names, then exit
With no selector at all, 4 random images are used.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

from .datasets import Sample, sample_from_path

Scorer = Callable[[Sample], "float | None"]
DEFAULT_RANDOM = 4


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("image selection")
    g.add_argument("--image", action="append", default=[], metavar="NAME_OR_PATH",
                   help="image by name/stem or file path (repeatable)")
    g.add_argument("--index", action="append", type=int, default=[], metavar="N",
                   help="image by index in the sorted dataset (repeatable; see --list)")
    g.add_argument("--random", type=int, default=None, metavar="N", help="N random images")
    g.add_argument("--worst", type=int, default=None, metavar="N",
                   help="the N lowest-scoring images (needs a reference mask)")
    g.add_argument("--pool", type=int, default=100, metavar="N",
                   help="candidates examined by --worst (0 = whole dataset; default 100)")
    g.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    g.add_argument("--list", action="store_true", help="list image indices and names, then exit")


def print_listing(samples: list[Sample]) -> None:
    for i, s in enumerate(samples):
        print(f"{i:5d}  {s.name}")


def _find(samples: list[Sample], token: str) -> Sample:
    path = Path(token).expanduser()
    if path.is_file():
        resolved = path.resolve()
        for s in samples:
            if s.image_path.resolve() == resolved:
                return s
        return sample_from_path(path)
    exact = [s for s in samples if token in (s.name, s.image_path.name)]
    if exact:
        return exact[0]
    prefix = [s for s in samples if s.name.startswith(token)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        raise ValueError(f"{token!r} is ambiguous ({len(prefix)} matches, e.g. {prefix[0].name})")
    raise ValueError(f"No image named {token!r} in this dataset (use --list)")


def select_samples(samples: list[Sample], args: argparse.Namespace,
                   scorer: Scorer | None = None) -> list[Sample]:
    rng = random.Random(args.seed)
    chosen: list[Sample] = [_find(samples, t) for t in args.image]
    for i in args.index:
        if not 0 <= i < len(samples):
            raise ValueError(f"--index {i} out of range (0..{len(samples) - 1})")
        chosen.append(samples[i])
    if args.random:
        chosen += rng.sample(samples, min(args.random, len(samples)))
    if args.worst:
        if scorer is None:
            raise ValueError("--worst is not available for this stage")
        pool = samples if args.pool == 0 else rng.sample(samples, min(args.pool, len(samples)))
        scored = []
        for n, s in enumerate(pool, 1):
            print(f"\rscoring {n}/{len(pool)}", end="", file=sys.stderr, flush=True)
            score = scorer(s)
            if score is not None:
                scored.append((score, s))
        print(file=sys.stderr)
        if not scored:
            raise ValueError("--worst needs a dataset with reference masks (try --dataset 1)")
        scored.sort(key=lambda t: t[0])
        chosen += [s for _, s in scored[:args.worst]]
    if not chosen:
        chosen = rng.sample(samples, min(DEFAULT_RANDOM, len(samples)))
    return chosen
