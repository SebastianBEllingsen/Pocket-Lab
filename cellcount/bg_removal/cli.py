"""Command line for the background-removal stage.

Run:  python -m cellcount.bg_removal [flags]      (see --help)

Typical use
    python -m cellcount.bg_removal --random 4                    original vs. result
    python -m cellcount.bg_removal --full --worst 6 --stats      6 lowest-IoU images, full comparison
    python -m cellcount.bg_removal --explain --image <name>      every step with its values
    python -m cellcount.bg_removal --dataset 2 --boxes --index 7 --method median

Views (choose one; default --minimal)
    --minimal   original and final result
    --full      original, background, flattened, distance map, final mask, result
    --explain   every pipeline step with the values used, then the result

Data and selection (shared by all stages, see cellcount/common/selection.py)
    --dataset NAME    1/rbc-masks (default) or 2/bccd        --data-root DIR
    --image/--index/--random/--worst/--pool/--seed/--list

Output
    --show            interactive window (default if --save is not given)
                      keys: <-/-> change image, o/g/d/b toggle layers, q close
    --save DIR        write one PNG per image
    --stats           print a score table (IoU/Dice need a reference mask)
    --no-ours --no-gt start with that outline hidden;  --diff --boxes start with that layer shown
    --thickness PX    outline width       --outline-ours/--outline-gt COLOR  override the auto colours

Algorithm parameters (defaults come from BgConfig)
    --method {mode,median} --scale --bins --tol --sigma-frac --kernel-frac --blur
    --min-area-frac --no-fill-holes
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields

import numpy as np

from ..common.datasets import available_datasets, get_dataset
from ..common.selection import add_selection_args, print_listing, select_samples
from .config import BgConfig
from .methods import METHODS
from .panels import ViewOptions, build_page, scores
from .pipeline import remove_background


def build_parser() -> argparse.ArgumentParser:
    d = BgConfig()
    p = argparse.ArgumentParser(prog="python -m cellcount.bg_removal", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="1", help=f"one of {', '.join(available_datasets())} (default 1)")
    g.add_argument("--data-root", default=None, help="folder containing Data_Set_1, ... (default: $CELLCOUNT_DATA_ROOT or ./Data_Sets)")
    add_selection_args(p)

    v = p.add_argument_group("view (choose one)").add_mutually_exclusive_group()
    v.add_argument("--minimal", dest="view", action="store_const", const="minimal", help="original and result (default)")
    v.add_argument("--full", dest="view", action="store_const", const="full", help="main stages side by side")
    v.add_argument("--explain", dest="view", action="store_const", const="explain", help="every step with its values")
    p.set_defaults(view="minimal")

    o = p.add_argument_group("output")
    o.add_argument("--show", action="store_true", help="interactive window (default when --save is absent)")
    o.add_argument("--save", metavar="DIR", help="write one PNG per image into DIR")
    o.add_argument("--stats", action="store_true", help="print a score table to the terminal")
    o.add_argument("--no-ours", action="store_true", help="start with our outline hidden")
    o.add_argument("--no-gt", action="store_true", help="start with the ground-truth outline hidden")
    o.add_argument("--diff", action="store_true", help="start with the difference layer shown")
    o.add_argument("--boxes", action="store_true", help="start with annotated boxes shown (BCCD)")
    o.add_argument("--thickness", type=int, default=None, help="outline width in px (default scales with image)")
    o.add_argument("--outline-ours", metavar="COLOR", help="name, #rrggbb or r,g,b (default: auto contrast)")
    o.add_argument("--outline-gt", metavar="COLOR", help="name, #rrggbb or r,g,b (default: auto contrast)")

    a = p.add_argument_group("algorithm")
    a.add_argument("--method", choices=sorted(METHODS), default=d.method)
    a.add_argument("--scale", type=float, default=d.scale, help="working resolution fraction")
    a.add_argument("--bins", type=int, default=d.bins, help="mode: Lab histogram bins per axis")
    a.add_argument("--tol", type=float, default=d.tol, help="mode: robust sigmas counted as background")
    a.add_argument("--sigma-frac", type=float, default=d.sigma_frac, help="mode: background smoothing (fraction of size)")
    a.add_argument("--kernel-frac", type=float, default=d.kernel_frac, help="median: kernel size (fraction of size)")
    a.add_argument("--blur", type=float, default=d.blur, help="distance-map blur sigma in px")
    a.add_argument("--min-area-frac", type=float, default=d.min_area_frac, help="remove blobs smaller than this fraction")
    a.add_argument("--no-fill-holes", action="store_true", help="do not fill enclosed holes")
    return p


def config_from_args(args: argparse.Namespace) -> BgConfig:
    kw = {f.name: getattr(args, f.name) for f in fields(BgConfig) if f.name != "fill_holes"}
    return BgConfig(**kw, fill_holes=not args.no_fill_holes)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        samples = get_dataset(args.dataset, args.data_root)
        if args.list:
            print_listing(samples)
            return 0
        cfg = config_from_args(args)

        def scorer(s):
            ref = s.load_mask()
            return None if ref is None else scores(remove_background(s.load_image(), cfg), ref)["iou"]

        chosen = select_samples(samples, args, scorer)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    opts = ViewOptions(args.view, not args.no_ours, not args.no_gt, args.diff, args.boxes,
                       args.thickness, args.outline_ours, args.outline_gt)
    pages, rows = [], []
    for s in chosen:
        img, ref, boxes = s.load_image(), s.load_mask(), s.load_boxes()
        result = remove_background(img, cfg)
        pages.append(build_page(s, img, result, ref, boxes, opts))
        rows.append((s.name, scores(result, ref)))

    if args.stats:
        print_stats(rows)
    if args.save:
        from ..viz import save_pages
        for path in save_pages(pages, args.save):
            print(f"saved {path}")
    if args.show or not args.save:
        from ..viz import show_pages
        show_pages(pages)
    return 0


def print_stats(rows: list[tuple[str, dict]]) -> None:
    print(f"{'image':40s} {'IoU':>6s} {'Dice':>6s} {'thresh':>7s} {'fg %':>6s} {'blobs':>6s}")
    for name, s in rows:
        iou = f"{s['iou']:.3f}" if "iou" in s else "-"
        dc = f"{s['dice']:.3f}" if "dice" in s else "-"
        print(f"{name[:40]:40s} {iou:>6s} {dc:>6s} {s['threshold']:7.1f} {s['foreground_pct']:6.1f} {int(s['blobs']):6d}")
    ious = [s["iou"] for _, s in rows if "iou" in s]
    if ious:
        print(f"mean IoU {np.mean(ious):.3f}  min {np.min(ious):.3f}  max {np.max(ious):.3f}  (n={len(ious)})")
