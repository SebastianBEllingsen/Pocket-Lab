"""Turn a background-removal result into viewer pages (the only place that knows both sides).

Public API
    build_page(sample, img, result, ref_mask, boxes, opts) -> Page

``opts`` is a ``ViewOptions``: which view (minimal/full/explain) and the initial state
of each overlay layer. Layer hotkeys: o = ours, g = ground truth, d = difference, b = boxes.

Reference ("ground truth") means the mask that ships with the dataset (Data_Set_1);
it is the dataset authors' answer, not our own golden set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..common.datasets import Box, Sample
from ..common.metrics import dice, iou
from ..viz import Layer, Page, Panel, boxes_rgba, diff_rgba, outline_rgba, parse_color, pick_contrasting_colors
from .pipeline import BgResult

View = Literal["minimal", "full", "explain"]


@dataclass
class ViewOptions:
    view: View = "minimal"
    show_ours: bool = True
    show_gt: bool = True
    show_diff: bool = False
    show_boxes: bool = False
    thickness: int | None = None
    color_ours: str | None = None
    color_gt: str | None = None


def scores(result: BgResult, ref_mask: np.ndarray | None) -> dict[str, float]:
    out = dict(result.stats)
    if ref_mask is not None:
        out["iou"] = iou(result.mask > 0, ref_mask)
        out["dice"] = dice(result.mask > 0, ref_mask)
    return out


def _overlay_panel(img, result: BgResult, ref_mask, boxes: list[Box], o: ViewOptions) -> Panel:
    t = o.thickness or max(1, img.shape[1] // 500)
    ours = result.mask > 0
    masks = [ours] + ([ref_mask] if ref_mask is not None else [])
    labels = sorted({b.label for b in boxes})
    palette = pick_contrasting_colors(img, masks, 2 + len(labels))
    c_ours = parse_color(o.color_ours) if o.color_ours else palette[0]
    c_gt = parse_color(o.color_gt) if o.color_gt else palette[1]
    layers = [Layer("ours", "o", outline_rgba(ours, c_ours, t), c_ours, "ours (this algorithm)", o.show_ours)]
    if ref_mask is not None:
        layers.append(Layer("gt", "g", outline_rgba(ref_mask, c_gt, t), c_gt,
                            "ground truth (dataset mask)", o.show_gt))
        layers.append(Layer("diff", "d", diff_rgba(ours, ref_mask, c_ours, c_gt), c_ours,
                            "difference (ours only / truth only)", o.show_diff))
    if boxes:
        box_colors = dict(zip(labels, palette[2:]))
        layers.append(Layer("boxes", "b", boxes_rgba(img.shape[:2], boxes, box_colors, t),
                            palette[2] if labels else c_ours, "annotated boxes: " + "/".join(labels), o.show_boxes))
    cap = "Outlines drawn in colours chosen to contrast with this image. Toggle layers with the hotkeys in the legend."
    return Panel("Result", "overlay", img, cap, layers)


def _step_panel(step) -> Panel:
    cap = step.caption + " | " + ", ".join(f"{k}: {v}" for k, v in step.values.items() if k != "threshold")
    if step.kind == "swatches":
        cap = step.caption
    if step.kind == "heatmap":
        return Panel(step.name, "heatmap", step.array, cap, extra={"cmap": "magma", "vmin": 0})
    if step.kind == "swatches":
        return Panel(step.name, "swatches", step.array, step.caption,
                     extra={k: step.values[k] for k in ("labels", "agree", "chosen")})
    if step.kind == "hist":
        return Panel(step.name, "hist", step.array, step.caption,
                     extra={"threshold": step.values.get("threshold"), "xlabel": "Lab distance to background"})
    return Panel(step.name, "image", step.array, cap)


def build_page(sample: Sample, img, result: BgResult, ref_mask, boxes: list[Box], o: ViewOptions) -> Page:
    sc = scores(result, ref_mask)
    cfg = result.config
    info = [
        f"image: {sample.name}  ({sample.dataset}, {img.shape[1]}x{img.shape[0]}px)",
        f"method={cfg.method} proposers={','.join(cfg.proposers)} agree_tol={cfg.agree_tol:g} scale={cfg.scale:g} bins={cfg.bins} tol={cfg.tol:g} sigma_frac={cfg.sigma_frac:g} "
        f"blur={cfg.blur:g} min_area_frac={cfg.min_area_frac:g} fill_holes={cfg.fill_holes}",
        f"background colour votes: {int(sc['votes_agree'])}/{int(sc['votes_total'])} agree" if sc["votes_total"] > 1 else "",
        f"Otsu threshold={sc['threshold']:.1f}  foreground={sc['foreground_pct']:.1f}%  blobs={int(sc['blobs'])}  "
        f"bg colour (Lab)=({', '.join(f'{v:.0f}' for v in result.reference_lab)})",
        ("IoU=%.3f  Dice=%.3f  (against the dataset's reference mask)" % (sc["iou"], sc["dice"]))
        if "iou" in sc else "no reference mask in this dataset: judge by eye (use --boxes for annotated boxes)",
    ]
    info = [line for line in info if line]
    overlay = _overlay_panel(img, result, ref_mask, boxes, o)
    original = Panel("Original", "image", img)
    if o.view == "minimal":
        panels, cols = [original, overlay], 2
    elif o.view == "full":
        by_name = {s.name: s for s in result.trace}
        wanted = ["Background colour votes", "Background estimate", "Flattened", "Distance to background"]
        panels = [original] + [_step_panel(by_name[n]) for n in wanted if n in by_name] + [_step_panel(result.trace[-1]), overlay]
        cols = 4
    else:  # explain: every recorded step, then the overlay
        panels = [_step_panel(s) for s in result.trace] + [overlay]
        cols = 4
    return Page(f"Background removal - {o.view}", info, panels, cols, f"{sample.name}_{o.view}")
