"""Scene generator: one synthetic smear tile with exact YOLO boxes (pure logic, no files written).

A scene starts from a glass-colour field (taken from a real photo's estimated background) and
places cut-outs on it. Cells are blended as *optical density* (how much light a cell absorbs):
densities of overlapping cells add up, so two overlapping red cells are darker where they cross,
as in a real smear. Stain strength and hue are jittered per scene, then lens blur, sensor noise,
white-balance, vignette, JPEG compression and an optional eyepiece rim are applied.

Public API
    CLASSES                     detector class order: rbc, wbc, platelet, smudge (teammate's order)
    SceneConfig                 every parameter with its default (frozen dataclass)
    Scene                       image (BGR uint8), boxes [(cls_id, x1, y1, x2, y2)], subtypes, params
    Bank                        cut-outs grouped by class/subtype plus glass fields (from_dir)
    make_smudge(rng, radius)    procedural pale, ragged smudge cell (a Cutout)
    generate_scene(bank, cfg, rng) -> Scene

Box convention (teammate, confirmed 2026-10-27): tight box around the whole cell outline (WBC with cytoplasm,
platelet around the dot, smudge around the pale patch), clipped to the tile; cells less than half visible or
smaller than 6 px are not labelled. Rim rule: outside the counting circle (count_frac x eyepiece radius minus one
red-cell radius) the image is black and only cells whose box centre is inside the circle are labelled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from .cutouts import BANK_R, Cutout, load_bank

CLASSES = ("rbc", "wbc", "platelet", "smudge")
WBC_TYPES = ("neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil", "ig", "erythroblast")
SMUDGE_OD = np.array([0.55, 0.85, 0.65], np.float32)     # purple: absorbs mostly green (BGR order)


@dataclass(frozen=True)
class SceneConfig:
    tile: int = 640
    rbc_radius: float = 16.0               # detector's canonical red-cell radius (px)
    coverage: tuple[float, float] = (0.15, 0.65)   # share of the tile covered by red-cell discs, before overlap (random and clumps layouts)
    min_dist: tuple[float, float] = (0.7, 1.05)    # smallest centre distance, in red-cell diameters (lower = more overlap)
    layouts: tuple[str, ...] = ("random", "packed", "clumps")   # how red cells are arranged: scattered, tight hexagonal-ish (100x look), small clumps (40x look)
    layout_weights: tuple[float, ...] = (0.4, 0.25, 0.35)
    size_jitter: float = 0.10              # per-cell size variation
    zoom_jitter: float = 0.08              # per-tile size variation of everything
    wbc_mean: float = 1.5                  # white cells per tile (Poisson mean)
    platelet_mean: float = 4.0
    smudge_mean: float = 1.0
    wbc_types: tuple[str, ...] = WBC_TYPES
    stain_strength: tuple[float, float] = (0.7, 1.3)
    tint_sigma: float = 0.12               # per-channel stain hue jitter
    blur_sigma: tuple[float, float] = (0.7, 1.7)
    noise_sigma: tuple[float, float] = (0.8, 3.5)
    jpeg_quality: tuple[int, int] = (55, 95)
    wb_sigma: float = 0.04                 # white-balance gain jitter
    rbc_od: tuple[float, float] = (0.10, 0.30)     # mean optical density of a red cell, drawn once per scene
    rbc_od_jitter: float = 0.15                    # per-cell spread around it (log scale)
    pbc_rbc_weight: float = 0.7                    # chance that a scene's red cells come from PBC (pale centre) rather than Data_Set_1
    platelet_radius: tuple[float, float] = (0.2, 0.45)   # platelet radius as a fraction of the red-cell radius
    vignette: tuple[float, float] = (0.0, 0.25)
    rim_prob: float = 0.15                 # tiles that contain the eyepiece edge
    rim_radius: tuple[float, float] = (0.55, 0.9)   # eyepiece radius as a fraction of the tile
    count_frac: float = 0.93               # counting circle = this fraction of the eyepiece radius minus one red-cell radius (teammate's rule)
    own_bg_weight: float = 0.7             # chance of a glass field from your own photos
    margin: int = 48                       # canvas margin so cells can be cut by the tile edge


@dataclass
class Scene:
    image: np.ndarray
    boxes: list[tuple[int, int, int, int, int]]      # (class id, x1, y1, x2, y2)
    subtypes: list[str | None]                       # WBC type per box, else None
    params: dict = field(default_factory=dict)


@dataclass
class Bank:
    rbc: list[Cutout]
    rbc_groups: dict[str, list[Cutout]]      # red cells by source dataset, so one scene can use one look
    wbc: dict[str, list[Cutout]]
    platelet: list[Cutout]
    fields: list[np.ndarray]
    field_kind: list[str]

    @classmethod
    def from_dir(cls, d: str | Path) -> "Bank":
        d = Path(d)
        cuts = load_bank(d)
        wbc: dict[str, list[Cutout]] = {}
        for c in cuts:
            if c.cls == "wbc":
                wbc.setdefault(c.subtype, []).append(c)
        info = json.loads((d / "bg_index.json").read_text())
        fields = [cv.imread(str(d / e["file"])) for e in info]
        rbc = [c for c in cuts if c.cls == "rbc"]
        groups: dict[str, list[Cutout]] = {}
        for c in rbc:
            groups.setdefault(c.source.split(":")[0], []).append(c)
        return cls(rbc, groups, wbc, [c for c in cuts if c.cls == "platelet"],
                   fields, [e["kind"] for e in info])


def make_smudge(rng: np.random.Generator, radius: float = BANK_R) -> Cutout:
    """Pale, ragged purple blob without a nucleus (broken or smeared cell), at bank scale."""
    r = radius * rng.uniform(0.9, 1.7)
    n = int(2.6 * r) | 1
    yy, xx = np.mgrid[:n, :n] - n // 2
    ang = np.arctan2(yy, xx)
    wobble = 1 + sum(rng.uniform(0.05, 0.16) / k * np.cos(k * ang + rng.uniform(0, 6.28)) for k in range(2, 6))
    d = np.hypot(yy, xx) / (r * wobble)
    alpha = np.clip((1.0 - d) / 0.25, 0, 1)
    tex = cv.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), r * 0.18)
    tex = 1 + 0.5 * tex / (tex.std() + 1e-6)
    od = np.clip(rng.uniform(0.10, 0.28) * tex, 0, None)[..., None] * SMUDGE_OD * alpha[..., None]
    bg = np.array([225.0, 230.0, 235.0], np.float32)
    bgr = np.clip(bg * np.exp(-od), 0, 255).astype(np.uint8)
    return Cutout("smudge", None, bgr, (alpha * 255).astype(np.uint8), tuple(bg), "synthetic")


def _od(c: Cutout) -> np.ndarray:
    ratio = np.clip(c.bgr.astype(np.float32), 1, 255) / np.clip(np.array(c.bg_bgr, np.float32), 1, 255)
    return np.clip(-np.log(np.clip(ratio, 1e-3, 1.0)), 0, 3.0)


def _stain_dir(c: Cutout) -> np.ndarray:
    """Colour direction of a cut-out's optical density (per-channel share, mean 1)."""
    od = _od(c)[c.alpha > 128]
    m = od.mean(0) if len(od) else np.ones(3, np.float32)
    return (m / max(float(m.mean()), 1e-3)).astype(np.float32)


def _warp(c: Cutout, scale: float, angle: float, flip: bool, target_od: float | None = None, stain: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Optical density and alpha of ``c`` scaled, optionally mirrored, rotated; canvas grown to fit."""
    h, w = c.alpha.shape
    diag = int(np.ceil(np.hypot(h, w) * scale)) + 4
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += diag / 2 - w / 2
    M[1, 2] += diag / 2 - h / 2
    od0 = _od(c)
    if target_od is not None:      # same mean density for every red cell, whatever stain its source image had
        inside = c.alpha > 128
        if inside.any():
            lum = od0.mean(axis=2, keepdims=True)           # keeps the cell's own pattern (pale centre, rim)
            od0 = lum * (target_od / max(float(lum[inside].mean()), 1e-3)) * (stain if stain is not None else 1.0)
    od = cv.warpAffine(od0, M, (diag, diag), flags=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
    al = cv.warpAffine(c.alpha, M, (diag, diag), flags=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
    if flip:
        od, al = od[:, ::-1], al[:, ::-1]
    return od, al.astype(np.float32) / 255.0


def _glass(bank: Bank, cfg: SceneConfig, rng: np.random.Generator, size: int) -> np.ndarray:
    own = [i for i, k in enumerate(bank.field_kind) if k == "own"]
    pub = [i for i, k in enumerate(bank.field_kind) if k != "own"]
    pool = own if own and (rng.random() < cfg.own_bg_weight or not pub) else pub or own
    f = bank.fields[int(rng.choice(pool))]
    n = f.shape[0]
    c = int(n * rng.uniform(0.6, 1.0))
    y, x = rng.integers(0, n - c + 1, 2)
    f = np.rot90(f[y:y + c, x:x + c], int(rng.integers(4)))
    if rng.random() < 0.5:
        f = f[:, ::-1]
    f = cv.resize(np.ascontiguousarray(f), (size, size), interpolation=cv.INTER_CUBIC).astype(np.float32)
    blot = cv.GaussianBlur(rng.normal(0, 1, (size // 16 + 2, size // 16 + 2)).astype(np.float32), (0, 0), 1.5)
    blot = cv.resize(blot, (size, size), interpolation=cv.INTER_CUBIC)
    return f * (1 + 0.02 * blot / (blot.std() + 1e-6))[..., None]


def _positions(n: int, size: int, min_d: float, rng: np.random.Generator, lo: float, hi: float, sampler=None) -> np.ndarray:
    """Dart throwing: centres at least ``min_d`` apart where possible; a few overlaps are allowed
    after 30 failed tries so dense scenes still fill up. ``sampler`` draws candidates (default uniform)."""
    pts: list[np.ndarray] = []
    for _ in range(n):
        for t in range(30):
            p = sampler() if sampler else rng.uniform(lo, hi, 2)
            if not pts or np.min(np.hypot(*(np.array(pts) - p).T)) >= min_d:
                break
        pts.append(p)
    return np.array(pts).reshape(-1, 2)


def _packed(size: int, R: float, rng: np.random.Generator) -> np.ndarray:
    """Jittered hexagonal grid at a random angle and spacing, with random gaps (the 100x look)."""
    a = 2 * R * rng.uniform(0.95, 1.2)
    ext = int(size * 0.75 / a) + 3
    ii, jj = np.meshgrid(np.arange(-ext, ext + 1), np.arange(-ext, ext + 1))
    pts = np.c_[(ii + 0.5 * (jj % 2)).ravel() * a, (jj * a * np.sqrt(3) / 2).ravel()]
    t = rng.uniform(0, np.pi)
    pts = pts @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T + size / 2 + rng.uniform(-a, a, 2)
    pts += rng.normal(0, 0.12 * R, pts.shape)
    keep = (pts > -0.5 * R).all(1) & (pts < size + 0.5 * R).all(1) & (rng.random(len(pts)) > rng.uniform(0.0, 0.3))
    return pts[keep]


def _layout(kind: str, n: int, size: int, R: float, min_d: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "packed":
        return _packed(size, R, rng)
    if kind == "clumps":
        k = int(rng.integers(3, 11))
        centres = rng.uniform(0, size, (k, 2))
        sig = rng.uniform(2.0, 5.0, k) * R

        def draw():
            if rng.random() < 0.12:
                return rng.uniform(-0.5 * R, size + 0.5 * R, 2)
            j = int(rng.integers(k))
            return centres[j] + rng.normal(0, sig[j], 2)
        return _positions(int(n * 0.7), size, min_d, rng, 0, size, draw)
    return _positions(n, size, min_d, rng, -0.5 * R, size + 0.5 * R)


def generate_scene(bank: Bank, cfg: SceneConfig, rng: np.random.Generator) -> Scene:
    S, m = cfg.tile, cfg.margin
    size = S + 2 * m
    zoom = 1 + rng.uniform(-cfg.zoom_jitter, cfg.zoom_jitter)
    R = cfg.rbc_radius * zoom
    base = R / BANK_R
    od = np.zeros((size, size, 3), np.float32)
    placed: list[tuple[int, str | None, np.ndarray]] = []       # class id, subtype, alpha canvas bbox info

    scene_od = rng.uniform(*cfg.rbc_od)
    groups = bank.rbc_groups
    pbc = [g for g in groups if g == "pbc"]
    others = [g for g in groups if g != "pbc"]
    group = groups[pbc[0] if pbc and (rng.random() < cfg.pbc_rbc_weight or not others) else others[int(rng.integers(len(others)))]]
    look = groups[pbc[0]] if pbc else group          # stain colour always from PBC's pink-purple cells (Data_Set_1's grey-blue does not match the photos)
    rbc_stain = _stain_dir(look[int(rng.integers(len(look)))]) * np.exp(rng.normal(0, cfg.tint_sigma, 3)).astype(np.float32)

    def put(c: Cutout, cx: float, cy: float, cid: int, sub: str | None, jitter: float = 0.0, radius: float | None = None) -> None:
        s = base * (1 + rng.uniform(-jitter, jitter))
        if radius is not None:         # size by target radius (platelets: the bank's are big thrombocytes)
            s = radius / max(float(np.sqrt((c.alpha > 128).sum() / np.pi)), 1.0)
        target = scene_od * float(np.exp(rng.normal(0, cfg.rbc_od_jitter))) if cid == 0 else None
        o, a = _warp(c, s, rng.uniform(0, 360), bool(rng.random() < 0.5), target, rbc_stain if cid == 0 else None)
        h, w = a.shape
        x0, y0 = int(round(cx - w / 2)) + m, int(round(cy - h / 2)) + m
        sx0, sy0, sx1, sy1 = max(0, -x0), max(0, -y0), min(w, size - x0), min(h, size - y0)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        od[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1] += o[sy0:sy1, sx0:sx1] * a[sy0:sy1, sx0:sx1, None]
        ys, xs = np.nonzero(a > 0.3)
        if len(xs):
            placed.append((cid, sub, np.array([x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max()])))

    cov = rng.uniform(*cfg.coverage)
    n_rbc = int(cov * S * S / (np.pi * R * R))
    min_d = rng.uniform(*cfg.min_dist) * 2 * R
    w = np.array(cfg.layout_weights[:len(cfg.layouts)], float)
    layout = str(rng.choice(cfg.layouts, p=w / w.sum()))
    for x, y in _layout(layout, n_rbc, S, R, min_d, rng):
        put(group[int(rng.integers(len(group)))], x, y, 0, None, cfg.size_jitter)
    for x, y in _positions(rng.poisson(cfg.wbc_mean), S, 3 * R, rng, 0.0, S):
        t = str(rng.choice([t for t in cfg.wbc_types if bank.wbc.get(t)]))
        put(bank.wbc[t][int(rng.integers(len(bank.wbc[t])))], x, y, 1, t, 0.08)
    for x, y in _positions(rng.poisson(cfg.platelet_mean), S, 1.5 * R, rng, 0.0, S):
        if bank.platelet:
            put(bank.platelet[int(rng.integers(len(bank.platelet)))], x, y, 2, None, 0.0, R * rng.uniform(*cfg.platelet_radius))
    for x, y in _positions(rng.poisson(cfg.smudge_mean), S, 2 * R, rng, 0.0, S):
        put(make_smudge(rng), x, y, 3, None)

    strength = rng.uniform(*cfg.stain_strength)
    tint = np.exp(rng.normal(0, cfg.tint_sigma, 3)).astype(np.float32)
    img = _glass(bank, cfg, rng, size) * np.exp(-od * strength * tint / tint.mean())
    img *= np.exp(rng.normal(0, cfg.wb_sigma, 3)).astype(np.float32)
    yy, xx = np.mgrid[:size, :size] / size - 0.5
    img *= (1 - rng.uniform(*cfg.vignette) * 2 * (xx ** 2 + yy ** 2))[..., None]
    img = cv.GaussianBlur(img, (0, 0), rng.uniform(*cfg.blur_sigma))
    img += rng.normal(0, rng.uniform(*cfg.noise_sigma), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)[m:m + S, m:m + S]

    rim = None
    if rng.random() < cfg.rim_prob:
        c = rng.uniform(0.3, 0.7, 2) * S
        rad = max(cfg.count_frac * rng.uniform(*cfg.rim_radius) * S - R, 0.1 * S)    # counting circle, not the eyepiece edge
        yy, xx = np.mgrid[:S, :S]
        rim = np.hypot(xx - c[0], yy - c[1]) <= rad
        img[~rim] = 0
    ok, enc = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, int(rng.integers(*cfg.jpeg_quality))])
    img = cv.imdecode(enc, cv.IMREAD_COLOR)

    boxes, subs = [], []
    for cid, sub, b in placed:
        full = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        x1, y1, x2, y2 = (int(np.clip(v - m, 0, S - 1)) for v in b)   # box centre decides below
        if (x2 - x1 + 1) * (y2 - y1 + 1) < 0.5 * full or x2 - x1 < 6 or y2 - y1 < 6:
            continue
        if rim is not None and not rim[(y1 + y2) // 2, (x1 + x2) // 2]:
            continue
        boxes.append((cid, x1, y1, x2, y2))
        subs.append(sub)
    return Scene(img, boxes, subs, dict(zoom=round(zoom, 3), layout=layout, coverage=round(cov, 3), min_dist=round(min_d / (2 * R), 3),
                                        stain=round(float(strength), 3), rbc_od=round(float(scene_od), 3), rim=rim is not None))
