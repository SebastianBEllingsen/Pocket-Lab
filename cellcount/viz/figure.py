"""Render pages of panels with matplotlib, interactively or to files.

A stage describes *what* to show with plain data objects; this module decides
*how*. It knows nothing about any pipeline stage.

Public API
    Layer   one toggleable RGBA overlay (name, hotkey, rgba array, colour, label, visible)
    Panel   one subplot: kind "image" | "heatmap" | "hist" | "overlay", plus title/caption
    Page    one figure: title, info lines (shown at the top), panels, column count
    show_pages(pages)        interactive window; <-/-> change image, layer hotkeys and
                             check-boxes toggle overlays one at a time, q closes
    save_pages(pages, dir)   write one PNG per page (layers use their initial visibility)

Panel.data by kind
    image     HxWx3 BGR uint8, or HxW (shown in grey)
    heatmap   HxW float; extra: cmap, vmin, vmax
    hist      1-D values; extra: threshold (drawn as a vertical line), xlabel
    overlay   base BGR image; the layers go in Panel.layers
"""
from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2 as cv
import numpy as np

RGB = tuple[int, int, int]


@dataclass
class Layer:
    name: str
    key: str
    rgba: np.ndarray
    color: RGB
    label: str
    visible: bool = True


@dataclass
class Panel:
    title: str
    kind: Literal["image", "heatmap", "hist", "overlay"]
    data: Any = None
    caption: str = ""
    layers: list[Layer] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    title: str
    info_lines: list[str]
    panels: list[Panel]
    columns: int = 3
    filename: str = "page"


def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr


class PageView:
    """Builds one Page into a Figure and handles layer toggling."""

    def __init__(self, fig, page: Page, interactive: bool):
        from matplotlib.widgets import CheckButtons

        self._CheckButtons = CheckButtons
        self.fig, self.page, self.interactive = fig, page, interactive
        self._layer_info: dict[str, Layer] = {}
        for p in page.panels:
            for layer in p.layers:
                self._layer_info.setdefault(layer.key, layer)
        self.visible = {k: l.visible for k, l in self._layer_info.items()}
        self._artists: dict[str, list] = {k: [] for k in self.visible}
        self._overlay_axes: list = []
        self._checks = None
        self._build()

    def _build(self) -> None:
        import matplotlib.pyplot as plt  # noqa: F401  (ensures the backend is initialised)

        page, fig = self.page, self.fig
        n = len(page.panels)
        cols = min(page.columns, n)
        rows = math.ceil(n / cols)
        header_in = 0.55 + 0.17 * len(page.info_lines)      # inches reserved for title + info lines
        has_layers = any(p.layers for p in page.panels)
        # Bottom strip, outside every image: holds the layer check boxes (interactive)
        # or the legend (saved PNG), so neither covers the pictures.
        n_layers = len(self._layer_info)
        footer_in = 0.6 + (0.3 * n_layers + 0.2 if has_layers else 0.0)
        fig_h = 3.9 * rows + header_in + footer_in
        self._fig_h, self._footer_in = fig_h, footer_in
        fig.set_size_inches(4.4 * cols, fig_h)
        gs = fig.add_gridspec(rows, cols, left=0.03, right=0.985, bottom=footer_in / fig_h,
                              top=1 - header_in / fig_h, hspace=0.5, wspace=0.12)
        fig.suptitle(page.title, fontsize=12, fontweight="bold", y=1 - 0.08 / fig_h)
        fig.text(0.01, 1 - 0.4 / fig_h, "\n".join(page.info_lines), va="top", ha="left",
                 fontsize=8, family="monospace")
        for i, panel in enumerate(page.panels):
            self._draw_panel(fig.add_subplot(gs[i // cols, i % cols]), panel)
        if self.visible:
            self._add_controls()
        self._apply()

    def _draw_panel(self, ax, panel: Panel) -> None:
        ax.set_title(panel.title, fontsize=9)
        if panel.kind == "image":
            d = panel.data
            ax.imshow(_rgb(d), cmap="gray", vmin=0, vmax=255) if d.ndim == 2 else ax.imshow(_rgb(d))
        elif panel.kind == "heatmap":
            e = panel.extra
            im = ax.imshow(panel.data, cmap=e.get("cmap", "magma"), vmin=e.get("vmin"), vmax=e.get("vmax"))
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        elif panel.kind == "hist":
            vals = np.asarray(panel.data).ravel()
            ax.hist(vals[:: max(1, len(vals) // 200_000)], bins=64, color="0.55", log=True)
            thr = panel.extra.get("threshold")
            if thr is not None:
                ax.axvline(thr, color="red", lw=1.5, label=f"Otsu threshold = {thr:.1f}")
                ax.legend(fontsize=7)
            ax.set_xlabel(panel.extra.get("xlabel", ""), fontsize=7)
            ax.tick_params(labelsize=7)
        elif panel.kind == "overlay":
            ax.imshow(_rgb(panel.data))
            for layer in panel.layers:
                self._artists[layer.key].append(ax.imshow(layer.rgba))
            self._overlay_axes.append((ax, panel))
        else:
            raise ValueError(f"Unknown panel kind {panel.kind!r}")
        if panel.kind != "hist":
            ax.set_xticks([])
            ax.set_yticks([])
        if panel.caption:
            ax.set_xlabel(textwrap.fill(panel.caption, 52), fontsize=7)

    def _add_controls(self) -> None:
        if not self.interactive:
            return
        keys = list(self.visible)
        h_in = 0.3 * len(keys) + 0.1
        ax = self.fig.add_axes([0.01, 0.05 / self._fig_h, 0.26, h_in / self._fig_h])
        labels = [f"[{k}] {self._layer_info[k].label}" for k in keys]
        self._checks = self._CheckButtons(ax, labels, [self.visible[k] for k in keys])
        for key, t in zip(keys, self._checks.labels):
            t.set_fontsize(8)
            t.set_bbox(dict(facecolor=np.array(self._layer_info[key].color) / 255, alpha=0.45,
                            edgecolor="none", pad=2))  # label highlighted in its outline colour
        self._keys = keys
        self._checks.on_clicked(lambda label: self._toggle(self._keys[labels.index(label)]))

    def _toggle(self, key: str) -> None:
        self.visible[key] = not self.visible[key]
        self._apply()
        self.fig.canvas.draw_idle()

    def press(self, key: str) -> bool:
        """Toggle the layer bound to ``key``; returns False if the key is unused."""
        if key not in self.visible:
            return False
        if self._checks is not None:
            self._checks.set_active(self._keys.index(key))  # fires _toggle via the callback
        else:
            self._toggle(key)
        return True

    def _apply(self) -> None:
        from matplotlib.lines import Line2D

        for key, arts in self._artists.items():
            for a in arts:
                a.set_visible(self.visible[key])
        if self.interactive:
            return  # the check boxes (coloured like the outlines) are the legend
        for ax, panel in self._overlay_axes:
            handles = [Line2D([0], [0], color=np.array(l.color) / 255, lw=3,
                              label=f"[{l.key}] {l.label}: {'ON' if self.visible[l.key] else 'off'}")
                       for l in panel.layers]
            if handles:
                n_cap = textwrap.fill(panel.caption, 52).count("\n") + 1 if panel.caption else 0
                ax.legend(handles=handles, loc="upper center", fontsize=7, framealpha=0.9,
                          bbox_to_anchor=(0.5, -0.04 - 0.055 * n_cap))


def save_pages(pages: list[Page], out_dir: str | Path, dpi: int = 110) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for page in pages:
        fig = plt.figure()
        PageView(fig, page, interactive=False)
        path = out / f"{page.filename}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)
    return written


def show_pages(pages: list[Page]) -> None:
    import matplotlib.pyplot as plt

    # Free our hotkeys from matplotlib's defaults (g = grid, o = zoom, p = pan, ...).
    ours = {l.key for pg in pages for pn in pg.panels for l in pn.layers} | {"n", "p"}
    for name in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in ours]

    fig = plt.figure()
    state = {"i": 0, "view": None}

    def draw() -> None:
        fig.clf()
        page = pages[state["i"]]
        state["view"] = PageView(fig, page, interactive=True)
        fig.text(0.99, 0.005, f"image {state['i'] + 1}/{len(pages)}   <- / -> change image   q close",
                 ha="right", fontsize=8)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        k = event.key
        if k in ("right", "n"):
            state["i"] = (state["i"] + 1) % len(pages)
            draw()
        elif k in ("left", "p"):
            state["i"] = (state["i"] - 1) % len(pages)
            draw()
        elif k == "q":
            plt.close(fig)
        elif state["view"] is not None:
            state["view"].press(k)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
