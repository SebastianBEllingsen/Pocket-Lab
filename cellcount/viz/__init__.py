"""Generic visualisation: contrasting outline colours, overlay layers and figures.

Nothing in here knows about a specific pipeline stage; stages describe what to
show with ``Page``/``Panel``/``Layer`` objects and this package renders them.
"""
from .colors import parse_color, pick_contrasting_colors
from .figure import Layer, Page, Panel, save_pages, show_pages
from .overlay import boxes_rgba, diff_rgba, outline_rgba

__all__ = ["Layer", "Page", "Panel", "boxes_rgba", "diff_rgba", "outline_rgba",
           "parse_color", "pick_contrasting_colors", "save_pages", "show_pages"]
