"""Shared plotting utilities: color palettes and palette helpers.

Single source of truth for the presentation color palette used across the
ESCan notebooks and the statistical plotting helpers.
"""

# Presentation slide palette (teal / lime / gold).
SLIDE_PALETTE = {
    "td": "#2F97A7",
    "asd": "#A7CF00",
    "teal_dark": "#1F6A78",
    "gold": "#E6BC34",
    "gold_dark": "#9F7D1C",
    "bg_alt": "#EAF2F4",
    "grid": "#C7D6DB",
    "neutral": "#7E8F95",
    "accent": "#2ca02c",
}


def get_palette(overrides=None):
    """Return a copy of :data:`SLIDE_PALETTE`, optionally updated with ``overrides``."""
    palette = dict(SLIDE_PALETTE)
    if overrides is not None:
        palette.update(dict(overrides))
    return palette
