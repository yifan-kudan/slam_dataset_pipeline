#!/usr/bin/env python3
"""One palette and one matplotlib theme for every figure in the re-run report.

Values come verbatim from the data-viz reference palette, so the documented
validation applies unchanged: categorical slots 1-3 (blue, orange, aqua) are the
subset that clears every all-pairs gate in BOTH modes -- worst pair CVD dE 9.2
light / 9.4 dark, normal-vision 24.0 / 20.9. Three series is therefore the cap;
a fourth would put yellow beside orange and fail the all-pairs floors.

Every figure is rendered twice, once per mode, and the page shows the one that
matches the reader's theme -- a PNG cannot adapt, and a light-only figure inside
a dark page is the usual way this goes wrong. `MODES` drives that loop.

Aqua and the light-mode magenta sit below 3:1 on the light surface, which triggers
the relief rule: every figure here is backed by the full table of the same numbers
in the report body, so no value is ever carried by colour alone.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

LIGHT = {
    "mode": "light",
    "surface": "#fcfcfb", "plane": "#f9f9f7",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
    # Sequential blue, 100 -> 700, for the coverage heat maps.
    "seq": ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
}
DARK = {
    "mode": "dark",
    "surface": "#1a1a19", "plane": "#0d0d0d",
    "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70"],
    "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
    "seq": ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
            "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"],
}
MODES = (LIGHT, DARK)

SANS = ["DejaVu Sans", "system-ui", "Helvetica", "Arial", "sans-serif"]


def apply(t):
    """Set the rcParams for one mode. Grid and axes recede; marks stay thin."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": SANS, "font.size": 9,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "savefig.edgecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "axes.titlecolor": t["ink"], "axes.titlesize": 10, "axes.titleweight": "medium",
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": t["grid"], "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": t["ink2"],
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.16,
    })


def seq_cmap(t):
    """The one-hue sequential ramp as a matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(f"kdseq_{t['mode']}", t["seq"])


def label(ax, x, y, text, t, dx=0, dy=0, **kw):
    """Direct label in INK, never in the series colour (text wears text tokens)."""
    kw.setdefault("fontsize", 7.5)
    kw.setdefault("color", t["ink2"])
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), **kw)


def short(seq):
    """site1_handheld_2 -> s1/hh2 -- axis labels have to fit 15 rows."""
    site = seq.split("_")[0].replace("site", "s")
    kind = "hh" if "handheld" in seq else "rb"
    return f"{site}/{kind}{seq.rsplit('_', 1)[-1]}"
