"""Shared figure style for the Hilti report.

Palette slots come from the data-viz reference palette unchanged (slots 1-3 for the
all-pairs case, the fixed status four for state), so no re-stepping is involved.
Figures render on the light surface and are shown on a light card in both page
themes -- a raster PNG cannot follow the viewer's theme, so it does not pretend to.
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# categorical, first three slots (validated all-pairs in both modes)
KD = "#2a78d6"        # slot 1 blue   - kd_localizer
FL = "#eb6834"        # slot 2 orange - FAST-LIO
GT = "#1baf7a"        # slot 3 aqua   - surveyed control point (always direct-labelled:
                      #                 it sits below 3:1 on the light surface)
# status four, fixed
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# The map is context behind the trajectories, so it gets a neutral ramp rather than a
# hue that would compete with the two series drawn on top of it (emphasis, not
# categorical). Height still reads, but recessively.
MAPGRAY = LinearSegmentedColormap.from_list("mapgray", ["#dedcd4", "#6f6d66", "#22211e"])
# Standalone map-geometry panels have no trajectory over them, so there the blue
# sequential ramp is free to carry magnitude at full strength.
SEQBLUE = LinearSegmentedColormap.from_list(
    "seqblue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"])


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "600",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2.0,       # 2px lines
        "lines.solid_capstyle": "round",
        "figure.dpi": 130,
    })


def ring(artist, lw=2.0, color=SURFACE):
    """2px surface ring so overlapping marks stay separable."""
    import matplotlib.patheffects as pe
    artist.set_path_effects([pe.withStroke(linewidth=lw + 2, foreground=color)])


def label(ax, x, y, text, color=INK2, **kw):
    t = ax.annotate(text, (x, y), color=color, fontsize=7.2, weight="600",
                    textcoords="offset points", **kw)
    import matplotlib.patheffects as pe
    t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=SURFACE)])
    return t


def tidy(ax):
    ax.tick_params(length=3, width=0.7)
    ax.set_axisbelow(True)
