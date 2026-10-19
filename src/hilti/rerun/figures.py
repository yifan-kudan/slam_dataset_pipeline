#!/usr/bin/env python3
"""Every figure in the re-run report, drawn twice (light + dark) from data.json.

    figures.py [--data <path>] [--out <dir>] [--only fig_frames fig_cp ...]

Each `fig_*` function takes (data, theme, out_dir) and writes <name>_<mode>.png.
The report body embeds both and shows the one matching the reader's theme.

Figures, and the question each one answers:

  fig_frames    did the DDS fix actually land -- what share of each bag's clouds
                FAST-LIO processed, before and after
  fig_cp        control-point error for the three configurations that exist for a
                sequence: FAST-LIO, kd_localizer in its own map, kd_localizer in
                its site's hub map
  fig_matrix    the coverage matrix per site -- how much of sequence B's cloud
                sequence A's map explains, which is what picks the hub
  fig_sites     top-down of each hub map with every sequence's registered route on
                it, so "coverage" is visible and not only tabulated
  fig_cross     error against the sequence's own FAST-LIO trajectory over time, one
                panel per cross-sequence run: where a run slips and whether it
                comes back
  fig_health    inlier ratio and covariance factor over every own-map run -- the
                two quantities the diagnostics threshold
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hilti import dataset                                          # noqa: E402
from hilti.rerun import style                                      # noqa: E402


def save(fig, out_dir, name, theme):
    path = os.path.join(out_dir, f"{name}_{theme['mode']}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def ordered(data):
    """Sequences grouped by site then platform, so every figure shares one order."""
    rows = list(data["sequences"].values())
    rows.sort(key=lambda r: (r["site"], r["platform"] != "robot", r["seq"]))
    return rows


# --------------------------------------------------------------------------
def fig_frames(data, t, out_dir):
    """Dumbbell: the share of each bag's clouds FAST-LIO saw, before -> after.

    A dumbbell and not a grouped bar because the story is one arrow per sequence,
    and 15 pairs of bars would read as 30 competing magnitudes.
    """
    rows = [r for r in ordered(data) if r.get("fastlio")]
    if not rows:
        return None
    style.apply(t)
    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(rows) + 1.5))
    y = np.arange(len(rows))
    before = [(r["fastlio_prefix"] or {}).get("frame_pct") for r in rows]
    after = [r["fastlio"]["frame_pct"] for r in rows]

    for i, (b, a) in enumerate(zip(before, after)):
        if b is not None:
            ax.plot([b, a], [i, i], color=t["axis"], lw=1.2, zorder=1,
                    solid_capstyle="round")
    ok = [(i, b) for i, b in enumerate(before) if b is not None]
    if ok:
        ax.scatter([b for _, b in ok], [i for i, _ in ok], s=34,
                   color=t["series"][1], zorder=3, label="before the DDS profile",
                   edgecolors=t["surface"], linewidths=1.2)
    ax.scatter(after, y, s=34, color=t["series"][0], zorder=3,
               label="rerun_r1.0", edgecolors=t["surface"], linewidths=1.2)
    for i, a in enumerate(after):
        style.label(ax, a, i, f"{a:.1f}", t, dx=7, dy=-3)

    ax.set_yticks(y, [style.short(r["seq"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("clouds processed, % of the bag's own count")
    ax.set_title("Every sequence now runs on all of its data", pad=30)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    # The title and a legend both default to the band above the axes; pad the title out
    # of the way and anchor the legend to the axes edge underneath it.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3)
    return save(fig, out_dir, "fig_frames", t)


# --------------------------------------------------------------------------
def fig_cp(data, t, out_dir):
    """Control-point error, three configurations per sequence, on a log axis.

    Dots and not bars: the range spans 1.7 cm to several metres, and a log axis is
    the only honest way to show that -- which rules bars out, since a log bar has
    no meaningful length.
    """
    rows = ordered(data)
    cross_by_seq = {c["seq"]: c for c in data["cross"].values()}
    series = [
        ("FAST-LIO", lambda r: (r.get("fastlio") or {}).get("err_mean_cm"), 0),
        ("kd_localizer, own map", lambda r: (r.get("localizer") or {}).get("err_mean_cm"), 1),
        ("kd_localizer, hub map",
         lambda r: ((cross_by_seq.get(r["seq"]) or {}).get("localizer") or {})
         .get("err_mean_cm"), 2),
    ]
    if not any(fn(r) for r in rows for _, fn, _ in series):
        return None
    style.apply(t)
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.6))
    y = np.arange(len(rows))
    offsets = (-0.22, 0.0, 0.22)
    markers = ("o", "s", "^")
    for (name, fn, slot), off, mk in zip(series, offsets, markers):
        xs, ys = [], []
        for i, r in enumerate(rows):
            v = fn(r)
            if v is not None and v > 0:
                xs.append(v)
                ys.append(i + off)
        if xs:
            ax.scatter(xs, ys, s=30, marker=mk, color=t["series"][slot], label=name,
                       edgecolors=t["surface"], linewidths=1.0, zorder=3)
    for i in y:
        ax.axhline(i + 0.5, color=t["grid"], lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(y, [style.short(r["seq"]) for r in rows])
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("mean control-point pair-distance error, cm (log)")
    ax.set_title("Accuracy against the survey, all three configurations", pad=30)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3)
    return save(fig, out_dir, "fig_cp", t)


# --------------------------------------------------------------------------
def fig_matrix(data, t, out_dir):
    """Per-site coverage matrix: row = map, column = sequence placed in it."""
    mats = {s: m for s, m in (data.get("matrix") or {}).items() if m}
    if not mats:
        return None
    style.apply(t)
    cmap = style.seq_cmap(t)
    n = len(mats)
    widths = []
    grids = {}
    for site, rows in sorted(mats.items()):
        names = sorted({r["hub"] for r in rows} | {r["seq"] for r in rows})
        grid = np.full((len(names), len(names)), np.nan)
        idx = {v: i for i, v in enumerate(names)}
        for r in rows:
            grid[idx[r["hub"]], idx[r["seq"]]] = r["overlap_0.5m"]
        grids[site] = (names, grid)
        widths.append(len(names))
    fig, axes = plt.subplots(1, n, figsize=(0.72 * sum(widths) + 2.2, 3.4),
                             gridspec_kw={"width_ratios": widths, "wspace": 0.42})
    axes = np.atleast_1d(axes)
    for ax, (site, (names, grid)) in zip(axes, sorted(grids.items())):
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), [style.short(v) for v in names],
                      rotation=45, ha="right")
        ax.set_yticks(range(len(names)), [style.short(v) for v in names])
        ax.set_title(f"site {site}")
        for i in range(len(names)):
            for j in range(len(names)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "—", ha="center", va="center",
                            color=t["muted"], fontsize=8)
                else:
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.5,
                            color="#ffffff" if grid[i, j] > 0.55 else t["ink"])
        ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
        ax.grid(which="minor", color=t["surface"], linewidth=2)
        ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("map built from")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.022, pad=0.02)
    cb.set_label("fraction of the column's cloud within 0.5 m", color=t["ink2"],
                 fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=t["muted"], labelcolor=t["ink2"])
    fig.suptitle("Which map explains which sequence", x=0.03, y=1.04, ha="left",
                 fontsize=10, color=t["ink"])
    return save(fig, out_dir, "fig_matrix", t)


# --------------------------------------------------------------------------
def fig_sites(data, t, out_dir, run="rerun_r1.0"):
    """Top-down of each hub map with every other sequence's registered route on it.

    Two series only, which is deliberate: the palette validates three slots and five
    routes would need more, so identity lives in the table and this figure answers the
    one question a picture can - does the map cover where the walks went. Axes are clipped
    to the hub cloud's 0.5-99.5 percentile, because a FAST-LIO cloud carries stray returns
    hundreds of metres out and they would otherwise shrink the building to a smudge.
    """
    hubs = data.get("hubs") or {}
    if not hubs:
        return None
    from hilti import xseq
    style.apply(t)
    sites = sorted(hubs.items())
    fig, axes = plt.subplots(1, len(sites), figsize=(3.9 * len(sites), 4.6))
    axes = np.atleast_1d(axes)
    drew = False
    for ax, (site, info) in zip(axes, sites):
        hub = info["hub"]
        try:
            H = xseq.levelled(hub, run, quiet=True, coarse_only=True)
        except SystemExit:
            ax.axis("off")
            continue
        xy = H["v_xyz"][:, :2]
        lo = np.percentile(xy, 0.5, axis=0)
        hi = np.percentile(xy, 99.5, axis=0)
        pad = 0.06 * (hi - lo).max()
        keep = ((xy >= lo) & (xy <= hi)).all(axis=1)
        ax.hist2d(xy[keep, 0], xy[keep, 1], bins=150, cmap=style.seq_cmap(t),
                  cmin=1, alpha=0.5, rasterized=True)
        log = dataset.pos_log(hub, run)
        if os.path.isfile(log):
            p_hub = np.loadtxt(log)[:, 4:7] @ np.array(H["R"]).T
            ax.plot(p_hub[:, 0], p_hub[:, 1], color=t["series"][0], lw=2.0, zorder=4,
                    label="the hub map's own route")
        first = True
        for seq in info.get("others", []):
            rec = (data["cross"].get(f"{seq}_in_{hub}") or {}).get("register")
            log = dataset.pos_log(seq, run)
            if not rec or not os.path.isfile(log):
                continue
            q = np.loadtxt(log)[:, 4:7] @ np.array(rec["R_align"]).T + np.array(rec["t_align"])
            ax.plot(q[:, 0], q[:, 1], color=t["series"][1], lw=1.1, alpha=0.85, zorder=3,
                    label="the other sequences, registered" if first else None)
            first = False
            drew = True
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_title(f"site {site} — map from {style.short(hub)}")
        ax.set_aspect("equal")
        ax.set_xlabel("x, m")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("y, m")
    if not drew:
        plt.close(fig)
        return None
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
               fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save(fig, out_dir, "fig_sites", t)


# --------------------------------------------------------------------------
def _err_panels(items, t, out_dir, name, title, ncol=3):
    """Small multiples of an error-vs-time curve, one panel per run."""
    items = [(lab, s) for lab, s in items if s and s.get("t")]
    if not items:
        return None
    style.apply(t)
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 1.9 * nrow),
                             squeeze=False, sharex=False)
    for ax, (lab, s) in zip(axes.ravel(), items):
        tt = np.asarray(s["t"], dtype=float)
        err = np.asarray(s["err"], dtype=float)
        ax.plot(tt, err, color=t["series"][0], lw=1.3)
        ax.axhline(1.0, color=t["critical"], lw=1.0, ls=(0, (4, 3)))
        ax.set_yscale("log")
        ax.set_ylim(0.005, max(3.0, float(err.max()) * 1.4))
        ax.set_title(lab, fontsize=8.5)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    # The x label belongs on the LAST PANEL OF EACH COLUMN that has data, not on the
    # bottom row: a partly filled final row leaves the columns to its right unlabelled.
    for col in range(ncol):
        last = [r for r in range(nrow) if r * ncol + col < len(items)]
        if last:
            axes[last[-1]][col].set_xlabel("bag time, s", fontsize=8)
    for row in axes:
        row[0].set_ylabel("error, m (log)", fontsize=8)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, color=t["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, out_dir, name, t)


def fig_cross(data, t, out_dir):
    items = [(f"{style.short(c['seq'])} in {style.short(c['hub'])}", c.get("mapframe"))
             for c in sorted(data["cross"].values(), key=lambda c: (c["site"], c["seq"]))]
    return _err_panels(items, t, out_dir, "fig_cross",
                       "Cross-map runs: distance from the sequence's own FAST-LIO "
                       "trajectory\n(the 1 m line is where a slip stops being noise)")


def fig_own(data, t, out_dir):
    items = [(style.short(r["seq"]), r.get("mapframe")) for r in ordered(data)]
    return _err_panels(items, t, out_dir, "fig_own",
                       "Own-map runs: distance from the same sequence's FAST-LIO "
                       "trajectory", ncol=4)


# --------------------------------------------------------------------------
def fig_health(data, t, out_dir):
    """The two quantities the diagnostics threshold, over every own-map run.

    Both are plotted as distributions rather than time series: the question is not
    when the ratio dipped but whether the thresholds -- min_ray_inlier_ratio and
    max_pose_sigma_factor_increase -- have any margin left across the whole set.
    """
    rows = [r for r in ordered(data) if r.get("localizer")]
    if not rows:
        return None
    style.apply(t)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 0.30 * len(rows) + 1.8))
    y = np.arange(len(rows))
    ax = axes[0]
    for i, r in enumerate(rows):
        d = r["localizer"]
        lo, mid = d.get("inlier_p05"), d.get("inlier_median")
        if lo is None or mid is None:
            continue
        ax.plot([lo, mid], [i, i], color=t["axis"], lw=1.2, solid_capstyle="round")
        ax.scatter([lo], [i], s=20, color=t["series"][1], zorder=3,
                   edgecolors=t["surface"], linewidths=0.9)
        ax.scatter([mid], [i], s=26, color=t["series"][0], zorder=3,
                   edgecolors=t["surface"], linewidths=0.9)
    ax.set_yticks(y, [style.short(r["seq"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("ray inlier ratio")
    ax.set_title("Inlier ratio: 5th percentile → median")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=t["series"][1],
                              label="p05", markersize=5),
                       Line2D([], [], marker="o", ls="", color=t["series"][0],
                              label="median", markersize=5)],
              loc="upper right")

    ax = axes[1]
    med = [(r["localizer"].get("sigma_median") or np.nan) for r in rows]
    peak = [(r["localizer"].get("sigma_peak") or np.nan) for r in rows]
    for i, (m, p) in enumerate(zip(med, peak)):
        ax.plot([m, p], [i, i], color=t["axis"], lw=1.2, solid_capstyle="round")
    ax.scatter(med, y, s=26, color=t["series"][0], zorder=3,
               edgecolors=t["surface"], linewidths=0.9, label="median")
    ax.scatter(peak, y, s=26, color=t["series"][2], zorder=3, marker="^",
               edgecolors=t["surface"], linewidths=0.9, label="peak")
    ax.axvline(200, color=t["critical"], lw=1.0, ls=(0, (4, 3)))
    style.label(ax, 200, -0.45, " limit 200", t, color=t["critical"], dy=-2)
    ax.set_xscale("log")
    ax.set_yticks(y, ["" for _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("pose sigma factor increase (log)")
    ax.set_title("Covariance factor: median → peak")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, out_dir, "fig_health", t)


ALL = [fig_frames, fig_cp, fig_matrix, fig_sites, fig_cross, fig_own, fig_health]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    res = dataset.results_dir()
    out_dir = a.out or os.path.join(res, "rerun_report")
    data = json.load(open(a.data or os.path.join(out_dir, "data.json")))
    os.makedirs(out_dir, exist_ok=True)
    fns = [f for f in ALL if not a.only or f.__name__ in a.only]
    for fn in fns:
        for t in style.MODES:
            try:
                path = fn(data, t, out_dir)
            except Exception as exc:                        # one bad figure, not all
                print(f"  {fn.__name__} [{t['mode']}] FAILED: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            print(f"  {fn.__name__} [{t['mode']}] "
                  f"{'-> ' + os.path.basename(path) if path else 'skipped (no data)'}")


if __name__ == "__main__":
    main()
