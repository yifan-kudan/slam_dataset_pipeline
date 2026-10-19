#!/usr/bin/env python3
"""One figure per sequence: the map the localizer matched against, plus the three series
that decide whether a run is trustworthy.

Layout, left to right:
  * plan and section views of the .kdm plane voxels -- the geometry the localizer actually
    matches, not the raw cloud -- with both trajectories over it. The section view is the
    one that matters on this dataset: Hilti's error is vertical.
  * height, ray inlier ratio, pose covariance factor against bag time.

The map is drawn on a neutral ramp so the two trajectory hues stay readable over it
(emphasis, not categorical: the map is context).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hilti import dataset                                             # noqa: E402
import style as S                                                     # noqa: E402

S.apply()
RES = dataset.results_dir()
OUT = dataset.report_dir()
D = json.load(open(f"{OUT}/data.json"))
T = np.load(f"{OUT}/traj.npz")
PLANES = dataset.planes_dir()


def planes(seq):
    d = np.loadtxt(f"{PLANES}/{seq}_planes.csv", delimiter=",", skiprows=1,
                   usecols=(0, 1, 2, 5))
    return d[:, :3], np.abs(d[:, 3])


def draw_map(ax, xy, z, a, b, lab_a, lab_b):
    """Scatter the plane voxels on axes (a, b) coloured by height."""
    o = np.argsort(z)
    ax.scatter(xy[o, a], xy[o, b], c=z[o], cmap=S.MAPGRAY, s=0.55, linewidths=0,
               rasterized=True, zorder=1, alpha=0.85)
    ax.set_xlabel(f"{lab_a}  (m)")
    ax.set_ylabel(f"{lab_b}  (m)")
    ax.set_aspect("equal", adjustable="box")


def traj(ax, seq, a, b):
    """FAST-LIO underneath as a wide band, kd_localizer thin on top.

    The two normally agree to millimetres, so drawing them as two equal lines just hides
    one under the other. As a band plus a line, agreement reads as a blue line with an
    orange rim and any real disagreement reads as two separated tracks.
    """
    fl = T.get(f"{seq}__fl_pos_aligned")
    if fl is None:
        fl = T[f"{seq}__fl_pos"]
    ax.plot(fl[:, a], fl[:, b], color=S.FL, lw=4.2, alpha=0.5, zorder=3,
            solid_capstyle="round", label="FAST-LIO — built this map")
    if f"{seq}__kd_pos" in T:
        kd = T[f"{seq}__kd_pos"]
        ax.plot(kd[:, a], kd[:, b], color=S.KD, lw=1.6, zorder=4, label="kd_localizer")
        ax.plot([kd[0, a]], [kd[0, b]], marker="o", ms=6, color=S.KD,
                markeredgecolor=S.SURFACE, markeredgewidth=1.8, zorder=6)
    return fl


def gcps(ax, seq, a, b, labels=True):
    """Mark where the estimate was at each control-point time.

    The survey xyz cannot go on this plot: it lives in the survey frame, which has no
    known relation to either map frame -- hence the frame-invariant pair-distance metric.
    What is in map coordinates is the estimated prism position at those instants.
    """
    m = D[seq]
    tip = m.get("kd_tip") or m.get("fl_tip")
    ok = m.get("kd_ok") if m.get("kd_tip") else m.get("fl_ok")
    if tip is None:
        return
    p = np.array(tip)
    ok = np.array(ok, dtype=bool)
    ax.scatter(p[ok, a], p[ok, b], s=66, marker="*", c=S.GT, linewidths=1.6,
               edgecolors=S.SURFACE, zorder=7,
               label="map position at a control-point time")
    if labels:
        for k in np.flatnonzero(ok):
            S.label(ax, p[k, a], p[k, b], f"cp{k}", color="#12805a", xytext=(6, 3))


def one(seq):
    m = D[seq]
    fig = plt.figure(figsize=(11.4, 6.4))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.42, 1], height_ratios=[1.32, 1, 1],
                  wspace=0.2, hspace=0.62)
    ax_xy, ax_xz = fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[2, 0])
    ax_h, ax_i, ax_s = (fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]),
                        fig.add_subplot(gs[2, 1]))

    if m["has_kdm"]:
        xyz, nz = planes(seq)
        draw_map(ax_xy, xyz, xyz[:, 2], 0, 1, "map x", "map y")
        draw_map(ax_xz, xyz, xyz[:, 2], 0, 2, "map x", "map z (up)")
        npl = len(xyz)
    else:
        npl = 0
        for ax in (ax_xy, ax_xz):
            ax.set_aspect("equal", adjustable="box")
    for ax, a, b in ((ax_xy, 0, 1), (ax_xz, 0, 2)):
        fl = traj(ax, seq, a, b)
        gcps(ax, seq, a, b, labels=(ax is ax_xy))
        # A handful of stray voxels sit tens of metres outside the building; framing on the
        # trajectory keeps the site legible instead of squeezing it into a corner.
        pts = [fl[:, [a, b]]]
        if f"{seq}__kd_pos" in T:
            pts.append(T[f"{seq}__kd_pos"][:, [a, b]])
        q = np.vstack(pts)
        lo, hi = q.min(0), q.max(0)
        pad = np.maximum((hi - lo) * 0.18, 4.0)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax_xy.set_title(f"Plan view — {npl:,} plane voxels" if npl else "Plan view — no map built")
    ax_xz.set_title("Section view — the axis Hilti gets wrong")

    for ax in (ax_xy, ax_xz):
        S.tidy(ax)

    # --- height against bag time.
    # Three different vertical datums (survey, levelled map, FAST-LIO map) but all three
    # measure height along gravity, so subtracting each series' own value at the first
    # usable control point makes the CHANGES directly comparable. Absolute z is not.
    g = np.array(m["gcp_xyz"])
    gt_rel = np.array(m["gcp_t_rel"])
    ok_fl = np.array(m["fl_ok"], dtype=bool)
    ok_kd = np.array(m.get("kd_ok", m["fl_ok"]), dtype=bool)
    ref = int(np.flatnonzero(ok_fl | ok_kd)[0]) if (ok_fl | ok_kd).any() else 0

    # Continuous lines are the body origin (shape of the height profile); the round markers
    # are the prism tip at each control-point time, which is the thing the survey measured.
    # The vertical gap between an aqua marker and a coloured one IS that pair's vertical error.
    fl_t, fl_h = T[f"{seq}__fl_t"], T[f"{seq}__fl_h"]
    fl_body_h, fl_tip_h = np.array(m["fl_body_h"]), np.array(m["fl_tip_h"])
    ax_h.plot(fl_t, fl_h - fl_body_h[ref], color=S.FL, lw=3.6, alpha=0.5,
              label="FAST-LIO — built this map", solid_capstyle="round", zorder=2)
    ax_h.scatter(gt_rel[ok_fl], (fl_tip_h - fl_tip_h[ref])[ok_fl], s=30, marker="o",
                 c=S.FL, linewidths=1.5, edgecolors=S.SURFACE, zorder=5)
    if f"{seq}__kd_pos" in T and m.get("kd_tip"):
        kt, kp = T[f"{seq}__kd_t"], T[f"{seq}__kd_pos"]
        kb = np.array(m["kd_body_h"])
        ktip = np.array(m["kd_tip"])[:, 2]
        ax_h.plot(kt, kp[:, 2] - kb[ref], color=S.KD, lw=1.6, label="kd_localizer",
                  zorder=3)
        ax_h.scatter(gt_rel[ok_kd], (ktip - ktip[ref])[ok_kd], s=30, marker="o", c=S.KD,
                     linewidths=1.5, edgecolors=S.SURFACE, zorder=6)
    ax_h.scatter(gt_rel, g[:, 2] - g[ref, 2], s=52, marker="o", c=S.GT,
                 linewidths=1.6, edgecolors=S.SURFACE, zorder=7, label="survey")
    for k, (tt, hh) in enumerate(zip(gt_rel, g[:, 2] - g[ref, 2], strict=False)):
        S.label(ax_h, tt, hh, f"cp{k}", color="#12805a", xytext=(5, 5))
    ax_h.set_ylabel(f"height rel. cp{ref}  (m)")
    ax_h.set_title("Height along gravity — lines are the body, dots the surveyed prism")
    S.tidy(ax_h)

    # --- diagnostics
    if f"{seq}__kd_diag_t" in T:
        dt, inl, sig = (T[f"{seq}__kd_diag_t"], T[f"{seq}__kd_inlier"],
                        T[f"{seq}__kd_sigma"])
        ax_i.plot(dt, inl, color=S.KD, lw=1.3)
        ax_i.axhline(0.1, color=S.CRIT, lw=1.3, ls=(0, (4, 3)))
        S.label(ax_i, dt[0], 0.1, "library default 0.1 — disabled here",
                color=S.CRIT, xytext=(2, 4))
        ax_i.set_ylim(-0.03, 1.05)
        ax_i.set_ylabel("ray inlier ratio")
        ax_i.set_title("Ray inlier ratio")
        ax_s.plot(dt, sig, color=S.KD, lw=1.3)
        ax_s.axhline(200, color=S.CRIT, lw=1.3, ls=(0, (4, 3)))
        S.label(ax_s, dt[0], 200, "limit 200", color=S.CRIT, xytext=(2, 4))
        ax_s.axhline(20, color=S.MUTED, lw=1.1, ls=(0, (3, 3)))
        S.label(ax_s, dt[0], 20, "default 20", color=S.MUTED, xytext=(2, 4))
        ax_s.set_yscale("log")
        ax_s.set_ylim(0.5, max(400, sig.max() * 2.2))
        ax_s.set_ylabel("pose σ factor")
        ax_s.set_title("Pose covariance factor")
    else:
        for ax, t in ((ax_i, "no kd_localizer run"), (ax_s, "no kd_localizer run")):
            ax.annotate(t, (0.5, 0.5), xycoords="axes fraction", ha="center",
                        va="center", color=S.MUTED, style="italic", fontsize=8)
    ax_s.set_xlabel("bag time  (s)")

    lost = m.get("kd_lost_t_rel")
    for ax in (ax_h, ax_i, ax_s):
        if lost:
            ax.axvline(lost, color=S.CRIT, lw=1.4)
        ax.set_xlim(0, m["duration_s"])
        S.tidy(ax)
    if lost:
        S.label(ax_h, lost, ax_h.get_ylim()[1], f"LOST t+{lost:.0f}s",
                color=S.CRIT, xytext=(-4, -10), ha="right")

    handles, labels = [], []
    for ax in (ax_xy, ax_h):
        for hh, ll in zip(*ax.get_legend_handles_labels(), strict=False):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.005, -0.035),
               ncols=len(labels), fontsize=7.6)

    kd = m.get("kd_score", {})
    trk = kd.get("tracked_s")
    bits = [f"{seq}", f"{m['platform']} · site{m['site']} · FAST-LIO r{m['rate']}",
            f"{m['bag_duration_s']:.0f} s bag"]
    if trk:
        bits.append(f"tracked {trk:.1f} s ({100 * trk / m['bag_duration_s']:.0f}%)")
    if kd.get("pairs"):
        bits.append(f"control points {kd['err_mean_cm']:.1f} / {kd['err_max_cm']:.1f} cm"
                    f"  ({kd['pairs']} pairs)")
    if m.get("mf"):
        bits.append(f"vs FAST-LIO {m['mf']['median'] * 1000:.0f} mm median")
    fig.suptitle("   ·   ".join(bits), x=0.006, ha="left", fontsize=10.5,
                 weight="600", color=S.INK, y=1.005)
    fig.savefig(f"{OUT}/fig_{seq}.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("ok", seq, f"{npl:,} planes")


SKIP = {"site2_handheld_5"}   # see fig_divergence.py

if __name__ == "__main__":
    for s in (sys.argv[1:] or [q for q in D if q not in SKIP]):
        one(s)
