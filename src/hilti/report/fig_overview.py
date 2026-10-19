#!/usr/bin/env python3
"""Cross-sequence figures: gravity tilt, the vertical-error finding, coverage, sigma headroom."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hilti import dataset                                             # noqa: E402
import style as S                                                     # noqa: E402

S.apply()
OUT = dataset.report_dir()
D = json.load(open(f"{OUT}/data.json"))
SEQS = list(D)

# site2_handheld_5's r0.3 FAST-LIO run reached 249 m/s and z = +2910 m. Its pair errors are
# 4 000 m and would flatten every real figure onto one log decade, so the vertical-vs-error
# figure excludes it and says so; it gets its own divergence figure instead.
BLOWUP = "site2_handheld_5"


def short(s):
    return s.replace("site", "s").replace("_handheld_", "·h").replace("_robot_", "·r")


def offscale(ax, xs, ys, xmax, color, marker):
    """Draw values past `xmax` as a caret on the edge rather than silently clipping them."""
    for x, y in zip(xs, ys, strict=False):
        if x > xmax:
            ax.plot([xmax * 0.93], [y], marker=marker, ms=6, color=color,
                    markeredgecolor=S.SURFACE, markeredgewidth=1.6,
                    clip_on=False, zorder=5)
            ax.annotate("", (xmax, y), (xmax * 0.86, y), arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.4, shrinkA=0, shrinkB=0))


# ----------------------------------------------------------------- 1. gravity tilt
def fig_tilt():
    order = sorted(SEQS, key=lambda s: D[s]["tilt_deg"])
    y = np.arange(len(order))
    tilt = np.array([D[s]["tilt_deg"] for s in order])
    isrobot = np.array([D[s]["platform"] == "robot" for s in order])

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for m, c, lab in ((isrobot, S.FL, "robot · XSens MTi-600 · mounted x-up"),
                      (~isrobot, S.KD, "handheld · Alphasense · mounted z-down")):
        ax.barh(y[m], tilt[m], height=0.62, color=c, label=lab,
                edgecolor=S.SURFACE, linewidth=2.0)
    ax.axvline(180, color=S.CRIT, lw=1.6, ls=(0, (4, 3)), zorder=5)
    ax.annotate("180° — the antipode of the 2-DOF sphere\nparameterisation, where the "
                "correction\ngradient vanishes",
                (181, 1.15), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.2, color=S.CRIT, weight="600")
    ax.axvline(0, color=S.AXIS, lw=0.8)
    for yy, t in zip(y, tilt, strict=False):
        S.label(ax, t, yy, f"{t:.2f}°", xytext=(5, -3), va="center")
    ax.set_yticks(y, [short(s) for s in order], fontsize=7.5)
    ax.set_xlim(0, 262)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("angle between the FAST-LIO map frame's true up and the +Z that "
                  "imu.cpp:139 assumes  (degrees)")
    ax.set_title("All 15 sequences violate the hard-coded gravity prior — by platform, not by chance")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.005, -0.13), ncols=2)
    S.tidy(ax)
    fig.savefig(f"{OUT}/fig_tilt.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------- 2. control-point error vs vertical separation
def fig_vertical():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                  gridspec_kw=dict(width_ratios=[1.22, 1], wspace=0.24))

    def pull(key):
        x, h, y = [], [], []
        for s in SEQS:
            if s == BLOWUP:
                continue
            for p in D[s].get(key, []):
                if abs(p["err"]) < 1e-9:
                    continue
                x.append(abs(p["gt_dz"]))
                h.append(p["gt_horiz"])
                y.append(abs(p["err"]) * 100)
        return np.array(x), np.array(h), np.array(y)

    for key, c, lab, mk in (("fl_pairs", S.FL, "FAST-LIO", "o"),
                            ("kd_pairs", S.KD, "kd_localizer", "D")):
        x, h, yv = pull(key)
        ax.scatter(x, yv, s=34, c=c, marker=mk, label=f"{lab}   n={len(x)} pairs",
                   linewidths=1.6, edgecolors=S.SURFACE, alpha=0.92, zorder=3)
        cross = x >= 0.2
        ax2.scatter(h[~cross], yv[~cross], s=28, c=c, marker=mk, linewidths=1.4,
                    edgecolors=S.SURFACE, alpha=0.3, zorder=2)
        ax2.scatter(h[cross], yv[cross], s=36, c=c, marker=mk, linewidths=1.6,
                    edgecolors=S.SURFACE, alpha=0.95, zorder=3)

    ax.axvspan(-0.01, 0.2, color="#eceae3", zorder=0)
    ax.axhline(20, color=S.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.set_xscale("symlog", linthresh=0.2, linscale=0.5)
    ax.set_yscale("log")
    ax.set_xlim(-0.008, 12)
    ax.set_ylim(4e-3, 900)
    ax.set_xticks([0, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                  ["0", "0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("surveyed vertical separation of the pair  |Δz|  (m)")
    ax.set_ylabel("control-point pair-distance error  (cm, log)")
    ax.set_title("Error is set by vertical separation")
    S.label(ax, 0.5, 21, "20 cm", xytext=(0, 3))
    ax.annotate("same floor\n|Δz| < 0.2 m", (0.075, 0.03), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=7.2, color=S.INK2, weight="600")
    ax.annotate("crosses a floor →", (0.99, 0.03), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=7.2, color=S.INK2, weight="600")
    ax.legend(loc="upper left", bbox_to_anchor=(0.20, 1.005))
    S.tidy(ax)

    ax2.set_yscale("log")
    ax2.set_ylim(4e-3, 900)
    ax2.set_xlim(0, 64)
    ax2.set_xlabel("surveyed horizontal baseline of the pair  (m)")
    ax2.set_ylabel("control-point pair-distance error  (cm, log)")
    ax2.set_title("…not by horizontal baseline: no trend")
    ax2.annotate("solid  = crosses a floor\nfaded  = same floor", (0.97, 0.04),
                 xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=7.2, color=S.INK2, weight="600")
    S.tidy(ax2)
    fig.savefig(f"{OUT}/fig_vertical.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- 3. coverage + accuracy
def fig_coverage():
    rows = []
    for s in SEQS:
        kd = D[s].get("kd_score", {})
        fl = D[s].get("fl_score", {})
        trk = kd.get("tracked_s")
        rows.append(dict(seq=s, pct=(100 * trk / D[s]["bag_duration_s"]) if trk else 0.0,
                         zero_lost=bool(trk) and not kd.get("first_lost"),
                         kd_cm=kd.get("err_mean_cm") if kd.get("pairs", 0) >= 2 else None,
                         kd_pairs=kd.get("pairs", 0), fl_pairs=fl.get("pairs", 0),
                         fl_cm=fl.get("err_mean_cm") if fl.get("pairs") else None))
    rows.sort(key=lambda r: (-r["pct"], r["seq"]))
    y = np.arange(len(rows))
    XMAX = 400.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.9), sharey=True,
                                  gridspec_kw=dict(width_ratios=[1, 1.06], wspace=0.06))
    pct = np.array([r["pct"] for r in rows])
    # The real criterion is whether a LOST ever fired, not a percentage threshold: the
    # zero-LOST runs sit at 98.9-99.9 %, the rest at 18-90 %, with nothing in between.
    col = [S.GOOD if r["zero_lost"] else (S.WARN if r["pct"] >= 60 else S.CRIT)
           for r in rows]
    ax.barh(y, pct, height=0.62, color=col, edgecolor=S.SURFACE, linewidth=2.0)
    for yy, (p, r) in zip(y, zip(pct, rows, strict=False), strict=False):
        txt = ("no map built" if p == 0 else
               f"{p:.1f}%  no LOST" if r["zero_lost"] else f"{p:.0f}%")
        S.label(ax, p, yy, txt, xytext=(5, -3), va="center")
    ax.set_xlim(0, 152)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(y, [short(r["seq"]) for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("share of the bag tracked  (%)")
    ax.set_title("kd_localizer coverage — green = a LOST never fired")
    ax.grid(axis="y", visible=False)
    S.tidy(ax)

    for i, r in enumerate(rows):
        vals = [(r["fl_cm"], S.FL, "o", "FAST-LIO"), (r["kd_cm"], S.KD, "D", "kd_localizer")]
        for v, c, mk, lab in vals:
            if v is None or v > XMAX:
                continue
            ax2.scatter([v], [i], s=44, c=c, marker=mk, linewidths=1.8,
                        edgecolors=S.SURFACE, zorder=4, label=lab if i == 0 else None)
        if r["fl_cm"] and r["kd_cm"] and max(r["fl_cm"], r["kd_cm"]) <= XMAX:
            ax2.plot([r["fl_cm"], r["kd_cm"]], [i, i], color=S.AXIS, lw=1.3, zorder=3)
        if r["kd_cm"] is None:
            note = ("no kd run — no map was built" if r["pct"] == 0 else
                    f"kd scored {r['kd_pairs']} pair{'' if r['kd_pairs'] == 1 else 's'}"
                    " — too few to quote")
            if r["seq"] == BLOWUP:
                note = "no kd run; FAST-LIO diverged — its pair error is 4 000 m, off scale"
            ax2.annotate(note, (0.015, i), xycoords=("axes fraction", "data"),
                         va="center", ha="left", fontsize=7.0,
                         color=S.MUTED, style="italic")
    ax2.axvline(10, color=S.AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax2.annotate("10 cm", (10, -0.75), xytext=(3, 0), textcoords="offset points",
                 fontsize=7.2, color=S.INK2, weight="600", va="center")
    ax2.set_xscale("log")
    ax2.set_xlim(0.8, XMAX)
    ax2.set_ylim(len(rows) - 0.4, -1.3)
    ax2.set_xticks([1, 10, 100], ["1", "10", "100"])
    ax2.set_xlabel("control-point pair-distance error, mean  (cm, log)")
    ax2.set_title("Accuracy against the survey — same bag, both systems")
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.99))
    S.tidy(ax2)
    fig.savefig(f"{OUT}/fig_coverage.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- 4. sigma headroom
def fig_sigma():
    rows = [(s, D[s]["kd_score"]["sigma_median"], D[s]["kd_score"]["sigma_peak"])
            for s in SEQS if D[s].get("kd_score", {}).get("sigma_peak")]
    rows.sort(key=lambda r: r[2])
    y = np.arange(len(rows))
    peak = np.array([r[2] for r in rows])
    med = np.array([r[1] for r in rows])

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.hlines(y, med, peak, color=S.AXIS, lw=1.4, zorder=2)
    ax.scatter(med, y, s=34, color="#9ec5f4", marker="D", linewidths=1.6,
               edgecolors=S.SURFACE, zorder=3, label="median while TRACKING")
    ax.scatter(peak, y, s=48, color=S.KD, marker="D", linewidths=1.8,
               edgecolors=S.SURFACE, zorder=4, label="peak while TRACKING")
    for lim, c, lab in ((20, S.MUTED, "library default  20"),
                        (200, S.CRIT, "configured  200")):
        ax.axvline(lim, color=c, lw=1.5, ls=(0, (4, 3)), zorder=1)
        ax.annotate(lab, (lim, len(rows) - 0.55), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.2, color=c, weight="600")
    for yy, p in zip(y, peak, strict=False):
        S.label(ax, p, yy, f"{p:.0f}", xytext=(6, -3), va="center")
    ax.set_xscale("log")
    ax.set_xlim(1, 620)
    ax.set_ylim(-0.7, len(rows) + 0.35)
    ax.set_yticks(y, [short(r[0]) for r in rows], fontsize=7.5)
    ax.set_xlabel("pose covariance factor — the quantity max_pose_sigma_factor_increase thresholds  (log)")
    ax.set_title("Headroom against the covariance limit: every robot and site3 run fits inside 200")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.005, -0.135), ncols=2)
    S.tidy(ax)
    fig.savefig(f"{OUT}/fig_sigma.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_tilt, fig_vertical, fig_coverage, fig_sigma):
        f()
        print("ok", f.__name__)
