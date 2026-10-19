#!/usr/bin/env python3
"""The four figures that carry a specific argument rather than one sequence's summary."""
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
RES = dataset.results_dir()
OUT = dataset.report_dir()
D = json.load(open(f"{OUT}/data.json"))
T = np.load(f"{OUT}/traj.npz")


def short(s):
    return s.replace("site", "s").replace("_handheld_", "·h").replace("_robot_", "·r")


# ------------------------------------------- 1. site2_handheld_5: FAST-LIO divergence
def fig_divergence():
    seq = "site2_handheld_5"
    t, sp = T[f"{seq}__fl_t"], T[f"{seq}__fl_speed"]
    pos = T[f"{seq}__fl_pos"]
    dist = np.linalg.norm(pos - pos[0], axis=1)
    onset = float(t[np.argmax(sp > 3.0)])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 4.9), sharex=True,
                                 gridspec_kw=dict(hspace=0.22))
    a1.plot(t, sp, color=S.FL, lw=1.6)
    a1.axhline(3, color=S.AXIS, lw=1.0, ls=(0, (3, 3)))
    S.label(a1, t[-1], 3, "3 m/s — a walking rig's ceiling", xytext=(-4, 4), ha="right")
    a1.set_yscale("log")
    a1.set_ylabel("speed  |v|  (m/s, log)")
    a1.set_title("site2_handheld_5 · FAST-LIO r0.3 — divergence has a definite onset and never recovers")
    a2.plot(t, dist, color=S.FL, lw=1.6)
    a2.axhline(40, color=S.AXIS, lw=1.0, ls=(0, (3, 3)))
    S.label(a2, t[-1], 40, "40 m — the site's own extent", xytext=(-4, 4), ha="right")
    a2.set_yscale("log")
    a2.set_ylabel("distance from start  (m, log)")
    a2.set_xlabel("bag time  (s)")
    a2.set_xlim(0, t[-1])
    for ax in (a1, a2):
        ax.axvline(onset, color=S.CRIT, lw=1.5)
        S.tidy(ax)
    S.label(a1, onset, sp.min() * 2.2, f"t+{onset:.1f} s — first breach of 3 m/s\n"
                                       "the survey puts a ~2.75 m stair climb here",
            color=S.CRIT, xytext=(6, 0))
    a1.annotate(f"peak {sp.max():.0f} m/s", (t[np.argmax(sp)], sp.max()),
                xytext=(-6, -12), textcoords="offset points", ha="right",
                fontsize=7.6, weight="600", color=S.CRIT)
    a2.annotate(f"ends {dist[-1]:,.0f} m from the start", (t[-1], dist[-1]),
                xytext=(-6, -12), textcoords="offset points", ha="right",
                fontsize=7.6, weight="600", color=S.CRIT)
    fig.savefig(f"{OUT}/fig_divergence.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------- 2. site2_robot_1: the localizer refuses the odometry's jump
def fig_incident():
    seq = "site2_robot_1"
    m = D[seq]
    fl_t = T[f"{seq}__fl_t"]
    fl_h = T[f"{seq}__fl_h"] - np.array(m["fl_body_h"])[0]
    kt, kp = T[f"{seq}__kd_t"], T[f"{seq}__kd_pos"]
    kh = kp[:, 2] - np.array(m["kd_body_h"])[0]
    mt, me = T[f"{seq}__mf_t"], T[f"{seq}__mf_err"]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.0), sharex=True,
                                 gridspec_kw=dict(hspace=0.24, height_ratios=[1.25, 1]))
    a1.plot(fl_t, fl_h, color=S.FL, lw=3.6, alpha=0.55, label="FAST-LIO — built this map",
            solid_capstyle="round")
    a1.plot(kt, kh, color=S.KD, lw=1.7, label="kd_localizer")
    g = np.array(m["gcp_xyz"])
    gt = np.array(m["gcp_t_rel"])
    ktip = np.array(m["kd_tip"])[:, 2]
    a1.scatter(gt, g[:, 2] - g[0, 2] + (ktip[0] - g[0, 2]) * 0, s=52, marker="o", c=S.GT,
               linewidths=1.6, edgecolors=S.SURFACE, zorder=7, label="survey")
    for k, (tt, hh) in enumerate(zip(gt, g[:, 2] - g[0, 2], strict=False)):
        S.label(a1, tt, hh, f"cp{k}", color="#12805a", xytext=(5, 4))
    a1.set_ylabel("height rel. cp0  (m)")
    a1.set_title("site2_robot_1 — the odometry gains 3.3 m in 2.2 s; the localizer stays on the floor")
    a1.legend(loc="lower left", bbox_to_anchor=(0, -0.02), ncols=3)

    a2.plot(mt, me, color=S.KD, lw=1.4)
    a2.axhline(1.0, color=S.AXIS, lw=1.0, ls=(0, (3, 3)))
    S.label(a2, 8, 1.0, "1 m", xytext=(0, 4))
    a2.set_yscale("log")
    a2.set_ylim(1e-3, 8)
    a2.set_ylabel("| kd − FAST-LIO |  (m, log)")
    a2.set_xlabel("bag time  (s)")
    a2.set_xlim(0, fl_t[-1])
    first = float(mt[np.argmax(me > 1.0)])
    for ax in (a1, a2):
        ax.axvspan(629.8, 634.3, color="#f7e3e2", zorder=0)
        ax.axvline(first, color=S.CRIT, lw=1.5, zorder=5)
        S.tidy(ax)
    S.label(a2, first, 1.6, f"first exceeds 1 m at t+{first:.1f} s\n"
                            f"— inside the odometry's 632.4–634.3 s incident",
            color=S.CRIT, xytext=(-8, 0), ha="right")
    a2.annotate(f"median before t+600 s: {np.median(me[mt < 600]) * 1000:.1f} mm",
                (0.015, 0.90), xycoords="axes fraction", fontsize=7.6, weight="600",
                color=S.INK2, va="top")
    fig.savefig(f"{OUT}/fig_incident.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------- 3. what the localizer matches: the plane maps
def fig_maps():
    seqs = [s for s in D if D[s]["has_kdm"]]
    ncol, nrow = 5, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.4, 7.4))
    for ax in axes.ravel():
        ax.set_visible(False)
    for k, seq in enumerate(seqs):
        ax = axes.ravel()[k]
        ax.set_visible(True)
        d = np.loadtxt(f"{RES}/planes/{seq}_planes.csv", delimiter=",", skiprows=1,
                       usecols=(0, 1, 2, 5))
        nz = np.abs(d[:, 3])
        # Split the way the localizer's constraints split: slabs and ramps hold height,
        # walls and columns hold horizontal position.
        horiz, vert = nz > 0.85, nz < 0.35
        ax.scatter(d[vert, 0], d[vert, 1], s=0.5, c=S.KD, linewidths=0, rasterized=True,
                   alpha=0.55)
        ax.scatter(d[horiz, 0], d[horiz, 1], s=0.5, c=S.FL, linewidths=0, rasterized=True,
                   alpha=0.55)
        fl = T.get(f"{seq}__fl_pos_aligned", T[f"{seq}__fl_pos"])
        lo, hi = fl[:, :2].min(0), fl[:, :2].max(0)
        pad = np.maximum((hi - lo) * 0.22, 5.0)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{short(seq)} · {len(d):,} planes\n"
                     f"{100 * horiz.mean():.0f}% horiz / {100 * vert.mean():.0f}% vert",
                     fontsize=7.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(S.GRID)
    h = [plt.Line2D([], [], marker="o", ls="", ms=6, color=S.FL,
                    label="horizontal plane |n$_z$| > 0.85 — slab, ramp (holds height)"),
         plt.Line2D([], [], marker="o", ls="", ms=6, color=S.KD,
                    label="vertical plane |n$_z$| < 0.35 — wall, column (holds horizontal position)")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.01, 0.005), ncols=2,
               fontsize=8.2)
    fig.suptitle("The geometry each run actually matches against — plan view of every .kdm plane map",
                 x=0.01, ha="left", fontsize=11, weight="600", color=S.INK, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965), w_pad=1.8, h_pad=2.4)
    fig.savefig(f"{OUT}/fig_maps.png", bbox_inches="tight", dpi=125)
    plt.close(fig)


# ---------------------- 4. agreement with the trajectory that built each map, over time
def fig_agreement():
    seqs = [s for s in D if f"{s}__mf_err" in T]
    seqs.sort(key=lambda s: D[s]["mf"]["median"])
    fig, axes = plt.subplots(4, 4, figsize=(12.2, 6.6), sharey=True)
    for ax in axes.ravel():
        ax.set_visible(False)
    for k, seq in enumerate(seqs):
        ax = axes.ravel()[k]
        ax.set_visible(True)
        t, e = T[f"{seq}__mf_t"], T[f"{seq}__mf_err"]
        ax.plot(t, e, color=S.KD, lw=1.1)
        ax.axhline(1.0, color=S.CRIT, lw=1.0, ls=(0, (3, 3)))
        lost = D[seq].get("kd_lost_t_rel")
        if lost:
            ax.axvline(lost, color=S.CRIT, lw=1.3)
        ax.set_yscale("log")
        ax.set_ylim(2e-4, 300)
        ax.set_xlim(0, D[seq]["duration_s"])
        mf = D[seq]["mf"]
        ax.set_title(f"{short(seq)}   {mf['median'] * 1000:.0f} mm median · "
                     f"{mf['over1m_pct']:.1f}% over 1 m", fontsize=7.8)
        S.tidy(ax)
        if k % 4 == 0:
            ax.set_ylabel("| kd − FAST-LIO |  (m)")
        if k >= len(seqs) - 4:
            ax.set_xlabel("bag time  (s)")
    fig.suptitle("Agreement with the FAST-LIO trajectory that built each map — no alignment, "
                 "so any gap is real disagreement",
                 x=0.005, ha="left", fontsize=11, weight="600", color=S.INK, y=1.0)
    fig.text(0.005, -0.035, "red dashes = 1 m   ·   red rule = the first LOST   ·   "
             "the map shares this trajectory's frame exactly, so the two should coincide",
             fontsize=8, color=S.INK2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_agreement.png", bbox_inches="tight", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_divergence, fig_incident, fig_maps, fig_agreement):
        f()
        print("ok", f.__name__)
