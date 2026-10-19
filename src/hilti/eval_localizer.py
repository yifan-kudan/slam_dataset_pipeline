#!/usr/bin/env python3
"""Score one kd_localizer run on a Hilti 2023 sequence.

    eval_localizer.py site2_robot_1                    # everything from the config
    eval_localizer.py site2_robot_1 --csv <run.csv> --align auto --json auto

Two independent accuracy figures, because Hilti's ground truth is only 2-8
surveyed points per sequence:

* **Control-point pair distances** against the survey -- the same metric the
  FAST-LIO report used, so the two systems are directly comparable on this
  dataset.  Needs no alignment (see gcp.py).
* **Map-frame error against the sequence's own FAST-LIO trajectory.**  A .kdm
  built from that sequence's FAST-LIO cloud shares its frame exactly, so this
  comparison needs no alignment either -- and unlike the control points it is
  dense (~5-10 Hz over the whole run), which is what makes a silent slip
  visible.  It measures agreement with FAST-LIO, not truth: where the two
  disagree, the control points arbitrate.

Plus the diagnostics that decide whether a run is trustworthy at all: tracked
span, any LOST, the ray inlier ratio distribution and the pose sigma factor
peak (the quantity `max_pose_sigma_factor_increase` thresholds).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_conslam_gt import anchor_time            # noqa: E402
from hilti import dataset                              # noqa: E402
from hilti.gcp import score                            # noqa: E402

POS_COLS, ROT_COLS = slice(4, 7), slice(1, 4)

ap = argparse.ArgumentParser()
ap.add_argument("seq")
ap.add_argument("--csv", help="monitor_pose.py output (default: <results>/run_<seq>.csv)")
ap.add_argument("--run", help="FAST-LIO run to compare against (default: the sequence's)")
ap.add_argument("--fastlio-bag", help="default: <root>/<seq>_fastlio_bag")
ap.add_argument("--pos-log", help="default: <run dir>/fastlio_log/pos_log.txt")
ap.add_argument("--root")
ap.add_argument("--results", help="results directory (default: datasets.hilti_2023.results)")
ap.add_argument("--topic", default=dataset.CLOUD_TOPIC)
ap.add_argument("--max-dt", type=float, default=0.15)
ap.add_argument("--align", metavar="JSON|auto|none", default="auto",
                help="push the FAST-LIO reference into the map frame the .kdm was built "
                     "in. `auto` uses <results>/align_<seq>.json (align_map.py, a pure "
                     "rotation); an xseq.py cross-sequence JSON works here too and "
                     "carries a translation as well. Pair distances and the "
                     "control-point score are frame-invariant and need neither")
ap.add_argument("--level-base", action="store_true",
                help="the run reported poses in the gravity-levelled `blevel` frame "
                     "(src/hilti/gen_reloc.py), not imu_link. The origins coincide, so "
                     "positions need nothing; the ORIENTATION is right-multiplied by "
                     "R_align to recover the IMU attitude the surveyed-prism lever arm "
                     "is defined against.")
ap.add_argument("--no-mapframe", action="store_true",
                help="skip the comparison against FAST-LIO (control points only)")
ap.add_argument("--json", help="`auto` writes <results>/kd_<seq>.json")
a = ap.parse_args()

csv_path = a.csv or dataset.monitor_csv(a.seq, a.results)
out_json = (dataset.score_json(a.seq, "kd", "", a.results) if a.json == "auto" else a.json)
align_path = (dataset.align_json(a.seq, a.results) if a.align == "auto"
              else (None if a.align in ("none", "") else a.align))

rows = list(csv.DictReader(open(csv_path)))
have = [r for r in rows if r.get("qw") and r["status"] == "TRACKING"]
if not have:
    raise SystemExit(f"{csv_path}: no TRACKING rows with a pose")
# One row per pose stamp: the monitor emits on every status message, so a pose
# that arrives between two of them is written out twice.
t = np.array([float(r["pose_stamp_s"]) for r in have])
_, keep = np.unique(t, return_index=True)
keep = np.sort(keep)
t = t[keep]
pos = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in have])[keep]
quat = np.array([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]
                 for r in have])[keep]

if a.level_base:
    # R_map_imu = R_map_blevel @ T_blevel_imu = R_map_blevel @ R_align.
    from scipy.spatial.transform import Rotation as _Rot
    R_own = np.array(json.load(open(dataset.align_json(a.seq, a.results)))["R_align"])
    quat = (_Rot.from_quat(quat) * _Rot.from_matrix(R_own)).as_quat()

lost = [r for r in rows if r["status"] == "LOST"]
# kd-localizer-v2.0.0 no longer publishes ray_inlier_ratio (the monitor writes it empty), so
# the inlier stats are None for a 2.0 run rather than a crash on float("").
inl = np.array([float(r["ray_inlier_ratio"]) for r in rows
                if r["status"] == "TRACKING" and float(r["sigma_factor"]) > 0
                and r.get("ray_inlier_ratio")])
# ...and 2.0 adds odometry mode: the share of TRACKING status messages spent in it, and the
# longest distance any one odometry-mode stretch covered (LOST past max_odometry_distance).
trk = [r for r in rows if r["status"] == "TRACKING" and r.get("odometry_mode")]
odo = [r for r in trk if r["odometry_mode"] == "1"]
# TRACKING only: after a LOST the node keeps republishing the frozen factor, which
# would otherwise dominate the median of any run that was declared lost.
sig = np.array([float(r["sigma_factor"]) for r in rows
                if r["status"] == "TRACKING" and float(r["sigma_factor"]) > 0])

out = {"seq": a.seq, "csv": csv_path, "pose_rows": len(t),
       "tracked_s": round(float(t[-1] - t[0]), 1),
       "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                         for s in ("UNINIT", "TRACKING", "LOST")},
       "first_lost": (lost[0]["lost_reason"], lost[0]["stamp_s"]) if lost else None,
       "inlier_median": round(float(np.median(inl)), 3) if len(inl) else None,
       "inlier_p05": round(float(np.percentile(inl, 5)), 3) if len(inl) else None,
       "sigma_median": round(float(np.median(sig)), 2) if len(sig) else None,
       "sigma_p95": round(float(np.percentile(sig, 95)), 2) if len(sig) else None,
       "sigma_peak": round(float(sig.max()), 2) if len(sig) else None,
       "odometry_mode_pct": round(100 * len(odo) / len(trk), 2) if trk else None,
       "odometry_distance_max": (round(max(float(r["odometry_distance"] or 0) for r in odo), 2)
                                 if odo else (0.0 if trk else None))}
out.update(score(a.seq, t, pos, quat))

# --- map-frame agreement with the sequence's own FAST-LIO run (no alignment)
if not a.no_mapframe:
    bag = a.fastlio_bag or dataset.bag_dir(a.seq, a.root)
    pos_log = a.pos_log or dataset.pos_log(a.seq, a.run, a.root)
    raw = np.loadtxt(pos_log)
    t0, anchor_err, _ = anchor_time(bag, raw[:, 0], a.topic)
    ref_t, ref_p = raw[:, 0] + t0, raw[:, POS_COLS]
    if align_path:
        # The .kdm was built from a cloud rotated by R_align, so the localizer reports
        # poses in that frame while pos_log is still in FAST-LIO's own. Rotate the
        # reference, not the estimate, to keep the estimate exactly as published.
        al = json.load(open(align_path))
        R = np.array(al["R_align"])
        # A cross-sequence prior (src/hilti/xseq.py) is a full rigid transform, not just
        # a levelling: the map belongs to another sequence, so the reference has to be
        # translated into it too. `t_align` is absent for an own-map run, where the two
        # frames share an origin by construction.
        t_al = np.array(al.get("t_align", [0.0, 0.0, 0.0]))
        ref_p = ref_p @ R.T + t_al
    j = np.searchsorted(t, ref_t).clip(1, len(t) - 1)
    best = np.where(np.abs(t[j - 1] - ref_t) < np.abs(t[j] - ref_t), j - 1, j)
    ok = np.abs(t[best] - ref_t) < a.max_dt
    d = np.linalg.norm(pos[best[ok]] - ref_p[ok], axis=1)
    out.update(mapframe_pos_log=pos_log,
               mapframe_pairs=int(ok.sum()), mapframe_anchor_ms=round(anchor_err * 1e3, 3),
               mapframe_median=round(float(np.median(d)), 3),
               mapframe_p95=round(float(np.percentile(d, 95)), 3),
               mapframe_max=round(float(d.max()), 3),
               mapframe_over_1m_pct=round(100 * float((d > 1.0).mean()), 2))
    np.savetxt(csv_path.replace(".csv", "_mapframe.txt"),
               np.column_stack([ref_t[ok], d, ref_p[ok], pos[best[ok]]]),
               header="t err_m ref_x ref_y ref_z est_x est_y est_z", fmt="%.6f")

print(json.dumps(out, indent=2))
if out_json:
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"wrote {out_json}", file=sys.stderr)
