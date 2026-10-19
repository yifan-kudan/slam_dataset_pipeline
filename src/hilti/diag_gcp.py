#!/usr/bin/env python3
"""Break a control-point pair error into vertical and horizontal parts.

    diag_gcp.py site2_handheld_4 [site2_handheld_6 ...] [--run qosfix_r0.3]

A pair distance is one scalar, which says a run is wrong but not how. Splitting it
along FAST-LIO's own estimated gravity vector (pos_log's last three columns, the
filter's gravity state in the map frame) separates the two failure modes this
dataset produces, because the survey frame's z IS vertical:

* vertical wrong  -> scale or z drift, usually a stair climb mis-integrated
* horizontal wrong, vertical fine -> yaw.  In a stairwell the point-to-plane
  ROTATION information collapses (it decays as distance squared) while the
  translation block stays healthy, and gravity pins roll and pitch -- so yaw is
  the only DOF left exposed.  This is docs/hilti_2023_report.md S3.1.

Also prints the sampling gaps around each control point: a dropped-frame run dead
reckons across them on IMU alone, which is where yaw actually leaks.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_conslam_gt import anchor_time                  # noqa: E402
from hilti import dataset                                    # noqa: E402
from hilti.gcp import interp_pose, load_gcp, rotvec_to_quat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("seqs", nargs="+")
# Runs are not only "r<rate>": the QoS fix and the b_acc_cov experiment each got their
# own <bag>/runs/<name>/, so the directory is named directly and defaults to whatever
# config/hilti/sequences.yaml says this sequence's numbers come from.
ap.add_argument("--run", help="run dir under <bag>/runs, e.g. qosfix_r0.3")
ap.add_argument("--root", help="dataset root (default: datasets.hilti_2023.root)")
a = ap.parse_args()

for seq in a.seqs:
    bag = dataset.bag_dir(seq, a.root)
    run = a.run or dataset.sequence(seq)["run"]
    raw = np.loadtxt(dataset.pos_log(seq, run, a.root))
    rel, pos, grav = raw[:, 0], raw[:, 4:7], raw[:, 22:25]
    t0, _, _ = anchor_time(bag, rel, dataset.CLOUD_TOPIC)
    t = rel + t0
    tg, pg, lever = load_gcp(seq)
    p, R, ok = interp_pose(tg, t, pos, rotvec_to_quat(raw[:, 1:4]))
    tip = p + np.einsum("nij,nj->ni", R, lever)

    # Gravity is a filter state and settles within the first second; take it from
    # mid-run rather than the start. -g/|g| is up in the map frame.
    g = np.median(grav[len(grav) // 4:3 * len(grav) // 4], axis=0)
    up = -g / np.linalg.norm(g)
    dt = np.diff(t)
    print(f"\n=== {seq}  ({run}, {len(raw)} samples, up={up.round(4)}, |g|={np.linalg.norm(g):.3f})")
    print(f"{'pair':>9} {'gt_dist':>8} {'est_dist':>9} {'err':>7} | "
          f"{'gt_vert':>8} {'est_vert':>9} | {'gt_horiz':>9} {'est_horiz':>10}")
    i_, j_ = np.triu_indices(len(tg), 1)
    for i, j in zip(i_, j_, strict=False):
        if not (ok[i] and ok[j]):
            print(f"  {i}-{j}  skipped (outside the trajectory)")
            continue
        d, gd = tip[j] - tip[i], pg[j] - pg[i]
        dv = float(d @ up)
        dh = float(np.linalg.norm(d - dv * up))
        print(f"{i:4d}-{j:<4d} {np.linalg.norm(gd):8.3f} {np.linalg.norm(d):9.3f} "
              f"{np.linalg.norm(d) - np.linalg.norm(gd):+7.3f} | {gd[2]:+8.3f} {dv:+9.3f} | "
              f"{np.linalg.norm(gd[:2]):9.3f} {dh:10.3f}")
    for k, tq in enumerate(tg):
        m = (t > tq - 5) & (t < tq + 5)
        gaps = dt[m[:-1]]
        print(f"  cp{k} t+{tq - t[0]:7.1f}s  gt_z={pg[k, 2]:+7.3f}  "
              f"gaps within +-5 s: n={len(gaps)} max={gaps.max() if len(gaps) else 0:.2f}s "
              f"#>0.3s={(gaps > 0.3).sum() if len(gaps) else 0}")
