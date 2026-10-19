#!/usr/bin/env python3
"""Score one GLIM run on a Hilti 2023 sequence -- same three checks as eval_fastlio.py.

    eval_glim.py site2_handheld_5 <dump_dir> [--json out.json]

Reads `traj_imu.txt` (TUM, the globally optimised trajectory in the IMU frame, which
is the frame gcp.py's lever arms are expressed in) and, when present, `odom_imu.txt`
(odometry only, no global optimisation) so the two can be compared: a large gap
between them means loop closure moved the map, which FAST-LIO cannot do at all.

Control points come from gcp.py, identical to the FAST-LIO/kd_localizer numbers, so
the columns are directly comparable. Health gates are recomputed here rather than
reused because GLIM writes poses, not velocities: speed is a finite difference over
the trajectory, which is noisier than a filter state but catches the same failure.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilti import dataset            # noqa: E402
from hilti.gcp import score          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("seq")
ap.add_argument("dump", nargs="?", help="GLIM dump directory "
                                       "(default: <results>/glim/<seq>)")
ap.add_argument("--window", type=float, default=9.0)
ap.add_argument("--json")
ap.add_argument("--results")
a = ap.parse_args()
a.dump = a.dump or f"{dataset.results_dir(a.results)}/glim/{a.seq}"


def load(path):
    d = np.atleast_2d(np.loadtxt(path))
    d = d[np.argsort(d[:, 0])]
    return d[:, 0], d[:, 1:4], d[:, 4:8]

t, pos, quat = load(f"{a.dump}/traj_imu.txt")
rel = t - t[0]
keep = np.concatenate([[0], np.flatnonzero(np.diff(np.floor(rel)) > 0) + 1])
path_1hz = float(np.linalg.norm(np.diff(pos[keep], axis=0), axis=1).sum())
dt = np.diff(rel)
v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.where(dt > 0, dt, np.inf)
j = np.searchsorted(rel, rel + a.window).clip(0, len(rel) - 1)
disp = np.linalg.norm(pos[j] - pos, axis=1)
w = int(np.argmax(disp))

out = {"seq": a.seq, "dump": a.dump, "frames": len(t),
       "duration_s": round(float(rel[-1]), 1),
       "rate_hz": round(len(t) / float(rel[-1]), 2),
       "path_1hz_m": round(path_1hz, 1),
       "max_v": round(float(v.max()), 2),
       f"worst_{a.window:.0f}s_m": round(float(disp[w]), 1),
       f"worst_{a.window:.0f}s_at": round(float(rel[w]), 1),
       "extent_m": [round(float(np.ptp(pos[:, k])), 1) for k in range(3)]}
try:
    t_o, pos_o, _ = load(f"{a.dump}/odom_imu.txt")
    common = np.intersect1d(np.round(t, 6), np.round(t_o, 6))
    if len(common) > 10:
        ia = np.searchsorted(np.round(t, 6), common)
        ib = np.searchsorted(np.round(t_o, 6), common)
        d = np.linalg.norm(pos[ia] - pos_o[ib], axis=1)
        out["loop_shift_median_m"] = round(float(np.median(d)), 3)
        out["loop_shift_max_m"] = round(float(d.max()), 3)
except OSError:
    pass
out.update(score(a.seq, t, pos, quat))
print(json.dumps(out, indent=2))
if a.json:
    json.dump(out, open(a.json, "w"), indent=2)
