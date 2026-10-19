#!/usr/bin/env python3
"""Re-check the numbers written into the report prose against data.json.

The tables are generated, so they cannot drift. The prose is hand-written, so it can --
this asserts the specific figures quoted in it. Run after build.sh; a CHECK line means the
prose and the data disagree and one of them has to change.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hilti import dataset                                             # noqa: E402

R = dataset.results_dir()
D = json.load(open(f"{R}/report/data.json"))
T = np.load(f"{R}/report/traj.npz")
B = json.load(open(f"{R}/report/bags.json"))
bad = 0


def ck(name, got, claim):
    global bad
    ok = str(got) == str(claim)
    bad += not ok
    print(f"{'OK   ' if ok else 'CHECK'} {name:34} data={got}  prose={claim}")


ck("bag total s", round(sum(b["dur"] for b in B)), 3341)
ck("bag total clouds", sum(b["lidar"] for b in B), 33359)
ck("bag total imu", sum(b["imu"] for b in B), 1061139)
ck("bag total GB", round(sum(b["gb"] for b in B), 1), 44.3)
ck("bags without tf_static", sum(1 for b in B if not b["tf"]), 10)

tilt = [m["tilt_deg"] for m in D.values()]
ck("tilt range rounded", f"{round(min(tilt))}-{round(max(tilt))}", "88-180")
rob = [m["tilt_deg"] for m in D.values() if m["platform"] == "robot"]
hh = [m["tilt_deg"] for m in D.values() if m["platform"] == "handheld"]
ck("robot tilt range", f"{min(rob):.2f}-{max(rob):.2f}", "87.75-91.79")
ck("handheld tilt range", f"{min(hh):.2f}-{max(hh):.2f}", "177.08-179.72")

mf = [m["mf"]["median"] * 1000 for m in D.values() if m.get("mf")]
ck("vs FAST-LIO median range", f"{min(mf):.0f}-{max(mf):.0f}", "7-31")
ck("maps built", sum(1 for m in D.values() if m["has_kdm"]), 14)

runs = [m for m in D.values() if m.get("kd_score")]
zero = [m for m in runs if not m["kd_score"].get("first_lost")]
ck("kd runs", len(runs), 14)
ck("zero-LOST runs", len(zero), 7)
pcts = [100 * m["kd_score"]["tracked_s"] / m["bag_duration_s"] for m in zero]
ck("zero-LOST coverage", f"{min(pcts):.1f}-{max(pcts):.1f}", "98.9-99.9")
trk = sum(m["kd_score"]["tracked_s"] for m in runs)
dur = sum(m["bag_duration_s"] for m in D.values())
ck("total tracked s", round(trk), 2697)
ck("total tracked pct", round(100 * trk / dur), 81)

s3 = [D[f"site3_handheld_{i}"]["kd_score"]["err_mean_cm"] for i in (1, 2, 3, 4)]
ck("site3 accuracy range", f"{min(s3)}-{max(s3)}", "1.7-6.9")
s1 = [D[f"site1_handheld_{i}"]["kd_score"]["err_mean_cm"] for i in (1, 2, 3, 4)]
ck("site1 accuracy range", f"{min(s1)}-{max(s1)}", "1.2-4.1")

lost = [r for r in csv.DictReader(open(f"{R}/run_site2_handheld_6.csv"))
        if r["status"] == "LOST"][0]
ck("h6 LOST inlier", round(float(lost["ray_inlier_ratio"]), 2), 0.98)
h6 = D["site2_handheld_6"]
ck("h6 tracked s", h6["kd_score"]["tracked_s"], 43.1)
ck("h6 vs FL median mm", round(h6["mf"]["median"] * 1000), 16)
ck("h6 vs FL max mm", round(h6["mf"]["max"] * 1000), 152)

h4 = D["site2_handheld_4"]
ck("h4 tracked s", h4["kd_score"]["tracked_s"], 17.0)
ck("h4 sigma peak", h4["kd_score"]["sigma_peak"], 190.3)
ck("h4 FAST-LIO path", h4["fl_score"]["path_1hz_m"], 63.2)
ext = np.ptp(np.loadtxt(f"{R}/planes/site2_handheld_4_planes.csv", delimiter=",",
                        skiprows=1, usecols=(0, 1, 2)), axis=0)
ck("h4 map extent", " x ".join(f"{v:.1f}" for v in ext), "149.7 x 46.7 x 13.2")

sp, pos, t = (T["site2_handheld_5__fl_speed"], T["site2_handheld_5__fl_pos"],
              T["site2_handheld_5__fl_t"])
ck("h5 peak speed", round(float(sp.max())), 249)
ck("h5 end distance", round(float(np.linalg.norm(pos[-1] - pos[0]))), 9259)
ck("h5 divergence onset", round(float(t[np.argmax(sp > 3.0)]), 1), 113.4)

r1 = D["site2_robot_1"]
ck("robot_1 kd control pts", f'{r1["kd_score"]["err_mean_cm"]} / {r1["kd_score"]["err_max_cm"]}',
   "20.6 / 81.0")
ck("robot_1 FL control pts", f'{r1["fl_score"]["err_mean_cm"]} / {r1["fl_score"]["err_max_cm"]}',
   "28.9 / 102.3")
ck("robot_1 sigma peak", r1["kd_score"]["sigma_peak"], 41.14)
me, mt = T["site2_robot_1__mf_err"], T["site2_robot_1__mf_t"]
ck("robot_1 median pre-600s mm", round(float(np.median(me[mt < 600])) * 1000, 1), 22.7)
ck("robot_1 first over 1 m", round(float(mt[np.argmax(me > 1.0)]), 1), 634.0)

for seq, span in (("site2_robot_1", 5.660), ("site2_robot_2", 0.047),
                  ("site2_robot_3", 0.175)):
    g = np.array(D[seq]["gcp_xyz"])
    ck(f"{seq} control-point z span", f"{np.ptp(g[:, 2]):.3f}", f"{span:.3f}")

print(f"\n{bad} disagreement(s)")
sys.exit(1 if bad else 0)
