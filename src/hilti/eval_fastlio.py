#!/usr/bin/env python3
"""Score one FAST-LIO run on a Hilti 2023 sequence: control points + health gates.

    eval_fastlio.py site2_robot_1                       # the run config/hilti/sequences.yaml names
    eval_fastlio.py site2_robot_1 --run qosfix_r0.3 --json auto
    eval_fastlio.py site2_robot_1 --pos-log <path>      # anything, e.g. a scratch run

Three independent checks, because their blind spots differ (docs/hilti_2023_report.md S4):

* **Control-point pair distances** -- the only accuracy figure with survey backing.
  See gcp.py for the two things that have to be right (prism lever arm, interpolation).
* **Trajectory health gates** -- `max|v|`, peak `|ba|`, worst 9 s displacement.  Not
  academic metrics; divergence detectors thresholded for a walking/driving rig.
  A sequence can pass the control points and still have blown up after the last one.
* **Frame rate** -- how many of the bag's scans FAST-LIO actually processed. Below
  ~100 % the map is built from a fraction of the data; this host drops half of them
  without the Fast DDS profile batch_fastlio.py now exports by default.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_conslam_gt import anchor_time            # noqa: E402
from hilti import dataset                              # noqa: E402
from hilti.gcp import rotvec_to_quat, score            # noqa: E402

# pos_log.txt layout from dump_lio_state_to_log() in laserMapping.cpp
ROT, POS, VEL, BA = slice(1, 4), slice(4, 7), slice(10, 13), slice(19, 22)

ap = argparse.ArgumentParser()
ap.add_argument("seq")
ap.add_argument("--run", help="run under <bag>/runs/ to score "
                              "(default: this sequence's run in config/hilti/sequences.yaml)")
ap.add_argument("--bag", help="the ROS 2 bag the run played, for the time anchor "
                             "(default: <root>/<seq>_fastlio_bag)")
ap.add_argument("--pos-log", help="default: <run dir>/fastlio_log/pos_log.txt")
ap.add_argument("--root", help="dataset root (default: datasets.hilti_2023.root)")
ap.add_argument("--topic", default=dataset.CLOUD_TOPIC)
ap.add_argument("--window", type=float, default=9.0, help="health window, seconds")
ap.add_argument("--json", help="write the record here; `auto` uses "
                              "<results>/fl_<seq>[_<tag>].json")
ap.add_argument("--tag", help="suffix for --json auto (default: the sequence's score_tag)")
a = ap.parse_args()

bag = a.bag or dataset.bag_dir(a.seq, a.root)
pos_log = a.pos_log or dataset.pos_log(a.seq, a.run, a.root)
out_json = dataset.score_json(a.seq, "fl", a.tag) if a.json == "auto" else a.json

raw = np.loadtxt(pos_log)
rel, pos, vel, ba = raw[:, 0], raw[:, POS], raw[:, VEL], raw[:, BA]
t0, anchor_err, n_frames = anchor_time(bag, rel, a.topic)

# path length at a fixed 1 Hz: the raw ~5 Hz samples carry their own jitter, and
# comparing runs at different frame rates needs a rate-independent number.
keep = np.concatenate([[0], np.flatnonzero(np.diff(np.floor(rel)) > 0) + 1])
path_1hz = float(np.linalg.norm(np.diff(pos[keep], axis=0), axis=1).sum())

# worst displacement over any `window` seconds -- a walking rig cannot cover 25 m
# in 9 s, so this catches the slide-and-recover failures the control points miss.
j = np.searchsorted(rel, rel + a.window).clip(0, len(rel) - 1)
disp = np.linalg.norm(pos[j] - pos, axis=1)
worst = int(np.argmax(disp))

out = {"seq": a.seq, "pos_log": pos_log, "run": a.run or dataset.sequence(a.seq)["run"],
       "duration_s": round(float(rel[-1] - rel[0]), 1),
       "frames_processed": len(raw), "frames_in_bag": n_frames,
       "frame_pct": round(100 * len(raw) / n_frames, 1),
       "anchor_residual_ms": round(anchor_err * 1e3, 3),
       "path_1hz_m": round(path_1hz, 1),
       "max_v": round(float(np.linalg.norm(vel, axis=1).max()), 2),
       "ba_peak": round(float(np.linalg.norm(ba, axis=1).max()), 3),
       f"worst_{a.window:.0f}s_m": round(float(disp[worst]), 1),
       f"worst_{a.window:.0f}s_at": round(float(rel[worst]), 1)}
out.update(score(a.seq, rel + t0, pos, rotvec_to_quat(raw[:, ROT])))
print(json.dumps(out, indent=2))
if out_json:
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"wrote {out_json}", file=sys.stderr)
