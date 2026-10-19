#!/usr/bin/env python3
"""Score what kd_relocalizer_node's 3D-BBS search actually returned.

    eval_reloc.py site2_handheld_4_in_site2_robot_1_cold --map-seq site2_robot_1 --seq site2_handheld_4

The monitor CSV only sees a pose hint after the localizer has applied it, and by then a
wrong hint and a right one look the same until the track falls apart. The relocalizer's
own log carries the measurement: one line per attempt with the score, the elapsed time and
the pose it published. This compares each published pose against the reference for that
run -- identity for an own-map run, src/hilti/xseq.py's registration for a cross-sequence
one -- in the `blevel` frame both are expressed in.

Reported per run: attempts, how many cleared the threshold, search time, and for every
published hint its distance and yaw error from the reference. `score` is BBS's inlier
count and `pct` its fraction of the SOURCE cloud -- the pair matters, because a frame whose
points all sit inside a 3 m box reaches 100 % on 50 points against any dense map, which is
the shipped threshold's blind spot.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilti import dataset                                          # noqa: E402

OK = re.compile(r"BBS relocalization succeeded: score=(\d+) \(([\d.]+)%\), time=([\d.]+) ms")
HINT = re.compile(r"Published 3D pose hint x: ([-\d.]+), y: ([-\d.]+), z: ([-\d.]+), "
                  r"yaw: ([-\d.]+)")
# v1.1 logs "starting BBS relocalization"; kd-localizer-v2.0.0 logs "Received <status>
# localization status, starting relocalization (cloud stamp: ...)". Match either.
START = re.compile(r"starting (?:BBS )?relocalization")
VOX = re.compile(r"Voxel downsample \([\d.]+ m\): (\d+) -> (\d+) points")


def reference(seq, map_seq, results):
    """T_map_blevel(t0): where the rig actually starts, in the frame the hint is in."""
    R_own = np.array(json.load(open(dataset.align_json(seq, results)))["R_align"])
    if not map_seq:
        return np.zeros(3), np.eye(3)                # own map: level, at the origin
    x = json.load(open(f"{results or dataset.results_dir()}/xreg_{seq}_in_{map_seq}.json"))
    return np.array(x["t_align"]), np.array(x["R_align"]) @ R_own.T


def reference_path(seq, map_seq, results, root=None, run=None):
    """The sequence's own FAST-LIO trajectory, pushed into the map being localized against.

    A hint published mid-run cannot be scored against the START pose -- the rig has moved.
    What it can be scored against is the path: a hint 30 m off it is a wrong basin, a hint
    on it is a real place found at the wrong moment, which is what a stale search returns.
    """
    R_own = np.array(json.load(open(dataset.align_json(seq, results)))["R_align"])
    raw = np.loadtxt(dataset.pos_log(seq, run, root))
    t_rel = raw[:, 0]
    pos, R_fl = raw[:, 4:7], Rot.from_rotvec(raw[:, 1:4]).as_matrix()
    R_prior = R_own if not map_seq else np.array(json.load(open(
        f"{results or dataset.results_dir()}/xreg_{seq}_in_{map_seq}.json"))["R_align"])
    t_prior = np.zeros(3) if not map_seq else np.array(json.load(open(
        f"{results or dataset.results_dir()}/xreg_{seq}_in_{map_seq}.json"))["t_align"])
    # The attitude the hint should have had at that point of the path, in blevel terms:
    # R_map_blevel(t) = R_prior @ R_fastlio(t) @ R_own^T.
    yaw = Rot.from_matrix(R_prior @ R_fl @ R_own.T).as_euler("xyz")[:, 2]
    return pos @ R_prior.T + t_prior, yaw, t_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="run name, i.e. reloc_<name>.log in the results dir")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--map-seq", help="hub sequence; omit for an own-map run")
    ap.add_argument("--results")
    ap.add_argument("--root")
    ap.add_argument("--run", help="FAST-LIO run whose trajectory is the reference path")
    ap.add_argument("--json")
    a = ap.parse_args()

    res = dataset.results_dir(a.results)
    text = open(f"{res}/reloc_{a.name}.log").read()
    t_ref, R_ref = reference(a.seq, a.map_seq, a.results)
    yaw_ref = Rot.from_matrix(R_ref).as_euler("xyz")[2]

    # Interleaved in log order: a voxel line, then either a success + hint, or a failure.
    hints, scores, times, voxels = [], [], [], []
    for line in text.split("\n"):
        if m := VOX.search(line):
            voxels.append((int(m.group(1)), int(m.group(2))))
        if m := OK.search(line):
            scores.append((int(m.group(1)), float(m.group(2))))
            times.append(float(m.group(3)))
        if m := HINT.search(line):
            hints.append([float(m.group(i)) for i in range(1, 5)])

    out = {"name": a.name, "seq": a.seq, "map_seq": a.map_seq,
           "attempts": len(START.findall(text)),
           "succeeded": len(hints),
           "timed_out": text.count("BBS relocalization timed out"),
           "below_threshold": text.count("below threshold"),
           "no_lidar_frame": text.count("no lidar frame available"),
           "map_loaded": "Loaded BBS map" in text,
           "search_ms_min": round(min(times), 1) if times else None,
           "search_ms_max": round(max(times), 1) if times else None,
           "src_voxels_median": int(np.median([v[1] for v in voxels])) if voxels else None,
           "reference_t": [round(v, 3) for v in t_ref],
           "reference_yaw_deg": round(float(np.degrees(yaw_ref)), 2),
           "hints": []}
    path, path_yaw, path_t = reference_path(a.seq, a.map_seq, a.results, a.root, a.run)

    def _stale(i, ms):
        j = int(np.searchsorted(path_t, path_t[i] + ms / 1000.0))
        if j >= len(path):
            return float("nan")
        return np.linalg.norm(np.diff(path[i:j + 1], axis=0), axis=1).sum()

    for i, (h, (sc, pct), ms) in enumerate(zip(hints, scores, times)):
        dyaw = np.degrees((h[3] - yaw_ref + np.pi) % (2 * np.pi) - np.pi)
        d_path = np.linalg.norm(path - np.array(h[:3]), axis=1)
        out["hints"].append({
            "i": i, "score": sc, "pct": pct, "search_ms": round(ms, 1),
            "t": h[:3], "yaw_deg": round(float(np.degrees(h[3])), 2),
            "dist_m": round(float(np.linalg.norm(np.array(h[:3]) - t_ref)), 3),
            "path_dist_m": round(float(d_path.min()), 3),
            # Yaw against the attitude the rig actually had at the nearest path point: a
            # hint published mid-run cannot be judged against the yaw it had at t0.
            # How far the rig moved WHILE the search ran. path_dist_m asks "is this a real
            # place on the route" and cannot see *when*: a hint 3 cm off the route can be
            # metres along it from where the rig was by the time LocalizerNode applies it,
            # because pose_hint_callback drops header.stamp. This is that gap, taken from
            # the reference trajectory between the matched point and `search_ms` later.
            "stale_m": round(float(_stale(d_path.argmin(), ms)), 3),
            "path_dyaw_deg": round(float(np.degrees(
                (h[3] - path_yaw[int(d_path.argmin())] + np.pi) % (2 * np.pi) - np.pi)), 2),
            "dyaw_deg": round(float(dyaw), 2)})

    first = out["hints"][0] if out["hints"] else None
    out["first_hint_dist_m"] = first["dist_m"] if first else None
    out["first_hint_dyaw_deg"] = first["dyaw_deg"] if first else None
    # A hint is only usable if the localizer's initial_align can close the gap from it.
    # "Usable": close enough to a real point of the path that the localizer's initial_align
    # can take it. 1 m / 10 deg, not 0.5 m / 1 deg -- the search grid's finest level IS
    # 0.5 m, so demanding better than the grid would count nothing.
    st = [h["stale_m"] for h in out["hints"] if h["stale_m"] == h["stale_m"]]
    out["stale_m_median"] = round(float(np.median(st)), 3) if st else None
    out["hints_usable"] = sum(
        1 for h in out["hints"]
        if h["path_dist_m"] < 1.0 and abs(h["path_dyaw_deg"]) < 10.0)
    out["hints_on_path_1m"] = sum(1 for h in out["hints"] if h["path_dist_m"] < 1.0)
    out["hints_off_path_5m"] = sum(1 for h in out["hints"] if h["path_dist_m"] > 5.0)
    out["hints_within_2m_5deg"] = sum(
        1 for h in out["hints"] if h["dist_m"] < 2.0 and abs(h["dyaw_deg"]) < 5.0)

    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
