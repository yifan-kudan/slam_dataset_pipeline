#!/usr/bin/env python3
"""Gather every artefact of the 2026-08-20 full-rate re-run into one data.json.

    collect.py [-o <results>/rerun_report/data.json]

Reads, and fails loudly on nothing -- a missing piece becomes a null in the record so
the report can say "not run" instead of quietly dropping a row:

    fl_<seq>_rerun.json          FAST-LIO score for the rerun_r1.0 run
    fl_<seq>[_<tag>].json        the pre-fix runs, for the frame-loss comparison
    align_<seq>.json             the map frame's gravity tilt
    kd_<seq>_rerun.json          kd_localizer against the sequence's OWN map
    xcov.json / xmat_site<N>.json / xhubs.json      coverage and hub choice
    xreg_<seq>_in_<hub>.json     the cross-sequence prior and its overlap
    kd_<seq>_in_<hub>.json       kd_localizer against the hub's map
    run_*.csv, run_*_mapframe.txt                   the per-run time series
    <kdm>/<seq>_fastlio_map.kdm                     map size on disk

The point of one file is that the figures and the tables cannot disagree: both read
this, and `rerun_report/build.py` regenerates it before it draws anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hilti import dataset                                          # noqa: E402

RUN = "rerun_r1.0"


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def bag_facts(seq):
    """Duration and cloud count straight out of the bag's own metadata."""
    import yaml
    meta = os.path.join(dataset.bag_dir(seq), "metadata.yaml")
    try:
        info = yaml.safe_load(open(meta))["rosbag2_bagfile_information"]
    except (OSError, KeyError, ValueError):
        return {}
    clouds = next((t["message_count"] for t in info["topics_with_message_count"]
                   if t["topic_metadata"]["name"] == dataset.CLOUD_TOPIC), None)
    return {"bag_duration_s": round(info["duration"]["nanoseconds"] / 1e9, 1),
            "bag_clouds": clouds,
            "bag_bytes": info.get("bag_size")}


def cloud_facts(seq):
    """Point count and extent of the cloud the map was built from, from the PCD header."""
    pcd = dataset.map_pcd(seq, RUN)
    out = {"pcd": pcd, "pcd_exists": os.path.isfile(pcd)}
    if out["pcd_exists"]:
        out["pcd_bytes"] = os.path.getsize(pcd)
        with open(pcd, "rb") as f:
            for _ in range(20):
                line = f.readline().decode("ascii", "replace").strip()
                if line.startswith("POINTS"):
                    out["pcd_points"] = int(line.split()[1])
                if line.startswith("DATA"):
                    break
    las = pcd.replace(".pcd", "_aligned.las")
    out["las_exists"] = os.path.isfile(las)
    out["las_bytes"] = os.path.getsize(las) if out["las_exists"] else None
    kdm = dataset.kdm(seq)
    out["kdm"] = kdm
    out["kdm_exists"] = os.path.isfile(kdm)
    out["kdm_bytes"] = os.path.getsize(kdm) if out["kdm_exists"] else None
    return out


def status_series(csv_path, stride=1):
    """The monitor CSV reduced to what a figure needs: t, status, inlier, sigma, xyz."""
    if not os.path.isfile(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    t0 = None
    out = {"t": [], "status": [], "inlier": [], "sigma": [], "x": [], "y": [], "z": []}
    for i, r in enumerate(rows):
        if i % stride:
            continue
        try:
            t = float(r["stamp_s"])
        except (KeyError, ValueError):
            continue
        t0 = t if t0 is None else t0
        out["t"].append(round(t - t0, 3))
        out["status"].append(r["status"])
        for key, col in (("inlier", "ray_inlier_ratio"), ("sigma", "sigma_factor"),
                         ("x", "x"), ("y", "y"), ("z", "z")):
            try:
                out[key].append(round(float(r[col]), 4))
            except (KeyError, ValueError, TypeError):
                out[key].append(None)
    return out


def mapframe_series(csv_path, stride=5):
    """The error-against-FAST-LIO curve eval_localizer.py wrote next to the CSV."""
    path = csv_path.replace(".csv", "_mapframe.txt")
    if not os.path.isfile(path):
        return None
    t, err, ref = [], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                continue
            if i % stride:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t.append(float(parts[0]))
            err.append(round(float(parts[1]), 4))
            ref.append([round(float(v), 3) for v in parts[2:5]])
    if not t:
        return None
    t0 = t[0]
    return {"t": [round(v - t0, 2) for v in t], "err": err, "ref_xyz": ref}


# The three arms of the experiment that found the IMU queue, and the file each one's score
# lives in. Written by hand because the arms are not sequences-under-test but
# configurations-under-test, and only the canonical one follows sequences.yaml.
ARMS = [
    ("mapfix", "-r 1.0, IMU queue 10",
     "the shared-memory and map-publish fixes only"),
    ("r03_v2", "-r 0.3, IMU queue 10",
     "same build, played at three-tenths speed"),
    ("rerun", "-r 1.0, IMU queue 2000",
     "the canonical run: both plus the deep IMU queue"),
]
ARM_SEQS = ["site1_handheld_1", "site1_handheld_5", "site2_handheld_4", "site2_handheld_5",
            "site2_handheld_6", "site3_handheld_2", "site2_robot_3"]


def rate_experiment(res):
    """The rate/queue comparison, so the report renders it instead of quoting it."""
    rows = []
    for seq in ARM_SEQS:
        row = {"seq": seq, "platform": dataset.platform(seq),
               "imu_hz": dataset.platform_config(seq)["imu_hz"]}
        for tag, _, _ in ARMS:
            d = load(f"{res}/fl_{seq}_{tag}.json")
            row[tag] = ({k: d.get(k) for k in ("frame_pct", "err_mean_cm", "err_max_cm",
                                               "err_rel_pct", "pairs")} if d else None)
        rows.append(row)
    return {"arms": [{"tag": t, "label": l, "note": n} for t, l, n in ARMS], "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out")
    ap.add_argument("--results")
    a = ap.parse_args()

    res = dataset.results_dir(a.results)
    out_dir = os.path.join(res, "rerun_report")
    os.makedirs(out_dir, exist_ok=True)
    out_path = a.out or os.path.join(out_dir, "data.json")

    seqs = dataset.sequences()
    data = {"run": RUN, "results": res, "kdm_dir": dataset.locations()["kdm"],
            "sequences": {}, "cross": {},
            "coverage": load(f"{res}/xcov.json"),
            "hubs": load(f"{res}/xhubs.json"),
            "matrix": {str(s): load(f"{res}/xmat_site{s}.json") for s in (1, 2, 3)},
            "rate_experiment": rate_experiment(res)}

    for seq, entry in seqs.items():
        rec = {"seq": seq, "site": entry["site"], "platform": entry["platform"],
               "alt_run": entry.get("alt_run"), "alt_score_tag": entry.get("alt_score_tag")}
        rec.update(bag_facts(seq))
        rec.update(cloud_facts(seq))
        rec["fastlio"] = load(dataset.score_json(seq, "fl", "rerun", a.results))
        # The pre-fix run of the same sequence: the frame-loss comparison rests on it.
        rec["fastlio_prefix"] = load(dataset.score_json(
            seq, "fl", entry.get("alt_score_tag") or "", a.results))
        rec["align"] = load(dataset.align_json(seq, a.results))
        rec["localizer"] = load(f"{res}/kd_{seq}_rerun.json")
        own_csv = f"{res}/run_{seq}.csv"
        rec["series"] = status_series(own_csv, stride=3)
        rec["mapframe"] = mapframe_series(own_csv)
        data["sequences"][seq] = rec

    hubs = data["hubs"] or {}
    for site, info in hubs.items():
        hub = info["hub"]
        for seq in info.get("others", []):
            name = f"{seq}_in_{hub}"
            csv_path = f"{res}/run_{name}.csv"
            data["cross"][name] = {
                "name": name, "hub": hub, "seq": seq, "site": int(site),
                "hub_platform": seqs[hub]["platform"],
                "seq_platform": seqs[seq]["platform"],
                "register": load(f"{res}/xreg_{name}.json"),
                # Two configurations were run on every pair: the canonical one, identical
                # to the own-map config so the two are comparable, and a second with the
                # gravity prior tightened and the covariance gate moved (sections 07, 09).
                "localizer": load(f"{res}/kd_{name}.json"),
                "localizer_v2": load(f"{res}/kd_{name}_v2.json"),
                "series": status_series(csv_path, stride=3),
                "mapframe": mapframe_series(csv_path),
                "mapframe_v2": mapframe_series(f"{res}/run_{name}_v2.csv")}

    with open(out_path, "w") as f:
        json.dump(data, f)
    n_fl = sum(1 for r in data["sequences"].values() if r["fastlio"])
    n_kd = sum(1 for r in data["sequences"].values() if r["localizer"])
    n_x = sum(1 for r in data["cross"].values() if r["localizer"])
    print(f"wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)\n"
          f"  fastlio {n_fl}/{len(seqs)}  own-map localizer {n_kd}/{len(seqs)}  "
          f"cross-map localizer {n_x}/{len(data['cross'])}")


if __name__ == "__main__":
    main()
