#!/usr/bin/env python3
"""Collect every number and every trajectory the Hilti report plots, into one JSON + NPZ.

Runs on the host (numpy only). Two things it has to get right, both inherited from
gcp.py / align_map.py:

* kd_localizer poses live in the GRAVITY-ALIGNED map frame (the .kdm was built from a
  cloud rotated by R_align), so up = +Z there. FAST-LIO's pos_log is still in its own
  map frame, where up = up_old. Decomposing a pair error into vertical/horizontal has
  to use the right one for each system.
* Control-point pair distances are frame-invariant, so those need no rotation at all.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate_conslam_gt import anchor_time                       # noqa: E402
from hilti import dataset                                          # noqa: E402
from hilti.gcp import interp_pose, load_gcp, rotvec_to_quat        # noqa: E402

RES = dataset.results_dir()
OUT = dataset.report_dir()

# Which run each sequence's numbers come from is config/hilti/sequences.yaml, not a
# table here: a map, its pos_log and the map frame they share all belong to one run,
# so pairing a map with another run's log would rotate the FAST-LIO reference by the
# wrong R_align. site2_handheld_6 is the standing example (r0.15, not r0.3).
SEQS = dataset.sequences()


def pair_table(tg, pg, tip, ok, up):
    """Per-pair gt/est distance plus the vertical/horizontal split along `up`."""
    rows = []
    i_, j_ = np.triu_indices(len(tg), 1)
    for i, j in zip(i_, j_, strict=False):
        if not (ok[i] and ok[j]):
            continue
        d, gd = tip[j] - tip[i], pg[j] - pg[i]
        dv = float(d @ up)
        dh = float(np.linalg.norm(d - dv * up))
        rows.append(dict(
            i=int(i), j=int(j),
            gt_dist=float(np.linalg.norm(gd)), est_dist=float(np.linalg.norm(d)),
            err=float(np.linalg.norm(d) - np.linalg.norm(gd)),
            gt_dz=float(gd[2]), est_vert=float(dv),
            gt_horiz=float(np.linalg.norm(gd[:2])), est_horiz=dh))
    return rows


def load_kd_csv(path):
    """TRACKING rows with a pose, deduplicated by pose stamp, plus the diagnostic series."""
    rows = list(csv.DictReader(open(path)))
    have = [r for r in rows if r.get("qw") and r["status"] == "TRACKING"]
    t = np.array([float(r["pose_stamp_s"]) for r in have])
    _, keep = np.unique(t, return_index=True)
    keep = np.sort(keep)
    pose = dict(
        t=t[keep],
        pos=np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in have])[keep],
        quat=np.array([[float(r[c]) for c in ("qx", "qy", "qz", "qw")] for r in have])[keep])
    # Non-frame ticks write 0 for both diagnostics (status 20 Hz, clouds 10 Hz).
    diag = [r for r in rows if r["status"] == "TRACKING" and float(r["sigma_factor"]) > 0]
    pose["diag_t"] = np.array([float(r["stamp_s"]) for r in diag])
    pose["inlier"] = np.array([float(r["ray_inlier_ratio"]) for r in diag])
    pose["sigma"] = np.array([float(r["sigma_factor"]) for r in diag])
    lost = [r for r in rows if r["status"] == "LOST"]
    pose["lost_t"] = float(lost[0]["stamp_s"]) if lost else None
    pose["lost_reason"] = lost[0]["lost_reason"] if lost else None
    return pose


meta, arrays, bags = {}, {}, []
for seq, entry in SEQS.items():
    plat, site, has_kdm = entry["platform"], entry["site"], entry["kdm"]
    run = entry["run"]
    bag = dataset.bag_dir(seq)
    raw = np.loadtxt(dataset.pos_log(seq, run))
    rel, fpos, grav = raw[:, 0], raw[:, 4:7], raw[:, 22:25]
    fvel = raw[:, 10:13]
    t0, anchor_err, n_frames = anchor_time(bag, rel, dataset.CLOUD_TOPIC)
    ft = rel + t0
    g = np.median(grav[len(grav) // 4:3 * len(grav) // 4], axis=0)
    up_fl = -g / np.linalg.norm(g)

    tg, pg, lever = load_gcp(seq)
    # Bag duration, not the FAST-LIO span: coverage is quoted as a share of the recording,
    # and FAST-LIO trims a little at each end while warming up.
    bag_meta = yaml.safe_load(open(f"{bag}/metadata.yaml"))["rosbag2_bagfile_information"]
    # `rate` is kept as the run label the per-sequence figure prints, which for the
    # r<rate> runs is exactly the playback rate.
    m = dict(seq=seq, platform=plat, site=site, has_kdm=has_kdm,
             run=run, rate=run.lstrip("r"),
             bag_duration_s=bag_meta["duration"]["nanoseconds"] / 1e9,
             bag_clouds=next(t["message_count"] for t in bag_meta["topics_with_message_count"]
                             if t["topic_metadata"]["name"] == dataset.CLOUD_TOPIC),
             bag_has_tf_static=any(t["topic_metadata"]["name"] == "/tf_static"
                                   for t in bag_meta["topics_with_message_count"]),
             up_fl=up_fl.tolist(), tilt_deg=float(np.degrees(np.arccos(np.clip(up_fl[2], -1, 1)))),
             duration_s=float(rel[-1] - rel[0]), frames_in_bag=int(n_frames),
             frames_processed=int(len(raw)), gcp_t_rel=(tg - ft[0]).tolist(),
             gcp_xyz=pg.tolist())

    # --- FAST-LIO
    p, R, ok = interp_pose(tg, ft, fpos, rotvec_to_quat(raw[:, 1:4]))
    fl_tip = p + np.einsum("nij,nj->ni", R, lever)
    m["fl_pairs"] = pair_table(tg, pg, fl_tip, ok, up_fl)
    m["fl_ok"] = ok.tolist()
    # The survey positions live in the SURVEY frame, which is unrelated to either map
    # frame -- that is the whole reason the metric is pair distances. So what a spatial
    # plot may show is the ESTIMATE at each control-point time, never the survey xyz.
    m["fl_tip"] = fl_tip.tolist()
    m["fl_tip_h"] = (fl_tip @ up_fl).tolist()
    # Body-origin height too: the continuous trajectory series is the body, while the
    # quantity comparable with the survey is the prism tip 0.2 m (handheld) / 0.77 m
    # (robot) away from it. Mixing the two shows up as a constant fake offset.
    m["fl_body_h"] = (p @ up_fl).tolist()
    arrays[f"{seq}__fl_t"] = rel
    arrays[f"{seq}__fl_pos"] = fpos
    arrays[f"{seq}__fl_speed"] = np.linalg.norm(fvel, axis=1)
    # Height along FAST-LIO's own gravity, so it is comparable with the survey z.
    arrays[f"{seq}__fl_h"] = fpos @ up_fl

    # --- kd_localizer
    csvp = dataset.monitor_csv(seq)
    if os.path.exists(csvp):
        kd = load_kd_csv(csvp)
        R_align = np.array(json.load(open(dataset.align_json(seq)))["R_align"])
        p, Rk, okk = interp_pose(tg, kd["t"], kd["pos"], kd["quat"])
        # kd poses are already levelled, so up is +Z.
        kd_tip = p + np.einsum("nij,nj->ni", Rk, lever)
        m["kd_pairs"] = pair_table(tg, pg, kd_tip, okk, np.array([0.0, 0.0, 1.0]))
        m["kd_ok"] = okk.tolist()
        m["kd_tip"] = kd_tip.tolist()
        m["kd_body_h"] = p[:, 2].tolist()
        m["kd_lost_t_rel"] = (kd["lost_t"] - ft[0]) if kd["lost_t"] else None
        m["kd_lost_reason"] = kd["lost_reason"]
        m["kd_tracked_s"] = float(kd["t"][-1] - kd["t"][0])
        arrays[f"{seq}__kd_t"] = kd["t"] - ft[0]
        arrays[f"{seq}__kd_pos"] = kd["pos"]
        arrays[f"{seq}__kd_diag_t"] = kd["diag_t"] - ft[0]
        arrays[f"{seq}__kd_inlier"] = kd["inlier"]
        arrays[f"{seq}__kd_sigma"] = kd["sigma"]
        # FAST-LIO reference rotated into the levelled map frame, for overlay on the map.
        arrays[f"{seq}__fl_pos_aligned"] = fpos @ R_align.T
        mf = f"{RES}/run_{seq}_mapframe.txt"
        if os.path.exists(mf):
            d = np.loadtxt(mf)
            arrays[f"{seq}__mf_t"] = d[:, 0] - ft[0]
            arrays[f"{seq}__mf_err"] = d[:, 1]
            m["mf"] = dict(n=len(d), median=float(np.median(d[:, 1])),
                           p95=float(np.percentile(d[:, 1], 95)), max=float(d[:, 1].max()),
                           over1m_pct=float(100 * (d[:, 1] > 1).mean()))
    # Carry the already-computed scorer output through unchanged. The FAST-LIO score has to
    # be the one for THIS sequence's rate, or handheld_6 would be reported against the r0.3
    # trajectory its map was rebuilt to replace.
    for kind, file in (("fl", dataset.score_json(seq, "fl", entry["score_tag"])),
                       ("kd", dataset.score_json(seq, "kd", ""))):
        if os.path.exists(file):
            m[f"{kind}_score"] = json.load(open(file))
    # The other run, kept for the playback-rate comparison in the reruns section.
    # Named in the sequence config so this stays true when a rerun replaces it.
    alt_run = entry.get("alt_run")
    if alt_run:
        alt = dataset.score_json(seq, "fl", entry.get("alt_score_tag", ""))
        if os.path.exists(alt):
            m["fl_score_alt"] = json.load(open(alt))
            m["run_alt"] = alt_run
            m["rate_alt"] = alt_run.lstrip("r")
    # The bag inventory table (section 02) and audit.py both read this.
    counts = {t["topic_metadata"]["name"]: t["message_count"]
              for t in bag_meta["topics_with_message_count"]}
    duration = bag_meta["duration"]["nanoseconds"] / 1e9
    bags.append(dict(
        seq=seq, dur=round(duration, 1),
        lidar=counts.get(dataset.CLOUD_TOPIC, 0),
        lhz=round(counts.get(dataset.CLOUD_TOPIC, 0) / duration, 3),
        imu=counts.get(dataset.IMU_TOPIC, 0),
        ihz=round(counts.get(dataset.IMU_TOPIC, 0) / duration, 1),
        # Storage files only: the bag directory also holds this run's maps/ and logs/.
        gb=round(sum(f.stat().st_size for f in Path(bag).iterdir()
                     if f.suffix in (".mcap", ".db3")) / 1e9, 2),
        tf="/tf_static" in counts,
        ver=bag_meta["version"]))

    meta[seq] = m
    print(f"{seq:22} {run:6} fl_pairs={len(m['fl_pairs']):3} "
          f"kd_pairs={len(m.get('kd_pairs', [])):3} tilt={m['tilt_deg']:6.2f} "
          f"dur={m['duration_s']:6.1f}")

os.makedirs(OUT, exist_ok=True)
json.dump(meta, open(f"{OUT}/data.json", "w"), indent=1)
json.dump(bags, open(f"{OUT}/bags.json", "w"), indent=1)
np.savez_compressed(f"{OUT}/traj.npz", **arrays)
print(f"\nwrote {OUT}/data.json, bags.json and traj.npz ({len(arrays)} arrays)")
