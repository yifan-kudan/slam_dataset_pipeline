#!/usr/bin/env python3
"""Score the runs of a KdLocalizer campaign: status, accuracy, relocalization, resources.

    evaluate.py <campaign dir> --reference <dir>                 every run that has an out/ directory
    evaluate.py <campaign dir> --reference <dir> warm_s2 ...     only these runs
    evaluate.py <campaign dir>                                   no reference: no accuracy figures

Writes under <campaign dir>/eval/ (or --out):
    <name>.json        everything below for one run
    <name>.est.tum     the localizer's TRACKING poses, for evo's own tools, e.g.
                       evo_ape tum <reference>.tum eval/<name>.est.tum --sync_method interpolation -p
    summary.json       all runs;  summary.csv  one row per run (also printed)

Reference trajectories are the dataset's part of the work: one TUM file per sequence, in the
map frame of the run, each row stamped at the instant it describes on the clock of the
localizer's pose stamps (the bag's header stamps). Looked up as <reference>/<sequence>_in_<map>.tum,
then <reference>/<sequence>.tum. Orientation is only used for the relocalization heading check;
a reference with identity quaternions only (position-only ground truth) skips that check.

status      success (tracked to the end of the bag, never LOST), LOST events with reasons, time in
            odometry mode, first TRACKING pose
accuracy    evo: the reference is associated with the TRACKING poses within --t-max-diff and
            interpolated at their stamps (evo's interpolation sync); pairs whose interpolation
            spans a gap longer than --max-gap are dropped. Position error (APE, translation part)
            mean / median / 95 % / max / RMSE, horizontal and vertical parts, mean split by
            odometry mode, the time spent more than --off-limit away, and the figures before the
            first LOST. Accuracy is High when the maximum stays below --high-limit.
reloc       every attempt and its outcome; each published pose hint against the reference at the
            scan it came from: correct when within --correct-xy and --correct-yaw
processing  per-update time from the localizer's statistics, updates against scans played
resources   per-node CPU and memory (procmon), GPU load and power (tegrastats)
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kdloc import logs  # noqa: E402


def find_reference(reference_dir, sequence, map_name):
    if reference_dir is None:
        return None
    for name in (f"{sequence}_in_{map_name}.tum", f"{sequence}.tum"):
        path = Path(reference_dir) / name
        if path.exists():
            return path
    return None


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz.T
    return np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))


def error_stats(err):
    if len(err) == 0:
        return None
    return {"n": int(len(err)), "mean_cm": float(err.mean() * 100), "median_cm": float(np.median(err) * 100),
            "p95_cm": float(np.percentile(err, 95) * 100), "max_cm": float(err.max() * 100),
            "rmse_cm": float(np.sqrt(np.mean(err ** 2)) * 100)}


def time_weights(stamps, max_gap):
    """Seconds each scored pose stands for: the gap to the next one, at most `max_gap`.

    A skipped scan doubles a gap and still counts in full; a longer gap (LOST, not scored)
    counts only up to `max_gap`. The last pose stands for the median gap.
    """
    if len(stamps) < 2:
        return np.full(len(stamps), 0.0)
    gaps = np.diff(stamps)
    return np.minimum(np.append(gaps, np.median(gaps)), max_gap)


def accuracy(reference, poses, lost_events, bag_start, args):
    """Position error of the TRACKING poses against the reference, with evo doing the association."""
    from evo.core import metrics, sync
    from evo.core.trajectory import PoseTrajectory3D

    stamps = np.array([p[0] for p in poses])
    xyz = np.array([p[1] for p in poses])
    quat_wxyz = np.array([[p[2][3], p[2][0], p[2][1], p[2][2]] for p in poses])
    odometry = np.array([p[3] for p in poses], dtype=bool)
    est = PoseTrajectory3D(positions_xyz=xyz, orientations_quat_wxyz=quat_wxyz, timestamps=stamps)

    try:
        ref_s, est_s = sync.associate_trajectories(
            reference, est, sync_method=sync.SyncMethod.interpolation,
            max_diff=args.t_max_diff, offset_2=args.t_offset)
    except sync.SyncException:
        return None

    # evo interpolates the trajectory with more poses (the reference on a tie) at the other's
    # stamps. Drop pairs where that interpolation bridged a gap (a LOST stretch, a stall)
    # longer than --max-gap, or where it held the first or last pose for a time outside the
    # trajectory. A stamp that hits a pose exactly needs no interpolation and is always kept.
    t_est = est_s.timestamps
    if reference.num_poses >= est.num_poses:
        long_stamps, query = reference.timestamps, t_est + args.t_offset
    else:
        long_stamps, query = stamps, t_est
    i = np.clip(np.searchsorted(long_stamps, query, side="left"), 1, len(long_stamps) - 1)
    exact = (np.abs(long_stamps[i - 1] - query) < 1e-6) | (np.abs(long_stamps[i] - query) < 1e-6)
    inside = (query >= long_stamps[0]) & (query <= long_stamps[-1])
    keep = exact | (inside & ((long_stamps[i] - long_stamps[i - 1]) <= args.max_gap))
    ref_s.reduce_to_ids(np.flatnonzero(keep))
    est_s.reduce_to_ids(np.flatnonzero(keep))
    if est_s.num_poses == 0:
        return None
    if args.align == "se3":
        est_s.align(ref_s)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref_s, est_s))
    err = np.asarray(ape.error)
    t = est_s.timestamps
    delta = est_s.positions_xyz - ref_s.positions_xyz
    err_xy = np.linalg.norm(delta[:, :2], axis=1)
    in_odometry = odometry[np.clip(np.searchsorted(stamps, t, side="right") - 1, 0, len(stamps) - 1)]
    weights = time_weights(t, args.max_gap)

    out = {"reference_poses": int(reference.num_poses), "scored_poses": int(len(err)),
           **error_stats(err),
           "mean_xy_cm": float(err_xy.mean() * 100), "max_xy_cm": float(err_xy.max() * 100),
           "mean_abs_z_cm": float(np.abs(delta[:, 2]).mean() * 100),
           "mean_cm_against_map": float(err[~in_odometry].mean() * 100) if (~in_odometry).any() else None,
           "mean_cm_odometry_mode": float(err[in_odometry].mean() * 100) if in_odometry.any() else None,
           f"seconds_over_{args.off_limit:g}m": round(float(weights[err > args.off_limit].sum()), 1),
           "high_accuracy": bool(err.max() < args.high_limit), "aligned": args.align}
    if lost_events:
        before = t < bag_start + lost_events[0]["at_s"]
        out["before_first_lost"] = error_stats(err[before])
    series = {"t": np.round(t - bag_start, 2).tolist(), "err_cm": np.round(err * 100, 2).tolist()}
    return out, series


def reference_at(reference, t):
    """Reference position and yaw at bag time t (linear interpolation, clamped to its span)."""
    stamps = reference.timestamps
    pos = np.array([np.interp(t, stamps, reference.positions_xyz[:, k]) for k in range(3)])
    yaw = float(np.interp(t, stamps, np.unwrap(yaw_of(reference.orientations_quat_wxyz))))
    return pos, yaw


def relocalization(attempts, info, reference, yaw_checked, bag_start, first_tracking_stamp, args):
    out = {"attempts": len(attempts), "outcomes": logs.relocalizer_counts(attempts),
           "map_loaded": info["map_loaded"], "errors": info["errors"][:3],
           "no_lidar_frame": info["no_lidar_frame"], "results": []}
    for a in attempts:
        if a["outcome"] != "succeeded":
            continue
        result = {"at_s": round(a["cloud_stamp"] - bag_start, 2) if a.get("cloud_stamp") else None,
                  "trigger": a.get("trigger"), "score_pct": a.get("score_pct"),
                  "search_ms": a.get("search_ms"),
                  "points": a.get("points"), "hint": a.get("hint")}
        if reference is not None and a.get("hint") and a.get("cloud_stamp"):
            pos, yaw = reference_at(reference, a["cloud_stamp"] + args.hint_time_offset)
            hx, hy, hz, hyaw = a["hint"]
            dxy = math.hypot(hx - pos[0], hy - pos[1])
            dyaw = math.degrees((hyaw - yaw + math.pi) % (2 * math.pi) - math.pi)
            heading_ok = not yaw_checked or abs(dyaw) < args.correct_yaw
            result.update(error_xy_m=round(dxy, 3), error_yaw_deg=round(dyaw, 2) if yaw_checked else None,
                          error_z_m=round(hz - float(pos[2]), 3),
                          correct=bool(dxy < args.correct_xy and heading_ok))
        out["results"].append(result)
    graded = [r for r in out["results"] if "correct" in r]
    out["correct"] = sum(r["correct"] for r in graded) if graded else None
    out["wrong"] = sum(not r["correct"] for r in graded) if graded else None
    if attempts:
        first = attempts[0]
        out["first_attempt"] = {"at_s": round(first["cloud_stamp"] - bag_start, 2) if first.get("cloud_stamp")
                                else None, "outcome": first["outcome"], "trigger": first.get("trigger")}
        if first_tracking_stamp is not None and first.get("cloud_stamp"):
            out["first_attempt_to_tracking_s"] = round(first_tracking_stamp - first["cloud_stamp"], 2)
    out["first_result"] = out["results"][0] if out["results"] else None
    return out


def evaluate_run(run_dir, reference_dir, args):
    from evo.tools import file_interface

    env = logs.read_env(run_dir / "run.env")
    out_dir = run_dir / "out"
    limit = float(logs.read_text(out_dir / "limit_s").strip() or 0)
    bag = logs.bag_info(out_dir / "bag_metadata.yaml", limit)
    rows = logs.read_pose_csv(out_dir / "pose.csv")
    status = logs.status_summary(rows, bag["start"], bag["end"], args.end_tolerance)
    poses = logs.tracking_poses(rows)

    result = {"name": env.get("NAME", run_dir.name), "kind": env.get("KIND"), "sequence": env.get("SEQUENCE"),
              "map": env.get("MAP_NAME"), "limit_s": limit or None, "status": status}

    ref_path = find_reference(reference_dir, env.get("SEQUENCE"), env.get("MAP_NAME"))
    reference, yaw_checked, series = None, False, None
    if ref_path is not None:
        reference = file_interface.read_tum_trajectory_file(str(ref_path))
        identity = np.allclose(np.abs(reference.orientations_quat_wxyz[:, 0]), 1.0)
        yaw_checked = not identity
        result["reference"] = str(ref_path)
    result["accuracy"] = None
    if reference is not None and len(poses) >= 2:
        scored = accuracy(reference, poses, status["lost_events"], bag["start"], args)
        if scored is not None:
            result["accuracy"], series = scored

    attempts, info = logs.parse_relocalizer_log(out_dir / "reloc.log")
    result["relocalization"] = relocalization(attempts, info, reference, yaw_checked, bag["start"],
                                              poses[0][0] if poses else None, args)

    processing = logs.parse_node_log(out_dir / "node.log")
    scans = bag["message_counts"].get(env.get("LIDAR_TOPIC"))
    if scans:
        processing["scans_played"] = scans
        if processing["updates"]:
            processing["updates_per_scan"] = round(processing["updates"] / scans, 4)
    result["processing"] = processing
    offset = logs.read_text(out_dir / "host_utc_offset").strip() or None
    tegra = logs.parse_tegrastats(out_dir / "tegra.log", offset)
    result["resources"] = {**logs.parse_procmon(out_dir / "proc.csv"),
                           "tegrastats": logs.tegrastats_summary(tegra)}
    return result, poses, series


SUMMARY_COLUMNS = ("name", "kind", "sequence", "map", "success", "lost_at_pct", "odometry_pct", "mean_cm",
                   "p95_cm", "max_cm", "accuracy", "over_limit_s", "first_result", "search_ms",
                   "tracking_from_s", "update_avg_ms", "updates_per_scan")


def summary_row(r, off_limit):
    s, a, rl, p = r["status"], r["accuracy"] or {}, r["relocalization"], r["processing"]
    odometry = s["odometry_mode_share"]
    first = rl.get("first_result") or {}
    if not first:
        verdict = "none" if rl["attempts"] else ""
    elif "correct" in first:
        verdict = "correct" if first["correct"] else "wrong"
    else:
        verdict = "result"

    def fmt(value, digits=1):
        return "" if value is None else f"{value:.{digits}f}"

    return {"name": r["name"], "kind": r["kind"], "sequence": r["sequence"], "map": r["map"],
            "success": "yes" if s["success"] else "no",
            "lost_at_pct": fmt(100 * s["lost_events"][0]["at_share"], 0) if s["lost_events"] else "",
            "odometry_pct": fmt(None if odometry is None else 100 * odometry, 0),
            "mean_cm": fmt(a.get("mean_cm")), "p95_cm": fmt(a.get("p95_cm")), "max_cm": fmt(a.get("max_cm")),
            "accuracy": ("High" if a["high_accuracy"] else "Low") if a and s["success"] else "",
            "over_limit_s": fmt(a.get(f"seconds_over_{off_limit:g}m")),
            "first_result": verdict, "search_ms": fmt(first.get("search_ms"), 0),
            "tracking_from_s": fmt(s["first_tracking_s"]), "update_avg_ms": fmt(p.get("process_avg_ms")),
            "updates_per_scan": fmt(p.get("updates_per_scan"), 3)}


def print_table(rows):
    widths = {c: max(len(c), *(len(r[c] or "") for r in rows)) for c in SUMMARY_COLUMNS}
    print("  ".join(c.ljust(widths[c]) for c in SUMMARY_COLUMNS))
    for r in rows:
        print("  ".join((r[c] or "").ljust(widths[c]) for c in SUMMARY_COLUMNS))


def write_tum(path, poses):
    with open(path, "w") as f:
        for t, p, q, _ in poses:
            f.write(f"{t:.6f} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("names", nargs="*", help="runs to score (default: every run with an out/ directory)")
    parser.add_argument("--reference", type=Path, help="directory of reference TUM files")
    parser.add_argument("--out", type=Path, help="output directory (default <campaign dir>/eval)")
    parser.add_argument("--t-max-diff", type=float, default=0.05,
                        help="largest time difference (s) at which a reference and a localizer pose "
                             "are paired")
    parser.add_argument("--t-offset", type=float, default=0.0,
                        help="seconds added to the localizer's stamps to put them on the reference clock")
    parser.add_argument("--max-gap", type=float, default=0.3,
                        help="drop pairs whose interpolation spans a gap longer than this (s)")
    parser.add_argument("--align", choices=("none", "se3"), default="none",
                        help="se3: rigidly align the poses to the reference first (only for a reference "
                             "that is not in the map frame)")
    parser.add_argument("--high-limit", type=float, default=0.10,
                        help="High accuracy: max error below this (m)")
    parser.add_argument("--off-limit", type=float, default=1.0, help="count the time spent this far off (m)")
    parser.add_argument("--correct-xy", type=float, default=1.0,
                        help="correct relocalization: horizontal error below this (m)")
    parser.add_argument("--correct-yaw", type=float, default=10.0,
                        help="correct relocalization: heading error below this (deg)")
    parser.add_argument("--hint-time-offset", type=float, default=0.05,
                        help="the pose hint is compared with the reference at the scan stamp plus this (s)")
    parser.add_argument("--end-tolerance", type=float, default=1.0,
                        help="tracked to the end: last TRACKING pose within this many s of the bag end")
    args = parser.parse_args(argv)

    campaign_dir = args.campaign_dir.resolve()
    out_dir = (args.out or campaign_dir / "eval").resolve()
    names = args.names or sorted(p.parent.name for p in (campaign_dir / "runs").glob("*/out"))
    if args.reference is not None and not args.reference.is_dir():
        sys.exit(f"error: reference directory {args.reference} does not exist")
    out_dir.mkdir(parents=True, exist_ok=True)

    results, rows = [], []
    for name in names:
        run_dir = campaign_dir / "runs" / name
        if not (run_dir / "out" / "bag_metadata.yaml").exists():
            print(f"skip {name}: no out/bag_metadata.yaml (not run yet?)")
            continue
        result, poses, series = evaluate_run(run_dir, args.reference, args)
        if series is not None:
            result["error_series"] = series
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))
        write_tum(out_dir / f"{name}.est.tum", poses)
        results.append({k: v for k, v in result.items() if k != "error_series"})
        rows.append(summary_row(result, args.off_limit))

    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        print_table(rows)
    print(f"{len(rows)} runs -> {out_dir}")


if __name__ == "__main__":
    main()
