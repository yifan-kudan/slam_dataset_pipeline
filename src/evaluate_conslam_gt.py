#!/usr/bin/env python3
"""Score a FAST-LIO run against ConSLAM's per-frame ground-truth poses.

ConSLAM ships ground truth as one ``<timestamp_us>.txt`` per keyframe, each a
4x4 SE(3) matrix, under ``<seq>/..._lidar_poses_normalized/normalized/``.
"Normalized" means the first pose is the identity, i.e. every pose is expressed
in the FIRST LiDAR frame rather than in the site survey frame that
``groundtruth_scan.ply`` uses. The poses are sparse (~0.7 Hz) and unevenly
spaced, so they are matched to the nearest estimate sample rather than
interpolated.

This is a separate entry point from ``evaluate_pose_csv_tum.py``: that one takes
ConPR's single ``pose.csv`` plus a TUM file, which is a different shape of input
entirely.

Two frame conventions have to be reconciled before any comparison:

  * ground truth is the LIDAR pose; FAST-LIO's state is the IMU ("body") pose in
    its own gravity-aligned ``camera_init`` frame. The two differ by a constant
    rigid transform (the LiDAR<->IMU extrinsic plus the initial attitude), which
    the Umeyama alignment absorbs -- so a nonzero ``extrinsic_T`` does not have
    to be known to score a run.
  * ``pos_log.txt`` timestamps are relative to ``first_lidar_time``, the first
    scan FAST-LIO actually synced (not the bag's first frame -- the IMU starts
    ~0.25 s late on ConSLAM, so several leading scans are skipped). The offset is
    recovered by testing each early bag frame as the anchor and keeping the one
    that lands every logged stamp on a real frame.

Reported both with and without scale: LiDAR-inertial odometry should already be
metric, so a Sim(3) scale far from 1.0 means something is wrong with the
per-point time or the IMU units rather than with the trajectory.

Example:
    python3 src/evaluate_conslam_gt.py \\
        /media/.../ConSLAM/seq5/seq5_lidar_poses_normalized/normalized \\
        ~/prj/dataset/ConSLAM/seq5_fastlio_bag \\
        ~/prj/dataset/ConSLAM/seq5_fastlio_bag/fastlio_log/pos_log.txt
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

# pos_log.txt column layout, from dump_lio_state_to_log() in laserMapping.cpp:
# t | rot_ang(3) | pos(3) | omega(3, zeroed) | vel(3) | acc(3, zeroed) |
# bg(3) | ba(3) | grav(3)
POS_COLS = slice(4, 7)

# Anchor search depth: how many leading bag frames to test as first_lidar_time.
# ConSLAM skips ~5 while the IMU warms up; 40 is slack well past that.
ANCHOR_SEARCH = 40


def load_gt(gt_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_sec, positions) from a directory of 4x4 pose files."""
    files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")),
                   key=lambda p: int(os.path.basename(p)[:-4]))
    if not files:
        raise SystemExit(f"no <timestamp>.txt pose files under {gt_dir}")
    stamps = np.array([int(os.path.basename(p)[:-4]) for p in files]) / 1e6
    poses = np.array([np.loadtxt(p) for p in files])
    if poses.shape[1:] != (4, 4):
        raise SystemExit(f"expected 4x4 matrices, got {poses.shape[1:]}")
    return stamps, poses[:, :3, 3]


def anchor_time(bag: str, est_rel: np.ndarray, topic: str) -> float:
    """Recover the absolute time that pos_log's t=0 corresponds to.

    FAST-LIO logs ``lidar_beg_time - first_lidar_time`` and never records
    first_lidar_time itself, so it is solved for: each candidate anchor implies
    a shift, and the right one puts every logged stamp back onto a real frame.
    """
    with AnyReader([Path(bag)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise SystemExit(f"topic {topic} not in {bag}")
        frames = np.array([ts for _, ts, _ in reader.messages(connections=conns)]) / 1e9
    best_t0, best_err = None, np.inf
    for k in range(min(ANCHOR_SEARCH, len(frames))):
        t0 = frames[k] - est_rel[0]
        idx = np.clip(np.searchsorted(frames, est_rel + t0), 1, len(frames) - 1)
        err = np.minimum(np.abs(est_rel + t0 - frames[idx - 1]),
                         np.abs(est_rel + t0 - frames[idx])).mean()
        if err < best_err:
            best_t0, best_err = t0, err
    return best_t0, best_err, len(frames)


def umeyama(target: np.ndarray, source: np.ndarray, with_scale: bool):
    """Least-squares similarity that maps ``source`` onto ``target``."""
    mt, ms = target.mean(0), source.mean(0)
    tc, sc = target - mt, source - ms
    u, sv, vt = np.linalg.svd(sc.T @ tc / len(target))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))   # reflection guard
    rot = (u @ d @ vt).T
    scale = (sv * np.diag(d)).sum() * len(target) / (sc ** 2).sum() if with_scale else 1.0
    return scale, rot, mt - scale * (rot @ ms)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt_dir", help="directory of <timestamp_us>.txt 4x4 pose files")
    ap.add_argument("bag", help="the ROS 2 bag the run played (for the time anchor)")
    ap.add_argument("pos_log", help="FAST-LIO pos_log.txt from that run")
    ap.add_argument("--topic", default="/velodyne_points", help="cloud topic in the bag")
    ap.add_argument("--max-dt", type=float, default=0.06, metavar="SEC",
                    help="pair a GT pose with the nearest estimate within this "
                         "window (default 0.06, over half a 10 Hz scan period)")
    ap.add_argument("--window", type=float, default=60.0, metavar="SEC",
                    help="bucket size for the error-over-time table (0 disables)")
    args = ap.parse_args(argv)

    gt_t, gt_p = load_gt(args.gt_dir)
    est = np.loadtxt(args.pos_log)
    est_rel, est_p = est[:, 0], est[:, POS_COLS]
    t0, anchor_err, n_frames = anchor_time(args.bag, est_rel, args.topic)
    print(f"anchor: pos_log t=0 -> bag time {t0:.6f}  (residual {anchor_err * 1e3:.3f} ms)")
    print(f"processed {len(est)} of {n_frames} frames in the bag "
          f"({len(est) / n_frames * 100:.1f}%)")

    est_abs = est_rel + t0
    pairs = [(i, j) for i, tg in enumerate(gt_t)
             if abs(est_abs[(j := int(np.argmin(np.abs(est_abs - tg))))] - tg) < args.max_dt]
    if len(pairs) < 3:
        raise SystemExit(f"only {len(pairs)} GT poses matched; check --max-dt / the anchor")
    gi = np.array([p[0] for p in pairs])
    ej = np.array([p[1] for p in pairs])
    print(f"matched {len(pairs)} of {len(gt_t)} GT poses (within {args.max_dt * 1e3:.0f} ms)")

    target, source = gt_p[gi], est_p[ej]
    for with_scale in (False, True):
        scale, rot, shift = umeyama(target, source, with_scale)
        err = np.linalg.norm(target - ((scale * (rot @ source.T)).T + shift), axis=1)
        label = f"Sim(3), scale {scale:.5f}" if with_scale else "SE(3)"
        print(f"\n{label}: ATE rmse={np.sqrt((err ** 2).mean()):.3f}  "
              f"mean={err.mean():.3f}  median={np.median(err):.3f}  max={err.max():.3f} m")
        if with_scale:
            continue
        # Path length is only comparable at equal sampling: the GT is ~0.7 Hz and
        # chords a curve, the estimate is ~6 Hz and follows it (plus its own
        # jitter), which alone inflates the estimate by several percent.
        sub = np.linalg.norm(np.diff(source, axis=0), axis=1).sum()
        gt_sub = np.linalg.norm(np.diff(target, axis=0), axis=1).sum()
        full = np.linalg.norm(np.diff(est_p, axis=0), axis=1).sum()
        print(f"  path length, matched samples: est {sub:.1f} m vs GT {gt_sub:.1f} m "
              f"({(sub / gt_sub - 1) * 100:+.1f}%)")
        print(f"  path length, every estimate sample: {full:.1f} m "
              f"(higher rate follows more of the curve -- not an error)")
        if args.window > 0:
            rel = gt_t[gi] - t0
            print(f"  ATE per {args.window:.0f}s:")
            for lo in np.arange(0, rel.max() + args.window, args.window):
                m = (rel >= lo) & (rel < lo + args.window)
                if m.any():
                    print(f"    {lo:5.0f}-{lo + args.window:5.0f}s  n={m.sum():3d}  "
                          f"median={np.median(err[m]):6.3f}  max={err[m].max():6.3f} m")


if __name__ == "__main__":
    main()
