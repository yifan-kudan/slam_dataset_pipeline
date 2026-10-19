#!/usr/bin/env python3
"""Hilti 2023 ground-control-point scoring, shared by every evaluator here.

Hilti 2023 has no dense ground truth: each sequence ships 2-8 surveyed control
points (``groundtruth_2023/<seq>.txt``, ``t x y z qx qy qz qw`` with an identity
quaternion -- only the position means anything). So the metric is the **distance
between pairs of control points**, which needs no alignment at all and is
therefore immune to whatever frame the estimate lives in.

Two details decide whether the number is real (both established in the FAST-LIO
report, docs/hilti_2023_report.md S4):

1. The survey measured a **prism, not the IMU origin**. Handheld: a constant 20 cm
   boom, ``platforms.handheld.prism_in_imu``. Robot: ``<seq>_lidar.txt`` gives the
   prism in the **LiDAR frame** at each control point, pushed into the IMU frame
   with ``platforms.robot.lidar_from_imu``. Using the handheld arm on the robot
   invents ~1.5 m of error, which is why both live in config/hilti/platforms.yaml
   rather than as constants here.
2. The pose has to be **interpolated at the control-point time** (linear position
   + slerp attitude). Nearest-neighbour is up to 117 ms off once frames are
   dropped, which is 16 cm -- larger than the error being measured.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilti import dataset                                          # noqa: E402


def load_gcp(seq: str, gt_dir: str | None = None):
    """Return (times, survey_positions, lever_arms_in_imu_frame) for one sequence.

    `lever` is per control point for the robot (each row of ``<seq>_lidar.txt`` is
    the prism as that scan saw it) and a constant for the handheld rig.
    """
    gt_dir = gt_dir or dataset.locations()["groundtruth"]
    survey = os.path.join(gt_dir, f"{seq}.txt")
    if not os.path.isfile(survey):
        raise SystemExit(f"no control points for {seq}: {survey}\n"
                         f"  point datasets.hilti_2023.groundtruth in config/paths.yaml "
                         f"at the challenge's groundtruth_2023 directory")
    gt = np.atleast_2d(np.loadtxt(survey))
    t, p = gt[:, 0], gt[:, 1:4]

    lidar_file = os.path.join(gt_dir, f"{seq}_lidar.txt")
    if os.path.exists(lidar_file):
        lid = np.atleast_2d(np.loadtxt(lidar_file))
        if not np.allclose(lid[:, 0], t):
            raise SystemExit(f"{seq}: <seq>_lidar.txt timestamps differ from <seq>.txt")
        T = np.linalg.inv(dataset.lidar_from_imu(seq))
        lever = lid[:, 1:4] @ T[:3, :3].T + T[:3, 3]
    else:
        arm = dataset.lever_arm(seq)
        if arm is None:
            raise SystemExit(f"{seq}: platform {dataset.platform(seq)} has no constant "
                             f"prism offset and {lidar_file} is missing")
        lever = np.repeat(arm[None, :], len(t), axis=0)
    return t, p, lever


def interp_pose(t_query, t_est, pos_est, quat_est):
    """Linear position + slerp attitude at `t_query`.  Returns (pos, R, ok mask)."""
    ok = (t_query >= t_est[0]) & (t_query <= t_est[-1])
    j = np.clip(np.searchsorted(t_est, t_query), 1, len(t_est) - 1)
    t0, t1 = t_est[j - 1], t_est[j]
    w = np.where(t1 > t0, (t_query - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
    pos = pos_est[j - 1] + (pos_est[j] - pos_est[j - 1]) * w[:, None]
    q0, q1 = quat_est[j - 1].copy(), quat_est[j].copy()
    flip = (q0 * q1).sum(1) < 0
    q1[flip] *= -1
    dot = np.clip((q0 * q1).sum(1), -1, 1)
    ang = np.arccos(dot)
    lin = ang < 1e-6
    s0 = np.where(lin, 1 - w, np.sin((1 - w) * ang) / np.where(lin, 1, np.sin(ang)))
    s1 = np.where(lin, w, np.sin(w * ang) / np.where(lin, 1, np.sin(ang)))
    q = q0 * s0[:, None] + q1 * s1[:, None]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return pos, quat_to_mat(q), ok


def quat_to_mat(q):
    """(N,4) xyzw -> (N,3,3)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rotvec_to_quat(v):
    """(N,3) so(3) log -> (N,4) xyzw.  FAST-LIO's pos_log stores rotations this way."""
    ang = np.linalg.norm(v, axis=1)
    small = ang < 1e-9
    axis = v / np.where(small, 1.0, ang)[:, None]
    s = np.sin(ang / 2)
    return np.column_stack([axis * s[:, None], np.cos(ang / 2)])


def score(seq, t_est, pos_est, quat_est, gt_dir: str | None = None):
    """Control-point pair-distance error for one estimated trajectory.

    Out-of-range control points are dropped rather than failing the sequence:
    FAST-LIO's first output is always a few tenths of a second after the bag
    head (warmup), and some sequences put a control point inside that window.
    """
    t_g, p_g, lever = load_gcp(seq, gt_dir)
    pos, R, ok = interp_pose(t_g, t_est, pos_est, quat_est)
    tip = pos + np.einsum("nij,nj->ni", R, lever)
    tip, p_g, n_all = tip[ok], p_g[ok], len(t_g)
    out = {"seq": seq, "gcp_total": n_all, "gcp_used": int(ok.sum()),
           "gcp_span_s": round(float(t_g[-1] - t_g[0]), 1)}
    if ok.sum() < 2:
        out["pairs"] = 0
        return out
    i, j = np.triu_indices(len(tip), 1)
    d_est = np.linalg.norm(tip[i] - tip[j], axis=1)
    d_gt = np.linalg.norm(p_g[i] - p_g[j], axis=1)
    err = np.abs(d_est - d_gt)
    out.update(pairs=len(err),
               gt_dist_min=round(float(d_gt.min()), 1), gt_dist_max=round(float(d_gt.max()), 1),
               err_mean_cm=round(float(err.mean()) * 100, 1),
               err_max_cm=round(float(err.max()) * 100, 1),
               err_rel_pct=round(float((err / d_gt).mean()) * 100, 3))
    return out
