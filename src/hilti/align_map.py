#!/usr/bin/env python3
"""Compute the rotation that makes a FAST-LIO map frame gravity-aligned.

    align_map.py site2_robot_1              # writes <results>/align_<seq>.json + PDAL matrix
    align_map.py site2_robot_1 --run qosfix_r0.3

WHY THIS IS NEEDED.  kd_localizer's ESKF seeds its gravity state at the MAP FRAME's
-Z and never questions the choice:

    kd_amr_libs/localization/src/motion_model/imu.cpp:139
        prior.gravity_direction = -Vector3d::UnitZ();

with a 2x2 (2-DOF, on-sphere) covariance from `prior_gravity_sigma`.  A FAST-LIO map
frame, though, is the IMU frame at the first processed scan -- so it inherits however
the IMU was bolted to the rig.  Measured from the raw /imu stream and confirmed
against FAST-LIO's own gravity state:

    Hilti handheld (Alphasense)   up = [ 0.04,  0.01, -1.00]   map z-DOWN  -> 180 deg off
    Hilti robot    (XSens MTi-600) up = [ 1.00, -0.01,  0.01]   map x-UP    ->  90 deg off
    ConSLAM        (Xsens)                                       0.98-9.51 deg off

180 deg is the antipode of a 2-DOF sphere parameterisation, where the correction
gradient vanishes.  Both Hilti platforms therefore diverge with a perfectly good map
and a 0.87 ray inlier ratio: the plane alignment fixes each frame while the IMU
prediction mis-compensates up to 9.8 m/s^2 between them, until alignment loses.

THE FIX.  Rotate the map into a gravity-aligned frame (+Z up), which is what a
surveyed map would be anyway, and start the run from the pose that rotation implies:

    p_new = R_align @ p_old,   R_align @ up_old = +Z
    T_new_base(t0) = R_align   (the origin is fixed, so only the rotation changes)

so `initial_pose.rotation` becomes euler(R_align) instead of zero.  `rotation:` is
[roll, pitch, yaw] with R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -- kd_amr_libs/math/src/
rotation.cpp:18, reproduced and checked numerically below.

Control-point pair distances are unaffected by any of this: distances are
frame-invariant, and the lever arm is applied in the base frame either way.  Only a
direct comparison against FAST-LIO's own trajectory needs R_align applied first.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilti import dataset                                          # noqa: E402


def minimal_rotation(a, b):
    """Smallest rotation taking unit vector `a` to unit vector `b` (no extra yaw)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # Antipodal: any axis perpendicular to `a` gives a valid 180 deg rotation.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def to_euler(R):
    """Inverse of Rz(yaw) @ Ry(pitch) @ Rx(roll); returns [roll, pitch, yaw]."""
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw])


def from_euler(e):
    roll, pitch, yaw = e
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--run", help="run under <bag>/runs/ whose gravity state to read "
                                 "(default: this sequence's run in "
                                 "config/hilti/sequences.yaml)")
    ap.add_argument("--pos-log", help="read this log instead of deriving it from --run")
    ap.add_argument("--root", help="dataset root (default: datasets.hilti_2023.root)")
    ap.add_argument("--out", help="output directory (default: datasets.hilti_2023.results)")
    a = ap.parse_args()

    out_dir = dataset.results_dir(a.out)
    log = a.pos_log or dataset.pos_log(a.seq, a.run, a.root)
    grav = np.loadtxt(log)[:, 22:25]
    # Gravity is a filter state; take the mid-run median, not the settling start.
    g = np.median(grav[len(grav) // 4:3 * len(grav) // 4], axis=0)
    up_old = -g / np.linalg.norm(g)

    R = minimal_rotation(up_old, np.array([0.0, 0.0, 1.0]))
    euler = to_euler(R)
    # The euler round-trip is the thing the run depends on, so assert it here rather
    # than discover a convention mismatch as a mystery 90 deg offset in the results.
    assert np.allclose(from_euler(euler), R, atol=1e-9), "euler round-trip mismatch"
    assert np.allclose(R @ up_old, [0, 0, 1], atol=1e-9), "R_align does not level gravity"

    tilt = np.degrees(np.arccos(np.clip(up_old @ np.array([0.0, 0.0, 1.0]), -1, 1)))
    out = {"seq": a.seq, "run": a.run or dataset.sequence(a.seq)["run"],
           "pos_log": log,
           "gravity_map_frame": [round(float(x), 4) for x in g],
           "up_old": [round(float(x), 6) for x in up_old],
           "tilt_from_plus_z_deg": round(float(tilt), 2),
           "R_align": [[round(float(x), 12) for x in row] for row in R],
           "initial_pose_rotation_rpy": [round(float(x), 12) for x in euler],
           "initial_pose_rotation_rpy_deg": [round(float(np.degrees(x)), 3) for x in euler]}
    T = np.eye(4)
    T[:3, :3] = R
    out["pdal_matrix"] = " ".join(f"{v:.12f}" for v in T.reshape(-1))
    json.dump(out, open(f"{out_dir}/align_{a.seq}.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in
                      ("seq", "up_old", "tilt_from_plus_z_deg", "initial_pose_rotation_rpy_deg")},
                     indent=2))
    print(out["pdal_matrix"])


if __name__ == "__main__":
    main()
