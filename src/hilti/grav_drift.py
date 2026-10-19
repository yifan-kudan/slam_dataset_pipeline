#!/usr/bin/env python3
"""How far FAST-LIO's gravity state and accel bias wandered during one run.

    grav_drift.py <pos_log.txt> [<pos_log.txt> ...] [--label name ...]

The point of the b_acc_cov experiment: with the stock 1e-4 accel-bias random walk
the ESKF can trade `ba` against the S2 gravity direction, and a slow rotation of
gravity tilts the whole map -- atan(0.32/9.807) = 1.87 deg hides in 0.32 m/s^2 of
bias.  So the two numbers to watch together are the gravity state's total rotation
and |ba|.  Columns follow dump_lio_state_to_log() in laserMapping.cpp:
  0 t | 1:4 rotvec | 4:7 pos | 10:13 vel | 19:22 ba | 22:25 grav
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--label", nargs="*", default=[])
a = ap.parse_args()

for k, path in enumerate(a.logs):
    raw = np.loadtxt(path)
    t, ba, grav = raw[:, 0], raw[:, 19:22], raw[:, 22:25]
    # The gravity state is initialised from the first ~20 IMU samples and settles
    # inside a second; measure drift from after that, not from sample 0.
    i0 = int(np.searchsorted(t, t[0] + 1.0))
    u = grav / np.linalg.norm(grav, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(u @ u[i0], -1, 1)))
    axis = np.cross(u[i0], u[-1])
    n = np.linalg.norm(axis)
    label = a.label[k] if k < len(a.label) else path
    print(f"=== {label}")
    print(f"  samples {len(raw)}  t {t[0]:.1f}-{t[-1]:.1f} s  |g| {np.linalg.norm(grav[i0]):.4f} "
          f"-> {np.linalg.norm(grav[-1]):.4f}")
    print(f"  gravity rotation from t+1s: end {ang[-1]:.3f} deg, max {ang.max():.3f} deg "
          f"at t+{t[np.argmax(ang)]:.1f}s")
    print(f"  rotation axis (map frame)   {(axis / n).round(3) if n > 1e-9 else axis}")
    # Where the rotation accumulates says whether it tracks a specific manoeuvre.
    q = np.searchsorted(t, t[0] + (t[-1] - t[0]) * np.array([0.25, 0.5, 0.75]))
    print("  cumulative deg at 25/50/75/100%%: "
          + " ".join(f"{ang[i]:.3f}" for i in list(q) + [len(ang) - 1]))
    nb = np.linalg.norm(ba, axis=1)
    print(f"  |ba| peak {nb.max():.4f} at t+{t[np.argmax(nb)]:.1f}s, "
          f"median {np.median(nb):.4f}, end {nb[-1]:.4f}")
    print(f"  ba end (map frame) {ba[-1].round(4)}")
