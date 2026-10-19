#!/usr/bin/env python3
"""
Evaluate a TUM trajectory against a ConPR pose.csv file.

pose.csv format:
  timestamp_ns,x,y,z

TUM format:
  timestamp_sec x y z qx qy qz qw

The script matches by nearest timestamp, aligns estimated positions to GT with
SE(3), and reports translation ATE statistics.
"""
import argparse
from pathlib import Path
from typing import Any

import numpy as np


def align_se3(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-align two already-corresponding point sets (Kabsch / Umeyama, no scale).

    ``src`` and ``dst`` are both (N, 3) and must already be in *matching order*:
    ``src[i]`` is the estimate paired with ground truth ``dst[i]`` (that pairing is
    what ``match_nearest`` produces). Returns the rotation ``r`` (3, 3) and
    translation ``t`` (3,) minimising the sum of squared distances between matched
    points, ``sum(|(r @ src[i] + t) - dst[i]|^2)``.

    Rotation only -- no scale factor is estimated, since both trajectories are in
    metres and a scaled fit would hide a real drift as a plausible-looking one.

    See https://en.wikipedia.org/wiki/Kabsch_algorithm -- centre both clouds, take
    the SVD of the cross-covariance, and read the rotation off the singular vectors.
    """
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    x = src - src_center
    y = dst - dst_center
    u, _, vt = np.linalg.svd(x.T @ y / len(src))
    r = vt.T @ u.T
    # A negative determinant means the SVD picked a reflection rather than a
    # rotation; flipping the last singular vector gives the best proper rotation.
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_center - r @ src_center
    return r, t


def match_nearest(gt_t: np.ndarray, est_t: np.ndarray, max_dt: float) -> list[tuple[int, int]]:
    """Pair each estimate stamp with its nearest ground-truth stamp.

    Both inputs are ascending stamps in seconds. Returns index pairs
    ``(gt_i, est_i)`` -- same order as the arguments -- for which the two stamps
    are within ``max_dt``; estimates with no GT that close are dropped. So every
    returned pair satisfies ``abs(gt_t[gt_i] - est_t[est_i]) <= max_dt``, and a
    caller that wants those differences can recover them from the indices.
    """
    # For each est stamp, the insertion point in gt_t: the nearest GT stamp is
    # either side of it, so only those two are candidates.
    insert_at = np.searchsorted(gt_t, est_t)
    pairs = []
    for est_i, stamp in enumerate(est_t):
        candidates = []
        if insert_at[est_i] > 0:
            candidates.append(insert_at[est_i] - 1)
        if insert_at[est_i] < len(gt_t):
            candidates.append(insert_at[est_i])
        gt_i = min(candidates, key=lambda i: abs(gt_t[i] - stamp))
        if abs(gt_t[gt_i] - stamp) <= max_dt:
            pairs.append((int(gt_i), est_i))
    return pairs


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def evaluate(gt_path: str, est_path: str, max_dt: float) -> dict[str, Any]:
    gt = np.loadtxt(gt_path, delimiter=",")
    est = np.loadtxt(est_path)
    gt_t = gt[:, 0] * 1e-9
    gt_p = gt[:, 1:4]
    est_t = est[:, 0]
    est_p = est[:, 1:4]

    pairs = match_nearest(gt_t, est_t, max_dt)
    if not pairs:
        raise RuntimeError("no timestamp matches found")

    gt_i, est_i = np.asarray(pairs, dtype=int).T
    # Recovered from the pairing rather than carried through it -- match_nearest
    # already guarantees every one of these is <= max_dt; reported to show how much
    # of that budget the matching actually used.
    dt = np.abs(gt_t[gt_i] - est_t[est_i])
    src = est_p[est_i]
    dst = gt_p[gt_i]

    raw_error = np.linalg.norm(src - dst, axis=1)
    r, t = align_se3(src, dst)
    aligned = (r @ src.T).T + t
    aligned_error = np.linalg.norm(aligned - dst, axis=1)

    return {
        "gt_count": len(gt),
        "est_count": len(est),
        "matched_count": len(pairs),
        "mean_abs_time_diff_sec": float(dt.mean()),
        "max_abs_time_diff_sec": float(dt.max()),
        "p95_abs_time_diff_sec": float(np.percentile(dt, 95)),
        "gt_path_length_m": float(np.linalg.norm(np.diff(dst, axis=0), axis=1).sum()),
        "raw": stats(raw_error),
        "se3": stats(aligned_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_csv", help="GT pose.csv")
    parser.add_argument("tum", help="estimated TUM trajectory")
    parser.add_argument("--max-dt", type=float, default=0.05, help="max timestamp difference in seconds")
    parser.add_argument("-o", "--output", help="optional text output path")
    args = parser.parse_args()

    result = evaluate(args.pose_csv, args.tum, args.max_dt)
    lines = [
        f"gt_count {result['gt_count']}",
        f"est_count {result['est_count']}",
        f"matched_count {result['matched_count']}",
        f"mean_abs_time_diff_sec {result['mean_abs_time_diff_sec']:.9f}",
        f"max_abs_time_diff_sec {result['max_abs_time_diff_sec']:.9f}",
        f"p95_abs_time_diff_sec {result['p95_abs_time_diff_sec']:.9f}",
        f"gt_path_length_m {result['gt_path_length_m']:.6f}",
    ]
    for prefix in ("raw", "se3"):
        for key, value in result[prefix].items():
            lines.append(f"{prefix}_ate_{key}_m {value:.6f}")

    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
