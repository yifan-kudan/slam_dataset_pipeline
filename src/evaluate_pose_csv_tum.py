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

import numpy as np


def align_se3(src: np.ndarray, dst: np.ndarray):
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    x = src - src_center
    y = dst - dst_center
    u, _, vt = np.linalg.svd(x.T @ y / len(src))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_center - r @ src_center
    return r, t


def match_nearest(gt_t: np.ndarray, est_t: np.ndarray, max_dt: float):
    idx = np.searchsorted(gt_t, est_t)
    pairs = []
    for est_i, stamp in enumerate(est_t):
        candidates = []
        if idx[est_i] > 0:
            candidates.append(idx[est_i] - 1)
        if idx[est_i] < len(gt_t):
            candidates.append(idx[est_i])
        gt_i = min(candidates, key=lambda i: abs(gt_t[i] - stamp))
        dt = abs(gt_t[gt_i] - stamp)
        if dt <= max_dt:
            pairs.append((est_i, gt_i, dt))
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


def evaluate(gt_path: str, est_path: str, max_dt: float):
    gt = np.loadtxt(gt_path, delimiter=",")
    est = np.loadtxt(est_path)
    gt_t = gt[:, 0] * 1e-9
    gt_p = gt[:, 1:4]
    est_t = est[:, 0]
    est_p = est[:, 1:4]

    pairs = match_nearest(gt_t, est_t, max_dt)
    if not pairs:
        raise RuntimeError("no timestamp matches found")

    est_i = np.array([p[0] for p in pairs])
    gt_i = np.array([p[1] for p in pairs])
    dt = np.array([p[2] for p in pairs])
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
