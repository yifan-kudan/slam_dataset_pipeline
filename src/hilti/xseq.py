#!/usr/bin/env python3
"""Cross-sequence localization on Hilti 2023: pick a hub map per site, register
every other sequence of that site into it, and emit the localizer prior.

    xseq.py coverage --run rerun_r1.0                  # per-map coverage table -> xcov.json
    xseq.py matrix   --run rerun_r1.0 --site 1         # all-pairs coarse overlap -> xmat_site<N>.json
    xseq.py register site1_handheld_2 site1_handheld_1 # hub, then the sequence to place in it
    xseq.py hubs     --run rerun_r1.0                  # choose the hub per site -> xhubs.json

WHY THIS EXISTS.  Every sequence's own map is the trivial case: a .kdm built from
that sequence's FAST-LIO cloud shares the cloud's frame, the rig starts at the
origin and `initial_pose` is just the gravity levelling.  Replaying sequence S
against the map of a *different* sequence H has no such luck -- a FAST-LIO map
frame is the IMU pose at the first processed scan, so H's map and S's bag share no
frame at all and kd_localizer needs a prior good enough for `initial_align`
(search_radius 0.5 m) to close.

WHAT MAKES IT EASY HERE.  align_map.py already measured each map's gravity from
FAST-LIO's own S2 gravity state -- a far better estimate than normals off the
cloud, because it comes from the filter that built it.  Levelling both clouds with
those known rotations collapses the 6-DOF search to yaw + xyz:

    L_H = R_H p_H          L_S = R_S p_S          (R from align_<seq>.json)
    find (Rz, t) with      Rz L_S + t ~ L_H
    =>  T_H_S = [Rz R_S | t]

and because S's bag starts at S's raw origin, that same matrix IS the prior:

    initial_pose.position = t
    initial_pose.rotation = euler(Rz R_S)

The matrix is written to the JSON under `R_align`/`t_align` as well, so
eval_localizer.py can push S's own FAST-LIO trajectory into H's map frame and give
the run a dense error curve instead of only the 2-8 surveyed control points.

Stages, each removing one unknown (after register_maps.py, which did this for
ConSLAM, but with the gravity stage replaced by the known rotation):

  yaw + xy   FFT phase correlation of 2D wall-occupancy grids over a yaw sweep.
             Walls carry the horizontal signal; floors are the same everywhere.
  z          1D correlation of the z histograms at the winning yaw/xy.
  ICP        multi-scale trimmed point-to-plane.  FAST-LIO writes normal_* but
             leaves them zero, so normals come from PCA over k nearest neighbours.

`overlap.within_0.5m` is the number that says whether the run is geometrically
possible at all -- it is reported for every pair and is what `hubs` maximises.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilti import dataset                                          # noqa: E402

PCD_TYPES = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2",
             ("U", 4): "<u4", ("I", 2): "<i2", ("I", 4): "<i4"}

# One cache per (sequence, run): reading + voxelising + normals is minutes, and
# every pair in a site wants the same two clouds.
_CACHE: dict[tuple, dict] = {}


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def read_pcd(path):
    """Read a binary PCD's xyz as float32 (memmapped, no full copy)."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if not line:
                raise SystemExit(f"{path}: header ended without DATA")
            parts = line.split()
            if parts:
                header[parts[0].upper()] = parts[1:]
            if parts and parts[0].upper() == "DATA":
                break
        offset = f.tell()
    if header["DATA"][0].lower() != "binary":
        raise SystemExit(f"{path}: only DATA binary is supported")
    dtype = np.dtype([(n, PCD_TYPES[(t.upper(), int(s))])
                      for n, s, t in zip(header["FIELDS"], header["SIZE"], header["TYPE"])])
    n = int(header["POINTS"][0])
    arr = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(n,))
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(1)]


def r_align(seq, results=None):
    """The gravity levelling align_map.py measured for this sequence's map frame."""
    path = dataset.align_json(seq, results)
    if not os.path.isfile(path):
        raise SystemExit(f"{seq}: no {path}\n  run src/hilti/align_map.py first")
    return np.array(json.load(open(path))["R_align"], dtype=float)


def voxel_mean(xyz, voxel):
    """Voxel-downsample, averaging the positions inside each voxel."""
    keys = np.floor(xyz / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    packed = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    _, inv = np.unique(packed, return_inverse=True)
    count = np.bincount(inv)
    out = np.stack([np.bincount(inv, w) / count for w in xyz.T], axis=1)
    return out.astype(np.float32)


def estimate_normals(xyz, k=20, chunk=200_000):
    """PCA normals over k nearest neighbours, plus a planarity score.

    planarity = 1 - lambda0/lambda1: near 1 on a clean plane, near 0 on an edge or
    in noise.  Point-to-plane ICP and the wall/floor split both want the flat
    points only, so the score is returned rather than thresholded here.
    """
    tree = cKDTree(xyz)
    nrm = np.empty_like(xyz)
    plan = np.empty(len(xyz), dtype=np.float32)
    for start in range(0, len(xyz), chunk):
        sl = slice(start, min(start + chunk, len(xyz)))
        _, idx = tree.query(xyz[sl], k=k, workers=-1)
        nb = xyz[idx].astype(np.float64)
        nb -= nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb) / k
        w, v = np.linalg.eigh(cov)
        nrm[sl] = v[:, :, 0].astype(np.float32)
        plan[sl] = (1.0 - w[:, 0] / np.maximum(w[:, 1], 1e-12)).astype(np.float32)
    return nrm, plan


def levelled(seq, run=None, root=None, results=None, voxel=0.4, fine_voxel=0.15,
             quiet=False, coarse_only=False):
    """The sequence's FAST-LIO cloud, rotated by its known R_align, at two scales.

    `coarse_only` skips the 0.15 m arrays, which are 4-8x the size of the 0.4 m ones and
    are only wanted by ICP. The all-pairs matrix never runs ICP, and it holds every
    sequence of a site in memory at once -- on a 16 GB machine already carrying a
    FAST-LIO run that is the difference between fitting and swapping.
    """
    key = (seq, run, voxel, fine_voxel, coarse_only)
    if key in _CACHE:
        return _CACHE[key]
    # A full-detail entry satisfies a coarse request; the reverse is not true.
    full = (seq, run, voxel, fine_voxel, False)
    if coarse_only and full in _CACHE:
        return _CACHE[full]
    pcd = dataset.map_pcd(seq, run, root)
    if not os.path.isfile(pcd):
        raise SystemExit(f"{seq}: no cloud at {pcd}")
    R = r_align(seq, results)
    xyz = read_pcd(pcd)
    n_raw = len(xyz)
    v_xyz = voxel_mean(xyz, voxel) @ R.T
    v_nrm, v_plan = estimate_normals(v_xyz)
    out = dict(seq=seq, pcd=pcd, R=R, raw_points=n_raw,
               v_xyz=v_xyz, v_nrm=v_nrm, v_plan=v_plan)
    if coarse_only:
        out.update(f_xyz=v_xyz, f_nrm=v_nrm, f_plan=v_plan)
        note = "coarse only"
    else:
        f_xyz = voxel_mean(xyz, fine_voxel) @ R.T
        f_nrm, f_plan = estimate_normals(f_xyz)
        out.update(f_xyz=f_xyz, f_nrm=f_nrm, f_plan=f_plan)
        note = f"{len(f_xyz):,} @ {fine_voxel} m"
    del xyz
    if not quiet:
        print(f"  {seq}: {n_raw:,} pts -> {len(v_xyz):,} @ {voxel} m, {note}",
              file=sys.stderr)
    _CACHE[key] = out
    return out


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def wall_grid(xyz, nrm, res, origin, shape, nz_max=0.4):
    """2D occupancy of vertical surfaces (walls / columns), binary per cell."""
    m = np.abs(nrm[:, 2]) < nz_max
    ij = np.floor((xyz[m, :2] - origin) / res).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    ij = ij[ok]
    g = np.zeros(shape, dtype=np.float32)
    np.add.at(g, (ij[:, 0], ij[:, 1]), 1.0)
    return (g > 0).astype(np.float32)


def coarse_yaw_xy(a_xyz, a_nrm, b_xyz, b_nrm, res, yaw_step_deg, half_extent):
    """FFT phase correlation over a yaw sweep -> (Rz, t, score)."""
    # The window is 2*half_extent across and CENTRED on each cloud's centroid. Off-centre
    # (the -half_extent/2 this was inherited with) it reaches from c-H/2 to c+3H/2, which
    # silently clips the near side: Hilti site1's real structure is ~100 m across at the
    # 0.1-99.9 percentile, so a window starting 35 m before the centroid drops walls that
    # the correlation needs. Both grids use the same offset, so the shift decoding below
    # is unchanged either way -- only the coverage differs.
    shape = (int(2 * half_extent / res), int(2 * half_extent / res))
    a_c = a_xyz[:, :2].mean(axis=0)
    b_c = b_xyz[:, :2].mean(axis=0)
    ga = wall_grid(a_xyz, a_nrm, res, a_c - half_extent, shape)
    ga -= ga.mean()
    Fa = np.fft.rfft2(ga)
    best = None
    for yaw in np.arange(0.0, 360.0, yaw_step_deg):
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        R = np.array([[c, -s], [s, c]])
        rb = (b_xyz[:, :2] - b_c) @ R.T
        rn = b_nrm[:, :2] @ R.T
        gb = wall_grid(np.column_stack([rb, b_xyz[:, 2]]),
                       np.column_stack([rn, b_nrm[:, 2]]), res,
                       np.array([-half_extent, -half_extent]), shape)
        gb -= gb.mean()
        corr = np.fft.irfft2(Fa * np.conj(np.fft.rfft2(gb)), s=shape)
        k = int(np.argmax(corr))
        i, j = np.unravel_index(k, shape)
        shift = np.array([i, j], dtype=float)
        shift -= np.where(shift > np.array(shape) / 2, shape, 0)
        sc = float(corr[i, j])
        if best is None or sc > best[3]:
            t = shift * res + a_c
            best = (np.radians(yaw), t[0], t[1], sc, b_c, R)
    yaw, tx, ty, score, b_c, R = best
    Rz = np.eye(3)
    Rz[:2, :2] = R
    t3 = np.array([tx, ty, 0.0]) - Rz @ np.array([b_c[0], b_c[1], 0.0])
    return Rz, t3, score


def coarse_z(a_xyz, b_xyz, Rz, t3, bin_z=0.1, max_dz=8.0):
    """1D histogram correlation on z -- floors and ceilings are strong z features."""
    bz = (b_xyz @ Rz.T + t3)[:, 2]
    lo = min(a_xyz[:, 2].min(), bz.min()) - max_dz
    hi = max(a_xyz[:, 2].max(), bz.max()) + max_dz
    bins = np.arange(lo, hi + bin_z, bin_z)
    ha, _ = np.histogram(a_xyz[:, 2], bins=bins)
    hb, _ = np.histogram(bz, bins=bins)
    ha = ha - ha.mean()
    hb = hb - hb.mean()
    n = len(ha)
    corr = np.fft.irfft(np.fft.rfft(ha, 2 * n) * np.conj(np.fft.rfft(hb, 2 * n)), 2 * n)
    k = np.arange(2 * n)
    shift = np.where(k > n, k - 2 * n, k)
    ok = np.abs(shift * bin_z) <= max_dz
    dz = shift[ok][int(np.argmax(corr[ok]))] * bin_z
    out = t3.copy()
    out[2] += dz
    return out, float(dz)


def icp_point_to_plane(src, tgt, tgt_nrm, R, t, max_dist, iters=30, trim=0.8):
    """Trimmed point-to-plane ICP; src/tgt already coarsely aligned."""
    tree = cKDTree(tgt)
    for _ in range(iters):
        p = src @ R.T + t
        d, idx = tree.query(p, workers=-1, distance_upper_bound=max_dist)
        m = np.isfinite(d)
        if m.sum() < 200:
            break
        dm = d[m]
        keep = dm <= np.quantile(dm, trim)
        sel = np.where(m)[0][keep]
        q = tgt[idx[sel]]
        n = tgt_nrm[idx[sel]]
        ps = p[sel]
        c = ps.mean(axis=0)
        A = np.column_stack([np.cross(ps - c, n), n])
        b = -np.einsum("ij,ij->i", ps - q, n)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        dR = Rotation.from_rotvec(x[:3]).as_matrix()
        dt = x[3:] + c - dR @ c
        R = dR @ R
        t = dR @ t + dt
        if np.linalg.norm(x[:3]) < 1e-6 and np.linalg.norm(x[3:]) < 1e-5:
            break
    return R, t


def overlap(src, tgt, R, t, radii=(0.2, 0.5, 1.0)):
    tree = cKDTree(tgt)
    d, _ = tree.query(src @ R.T + t, workers=-1, distance_upper_bound=max(radii) * 4)
    finite = d[np.isfinite(d)]
    out = {f"within_{r}m": round(float((d <= r).mean()), 4) for r in radii}
    out["median_nn_m"] = round(float(np.median(finite)), 4) if finite.size else None
    return out


def local_check(A, B, R, t, radius=20.0, max_dist=0.5):
    """Re-fit using only the points near `seq`'s own origin, and report the disagreement.

    The prior that matters is the one at t=0: the global fit minimises residual over the
    whole site, so if the far end of the building dragged it, the START pose would be off
    even with a healthy overall overlap. Restricting both clouds to a ball around seq's
    origin (which is 0 in its levelled frame, so the ball is around `t` on the hub side)
    and re-running the finest ICP stage answers that directly. A few centimetres and a
    tenth of a degree means the global fit is trustworthy where the run begins.
    """
    near_b = np.linalg.norm(B["f_xyz"], axis=1) <= radius
    hub_centre = t
    near_a = np.linalg.norm(A["f_xyz"] - hub_centre, axis=1) <= radius * 1.5
    if near_b.sum() < 500 or near_a.sum() < 500:
        return None
    R2, t2 = icp_point_to_plane(B["f_xyz"][near_b], A["f_xyz"][near_a],
                                A["f_nrm"][near_a], R, t, max_dist)
    dR = R2 @ R.T
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return {"radius_m": radius, "src_points": int(near_b.sum()),
            "d_translation_m": round(float(np.linalg.norm(t2 - t)), 4),
            "d_rotation_deg": round(float(ang), 4)}


def route_coverage(A, B, R, t, seq, run=None, root=None, ball=15.0, radius=0.5,
                   samples=200):
    """Local overlap in a ball around each pose along `seq`'s own trajectory.

    Cloud-level overlap is one number for the whole site and hides where the map runs
    out. This is the same measurement per LOCATION: at sampled poses along the route,
    what fraction of seq's cloud inside a `ball` metre sphere has a hub point within
    `radius`. The minimum over the route, not the mean, is what decides whether the run
    can survive the whole walk.
    """
    log = dataset.pos_log(seq, run, root)
    if not os.path.isfile(log):
        return None
    traj = np.loadtxt(log)[:, 4:7] @ B["R"].T          # seq's raw frame -> seq levelled
    idx = np.linspace(0, len(traj) - 1, min(samples, len(traj))).astype(int)
    src = B["v_xyz"]
    tgt_tree = cKDTree(A["v_xyz"])
    src_moved = src @ R.T + t
    src_tree = cKDTree(src)
    fracs = []
    for pose in traj[idx]:
        near = src_tree.query_ball_point(pose, ball)
        if len(near) < 200:
            continue
        d, _ = tgt_tree.query(src_moved[near], workers=-1,
                              distance_upper_bound=radius * 4)
        fracs.append(float((d <= radius).mean()))
    if not fracs:
        return None
    f = np.array(fracs)
    return {"ball_m": ball, "poses": len(f),
            "median": round(float(np.median(f)), 4),
            "p05": round(float(np.percentile(f, 5)), 4),
            "min": round(float(f.min()), 4),
            "below_0.5_pct": round(100 * float((f < 0.5).mean()), 1)}


def to_euler(R):
    """Inverse of Rz(yaw) @ Ry(pitch) @ Rx(roll) -- kd_amr_libs/math/src/rotation.cpp."""
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw])


def register(hub, seq, run=None, root=None, results=None, coarse_only=False,
             grid_res=0.4, yaw_step=1.0, half_extent=140.0, quiet=False):
    """Place `seq`'s cloud in `hub`'s levelled map frame.  Returns the record."""
    A = levelled(hub, run, root, results, quiet=quiet, coarse_only=coarse_only)
    B = levelled(seq, run, root, results, quiet=quiet, coarse_only=coarse_only)
    Rz, t3, score = coarse_yaw_xy(A["v_xyz"], A["v_nrm"], B["v_xyz"], B["v_nrm"],
                                  grid_res, yaw_step, half_extent)
    t3, dz = coarse_z(A["v_xyz"], B["v_xyz"], Rz, t3)
    out = {"hub": hub, "seq": seq, "run": run or dataset.sequence(seq)["run"],
           "hub_pcd": A["pcd"], "seq_pcd": B["pcd"],
           "coarse": {"yaw_deg": round(float(np.degrees(np.arctan2(Rz[1, 0], Rz[0, 0]))), 3),
                      "t": [round(float(x), 3) for x in t3], "dz": round(dz, 3),
                      "score": round(score, 1)},
           "coarse_overlap": overlap(B["v_xyz"], A["v_xyz"], Rz, t3)}
    if not quiet:
        print(f"  coarse {hub} <- {seq}: yaw {out['coarse']['yaw_deg']:.1f} deg "
              f"t {out['coarse']['t']} overlap {out['coarse_overlap']['within_0.5m']:.3f}",
              file=sys.stderr)
    R, t = Rz, t3
    if not coarse_only:
        for max_dist, tag in [(2.0, "v"), (1.0, "v"), (0.5, "f"), (0.25, "f")]:
            R, t = icp_point_to_plane(B[f"{tag}_xyz"], A[f"{tag}_xyz"], A[f"{tag}_nrm"],
                                      R, t, max_dist)
            if not quiet:
                ov = overlap(B["v_xyz"], A["v_xyz"], R, t)
                print(f"  icp @ {max_dist} m: yaw "
                      f"{np.degrees(np.arctan2(R[1, 0], R[0, 0])):.2f} "
                      f"t {np.round(t, 3)} overlap {ov['within_0.5m']:.3f}", file=sys.stderr)
    out["overlap"] = overlap(B["v_xyz"], A["v_xyz"], R, t)
    if not coarse_only:
        out["local_check"] = local_check(A, B, R, t)
        out["route_coverage"] = route_coverage(A, B, R, t, seq, run, root)
        if not quiet:
            print(f"  local check {out['local_check']}\n"
                  f"  route coverage {out['route_coverage']}", file=sys.stderr)

    # The prior.  R_full maps seq's RAW FAST-LIO frame into hub's LEVELLED map frame,
    # which is both the initial pose (seq's bag starts at its own raw origin) and the
    # transform eval_localizer.py needs to compare against seq's own pos_log.
    R_full = R @ B["R"]
    euler = to_euler(R_full)
    out["T_hub_seq"] = {"matrix": np.vstack([np.column_stack([R_full, t]),
                                             [0, 0, 0, 1]]).tolist(),
                        "position": [float(x) for x in t],
                        "rotation_rpy": [float(x) for x in euler],
                        "rotation_rpy_deg": [round(float(np.degrees(x)), 3) for x in euler]}
    # align-JSON compatible keys, so eval_localizer.py --align <this file> works.
    out["R_align"] = R_full.tolist()
    out["t_align"] = [float(x) for x in t]
    out["initial_pose_rotation_rpy"] = [float(x) for x in euler]
    out["initial_pose_rotation_rpy_deg"] = out["T_hub_seq"]["rotation_rpy_deg"]
    out["tilt_from_plus_z_deg"] = round(float(np.degrees(
        np.arccos(np.clip(R_full[2, 2], -1, 1)))), 2)
    return out


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------
def coverage(seq, run=None, root=None, results=None, voxel=1.0):
    """How much building one sequence's map actually holds, in its levelled frame.

    `occupied_voxels` at 1 m is the honest size measure: point count is dominated by
    how long the rig lingered, and the bbox by a single stray return.  Path length
    comes from the run's own pos_log, so a folded trajectory shows up as a long path
    inside a small box.
    """
    # Only the 0.4 m arrays are read below, so do not pay for the fine ones.
    L = levelled(seq, run, root, results, quiet=True, coarse_only=True)
    xyz = L["v_xyz"]
    keys = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    entry = dataset.sequences()[seq]
    out = {"seq": seq, "site": entry["site"], "platform": entry["platform"],
           "raw_points": L["raw_points"], "occupied_voxels_1m": int(len(keys)),
           "bbox_m": [round(float(v), 1) for v in (hi - lo)],
           "bbox_volume_m3": round(float(np.prod(hi - lo)), 1),
           "z_span_m": round(float(hi[2] - lo[2]), 2),
           "wall_voxel_frac": round(float((np.abs(L["v_nrm"][:, 2]) < 0.4).mean()), 3)}
    log = dataset.pos_log(seq, run, root)
    if os.path.isfile(log):
        p = np.loadtxt(log)[:, 4:7]
        out["path_m"] = round(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), 1)
        out["frames"] = int(len(p))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("coverage", "matrix", "register", "hubs"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--run", help="FAST-LIO run whose cloud to use (default: per sequence)")
    ap.add_argument("--root")
    ap.add_argument("--results")
    ap.add_argument("--site", type=int, help="restrict to one site")
    ap.add_argument("--coarse-only", action="store_true",
                    help="skip ICP (the `matrix` default -- coarse is enough to rank)")
    ap.add_argument("--half-extent", type=float, default=140.0)
    ap.add_argument("--yaw-step", type=float, default=1.0)
    ap.add_argument("-o", "--out", help="write the JSON here")
    a = ap.parse_args()

    res = dataset.results_dir(a.results)
    seqs = {s: e for s, e in dataset.sequences().items()
            if a.site is None or e["site"] == a.site}

    if a.cmd == "coverage":
        rows = []
        for s in seqs:
            try:
                rows.append(coverage(s, a.run, a.root, a.results))
            except SystemExit as e:
                print(f"SKIP {s}: {e}", file=sys.stderr)
        rows.sort(key=lambda r: (r["site"], -r["occupied_voxels_1m"]))
        hdr = f"{'sequence':<18} {'site':<5} {'plat':<9} {'vox@1m':>8} {'bbox':>18} " \
              f"{'z':>6} {'path':>8} {'wall%':>6}"
        print(hdr)
        for r in rows:
            print(f"{r['seq']:<18} {r['site']:<5} {r['platform']:<9} "
                  f"{r['occupied_voxels_1m']:>8,} "
                  f"{'x'.join(str(v) for v in r['bbox_m']):>18} "
                  f"{r['z_span_m']:>6.1f} {r.get('path_m', 0):>8.1f} "
                  f"{100 * r['wall_voxel_frac']:>6.1f}")
        out = a.out or f"{res}/xcov.json"
        json.dump(rows, open(out, "w"), indent=2)
        print(f"wrote {out}", file=sys.stderr)

    elif a.cmd == "matrix":
        if a.site is None:
            raise SystemExit("matrix needs --site")
        names = list(seqs)
        rows = []
        for hub in names:
            for seq in names:
                if hub == seq:
                    continue
                try:
                    rec = register(hub, seq, a.run, a.root, a.results, coarse_only=True,
                                   half_extent=a.half_extent, yaw_step=a.yaw_step)
                except SystemExit as e:
                    print(f"SKIP {hub} <- {seq}: {e}", file=sys.stderr)
                    continue
                rows.append({"hub": hub, "seq": seq,
                             "overlap_0.5m": rec["coarse_overlap"]["within_0.5m"],
                             "overlap_1.0m": rec["coarse_overlap"]["within_1.0m"],
                             "yaw_deg": rec["coarse"]["yaw_deg"]})
        out = a.out or f"{res}/xmat_site{a.site}.json"
        json.dump(rows, open(out, "w"), indent=2)
        print(f"wrote {out}", file=sys.stderr)
        for r in rows:
            print(f"{r['hub']:<18} <- {r['seq']:<18} {r['overlap_0.5m']:.3f}")

    elif a.cmd == "register":
        if len(a.args) != 2:
            raise SystemExit("register needs: <hub> <seq>")
        hub, seq = a.args
        rec = register(hub, seq, a.run, a.root, a.results, coarse_only=a.coarse_only,
                       half_extent=a.half_extent, yaw_step=a.yaw_step)
        out = a.out or f"{res}/xreg_{seq}_in_{hub}.json"
        json.dump(rec, open(out, "w"), indent=2)
        print(json.dumps({k: rec[k] for k in ("hub", "seq", "overlap",
                                              "initial_pose_rotation_rpy_deg")}, indent=2))
        print("\ninitial_pose:")
        p = rec["T_hub_seq"]["position"]
        e = rec["initial_pose_rotation_rpy"]
        print(f"  position: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
        print(f"  rotation: [{e[0]:.6f}, {e[1]:.6f}, {e[2]:.6f}]")
        print(f"wrote {out}", file=sys.stderr)

    elif a.cmd == "hubs":
        # The hub is the map with the highest COVERAGE OF THE OTHER SEQUENCES, which is
        # what decides whether the localizer can match at all -- not the biggest map.
        # `matrix` measures that directly (mean fraction of every other sequence's cloud
        # landing within 0.5 m of this one), so it wins when it has been run; the 1 m
        # voxel count is the fallback and is only a proxy for mapped volume.
        cov = json.load(open(f"{res}/xcov.json"))
        by_site: dict[int, list] = {}
        for r in cov:
            by_site.setdefault(r["site"], []).append(r)
        hubs = {}
        for site, rows in sorted(by_site.items()):
            names = [r["seq"] for r in rows]
            mat_path = f"{res}/xmat_site{site}.json"
            mean_ov, basis = {}, "occupied_voxels_1m"
            if os.path.isfile(mat_path):
                mat = json.load(open(mat_path))
                for hub in names:
                    got = [m["overlap_0.5m"] for m in mat if m["hub"] == hub]
                    if len(got) == len(names) - 1:
                        mean_ov[hub] = float(np.mean(got))
                if mean_ov:
                    basis = "mean_coarse_overlap_0.5m"
            if basis == "mean_coarse_overlap_0.5m":
                best_name = max(mean_ov, key=mean_ov.get)
            else:
                best_name = max(rows, key=lambda r: r["occupied_voxels_1m"])["seq"]
            best = next(r for r in rows if r["seq"] == best_name)
            hubs[str(site)] = {
                "hub": best_name, "basis": basis,
                "occupied_voxels_1m": best["occupied_voxels_1m"],
                "mean_coarse_overlap_0.5m": (round(mean_ov[best_name], 4)
                                             if best_name in mean_ov else None),
                "ranking": ([[k, round(v, 4)] for k, v in
                             sorted(mean_ov.items(), key=lambda kv: -kv[1])]
                            if mean_ov else
                            [[r["seq"], r["occupied_voxels_1m"]] for r in
                             sorted(rows, key=lambda r: -r["occupied_voxels_1m"])]),
                "others": [n for n in names if n != best_name]}
            print(f"site {site}: hub {best_name}  (by {basis}) "
                  f"{len(hubs[str(site)]['others'])} others")
            for name, value in hubs[str(site)]["ranking"]:
                mark = "  <-- hub" if name == best_name else ""
                print(f"    {name:<18} {value}{mark}")
        json.dump(hubs, open(f"{res}/xhubs.json", "w"), indent=2)
        print(f"wrote {res}/xhubs.json", file=sys.stderr)


if __name__ == "__main__":
    main()
