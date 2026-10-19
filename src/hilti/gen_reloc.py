#!/usr/bin/env python3
"""Turn a gen_cfg.py run into a RELOCALIZER run: params for kd_relocalizer_node, a
gravity-levelled base frame, and (optionally) no prior at all.

    gen_reloc.py site1_handheld_1                       own map, prior kept, BBS on LOST
    gen_reloc.py site1_handheld_2 --map-seq site1_handheld_1 --cold-start
                                                        cross map, NO prior: BBS finds it

Why this is not just `use_relocalizer:=true`.  Three things block that flag:

1. kd_relocalizer_node throws in its constructor when `map` is unset
   (relocalizer_node.cpp:38), and gen_cfg.py's params file has only a
   kd_localizer_node section -- so the node dies at startup and the run looks
   normal apart from one line in the log.
2. The BBS angular search is roll/pitch in [-max_tilt, +max_tilt] and yaw over the
   full 2*pi (relocalizer_node.cpp:configure_bbs).  max_tilt defaults to 0.02 rad.
   Our base frame is `imu_link`, and the .kdm is levelled, so the rig's TRUE
   attitude at t0 is R_align -- 178-180 deg from level on the handheld rig and
   ~90 deg on the robot (see any cfg_*.yaml `initial_pose`).  The correct answer is
   not in the search space at any max_tilt worth paying for: widening roll/pitch to
   +-pi turns the per-node branching from 8*1*1*2 into 8*2*2*2 at every level of a
   7-level pyramid.
3. The pose hint carries no usable time: LocalizerNode::pose_hint_callback
   (localizer_node.cpp:881) drops header.stamp and applies the pose at the next
   process tick, while the relocalizer stamps it with the cloud it started the
   search on.  A slow search on a walking rig therefore hands back a pose from
   several metres ago -- which is why --reloc-timeout-ms defaults to 2000 here and
   not to the library's 30000.

`blevel` fixes (2).  It is a frame rigidly attached to the IMU with

    T_imu_blevel = R_align^T          (pure rotation, same origin)

so that T_map_blevel(t0) = R_align @ R_align^T = I: level at t0 by construction.
Both nodes take base_frame_id: blevel, run_seq.sh publishes imu_link -> blevel from
the .args file written here, and BBS then searches yaw + xyz, which is exactly the
4-DOF space xseq.py searches offline -- so a cold-start cross run is a like-for-like
comparison with the registration in the report.

What it costs: the reported pose is T_map_blevel, not T_map_imu.  The origins
coincide, so every POSITION is unchanged and the trajectory scoring needs nothing.
Only the orientation differs, by the constant R_align, which matters solely for the
surveyed-prism lever arm -- pass --level-base to eval_localizer.py there.

Mid-run recovery has a limit worth knowing before reading the results: measured off
each run's own FAST-LIO attitude, blevel's tilt from level has a median of 0.8-17 deg
and a 95th percentile of 1.4-56 deg depending on rig and site.  A single BBS attempt
at --max-tilt 0.35 (20 deg) therefore covers roughly half the frames of a site1
handheld walk and nearly all of a robot run; the node retries on every LOST status
message, so what matters is that a low-tilt frame comes along, not that any one does.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths                                                       # noqa: E402
from hilti import dataset                                          # noqa: E402

LEVEL_NOTE = """\
# LEVELLED BASE FRAME. base_frame_id is `blevel`, not `imu_link`: a frame rigidly
# attached to the IMU with T_imu_blevel = R_align^T, published by run_seq.sh from
# tf_blevel_{name}.args. It exists so the 3D-BBS relocalizer's roll/pitch search
# window (+-max_tilt about level) can contain the answer -- imu_link is {tilt} deg
# from level in this map. Origins coincide, so reported POSITIONS are unchanged and
# only the orientation differs, by R_align."""

COLD_NOTE = """\
# COLD START: no initial_pose. The node stays UNINITIALIZED, publishes that status,
# and kd_relocalizer_node answers it with a 3D-BBS search over the .kdm's voxel
# pyramid. This is the experiment: whether the shipped global relocalizer finds by
# itself the pose that xseq.py computed offline{ref}."""

RELOC_PARAMS = """
# kd_relocalizer_node: the shipped 3D-BBS global relocalizer, started by run_seq.sh
# (RELOC=1) rather than by localizer.launch.py's use_relocalizer, so that the moment
# it joins the graph is controllable -- on an own-map run with a good initial_pose it
# must NOT fire during the startup UNINITIALIZED window and clobber the prior.
#
# base_frame_id must equal kd_localizer_node's: the hint is T_map_base and
# LocalizerNode feeds it straight into Localizer::relocalize().
kd_relocalizer_node:
  ros__parameters:
    # `map` unset makes the constructor throw and the node exit (relocalizer_node.cpp:38).
    map: {map_path}
    map_frame_id: map
    base_frame_id: {base}
    lidar_exclude_radius: 0.5
    # The BBS pyramid's finest level is 0.5 m (plane voxel 0.25 m x --bbs-voxel-factor 2),
    # so a finer source cloud only costs scoring time.
    lidar_voxel_size: {voxel}
    # roll/pitch search half-width in radians ({tilt_deg} deg); yaw is always the full 2*pi.
    max_tilt: {max_tilt}
    # 0.9 is the library default and is a fraction of the SOURCE points that must fall in
    # an occupied voxel. A different visit to a construction site does not reach it -- the
    # report's registrations land 0.37-0.75 of the cloud within 0.5 m.
    score_threshold_percentage: {score}
    # Not the 30000 default: the hint is applied as `now` however old the cloud it was
    # computed from, so a long search is worse than no answer on a moving rig.
    timeout_msec: {timeout}
    relocalization_wait_time_ms: {wait}
    # Not 8. This host has 16 cores and kd_localizer_node already asks for 8; a BBS search
    # is CPU-bound for seconds at a time, and the localizer has a 20 Hz process deadline to
    # miss. The relocalizer gets the slack, not half the machine.
    num_threads: {threads}
"""


def rewrite_initial_pose(text, block, note):
    """Replace the `initial_pose:` 3-line block, keeping every other line and comment."""
    out, i, lines = [], 0, text.split("\n")
    while i < len(lines):
        if lines[i].startswith("initial_pose:"):
            out.append(note)
            out.extend(block)
            i += 3                      # initial_pose: / position: / rotation:
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--map-seq", help="hub sequence whose .kdm to localize against")
    ap.add_argument("--prior", help="xseq.py register JSON (default: the standard path)")
    ap.add_argument("--tag", default="reloc", help="artefact suffix (default: reloc)")
    ap.add_argument("--cold-start", action="store_true",
                    help="drop initial_pose entirely and let BBS find the pose")
    ap.add_argument("--keep-imu-base", action="store_true",
                    help="leave base_frame_id at imu_link -- the naive `just turn the "
                         "flag on` configuration, kept so its failure is measured "
                         "rather than argued")
    ap.add_argument("--max-tilt", type=float, default=0.35)
    ap.add_argument("--score", type=float, default=0.6)
    ap.add_argument("--reloc-timeout-ms", type=int, default=2000)
    ap.add_argument("--reloc-voxel", type=float, default=0.5)
    ap.add_argument("--reloc-wait-ms", type=int, default=1000)
    ap.add_argument("--reloc-threads", type=int, default=4)
    # A BBS hint lands on the pyramid's finest grid (0.5 m here) and is applied as `now`
    # however stale it is, so the first align after a relocalization has a metre-scale gap
    # to close -- and gen_cfg.py's initial_align.search_radius is 0.5 m, tighter than the
    # library's own 1.0. Raising it is the handoff, not a loosening of the tracker: only
    # the first update after a relocalization uses initial_align (localizer.cpp:136,164).
    ap.add_argument("--initial-radius", type=float,
                    help="override localizer.initial_align.search_radius")
    ap.add_argument("--initial-sigma", type=float,
                    help="override localizer.initial_align.match_sigma")
    ap.add_argument("--grav", type=float, help="override prior_gravity_sigma")
    ap.add_argument("--gate", type=int, help="override max_pose_sigma_factor_increase")
    ap.add_argument("-o", "--out", help="results dir (default: the dataset's)")
    ap.add_argument("--work", help="path the configs are mounted at in the container")
    ap.add_argument("--map-dir")
    a = ap.parse_args()

    out = dataset.results_dir(a.out)
    work = a.work or paths.get("localizer_container.work_mount")
    map_dir = a.map_dir or dataset.locations()["kdm"]
    here = Path(__file__).resolve().parent
    base_name = f"{a.seq}_in_{a.map_seq}" if a.map_seq else a.seq
    name = f"{base_name}_{a.tag}"

    # gen_cfg.py stays the single source of every localizer setting; this script only
    # adds what a relocalizer run needs on top, so the two cannot drift apart.
    cmd = [sys.executable, str(here / "gen_cfg.py"), a.seq]
    for flag, value in (("-o", a.out), ("--work", a.work), ("--map-dir", a.map_dir),
                        ("--map-seq", a.map_seq), ("--prior", a.prior)):
        if value:
            cmd += [flag, value]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    cfg = open(f"{out}/cfg_{base_name}.yaml").read()

    # R_own levels THIS sequence's own FAST-LIO frame; it defines blevel and is the same
    # transform whether the run is own-map or cross-sequence.
    own = json.load(open(dataset.align_json(a.seq, a.out)))
    R_own = np.array(own["R_align"])
    # R_prior places the rig at t0 in the map being localized against: R_own for an
    # own-map run, and the registration's rotation for a cross-sequence one.
    prior_json = (a.prior or f"{out}/xreg_{a.seq}_in_{a.map_seq}.json") if a.map_seq else None
    al = json.load(open(prior_json)) if prior_json else own
    R_prior = np.array(al["R_align"])
    t_prior = np.array(al.get("t_align", [0.0, 0.0, 0.0]))

    base = "imu_link" if a.keep_imu_base else "blevel"
    if a.keep_imu_base:
        note, R_base = "", R_prior
    else:
        note = LEVEL_NOTE.format(name=name, tilt=al["tilt_from_plus_z_deg"])
        # T_map_blevel(t0) = T_map_imu(t0) @ T_imu_blevel = R_prior @ R_own^T.
        # Own map: identity. Cross: pure yaw, because R_prior is Rz @ R_own.
        R_base = R_prior @ R_own.T

    if a.cold_start:
        ref = (f" for {a.seq} in {a.map_seq}'s map -- and how far off it lands"
               if a.map_seq else "")
        cfg = rewrite_initial_pose(cfg, [], COLD_NOTE.format(ref=ref))
    else:
        rpy = Rot.from_matrix(R_base).as_euler("xyz")
        deg = np.round(np.degrees(rpy), 3)
        cfg = rewrite_initial_pose(cfg, [
            "initial_pose:",
            f"  position: [{t_prior[0]}, {t_prior[1]}, {t_prior[2]}]",
            f"  rotation: [{rpy[0]:.12f}, {rpy[1]:.12f}, {rpy[2]:.12f}]"
            f"     # = [{deg[0]}, {deg[1]}, {deg[2]}] deg"
            + ("  (level by construction)" if not a.map_seq and not a.keep_imu_base else ""),
        ], note)

    if a.initial_radius is not None or a.initial_sigma is not None:
        # `match_sigma: 0.1` appears in both align blocks, so replace the initial one whole.
        old_block = ("  initial_align:\n    search_radius: 0.5\n"
                     "    match_sigma: 0.1\n    max_iterations: 30")
        assert old_block in cfg, "initial_align block is not the shape gen_cfg.py writes"
        cfg = cfg.replace(old_block,
                          "  initial_align:\n"
                          f"    search_radius: {a.initial_radius or 0.5}\n"
                          f"    match_sigma: {a.initial_sigma or 0.1}\n"
                          "    max_iterations: 30")
    if a.grav is not None:
        cfg = cfg.replace("prior_gravity_sigma: 0.3",
                          f"prior_gravity_sigma: {a.grav}")
    if a.gate is not None:
        cfg = cfg.replace("max_pose_sigma_factor_increase: 200",
                          f"max_pose_sigma_factor_increase: {a.gate}")
    open(f"{out}/cfg_{name}.yaml", "w").write(cfg)

    params = open(f"{out}/params_{base_name}.yaml").read()
    params = params.replace(f"cfg_{base_name}.yaml", f"cfg_{name}.yaml")
    if base != "imu_link":
        params = params.replace(
            "    # imu_link, not velodyne: the .kdm frame is the IMU frame at t=0, so this is\n"
            "    # what makes the reported pose comparable to FAST-LIO and to the survey points.\n"
            "    base_frame_id: imu_link",
            "    # blevel, not imu_link: see cfg_%s.yaml. Same origin, so positions stay\n"
            "    # comparable to FAST-LIO and to the survey points; the orientation does not.\n"
            "    base_frame_id: %s" % (name, base))
    map_path = f"{map_dir}/{a.map_seq or a.seq}_fastlio_map.kdm"
    params += RELOC_PARAMS.format(
        map_path=map_path, base=base, voxel=a.reloc_voxel, max_tilt=a.max_tilt,
        tilt_deg=round(np.degrees(a.max_tilt), 1), score=a.score,
        timeout=a.reloc_timeout_ms, wait=a.reloc_wait_ms, threads=a.reloc_threads)
    open(f"{out}/params_{name}.yaml", "w").write(params)

    # The blevel transform, for run_seq.sh. Written even for --keep-imu-base so the two
    # configurations differ in exactly one place (the frame the two nodes name).
    q = Rot.from_matrix(R_own.T).as_quat()      # xyzw, T_imu_blevel
    open(f"{out}/tf_blevel_{name}.args", "w").write(
        f"--x 0 --y 0 --z 0 --qx {q[0]:.12f} --qy {q[1]:.12f} --qz {q[2]:.12f} "
        f"--qw {q[3]:.12f} --frame-id imu_link --child-frame-id blevel\n")

    os.makedirs(out, exist_ok=True)
    print(f"wrote {out}/cfg_{name}.yaml, params_{name}.yaml, tf_blevel_{name}.args "
          f"(base {base}, max_tilt {a.max_tilt}, "
          f"{'cold start' if a.cold_start else 'prior kept'})")


if __name__ == "__main__":
    main()
