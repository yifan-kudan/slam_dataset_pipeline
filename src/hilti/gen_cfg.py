#!/usr/bin/env python3
"""Emit the kd_localizer config + node params for one Hilti 2023 sequence.

    gen_cfg.py site2_robot_1 [-o /work]

Two modes:

  own map (default)   the .kdm was built from THIS sequence's FAST-LIO cloud, so the
                      map frame is the IMU pose at its first processed scan and the
                      rig starts at the origin -- `initial_pose` is the gravity
                      levelling and nothing else.
  --map-seq HUB       the .kdm belongs to a DIFFERENT sequence of the same site. The
                      two share no frame, so the prior comes from
                      src/hilti/xseq.py register (which writes the 4x4 that places
                      this sequence's raw FAST-LIO frame in the hub's levelled map
                      frame) and `initial_pose` carries a translation as well.

The settings that are not library defaults, and why (all three carried over from
the ConSLAM evaluation, where each unblocked exactly one failure):

  min_ray_inlier_ratio 0.1 -> 0.0            an open construction site lets rays
      leave through gaps and return nothing the map can confirm; the ratio sits
      near zero for minutes while the pose stays ~10 cm accurate.
  predicted_pose_time_tolerance 0.001 -> 0.02  a violation of the 1 ms default is
      an *uncaught exception* that kills the node, not a dropped frame.  The
      robot's IMU is 200 Hz (5 ms), so 1 ms has no margin at all.
  max_pose_sigma_factor_increase 20 -> 200   the factor spikes into the 30-60 band
      on healthy poses; anything near the top of that band is a false Lost.

  prior_gravity_sigma 0.1 -> 0.3             a FAST-LIO map frame is NOT gravity
      aligned -- it is the IMU frame at t=0, so map +Z is off vertical by however
      the rig was held.  Measured per sequence below; 0.3 rad covers all of them.

And one that applies to --map-seq only:

  prior_{gyro,accel}_bias_sigma 10 -> 0.1/0.5   a cross-map startup disagrees with its
      prior by 10-20 cm, and a bias of ~1 g is an available explanation the filter
      takes; see CROSS_BIAS below.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths                                                       # noqa: E402
from hilti import dataset                                          # noqa: E402

# Why the deskewing note differs per platform. The IMU rate that goes with it comes
# from config/hilti/platforms.yaml, so it is not repeated here.
TIME_COLUMN_NOTE = {
    "robot": "The robot's header stamp marks the scan END, which the\n    # bag converter "
             "already pulled back to the first point (header_at: scan_end).",
    "handheld": "Hesai per-point time is absolute Unix seconds,\n    # rewritten relative by "
                "the converter's hesai_to_velodyne repack.",
}

# Library defaults. On an own-map run the first pose constraints agree with the prior to
# millimetres, so the slack is never called on.
OWN_BIAS = """\
    prior_gyro_bias_sigma: 10
    prior_accel_bias_sigma: 10"""

# Tightened for cross-sequence, carried over from the ConSLAM cross-map evaluation where
# it was the ONLY change that fixed anything. With the library defaults the startup blows
# up vertically at 2-3 m/s inside the first second and dies on `Motion predictor update
# failed`: the first pose constraints disagree with the registered prior by 10-20 cm,
# because the map is a different visit, and ~1 g of accelerometer bias is an available
# explanation -- ImuModel::prior() seeds it with prior_accel_bias_sigma, and 10 m/s^2 is
# about one g of freedom, from which a single 0.1 s prediction step can manufacture a
# metre per second. 2 of 10 ConSLAM startups survived the defaults, 6 of 6 the pair below.
# 0.5 m/s^2 is still ~50x the datasheet bias, so nothing physical is excluded.
CROSS_BIAS = """\
    prior_gyro_bias_sigma: 0.1
    prior_accel_bias_sigma: 0.5"""

OWN_NOTE = """\
# The .kdm was GRAVITY-ALIGNED at build time (see align_map.py): its cloud was rotated
# by R_align so that +Z is up, because kd_localizer seeds its ESKF gravity at the map
# frame's -Z (imu.cpp:139) and a raw FAST-LIO map frame is {tilt} deg off that.
# The rig still starts at the map origin, so only the rotation is non-zero:
#   T_newmap_base(t0) = R_align"""

CROSS_NOTE = """\
# CROSS-SEQUENCE run: the map is {hub}'s, not this sequence's. Two FAST-LIO map frames
# share nothing -- each is its own rig's IMU pose at its own first scan -- so the prior
# below is a registration, not a levelling: src/hilti/xseq.py levelled both clouds with
# their known R_align (collapsing the search to yaw + xyz), matched walls by FFT phase
# correlation, then ran multi-scale point-to-plane ICP. What it produced is
#   T_hubmap_base(t0) = [Rz @ R_align({seq}) | t]
# with {ov} of this sequence's cloud landing within 0.5 m of {hub}'s. The rotation is
# {tilt} deg off vertical because it carries this rig's own mounting as well as the yaw."""

CFG = """\
# kd_localizer config for {seq}: the {seq}_fastlio_bag (LiDAR + IMU + tf_static)
# replayed against {map_desc}. Generated by src/hilti/gen_cfg.py -- see that
# file for why each non-default value is here.
#
{prior_note}
# `rotation` is [roll, pitch, yaw] in RADIANS, R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
initial_pose:
  position: [{p0}, {p1}, {p2}]
  rotation: [{r0}, {r1}, {r2}]     # = [{d0}, {d1}, {d2}] deg

localizer:
  lidar_preprocessing:
    voxel_size: 0.2
    # The repacked clouds carry a float32 `time` column relative to scan start,
    # which is what deskew_lidar_frame reads. {platform_note}
    deskew: true
  motion_predictor:
    type: imu
    gyro_sigma: 0.1
    accel_sigma: 0.1
    gyro_bias_sigma: 1e-3
    accel_bias_sigma: 1e-3
    prior_lin_vel_sigma: 0.1
{bias_priors}
    # FAST-LIO map frames are not gravity aligned; 0.3 rad (17 deg) covers the tilt.
    prior_gravity_sigma: 0.3
    # {imu_hz} Hz IMU, so the 1 ms library default has no margin -- and violating
    # it throws an uncaught std::invalid_argument that kills the node.
    predicted_pose_time_tolerance: 0.02
  initial_align:
    search_radius: 0.5
    match_sigma: 0.1
    max_iterations: 30
  align:
    search_radius: 0.2
    match_sigma: 0.1
    max_iterations: 10
  diagnostics:
    max_pose_sigma_factor_increase: 200
    ray_inlier_error_threshold: 0.4
    # Disabled: on this site the ratio dips near zero on correct poses.
    min_ray_inlier_ratio: 0.0
  num_threads: 8
"""

PARAMS = """\
# kd_localizer_node params for {seq}. `map` is a NODE PARAMETER, not a launch
# argument -- `map:=...` on localizer.launch.py is silently ignored.
kd_localizer_node:
  ros__parameters:
    map: {map_path}
    config_file: {work}/cfg_{seq}.yaml
    map_frame_id: map
    odom_frame_id: odom
    # imu_link, not velodyne: the .kdm frame is the IMU frame at t=0, so this is
    # what makes the reported pose comparable to FAST-LIO and to the survey points.
    base_frame_id: imu_link
    track_only: true
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("-o", "--out", help="output directory "
                                       "(default: datasets.hilti_2023.results)")
    ap.add_argument("--work", help="path the configs will be mounted at inside the "
                                  "container (default: localizer_container.work_mount)")
    ap.add_argument("--map-dir", help="directory holding the .kdm maps "
                                     "(default: datasets.hilti_2023.kdm)")
    ap.add_argument("--map-seq", help="localize against THIS sequence's .kdm instead of "
                                     "the sequence's own (cross-sequence run); needs "
                                     "--prior from src/hilti/xseq.py register")
    ap.add_argument("--prior", help="xseq.py register JSON with the initial pose "
                                   "(default: <results>/xreg_<seq>_in_<map-seq>.json)")
    a = ap.parse_args()

    out = dataset.results_dir(a.out)
    work = a.work or paths.get("localizer_container.work_mount")
    map_dir = a.map_dir or dataset.locations()["kdm"]
    platform = dataset.platform(a.seq)
    hz = dataset.platform_config(a.seq)["imu_hz"]
    note = TIME_COLUMN_NOTE[platform]
    os.makedirs(out, exist_ok=True)
    # align_map.py must have run first: it measures the map frame's gravity tilt and
    # supplies both the rotation the .kdm was built with and the matching initial pose.
    if a.map_seq:
        prior = a.prior or f"{out}/xreg_{a.seq}_in_{a.map_seq}.json"
        al = json.load(open(prior))
        pos = al["T_hub_seq"]["position"]
        prior_note = CROSS_NOTE.format(
            hub=a.map_seq, seq=a.seq, tilt=al["tilt_from_plus_z_deg"],
            ov=f"{100 * al['overlap']['within_0.5m']:.1f} %")
        map_seq, name = a.map_seq, f"{a.seq}_in_{a.map_seq}"
        map_desc = f"the .kdm built from {a.map_seq}'s FAST-LIO cloud"
        bias = CROSS_BIAS
    else:
        al = json.load(open(dataset.align_json(a.seq, a.out)))
        pos = [0.0, 0.0, 0.0]
        prior_note = OWN_NOTE.format(tilt=al["tilt_from_plus_z_deg"])
        map_seq, name = a.seq, a.seq
        map_desc = "the .kdm\n# built from its own FAST-LIO cloud"
        bias = OWN_BIAS
    rpy, deg = al["initial_pose_rotation_rpy"], al["initial_pose_rotation_rpy_deg"]
    open(f"{out}/cfg_{name}.yaml", "w").write(
        CFG.format(seq=a.seq, imu_hz=hz, platform_note=note, map_desc=map_desc,
                   prior_note=prior_note, bias_priors=bias,
                   p0=pos[0], p1=pos[1], p2=pos[2],
                   r0=rpy[0], r1=rpy[1], r2=rpy[2],
                   d0=deg[0], d1=deg[1], d2=deg[2]))
    open(f"{out}/params_{name}.yaml", "w").write(
        PARAMS.format(seq=name, work=work,
                      map_path=f"{map_dir}/{map_seq}_fastlio_map.kdm"))
    print(f"wrote {out}/cfg_{name}.yaml and params_{name}.yaml")


if __name__ == "__main__":
    main()
