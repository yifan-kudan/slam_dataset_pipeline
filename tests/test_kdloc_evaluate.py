"""src/kdloc/evaluate.py: accuracy (evo), relocalization grading and the run summary.

Synthetic trajectories with a known answer: a cart moving at 1 m/s along x, a localizer that
is a fixed 5 cm off, skips scans and loses track for a while. The evaluator has to recover the
5 cm, score only where it should, and grade pose hints against the reference.
"""
import json
import math

import numpy as np
import pytest

pytest.importorskip("evo")

from kdloc import evaluate  # noqa: E402

BAG_START = 1000.0
POSE_HEADER = ("wall_s,stamp_s,status,lost_reason,ray_inlier_ratio,sigma_factor,"
               "pose_stamp_s,x,y,z,qx,qy,qz,qw,sig_pos_x,sig_pos_y,sig_pos_z,sig_rot_x,sig_rot_y,sig_rot_z,odometry_mode,odometry_distance\n")


def write_campaign(tmp_path, pose_rows, reloc_log="", identity_reference=False):
    run = tmp_path / "runs" / "cold_s1"
    out = run / "out"
    out.mkdir(parents=True)
    (run / "run.env").write_text("NAME=cold_s1\nKIND=cold\nSEQUENCE=s1\nMAP_NAME=m\nLIDAR_TOPIC=/points\n")
    (out / "bag_metadata.yaml").write_text(f"""
rosbag2_bagfile_information:
  duration: {{nanoseconds: 60000000000}}
  starting_time: {{nanoseconds_since_epoch: {int(BAG_START * 1e9)}}}
  topics_with_message_count:
    - topic_metadata: {{name: /points}}
      message_count: 600
""")
    (out / "pose.csv").write_text(POSE_HEADER + "".join(pose_rows))
    (out / "reloc.log").write_text(reloc_log)
    (out / "node.log").write_text("Process timing (ms) over 500 updates: process avg=50.00 max=90.00\n")
    ref = tmp_path / "ref"
    ref.mkdir()
    with open(ref / "s1.tum", "w") as f:
        # Reference at the scan instants, heading 90 deg (yaw pi/2) unless position-only.
        qz, qw = (0.0, 1.0) if identity_reference else (math.sin(math.pi / 4), math.cos(math.pi / 4))
        for k in range(601):
            t = BAG_START + 0.1 * k
            f.write(f"{t:.3f} {t - BAG_START:.4f} 0 0 0 0 {qz:.8f} {qw:.8f}\n")
    return tmp_path, ref


def localizer_rows(lost_from=None, lost_to=None, jump_after_lost=0.0):
    """TRACKING poses 5 cm off in y, stamped 0.03 s after the reference rows, every 4th scan
    skipped, LOST between lost_from and lost_to (bag seconds)."""
    rows = []
    for k in range(600):
        if k % 4 == 3:
            continue
        t = BAG_START + 0.1 * k + 0.03
        rel = t - BAG_START
        if lost_from is not None and lost_from <= rel < lost_to:
            rows.append(f"0,{t:.3f},LOST,\"Odometry distance exceeded\",,1.0,,,,,,,,,,,,,,,1,0\n")
            continue
        x = rel + (jump_after_lost if lost_to is not None and rel >= lost_to else 0.0)
        odometry = 1 if k % 2 else 0
        rows.append(f"0,{t:.3f},TRACKING,\"\",,1.0,{t:.6f},{x:.4f},0.0500,0,0,0,0.70710678,0.70710678,"
                    f",,,,,,{odometry},0\n")
    return rows


def run_evaluate(campaign, ref, *extra):
    evaluate.main([str(campaign), "--reference", str(ref), *extra])
    return json.loads((campaign / "eval" / "cold_s1.json").read_text())


def test_constant_offset_is_recovered(tmp_path):
    campaign, ref = write_campaign(tmp_path, localizer_rows())
    result = run_evaluate(campaign, ref)
    acc = result["accuracy"]
    assert acc["mean_cm"] == pytest.approx(5.0, abs=0.01) and acc["max_cm"] == pytest.approx(5.0, abs=0.01)
    assert acc["mean_xy_cm"] == pytest.approx(5.0, abs=0.01) and acc["mean_abs_z_cm"] == pytest.approx(0.0)
    assert acc["scored_poses"] == 450 and acc["high_accuracy"]
    assert acc["mean_cm_odometry_mode"] == pytest.approx(5.0, abs=0.01)
    assert result["status"]["success"]
    assert result["processing"]["updates_per_scan"] == pytest.approx(500 / 600, abs=1e-4)
    assert (campaign / "eval" / "cold_s1.est.tum").read_text().count("\n") == 450


def test_lost_stretch_is_not_scored_and_time_off_is_counted(tmp_path):
    rows = localizer_rows(lost_from=20.0, lost_to=30.0, jump_after_lost=2.0)
    campaign, ref = write_campaign(tmp_path, rows)
    result = run_evaluate(campaign, ref)
    acc, status = result["accuracy"], result["status"]
    assert status["lost_events"][0]["at_s"] == pytest.approx(20.03) and not status["success"]
    assert acc["before_first_lost"]["max_cm"] == pytest.approx(5.0, abs=0.01)
    # 2 m off from 30 s to the end of the 60 s bag
    assert acc["seconds_over_1m"] == pytest.approx(30.0, abs=0.3)
    assert acc["max_cm"] == pytest.approx(100 * math.hypot(2.0, 0.05), abs=0.1)


HINT = ("[INFO] [2000.0] [kd_relocalizer_node]: Received UNINITIALIZED localization status, "
        "starting relocalization (cloud stamp: {stamp})\n"
        "[INFO] [2000.3] [kd_relocalizer_node]: "
        "BBS relocalization succeeded: score=50 (70.0%), time=300.0 ms\n"
        "[INFO] [2000.3] [kd_relocalizer_node]: Published 3D pose hint x: {x}, y: 0.2, z: 0.0, yaw: {yaw}\n")


def test_pose_hints_are_graded_against_the_reference(tmp_path):
    reloc = (HINT.format(stamp=BAG_START + 10.0, x=10.35, yaw=math.pi / 2 + 0.05)
             + HINT.format(stamp=BAG_START + 40.0, x=52.0, yaw=math.pi / 2))
    campaign, ref = write_campaign(tmp_path, localizer_rows(), reloc_log=reloc)
    rl = run_evaluate(campaign, ref)["relocalization"]
    first, second = rl["results"]
    assert first["correct"] and first["error_xy_m"] == pytest.approx(math.hypot(0.30, 0.2), abs=1e-3)
    assert first["error_yaw_deg"] == pytest.approx(math.degrees(0.05), abs=0.01)
    assert not second["correct"] and second["error_xy_m"] > 11
    assert (rl["correct"], rl["wrong"]) == (1, 1)
    assert rl["first_attempt_to_tracking_s"] == pytest.approx(-9.97)


def test_position_only_reference_skips_the_heading_check(tmp_path):
    reloc = HINT.format(stamp=BAG_START + 10.0, x=10.35, yaw=3.0)
    campaign, ref = write_campaign(tmp_path, localizer_rows(), reloc_log=reloc, identity_reference=True)
    result = run_evaluate(campaign, ref)["relocalization"]["results"][0]
    assert result["correct"] and result["error_yaw_deg"] is None


def test_no_reference_still_reports_status(tmp_path):
    campaign, _ = write_campaign(tmp_path, localizer_rows())
    evaluate.main([str(campaign)])
    result = json.loads((campaign / "eval" / "cold_s1.json").read_text())
    assert result["accuracy"] is None and result["status"]["success"]
    assert (campaign / "eval" / "summary.csv").read_text().startswith("name,kind,sequence")


def test_interpolation_never_spans_a_long_gap(tmp_path):
    """A localizer denser than the reference gets interpolated by evo. A pair whose
    interpolation would span the LOST stretch, or hold the first or last pose for a time
    outside the trajectory, has to be dropped rather than scored."""
    rows = []
    for k in range(1200):                      # 20 Hz, 0.03 s after the reference rows
        t = BAG_START + 0.05 * k + 0.03
        rel = t - BAG_START
        if 20.0 <= rel < 30.0:                 # LOST, then 2 m off
            continue
        x = rel + (2.0 if rel >= 30.0 else 0.0)
        rows.append(f"0,{t:.3f},TRACKING,\"\",,1.0,{t:.6f},{x:.4f},0.0,0,0,0,0.70710678,0.70710678,,,,,,,0,0\n")
    campaign, ref = write_campaign(tmp_path, rows)
    result = run_evaluate(campaign, ref, "--t-max-diff", "0.04")
    errors = np.array(result["error_series"]["err_cm"])
    assert np.all((errors < 0.1) | (np.abs(errors - 200.0) < 0.1))
    # 601 reference rows - 99 inside the gap (no pose within 0.04 s) - 2 at its edges
    # (20.0 and 30.0 would interpolate across it) - 2 outside the trajectory (0.0 and 60.0)
    assert result["accuracy"]["scored_poses"] == 498
