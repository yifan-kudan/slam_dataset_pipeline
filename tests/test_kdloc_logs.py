"""src/kdloc/logs.py: readers for the files one localizer run leaves behind.

The log lines below are copied from real kd_relocalizer_node / kd_localizer_node output
(LUH campaign, kd_amr_libs main a35f8f9). The readers are what every campaign's numbers come
from, so each outcome and each counter is pinned here. The fixtures keep the log lines whole,
hence the line-length exemption.
"""
# ruff: noqa: E501
import numpy as np
import pytest

from kdloc import logs

RELOC_LOG = """\
[INFO] [1792092057.500346593] [kd_relocalizer_node]: Relocalization method set to 'bbs3d_gpu'
[INFO] [1792092057.727001500] [kd_relocalizer_node]: Loaded BBS map from /maps/localizer_map.kdm (min_resolution=0.400 m, depth=6)
[WARN] [1792092058.717998938] [kd_relocalizer_node]: Cannot relocalize: no lidar frame available
[INFO] [1792092060.344197290] [kd_relocalizer_node]: Received UNINITIALIZED localization status, starting relocalization (cloud stamp: 1741091997.742853642)
[INFO] [1792092060.374326906] [kd_relocalizer_node]: Voxel downsample (1.000 m): 203776 -> 794 points
[WARN] [1792092060.779675013] [kd_relocalizer_node]: BBS relocalization failed: score below threshold
[WARN] [1792092060.779895554] [kd_relocalizer_node]: Cannot relocalize: no lidar frame available
[INFO] [1792092060.817852803] [kd_relocalizer_node]: Received UNINITIALIZED localization status, starting relocalization (cloud stamp: 1741091998.142941999)
[INFO] [1792092061.817852803] [kd_relocalizer_node]: Received LOST localization status, starting relocalization (cloud stamp: 1741091999.0)
[INFO] [1792092061.846729927] [kd_relocalizer_node]: Voxel downsample (1.000 m): 205355 -> 229 points
[INFO] [1792092062.440384862] [kd_relocalizer_node]: BBS relocalization succeeded: score=211 (92.1%), time=687.2 ms
[INFO] [1792092062.440786136] [kd_relocalizer_node]: Published 3D pose hint x: 49.600002, y: 14.800000, z: -0.800000, yaw: 1.811771
[INFO] [1792092062.441007604] [kd_relocalizer_node]: Ignoring localization status LOST: too soon after last pose hint
"""

NODE_LOG = """\
[kd_localizer-1] [INFO] [1792082330.197413837] [kd_localizer_node]: Loaded initial pose from /x/init.yaml
[kd_localizer-1] [WARN] [1792082334.432013513] [kd_localizer_node]: Timings (ms): Total: 102.80, Update: 102.67, Align: 39.20, MapUpdate: 56.14
[kd_localizer-1] [INFO] [1792082340.403151974] [kd_localizer_node]: Process timing (ms) over 60 updates: process avg=80.00 max=102.80
[kd_localizer-1] [INFO] [1792082350.403151974] [kd_localizer_node]: Process timing (ms) over 40 updates: process avg=90.00 max=99.00
"""

TEGRA_LOG = (
    "10-15-2026 17:38:47 RAM 2094/30684MB (lfb 3x2MB) SWAP 1056/15342MB (cached 1MB) "
    "CPU [20%@1574,40%@1574] GR3D_FREQ 10% cpu@47.125C VDD_GPU_SOC 3490mW/3490mW VDD_CPU_CV 1552mW/1552mW\n"
    "10-15-2026 17:38:48 RAM 2100/30684MB (lfb 2x4MB) SWAP 1056/15342MB (cached 1MB) "
    "CPU [30%@729,50%@729] GR3D_FREQ 30% cpu@47.031C VDD_GPU_SOC 3104mW/3297mW VDD_CPU_CV 1552mW/1552mW\n")

POSE_HEADER = ("wall_s,stamp_s,status,lost_reason,ray_inlier_ratio,sigma_factor,pose_stamp_s,x,y,z,qx,qy,qz,qw,"
               "sig_pos_x,sig_pos_y,sig_pos_z,sig_rot_x,sig_rot_y,sig_rot_z,odometry_mode,odometry_distance\n")


def pose_row(stamp, status, pose_stamp=None, odometry=0, reason=""):
    pose = (f"{pose_stamp},{pose_stamp},0,0,0,0,0,1" if pose_stamp is not None else ",,,,,,,")
    return f"0,{stamp},{status},\"{reason}\",,1.0,{pose},,,,,,,{odometry},0\n"


def test_relocalizer_attempts_and_outcomes(tmp_path):
    (tmp_path / "reloc.log").write_text(RELOC_LOG)
    attempts, info = logs.parse_relocalizer_log(tmp_path / "reloc.log")
    assert [a["outcome"] for a in attempts] == ["below_threshold", "no_answer", "succeeded"]
    assert [a["trigger"] for a in attempts] == ["UNINITIALIZED", "UNINITIALIZED", "LOST"]
    assert attempts[0]["cloud_stamp"] == pytest.approx(1741091997.742853642)
    assert attempts[0]["points"] == 794
    assert attempts[0]["search_ms"] == pytest.approx(435.5, abs=0.1)
    ok = attempts[2]
    assert (ok["score"], ok["score_pct"], ok["search_ms"]) == (211, 92.1, 687.2)
    assert ok["hint"] == pytest.approx([49.6, 14.8, -0.8, 1.811771])
    assert info["map_loaded"] and info["no_lidar_frame"] == 2 and info["ignored_status"] == 1
    assert logs.relocalizer_counts(attempts)["succeeded"] == 1


def test_node_log_weights_the_windows_by_update_count(tmp_path):
    (tmp_path / "node.log").write_text(NODE_LOG + "[x] Failed to create segment\n")
    node = logs.parse_node_log(tmp_path / "node.log")
    assert node["updates"] == 100
    assert node["process_avg_ms"] == pytest.approx(84.0)
    assert node["process_max_ms"] == 102.8 and node["slow_updates"] == 1
    assert node["slow_update_mean_ms"]["map_update"] == pytest.approx(56.14)
    assert node["udp_fallback"] and node["initial_pose_loaded"]


def test_status_summary(tmp_path):
    rows = (POSE_HEADER + pose_row(100.0, "UNINIT") + pose_row(100.5, "TRACKING", 100.4)
            + pose_row(100.6, "TRACKING", 100.4) + pose_row(101.0, "TRACKING", 100.9, odometry=1)
            + pose_row(105.0, "LOST", 104.9, reason="Odometry distance exceeded")
            + pose_row(106.0, "TRACKING", 109.8))
    (tmp_path / "pose.csv").write_text(rows)
    parsed = logs.read_pose_csv(tmp_path / "pose.csv")
    assert len(logs.tracking_poses(parsed)) == 3          # the repeated pose stamp counts once
    s = logs.status_summary(parsed, 100.0, 110.0)
    assert s["lost_events"] == [{"at_s": 5.0, "at_share": 0.5, "reason": "Odometry distance exceeded"}]
    assert s["status_share"]["TRACKING"] == pytest.approx(4 / 6)
    assert s["first_tracking_s"] == pytest.approx(0.4) and s["odometry_mode_share"] == pytest.approx(1 / 3)
    assert s["tracked_to_end"] and not s["success"]


def test_tegrastats_and_procmon(tmp_path):
    (tmp_path / "tegra.log").write_text(TEGRA_LOG)
    samples = logs.parse_tegrastats(tmp_path / "tegra.log", "+0100")
    assert samples[0][0] == pytest.approx(1792082327.0)    # 17:38:47 BST = 16:38:47 UTC
    summary = logs.tegrastats_summary(samples)
    assert summary["ram_used_mb_max"] == 2100 and summary["cpu_pct_mean"] == 35.0
    assert summary["gpu_pct_max"] == 30 and summary["gpu_power_mw_mean"] == 3297
    (tmp_path / "proc.csv").write_text("t_s,kd_localizer_cpu_cores,kd_localizer_rss_mb,kd_localizer_threads,"
                                       "kd_relocalizer_cpu_cores,kd_relocalizer_rss_mb,kd_relocalizer_threads\n"
                                       "1.0,,,,,,\n2.0,1.5,500,20,,,\n3.0,2.5,520,20,0.1,300,8\n")
    proc = logs.parse_procmon(tmp_path / "proc.csv")
    assert proc["kd_localizer"]["cpu_cores_mean"] == 2.0 and proc["kd_localizer"]["rss_mb_max"] == 520.0
    assert proc["kd_relocalizer"]["cpu_cores_max"] == 0.1


def test_bag_info_cuts_a_limited_run(tmp_path):
    (tmp_path / "metadata.yaml").write_text("""
rosbag2_bagfile_information:
  duration: {nanoseconds: 100000000000}
  starting_time: {nanoseconds_since_epoch: 1700000000000000000}
  topics_with_message_count:
    - topic_metadata: {name: /points}
      message_count: 1000
""")
    full = logs.bag_info(tmp_path / "metadata.yaml")
    assert (full["start"], full["end"], full["message_counts"]["/points"]) == (1.7e9, 1.7e9 + 100, 1000)
    part = logs.bag_info(tmp_path / "metadata.yaml", limit_s=60)
    assert part["end"] == 1.7e9 + 60 and part["message_counts"]["/points"] == 600


def test_percentile_matches_numpy():
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
    for q in (0, 50, 95, 100):
        assert logs.percentile(values, q) == pytest.approx(np.percentile(values, q))


def test_summary_flags_invalid_runs(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "run.env").write_text("NAME=x\nLIMIT=0\n")
    (out / "pose.csv").write_text(POSE_HEADER + pose_row(100.5, "TRACKING", 100.4))
    (out / "node.log").write_text(NODE_LOG + "Failed to create segment\n")
    (out / "play.log").write_text("[ERROR] SQLite error: database disk image is malformed\n")
    lines = logs.summary_lines(tmp_path)
    assert sum(line.startswith("[INVALID]") for line in lines) == 2
