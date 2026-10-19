"""The Hilti layout and rig facts, which decide whether a score means anything.

Two things here are not cosmetic. Pairing a map with another run's pos_log rotates
the FAST-LIO reference by the wrong R_align, so `run` has to come from the sequence
config and follow through to every derived path. And the prism lever arm differs per
platform -- using the handheld's 20 cm boom on the robot invents ~1.5 m of error --
so the platform lookup is covered directly.
"""
import numpy as np
import pytest

import paths
from hilti import dataset


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Point the dataset locations at a scratch tree, leaving the real configs alone."""
    file = tmp_path / "paths.yaml"
    file.write_text(
        "datasets:\n"
        "  hilti_2023:\n"
        f"    root: {tmp_path}/bags\n"
        f"    raw: {tmp_path}/raw\n"
        "    groundtruth: ${datasets.hilti_2023.raw}/groundtruth_2023\n"
        "    kdm: ${datasets.hilti_2023.root}/maps/kdm\n"
        f"    results: {tmp_path}/results\n"
        "localizer_container:\n"
        "  work_mount: /work\n", encoding="utf-8")
    monkeypatch.setenv("SDP_PATHS", str(file))
    paths.reload()
    yield tmp_path
    paths.reload()


# --------------------------------------------------------------------------
# Sequence config
# --------------------------------------------------------------------------
def test_every_sequence_has_a_run_and_the_default_is_filled_in():
    seqs = dataset.sequences()
    assert len(seqs) == 15
    assert all(entry["run"] for entry in seqs.values())
    assert seqs["site2_robot_1"]["run"] == "r0.3"


def test_the_sequence_whose_map_was_rebuilt_keeps_its_own_run():
    """site2_handheld_6's .kdm came from its r0.15 cloud, not the r0.3 one."""
    assert dataset.sequence("site2_handheld_6")["run"] == "r0.15"
    assert dataset.sequence("site2_handheld_6")["score_tag"] == "r015"


def test_an_unknown_sequence_lists_the_known_ones():
    with pytest.raises(SystemExit, match="site2_robot_1"):
        dataset.sequence("site9_nope")


# --------------------------------------------------------------------------
# Rig facts
# --------------------------------------------------------------------------
def test_the_handheld_arm_is_20_cm_and_the_robot_has_no_constant_arm():
    arm = dataset.lever_arm("site3_handheld_2")
    assert np.isclose(np.linalg.norm(arm), 0.205, atol=0.01)
    # The robot's prism is given per control point in <seq>_lidar.txt instead.
    assert dataset.lever_arm("site2_robot_1") is None


def test_the_robot_extrinsic_is_a_rigid_4x4():
    T = dataset.lidar_from_imu("site2_robot_1")
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0, 0, 0, 1])
    assert np.isclose(abs(np.linalg.det(T[:3, :3])), 1.0, atol=1e-6)


def test_platform_comes_from_the_sequence_table_and_falls_back_to_the_name():
    assert dataset.platform("site2_robot_2") == "robot"
    assert dataset.platform("site3_handheld_1") == "handheld"
    assert dataset.platform("site9_robot_9") == "robot"      # not in the table


def test_static_tf_is_a_unit_quaternion_per_platform():
    for seq in ("site2_robot_1", "site1_handheld_1"):
        tf = dataset.static_tf(seq)
        assert len(tf["translation"]) == 3
        assert np.isclose(np.linalg.norm(tf["rotation_xyzw"]), 1.0, atol=1e-3)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def test_paths_follow_the_configured_root(layout):
    assert dataset.bag_dir("site2_robot_1") == f"{layout}/bags/site2_robot_1_fastlio_bag"
    assert dataset.kdm("site2_robot_1").endswith("maps/kdm/site2_robot_1_fastlio_map.kdm")
    assert dataset.monitor_csv("site2_robot_1") == f"{layout}/results/run_site2_robot_1.csv"


def test_the_run_directory_defaults_to_the_sequences_run(layout):
    assert dataset.run_dir("site2_robot_1").endswith("_fastlio_bag/runs/r0.3")
    assert dataset.run_dir("site2_handheld_6").endswith("_fastlio_bag/runs/r0.15")
    assert dataset.pos_log("site2_handheld_6").endswith("runs/r0.15/fastlio_log/pos_log.txt")


def test_an_explicit_run_overrides_it_and_the_empty_run_is_the_bag_itself(layout):
    assert dataset.run_dir("site2_robot_1", "qosfix_r0.3").endswith("runs/qosfix_r0.3")
    assert dataset.run_dir("site2_robot_1", "") == dataset.bag_dir("site2_robot_1")


def test_score_file_names_carry_the_tag_that_distinguishes_runs(layout):
    assert dataset.score_json("site2_robot_1", "fl").endswith("fl_site2_robot_1.json")
    # Without the tag site2_handheld_6's two runs would overwrite each other's score.
    assert dataset.score_json("site2_handheld_6", "fl").endswith("fl_site2_handheld_6_r015.json")
    assert dataset.score_json("site2_handheld_6", "fl", "").endswith("fl_site2_handheld_6.json")
