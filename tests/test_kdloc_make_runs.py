"""src/kdloc/make_runs.py: campaign.yaml -> one directory of inputs per run.

What matters is that the generated files say exactly what the campaign asked for (a wrong
frame or threshold would silently produce a different test), that the run list is complete
and ordered in passes, and that a finished run's inputs cannot be changed under its results.
"""
import math
from pathlib import Path

import pytest
import yaml

from kdloc import logs, make_runs

CAMPAIGN = """
container: {service: kdloc}
topics: {lidar: /points, imu: /imu}
frames: {base: lidar_link}
static_tf: ["--x 0.1 --frame-id lidar_link --child-frame-id imu_link"]
maps:
  a: maps/a.kdm
sequences:
  s1:
    bag: bags/s1
    initial_pose: {a: {position: [1, 2, 3], orientation: [0, 0, 0.7071067811865476, 0.7071067811865476]}}
  s2:
    bag: bags/s2
groups:
  - {kind: warm, initial_pose: true}
  - {kind: coldcpu, method: bbs3d_cpu}
  - kind: coldgpu_t06
    repeats: 2
    relocalizer_params: {score_threshold_percentage: 0.6}
"""


def make(tmp_path, text=CAMPAIGN, *extra):
    (tmp_path / "campaign.yaml").write_text(text)
    make_runs.main([str(tmp_path), *extra])
    return tmp_path


def names(campaign_dir):
    return [line.split("\t")[0] for line in (campaign_dir / "runs.tsv").read_text().splitlines()[1:]]


def test_runs_are_ordered_in_passes_and_warm_runs_need_an_initial_pose(tmp_path):
    camp = make(tmp_path)
    assert names(camp) == ["warm_s1", "coldcpu_s1", "coldgpu_t06_s1_r1", "coldcpu_s2", "coldgpu_t06_s2_r1",
                           "coldgpu_t06_s1_r2", "coldgpu_t06_s2_r2"]


def test_params_carry_frames_method_and_group_overrides(tmp_path):
    camp = make(tmp_path)
    params = make_runs.load_yaml(camp / "runs/coldgpu_t06_s2_r1/params.yaml")
    loc = params["kd_localizer_node"]["ros__parameters"]
    rel = params["kd_relocalizer_node"]["ros__parameters"]
    assert loc["map"] == str(camp / "maps/a.kdm")
    assert loc["base_frame_id"] == rel["base_frame_id"] == "lidar_link"
    assert loc["config_file"] == str(camp / "runs/coldgpu_t06_s2_r1/cfg.yaml")
    assert loc["publish_tf"] is False and "initial_pose_file" not in loc
    assert rel["method"] == "bbs3d_gpu" and rel["score_threshold_percentage"] == 0.6
    cold_cpu = make_runs.load_yaml(camp / "runs/coldcpu_s1/params.yaml")
    cold_cpu = cold_cpu["kd_relocalizer_node"]["ros__parameters"]
    assert cold_cpu["method"] == "bbs3d_cpu" and "score_threshold_percentage" not in cold_cpu


def test_initial_pose_is_written_as_roll_pitch_yaw(tmp_path):
    camp = make(tmp_path)
    init = yaml.safe_load((camp / "runs/warm_s1/init.yaml").read_text())
    assert init["position"] == [1.0, 2.0, 3.0]
    assert init["rotation"] == pytest.approx([0.0, 0.0, math.pi / 2])
    loc = make_runs.load_yaml(camp / "runs/warm_s1/params.yaml")["kd_localizer_node"]["ros__parameters"]
    assert loc["initial_pose_file"] == str(camp / "runs/warm_s1/init.yaml")


def test_quaternion_to_rpy_round_trip():
    roll, pitch, yaw = 0.1, -0.2, 2.5
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2),
                              math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2))
    q = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
         cr * cp * cy + sr * sp * sy)
    assert make_runs.quaternion_to_rpy(*q) == pytest.approx((roll, pitch, yaw))


def test_run_env_and_static_tf(tmp_path):
    camp = make(tmp_path)
    warm = logs.read_env(camp / "runs/warm_s1/run.env")
    cold = logs.read_env(camp / "runs/coldcpu_s1/run.env")
    assert warm["RELOC_DELAY"] == "30" and cold["RELOC_DELAY"] == "0"
    assert warm["BAG"] == str(camp / "bags/s1") and warm["LIDAR_TOPIC"] == "/points"
    assert warm["DDS_PROFILE"].endswith("config/dds/fastdds_shm_big.xml")
    assert (camp / "runs/warm_s1/static_tf.args").read_text() == \
        "--x 0.1 --frame-id lidar_link --child-frame-id imu_link\n"
    env = logs.read_env(camp / "campaign.env")
    assert env["COMPOSE_SERVICE"] == "kdloc" and env["CAMPAIGN"] == str(camp)


def test_standard_config_and_group_merge(tmp_path):
    text = CAMPAIGN.replace("  - {kind: coldcpu, method: bbs3d_cpu}",
                            "  - {kind: coldcpu, method: bbs3d_cpu,"
                            " localizer_config: {align: {max_iterations: 5}}}")
    camp = make(tmp_path, text)
    cfg = make_runs.load_yaml(camp / "runs/coldcpu_s1/cfg.yaml")
    assert cfg["align"] == {"search_radius": 0.2, "match_sigma": 0.1, "max_iterations": 5}
    standard = make_runs.load_yaml(camp / "runs/warm_s1/cfg.yaml")
    assert standard == make_runs.STANDARD_LOCALIZER_CONFIG


def test_exponent_without_a_dot_reads_as_a_float(tmp_path):
    (tmp_path / "x.yaml").write_text("a: 1e-3\nb: 2E+2\n")
    assert make_runs.load_yaml(tmp_path / "x.yaml") == {"a": 0.001, "b": 200.0}


def test_several_maps_put_the_map_in_the_name(tmp_path):
    text = CAMPAIGN.replace("  a: maps/a.kdm", "  a: maps/a.kdm\n  b: maps/b.kdm")
    camp = make(tmp_path, text)
    assert "warm_s1_in_a" in names(camp) and "warm_s1_in_b" not in names(camp)
    assert "coldcpu_s2_in_b" in names(camp)


@pytest.mark.parametrize("group, message", [
    ("  - {kind: warm, initial_pose: true, repeat: 2}", "unknown key"),
    ("  - {kind: warm, method: bbs3d_tpu}", "method must be one of"),
    ("  - {kind: warm-run}", "`kind` must be"),
    ("  - {kind: warm, sequences: [s9]}", "not defined"),
])
def test_mistakes_are_refused(tmp_path, group, message):
    with pytest.raises(SystemExit) as e:
        make(tmp_path, CAMPAIGN.replace("  - {kind: warm, initial_pose: true}", group))
    assert message in str(e.value)


def test_inputs_of_a_finished_run_are_protected(tmp_path):
    camp = make(tmp_path)
    (camp / "runs/coldcpu_s1/done").touch()
    changed = CAMPAIGN.replace("lidar: /points", "lidar: /other")
    with pytest.raises(SystemExit) as e:
        make(tmp_path, changed)
    assert "coldcpu_s1" in str(e.value)
    assert logs.read_env(camp / "runs/coldcpu_s1/run.env")["LIDAR_TOPIC"] == "/points"
    make(tmp_path, changed, "--force")
    assert logs.read_env(camp / "runs/coldcpu_s1/run.env")["LIDAR_TOPIC"] == "/other"


def test_the_example_campaign_is_valid(tmp_path):
    example = Path(__file__).resolve().parents[1] / "config/examples/kdloc_campaign.example.yaml"
    camp = make(tmp_path, example.read_text())
    assert len(names(camp)) == 2 * (1 + 1 + 1 + 3 + 3)
