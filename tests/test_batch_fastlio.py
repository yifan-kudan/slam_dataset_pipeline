"""Bag discovery in batch_fastlio.py.

``--root`` is the dataset root (it is where ``<sequence>/pose.csv`` is looked up),
but batch_to_bag.py writes its bags to ``<root>/Converted_bag/``. Searching only
one level under ``--root`` therefore found nothing, and pointing ``--root`` at
``Converted_bag`` instead silently broke ``--eval-odom``. These cover the
resolution order that fixes it, plus the sequence-name derivation the eval path
depends on.
"""
import pytest

import batch_fastlio as bf


def make(tmp_path, layout: dict[str, list[str]]):
    for sub, names in layout.items():
        base = tmp_path / sub if sub else tmp_path
        for n in names:
            (base / n).mkdir(parents=True, exist_ok=True)
    return str(tmp_path)


# --------------------------------------------------------------------------
# Root resolution
# --------------------------------------------------------------------------
def test_converted_bag_is_searched_before_the_root(tmp_path):
    root = make(tmp_path, {"": ["20230706"], "Converted_bag": ["20230706_pc2_bag"]})
    roots = bf.bag_roots_for(root, None)
    bags, used = bf.find_bags(roots, "*_pc2_bag")
    assert used.endswith("Converted_bag")
    assert [b.split("/")[-1] for b in bags] == ["20230706_pc2_bag"]


def test_falls_back_to_the_root_when_there_is_no_converted_bag_dir(tmp_path):
    """The older layout: bags sit next to the sequence folders."""
    root = make(tmp_path, {"": ["seq1", "seq1_fastlio_bag"]})
    bags, used = bf.find_bags(bf.bag_roots_for(root, None), "*_fastlio_bag")
    assert used == root
    assert [b.split("/")[-1] for b in bags] == ["seq1_fastlio_bag"]


def test_falls_back_to_the_root_when_converted_bag_has_no_match(tmp_path):
    """Converted_bag exists but holds a different flavour of bag, and the wanted
    ones are next to the sequences -- take the root rather than reporting none."""
    root = make(tmp_path, {"": ["seq1", "seq1_fastlio_bag"],
                           "Converted_bag": ["seq1_pc2_bag"]})
    bags, used = bf.find_bags(bf.bag_roots_for(root, None), "*_fastlio_bag")
    assert used == root
    assert [b.split("/")[-1] for b in bags] == ["seq1_fastlio_bag"]


def test_explicit_bag_root_wins(tmp_path):
    root = make(tmp_path, {"": ["20230706"], "Converted_bag": ["20230706_pc2_bag"],
                           "elsewhere": ["20230706_pc2_bag"]})
    roots = bf.bag_roots_for(root, str(tmp_path / "elsewhere"))
    assert roots == [str(tmp_path / "elsewhere")]
    _, used = bf.find_bags(roots, "*_pc2_bag")
    assert used.endswith("elsewhere")


def test_no_match_anywhere_reports_empty_rather_than_raising(tmp_path):
    root = make(tmp_path, {"": ["20230706"], "Converted_bag": ["20230706_pc2_bag"]})
    bags, _ = bf.find_bags(bf.bag_roots_for(root, None), "*_custom_bag")
    assert bags == []


def test_missing_root_directory_is_skipped_not_crashed(tmp_path):
    """bag_roots_for can hand find_bags a Converted_bag that vanished mid-run."""
    bags, _ = bf.find_bags([str(tmp_path / "gone"), str(tmp_path)], "*_bag")
    assert bags == []


# --------------------------------------------------------------------------
# Positional bag arguments
# --------------------------------------------------------------------------
def test_relative_positional_bag_resolves_against_converted_bag(tmp_path):
    root = make(tmp_path, {"": ["20230706"], "Converted_bag": ["20230706_pc2_bag"]})
    got = bf.resolve_bag_path("20230706_pc2_bag", bf.bag_roots_for(root, None))
    assert got == str(tmp_path / "Converted_bag" / "20230706_pc2_bag")


def test_absolute_positional_bag_is_left_alone(tmp_path):
    root = make(tmp_path, {"Converted_bag": ["b_bag"]})
    absolute = "/somewhere/else/b_bag"
    assert bf.resolve_bag_path(absolute, bf.bag_roots_for(root, None)) == absolute


def test_unresolvable_positional_bag_returns_first_root_for_reporting(tmp_path):
    root = make(tmp_path, {"Converted_bag": ["b_bag"]})
    got = bf.resolve_bag_path("nope_bag", bf.bag_roots_for(root, None))
    assert got == str(tmp_path / "Converted_bag" / "nope_bag")


# --------------------------------------------------------------------------
# Sequence name -- decides where pose.csv is looked up
# --------------------------------------------------------------------------
def test_sequence_name_from_a_plain_suffix_glob():
    assert bf.sequence_name("20230706_pc2_bag", "*_pc2_bag") == "20230706"
    assert bf.sequence_name("20230706_custom_bag", "*_custom_bag") == "20230706"


def test_sequence_name_falls_back_to_the_regex_on_a_mismatched_glob():
    """The suffix strip only applies when the glob is a plain `*<suffix>`; a bag
    run positionally under the default glob still has to resolve."""
    assert bf.sequence_name("20230706_pc2_bag", "*_custom_bag") == "20230706"
    assert bf.sequence_name("20230706_bag", "*") == "20230706"


def test_sequence_name_leaves_a_name_without_a_bag_suffix_alone():
    assert bf.sequence_name("20230706", "*_pc2_bag") == "20230706"


def test_only_matches_a_sequence_name_for_a_digit_bearing_suffix(tmp_path):
    """`--only 20230706` against `20230706_pc2_bag`: the sequence name has to come
    out as `20230706`, which the old `[a-z]+` fallback could not produce."""
    bags = [str(tmp_path / "20230706_pc2_bag"), str(tmp_path / "20230809_pc2_bag")]
    kept = bf.select(bags, None, ["20230706"], "*_pc2_bag")
    assert kept == [bags[0]]


def test_only_still_matches_the_bag_directory_name(tmp_path):
    bags = [str(tmp_path / "20230706_pc2_bag")]
    assert bf.select(bags, None, ["20230706_pc2_bag"], "*_pc2_bag") == bags


def test_start_from_uses_the_sequence_name_too(tmp_path):
    bags = [str(tmp_path / f"{s}_pc2_bag") for s in ("20230531", "20230706", "20230809")]
    assert bf.select(bags, "20230706", None, "*_pc2_bag") == bags[1:]


# --------------------------------------------------------------------------
# Frame accounting
#
# The silent failure this exists for: FAST-LIO processed ~50 % of the clouds on this
# host for weeks (a writer-side shared-memory drop), and every map, trajectory and
# report number was quietly computed from half the data. A run that consumed half the
# bag has to fail, not warn in passing.
# --------------------------------------------------------------------------
import argparse                                                       # noqa: E402
import textwrap                                                       # noqa: E402


def make_bag(tmp_path, clouds=929, topic="/velodyne_points", imu=37243):
    bag = tmp_path / "seq_fastlio_bag"
    bag.mkdir(parents=True, exist_ok=True)
    (bag / "metadata.yaml").write_text(textwrap.dedent(f"""
        rosbag2_bagfile_information:
          version: 8
          duration:
            nanoseconds: 93300000000
          topics_with_message_count:
            - topic_metadata:
                name: {topic}
                type: sensor_msgs/msg/PointCloud2
              message_count: {clouds}
            - topic_metadata:
                name: /imu
                type: sensor_msgs/msg/Imu
              message_count: {imu}
        """), encoding="utf-8")
    return bag


def make_pos_log(bag, rows):
    log_dir = bag / "fastlio_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "pos_log.txt").write_text("".join(f"{i} 0 0 0\n" for i in range(rows)),
                                         encoding="utf-8")
    return str(log_dir)


def test_cloud_count_is_found_by_type_not_by_topic_name(tmp_path):
    bag = make_bag(tmp_path, clouds=2053, topic="/livox/lidar")
    assert bf.bag_cloud_count(str(bag)) == ("/livox/lidar", 2053)


def test_a_full_run_passes(tmp_path):
    bag = make_bag(tmp_path, clouds=929)
    job = {"bag": str(bag), "fastlio_log_dir": make_pos_log(bag, 926)}
    message, ok = bf.frame_report(job, 95.0)
    assert ok and "926/929 (99.7%)" in message


def test_half_the_clouds_fails_the_run_and_says_why(tmp_path):
    """The exact site2_handheld_4 numbers from before the DDS profile."""
    bag = make_bag(tmp_path, clouds=929)
    job = {"bag": str(bag), "fastlio_log_dir": make_pos_log(bag, 457)}
    message, ok = bf.frame_report(job, 95.0)
    assert not ok
    assert "457/929 (49.2%)" in message and "dds-profile" in message


def test_the_check_can_be_switched_off(tmp_path):
    bag = make_bag(tmp_path, clouds=929)
    job = {"bag": str(bag), "fastlio_log_dir": make_pos_log(bag, 457)}
    assert bf.frame_report(job, 0.0)[1] is True


def test_a_missing_log_or_cloud_topic_does_not_fail_the_run(tmp_path):
    bag = make_bag(tmp_path, clouds=929)
    job = {"bag": str(bag), "fastlio_log_dir": str(tmp_path / "nope")}
    message, ok = bf.frame_report(job, 95.0)
    assert ok and "unreadable" in message

    bare = tmp_path / "bare_fastlio_bag"
    bare.mkdir()
    message, ok = bf.frame_report({"bag": str(bare), "fastlio_log_dir": ""}, 95.0)
    assert ok and "not checked" in message


# --------------------------------------------------------------------------
# The DDS profile, exported for the node and the player from one environment
# --------------------------------------------------------------------------
def test_the_profile_is_exported_for_every_child(tmp_path):
    profile = tmp_path / "shm.xml"
    profile.write_text("<dds/>", encoding="utf-8")
    env = bf.with_dds_profile({"PATH": "/usr/bin"}, str(profile))
    assert env["FASTRTPS_DEFAULT_PROFILES_FILE"] == str(profile)
    assert env["PATH"] == "/usr/bin"


def test_none_runs_without_a_profile_and_a_missing_file_is_fatal(tmp_path):
    assert "FASTRTPS_DEFAULT_PROFILES_FILE" not in bf.with_dds_profile({}, "none")
    with pytest.raises(SystemExit, match="DDS profile not found"):
        bf.with_dds_profile({}, str(tmp_path / "absent.xml"))


# --------------------------------------------------------------------------
# --run-name: a sweep must not overwrite the outputs a report was built from
# --------------------------------------------------------------------------
def job_args(**over):
    args = dict(glob="*_fastlio_bag", run_name=None, odom_dir=None, record_odom=False,
                fastlio_log_dir=None, ros_log_dir=None, root="/root")
    args.update(over)
    return argparse.Namespace(**args)


def test_without_a_run_name_the_canonical_paths_are_used():
    job = bf.build_jobs(["/bags/site2_robot_1_fastlio_bag"], job_args(), "cfg.yaml")[0]
    assert job["map"] == "/bags/site2_robot_1_fastlio_bag/maps/fastlio_map.pcd"
    assert job["fastlio_log_dir"] == "/bags/site2_robot_1_fastlio_bag/fastlio_log"


def test_a_run_name_moves_every_output_under_runs_and_names_the_map():
    job = bf.build_jobs(["/bags/site2_robot_1_fastlio_bag"],
                        job_args(run_name="qosfix_r0.3"), "cfg.yaml")[0]
    run = "/bags/site2_robot_1_fastlio_bag/runs/qosfix_r0.3"
    assert job["map"] == f"{run}/maps/site2_robot_1_fastlio_map.pcd"
    assert job["fastlio_log_dir"] == f"{run}/fastlio_log"
    assert job["ros_log_dir"] == f"{run}/ros_logs"
    assert job["odom_dir"] == f"{run}/results"
