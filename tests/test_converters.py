"""Regression tests for the failure modes that used to be silent.

Every case here corresponds to something that produced a wrong bag (or destroyed a
good one) without raising: an out-of-range cast that wrapped, a NaN that became
INT64_MIN, an output deleted before its replacement existed. They are cheap and
need no ROS install -- ``rosbags`` is pure python.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer as Rosbag1Writer
from rosbags.rosbag2 import Reader

import batch_to_bag
import rosbag1_to_ros2
import to_bag
from ros2_bag_io import (
    ROS1_STORE, apply_distro, atomic_bag, bag_version, build_camera_info, check_window,
    filter_groups, load_yaml, select_groups, write_tf_static,
)

CFG: dict = {"output": {"storage_id": "sqlite3", "version": 9}}


# --------------------------------------------------------------------------
# Timestamps: fatal, because they define the bag's clock
# --------------------------------------------------------------------------
def test_int_ns_stamps_are_exact():
    """1.7e18 ns is past float64's exact-integer range; the int path must not
    round trip through a float multiply."""
    raw = np.array([1685500000123456789, 1685500000223456789], dtype=np.int64)
    out = to_bag.stamps_to_ns(raw, "ns", 0)
    assert out.tolist() == raw.tolist()


def test_stamp_shift_is_applied():
    out = to_bag.stamps_to_ns(np.array([1_000_000_000], dtype=np.int64), "ns", -250)
    assert out.tolist() == [999_999_750]


def test_seconds_stamps_are_scaled():
    out = to_bag.stamps_to_ns(np.array([1.5, 2.25]), "s", 0)
    assert out.tolist() == [1_500_000_000, 2_250_000_000]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_stamp_is_fatal(bad):
    """Used to cast to INT64_MIN and be written to the bag as a valid-looking stamp."""
    with pytest.raises(SystemExit, match="NaN/Inf"):
        to_bag.stamps_to_ns(np.array([1.0, bad]), "s", 0, "/points")


def test_out_of_int64_range_stamp_is_fatal():
    with pytest.raises(SystemExit, match="int64"):
        to_bag.stamps_to_ns(np.array([1e30]), "s", 0, "/points")


def test_uint64_above_int64_is_fatal():
    """astype(np.int64) wraps such a value negative without complaint."""
    with pytest.raises(SystemExit, match="int64"):
        to_bag.stamps_to_ns(np.array([2**63 + 5], dtype=np.uint64), "ns", 0, "/points")


def test_stamp_shift_overflow_is_fatal():
    stamps = np.array([np.iinfo(np.int64).max - 10], dtype=np.int64)
    with pytest.raises(SystemExit, match="stamp shift"):
        to_bag.stamps_to_ns(stamps, "ns", 1000, "/points")


def test_empty_stamps_are_allowed():
    assert to_bag.stamps_to_ns(np.array([], dtype=np.int64), "ns", 5).tolist() == []


# --------------------------------------------------------------------------
# Narrow output columns: clamped, not wrapped
# --------------------------------------------------------------------------
def test_uint32_overflow_clamps_instead_of_wrapping():
    """2**32 + 5 used to become 5 -- a point 4.3 s into the scan read as 5 ns in."""
    out = to_bag.clamp_to_type(np.array([2.0**32 + 5]), "u4", "/points:clamp", "offset_time")
    assert out.astype(np.uint32).tolist() == [2**32 - 1]


def test_negative_and_non_finite_clamp_to_zero():
    out = to_bag.clamp_to_type(np.array([-7.0, np.nan]), "u4", "/points:clamp", "offset_time")
    assert out.astype(np.uint32).tolist() == [0, 0]


def test_in_range_values_pass_through_unchanged(capsys):
    vals = np.array([0.0, 1234.0, float(2**32 - 1)])
    out = to_bag.clamp_to_type(vals, "u4", "/points:clamp:pass", "offset_time")
    assert out.tolist() == vals.tolist()
    assert capsys.readouterr().out == ""      # no warning when nothing was clamped


def test_clamping_is_reported(capsys):
    to_bag.clamp_to_type(np.array([2.0**32]), "u4", "/points:clamp:warn", "offset_time")
    assert "clamped" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Output paths
# --------------------------------------------------------------------------
@pytest.mark.parametrize("suffix", ["../escape", "a/b", "..", ".", ""])
def test_suffix_escaping_the_output_root_is_rejected(suffix):
    with pytest.raises(SystemExit, match="suffix"):
        batch_to_bag.check_suffix(suffix)


def test_plain_suffix_is_accepted(tmp_path):
    assert batch_to_bag.check_suffix("_fastlio_bag") == "_fastlio_bag"
    out = batch_to_bag.resolve_output(str(tmp_path), "20230531", "_fastlio_bag")
    assert out == os.path.join(os.path.realpath(tmp_path), "20230531_fastlio_bag")


def test_resolve_output_rejects_a_name_that_climbs_out(tmp_path):
    with pytest.raises(SystemExit, match="escapes"):
        batch_to_bag.resolve_output(str(tmp_path), "..", "")


def test_check_window_rejects_an_inverted_window():
    with pytest.raises(SystemExit, match="start-sec"):
        check_window(30.0, 10.0)
    check_window(10.0, 30.0)      # valid
    check_window(None, None)


# --------------------------------------------------------------------------
# Atomic output
# --------------------------------------------------------------------------
def _write_bag(path: str, topic: str, force: bool = False) -> None:
    with atomic_bag(path, CFG, force) as writer:
        conn = writer.add_connection(topic, "std_msgs/msg/Header",
                                     typestore=to_bag.ROS2_STORE)
        writer.write(conn, 1_000_000_000, b"\x00" * 8)


def _topics(path: str) -> list[str]:
    with Reader(path) as reader:
        return sorted(c.topic for c in reader.connections)


def test_atomic_bag_publishes_on_success(tmp_path):
    out = str(tmp_path / "bag")
    _write_bag(out, "/first")
    assert _topics(out) == ["/first"]
    assert not os.path.exists(os.path.join(str(tmp_path), ".bag.partial"))


def test_published_bag_files_are_named_after_the_bag(tmp_path):
    """rosbags names the storage file after the bag's own directory, so staging
    under a temporary *name* published a hidden `.bag.partial.db3` inside it.
    Staging under a temporary *parent* keeps the final basename throughout."""
    out = str(tmp_path / "20230531_bag")
    _write_bag(out, "/first")
    assert sorted(os.listdir(out)) == ["20230531_bag.db3", "metadata.yaml"]


def test_atomic_bag_refuses_to_overwrite_without_force(tmp_path):
    out = str(tmp_path / "bag")
    _write_bag(out, "/first")
    with pytest.raises(SystemExit, match="already exists"):
        _write_bag(out, "/second")
    assert _topics(out) == ["/first"]


def test_failed_replacement_leaves_the_previous_bag_intact(tmp_path):
    """The bug this exists for: --force used to rmtree the old bag *before*
    converting, so a mid-conversion failure lost both."""
    out = str(tmp_path / "bag")
    _write_bag(out, "/first")

    with pytest.raises(RuntimeError):
        with atomic_bag(out, CFG, force=True) as writer:
            writer.add_connection("/second", "std_msgs/msg/Header", typestore=to_bag.ROS2_STORE)
            raise RuntimeError("disk full")

    assert _topics(out) == ["/first"]                      # old bag still readable
    assert sorted(os.listdir(str(tmp_path))) == ["bag"]    # no .partial / .trash left


def test_interrupt_leaves_no_staging_directory(tmp_path):
    """KeyboardInterrupt is a BaseException, so a plain `except Exception` would
    have let the staging directory survive."""
    out = str(tmp_path / "bag")
    with pytest.raises(KeyboardInterrupt):
        with atomic_bag(out, CFG) as writer:
            writer.add_connection("/x", "std_msgs/msg/Header", typestore=to_bag.ROS2_STORE)
            raise KeyboardInterrupt
    assert os.listdir(str(tmp_path)) == []


def test_force_replaces_the_bag_when_conversion_succeeds(tmp_path):
    out = str(tmp_path / "bag")
    _write_bag(out, "/first")
    _write_bag(out, "/second", force=True)
    assert _topics(out) == ["/second"]
    assert sorted(os.listdir(str(tmp_path))) == ["bag"]


def test_stale_staging_directory_is_cleared(tmp_path):
    """A killed run leaves .<name>.partial behind; the next one must not trip on it."""
    out = str(tmp_path / "bag")
    stale = tmp_path / ".bag.partial"
    stale.mkdir()
    (stale / "junk").write_text("leftover")
    _write_bag(out, "/first")
    assert _topics(out) == ["/first"]
    assert not stale.exists()


# --------------------------------------------------------------------------
# The ROS 1 converter shares atomic_bag, and no .bag files are kept on disk, so
# build a tiny one here to cover that path too.
# --------------------------------------------------------------------------
def _ros1_imu_bag(path: str, n: int = 5) -> None:
    Imu = ROS1_STORE.types["sensor_msgs/msg/Imu"]
    Header = ROS1_STORE.types["std_msgs/msg/Header"]
    Time = ROS1_STORE.types["builtin_interfaces/msg/Time"]
    Vector3 = ROS1_STORE.types["geometry_msgs/msg/Vector3"]
    Quaternion = ROS1_STORE.types["geometry_msgs/msg/Quaternion"]
    zeros9 = np.zeros(9, dtype=np.float64)

    writer = Rosbag1Writer(path)
    writer.open()
    conn = writer.add_connection("/alphasense/imu", "sensor_msgs/msg/Imu", typestore=ROS1_STORE)
    for i in range(n):
        ns = 1_685_528_702_000_000_000 + i * 2_500_000
        msg = Imu(
            header=Header(seq=i, stamp=Time(sec=ns // 10**9, nanosec=ns % 10**9),
                          frame_id="imu"),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            orientation_covariance=zeros9.copy(),
            angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
            angular_velocity_covariance=zeros9.copy(),
            linear_acceleration=Vector3(x=0.0, y=0.0, z=9.81),
            linear_acceleration_covariance=zeros9.copy(),
        )
        writer.write(conn, ns, ROS1_STORE.serialize_ros1(msg, "sensor_msgs/msg/Imu"))
    writer.close()


ROS1_CFG: dict = {
    "output": {"storage_id": "sqlite3", "version": 9},
    "topics": [{"in": "/alphasense/imu", "out": "/imu", "type": "imu", "frame_id": "imu_link"}],
}


# --------------------------------------------------------------------------
# repack: hesai_to_velodyne
#
# The vendors disagree about which end of the sweep the header stamp marks --
# Hilti's handheld Hesai says scan start, its robot's Robosense Bpearl says scan
# END -- and differencing against the wrong one negates every per-point time,
# which no downstream check would catch (the span is still ~0.1 s).
# --------------------------------------------------------------------------
HESAI_SCAN_NS = 100_000_000                 # a 10 Hz sweep
HESAI_T0_NS = 1_674_573_641_403_000_000     # absolute, like the recording


def _ros1_hesai_bag(path: str, header_at: str = "scan_start", n: int = 3,
                    pts: int = 6) -> list[int]:
    """A ROS 1 bag of Hesai-layout clouds: x,y,z,intensity (f32), timestamp (f64
    ABSOLUTE Unix seconds), ring (u16), point_step 48 -- the real field offsets,
    gaps included, so the strided read is exercised. Returns the header stamps."""
    PointCloud2 = ROS1_STORE.types["sensor_msgs/msg/PointCloud2"]
    PointField = ROS1_STORE.types["sensor_msgs/msg/PointField"]
    Header = ROS1_STORE.types["std_msgs/msg/Header"]
    Time = ROS1_STORE.types["builtin_interfaces/msg/Time"]
    fields = [PointField(name="x", offset=0, datatype=7, count=1),
              PointField(name="y", offset=4, datatype=7, count=1),
              PointField(name="z", offset=8, datatype=7, count=1),
              PointField(name="intensity", offset=16, datatype=7, count=1),
              PointField(name="timestamp", offset=24, datatype=8, count=1),
              PointField(name="ring", offset=32, datatype=4, count=1)]

    stamps = []
    writer = Rosbag1Writer(path)
    writer.open()
    conn = writer.add_connection("/hesai/pandar", "sensor_msgs/msg/PointCloud2",
                                 typestore=ROS1_STORE)
    for i in range(n):
        start_ns = HESAI_T0_NS + i * HESAI_SCAN_NS
        # Points spread evenly across the sweep; the last one lands at the very end.
        t_pts = start_ns + np.linspace(0, HESAI_SCAN_NS, pts, dtype=np.int64)
        hdr_ns = start_ns if header_at == "scan_start" else int(t_pts[-1])
        stamps.append(hdr_ns)
        buf = np.zeros(pts, dtype=np.dtype({
            "names": ["x", "y", "z", "intensity", "timestamp", "ring"],
            "formats": ["<f4", "<f4", "<f4", "<f4", "<f8", "<u2"],
            "offsets": [0, 4, 8, 16, 24, 32], "itemsize": 48}))
        buf["x"] = np.arange(pts) + 1.0
        buf["y"], buf["z"], buf["intensity"] = 2.0, 3.0, 42.0
        buf["timestamp"] = t_pts / 1e9
        buf["ring"] = np.arange(pts) % 32
        msg = PointCloud2(
            header=Header(seq=i, stamp=Time(sec=hdr_ns // 10**9, nanosec=hdr_ns % 10**9),
                          frame_id="PandarXT-32"),
            height=1, width=pts, fields=fields, is_bigendian=False,
            point_step=48, row_step=48 * pts, data=buf.view(np.uint8).reshape(-1),
            is_dense=True)
        writer.write(conn, hdr_ns, ROS1_STORE.serialize_ros1(msg, "sensor_msgs/msg/PointCloud2"))
    writer.close()
    return stamps


def _hesai_cfg(**cloud) -> dict:
    return {
        "output": {"storage_id": "sqlite3", "version": 9},
        "topics": [{"in": "/hesai/pandar", "out": "/velodyne_points", "type": "pointcloud2",
                    "frame_id": "velodyne", "repack": "hesai_to_velodyne",
                    "time_field": "timestamp", "time_unit": "s", "ring_field": "ring",
                    **cloud}],
    }


def _cloud_time(msg) -> np.ndarray:
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    off = {f.name: f.offset for f in msg.fields}["time"]
    return buf[:, off:off + 4].copy().view("<f4").ravel()


def test_hesai_repack_targets_the_velodyne_layout(tmp_path):
    in_bag = str(tmp_path / "in.bag")
    stamps = _ros1_hesai_bag(in_bag)
    out = str(tmp_path / "out_bag")
    rosbag1_to_ros2.convert(in_bag, out, _hesai_cfg())

    msgs = _read(out, "/velodyne_points")
    assert len(msgs) == 3
    _, m = msgs[0]
    assert [f.name for f in m.fields] == ["x", "y", "z", "intensity", "time", "ring"]
    assert [f.datatype for f in m.fields] == [7, 7, 7, 7, 7, 4]
    assert (m.point_step, m.row_step) == (22, 22 * m.width)
    assert m.header.frame_id == "velodyne"
    # header at scan start -> per-point time is [0, sweep], and nothing moved
    assert _cloud_time(m) == pytest.approx([0.0, 0.02, 0.04, 0.06, 0.08, 0.1], abs=1e-6)
    assert [ts for ts, _ in msgs] == stamps
    assert m.header.stamp.sec * 10**9 + m.header.stamp.nanosec == stamps[0]


def test_hesai_repack_rebases_a_header_stamped_at_scan_end(tmp_path):
    """Robosense Bpearl (Hilti robot): the stamp marks the LAST point, so
    subtracting it would make every `time` negative. The header moves back to the
    first point instead -- and the bag write stamp with it, so playback still
    delivers the frame when it was recorded."""
    in_bag = str(tmp_path / "in.bag")
    stamps = _ros1_hesai_bag(in_bag, header_at="scan_end")
    out = str(tmp_path / "out_bag")
    rosbag1_to_ros2.convert(in_bag, out, _hesai_cfg(header_at="scan_end"))

    msgs = _read(out, "/velodyne_points")
    _, m = msgs[0]
    assert _cloud_time(m) == pytest.approx([0.0, 0.02, 0.04, 0.06, 0.08, 0.1], abs=1e-6)
    assert (_cloud_time(m) >= 0).all()
    moved = stamps[0] - HESAI_SCAN_NS
    assert m.header.stamp.sec * 10**9 + m.header.stamp.nanosec == pytest.approx(moved, abs=1000)
    assert [ts for ts, _ in msgs] == pytest.approx([s - HESAI_SCAN_NS for s in stamps], abs=1000)


def test_hesai_repack_without_the_flag_would_negate_the_times(tmp_path):
    """The failure this guards against: reading a scan_end recording as scan_start
    is not an error anywhere downstream, it just silently flips the sign."""
    in_bag = str(tmp_path / "in.bag")
    _ros1_hesai_bag(in_bag, header_at="scan_end")
    out = str(tmp_path / "out_bag")
    rosbag1_to_ros2.convert(in_bag, out, _hesai_cfg())      # default scan_start
    _, m = _read(out, "/velodyne_points")[0]
    assert _cloud_time(m) == pytest.approx([-0.1, -0.08, -0.06, -0.04, -0.02, 0.0], abs=1e-6)


def test_hesai_repack_rejects_an_unknown_header_position(tmp_path):
    in_bag = str(tmp_path / "in.bag")
    _ros1_hesai_bag(in_bag)
    with pytest.raises(SystemExit, match="header_at must be"):
        rosbag1_to_ros2.convert(in_bag, str(tmp_path / "out_bag"),
                                _hesai_cfg(header_at="middle"))


def test_hesai_repack_refuses_relative_per_point_time(tmp_path):
    """Feeding an already-relative column here would subtract ~1.7e9 s per point."""
    in_bag = str(tmp_path / "in.bag")
    _ros1_hesai_bag(in_bag)
    out = str(tmp_path / "out_bag")
    with pytest.raises(SystemExit, match="not absolute Unix time"):
        rosbag1_to_ros2.convert(in_bag, out, _hesai_cfg(time_unit="ns"))


def test_rosbag1_converter_writes_a_readable_bag(tmp_path):
    in_bag = str(tmp_path / "in.bag")
    _ros1_imu_bag(in_bag)
    out = str(tmp_path / "out_bag")
    rosbag1_to_ros2.convert(in_bag, out, ROS1_CFG)
    assert _topics(out) == ["/imu"]
    assert sorted(os.listdir(out)) == ["metadata.yaml", "out_bag.db3"]


def test_rosbag1_converter_keeps_the_old_bag_when_it_fails(tmp_path):
    in_bag = str(tmp_path / "in.bag")
    _ros1_imu_bag(in_bag)
    out = str(tmp_path / "out_bag")
    rosbag1_to_ros2.convert(in_bag, out, ROS1_CFG)

    # A topic list that matches nothing is a SystemExit from inside the writer
    # block -- the previous bag must survive it, with no staging left behind.
    bad = dict(ROS1_CFG, topics=[{"in": "/nope", "out": "/imu", "type": "imu"}])
    with pytest.raises(SystemExit):
        rosbag1_to_ros2.convert(in_bag, out, bad, force=True)
    assert _topics(out) == ["/imu"]
    assert sorted(os.listdir(str(tmp_path))) == ["in.bag", "out_bag"]


# --------------------------------------------------------------------------
# Distro compatibility
#
# Metadata version 9 stores `offered_qos_profiles` as a YAML sequence, which
# Humble reads with .as<std::string>() and dies on ("yaml-cpp: bad conversion")
# -- the whole bag fails to open, on info and on play. Version 8 stores it as a
# YAML string with integer enum values, which both Humble and Jazzy read. These
# tests pin the encoding, since the failure is in another process on another
# distro and nothing here would otherwise notice a regression.
# --------------------------------------------------------------------------
def _metadata(path: str) -> dict:
    return load_yaml(os.path.join(path, "metadata.yaml"))["rosbag2_bagfile_information"]


def _qos_of(path: str, topic: str):
    entry = next(t for t in _metadata(path)["topics_with_message_count"]
                 if t["topic_metadata"]["name"] == topic)
    return entry["topic_metadata"]["offered_qos_profiles"]


def _tf_static_bag(path: str, cfg: dict) -> None:
    with atomic_bag(path, cfg) as writer:
        write_tf_static(writer, [{"parent": "base_link", "child": "lidar",
                                  "translation": [1.0, 0.0, 0.0]}], 1_700_000_000_000_000_000)


def test_default_output_is_readable_on_humble_and_jazzy(tmp_path):
    out = str(tmp_path / "bag")
    _tf_static_bag(out, {"output": {"storage_id": "sqlite3"}})
    assert _metadata(out)["version"] == 8
    # A string, not a list: this is the whole difference Humble trips over.
    assert isinstance(_qos_of(out, "/tf_static"), str)


def test_humble_readable_qos_keeps_tf_static_latched(tmp_path):
    """The v8 QoS string uses the rmw integer enums (transient_local == 1). Getting
    them wrong would silently downgrade /tf_static to volatile, and a localizer
    subscribing after playback starts would never receive the extrinsics."""
    out = str(tmp_path / "bag")
    _tf_static_bag(out, {"output": {"ros_distro": "humble"}})
    qos = yaml.safe_load(_qos_of(out, "/tf_static"))
    assert qos[0]["durability"] == 1 and qos[0]["reliability"] == 1
    assert qos[0]["history"] == 1 and qos[0]["depth"] == 1


def test_jazzy_target_writes_native_v9_metadata(tmp_path):
    out = str(tmp_path / "bag")
    _tf_static_bag(out, {"output": {"ros_distro": "jazzy"}})
    assert _metadata(out)["version"] == 9
    assert isinstance(_qos_of(out, "/tf_static"), list)      # sequence form


@pytest.mark.parametrize("distro,expected", [("both", 8), ("humble", 8), ("jazzy", 9)])
def test_distro_maps_to_metadata_version(distro, expected):
    assert bag_version({"output": {"ros_distro": distro}}) == expected


def test_explicit_version_is_honoured():
    assert bag_version({"output": {"version": 9}}) == 9
    assert bag_version({"output": {"version": 8, "ros_distro": "humble"}}) == 8


def test_version_that_cannot_be_written_is_refused():
    """Humble's own metadata version. rosbags can only emit 8 or 9, and silently
    writing 8 instead would be a bag the config did not ask for."""
    with pytest.raises(SystemExit, match="unsupported rosbag2 metadata version 5"):
        bag_version({"output": {"version": 5}})


def test_conflicting_distro_and_version_is_refused():
    with pytest.raises(SystemExit, match="drop one"):
        bag_version({"output": {"ros_distro": "humble", "version": 9}})


def test_unknown_distro_is_refused():
    with pytest.raises(SystemExit, match="unknown ros_distro"):
        bag_version({"output": {"ros_distro": "foxy"}})


def test_cli_distro_overrides_a_config_pinned_to_jazzy():
    """--ros-distro humble must win over `version: 9` in the YAML, not collide
    with it -- otherwise re-targeting a bag would mean editing the config."""
    cfg = {"output": {"storage_id": "mcap", "version": 9}, "sources": []}
    out = apply_distro(cfg, "humble")
    assert bag_version(out) == 8
    assert out["output"]["storage_id"] == "mcap"      # other keys survive
    assert cfg["output"]["version"] == 9              # caller's config untouched


def test_no_cli_distro_leaves_the_config_alone():
    cfg = {"output": {"version": 9}}
    assert apply_distro(cfg, None) is cfg


# --------------------------------------------------------------------------
# Calibration injection
# --------------------------------------------------------------------------
def _cfg_with_two_transforms() -> dict:
    eye = np.eye(4).tolist()
    return {
        "sources": [
            {"out": "/points", "type": "pointcloud2", "frame_id": "livox_frame"},
            {"out": "/imu", "type": "imu", "frame_id": "imu_link"},
        ],
        "tf_static": [
            {"parent": "imu_link", "child": "livox_frame", "matrix": eye},
            {"parent": "imu_link", "child": "camera_link", "matrix": eye},
        ],
    }


def test_calibration_only_touches_the_lidar_imu_transform():
    """It used to overwrite every entry with a `matrix`, camera included."""
    cfg = _cfg_with_two_transforms()
    cal = {"matrix": (np.eye(4) * 2).tolist(), "stamp_shift_sec": 0.25}
    notes = batch_to_bag.apply_calibration(cfg, cal)

    lidar_tf, camera_tf = cfg["tf_static"]
    assert lidar_tf["matrix"] == cal["matrix"]
    assert lidar_tf["matrix_is"] == "parent_to_child"
    assert camera_tf["matrix"] == np.eye(4).tolist()          # untouched
    assert cfg["sources"][0]["stamp_shift_sec"] == 0.25
    assert any("livox_frame" in n for n in notes)


def test_reversed_transform_is_flagged_for_inversion():
    cfg = _cfg_with_two_transforms()
    cfg["tf_static"][0] = {"parent": "livox_frame", "child": "imu_link", "matrix": np.eye(4).tolist()}
    batch_to_bag.apply_calibration(cfg, {"matrix": (np.eye(4) * 2).tolist(), "stamp_shift_sec": 0.0})
    assert cfg["tf_static"][0]["matrix_is"] == "child_to_parent"


def test_ambiguous_calibration_target_is_refused():
    cfg = _cfg_with_two_transforms()
    cfg["tf_static"][1] = {"parent": "livox_frame", "child": "imu_link", "matrix": np.eye(4).tolist()}
    with pytest.raises(SystemExit, match="unambiguously"):
        batch_to_bag.calibration_target(cfg)


def test_singular_calibration_matrix_is_refused(tmp_path):
    path = tmp_path / "calibration_lidar.yaml"
    path.write_text("lidar0:\n  T_lidar_imu0: [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]\n")
    with pytest.raises(SystemExit, match="singular"):
        batch_to_bag.read_calibration(str(path))


def test_non_finite_calibration_matrix_is_refused(tmp_path):
    path = tmp_path / "calibration_lidar.yaml"
    path.write_text("lidar0:\n  T_lidar_imu0: [[.nan,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n")
    with pytest.raises(SystemExit, match="NaN/Inf"):
        batch_to_bag.read_calibration(str(path))


# --------------------------------------------------------------------------
# Images + camera_info
#
# The images themselves are a pass-through (sensor_msgs/Image is field-identical
# in both typestores), so what needs pinning is the camera_info that goes with
# them: the recorded one cannot be forwarded at all (the ROS 1 typestore names the
# matrices D/K/R/P, the ROS 2 one d/k/r/p) and the one ConSLAM records has an
# all-zero P, which silently sends every projected point to NaN downstream.
# --------------------------------------------------------------------------
CALIB_RGB = """
image_width: 8
image_height: 4
camera_name: pp_rgb
camera_matrix:
  rows: 3
  cols: 3
  data: [2351.85909, 0.0, 1017.24032, 0.0, 2353.56734, 788.70774, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.054366, 0.118389, 0.001781, -0.003908, 0.0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1., 0., 0., 0., 1., 0., 0., 0., 1.]
projection_matrix:   # declared 3x3 but holds 12 numbers, and is for CROPPED images
  rows: 3
  cols: 3
  data: [2333.62549, 0.0, 1010.86063, 0.0, 0.0, 2330.45288, 785.218867, 0.0,
         0.0, 0.0, 1.0, 0.0]
"""


# Kalibr multi-camera calibration (Hilti's calibration_cams.yaml layout, cut to
# two cameras and to this fixture's 8x4 image). cam1 is deliberately a different
# resolution and distortion model so a mis-selected camera is visible.
CALIB_KALIBR = """
cam0:
  camera_model: pinhole
  distortion_coeffs: [-0.03696737, -0.00891788, 0.00891297, -0.00376860]
  distortion_model: equidistant
  intrinsics: [351.31400364193297, 351.4911744656785, 3.5, 2.5]
  resolution: [8, 4]
  rostopic: pp_rgb/synced2points
  T_here_imu0:
  - [1.0, 0.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
cam1:
  camera_model: pinhole
  distortion_coeffs: [-0.05, 0.11, 0.0017, -0.0039]
  distortion_model: radtan
  intrinsics: [352.6, 352.8, 8.0, 5.0]
  resolution: [16, 10]
  rostopic: /alphasense/cam1/image_raw
"""


def _ros1_image_bag(path: str, n: int = 3, width: int = 8, height: int = 4) -> list[int]:
    """A tiny ROS 1 bag of rgb8 images. Returns the header stamps in ns."""
    Image = ROS1_STORE.types["sensor_msgs/msg/Image"]
    Header = ROS1_STORE.types["std_msgs/msg/Header"]
    Time = ROS1_STORE.types["builtin_interfaces/msg/Time"]

    stamps = []
    writer = Rosbag1Writer(path)
    writer.open()
    conn = writer.add_connection("pp_rgb/synced2points", "sensor_msgs/msg/Image",
                                 typestore=ROS1_STORE)
    for i in range(n):
        ns = 1_650_963_683_550_153_000 + i * 100_000_000
        stamps.append(ns)
        msg = Image(
            header=Header(seq=i, stamp=Time(sec=ns // 10**9, nanosec=ns % 10**9),
                          frame_id="pp_rgb_raw"),
            height=height, width=width, encoding="rgb8", is_bigendian=0,
            step=width * 3, data=np.full(width * height * 3, i, dtype=np.uint8),
        )
        writer.write(conn, ns, ROS1_STORE.serialize_ros1(msg, "sensor_msgs/msg/Image"))
    writer.close()
    return stamps


def _image_cfg(calib: str, **camera_info) -> dict:
    return {
        "output": {"storage_id": "sqlite3", "version": 9},
        "topics": [{
            "in": "pp_rgb/synced2points", "out": "/camera/rgb/image_raw",
            "type": "image", "frame_id": "pp_rgb",
            "camera_info": {"from": calib, **camera_info},
        }],
    }


def _read(path: str, topic: str) -> list:
    with AnyReader([Path(path)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        return [(ts, reader.deserialize(raw, c.msgtype))
                for c, ts, raw in reader.messages(connections=conns)]


def test_images_pass_through_unchanged(tmp_path):
    """No repack, no rebase: the ROS 1 payload must survive re-serialization as-is."""
    in_bag = str(tmp_path / "in.bag")
    _ros1_image_bag(in_bag)
    out = str(tmp_path / "images_bag")
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB)
    rosbag1_to_ros2.convert(in_bag, out, _image_cfg(str(calib)))

    imgs = _read(out, "/camera/rgb/image_raw")
    assert len(imgs) == 3
    _, first = imgs[0]
    assert (first.width, first.height, first.encoding) == (8, 4, "rgb8")
    assert first.header.frame_id == "pp_rgb"          # frame_id override applied
    assert np.array_equal(np.asarray(first.data), np.zeros(8 * 4 * 3, dtype=np.uint8))


def test_camera_info_is_written_once_per_image_with_the_image_stamp(tmp_path):
    """image_transport's CameraSubscriber pairs the two by EXACT time, so a single
    latched message -- or one stamped off the bag clock -- would never pair."""
    in_bag = str(tmp_path / "in.bag")
    stamps = _ros1_image_bag(in_bag)
    out = str(tmp_path / "images_bag")
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB)
    rosbag1_to_ros2.convert(in_bag, out, _image_cfg(str(calib)))

    assert _topics(out) == ["/camera/rgb/camera_info", "/camera/rgb/image_raw"]
    infos = _read(out, "/camera/rgb/camera_info")
    assert len(infos) == len(stamps)
    for (bag_ns, ci), ns in zip(infos, stamps, strict=True):
        assert ci.header.stamp.sec * 10**9 + ci.header.stamp.nanosec == ns
        assert bag_ns == ns
        assert ci.header.frame_id == "pp_rgb"          # same frame as the image


def test_camera_info_p_is_built_from_k_not_the_files_projection_matrix(tmp_path):
    """The file's projection_matrix is for undistorted AND CROPPED images
    (fx 2333.6 vs K's 2351.9); using it on the raw frame biases every point."""
    in_bag = str(tmp_path / "in.bag")
    _ros1_image_bag(in_bag)
    out = str(tmp_path / "images_bag")
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB)
    rosbag1_to_ros2.convert(in_bag, out, _image_cfg(str(calib)))

    _, ci = _read(out, "/camera/rgb/camera_info")[0]
    k, p = np.asarray(ci.k), np.asarray(ci.p).reshape(3, 4)
    assert k[0] == 2351.85909 and k[2] == 1017.24032
    assert np.array_equal(p[:, :3].reshape(9), k)      # P == [K | 0]
    assert np.array_equal(p[:, 3], np.zeros(3))
    assert np.array_equal(np.asarray(ci.r), np.eye(3).reshape(9))
    assert ci.distortion_model == "plumb_bob" and len(np.asarray(ci.d)) == 5
    assert (ci.width, ci.height) == (8, 4)


def test_camera_info_for_the_wrong_image_size_is_fatal(tmp_path):
    """A calibration from another camera (or from before a crop/rescale) would
    otherwise project into the wrong pixels with nothing complaining."""
    in_bag = str(tmp_path / "in.bag")
    _ros1_image_bag(in_bag, width=16)
    out = str(tmp_path / "images_bag")
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB)                        # declares 8x4
    with pytest.raises(SystemExit, match="wrong calibration"):
        rosbag1_to_ros2.convert(in_bag, out, _image_cfg(str(calib)))


def test_camera_info_topic_can_be_overridden(tmp_path):
    in_bag = str(tmp_path / "in.bag")
    _ros1_image_bag(in_bag)
    out = str(tmp_path / "images_bag")
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB)
    rosbag1_to_ros2.convert(in_bag, out, _image_cfg(str(calib), out="/rgb/info"))
    assert "/rgb/info" in _topics(out)


def test_relative_calibration_path_resolves_against_the_config(tmp_path):
    in_bag = str(tmp_path / "in.bag")
    _ros1_image_bag(in_bag)
    (tmp_path / "calib_rgb.yaml").write_text(CALIB_RGB)
    rosbag1_to_ros2.convert(in_bag, str(tmp_path / "images_bag"),
                            _image_cfg("calib_rgb.yaml"), base_dir=str(tmp_path))
    assert build_camera_info({"from": "calib_rgb.yaml"}, str(tmp_path)).width == 8


def test_missing_calibration_file_fails_before_any_message_is_read(tmp_path):
    """The 63 GB source takes ~1 h to read; a bad path must not surface at the end."""
    in_bag = str(tmp_path / "in.bag")
    _ros1_image_bag(in_bag)
    with pytest.raises(SystemExit, match="not found"):
        rosbag1_to_ros2.convert(in_bag, str(tmp_path / "images_bag"),
                                _image_cfg(str(tmp_path / "nope.yaml")))


def test_unknown_camera_info_format_is_refused(tmp_path):
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB)
    with pytest.raises(SystemExit, match="unknown format"):
        build_camera_info({"from": str(calib), "format": "opencv_xml"})


# --------------------------------------------------------------------------
# Kalibr calibrations (Hilti). One file holds every camera on the rig, so the
# selection is part of the contract, and the fisheye model has to survive the
# trip into CameraInfo intact.
# --------------------------------------------------------------------------

def _kalibr(tmp_path, **spec):
    calib = tmp_path / "calibration_cams.yaml"
    calib.write_text(CALIB_KALIBR)
    return build_camera_info({"from": str(calib), "format": "kalibr", **spec})


@pytest.mark.parametrize("camera", ["cam0", "pp_rgb/synced2points"])
def test_kalibr_camera_selected_by_key_or_by_rostopic(tmp_path, camera):
    """Both spellings must resolve to the same camera: the key is what the file
    is organized by, the rostopic is what cannot drift out of sync with the bag."""
    info = _kalibr(tmp_path, camera=camera)
    assert (info.width, info.height) == (8, 4)
    assert info.distortion_model == "equidistant"
    assert list(info.d) == pytest.approx([-0.03696737, -0.00891788, 0.00891297, -0.00376860])
    # K = [fx, 0, cx; 0, fy, cy; 0, 0, 1] out of `intrinsics: [fx, fy, cx, cy]`,
    # and P = [K | 0] -- the reordering is exactly what a hand-written K gets wrong.
    assert list(info.k) == pytest.approx([351.31400364193297, 0.0, 3.5,
                                          0.0, 351.4911744656785, 2.5, 0.0, 0.0, 1.0])
    assert list(info.p) == pytest.approx([351.31400364193297, 0.0, 3.5, 0.0,
                                          0.0, 351.4911744656785, 2.5, 0.0,
                                          0.0, 0.0, 1.0, 0.0])


def test_kalibr_radtan_becomes_plumb_bob_with_five_coefficients(tmp_path):
    """ROS's plumb_bob D is [k1,k2,p1,p2,k3]; kalibr radtan omits k3."""
    info = _kalibr(tmp_path, camera="cam1")
    assert info.distortion_model == "plumb_bob"
    assert list(info.d) == pytest.approx([-0.05, 0.11, 0.0017, -0.0039, 0.0])


@pytest.mark.parametrize("spec,match", [
    ({}, "needs `camera:`"),                       # five look-alike cameras, no default
    ({"camera": "cam9"}, "no camera 'cam9'"),
    ({"camera": "/alphasense/cam2/image_raw"}, "no camera"),
])
def test_kalibr_camera_selection_must_be_unambiguous(tmp_path, spec, match):
    with pytest.raises(SystemExit, match=match):
        _kalibr(tmp_path, **spec)


@pytest.mark.parametrize("edit,replacement,match", [
    # A model CameraInfo's K cannot express, and one this reader does not know.
    ("camera_model: pinhole", "camera_model: omni", "camera_model 'omni'"),
    ("distortion_model: equidistant", "distortion_model: fov", "distortion_model 'fov'"),
    # Truncated matrices: `intrinsics` short one number reads cx as cy, and a
    # 3-coefficient equidistant D is not a KB4 model at all.
    ("intrinsics: [351.31400364193297, 351.4911744656785, 3.5, 2.5]",
     "intrinsics: [351.31400364193297, 351.4911744656785, 3.5]", "intrinsics"),
    ("distortion_coeffs: [-0.03696737, -0.00891788, 0.00891297, -0.00376860]",
     "distortion_coeffs: [-0.03696737, -0.00891788, 0.00891297]", "distortion_coeffs"),
    ("resolution: [8, 4]", "", "resolution"),
    ("intrinsics: [351.31400364193297, 351.4911744656785, 3.5, 2.5]",
     "intrinsics: [0.0, 351.4911744656785, 3.5, 2.5]", "focal length"),
])
def test_kalibr_unusable_calibration_is_refused(tmp_path, edit, replacement, match):
    """The intrinsics are the one thing no downstream check catches, so anything
    unexpressible or truncated has to fail here rather than be guessed at."""
    calib = tmp_path / "calibration_cams.yaml"
    calib.write_text(CALIB_KALIBR.replace(edit, replacement, 1))
    with pytest.raises(SystemExit, match=match):
        build_camera_info({"from": str(calib), "format": "kalibr", "camera": "cam0"})


def test_kalibr_principal_point_outside_the_image_is_refused(tmp_path):
    """The guard against pairing cam3's intrinsics with cam0's images."""
    calib = tmp_path / "calibration_cams.yaml"
    calib.write_text(CALIB_KALIBR.replace("resolution: [8, 4]", "resolution: [2, 1]", 1))
    with pytest.raises(SystemExit, match="principal point"):
        build_camera_info({"from": str(calib), "format": "kalibr", "camera": "cam0"})


def test_ros_camera_info_refuses_a_kalibr_camera_selector(tmp_path):
    """`camera:` with the default format means the wrong `format:` was set."""
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB)
    with pytest.raises(SystemExit, match="format: kalibr"):
        build_camera_info({"from": str(calib), "camera": "cam0"})


def test_kalibr_calibration_lands_in_the_bag(tmp_path):
    """End to end: the reader is reachable through a topic's `camera_info:` block."""
    in_bag = str(tmp_path / "in.bag")
    stamps = _ros1_image_bag(in_bag)
    calib = tmp_path / "calibration_cams.yaml"
    calib.write_text(CALIB_KALIBR)
    out = str(tmp_path / "images_bag")
    rosbag1_to_ros2.convert(in_bag, out, _image_cfg(str(calib), format="kalibr", camera="cam0"))

    infos = _read(out, "/camera/rgb/camera_info")
    assert [ts for ts, _ in infos] == stamps
    assert {m.distortion_model for _, m in infos} == {"equidistant"}
    assert {m.header.frame_id for _, m in infos} == {"pp_rgb"}


def test_camera_info_on_a_non_image_topic_is_refused(tmp_path):
    """camera_info alongside an Imu (or a cloud, whose `width` is a point count)
    would pass the size check by accident."""
    in_bag = str(tmp_path / "in.bag")
    _ros1_imu_bag(in_bag)
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB)
    cfg = dict(ROS1_CFG, topics=[dict(ROS1_CFG["topics"][0],
                                      camera_info={"from": str(calib)})])
    with pytest.raises(SystemExit, match="not sensor_msgs/msg/Image"):
        rosbag1_to_ros2.convert(in_bag, str(tmp_path / "out_bag"), cfg)


@pytest.mark.parametrize("edit,match", [
    ("image_width: 8", "image_width"),                        # key removed below
    ("data: [2351.85909, 0.0, 1017.24032, 0.0, 2353.56734, 788.70774, 0.0, 0.0, 1.0]",
     "camera_matrix"),
])
def test_unusable_calibration_is_refused(tmp_path, edit, match):
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB.replace(edit, "", 1))
    with pytest.raises(SystemExit, match=match):
        build_camera_info({"from": str(calib)})


def test_zero_focal_length_is_refused(tmp_path):
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB.replace("2351.85909", "0.0", 1))
    with pytest.raises(SystemExit, match="focal length"):
        build_camera_info({"from": str(calib)})


# --------------------------------------------------------------------------
# Topic groups
#
# One config per consumer declares everything the dataset offers; `--groups`
# decides what lands in the bag. The failure modes worth pinning are a typo'd
# group silently producing an odometry-only bag, and an untagged entry (the
# imu->lidar extrinsic, an old config) being dropped by a filter it predates.
# --------------------------------------------------------------------------
GROUPED_CFG: dict = {
    "include_groups": ["lidar", "imu"],
    "topics": [
        {"in": "/points", "groups": ["lidar"]},
        {"in": "/imu", "groups": ["imu"]},
        {"in": "/cam0", "groups": ["camera", "cam0"]},
        {"in": "/cam1", "groups": ["camera", "cam1"]},
        {"in": "/marks"},                                   # untagged
    ],
    "tf_static": [
        {"parent": "imu_link", "child": "velodyne"},         # untagged
        {"parent": "velodyne", "child": "cam0", "groups": ["camera"]},
    ],
}


def _kept(cfg: dict, requested):
    groups = select_groups(cfg, requested)
    out = filter_groups(cfg, groups)
    return [t["in"] for t in out["topics"]], len(out.get("tf_static") or [])


def test_config_default_groups_are_used_when_no_flag_is_given():
    assert _kept(GROUPED_CFG, None) == (["/points", "/imu", "/marks"], 1)


def test_all_takes_every_declared_group():
    topics, n_tf = _kept(GROUPED_CFG, "all")
    assert topics == ["/points", "/imu", "/cam0", "/cam1", "/marks"] and n_tf == 2


def test_a_single_camera_can_be_selected():
    topics, n_tf = _kept(GROUPED_CFG, "lidar,imu,cam0")
    assert topics == ["/points", "/imu", "/cam0", "/marks"]
    assert n_tf == 1          # the camera extrinsic belongs to `camera`, not `cam0`


def test_untagged_entries_survive_every_selection():
    """An untagged entry is unconditional -- otherwise adding tags to one topic
    would silently drop the imu->lidar extrinsic next to it."""
    for requested in (None, "all", "camera", "lidar"):
        topics, n_tf = _kept(GROUPED_CFG, requested)
        assert "/marks" in topics and n_tf >= 1


def test_none_keeps_only_the_untagged_entries():
    topics, n_tf = _kept(GROUPED_CFG, "none")
    assert topics == ["/marks"] and n_tf == 1


def test_untagged_config_is_unaffected_by_the_filter():
    cfg = {"topics": [{"in": "/points"}, {"in": "/imu"}]}
    assert _kept(cfg, None)[0] == ["/points", "/imu"]


def test_typo_in_group_name_is_fatal():
    """It used to be indistinguishable from `that dataset has no images`."""
    with pytest.raises(SystemExit, match="unknown topic group"):
        select_groups(GROUPED_CFG, "lidar,camrea")


def test_selection_that_keeps_no_topic_is_fatal():
    cfg = {"topics": [{"in": "/cam0", "groups": ["camera"]}]}
    with pytest.raises(SystemExit, match="select no topics"):
        filter_groups(cfg, ["lidar"])


def test_a_selected_topic_missing_from_the_input_is_fatal(tmp_path):
    """The silent case this guards: bag_conslam_kdamr.yaml selects the camera group
    by default, so re-timing a _fastlio_bag that was written WITHOUT images used to
    produce a valid kdamr bag with no images in it and say nothing."""
    in_bag = str(tmp_path / "in.bag")
    _ros1_imu_bag(in_bag)
    cfg = dict(ROS1_CFG, topics=[
        *ROS1_CFG["topics"],
        {"in": "/camera/rgb/image_raw", "type": "image", "groups": ["camera"]},
    ])
    with pytest.raises(SystemExit, match="missing from"):
        rosbag1_to_ros2.convert(in_bag, str(tmp_path / "out_bag"), cfg)
    # ... and deselecting the group is the way out. Here the IMU entry is untagged,
    # so `none` (untagged only) is what drops the camera.
    kept = filter_groups(cfg, select_groups(cfg, "none"))
    rosbag1_to_ros2.convert(in_bag, str(tmp_path / "imu_bag"), kept)
    assert _topics(str(tmp_path / "imu_bag")) == ["/imu"]


def test_missing_topic_error_names_the_group(tmp_path):
    in_bag = str(tmp_path / "in.bag")
    _ros1_imu_bag(in_bag)
    cfg = dict(ROS1_CFG, topics=[
        *ROS1_CFG["topics"],
        {"in": "/cam0", "type": "image", "groups": ["camera", "cam0"]},
    ])
    with pytest.raises(SystemExit, match=r"/cam0 \[camera, cam0\]"):
        rosbag1_to_ros2.convert(in_bag, str(tmp_path / "out_bag"), cfg)


def test_groups_reach_the_bag(tmp_path):
    """End to end: the same config, two different bags."""
    in_bag = str(tmp_path / "in.bag")
    _ros1_image_bag(in_bag)
    calib = tmp_path / "calib_rgb.yaml"
    calib.write_text(CALIB_RGB)
    cfg = dict(_image_cfg(str(calib)), include_groups=["camera"])
    cfg["topics"][0]["groups"] = ["camera"]
    cfg["topics"].append({"in": "pp_rgb/synced2points", "out": "/other",
                          "type": "image", "groups": ["spare"]})

    out = str(tmp_path / "camera_bag")
    rosbag1_to_ros2.convert(in_bag, out, filter_groups(cfg, select_groups(cfg, None)))
    assert _topics(out) == ["/camera/rgb/camera_info", "/camera/rgb/image_raw"]

    out2 = str(tmp_path / "spare_bag")
    rosbag1_to_ros2.convert(in_bag, out2, filter_groups(cfg, select_groups(cfg, "spare")))
    assert _topics(out2) == ["/other"]


def test_a_deselected_cameras_calibration_is_never_opened(tmp_path):
    """No camera selected -> no calibration needed, not even present.

    `camera_info:` is read per SELECTED topic, so an odometry-only run (and a
    dataset with no cameras at all, which simply has no such block) must not
    depend on a calibration file existing. The kalibr reader's demands --
    `camera:`, a pinhole model, four intrinsics -- likewise only ever apply to a
    camera that is actually being written.
    """
    in_bag = str(tmp_path / "in.bag")
    _ros1_image_bag(in_bag)
    missing = tmp_path / "not_downloaded_yet.yaml"          # deliberately absent
    cfg = dict(_image_cfg(str(missing), format="kalibr", camera="cam0"),
               include_groups=["spare"])
    cfg["topics"][0]["groups"] = ["camera"]
    cfg["topics"].append({"in": "pp_rgb/synced2points", "out": "/other",
                          "type": "image", "groups": ["spare"]})

    out = str(tmp_path / "no_camera_bag")
    rosbag1_to_ros2.convert(in_bag, out, filter_groups(cfg, select_groups(cfg, None)))
    assert _topics(out) == ["/other"]

    # ...while selecting that camera fails on the missing file, before any message
    # is read -- the point of validating up front.
    with pytest.raises(SystemExit, match="not found"):
        rosbag1_to_ros2.convert(in_bag, str(tmp_path / "camera_bag"),
                                filter_groups(cfg, select_groups(cfg, "camera")))


# --------------------------------------------------------------------------
# Per-run state
# --------------------------------------------------------------------------
def test_warnings_are_not_suppressed_across_runs(capsys):
    """warn_once is module state; a second in-process conversion used to inherit
    the first one's 'already reported' set and go quiet."""
    to_bag.warn_once("/points:demo", "first")
    capsys.readouterr()
    to_bag.warn_once("/points:demo", "first")
    assert capsys.readouterr().out == ""          # same run: still once

    to_bag.reset_warnings()
    to_bag.warn_once("/points:demo", "first")
    assert "first" in capsys.readouterr().out     # new run: reported again
