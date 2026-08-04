"""Shared ROS 2 bag *writing* layer for the converters.
"""
import contextlib
import math
import os
import shutil
from collections.abc import Generator
from typing import Any

import numpy as np
import yaml
from rosbags.rosbag2 import Writer
from rosbags.rosbag2.enums import StoragePlugin
from rosbags.typesys import Stores, get_typestore
from rosbags.interfaces import (
    Qos, QosDurability, QosHistory, QosLiveliness, QosReliability, QosTime,
)

ROS1_STORE = get_typestore(Stores.ROS1_NOETIC)   # deserialize ROS 1 input
ROS2_STORE = get_typestore(Stores.ROS2_JAZZY)    # serialize ROS 2 output

# Two storage plugins are supported: SQLite3 and MCAP.
STORAGE_PLUGINS = {
    "sqlite3": StoragePlugin.SQLITE3,
    "mcap": StoragePlugin.MCAP,
}

# These converters only WRITE standard ROS 2 messages (PointCloud2, Imu,
# Odometry, TFMessage), so ROS2_STORE needs no vendor types registered. Bags
# previously written as livox_ros_driver2/CustomMsg stay readable: rosbag2
# embeds each type's .msg text in the bag, so `rosbags.highlevel.AnyReader`
# (what inspect_bag.py uses) registers it from the bag itself.


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_topic(name: str) -> str:
    return name if name.startswith("/") else "/" + name


def check_window(start_sec: float | None, limit_sec: float | None) -> None:
    """Validate a --start-sec/--limit-sec pair.

    Shared so the batch driver can reject a bad window *before* it starts
    replacing outputs, rather than letting every subprocess discover it.
    """
    if start_sec is not None and limit_sec is not None and start_sec >= limit_sec:
        raise SystemExit(f"--start-sec ({start_sec}) must be < --limit-sec ({limit_sec})")


def open_writer(out_bag: str, cfg: dict[str, Any]) -> Writer:
    """Open an output bag using the shared ``output:`` config block."""
    out_cfg = cfg.get("output", {})
    storage = out_cfg.get("storage_id", "sqlite3")
    if storage not in STORAGE_PLUGINS:
        raise SystemExit(f"unknown storage_id {storage!r}; known: {sorted(STORAGE_PLUGINS)}")
    writer = Writer(out_bag, version=int(out_cfg.get("version", 9)),
                    storage_plugin=STORAGE_PLUGINS[storage])
    writer.open()
    return writer


def _sidecar(out_bag: str, kind: str) -> str:
    """A hidden sibling path of ``out_bag`` -- same directory, so same filesystem,
    which is what makes the final os.rename atomic."""
    parent, name = os.path.split(os.path.abspath(out_bag))
    return os.path.join(parent, f".{name}.{kind}")


@contextlib.contextmanager
def atomic_bag(out_bag: str, cfg: dict[str, Any],
               force: bool = False) -> Generator[Writer, None, None]:
    """Yield a Writer that only becomes ``out_bag`` once the body succeeds.

    Two failure modes this exists to prevent, both of which used to lose data:

      * the batch driver deleted the previous bag *before* converting, so a
        config error or a full disk left neither an old bag nor a complete new one;
      * a read/serialize error mid-conversion skipped ``writer.close()`` and left a
        half-written directory behind, which the next run then refused to touch
        ("output already exists").

    So: build the bag inside a staging *parent* directory ``.<name>.partial/`` and
    swap it into place with os.rename (atomic within a filesystem) once it is
    complete. A failure removes the staging tree and leaves any existing bag exactly
    as it was.

    The staging directory is a parent rather than a renamed bag because rosbags names
    the storage file after the bag's own directory: staging as ``.<name>.partial``
    directly would publish a bag containing a hidden ``.<name>.partial.db3``. Nesting
    keeps the bag's basename -- and therefore its db3/mcap file names -- final from
    the moment it is opened.
    """
    if os.path.exists(out_bag) and not force:
        raise SystemExit(f"output already exists: {out_bag}")
    out_bag = os.path.abspath(out_bag)
    parent = os.path.dirname(out_bag)
    if parent:
        os.makedirs(parent, exist_ok=True)

    staging_dir = _sidecar(out_bag, "partial")
    if os.path.exists(staging_dir):     # left over from a killed run
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)
    staging_bag = os.path.join(staging_dir, os.path.basename(out_bag))

    writer = open_writer(staging_bag, cfg)
    try:
        yield writer
        writer.close()
    except BaseException:           # BaseException: Ctrl-C must clean up too
        with contextlib.suppress(Exception):
            writer.close()
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    trash = _sidecar(out_bag, "trash")
    shutil.rmtree(trash, ignore_errors=True)
    replaced = os.path.exists(out_bag)
    if replaced:
        os.rename(out_bag, trash)
    try:
        os.rename(staging_bag, out_bag)
    except OSError:
        if replaced:                # put the old bag back rather than losing both
            os.rename(trash, out_bag)
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    if replaced:
        shutil.rmtree(trash, ignore_errors=True)


def ns_to_time(ns: int):
    Time = ROS2_STORE.types["builtin_interfaces/msg/Time"]
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def advance_stamp(stamp, delta_ns: int):
    total = stamp.sec * 1_000_000_000 + stamp.nanosec + delta_ns
    stamp.sec = int(total // 1_000_000_000)
    stamp.nanosec = int(total % 1_000_000_000)
    return stamp


# rmw "infinite" duration sentinel used by ROS 2 QoS profiles.
_QOS_INFINITE = QosTime(sec=9223372036, nsec=854775807)

# /tf_static is published transient_local (latched) so late subscribers -- like
# kd_localizer coming up after bag playback starts -- still receive the one-shot
# extrinsics. Record the connection with that QoS so `ros2 bag play` re-offers it.
TF_STATIC_QOS = Qos(
    history=QosHistory.KEEP_LAST, depth=1,
    reliability=QosReliability.RELIABLE, durability=QosDurability.TRANSIENT_LOCAL,
    deadline=_QOS_INFINITE, lifespan=_QOS_INFINITE,
    liveliness=QosLiveliness.AUTOMATIC, liveliness_lease_duration=_QOS_INFINITE,
    avoid_ros_namespace_conventions=False,
)


def rot_to_quat(r) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> (x, y, z, w). Shepperd's method."""
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return ((r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s,
                (r[1][0] - r[0][1]) / s, 0.25 * s)
    if r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        return (0.25 * s, (r[0][1] + r[1][0]) / s,
                (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s)
    if r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        return ((r[0][1] + r[1][0]) / s, 0.25 * s,
                (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s)
    s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
    return ((r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s,
            0.25 * s, (r[1][0] - r[0][1]) / s)


def transform_components(t: dict[str, Any]) -> tuple[list[float], tuple[float, float, float, float]]:
    """Read one tf_static entry as (translation xyz, rotation xyzw).

    Two spellings are accepted: explicit ``translation`` + ``rotation_xyzw``, or a
    4x4 ``matrix`` (what calibration files ship), optionally flagged
    ``matrix_is: child_to_parent`` to be inverted first.
    """
    child = t.get("child")
    if "matrix" in t:
        m = np.array(t["matrix"], dtype=float)
        if m.shape != (4, 4):
            raise SystemExit(f"tf_static matrix for {child} must be 4x4, got {m.shape}")
        if not np.isfinite(m).all():
            raise SystemExit(f"tf_static matrix for {child} contains NaN/Inf")
        if t.get("matrix_is", "parent_to_child") == "child_to_parent":
            try:
                m = np.linalg.inv(m)
            except np.linalg.LinAlgError as exc:
                raise SystemExit(f"tf_static matrix for {child} is singular, so "
                                 f"`matrix_is: child_to_parent` cannot invert it: {exc}") from None
        translation = [float(m[0, 3]), float(m[1, 3]), float(m[2, 3])]
        rotation = rot_to_quat(m[:3, :3].tolist())
    else:
        tx, ty, tz = t.get("translation", [0.0, 0.0, 0.0])
        translation = [float(tx), float(ty), float(tz)]
        rotation = tuple(float(v) for v in t.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
        if not all(math.isfinite(v) for v in (*translation, *rotation)):
            raise SystemExit(f"tf_static translation/rotation for {child} contains NaN/Inf")
    # Calibration matrices are published rounded, so they are only orthonormal to
    # ~1e-6 and the quaternion derived from one comes out slightly off unit
    # length. tf2 rejects a quaternion that drifts past its tolerance, so
    # normalize here rather than shipping one that is nearly-but-not-valid.
    q = np.array(rotation, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm == 0.0 or not math.isfinite(norm):
        raise SystemExit(f"tf_static rotation for {child} is not a usable quaternion "
                         f"(norm={norm})")
    return translation, tuple(float(v) for v in q / norm)


def write_tf_static(writer: Writer, transforms: list[dict[str, Any]], stamp_ns: int) -> int:
    """Write a single latched /tf_static message from the config `tf_static` list.

    Each entry needs `parent`, `child`, and either `translation` [x,y,z] +
    `rotation_xyzw` [x,y,z,w] or a 4x4 `matrix`. kd_localizer reads
    base_frame_id->lidar (and ->imu when use_imu) from here; it is NOT needed by
    FAST-LIO (which takes extrinsics from its own YAML). All transforms are
    stamped at the bag start time so they precede the first sensor message.
    """
    Header = ROS2_STORE.types["std_msgs/msg/Header"]
    Vector3 = ROS2_STORE.types["geometry_msgs/msg/Vector3"]
    Quaternion = ROS2_STORE.types["geometry_msgs/msg/Quaternion"]
    Transform = ROS2_STORE.types["geometry_msgs/msg/Transform"]
    TransformStamped = ROS2_STORE.types["geometry_msgs/msg/TransformStamped"]
    TFMessage = ROS2_STORE.types["tf2_msgs/msg/TFMessage"]

    stamp = ns_to_time(stamp_ns)
    tfs = []
    for t in transforms:
        (tx, ty, tz), (qx, qy, qz, qw) = transform_components(t)
        tfs.append(TransformStamped(
            header=Header(stamp=stamp, frame_id=t["parent"]),
            child_frame_id=t["child"],
            transform=Transform(
                translation=Vector3(x=tx, y=ty, z=tz),
                rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
            ),
        ))
    conn = writer.add_connection("/tf_static", "tf2_msgs/msg/TFMessage",
                                 typestore=ROS2_STORE, offered_qos_profiles=[TF_STATIC_QOS])
    writer.write(conn, stamp_ns,
                 ROS2_STORE.serialize_cdr(TFMessage(transforms=tfs), "tf2_msgs/msg/TFMessage"))
    return len(tfs)
