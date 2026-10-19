"""Shared ROS 2 bag *writing* layer for the converters.
"""
import argparse
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

# --------------------------------------------------------------------------
# Which ROS 2 distros can read the bag we write
#
# `rosbags` can emit rosbag2 metadata version 8 or 9, and that choice alone
# decides whether Humble can open the bag at all:
#
#   * v9 (Jazzy's native version) writes `offered_qos_profiles` as a YAML
#     *sequence* with named enum values. Humble's metadata parser reads that key
#     with `.as<std::string>()`, so the whole bag fails to open --
#     "Exception on parsing info file: yaml-cpp: ... bad conversion" -- for
#     sqlite3 and mcap alike, on `ros2 bag info` and on `ros2 bag play`.
#   * v8 writes the same QoS as a YAML *string* with integer enum values. That is
#     what Humble expects, and Jazzy reads it too: its decoder branches on
#     `version < 9` and keeps the string path for older bags.
#
# So v8 is the one format BOTH distros read, and it is the default here. Nothing
# in these bags needs v9: the storage files are identical either way, only
# metadata.yaml (and the copy of it in the .db3 `metadata` table) differs. The
# message definitions are byte-identical between the Humble and Jazzy typestores
# for every type these converters write (PointCloud2, Imu, Odometry, TFMessage),
# so ROS2_STORE serves both -- the distro is a metadata-encoding question only.
#
# Verified with `ros2 bag info` + `ros2 bag play` on Humble and Jazzy, both
# storage plugins. Distros older than Humble are out of reach: they want metadata
# version <= 5 and rosbags cannot write it.
BAG_VERSIONS = (8, 9)
DISTRO_VERSIONS = {
    "both": 8,      # readable on Humble and on Jazzy -- the default
    "humble": 8,
    "jazzy": 9,     # Jazzy-native metadata; NOT readable on Humble
}
DEFAULT_DISTRO = "both"

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


def bag_version(cfg: dict[str, Any]) -> int:
    """Resolve the rosbag2 metadata version from the ``output:`` block.

    Two spellings, because the useful question ("which distro must read this?")
    and the underlying knob are not the same thing:

      * ``ros_distro: both|humble|jazzy`` -- the intent, mapped by DISTRO_VERSIONS;
      * ``version: 8|9`` -- the raw metadata version, for a config that wants to
        name the format directly.

    Neither key is required (the default is Humble+Jazzy compatible). Giving both
    is only an error when they disagree, so a config can spell out both without
    the two silently ranking against each other.
    """
    out_cfg = cfg.get("output") or {}
    distro = out_cfg.get("ros_distro")
    if distro is not None:
        distro = str(distro).lower()
        if distro not in DISTRO_VERSIONS:
            raise SystemExit(f"unknown ros_distro {distro!r}; known: {sorted(DISTRO_VERSIONS)} "
                             f"(pre-Humble distros need metadata version <= 5, which cannot "
                             f"be written)")
    if "version" not in out_cfg:
        return DISTRO_VERSIONS[distro or DEFAULT_DISTRO]

    try:
        version = int(out_cfg["version"])
    except (TypeError, ValueError):
        raise SystemExit(f"output version must be an integer, got "
                         f"{out_cfg['version']!r}") from None
    if version not in BAG_VERSIONS:
        raise SystemExit(
            f"unsupported rosbag2 metadata version {version}; known: {list(BAG_VERSIONS)} "
            f"(8 is readable on Humble and Jazzy, 9 only from Jazzy on -- "
            f"prefer `ros_distro:` over `version:`)"
        )
    if distro is not None and DISTRO_VERSIONS[distro] != version:
        raise SystemExit(
            f"output block asks for both ros_distro: {distro} (metadata version "
            f"{DISTRO_VERSIONS[distro]}) and version: {version}; drop one"
        )
    return version


def apply_distro(cfg: dict[str, Any], distro: str | None) -> dict[str, Any]:
    """Return ``cfg`` with its target distro overridden from the command line.

    Any explicit ``version:`` is dropped rather than left to conflict, so
    ``--ros-distro humble`` re-targets a config pinned to ``version: 9`` without
    editing the YAML. Copies only the ``output:`` block; the rest is shared.
    """
    if distro is None:
        return cfg
    if distro not in DISTRO_VERSIONS:
        raise SystemExit(f"unknown --ros-distro {distro!r}; known: {sorted(DISTRO_VERSIONS)}")
    out_cfg = dict(cfg.get("output") or {})
    out_cfg.pop("version", None)
    out_cfg["ros_distro"] = distro
    return {**cfg, "output": out_cfg}


def add_distro_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``--ros-distro`` on a converter's parser (shared help text)."""
    parser.add_argument("--ros-distro", choices=sorted(DISTRO_VERSIONS), default=None,
                        help="ROS 2 distro that must be able to read the bag, overriding the "
                             f"config's `output: ros_distro:` (default: {DEFAULT_DISTRO} -- "
                             "metadata version 8, which Humble and Jazzy both read; "
                             "jazzy writes version 9, which Humble cannot open)")


# --------------------------------------------------------------------------
# Topic groups: one config per consumer, content chosen at conversion time
#
# A config declares everything a dataset offers -- clouds, IMU, images,
# camera_info, control-point marks -- and tags each entry with `groups:`. What
# actually lands in the bag is then a run-time choice (`--groups`), which is the
# difference between "one bag with everything" and "just the odometry topics"
# without maintaining two nearly identical configs that drift apart.
#
# It matters because the payloads differ by three orders of magnitude: ConSLAM's
# RGB stream is 9.56 MB per frame at 10 Hz (39.9 GB and ~96 MB/s of playback
# bandwidth per sequence) against ~0.9 MB/s for cloud + IMU. Which of those you
# want depends on the run, not on the dataset.
#
# An UNTAGGED entry is unconditional: it is converted whatever `--groups` says.
# That is what keeps a config without any tags behaving exactly as it did before
# groups existed, and it is also the right default for the entries a bag is
# meaningless without -- an imu_link->velodyne extrinsic is not "part of the lidar
# group", it is part of every bag that has a cloud in it.
def entry_groups(entry: dict[str, Any]) -> list[str]:
    """The groups one `topics:` / `tf_static:` entry declares ([] = unconditional)."""
    raw = entry.get("groups", entry.get("group"))
    if raw is None:
        return []
    return [str(g) for g in (raw if isinstance(raw, list) else [raw])]


def declared_groups(cfg: dict[str, Any]) -> list[str]:
    """Every group the config mentions, in declaration order (no duplicates)."""
    seen: dict[str, None] = {}
    for entry in [*(cfg.get("topics") or []), *(cfg.get("sources") or []),
                  *(cfg.get("tf_static") or [])]:
        for g in entry_groups(entry):
            seen.setdefault(g, None)
    return list(seen)


def select_groups(cfg: dict[str, Any], requested: str | None) -> list[str]:
    """Resolve which groups to convert: CLI > config `include_groups:` > all.

    ``requested`` is a comma-separated list, "all", or "none" (untagged entries
    only -- the way to drop every optional group in a config whose core topics are
    untagged). Unknown names are fatal rather than silently empty: a typo'd
    `--groups camrea` used to be indistinguishable from "that dataset has no
    images".
    """
    known = declared_groups(cfg)
    asked = None if requested is None else str(requested).strip()
    if asked in ("all", "none"):
        want = list(known) if asked == "all" else []
    elif asked is not None:
        want = [g.strip() for g in asked.split(",") if g.strip()]
    else:
        configured = cfg.get("include_groups")
        want = ([str(g) for g in configured] if configured is not None else list(known))
    if asked == "none":
        return want          # explicitly empty: skip the "nothing selected" check
    unknown = [g for g in want if g not in known]
    if unknown:
        raise SystemExit(f"unknown topic group(s) {unknown}; this config declares "
                         f"{known} (or pass --groups all)")
    # `known` empty means nothing is tagged, so an empty selection is correct: every
    # entry is unconditional. Only an explicit request for nothing is an error.
    if known and not want:
        raise SystemExit(f"no topic groups selected; this config declares {known}")
    return want


def filter_groups(cfg: dict[str, Any], groups: list[str]) -> dict[str, Any]:
    """Return ``cfg`` with `topics:` / `sources:` / `tf_static:` cut down to
    ``groups``. Copies only the filtered keys; the rest is shared."""
    keep = set(groups)
    out = dict(cfg)
    for key in ("topics", "sources", "tf_static"):
        entries = cfg.get(key)
        if entries is None:
            continue
        # `not tags` = untagged = unconditional.
        out[key] = [e for e in entries
                    if not (tags := entry_groups(e)) or keep & set(tags)]
    if not out.get("topics") and not out.get("sources"):
        raise SystemExit(f"groups {groups} select no topics; this config declares "
                         f"{declared_groups(cfg)}")
    return out


def check_topics_present(topic_cfgs: dict[str, Any], present: set[str], source: str) -> None:
    """Fail when a SELECTED topic is not in the input.

    A partial miss used to be silent, and the way it happened is not exotic:
    ``bag_conslam_kdamr.yaml`` selects the camera group by default, but rebag.py's
    input is an already-converted bag, so pointing it at a ``_fastlio_bag`` that
    was written with ``--groups lidar,imu`` produced a perfectly valid kdamr bag
    with no images in it and said nothing. The same shape catches a typo in `in:`
    -- ConSLAM's topics have no leading slash, which is easy to get wrong.

    The escape hatch is `--groups`: a sequence genuinely missing a topic (Hilti's
    `/Point_name` is only in some recordings) is converted by deselecting it,
    which is a decision on the command line rather than a surprise in the bag.
    """
    missing = sorted(t for t in topic_cfgs if t not in present)
    if not missing:
        return
    detail = ", ".join(
        f"{t} [{', '.join(entry_groups(topic_cfgs[t])) or 'untagged'}]" for t in missing)
    raise SystemExit(
        f"configured topics missing from {source}: {detail}\n"
        f"  present: {sorted(present)}\n"
        f"  either the input lacks them (re-convert the source with the group "
        f"selected -- rebag.py can only re-time what its input already contains), "
        f"or deselect them with --groups"
    )


def add_group_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--groups", default=None,
                        help="comma-separated topic groups to convert, `all`, or `none` "
                             "(untagged entries only). Default: the config's "
                             "`include_groups:`, or every group it declares. Groups are "
                             "tagged per topic in the config -- e.g. `--groups lidar,imu` "
                             "for an odometry-only bag, `--groups all` to include the "
                             "images. Untagged entries are always converted")


def output_summary(cfg: dict[str, Any]) -> str:
    """``sqlite3, rosbag2 v8 (Humble+Jazzy)`` -- what the done: line reports."""
    storage = (cfg.get("output") or {}).get("storage_id", "sqlite3")
    version = bag_version(cfg)
    readers = "Humble+Jazzy" if version == 8 else "Jazzy and newer"
    return f"{storage}, rosbag2 v{version} ({readers})"


def open_writer(out_bag: str, cfg: dict[str, Any]) -> Writer:
    """Open an output bag using the shared ``output:`` config block."""
    out_cfg = cfg.get("output") or {}
    storage = out_cfg.get("storage_id", "sqlite3")
    if storage not in STORAGE_PLUGINS:
        raise SystemExit(f"unknown storage_id {storage!r}; known: {sorted(STORAGE_PLUGINS)}")
    writer = Writer(out_bag, version=bag_version(cfg),
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


# --------------------------------------------------------------------------
# camera_info: intrinsics come from the calibration FILE, not from the recording
#
# A camera driver publishes a camera_info next to every frame, so a recording
# normally carries one already. Rebuilding it from the calibration instead is
# deliberate, and specific to what these datasets actually ship:
#
#   * ConSLAM's recorded `pp_rgb/camera_info` has an ALL-ZERO P. P is the 3x4
#     projection matrix of the rectified image, and the standard ROS path reads
#     P, not K: image_geometry's project3dToPixel (and RViz's Camera display,
#     and image_proc) computes P * [X Y Z 1] and divides by the third row, so a
#     zero P sends every point to NaN. K and D in the recording are correct, but
#     they are not what those consumers use.
#   * Hilti 2023 publishes no camera_info topic at all, so there is nothing to
#     pass through for that dataset in the first place.
#   * rosbags names the matrices D/K/R/P in its ROS 1 typestore and d/k/r/p in
#     its ROS 2 one, so copying a recorded message across the two needs a
#     field-by-field rebuild anyway -- the "just forward it" option does not
#     exist.
#
# That makes the calibration file the single source for intrinsics, exactly as it
# already is for the extrinsics in `tf_static:`.
CAMERA_INFO_FORMATS = ("ros_camera_info", "kalibr")


def _calib_numbers(entry: Any, key: str, path: str, expected: int | None = None) -> list[float]:
    """Read a `{rows, cols, data: [...]}` block (or a bare list) as finite floats.

    ``rows``/``cols`` are deliberately ignored: ConSLAM's calib_rgb.yaml declares
    ``rows: 3, cols: 3`` for its projection_matrix and then lists 12 numbers, so
    trusting the declared shape over the actual length would mis-read the file.
    """
    data = entry.get("data") if isinstance(entry, dict) else entry
    if data is None:
        raise SystemExit(f"camera_info: {path} has no `{key}`")
    try:
        vals = [float(v) for v in data]
    except (TypeError, ValueError):
        raise SystemExit(f"camera_info: `{key}` in {path} is not a list of numbers") from None
    if expected is not None and len(vals) != expected:
        raise SystemExit(f"camera_info: `{key}` in {path} has {len(vals)} values, "
                         f"expected {expected}")
    if not vals or not all(math.isfinite(v) for v in vals):
        raise SystemExit(f"camera_info: `{key}` in {path} is empty or contains NaN/Inf")
    return vals


def _read_ros_camera_info_yaml(path: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Parse the YAML that ROS's own camera_calibration writes (ConSLAM's
    calib_rgb.yaml / calib_nir.yaml): image_width/image_height, a 3x3
    camera_matrix, distortion_model and distortion_coefficients.

    One camera per file, so `camera:` (which selects one of kalibr's cam0..camN)
    is meaningless here and is refused rather than silently ignored -- it would
    mean the wrong `format:` was set.
    """
    if spec.get("camera") is not None:
        raise SystemExit(f"camera_info: `camera:` selects one camera out of a multi-camera "
                         f"calibration and only applies to format: kalibr, but {path} is "
                         f"read as ros_camera_info (one camera per file)")
    calib = load_yaml(path)
    try:
        width, height = int(calib["image_width"]), int(calib["image_height"])
    except KeyError as exc:
        raise SystemExit(f"camera_info: {path} has no {exc.args[0]} "
                         f"(is it really a ROS camera_info YAML? try another `format:`)") from None
    if width <= 0 or height <= 0:
        raise SystemExit(f"camera_info: {path} declares a {width}x{height} image")
    k = _calib_numbers(calib.get("camera_matrix"), "camera_matrix", path, 9)
    if k[0] == 0.0 or k[4] == 0.0:
        raise SystemExit(f"camera_info: {path} has a zero focal length (fx={k[0]}, fy={k[4]})")
    return {
        "width": width, "height": height,
        "distortion_model": str(calib.get("distortion_model", "plumb_bob")),
        "d": _calib_numbers(calib.get("distortion_coefficients"), "distortion_coefficients", path),
        "k": k,
    }


# kalibr's `distortion_model` -> the CameraInfo `distortion_model` string, with
# the number of coefficients each one carries in the file. ROS's plumb_bob D is
# [k1, k2, p1, p2, k3], so kalibr's 4-coefficient radtan is padded with k3 = 0.
# `equidistant` is Kannala-Brandt (KB4) fisheye: the name is what ROS 2 drivers
# publish for it, but note that image_geometry's PinholeCameraModel does NOT
# implement it -- undistort/project has to go through cv2.fisheye, and the
# `P = [K | 0]` this builds is a raw-image projection, not a rectified one.
_KALIBR_DISTORTION = {"equidistant": ("equidistant", 4), "radtan": ("plumb_bob", 4)}


def _read_kalibr_camera_info_yaml(path: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Parse one camera out of a kalibr multi-camera YAML (Hilti's
    calibration_cams.yaml): `intrinsics: [fx, fy, cx, cy]`, `resolution: [w, h]`,
    `distortion_model` + `distortion_coeffs`, keyed cam0..camN.

    `camera:` picks the entry, by kalibr key (``cam0``) or by its ``rostopic``
    (``/alphasense/cam0/image_raw``) -- the topic spelling is the one that cannot
    drift out of sync with the bag, and with five look-alike 720x540 cameras
    per recording, picking the wrong one is the mistake worth making impossible.
    """
    calib = load_yaml(path)
    if not isinstance(calib, dict) or not calib:
        raise SystemExit(f"camera_info: {path} is not a kalibr camera calibration (no cam* keys)")
    # Only mappings that actually describe a camera; kalibr writes nothing else at
    # the top level, but a merged calibration file could.
    cams = {k: v for k, v in calib.items() if isinstance(v, dict) and "intrinsics" in v}
    if not cams:
        raise SystemExit(f"camera_info: {path} has no camera with `intrinsics:` "
                         f"(is it really a kalibr file? try another `format:`)")
    by_topic = {str(v["rostopic"]): k for k, v in cams.items() if v.get("rostopic")}
    want = spec.get("camera")
    if want is None:
        raise SystemExit(f"camera_info: format: kalibr needs `camera:` to pick one of "
                         f"{sorted(cams)} in {path} (or one of {sorted(by_topic)})")
    want = str(want)
    key = want if want in cams else by_topic.get(want)
    if key is None:
        raise SystemExit(f"camera_info: no camera {want!r} in {path}; "
                         f"available: {sorted(cams)} / {sorted(by_topic)}")
    cam = cams[key]

    model = str(cam.get("camera_model", "pinhole"))
    if model != "pinhole":
        # `omni` carries a fifth intrinsic (the mirror parameter xi) that has no
        # place in a CameraInfo K, so it cannot be expressed here at all.
        raise SystemExit(f"camera_info: {path} {key} is camera_model {model!r}; "
                         f"only 'pinhole' maps onto CameraInfo's K")
    dist = str(cam.get("distortion_model", "radtan"))
    if dist not in _KALIBR_DISTORTION:
        raise SystemExit(f"camera_info: {path} {key} has distortion_model {dist!r}; "
                         f"known: {sorted(_KALIBR_DISTORTION)}")
    ros_dist, n_coeffs = _KALIBR_DISTORTION[dist]
    d = _calib_numbers(cam.get("distortion_coeffs"), f"{key}.distortion_coeffs", path, n_coeffs)
    if ros_dist == "plumb_bob":
        d = [*d, 0.0]   # kalibr radtan is k1,k2,p1,p2; plumb_bob wants a fifth (k3)

    fx, fy, cx, cy = _calib_numbers(cam.get("intrinsics"), f"{key}.intrinsics", path, 4)
    width, height = (int(v) for v in _calib_numbers(cam.get("resolution"),
                                                    f"{key}.resolution", path, 2))
    if width <= 0 or height <= 0:
        raise SystemExit(f"camera_info: {path} {key} declares a {width}x{height} image")
    if fx == 0.0 or fy == 0.0:
        raise SystemExit(f"camera_info: {path} {key} has a zero focal length (fx={fx}, fy={fy})")
    # A principal point outside the image means the entry does not belong to this
    # topic (or the file lists them in another order) -- catching that here beats
    # discovering it as points landing off-frame after a 13 GB conversion.
    if not (0.0 <= cx <= width and 0.0 <= cy <= height):
        raise SystemExit(f"camera_info: {path} {key} has principal point ({cx}, {cy}) "
                         f"outside its {width}x{height} image")
    return {
        "width": width, "height": height,
        "distortion_model": ros_dist,
        "d": d,
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
    }


CAMERA_INFO_READERS = {"ros_camera_info": _read_ros_camera_info_yaml,
                       "kalibr": _read_kalibr_camera_info_yaml}


def build_camera_info(spec: dict[str, Any], base_dir: str | None = None) -> Any:
    """Build one sensor_msgs/msg/CameraInfo from a `camera_info:` config block.

    The header is left blank: the caller stamps each copy from the image it
    accompanies, because image_transport's CameraSubscriber pairs image and
    camera_info by EXACT time and would otherwise never deliver a pair.

    ``R`` is identity and ``P`` is ``[K | 0]`` -- the ROS convention for an
    unrectified monocular image, which is what these bags carry. Note that the
    file's own `projection_matrix`, when it has one, is NOT used: ConSLAM's is
    for undistorted *and cropped* images (fx 2333.6 vs K's 2351.9), so applying
    it to the full-resolution raw frame would bias every projected point.

    For an `equidistant` (KB4 fisheye) camera -- every Hilti one -- ``P``
    carries less than it does for plumb_bob: the pinhole part of the model is
    still K, but no linear matrix undoes KB4 distortion, so a consumer must
    project through cv2.fisheye (or rectify first) rather than trusting ``P``.
    """
    if not isinstance(spec, dict) or not spec.get("from"):
        raise SystemExit("camera_info: needs a `from:` path to a calibration file")
    fmt = str(spec.get("format", "ros_camera_info"))
    if fmt not in CAMERA_INFO_READERS:
        raise SystemExit(f"camera_info: unknown format {fmt!r}; "
                         f"known: {sorted(CAMERA_INFO_READERS)}")
    path = str(spec["from"])
    # Relative paths resolve against the config file, not the working directory,
    # so a config can sit next to its calibration and be run from anywhere.
    if not os.path.isabs(path) and base_dir:
        path = os.path.join(base_dir, path)
    if not os.path.isfile(path):
        raise SystemExit(f"camera_info: calibration file not found: {path}")

    calib = CAMERA_INFO_READERS[fmt](path, spec)
    k = calib["k"]
    p = [k[0], k[1], k[2], 0.0, k[3], k[4], k[5], 0.0, k[6], k[7], k[8], 0.0]
    CameraInfo = ROS2_STORE.types["sensor_msgs/msg/CameraInfo"]
    RegionOfInterest = ROS2_STORE.types["sensor_msgs/msg/RegionOfInterest"]
    Header = ROS2_STORE.types["std_msgs/msg/Header"]
    return CameraInfo(
        header=Header(stamp=ns_to_time(0), frame_id=""),
        height=calib["height"], width=calib["width"],
        distortion_model=calib["distortion_model"],
        d=np.array(calib["d"], dtype=np.float64),
        k=np.array(k, dtype=np.float64),
        r=np.eye(3, dtype=np.float64).reshape(9),
        p=np.array(p, dtype=np.float64),
        binning_x=0, binning_y=0,
        roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
    )


def camera_info_topic(image_topic: str) -> str:
    """`/camera/rgb/image_raw` -> `/camera/rgb/camera_info`.

    The sibling name is what image_transport's CameraSubscriber (and RViz's
    Camera display) resolves from the image topic, so it is the default.
    """
    namespace = image_topic.rsplit("/", 1)[0]
    return f"{namespace}/camera_info" if namespace else "/camera_info"


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
