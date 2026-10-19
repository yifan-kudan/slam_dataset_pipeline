#!/usr/bin/env python3
"""Where the Hilti 2023 artefacts are, and what differs between the two rigs.

One place for the layout every tool here walks:

    <root>/<seq>_fastlio_bag/                     the converted LiDAR+IMU bag
        maps/fastlio_map.pcd                      canonical map
        fastlio_log/pos_log.txt                   canonical FAST-LIO log
        runs/<run>/{maps,fastlio_log,ros_logs}/    one parameter set, kept apart
    <results>/                                    align_*/cfg_*/params_*/run_*.csv,
                                                  fl_*/kd_*.json, report/

and one place for the two rig-dependent facts that decide whether a score is real:
the prism lever arm and, for the 10 bags rebagged without it, the imu->velodyne
static transform. Both come from config/hilti/platforms.yaml.

    python3 src/hilti/dataset.py                  # resolved layout + sequence table
    python3 src/hilti/dataset.py site2_robot_1    # one sequence's paths
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths                                                       # noqa: E402

DATASET = "hilti_2023"
# Overridable so a campaign can run against a differently-named bag set (the
# unified kdamr bags are <seq>_kdamr_bag) without every caller growing a flag.
BAG_SUFFIX = os.environ.get("SDP_BAG_SUFFIX", "_fastlio_bag")
CLOUD_TOPIC = "/velodyne_points"
IMU_TOPIC = "/imu"

_config_cache: dict[str, dict] = {}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def _config(name: str) -> dict:
    """A yaml under config/hilti/, parsed once."""
    if name not in _config_cache:
        file = paths.CONFIG_DIR / "hilti" / f"{name}.yaml"
        if not file.is_file():
            raise SystemExit(f"missing {file}\n"
                             f"  it is committed with the repo; restore it from git")
        with open(file, "r", encoding="utf-8") as f:
            _config_cache[name] = yaml.safe_load(f) or {}
    return _config_cache[name]


def locations() -> dict[str, str]:
    """datasets.hilti_2023 from config/paths.yaml: root, raw, groundtruth, kdm, results."""
    return paths.dataset(DATASET)


def sequences() -> dict[str, dict]:
    """Every sequence with its platform, site, map flag and run, defaults filled in."""
    cfg = _config("sequences")
    default_run = cfg.get("default_run", "r0.3")
    out = {}
    for seq, entry in cfg["sequences"].items():
        merged = dict(entry)
        merged.setdefault("run", default_run)
        merged.setdefault("score_tag", "")
        merged["seq"] = seq
        out[seq] = merged
    return out


def sequence(seq: str) -> dict:
    try:
        return sequences()[seq]
    except KeyError:
        raise SystemExit(f"unknown sequence {seq!r}; known: "
                         f"{', '.join(sequences())}") from None


def platform(seq: str) -> str:
    """'robot' or 'handheld', from the sequence table (name pattern as the fallback)."""
    entry = sequences().get(seq)
    if entry:
        return entry["platform"]
    return "robot" if _config("platforms")["robot_name_pattern"] in seq else "handheld"


def platform_config(seq: str) -> dict:
    return _config("platforms")["platforms"][platform(seq)]


def static_tf(seq: str) -> dict[str, list[float]]:
    """imu_link -> velodyne for the bags that were rebagged without /tf_static.

    kd_localizer has base_frame_id imu_link and clouds stamped `velodyne`, so without
    this transform in TF it never leaves UNINITIALIZED. FAST-LIO does not care: it
    reads its extrinsic from its own config and ignores tf.
    """
    return platform_config(seq)["static_tf"]


def lever_arm(seq: str):
    """Constant prism offset in the IMU frame, or None when it is per control point.

    The robot has no constant arm: <seq>_lidar.txt gives the prism in the LiDAR frame
    at every control point (see gcp.py). Using the handheld's 20 cm boom there instead
    invents ~1.5 m of error.
    """
    import numpy as np           # imported here: see the note on the yaml import above
    arm = platform_config(seq).get("prism_in_imu")
    return None if arm is None else np.asarray(arm, dtype=float)


def lidar_from_imu(seq: str):
    """imu -> lidar for the robot rig, as a 4x4."""
    import numpy as np
    matrix = platform_config(seq).get("lidar_from_imu")
    if matrix is None:
        raise SystemExit(f"{seq}: platform {platform(seq)} has no lidar_from_imu in "
                         f"config/hilti/platforms.yaml")
    return np.asarray(matrix, dtype=float)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def root(override: str | None = None) -> str:
    return override or locations()["root"]


def results_dir(override: str | None = None) -> str:
    return override or locations()["results"]


def bag_dir(seq: str, root_override: str | None = None) -> str:
    return os.path.join(root(root_override), f"{seq}{BAG_SUFFIX}")


def run_dir(seq: str, run: str | None = None, root_override: str | None = None) -> str:
    """``<bag>/runs/<run>``, or the bag itself for the canonical (unnamed) run."""
    bag = bag_dir(seq, root_override)
    resolved = run if run is not None else sequence(seq)["run"]
    return bag if resolved in (None, "", ".") else os.path.join(bag, "runs", resolved)


def pos_log(seq: str, run: str | None = None, root_override: str | None = None) -> str:
    return os.path.join(run_dir(seq, run, root_override), "fastlio_log", "pos_log.txt")


def map_pcd(seq: str, run: str | None = None, root_override: str | None = None) -> str:
    """The cloud a run wrote. Named per sequence inside a run dir, generic outside it."""
    directory = run_dir(seq, run, root_override)
    generic = os.path.join(directory, "maps", "fastlio_map.pcd")
    named = os.path.join(directory, "maps", f"{seq}_fastlio_map.pcd")
    return named if os.path.exists(named) or directory != bag_dir(seq, root_override) else generic


def kdm(seq: str) -> str:
    return os.path.join(locations()["kdm"], f"{seq}_fastlio_map.kdm")


def score_json(seq: str, kind: str = "fl", tag: str | None = None,
               results_override: str | None = None) -> str:
    """``<results>/fl_<seq>[_<tag>].json`` -- the evaluators' output for one run."""
    resolved = sequence(seq)["score_tag"] if tag is None else tag
    suffix = f"_{resolved}" if resolved else ""
    return os.path.join(results_dir(results_override), f"{kind}_{seq}{suffix}.json")


def monitor_csv(seq: str, results_override: str | None = None) -> str:
    return os.path.join(results_dir(results_override), f"run_{seq}.csv")


def align_json(seq: str, results_override: str | None = None) -> str:
    return os.path.join(results_dir(results_override), f"align_{seq}.json")


def report_dir(results_override: str | None = None) -> str:
    """Where the report chain writes data.json, the figures and the tables."""
    return os.path.join(results_dir(results_override), "report")


def planes_dir(results_override: str | None = None) -> str:
    """Plane-voxel CSVs dumped from the .kdm files by scripts/hilti/map_dump.cpp."""
    return os.path.join(results_dir(results_override), "planes")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seq", nargs="?", help="print this sequence's paths only")
    ap.add_argument("--tf-args", action="store_true",
                    help="print the static_transform_publisher arguments for this "
                         "sequence's imu_link -> velodyne transform, for the bags that "
                         "were rebagged without /tf_static")
    a = ap.parse_args(argv)

    if a.tf_args:
        if not a.seq:
            raise SystemExit("--tf-args needs a sequence")
        tf = static_tf(a.seq)
        x, y, z = tf["translation"]
        qx, qy, qz, qw = tf["rotation_xyzw"]
        print(f"--x {x} --y {y} --z {z} --qx {qx} --qy {qy} --qz {qz} --qw {qw} "
              f"--frame-id imu_link --child-frame-id velodyne")
        return 0

    if a.seq:
        entry = sequence(a.seq)
        print(f"{a.seq}  platform={entry['platform']} site={entry['site']} "
              f"kdm={entry['kdm']} run={entry['run']}")
        for label, value in (("bag", bag_dir(a.seq)), ("run dir", run_dir(a.seq)),
                            ("pos_log", pos_log(a.seq)), ("map", map_pcd(a.seq)),
                            ("kdm", kdm(a.seq)), ("monitor csv", monitor_csv(a.seq)),
                            ("align json", align_json(a.seq)),
                            ("fl score", score_json(a.seq, "fl"))):
            print(f"  {label:<12} {value}{'' if os.path.exists(value) else '   <- missing'}")
        return 0

    for key, value in locations().items():
        print(f"{key:<12} {value}{'' if os.path.exists(value) else '   <- missing'}")
    print()
    print(f"{'sequence':<18} {'platform':<9} {'site':<5} {'kdm':<5} {'run':<8} pos_log")
    for seq, entry in sequences().items():
        log = pos_log(seq)
        print(f"{seq:<18} {entry['platform']:<9} {entry['site']:<5} "
              f"{str(entry['kdm']):<5} {entry['run']:<8} "
              f"{'ok' if os.path.exists(log) else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
