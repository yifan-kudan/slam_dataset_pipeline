#!/usr/bin/env python3
"""Turn a campaign.yaml into one directory of inputs per localizer run.

    make_runs.py <campaign dir>              reads <dir>/campaign.yaml
    make_runs.py <campaign dir> --dry-run    only list the runs it would write
    make_runs.py <campaign dir> --force      also rewrite the inputs of runs already done

Writes, under the campaign directory:

    campaign.env               CAMPAIGN, SDP and COMPOSE_SERVICE for scripts/kdloc/campaign.sh
    runs.tsv                   one row per run, in the order campaign.sh runs them
    runs/<name>/run.env        what run_one.sh needs: bag, topics, relocalizer delay, DDS, limits
    runs/<name>/params.yaml    kd_localizer_node + kd_relocalizer_node parameters
    runs/<name>/cfg.yaml       the localizer config (config_file)
    runs/<name>/init.yaml      initial pose, only for groups with `initial_pose: true`
    runs/<name>/static_tf.args one static_transform_publisher argument line per transform

Run outputs go to runs/<name>/out/ and are never touched here. Changing the inputs of a run
that is already done (runs/<name>/done exists) is refused unless --force is given, because the
results on disk would then describe parameters that are no longer in its directory.

Run names are <kind>_<sequence>, plus _in_<map> when the campaign has more than one map and
_r<k> when the group repeats. Runs are ordered in passes: the first repeat of every group on
every sequence comes before any second repeat, so a partly finished campaign covers everything.

The campaign.yaml schema is documented in config/examples/kdloc_campaign.example.yaml.
"""
import argparse
import copy
import math
import re
import shlex
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

# The localizer config used by every Jetson campaign since the 2.0.0 retest (ConSLAM, Hilti,
# Shinjuku, LUH). A campaign replaces it with `localizer_config`; a group can override single
# keys with its own `localizer_config`, which is merged into the campaign's.
STANDARD_LOCALIZER_CONFIG = {
    "lidar_preprocessing": {"voxel_size": 0.2, "deskew": True},
    "motion_predictor": {
        "type": "imu",
        "predicted_pose_time_tolerance": 0.02,
        "motion_model": {
            "gyro_sigma": 0.1, "accel_sigma": 0.1, "gyro_bias_sigma": 1e-3, "accel_bias_sigma": 1e-3,
            "prior_lin_vel_sigma": 0.1, "prior_gyro_bias_sigma": 0.1, "prior_accel_bias_sigma": 0.5,
            "prior_gravity_sigma": 0.3,
        },
    },
    "initial_align": {"search_radius": 0.5, "match_sigma": 0.1, "max_iterations": 30},
    "align": {"search_radius": 0.2, "match_sigma": 0.1, "max_iterations": 10},
    "num_threads": 8,
}

LOCALIZER_DEFAULTS = {"publish_tf": False, "lidar_exclude_radius": 0.5, "process_stats_log_interval_sec": 10}
RELOCALIZER_DEFAULTS = {"lidar_exclude_radius": 0.5}
METHODS = ("bbs3d_gpu", "bbs3d_cpu")

CAMPAIGN_KEYS = {"container", "sdp", "topics", "frames", "static_tf", "dds_profile", "ros_domain_id",
                 "startup_timeout_s", "play_rate", "localizer_config", "localizer_params",
                 "relocalizer_params", "maps", "sequences", "groups"}
GROUP_KEYS = {"kind", "method", "initial_pose", "reloc_delay", "repeats", "sequences", "maps",
              "localizer_config", "localizer_params", "relocalizer_params", "limit_s"}
SEQUENCE_KEYS = {"bag", "initial_pose"}
CONTAINER_KEYS = {"service", "ros_setup", "kd_setup"}


class Loader(yaml.SafeLoader):
    """SafeLoader that also reads 1e-3 as a float (YAML 1.1 wants 1.0e-3), as yaml-cpp does."""


Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                   |\.[0-9_]+(?:[eE][-+][0-9]+)?
                   |[-+]?\.(?:inf|Inf|INF)
                   |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=Loader)


class CampaignError(Exception):
    pass


def check_keys(where, mapping, allowed):
    """Refuse unknown keys: a misspelt option would otherwise be ignored without a word."""
    unknown = set(mapping) - allowed
    if unknown:
        raise CampaignError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def merge(base, override):
    """Deep merge of two mappings; values of `override` win."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve(campaign_dir, path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else (campaign_dir / p)


def quaternion_to_rpy(qx, qy, qz, qw):
    """(roll, pitch, yaw) with R = Rz(yaw) Ry(pitch) Rx(roll), kd::Rotation3d's convention."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r10 = 2 * (qx * qy + qz * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    return math.atan2(r21, r22), -math.asin(max(-1.0, min(1.0, r20))), math.atan2(r10, r00)


def initial_pose_yaml(pose, where):
    """The localizer's initial_pose_file: position [x, y, z], rotation [roll, pitch, yaw] in rad."""
    check_keys(where, pose, {"position", "orientation", "rotation"})
    position = [float(v) for v in pose["position"]]
    if "rotation" in pose:
        rpy = [float(v) for v in pose["rotation"]]
    elif "orientation" in pose:
        rpy = list(quaternion_to_rpy(*[float(v) for v in pose["orientation"]]))
    else:
        raise CampaignError(f"{where}: needs `orientation` [qx, qy, qz, qw] or `rotation` [roll, pitch, yaw]")
    if len(position) != 3 or len(rpy) != 3:
        raise CampaignError(f"{where}: position and rotation need three values each")
    return ("# rotation = roll, pitch, yaw (rad), R = Rz(yaw) Ry(pitch) Rx(roll)\n"
            f"position: [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]\n"
            f"rotation: [{rpy[0]:.8f}, {rpy[1]:.8f}, {rpy[2]:.8f}]\n")


def plan_runs(campaign, campaign_dir):
    """Every run of the campaign as a dict, in execution order."""
    check_keys("campaign.yaml", campaign, CAMPAIGN_KEYS)
    for key in ("container", "topics", "frames", "maps", "sequences", "groups"):
        if not campaign.get(key):
            raise CampaignError(f"campaign.yaml: `{key}` is required")
    check_keys("container", campaign["container"], CONTAINER_KEYS)
    if not campaign["container"].get("service"):
        raise CampaignError("campaign.yaml: container.service is required")
    for key in ("lidar", "imu"):
        if not campaign["topics"].get(key):
            raise CampaignError(f"campaign.yaml: topics.{key} is required")
    if not campaign["frames"].get("base"):
        raise CampaignError("campaign.yaml: frames.base is required")

    maps = {name: resolve(campaign_dir, path) for name, path in campaign["maps"].items()}
    sequences = campaign["sequences"]
    for seq, entry in sequences.items():
        check_keys(f"sequences.{seq}", entry, SEQUENCE_KEYS)
        if "bag" not in entry:
            raise CampaignError(f"sequences.{seq}: `bag` is required")
        for map_name in entry.get("initial_pose") or {}:
            if map_name not in maps:
                raise CampaignError(f"sequences.{seq}.initial_pose: no map called {map_name!r}")

    groups = []
    for i, group in enumerate(campaign["groups"]):
        where = f"groups[{i}]"
        check_keys(where, group, GROUP_KEYS)
        kind = group.get("kind", "")
        if not re.fullmatch(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*", kind):
            raise CampaignError(f"{where}: `kind` must be letters, digits and single underscores, "
                                f"got {kind!r}")
        method = group.get("method", "bbs3d_gpu")
        if method not in METHODS:
            raise CampaignError(f"{where}: method must be one of {METHODS}, got {method!r}")
        for key, known in (("sequences", sequences), ("maps", maps)):
            for name in group.get(key) or []:
                if name not in known:
                    raise CampaignError(f"{where}.{key}: {name!r} is not defined in campaign.yaml")
        groups.append(dict(group, method=method))

    several_maps = len(maps) > 1
    runs = []
    for repeat in range(1, max(int(g.get("repeats", 1)) for g in groups) + 1):
        for seq in sequences:
            for group in groups:
                repeats = int(group.get("repeats", 1))
                if repeat > repeats or seq not in (group.get("sequences") or sequences):
                    continue
                for map_name in group.get("maps") or maps:
                    pose = None
                    if group.get("initial_pose"):
                        pose = (sequences[seq].get("initial_pose") or {}).get(map_name)
                        if pose is None:
                            if repeat == 1:
                                print(f"note: no initial pose for {seq} in map {map_name}; "
                                      f"group {group['kind']} skips it")
                            continue
                    name = f"{group['kind']}_{seq}"
                    name += f"_in_{map_name}" if several_maps else ""
                    name += f"_r{repeat}" if repeats > 1 else ""
                    runs.append({"name": name, "kind": group["kind"], "sequence": seq, "map_name": map_name,
                                 "map": maps[map_name], "bag": resolve(campaign_dir, sequences[seq]["bag"]),
                                 "group": group, "initial_pose": pose})
    names = [r["name"] for r in runs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise CampaignError(f"two groups produce the same run names {duplicates}; give them different kinds")
    return runs


def run_files(run, campaign, campaign_dir, sdp):
    """File name -> content for one run directory."""
    group = run["group"]
    run_dir = campaign_dir / "runs" / run["name"]
    frames = campaign["frames"]
    header = (f"# {run['name']}: {run['sequence']} in map {run['map_name']}, group {run['kind']}, "
              f"relocalizer {group['method']}\n")
    files = {}

    config = campaign.get("localizer_config") or STANDARD_LOCALIZER_CONFIG
    config = merge(config, group.get("localizer_config"))
    files["cfg.yaml"] = header + yaml.safe_dump(config, sort_keys=False)

    localizer = {"map": str(run["map"]), "config_file": str(run_dir / "cfg.yaml"),
                 "map_frame_id": frames.get("map", "map"), "odom_frame_id": frames.get("odom", "odom"),
                 "base_frame_id": frames["base"], "session_data_directory": str(run_dir / "out" / "session")}
    localizer = merge(merge(merge(localizer, LOCALIZER_DEFAULTS), campaign.get("localizer_params")),
                      group.get("localizer_params"))
    if run["initial_pose"] is not None:
        files["init.yaml"] = initial_pose_yaml(
            run["initial_pose"], f"sequences.{run['sequence']}.initial_pose.{run['map_name']}")
        localizer["initial_pose_file"] = str(run_dir / "init.yaml")
    relocalizer = {"map": str(run["map"]), "map_frame_id": frames.get("map", "map"),
                   "base_frame_id": frames["base"], "method": group["method"]}
    relocalizer = merge(merge(merge(relocalizer, RELOCALIZER_DEFAULTS), campaign.get("relocalizer_params")),
                        group.get("relocalizer_params"))
    files["params.yaml"] = header + yaml.safe_dump(
        {"kd_localizer_node": {"ros__parameters": localizer},
         "kd_relocalizer_node": {"ros__parameters": relocalizer}}, sort_keys=False)

    static_tf = campaign.get("static_tf") or []
    if isinstance(static_tf, str):
        static_tf = [static_tf]
    if static_tf:
        files["static_tf.args"] = "".join(" ".join(str(line).split()) + "\n" for line in static_tf)

    # A warm run starts the relocalizer `reloc_delay` s after the bag, so it cannot answer the
    # start-up UNINITIALIZED window and replace the initial pose; a cold run needs it at once.
    reloc_delay = group.get("reloc_delay", 30 if run["initial_pose"] is not None else 0)
    container = campaign["container"]
    env = {
        "NAME": run["name"], "KIND": run["kind"], "SEQUENCE": run["sequence"], "MAP_NAME": run["map_name"],
        "MAP": str(run["map"]), "BAG": str(run["bag"]), "PARAMS": str(run_dir / "params.yaml"),
        "LIDAR_TOPIC": campaign["topics"]["lidar"], "IMU_TOPIC": campaign["topics"]["imu"],
        "RELOC_DELAY": str(reloc_delay), "PLAY_RATE": str(campaign.get("play_rate", 1.0)),
        "LIMIT": str(group.get("limit_s", 0)),
        "DDS_PROFILE": str(resolve(sdp, campaign.get("dds_profile", "config/dds/fastdds_shm_big.xml"))),
        "ROS_DOMAIN_ID": str(campaign.get("ros_domain_id", 42)),
        "STARTUP_TIMEOUT": str(campaign.get("startup_timeout_s", 180)),
        # Empty means run_one.sh's defaults: /opt/ros/humble and ~/ros2_ws/install in the container.
        "ROS_SETUP": container.get("ros_setup", ""),
        "KD_SETUP": container.get("kd_setup", ""),
    }
    files["run.env"] = "".join(f"{key}={shlex.quote(value)}\n" for key, value in env.items())
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="list the runs, write nothing")
    parser.add_argument("--force", action="store_true", help="rewrite the inputs of runs already done")
    args = parser.parse_args(argv)

    campaign_dir = args.campaign_dir.resolve()
    campaign = load_yaml(campaign_dir / "campaign.yaml")
    sdp = resolve(campaign_dir, campaign["sdp"]) if campaign.get("sdp") else REPO
    try:
        runs = plan_runs(campaign, campaign_dir)
        planned = [(run, run_files(run, campaign, campaign_dir, sdp)) for run in runs]
    except CampaignError as e:
        sys.exit(f"error: {e}")

    for path in [*{r["map"] for r in runs}, *{r["bag"] for r in runs}]:
        if not path.exists():
            print(f"warning: {path} does not exist (yet)")

    # Refuse to rewrite the inputs of a finished run: its results would no longer match them.
    changed_done = []
    for run, files in planned:
        run_dir = campaign_dir / "runs" / run["name"]
        if (run_dir / "done").exists():
            stale = [f for f, text in files.items()
                     if not (run_dir / f).exists() or (run_dir / f).read_text() != text]
            if stale:
                changed_done.append(f"{run['name']} ({', '.join(stale)})")
    if changed_done and not args.force:
        sys.exit("error: these runs are done but their inputs would change:\n  " + "\n  ".join(changed_done)
                 + "\nRe-run with --force to rewrite them (then delete runs/<name>/done to repeat them).")

    rows = ["name\tkind\tsequence\tmap\tbag\treloc_delay\tmethod"]
    for run, files in planned:
        env = dict(line.split("=", 1) for line in files["run.env"].splitlines())
        rows.append("\t".join([run["name"], run["kind"], run["sequence"], run["map_name"], str(run["bag"]),
                               shlex.split(env["RELOC_DELAY"])[0], run["group"]["method"]]))
    if args.dry_run:
        print("\n".join(rows))
        print(f"{len(runs)} runs (dry run, nothing written)")
        return

    for run, files in planned:
        run_dir = campaign_dir / "runs" / run["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("init.yaml", "static_tf.args"):
            if name not in files and (run_dir / name).exists():
                (run_dir / name).unlink()
        for name, text in files.items():
            (run_dir / name).write_text(text)
    (campaign_dir / "runs.tsv").write_text("\n".join(rows) + "\n")
    (campaign_dir / "campaign.env").write_text(
        f"CAMPAIGN={shlex.quote(str(campaign_dir))}\nSDP={shlex.quote(str(sdp))}\n"
        f"COMPOSE_SERVICE={shlex.quote(str(campaign['container']['service']))}\n")
    known = {r["name"] for r in runs}
    orphans = sorted(p.name for p in (campaign_dir / "runs").iterdir() if p.is_dir() and p.name not in known)
    print(f"{len(runs)} runs -> {campaign_dir / 'runs.tsv'}")
    if orphans:
        print(f"note: runs/ also holds {len(orphans)} directories no longer in campaign.yaml: "
              f"{', '.join(orphans[:5])}{' ...' if len(orphans) > 5 else ''}")


if __name__ == "__main__":
    main()
