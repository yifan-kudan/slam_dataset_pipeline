#!/usr/bin/env python3
"""
Run FAST-LIO over a batch of ROS 2 bags, one bag at a time.

Python port of ``run_fastlio_bags.sh`` with the same per-bag sequence -- launch
the node, play the bag against ``--clock``, call ``/map_save``, shut the node
down with SIGINT so its logs flush, copy the logs out -- and the same layout of
outputs next to each bag::

    <bag>/maps/fastlio_map.pcd   map, written directly by FAST-LIO
    <bag>/fastlio_log/           pos_log.txt, mat_pre/mat_out/dbg/imu.txt, time log
    <bag>/ros_logs/              ROS 2 launch and node logs
    <bag>/results/               recorded odometry bag, .tum export, eval report

Three things this does beyond the shell version:

  * **A failed bag does not end the batch.** The shell script runs under
    ``set -e``, so a non-zero ``ros2 bag play`` or a ``/map_save`` timeout aborts
    every sequence still queued. Here each bag is isolated and the run ends with
    a pass/fail table; the exit status is still non-zero if anything failed.
  * **Resume and select.** ``--start-from`` / ``--only`` pick up a batch that
    died halfway without hand-editing a bag list.
  * **--dry-run** prints the plan (bags, staged config, output paths) so a
    12-sequence run can be checked before it takes hours.

Bags are processed strictly serially: FAST-LIO is a single global ROS graph
driven by ``/clock``, so two bags at once would interleave into one map.

Two things every run gets whether it asks or not:

  * **A Fast DDS profile with a big shared-memory segment**, exported for the node
    AND for the player (they share one environment). Without it this host drops
    ~50 % of 1.37 MB clouds at the *writer* -- intra-host payload goes over shared
    memory, the default segment cannot hold that sample stream, and the writer
    discards rather than waits. ``--dds-profile none`` opts out; see
    config/dds/fastdds_shm_big.xml.
  * **Frame accounting.** After each bag the processed-scan count (pos_log.txt rows)
    is compared with the bag's own cloud count and the run FAILS below
    ``--min-frame-pct``. That loss is silent otherwise: the map thins out, the
    trajectory keeps looking plausible, and every number downstream is quietly
    computed from half the data.

No path in this file is absolute: the ROS setup files, the FAST-LIO log directory,
the RViz config and the dataset roots all come from config/paths.yaml (src/paths.py).

``--root`` is the *dataset* root (where the ``<sequence>/`` folders with
``pose.csv`` are). Bags are looked for in ``<root>/Converted_bag`` -- where
batch_to_bag.py writes them -- falling back to ``<root>`` itself; override with
``--bag-root``.

Examples::

    # ConPR, PointCloud2 bags from batch_to_bag.py (Converted_bag/<seq>_pc2_bag).
    # --root defaults to datasets.conpr.root in config/paths.yaml.
    python3 src/batch_fastlio.py --glob '*_pc2_bag' \\
        --config-file conpr/fast_lio_conpr.yaml

    # ConSLAM, no RViz, record + evaluate odometry against pose.csv
    python3 src/batch_fastlio.py --root .../ConSLAM --glob '*_fastlio_bag' \\
        --config-file conslam/fast_lio_conslam.yaml --no-rviz --eval-odom

    # resume an aborted batch, collecting every map into one directory
    python3 src/batch_fastlio.py --root .../ConPR --glob '*_pc2_bag' \\
        --start-from 20230706 --map-dir /media/kudan/seagate-2tb/maps

    # bags somewhere else entirely
    python3 src/batch_fastlio.py --root .../ConPR --bag-root /mnt/scratch/bags

    # one Hilti sequence into its own run directory, leaving the canonical
    # <bag>/maps and <bag>/fastlio_log untouched (what run_fastlio_seq.sh used to do)
    python3 src/batch_fastlio.py --dataset hilti_2023 --glob '*_fastlio_bag' \\
        --config-file hilti/fast_lio_hilti.yaml --only site2_handheld_5 \\
        --run-name qosfix --no-rviz
"""
import argparse
import fnmatch
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import yaml

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# FAST-LIO writes its debug logs to a path baked in at compile time
# (CMakeLists.txt sets -DROOT_DIR=<source dir>/), so they always land in the
# source tree and every bag overwrites the previous one's. Copied out per bag.
# Set from config/paths.yaml in main(); the module level default keeps the
# functions below usable (and testable) without a config.
FASTLIO_LOG_SRC = ""
# imu.txt comes from IMU_Processing.hpp; empty files are skipped when copying.
# Nothing to collect from PCD/: the pcd_save_en block in laserMapping.cpp is
# commented out, so scans.pcd is never written despite the config key.
FASTLIO_LOG_FILES = ("pos_log.txt", "mat_pre.txt", "mat_out.txt", "dbg.txt",
                     "imu.txt", "fast_lio_time_log.csv")

WAIT_AFTER_LAUNCH = 8.0
WAIT_AFTER_PLAY = 5.0
WAIT_AFTER_SAVE = 2.0
SERVICE_TIMEOUT = 60.0
STOP_GRACE = 10.0

LIVE: list[subprocess.Popen] = []


# --------------------------------------------------------------------------
# ROS environment
# --------------------------------------------------------------------------
def ros_env(setups: list[str]) -> dict[str, str]:
    """Env after sourcing the ROS / overlay setup files.

    ``source setup.bash`` only mutates the shell that runs it, so the sourcing
    is done once in a throwaway bash and the resulting environment is captured
    for every child process. ``env -0`` rather than ``env`` because ROS values
    (AMENT_PREFIX_PATH and friends) may contain anything but NUL.
    """
    script = "".join(f'source "{p}" || exit 1\n' for p in setups) + "env -0"
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"failed to source setup files:\n{proc.stderr.strip()}")
    env = {}
    for entry in proc.stdout.split("\0"):
        key, sep, value = entry.partition("=")
        if sep:
            env[key] = value
    return env


def with_dds_profile(env: dict[str, str], profile: str | None) -> dict[str, str]:
    """Add FASTRTPS_DEFAULT_PROFILES_FILE to the environment every child inherits.

    One environment covers the node and ``ros2 bag play`` both, which is the point:
    the loss this fixes is at the WRITER, so the player needs the profile as much as
    the subscriber does. Exporting it for only one of them fixes nothing.
    """
    if profile is None:
        profile = paths.get("ros.dds_profile")
    if profile in ("none", "off", ""):
        print("DDS profile: none (stock transport -- expect frame loss on big clouds)",
              file=sys.stderr)
        return env
    if not os.path.isfile(profile):
        raise SystemExit(f"DDS profile not found: {profile}\n"
                         f"  fix ros.dds_profile in config/paths.yaml, pass "
                         f"--dds-profile <xml>, or --dds-profile none to run without one")

    rmw = env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    if "fastrtps" not in rmw:
        print(f"warning: RMW_IMPLEMENTATION={rmw} ignores a Fast DDS profile; "
              f"{profile} will have no effect", file=sys.stderr)
    print(f"DDS profile: {profile}")
    return dict(env, FASTRTPS_DEFAULT_PROFILES_FILE=profile)


# --------------------------------------------------------------------------
# Config staging
# --------------------------------------------------------------------------
MAP_PATH_RE = re.compile(r"^([ \t]*map_file_path:).*$", re.MULTILINE)


def stage_config(src: str, rel_name: str, map_path: str, tmp_dir: str) -> str:
    """Copy the config with ``map_file_path`` pointed at this bag's map.

    FAST-LIO reads map_file_path once at startup and mapping.launch.py exposes
    no pass-through for parameter overrides, so the only way to place the map
    per bag is a rewritten copy of the config.
    """
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()
    staged, n = MAP_PATH_RE.subn(rf'\1 "{map_path}"', text)
    if n == 0:
        print(f"warning: no map_file_path key in {src}; "
              f"map will not land in {map_path}", file=sys.stderr)

    dest = os.path.join(tmp_dir, rel_name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(staged)
    return dest


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def bag_roots_for(root: str, bag_root: str | None) -> list[str]:
    """Directories to search for bags, in priority order.

    ``--root`` has to stay at the dataset root, because that is where the
    per-sequence ``pose.csv`` lives (``sequence_dir``). But batch_to_bag.py writes
    its output to ``<root>/Converted_bag/``, so searching only one level under
    ``--root`` finds nothing -- the two scripts disagreed about what ``--root``
    means. Search the conversion output directory first, then the root itself
    (the older layout, where bags sit next to the sequence folders).
    """
    if bag_root:
        return [os.path.abspath(bag_root)]
    converted = os.path.join(root, "Converted_bag")
    return [converted, root] if os.path.isdir(converted) else [root]


def find_bags(roots: list[str], pattern: str) -> tuple[list[str], str]:
    """(matching bag dirs, the root they came from) for the first root that has any."""
    for root in roots:
        if not os.path.isdir(root):
            continue
        bags = sorted(os.path.join(root, n) for n in os.listdir(root)
                      if os.path.isdir(os.path.join(root, n)) and fnmatch.fnmatch(n, pattern))
        if bags:
            return bags, root
    return [], roots[0]


def resolve_bag_path(name: str, roots: list[str]) -> str:
    """A positional bag argument -> absolute path, tried against each bag root."""
    if os.path.isabs(name):
        return name
    for root in roots:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(roots[0], name)   # the caller reports the missing directory


def sequence_name(bag_name: str, pattern: str) -> str:
    """`20230531_custom_bag` -> `20230531`, i.e. the sibling folder with pose.csv.

    The suffix comes from the discovery glob when it is a plain `*<suffix>`
    (so `--glob '*_fastlio_bag'` strips `_fastlio_bag`), which keeps this
    working across datasets instead of hardcoding one dataset's naming.
    """
    if pattern.startswith("*") and "*" not in pattern[1:] and "?" not in pattern[1:]:
        suffix = pattern[1:]
        if suffix and bag_name.endswith(suffix):
            return bag_name[: -len(suffix)]
    # Fallback for a glob that is not a plain `*<suffix>`. The middle segment has to
    # allow digits: `[a-z]+` could not match `pc2`, so `20230706_pc2_bag` came back
    # as `20230706_pc2` and pose.csv was then looked for in a directory that does
    # not exist (--eval-odom silently skipped, --only silently matching nothing).
    return re.sub(r"_(?:[a-z0-9]+_)?bag$", "", bag_name)


def select(bags: list[str], start_from: str | None, only: list[str] | None,
           pattern: str = "*") -> list[str]:
    """Apply --only / --start-from, matching on either bag or sequence name.

    ``pattern`` is the discovery glob, threaded through so the sequence name comes
    from the reliable suffix-from-glob path rather than always from the regex
    fallback -- otherwise `--only 20230706` could not match a `20230706_pc2_bag`.
    """
    def names(path: str) -> tuple[str, str]:
        base = os.path.basename(path.rstrip("/"))
        return base, sequence_name(base, pattern)

    if only:
        wanted = {os.path.basename(o.rstrip("/")) for o in only}
        kept = [b for b in bags if wanted & set(names(b))]
        found = {n for b in kept for n in names(b)}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"--only {missing} matched none of: "
                             f"{[os.path.basename(b) for b in bags]}")
        bags = kept

    if start_from:
        target = os.path.basename(start_from.rstrip("/"))
        for i, bag in enumerate(bags):
            if target in names(bag):
                return bags[i:]
        raise SystemExit(f"--start-from {start_from} matched none of: "
                         f"{[os.path.basename(b) for b in bags]}")
    return bags


# --------------------------------------------------------------------------
# Process control
# --------------------------------------------------------------------------
def spawn(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    """Start a child in its own process group.

    ``ros2 launch`` and ``ros2 bag record`` are process trees; a signal has to
    reach the whole group or the real node survives its launcher.
    """
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    LIVE.append(proc)
    return proc


def signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def stop(proc: subprocess.Popen | None, grace: float = STOP_GRACE,
         first: int = signal.SIGINT, label: str = "process") -> None:
    """Signal a child's group and reap it, escalating only if it hangs.

    laserMapping.cpp only writes fast_lio_time_log.csv after rclcpp::spin()
    returns, and flushes mat_pre/mat_out/dbg in the node destructor, so
    FAST-LIO needs a clean SIGINT shutdown to leave usable logs.
    """
    if proc is None or proc.poll() is not None:
        if proc in LIVE:
            LIVE.remove(proc)
        return

    signal_group(proc, first)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    else:
        print(f"{label} did not exit {grace:.0f}s after signal; escalating",
              file=sys.stderr)
        signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            signal_group(proc, signal.SIGKILL)
            proc.wait(timeout=5)

    if proc in LIVE:
        LIVE.remove(proc)


def kill_all() -> None:
    for proc in list(LIVE):
        stop(proc, grace=2.0, first=signal.SIGTERM, label="child")


# --------------------------------------------------------------------------
# FAST-LIO log handling
# --------------------------------------------------------------------------
def clear_fastlio_logs() -> None:
    """Stale logs from the previous bag would otherwise be copied as this bag's."""
    for name in FASTLIO_LOG_FILES:
        try:
            os.remove(os.path.join(FASTLIO_LOG_SRC, name))
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------
# Frame accounting
# --------------------------------------------------------------------------
def bag_cloud_count(bag: str) -> tuple[str, int]:
    """(topic, message count) for the bag's point cloud topic, from metadata.yaml.

    The topic is found by type rather than by name so this works on every dataset
    here (/velodyne_points, /livox/lidar, ...); with more than one cloud topic the
    busiest wins, which is the one FAST-LIO is being fed.
    """
    meta_path = os.path.join(bag, "metadata.yaml")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)["rosbag2_bagfile_information"]
    except (OSError, KeyError, yaml.YAMLError) as exc:
        print(f"warning: cannot read cloud count from {meta_path}: {exc}", file=sys.stderr)
        return "", 0

    clouds = [(entry["topic_metadata"]["name"], int(entry["message_count"]))
              for entry in meta.get("topics_with_message_count", [])
              if entry["topic_metadata"]["type"].endswith("PointCloud2")]
    if not clouds:
        return "", 0
    return max(clouds, key=lambda pair: pair[1])


def frame_report(job: dict, min_pct: float) -> tuple[str, bool]:
    """Compare processed scans with the bag's clouds. Returns (message, ok).

    pos_log.txt gets one row per processed scan (dump_lio_state_to_log() runs once
    per frame), so its length is what FAST-LIO actually consumed.
    """
    pos_log = os.path.join(job["fastlio_log_dir"], "pos_log.txt")
    topic, in_bag = bag_cloud_count(job["bag"])
    if not in_bag:
        return "frames: no PointCloud2 topic in metadata.yaml, not checked", True
    try:
        with open(pos_log, "r", encoding="utf-8") as f:
            processed = sum(1 for line in f if line.strip())
    except OSError:
        return f"frames: {pos_log} unreadable, not checked "\
               f"(runtime_pos_log_enable must be true)", True

    pct = 100.0 * processed / in_bag
    message = f"frames {processed}/{in_bag} ({pct:.1f}%) of {topic}"
    if pct >= min_pct:
        return message, True
    return (f"{message} -- BELOW {min_pct:.0f}%: the map is built from a fraction of "
            f"the data and every number derived from it is wrong. Check that the DDS "
            f"profile is in effect (--dds-profile) before trusting this run.", False)


def collect_fastlio_logs(dest: str) -> int:
    """Copy the logs out. Must run after the node exits, or they are still buffered."""
    if not os.path.isdir(FASTLIO_LOG_SRC):
        print(f"warning: FAST-LIO log dir not found: {FASTLIO_LOG_SRC}", file=sys.stderr)
        return 0

    os.makedirs(dest, exist_ok=True)
    copied = 0
    for name in FASTLIO_LOG_FILES:
        src = os.path.join(FASTLIO_LOG_SRC, name)
        if os.path.isfile(src) and os.path.getsize(src) > 0:
            shutil.copy2(src, os.path.join(dest, name))
            copied += 1

    if copied:
        print(f"saved {copied} FAST-LIO log file(s): {dest}")
    else:
        print(f"warning: no FAST-LIO logs to copy from {FASTLIO_LOG_SRC}\n"
              f"  (runtime_pos_log_enable must be true in the config)", file=sys.stderr)
    return copied


# --------------------------------------------------------------------------
# One bag
# --------------------------------------------------------------------------
def run_one(job: dict, args: argparse.Namespace, env: dict[str, str],
            tmp_dir: str) -> dict:
    """Play one bag through FAST-LIO. Returns a result record; never raises."""
    result = {"name": job["bag_name"], "ok": False, "note": "", "seconds": 0.0,
              "map": ""}
    started = time.monotonic()
    fastlio = record = None

    # PCL's writeBinary will not create the parent dir.
    os.makedirs(job["map_dir"], exist_ok=True)
    if os.path.exists(job["map"]):
        os.remove(job["map"])
    staged = stage_config(job["config_src"], args.config_file, job["map"], tmp_dir)
    clear_fastlio_logs()

    # Each bag logs to its own dir unless --ros-log-dir collects them all.
    bag_env = dict(env, ROS_LOG_DIR=job["ros_log_dir"])
    os.makedirs(job["ros_log_dir"], exist_ok=True)

    try:
        fastlio = spawn(["ros2", "launch", "fast_lio", "mapping.launch.py",
                         f"config_path:={os.path.dirname(staged)}",
                         f"config_file:={os.path.basename(staged)}",
                         "use_sim_time:=true", "rviz:=false"], bag_env)
        time.sleep(WAIT_AFTER_LAUNCH)
        if fastlio.poll() is not None:
            result["note"] = f"FAST-LIO launch exited (rc={fastlio.returncode}) before playback"
            return result

        if args.record_odom:
            os.makedirs(job["odom_dir"], exist_ok=True)
            print(f"recording odometry: {args.odom_topic} -> {job['odom_bag']}")
            record = spawn(["ros2", "bag", "record", args.odom_topic,
                            "/tf", "/tf_static", "-o", job["odom_bag"]], bag_env)
            time.sleep(2.0)

        play_cmd = ["ros2", "bag", "play", job["bag"], "--clock",
                    "-r", str(args.rate),
                    "--read-ahead-queue-size", str(args.queue_size)]
        if args.play_qos_overrides:
            play_cmd += ["--qos-profile-overrides-path", args.play_qos_overrides]
        play = subprocess.run(play_cmd, env=bag_env)
        if play.returncode != 0:
            result["note"] = f"ros2 bag play failed (rc={play.returncode})"
            return result

        stop(record, grace=5.0, label="odom recorder")
        record = None

        print(f"bag play complete, waiting {WAIT_AFTER_PLAY:.0f}s before saving map")
        time.sleep(WAIT_AFTER_PLAY)

        # Call /map_save directly rather than gating on `ros2 service list`:
        # `service call` already waits for the service to become available, and the
        # enumeration APIs are not reliable. On this machine `ros2 service list`
        # returns nothing at all -- not even the CLI node's own services -- while
        # `service call /map_save` succeeds and `ros2 bag record <topic>` subscribes
        # in 0.2 s, i.e. endpoint matching works and only graph enumeration is
        # broken. Gating on the enumeration turned a working call into
        # "/map_save never appeared" for every bag in the batch.
        try:
            call = subprocess.run(["ros2", "service", "call", "/map_save",
                                   "std_srvs/srv/Trigger", "{}"],
                                  env=bag_env, timeout=SERVICE_TIMEOUT,
                                  capture_output=True, text=True, check=False)
        except subprocess.TimeoutExpired:
            result["note"] = "/map_save call timed out (service never became available)"
            return result
        if "success=True" not in call.stdout:
            tail = (call.stdout + call.stderr).strip().splitlines()[-3:]
            result["note"] = f"/map_save did not report success: {' | '.join(tail)}"
            return result
        time.sleep(WAIT_AFTER_SAVE)

        if not os.path.isfile(job["map"]):
            result["note"] = f"map not found after save: {job['map']}"
            return result
        result["map"] = job["map"]
        print(f"saved map: {job['map']}")
        if args.map_dir:
            copy = os.path.join(args.map_dir, f"{job['sequence']}_fastlio_map.pcd")
            shutil.copy2(job["map"], copy)
            print(f"saved map copy: {copy}")

        if args.eval_odom:
            result["note"] = evaluate(job, args, bag_env)
        result["ok"] = True
        return result

    except Exception as exc:  # keep the batch alive; the summary reports it
        result["note"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        stop(record, grace=5.0, label="odom recorder")
        # The node must be down before its logs are readable.
        stop(fastlio, label="FAST-LIO")
        collect_fastlio_logs(job["fastlio_log_dir"])
        # Frame accounting last, and it can fail a run that otherwise succeeded:
        # a map built from half the clouds looks fine and is not.
        frames, frames_ok = frame_report(job, args.min_frame_pct)
        print(f"    {frames}")
        result["frames"] = frames
        if not frames_ok:
            result["ok"] = False
            result["note"] = f"{result['note']}; {frames}" if result["note"] else frames
        result["seconds"] = time.monotonic() - started


def evaluate(job: dict, args: argparse.Namespace, env: dict[str, str]) -> str:
    """Export the recorded odometry to TUM and score it against pose.csv."""
    pose_csv = os.path.join(job["sequence_dir"], "pose.csv")
    if not os.path.isdir(job["odom_bag"]) or not os.path.isfile(pose_csv):
        return f"eval skipped: missing odom bag or {pose_csv}"

    # These import rclpy/rosbag2_py, so they need the ROS python, not whatever
    # interpreter (venv, conda) is running this script.
    subprocess.run([args.python, os.path.join(args.eval_dir, "export_odometry_tum.py"),
                    job["odom_bag"], "-t", args.odom_topic, "-o", job["odom_tum"]],
                   env=env, check=True)
    subprocess.run([args.python, os.path.join(args.eval_dir, "evaluate_pose_csv_tum.py"),
                    pose_csv, job["odom_tum"], "-o", job["odom_eval"]],
                   env=env, check=True)
    print(f"saved odom trajectory: {job['odom_tum']}\n"
          f"saved odom evaluation: {job['odom_eval']}")
    return f"eval: {job['odom_eval']}"


# --------------------------------------------------------------------------
def build_jobs(bags: list[str], args: argparse.Namespace, config_src: str) -> list[dict]:
    jobs = []
    for bag in bags:                      # already absolute (resolved in main)
        bag_name = os.path.basename(bag.rstrip("/"))
        sequence = sequence_name(bag_name, args.glob)
        # --run-name keeps a parameter sweep out of the canonical outputs: everything
        # lands under <bag>/runs/<name>/ instead of <bag>/, so the map and the logs a
        # report was built from are not overwritten by the next experiment. The map is
        # named after the sequence there, matching what the evaluators expect to find.
        out_dir = os.path.join(bag, "runs", args.run_name) if args.run_name else bag
        map_name = f"{sequence}_fastlio_map.pcd" if args.run_name else "fastlio_map.pcd"
        odom_dir = args.odom_dir or os.path.join(out_dir, "results")
        odom_bag = os.path.join(odom_dir, f"{sequence}_fastlio_odom")
        if args.record_odom and os.path.exists(odom_bag):
            odom_bag = f"{odom_bag}_{time.strftime('%Y%m%d_%H%M%S')}"
        jobs.append({
            "bag": bag,
            "bag_name": bag_name,
            "sequence": sequence,
            "sequence_dir": os.path.join(args.root, sequence),
            "config_src": config_src,
            "map_dir": os.path.join(out_dir, "maps"),
            "map": os.path.join(out_dir, "maps", map_name),
            "fastlio_log_dir": args.fastlio_log_dir or os.path.join(out_dir, "fastlio_log"),
            "ros_log_dir": args.ros_log_dir or os.path.join(out_dir, "ros_logs"),
            "odom_dir": odom_dir,
            "odom_bag": odom_bag,
            "odom_tum": f"{odom_bag}.tum",
            "odom_eval": f"{odom_bag}_vs_pose_eval.txt",
        })
    return jobs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="*", help="bag directories; skips auto-discovery")
    ap.add_argument("--dataset", default="conpr",
                    help="which datasets.<name>.root in config/paths.yaml --root "
                         "defaults to. Default: conpr")
    ap.add_argument("--root", default=None,
                    help="dataset root: the directory holding the <sequence>/ folders "
                         "(that is where pose.csv is looked up for --eval-odom). "
                         "Default: datasets.<--dataset>.root from config/paths.yaml")
    ap.add_argument("--bag-root", default=None,
                    help="directory holding the bags. Default: <root>/Converted_bag "
                         "when it exists (where batch_to_bag.py writes), else <root>")
    ap.add_argument("--glob", default="*_pc2_bag",
                    help="auto-discovery pattern, matched against directory names "
                         "(needs the wildcard). Default: *_pc2_bag, what "
                         "config/conpr/bag_conpr_pc2.yaml produces")
    ap.add_argument("--start-from", help="skip every bag before this one "
                                         "(bag or sequence name)")
    ap.add_argument("--only", nargs="+", help="run just these bags/sequences")
    ap.add_argument("--config-file", default="conpr/fast_lio_conpr.yaml",
                    help="FAST-LIO config, relative to --config-path")
    ap.add_argument("--config-path", default=os.path.join(REPO, "config"),
                    help="directory holding the FAST-LIO config")
    ap.add_argument("--run-name", default=None,
                    help="write this run's map, logs and odometry under "
                         "<bag>/runs/<name>/ instead of <bag>/, so a parameter sweep "
                         "cannot overwrite the canonical outputs a report was built from")
    ap.add_argument("-r", "--rate", default="1.0", help="ros2 bag play rate")
    ap.add_argument("--queue-size", type=int, default=10000,
                    help="ros2 bag play --read-ahead-queue-size")
    ap.add_argument("--no-rviz", dest="rviz", action="store_false", help="do not open RViz")
    ap.add_argument("--rviz-config", default=None,
                    help="Default: fastlio.rviz_config from config/paths.yaml")
    ap.add_argument("--record-odom", action="store_true",
                    help="record the odometry topic while playing each bag")
    ap.add_argument("--eval-odom", action="store_true",
                    help="also export the odometry to TUM and score it against "
                         "<sequence>/pose.csv (implies --record-odom)")
    ap.add_argument("--odom-topic", default="/Odometry")
    ap.add_argument("--odom-dir", help="one directory for every bag's odometry "
                                      "(default: <bag>/results)")
    ap.add_argument("--ros-log-dir", help="one ROS log directory for every bag "
                                          "(default: <bag>/ros_logs)")
    ap.add_argument("--fastlio-log-dir", help="one FAST-LIO log directory for every bag "
                                              "(default: <bag>/fastlio_log). Point this at a "
                                              "per-run directory to keep pos_log.txt from one "
                                              "parameter sweep from overwriting the next; note "
                                              "the filenames inside are fixed, so a MULTI-bag "
                                              "batch would clobber itself here")
    ap.add_argument("--map-dir", help="extra directory to copy each map into as "
                                     "<sequence>_fastlio_map.pcd")
    ap.add_argument("--ros-setup", default=None,
                    help="ROS setup.bash. Default: ros.setup from config/paths.yaml")
    ap.add_argument("--overlay", action="append", default=None,
                    help="workspace setup.bash to source after ROS; repeatable. "
                         "Default: ros.overlays from config/paths.yaml")
    ap.add_argument("--dds-profile", default=None, metavar="XML|none",
                    help="Fast DDS profile exported as FASTRTPS_DEFAULT_PROFILES_FILE "
                         "for the node and the player alike. Default: ros.dds_profile "
                         "from config/paths.yaml, which enlarges the shared-memory "
                         "segment so the writer stops dropping big clouds. `none` "
                         "runs with the stock transport -- expect ~50%% frame loss")
    ap.add_argument("--play-qos-overrides", default=None, metavar="YAML|auto",
                    help="ros2 bag play --qos-profile-overrides-path. `auto` uses "
                         "ros.play_qos_overrides from config/paths.yaml (keep_all on the "
                         "cloud topic, so the player blocks instead of overwriting). "
                         "Off by default: redundant once --dds-profile is in effect")
    ap.add_argument("--min-frame-pct", type=float, default=95.0,
                    help="fail a bag whose processed-scan count falls below this "
                         "percentage of the clouds in the bag. 0 disables the check")
    ap.add_argument("--fastlio-log-src", default=None,
                    help="FAST-LIO's compiled-in Log directory, copied out per bag. "
                         "Default: fastlio.log_dir from config/paths.yaml")
    ap.add_argument("--eval-dir", default=HERE,
                    help="directory holding export_odometry_tum.py / "
                         "evaluate_pose_csv_tum.py")
    ap.add_argument("--python", default=None,
                    help="interpreter for the eval scripts (needs rclpy/rosbag2_py). "
                         "Default: ros.python from config/paths.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the resolved output paths only")
    args = ap.parse_args(argv)
    if args.eval_odom:
        args.record_odom = True

    # Machine paths come from config/paths.yaml, so nothing here is absolute. Resolved
    # after parsing rather than as argparse defaults: a broken config should not stop
    # --help from working, and the error should name the key that is wrong.
    global FASTLIO_LOG_SRC
    if args.root is None:
        args.root = paths.dataset(args.dataset)["root"]
    args.ros_setup = args.ros_setup or paths.get("ros.setup")
    args.python = args.python or paths.get("ros.python")
    args.rviz_config = args.rviz_config or paths.get("fastlio.rviz_config")
    FASTLIO_LOG_SRC = args.fastlio_log_src or paths.get("fastlio.log_dir")
    if args.play_qos_overrides == "auto":
        args.play_qos_overrides = paths.get("ros.play_qos_overrides")

    args.root = os.path.abspath(args.root)
    if not os.path.isdir(args.root):
        raise SystemExit(f"--root is not a directory: {args.root}")
    config_src = os.path.join(args.config_path, args.config_file)
    if not os.path.isfile(config_src):
        raise SystemExit(f"config file not found: {config_src}\n"
                         f"  --config-file is relative to --config-path ({args.config_path})")

    roots = bag_roots_for(args.root, args.bag_root)
    if args.bags:
        bags, bag_root = [resolve_bag_path(b, roots) for b in args.bags], roots[0]
    else:
        bags, bag_root = find_bags(roots, args.glob)
    if not bags:
        raise SystemExit(f"no bags found with glob {args.glob!r} under "
                         f"{' or '.join(roots)}\n"
                         f"  point --bag-root at the directory holding the bags "
                         f"(--root must stay at the dataset root, where "
                         f"<sequence>/pose.csv lives)")
    jobs = build_jobs(select(bags, args.start_from, args.only, args.glob),
                      args, config_src)

    print(f"root     {args.root}\nbags in  {bag_root}\n"
          f"config   {config_src}\nbags     {len(jobs)}")
    for job in jobs:
        print(f"  {job['bag_name']}  (sequence {job['sequence']})")
        if not os.path.isdir(job["bag"]):
            print(f"    ! not a directory: {job['bag']}", file=sys.stderr)
    if args.dry_run:
        job = jobs[0]
        print(f"\nfirst bag would write:\n  map    {job['map']}\n"
              f"  logs   {job['fastlio_log_dir']}\n  ros    {job['ros_log_dir']}"
              + (f"\n  odom   {job['odom_bag']}" if args.record_odom else ""))
        return

    # Overlays: ws_livox is only needed to replay a pre-existing
    # livox_ros_driver2/CustomMsg bag (`*_custom_bag`, lidar_type: 1). The
    # converters no longer write that type, and no PointCloud2 config touches it,
    # so a missing overlay is a warning rather than a hard stop.
    overlays = (args.overlay if args.overlay is not None
                else paths.get_list("ros.overlays"))
    setups = [args.ros_setup]
    for path in overlays:
        if os.path.isfile(path):
            setups.append(path)
        else:
            print(f"warning: skipping missing setup file: {path}", file=sys.stderr)
    if not os.path.isfile(args.ros_setup):
        raise SystemExit(f"missing setup file: {args.ros_setup}")
    env = ros_env(setups)
    env = with_dds_profile(env, args.dds_profile)

    if args.map_dir:
        os.makedirs(args.map_dir, exist_ok=True)
    # RViz spans the whole run, so it logs to the root rather than to one bag.
    rviz_log_dir = args.ros_log_dir or os.path.join(args.root, "ros_logs")
    os.makedirs(rviz_log_dir, exist_ok=True)

    results: list[dict] = []
    rviz = None
    started = time.monotonic()
    try:
        if args.rviz:
            rviz = spawn(["rviz2", "-d", args.rviz_config,
                          "--ros-args", "-p", "use_sim_time:=true"],
                         dict(env, ROS_LOG_DIR=rviz_log_dir))
            time.sleep(3.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, job in enumerate(jobs, 1):
                if not os.path.isdir(job["bag"]):
                    results.append({"name": job["bag_name"], "ok": False,
                                    "note": "missing bag directory", "seconds": 0.0,
                                    "map": ""})
                    continue
                print(f"\n=== [{i}/{len(jobs)}] FAST-LIO map: {job['bag_name']} ===")
                result = run_one(job, args, env, tmp_dir)
                results.append(result)
                print(f"    {'ok' if result['ok'] else 'FAILED'} "
                      f"({result['seconds']:.0f}s)"
                      + (f" -- {result['note']}" if result["note"] else ""))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        kill_all()
        raise SystemExit(130) from None
    finally:
        stop(rviz, grace=5.0, first=signal.SIGTERM, label="rviz")
        kill_all()

    good = [r for r in results if r["ok"]]
    print(f"\n{'=' * 72}\nmapped {len(good)}/{len(results)} bags in "
          f"{(time.monotonic() - started) / 60:.1f} min\n")
    for r in results:
        print(f"{r['name']}  {'ok' if r['ok'] else 'FAILED'}  ({r['seconds']:.0f}s)"
              + (f"  {r['frames']}" if r.get("frames") else ""))
        if r["note"]:
            print(f"    ! {r['note']}")
    print("\nper bag: maps/, fastlio_log/, ros_logs/, results/")
    if args.map_dir:
        print(f"map copies in: {args.map_dir}")
    if len(good) != len(results):
        raise SystemExit(1)


def on_sigterm(signum, frame):
    """Route SIGTERM through the same cleanup path as Ctrl-C."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, on_sigterm)
    main()
