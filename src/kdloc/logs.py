#!/usr/bin/env python3
"""Readers for what one KdLocalizer run leaves in runs/<name>/out/.

    logs.py summary <run dir>      the short report run_one.sh prints at the end of a run

    pose.csv            read_pose_csv          src/kdloc/monitor.py (or src/hilti/monitor_pose.py)
    reloc.log           parse_relocalizer_log  kd_relocalizer_node output
    node.log            parse_node_log         localizer.launch.py output
    proc.csv            parse_procmon          src/kdloc/procmon.py
    tegra.log           parse_tegrastats       tegrastats --logfile
    bag_metadata.yaml   bag_info               copy of the bag's metadata.yaml
    play.log            parse_play_log         ros2 bag play output

Standard library only (bag_info also needs PyYAML), so the same code runs inside the
localizer container and on the evaluation machine. Times called "wall" are the epoch seconds
the ROS log lines carry; "stamp" times are bag time, the clock of the localizer's poses.
"""
import csv
import datetime
import math
import re
import sys
from pathlib import Path

STATUSES = ("UNINIT", "TRACKING", "LOST")


def percentile(values, q):
    """The q-th percentile with linear interpolation (numpy's default method)."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def mean(values):
    return sum(values) / len(values) if values else None


def read_text(path):
    path = Path(path)
    return path.read_text(errors="replace") if path.exists() else ""


# ------------------------------------------------------------------------------------------
# pose.csv
# ------------------------------------------------------------------------------------------
def read_pose_csv(path):
    """Rows of the monitor CSV as dicts of strings (empty list when the file is missing)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def tracking_poses(rows):
    """TRACKING rows with a pose, one per pose stamp, sorted by stamp.

    Each item: (stamp, (x, y, z), (qx, qy, qz, qw), odometry_mode). A pose is repeated on every
    status message until the next one arrives, so only its first row is kept.
    """
    seen, out = set(), []
    for r in rows:
        if r.get("status") != "TRACKING" or not r.get("pose_stamp_s"):
            continue
        t = float(r["pose_stamp_s"])
        if t in seen:
            continue
        seen.add(t)
        out.append((t, (float(r["x"]), float(r["y"]), float(r["z"])),
                    (float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])),
                    r.get("odometry_mode") == "1"))
    out.sort(key=lambda item: item[0])
    return out


def status_summary(rows, bag_start, bag_end, end_tolerance=1.0):
    """What the localizer reported over the bag, from the monitor rows.

    tracked_to_end: the last TRACKING pose is within `end_tolerance` s of the bag end and the
    last status inside the bag is TRACKING. success: tracked to the end with no LOST on the way.
    """
    span = bag_end - bag_start
    inside = [r for r in rows if bag_start <= float(r["stamp_s"]) <= bag_end]
    shares = {s: (sum(r["status"] == s for r in inside) / len(inside) if inside else 0.0) for s in STATUSES}

    lost_events, previous = [], None
    for r in rows:
        if r["status"] == "LOST" and previous != "LOST":
            t = float(r["stamp_s"])
            lost_events.append({"at_s": round(t - bag_start, 2), "at_share": round((t - bag_start) / span, 4),
                                "reason": r.get("lost_reason", "")})
        previous = r["status"]

    poses = tracking_poses(rows)
    out = {"bag_seconds": round(span, 2), "status_share": shares, "lost_events": lost_events,
           "tracking_poses": len(poses), "first_tracking_s": None, "last_tracking_s": None,
           "last_tracking_share": None, "odometry_mode_share": None, "tracked_to_end": False,
           "success": False}
    if poses:
        first, last = poses[0][0], poses[-1][0]
        last_status = inside[-1]["status"] if inside else None
        out.update(first_tracking_s=round(first - bag_start, 2), last_tracking_s=round(last - bag_start, 2),
                   last_tracking_share=round(min(1.0, (last - bag_start) / span), 4),
                   odometry_mode_share=sum(p[3] for p in poses) / len(poses),
                   tracked_to_end=bool(last >= bag_end - end_tolerance and last_status == "TRACKING"))
        out["success"] = out["tracked_to_end"] and not lost_events
    return out


# ------------------------------------------------------------------------------------------
# reloc.log
# ------------------------------------------------------------------------------------------
WALL = re.compile(r"\[(\d{9,}\.\d+)\]")
ATTEMPT = re.compile(r"(?:Received (\w+) localization status, )?starting (?:BBS )?relocalization")
CLOUD_STAMP = re.compile(r"cloud stamp: ([\d.]+)")
DOWNSAMPLE = re.compile(r"Voxel downsample \(([\d.]+) m\): (\d+) -> (\d+) points")
SUCCEEDED = re.compile(r"succeeded: score=(\d+) \(([\d.]+)%\), time=([\d.]+) ms")
HINT = re.compile(r"Published 3D pose hint x: ([-\d.]+), y: ([-\d.]+), z: ([-\d.]+), yaw: ([-\d.]+)")


def parse_relocalizer_log(path):
    """Every relocalization attempt in log order, plus how the relocalizer started.

    An attempt starts at "starting relocalization" and ends with one outcome:
    succeeded (score, search ms, published pose hint), below_threshold, timed_out, no_points,
    failed (any other failure line), or no_answer when the next attempt started first.
    search_ms of a failed attempt is the wall time from its start to its outcome line.
    """
    text = read_text(path)
    attempts, current = [], None
    info = {"map_loaded": "Loaded BBS map" in text, "errors": [], "no_lidar_frame": 0, "ignored_status": 0,
            "start_wall": None}

    def close(outcome, wall, **extra):
        current.update(outcome=outcome, end_wall=wall, **extra)
        if "search_ms" not in current and wall is not None:
            current["search_ms"] = round(1000 * (wall - current["start_wall"]), 1)

    for line in text.splitlines():
        m = WALL.search(line)
        wall = float(m.group(1)) if m else None
        if info["start_wall"] is None and wall is not None:
            info["start_wall"] = wall
        if "Failed to load" in line or "what()" in line:
            info["errors"].append(line.strip()[:200])
        if "Cannot relocalize: no lidar frame available" in line:
            info["no_lidar_frame"] += 1
            continue
        if "Ignoring localization status" in line:
            info["ignored_status"] += 1
            continue
        m = ATTEMPT.search(line)
        if m:
            cloud = CLOUD_STAMP.search(line)
            current = {"start_wall": wall, "trigger": m.group(1) or "", "outcome": "no_answer",
                       "cloud_stamp": float(cloud.group(1)) if cloud else None}
            attempts.append(current)
            continue
        if current is None:
            continue
        m = DOWNSAMPLE.search(line)
        if m:
            current["points"] = int(m.group(3))
            continue
        m = SUCCEEDED.search(line)
        if m:
            close("succeeded", wall, score=int(m.group(1)), score_pct=float(m.group(2)),
                  search_ms=float(m.group(3)))
            continue
        m = HINT.search(line)
        if m and current["outcome"] == "succeeded":
            current["hint"] = [float(v) for v in m.groups()]
            continue
        if current["outcome"] != "no_answer":
            continue
        if "score below threshold" in line:
            close("below_threshold", wall)
        elif "timed out" in line:
            close("timed_out", wall)
        elif "No lidar points for BBS" in line:
            close("no_points", wall)
        elif "relocalization failed" in line.lower():
            close("failed", wall)
    return attempts, info


def relocalizer_counts(attempts):
    outcomes = ("succeeded", "below_threshold", "timed_out", "no_points", "failed", "no_answer")
    return {o: sum(a["outcome"] == o for a in attempts) for o in outcomes}


# ------------------------------------------------------------------------------------------
# node.log, play.log
# ------------------------------------------------------------------------------------------
PROCESS_TIMING = re.compile(r"Process timing \(ms\) over (\d+) updates: process avg=([\d.]+) max=([\d.]+)")
SLOW_TICK = re.compile(r"Timings \(ms\): Total: ([\d.]+), Update: ([\d.]+), Align: ([\d.]+), "
                       r"MapUpdate: ([\d.]+)")


def parse_node_log(path):
    """Per-update processing time from the localizer's own statistics.

    The localizer logs "Process timing (ms) over N updates: process avg=A max=M" every
    process_stats_log_interval_sec and a "Timings (ms): Total ..." warning for each update above
    process_total_warn_threshold_ms (100 ms by default).
    """
    text = read_text(path)
    out = {"updates": 0, "udp_fallback": "Failed to create segment" in text,
           "initial_pose_loaded": "Loaded initial pose from" in text}
    windows = [(int(n), float(a), float(m)) for n, a, m in PROCESS_TIMING.findall(text)]
    if windows:
        n = sum(w[0] for w in windows)
        out.update(updates=n, process_avg_ms=round(sum(w[0] * w[1] for w in windows) / n, 2),
                   process_window_avg_max_ms=max(w[1] for w in windows),
                   process_max_ms=max(w[2] for w in windows))
    slow = [[float(v) for v in m] for m in SLOW_TICK.findall(text)]
    out["slow_updates"] = len(slow)
    if slow:
        out["slow_update_mean_ms"] = {key: round(mean([s[i] for s in slow]), 2)
                                      for i, key in enumerate(("total", "update", "align", "map_update"))}
    return out


PLAY_ERRORS = re.compile(r"SQLite error|malformed|Exception", re.I)


def parse_play_log(path):
    text = read_text(path)
    errors = [line.strip()[:200] for line in text.splitlines() if PLAY_ERRORS.search(line)]
    return {"errors": errors[:5]}


# ------------------------------------------------------------------------------------------
# proc.csv, tegra.log
# ------------------------------------------------------------------------------------------
def parse_procmon(path):
    """Per node: CPU cores mean / 95th percentile / max and peak resident memory (MB)."""
    rows = read_pose_csv(path)
    out = {}
    for node in ("kd_localizer", "kd_relocalizer"):
        cpu = [float(r[f"{node}_cpu_cores"]) for r in rows if r.get(f"{node}_cpu_cores")]
        rss = [float(r[f"{node}_rss_mb"]) for r in rows if r.get(f"{node}_rss_mb")]
        if cpu:
            out[node] = {"cpu_cores_mean": round(mean(cpu), 3),
                         "cpu_cores_p95": round(percentile(cpu, 95), 3),
                         "cpu_cores_max": round(max(cpu), 3),
                         "rss_mb_max": round(max(rss), 1) if rss else None}
    return out


TEGRA = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) RAM (\d+)/(\d+)MB")
TEGRA_CPU = re.compile(r"CPU \[([^\]]*)\]")
TEGRA_GPU = re.compile(r"GR3D_FREQ (\d+)%")
TEGRA_GPU_POWER = re.compile(r"VDD_GPU_SOC (\d+)mW")


def parse_tegrastats(path, utc_offset=None):
    """tegrastats samples: (epoch s or None, RAM used MB, mean CPU %, GPU %, GPU+SoC power mW).

    tegrastats writes the host's local time without a zone; `utc_offset` (e.g. "+0100", written by
    campaign.sh into out/host_utc_offset) turns it into epoch seconds. Fields a Jetson model does
    not report are None.
    """
    tz = None
    if utc_offset:
        m = re.fullmatch(r"([+-])(\d\d):?(\d\d)", utc_offset.strip())
        if m:
            minutes = (int(m.group(2)) * 60 + int(m.group(3))) * (1 if m.group(1) == "+" else -1)
            tz = datetime.timezone(datetime.timedelta(minutes=minutes))
    samples = []
    for line in read_text(path).splitlines():
        m = TEGRA.match(line)
        if not m:
            continue
        stamp = None
        if tz is not None:
            stamp = datetime.datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S").replace(tzinfo=tz).timestamp()
        cpu = TEGRA_CPU.search(line)
        cores = [int(c.split("%")[0]) for c in cpu.group(1).split(",") if "%" in c] if cpu else []
        gpu, power = TEGRA_GPU.search(line), TEGRA_GPU_POWER.search(line)
        samples.append((stamp, int(m.group(2)), mean(cores), int(gpu.group(1)) if gpu else None,
                        int(power.group(1)) if power else None))
    return samples


def tegrastats_summary(samples):
    if not samples:
        return {}

    def column(i):
        return [s[i] for s in samples if s[i] is not None]

    ram, cpu, gpu, power = column(1), column(2), column(3), column(4)
    return {"samples": len(samples), "ram_used_mb_max": max(ram),
            "cpu_pct_mean": round(mean(cpu), 1) if cpu else None,
            "gpu_pct_mean": round(mean(gpu), 1) if gpu else None, "gpu_pct_max": max(gpu) if gpu else None,
            "gpu_power_mw_mean": round(mean(power)) if power else None,
            "gpu_power_mw_max": max(power) if power else None}


# ------------------------------------------------------------------------------------------
# bag metadata, run.env
# ------------------------------------------------------------------------------------------
def bag_info(path, limit_s=0.0):
    """Start, end (bag time, epoch s) and per-topic message counts from a rosbag2 metadata.yaml.

    With `limit_s` > 0 (a run that played only the first limit_s seconds) the end and the
    counts are cut down to the part that was played.
    """
    import yaml

    with open(path) as f:
        info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    start = info["starting_time"]["nanoseconds_since_epoch"] / 1e9
    duration = info["duration"]["nanoseconds"] / 1e9
    played = min(duration, limit_s) if limit_s and limit_s > 0 else duration
    counts = {t["topic_metadata"]["name"]: t["message_count"] for t in info["topics_with_message_count"]}
    if played < duration:
        counts = {k: int(round(v * played / duration)) for k, v in counts.items()}
    return {"start": start, "end": start + played, "duration": duration, "played": played,
            "message_counts": counts}


def read_env(path):
    """KEY=value lines of run.env (values written with shlex.quote)."""
    import shlex

    out = {}
    for line in read_text(path).splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            out[key.strip()] = (shlex.split(value) or [""])[0]
    return out


# ------------------------------------------------------------------------------------------
# summary <run dir>
# ------------------------------------------------------------------------------------------
def summary_lines(run_dir):
    """Human-readable one-run summary; lines starting with [INVALID] make the run not count."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "out"
    env = read_env(run_dir / "run.env")
    lines = []
    rows = read_pose_csv(out_dir / "pose.csv")
    if not rows:
        return ["[pose] NO POSE LOG"]
    limit = float(read_text(out_dir / "limit_s").strip() or env.get("LIMIT", 0) or 0)
    if (out_dir / "bag_metadata.yaml").exists():
        bag = bag_info(out_dir / "bag_metadata.yaml", limit)
        s = status_summary(rows, bag["start"], bag["end"])
        shares = ", ".join(f"{k} {100 * v:.0f} %" for k, v in s["status_share"].items())
        text = f"[pose] {shares}"
        if s["tracking_poses"]:
            text += (f"; tracking from {s['first_tracking_s']} s to {s['last_tracking_s']} s of "
                     f"{s['bag_seconds']} s, odometry mode {100 * s['odometry_mode_share']:.1f} %")
        if s["lost_events"]:
            first = s["lost_events"][0]
            text += f"; first LOST at {first['at_s']} s: {first['reason']}"
        lines.append(text)
    attempts, info = parse_relocalizer_log(out_dir / "reloc.log")
    if (out_dir / "reloc.log").exists():
        counts = relocalizer_counts(attempts)
        searches = [f"{a['search_ms']:.0f}" for a in attempts if a["outcome"] == "succeeded"]
        outcomes = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        lines.append(f"[reloc] {len(attempts)} attempts" + (f": {outcomes}" if outcomes else "")
                     + (f"; search ms of results: {' '.join(searches[:10])}" if searches else "")
                     + ("" if info["map_loaded"] else "; BBS MAP NOT LOADED"))
        lines += [f"[reloc] {e}" for e in info["errors"][:3]]
    node = parse_node_log(out_dir / "node.log")
    if node["updates"]:
        lines.append(f"[timing] {node['updates']} updates, process avg {node['process_avg_ms']:.1f} ms, "
                     f"worst window max {node['process_max_ms']:.1f} ms, "
                     f"updates over 100 ms: {node['slow_updates']}")
    else:
        lines.append("[timing] no 'Process timing' lines in node.log")
    if node["udp_fallback"]:
        lines.append("[INVALID] Fast DDS could not create its shared-memory segment and fell back to UDP; "
                     "the frame rate of this run is not comparable (is /dev/shm full?)")
    play = parse_play_log(out_dir / "play.log")
    if play["errors"]:
        lines.append(f"[INVALID] bag play error: {play['errors'][0]}")
    return lines


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "summary":
        sys.exit("usage: logs.py summary <run dir>")
    print("\n".join(summary_lines(argv[1])))


if __name__ == "__main__":
    main()
