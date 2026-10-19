#!/usr/bin/env python3
"""
Batch-convert every sequence folder under a dataset root with ``to_bag.py``.

Written for ConPR's layout, where the per-sequence inputs sit side by side and
the calibration is shared by all of them::

    <root>/20230531/LiDAR/<unix_ns>.bin     per-frame Livox records
    <root>/20230531/imu.csv                 t, qx..qw, wx..wz, ax..az
    <root>/20230531/pose.csv                t, x, y, z
    <root>/20230531/data_stamp.csv          acquisition log (not used)
    <root>/Calibration/calibration_lidar.yaml
    <root>/Converted_bag/20230531_<suffix>  output

Three things this does beyond a shell for-loop:

  * **Checks every sequence before converting any of them.** A 12-sequence run
    takes ~25 minutes; discovering a missing imu.csv at minute 20 is a waste.
  * **Injects the shared calibration.** ``calibration_lidar.yaml`` carries both
    the LiDAR<->IMU extrinsic (``T_lidar_imu0``) and the clock offset
    (``timeshift_to_imu0``), so they are read from the dataset instead of being
    duplicated into every bag config -- and used at full precision rather than
    the 9-decimal rounding the configs carry.
  * **Logs CPU / memory / disk usage** for the whole run to
    ``<out-dir>/logs/resources_<timestamp>.csv`` plus a line chart beside it,
    banded per sequence, so "which sequence pinned the disk" is answerable
    afterwards. ``--no-resource-log`` turns it off; see ``resource_log.py``.

Examples::

    # every sequence -> ConPR/Converted_bag/<seq>_pc2_bag
    python3 src/batch_to_bag.py --root /media/kudan/seagate-2tb/datasets/ConPR \\
        -c config/conpr/bag_conpr_pc2.yaml

    # 20 s of each sequence first, to check the config end to end
    python3 src/batch_to_bag.py --root ... -c ... --limit-sec 20 --suffix _probe_bag

    # one sequence, forced overwrite
    python3 src/batch_to_bag.py --root ... -c ... --only 20230706 --force
"""
import argparse
import fnmatch
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import yaml

import resource_log
from ros2_bag_io import add_distro_arg, check_window, load_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TO_BAG = os.path.join(HERE, "to_bag.py")

CLOUD_TYPES = ("pointcloud2",)      # keep in sync with to_bag.CLOUD_TYPES


def default_suffix(config_path: str) -> str:
    """`bag_conpr_pc2.yaml` -> `_pc2_bag`, `bag_conpr.yaml` -> `_bag`.

    Matches the naming the existing bags and run_fastlio_bags.sh's `*_custom_bag`
    glob already use. Override with --suffix when the guess is wrong.
    """
    parts = os.path.splitext(os.path.basename(config_path))[0].split("_")
    return f"_{parts[-1]}_bag" if len(parts) > 2 else "_bag"


# --------------------------------------------------------------------------
# Calibration (Curretnly only for Lidar and IMU)
# --------------------------------------------------------------------------
def read_calibration(path: str) -> dict[str, Any]:
    """Extract the LiDAR->IMU extrinsic and clock offset from calibration_lidar.yaml.

    ConPR ships ``T_lidar_imu0`` (the IMU pose in the LiDAR frame). tf_static and
    FAST-LIO both want the opposite direction, so it is inverted here -- the same
    ``inverse(T_lidar_imu0)`` the bag configs carry hardcoded.
    """
    cal = load_yaml(path)
    key = next((k for k in cal if k.startswith("lidar")), None)
    if key is None:
        raise SystemExit(f"{path}: no `lidar*` section; found {sorted(cal)}")
    node = cal[key]
    if "T_lidar_imu0" not in node:
        raise SystemExit(f"{path}: {key} has no `T_lidar_imu0`; found {sorted(node)}")
    T_lidar_imu = np.array(node["T_lidar_imu0"], dtype=float)
    if T_lidar_imu.shape != (4, 4):
        raise SystemExit(f"{path}: {key}.T_lidar_imu0 must be 4x4, got {T_lidar_imu.shape}")
    if not np.isfinite(T_lidar_imu).all():
        raise SystemExit(f"{path}: {key}.T_lidar_imu0 contains NaN/Inf")
    try:
        T_imu_lidar = np.linalg.inv(T_lidar_imu)
    except np.linalg.LinAlgError as exc:
        raise SystemExit(f"{path}: {key}.T_lidar_imu0 is singular and cannot be "
                         f"inverted: {exc}") from None
    shift = float(node.get("timeshift_to_imu0", 0.0))
    if not np.isfinite(shift):
        raise SystemExit(f"{path}: {key}.timeshift_to_imu0 is not finite ({shift})")
    return {"matrix": T_imu_lidar.tolist(), "stamp_shift_sec": shift}


def calibration_target(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The one tf_static entry the LiDAR<->IMU extrinsic belongs to, or None.

    Matching on the source `frame_id`s rather than on "has a `matrix` key": a
    config with a second matrix-valued transform (a camera, a base_link offset)
    would otherwise get that one overwritten with the LiDAR extrinsic too.
    """
    sources = cfg.get("sources", []) or []
    lidar_frames = {s.get("frame_id") for s in sources if s.get("type") in CLOUD_TYPES}
    imu_frames = {s.get("frame_id") for s in sources if s.get("type") == "imu"}
    lidar_frames.discard(None)
    imu_frames.discard(None)

    transforms = cfg.get("tf_static", []) or []
    matched = [tf for tf in transforms
               if {tf.get("parent"), tf.get("child")} & imu_frames
               and {tf.get("parent"), tf.get("child")} & lidar_frames]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise SystemExit(
            f"{len(matched)} tf_static entries connect the IMU frame(s) {sorted(imu_frames)} to "
            f"the LiDAR frame(s) {sorted(lidar_frames)}, so the calibration cannot be injected "
            f"unambiguously -- keep one, or pass --no-calibration and set them in the config")
    # No frame match: fall back only when there is exactly one matrix-valued entry,
    # which is the shape every current config has.
    with_matrix = [tf for tf in transforms if "matrix" in tf]
    if len(with_matrix) == 1:
        return with_matrix[0]
    if len(with_matrix) > 1:
        raise SystemExit(
            f"config has {len(with_matrix)} tf_static entries with a `matrix` and none of them "
            f"connects {sorted(imu_frames)} to {sorted(lidar_frames)}; name the frames to match "
            f"the source `frame_id`s, or pass --no-calibration")
    return None


def apply_calibration(cfg: dict[str, Any], cal: dict[str, Any]) -> list[str]:
    """Overwrite the config's extrinsic / clock offset in place; report changes."""
    notes = []
    for src in cfg.get("sources", []):
        if src.get("type") in CLOUD_TYPES:
            was = src.get("stamp_shift_sec")
            if was != cal["stamp_shift_sec"]:
                src["stamp_shift_sec"] = cal["stamp_shift_sec"]
                notes.append(f"{src.get('out')}: stamp_shift_sec {was} -> {cal['stamp_shift_sec']}")
    tf = calibration_target(cfg)
    if tf is not None:
        # read_calibration returns T_imu_lidar, i.e. parent_to_child only when the
        # entry is written imu -> lidar. If it is written the other way round, flag
        # it for inversion rather than silently installing a transform that points
        # backwards. Frames we cannot identify keep the historical parent_to_child.
        sources = cfg.get("sources", []) or []
        lidar_frames = {s.get("frame_id") for s in sources if s.get("type") in CLOUD_TYPES}
        imu_frames = {s.get("frame_id") for s in sources if s.get("type") == "imu"}
        reversed_ = tf.get("parent") in lidar_frames and tf.get("child") in imu_frames
        tf["matrix"] = cal["matrix"]
        tf["matrix_is"] = "child_to_parent" if reversed_ else "parent_to_child"
        notes.append(f"tf_static {tf.get('parent')}->{tf.get('child')}: matrix from calibration "
                     f"(exact inverse, unrounded, {tf['matrix_is']})")
    return notes


# --------------------------------------------------------------------------
# Discovery / validation
# --------------------------------------------------------------------------
def required_inputs(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(directories, files) each sequence must contain, read off the config."""
    dirs, files = [], []
    for src in cfg.get("sources", []):
        if not src.get("enabled", True):
            continue
        if src.get("type") in CLOUD_TYPES:
            dirs.append(src["dir"])
            if src.get("stamp_from") == "file":
                files.append(src["stamp_file"])
        else:
            files.append(src["path"])
    return dirs, files


def check_suffix(suffix: str) -> str:
    """Reject a --suffix that would move the output out of the output root.

    The suffix is concatenated onto a sequence name to form a path that --force
    later replaces, so `--suffix ../../scratch` must not be able to aim that
    replacement somewhere else on disk.
    """
    seps = [s for s in (os.sep, os.altsep) if s]
    if not suffix or suffix in (".", "..") or any(s in suffix for s in seps):
        raise SystemExit(f"--suffix must be a plain name fragment without a path separator, "
                         f"got {suffix!r}")
    return suffix


def resolve_output(out_root: str, name: str, suffix: str) -> str:
    """``<out_root>/<name><suffix>``, verified to be a direct child of out_root."""
    out_bag = os.path.realpath(os.path.join(out_root, f"{name}{suffix}"))
    if os.path.dirname(out_bag) != os.path.realpath(out_root):
        raise SystemExit(f"output path {out_bag} escapes the output root {out_root} "
                         f"(sequence {name!r}, suffix {suffix!r})")
    return out_bag


def find_sequences(root: str, pattern: str, only: list[str] | None) -> list[str]:
    names = sorted(n for n in os.listdir(root)
                   if os.path.isdir(os.path.join(root, n)) and fnmatch.fnmatch(n, pattern))
    if only:
        missing = [o for o in only if o not in names]
        if missing:
            raise SystemExit(f"--only {missing} not found under {root}; "
                             f"{pattern!r} matches {names}")
        names = [n for n in names if n in only]
    return names


def check_sequence(seq_dir: str, dirs: list[str], files: list[str]) -> list[str]:
    problems = []
    for d in dirs:
        p = os.path.join(seq_dir, d)
        if not os.path.isdir(p):
            problems.append(f"missing dir {d}/")
        elif not os.listdir(p):
            problems.append(f"empty dir {d}/")
    for f in files:
        if not os.path.isfile(os.path.join(seq_dir, f)):
            problems.append(f"missing file {f}")
    return problems


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def convert_one(job: dict[str, Any]) -> dict[str, Any]:
    """Run to_bag.py in a subprocess.

    A subprocess rather than an in-process call on purpose: it isolates one
    sequence's crash (or MemoryError on a huge frame) from the rest of the batch,
    and it is what lets --jobs run conversions concurrently. `--force` is handed
    down rather than acted on here, so the old bag is only replaced once to_bag has
    written a complete new one.
    """
    started = time.time()
    cmd = [sys.executable, TO_BAG, job["seq_dir"], "-o", job["out_bag"], "-c", job["config"]]
    for flag in ("start_sec", "limit_sec"):
        if job.get(flag) is not None:
            cmd += [f"--{flag.replace('_', '-')}", str(job[flag])]
    if job.get("ros_distro"):
        cmd += ["--ros-distro", job["ros_distro"]]
    if job.get("force"):
        cmd.append("--force")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
    return {
        "name": job["name"],
        "ok": proc.returncode == 0,
        "seconds": time.time() - started,
        "done": next((ln for ln in lines if ln.startswith("done:")), ""),
        "warnings": [ln for ln in lines if ln.startswith("WARNING")],
        "tally": [ln.strip() for ln in lines if ln.strip().startswith("[")],
        "error": "" if proc.returncode == 0 else "\n".join(lines[-6:]),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root containing the sequence folders")
    ap.add_argument("-c", "--config", required=True, help="to_bag.py YAML config")
    ap.add_argument("--glob", default="2023*", help="sequence folder pattern (default: 2023*)")
    ap.add_argument("--only", nargs="+", help="convert just these sequence names")
    ap.add_argument("--out-dir", help="output root (default: <root>/Converted_bag)")
    ap.add_argument("--suffix", help="output name suffix (default: derived from the config name)")
    ap.add_argument("--calibration", help="calibration_lidar.yaml "
                                          "(default: <root>/Calibration/calibration_lidar.yaml)")
    ap.add_argument("--no-calibration", action="store_true",
                    help="use the extrinsic / clock offset written in the config instead")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel conversions (default 1; >1 only helps on an SSD -- "
                         "concurrent readers thrash a spinning disk)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild existing outputs (each is replaced only once its "
                         "replacement is complete; a failure leaves the old bag intact)")
    ap.add_argument("--dry-run", action="store_true", help="check inputs and print the plan only")
    ap.add_argument("--start-sec", type=float, help="passed through to to_bag.py")
    ap.add_argument("--limit-sec", type=float, help="passed through to to_bag.py")
    add_distro_arg(ap)
    ap.add_argument("--resource-log", metavar="CSV",
                    help="CPU/memory/disk log path (default: "
                         "<out-dir>/logs/resources_<timestamp>.csv)")
    ap.add_argument("--no-resource-log", action="store_true",
                    help="do not sample CPU/memory/disk usage")
    ap.add_argument("--no-resource-chart", action="store_true",
                    help="write the resource CSV but skip the PNG chart")
    ap.add_argument("--resource-interval", type=float, default=1.0, metavar="SEC",
                    help="resource sampling period (default: 1.0)")
    args = ap.parse_args(argv)

    # Validate the arguments before touching anything: with --force, a window this
    # script cannot use would otherwise be discovered by each subprocess in turn,
    # one output at a time.
    check_window(args.start_sec, args.limit_sec)
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"--root is not a directory: {root}")
    out_root = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(root, "Converted_bag")
    suffix = check_suffix(args.suffix if args.suffix else default_suffix(args.config))
    cfg = load_yaml(args.config)
    if "sources" not in cfg:
        raise SystemExit(f"{args.config} has no `sources:` list -- is it a to_bag.py config?")

    # Calibration: read once, injected into the config every sequence shares.
    cal_path = args.calibration or os.path.join(root, "Calibration", "calibration_lidar.yaml")
    notes: list[str] = []
    if not args.no_calibration and os.path.isfile(cal_path):
        notes = apply_calibration(cfg, read_calibration(cal_path))
        print(f"calibration: {cal_path}")
        for n in notes:
            print(f"  {n}")
        if not notes:
            print("  (config already matches the calibration; nothing overridden)")
    elif not args.no_calibration:
        print(f"calibration: not found at {cal_path}; using the values in {args.config}")

    names = find_sequences(root, args.glob, args.only)
    if not names:
        raise SystemExit(f"no sequence folders matching {args.glob!r} under {root}")

    # Validate everything up front -- a full run is ~25 min, so failing at the
    # end because one sequence lacks imu.csv is a waste of the whole batch.
    dirs, files = required_inputs(cfg)
    jobs, skipped, broken = [], [], []
    for name in names:
        seq_dir = os.path.join(root, name)
        problems = check_sequence(seq_dir, dirs, files)
        if problems:
            broken.append((name, problems))
            continue
        out_bag = resolve_output(out_root, name, suffix)
        if os.path.exists(out_bag) and not args.force:
            skipped.append(name)
            continue
        # Nothing is deleted here: to_bag --force stages the new bag and only swaps
        # it in once it is complete, so a failure leaves the existing bag usable.
        jobs.append({"name": name, "seq_dir": seq_dir, "out_bag": out_bag,
                     "config": args.config, "start_sec": args.start_sec,
                     "limit_sec": args.limit_sec, "force": args.force,
                     "ros_distro": args.ros_distro})

    print(f"\nroot     {root}\nconfig   {args.config}\noutput   {out_root}/<seq>{suffix}")
    print(f"found {len(names)} sequences: {len(jobs)} to convert, "
          f"{len(skipped)} already present, {len(broken)} unusable")
    for name, problems in broken:
        print(f"  SKIP {name}: {', '.join(problems)}")
    if skipped:
        print(f"  present (use --force to rebuild): {', '.join(skipped)}")
    if args.dry_run:
        for j in jobs:
            print(f"  would convert {j['name']} -> {j['out_bag']}")
        return
    if not jobs:
        print("nothing to do")
        return

    os.makedirs(out_root, exist_ok=True)
    res_csv = None if args.no_resource_log else (
        args.resource_log or os.path.join(out_root, "logs",
                                          f"resources_{time.strftime('%Y%m%d-%H%M%S')}.csv"))

    # The sampler measures this process and every descendant, so it has to be
    # running around the conversions themselves; the summary is printed inside
    # the block so the resource lines land at the very end of the run, after
    # (and including) a failing exit.
    with resource_log.record(res_csv, args.resource_interval, disk_path=out_root,
                             chart=not args.no_resource_chart) as reslog:
        # The injected calibration has to reach the subprocess, so hand it a derived
        # config rather than the original path.
        with tempfile.TemporaryDirectory() as tmp:
            if notes:
                derived = os.path.join(tmp, os.path.basename(args.config))
                with open(derived, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                for j in jobs:
                    j["config"] = derived

            results, started = [], time.time()
            print()
            if args.jobs > 1:
                # No per-sequence label here: several conversions overlap, so a
                # band naming one of them would misattribute the other's load.
                if reslog:
                    reslog.stage(f"{len(jobs)} sequences, {args.jobs} jobs")
                with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                    for r in pool.map(convert_one, jobs):
                        results.append(r)
                        print(f"[{len(results)}/{len(jobs)}] {r['name']}: "
                              f"{'ok' if r['ok'] else 'FAILED'} ({r['seconds']:.0f}s)")
            else:
                for i, j in enumerate(jobs, 1):
                    print(f"[{i}/{len(jobs)}] {j['name']} ...", flush=True)
                    if reslog:
                        reslog.stage(j["name"])
                    results.append(convert_one(j))
                    r = results[-1]
                    print(f"      {'ok' if r['ok'] else 'FAILED'} ({r['seconds']:.0f}s)")
            if reslog:
                reslog.stage("")

        # Summary. Warnings are reprinted here because a 25-minute scroll buries them,
        # and they are the part that decides whether the bags are trustworthy.
        elapsed = time.time() - started
        good = [r for r in results if r["ok"]]
        print(f"\n{'='*72}\nconverted {len(good)}/{len(results)} in {elapsed/60:.1f} min\n")
        for r in results:
            print(f"{r['name']}  {r['done'].replace('done: ', '') if r['ok'] else 'FAILED'}")
            for w in r["warnings"]:
                print(f"    ! {w.replace('WARNING ', '')}")
            for t in r["tally"]:
                print(f"    ! {t}")
            if not r["ok"]:
                print(f"    {r['error']}")
        flagged = [r["name"] for r in results if r["warnings"] or r["tally"]]
        if flagged:
            print(f"\n{len(flagged)} sequence(s) produced warnings: {', '.join(flagged)}")
        if len(good) != len(results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
