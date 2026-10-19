#!/usr/bin/env python3
"""Batch-convert PCD point clouds to PLY or LAS/LAZ.

The clouds this pipeline produces are PCDs -- FAST-LIO writes ``fastlio_map.pcd``
per run, and several datasets ship per-frame ``.pcd`` scans -- but nothing
downstream reads PCD. This converts a whole directory into whichever of the two
formats the consumer needs, sharing the PCD reader with ``to_bag.py``
(``pcd_io.py``) so a cloud is parsed the same way whether it ends up in a bag or
in a file on disk.

Which format, and why it is a choice rather than a default:

``--format ply`` (default)
  For **looking at** a cloud: CloudCompare, MeshLab, Open3D and Blender all read
  PLY. It is schema-free, so the conversion is a lossless re-container -- every
  field arrives under its own name with the width the PCD header declared.

``--format las`` / ``--format laz``
  For **feeding PDAL**, which is the only point-cloud reader ``build_map`` has.
  LAS is a survey format and costs two things a map should not pay silently:

    1. coordinates are quantised -- LAS stores ``int32 * scale + offset``, so the
       header scale is the resolution of the file. ``--las-scale`` defaults to
       1 mm (0.5 mm worst-case error) rather than PDAL's 1 cm, which is coarser
       than the voxel sizes these maps get built and compared at, and the error it
       introduces is reported per run rather than left to be discovered;
    2. the dimension list is fixed. x/y/z, intensity and RGB have a home; the
       ``normal_x``, ``ring``, ``t``, ``curvature`` and ``label`` fields these
       clouds actually carry become ExtraBytes dimensions -- named and exactly
       typed, but scalars a viewer cannot interpret as normals or colour. Which
       fields those were is reported per cloud too.

  So: LAS to build a map from, PLY to check the map in.

Beyond a shell loop over ``pcl_pcd2ply`` or ``pdal translate``:

  * **needs no PCL or PDAL install** -- numpy, plus laspy for LAS output;
  * **checks every input before converting any of them.** Headers are cheap to
    read, so a missing field, a ``binary_compressed`` payload or two inputs whose
    names would collide in one output directory surface in the first second;
  * **translates names instead of dumping fields verbatim.** PCL's ``normal_x`` is
    PLY's ``nx``, and a packed ``rgb`` float becomes red/green/blue, so a viewer
    shows normals and colour rather than anonymous scalars;
  * **never destroys an output it has not replaced.** Each file is written beside
    its target and renamed into place when complete, so a Ctrl-C on a 10-million
    point map leaves the previous file intact rather than a truncated one.

Examples::

    # one map -> /media/.../maps/fastlio_map.ply
    python3 src/pcd_convert.py /media/kudan/seagate-2tb/datasets/hilti_2023/maps

    # every sequence's map, each output beside its own <bag>/maps/fastlio_map.pcd.
    # No -o: the maps are all called fastlio_map, so one output directory would
    # be a name clash (check_collisions) -- the sequence name lives in the path.
    python3 src/pcd_convert.py /media/.../ConPR/Converted_bag --recursive

    # the same maps as LAS, for build_map's PDAL loader
    python3 src/pcd_convert.py /media/.../ConPR/Converted_bag --recursive --format las

    # per-frame scans into one directory, coordinates + intensity only
    python3 src/pcd_convert.py <seq>/lidar-pcd -o <seq>/ply --fields x y z intensity
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from pcd_io import (LAS_SCALE, check_laz_backend, import_laspy, las_extra_fields, parse_pcd,
                    read_pcd_header, to_las_columns, unpack_rgb, write_las, write_ply)

# x, y, z first and in that order: a PLY consumer identifies the geometry by
# those property names, and every tool expects them at the head of the vertex.
XYZ = ("x", "y", "z")

# Output formats, and the file extension each gets (which is the format name).
FORMATS = ("ply", "las", "laz")

# Per-file progress / summary lines are printed in full up to this many clouds;
# past it a per-frame directory would print thousands of them (see report()).
PROGRESS_LINES = 40


# --------------------------------------------------------------------------
# Discovery / validation
# --------------------------------------------------------------------------
def find_inputs(inputs: list[str], pattern: str, recursive: bool) -> list[str]:
    """Expand files and directories into a sorted, de-duplicated list of PCDs."""
    found: list[str] = []
    for item in inputs:
        if os.path.isfile(item):
            found.append(os.path.abspath(item))
        elif os.path.isdir(item):
            where = os.path.join(item, "**", pattern) if recursive else os.path.join(item, pattern)
            matched = sorted(glob.glob(where, recursive=recursive))
            if not matched:
                raise SystemExit(f"no files matching {pattern!r} "
                                 f"{'under' if recursive else 'in'} {item}")
            found += [os.path.abspath(p) for p in matched if os.path.isfile(p)]
        else:
            raise SystemExit(f"input not found: {item}")
    return sorted(dict.fromkeys(found))


def labeller(paths: list[str]):
    """A naming function that tells the inputs apart in the progress and summary.

    Basenames do not: a FAST-LIO batch is a dozen ``fastlio_map.pcd`` under a
    dozen sequence directories, and a summary of twelve identical lines says
    nothing about which sequence is which. The path relative to what the inputs
    have in common is the shortest label that stays unambiguous.
    """
    if len(paths) < 2:
        return os.path.basename
    root = os.path.commonpath(paths)
    return lambda p: os.path.relpath(p, root)


def probe(path: str) -> dict[str, Any]:
    """Read one PCD's header only: fields, types and point count, without the payload."""
    with open(path, "rb") as f:
        header = read_pcd_header(f)
    return {"fields": header["fields"], "points": header["points"], "data": header["data"],
            "dtype": header["dtype"]}


def resolve_output(path: str, out_dir: str | None, fmt: str) -> str:
    """``<out_dir>/<stem>.<fmt>``, or ``<stem>.<fmt>`` beside the input."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(out_dir if out_dir else os.path.dirname(path), f"{stem}.{fmt}")


def select_fields(available: list[str], wanted: list[str] | None, path: str) -> list[str]:
    """Which fields to write, in output order.

    Without ``--fields`` everything the cloud carries is written -- for PLY that is
    a lossless re-container, which is the useful default for a map that took an
    hour to build. For LAS it is still the right default (an extra byte carries a
    field exactly, and dropping one cannot be undone) but it is not lossless: see
    ``pcd_io.to_las_columns``, and use ``--fields x y z intensity`` for a cloud
    whose trailing fields are of no use to the consumer.

    x/y/z are hoisted to the front either way, and required in both cases: a file
    without them is not a point cloud.
    """
    missing = [a for a in XYZ if a not in available]
    if missing:
        raise SystemExit(f"{path}: no {missing} field(s); present: {available}")
    if wanted is None:
        rest = [f for f in available if f not in XYZ]
    else:
        unknown = [w for w in wanted if w not in available]
        if unknown:
            raise SystemExit(f"{path}: --fields names {unknown}, which the cloud does not "
                             f"carry; present: {available}")
        rest = [w for w in wanted if w not in XYZ]
    return [*XYZ, *rest]


def check_collisions(jobs: list[dict[str, Any]]) -> None:
    """Refuse a batch in which two inputs map to the same output.

    ``--recursive`` over per-run map directories is exactly the case that bites:
    a dozen ``fastlio_map.pcd`` under different parents all want
    ``<out_dir>/fastlio_map.ply``, and without this the last one silently wins.
    """
    by_out: dict[str, list[str]] = {}
    for job in jobs:
        by_out.setdefault(job["out"], []).append(job["path"])
    clashes = {out: srcs for out, srcs in by_out.items() if len(srcs) > 1}
    if clashes:
        lines = "\n".join(f"  {out} <- {', '.join(srcs)}" for out, srcs in sorted(clashes.items()))
        raise SystemExit(f"{len(clashes)} output name(s) would be written by more than one "
                         f"input:\n{lines}\nconvert the directories one at a time, or drop "
                         f"-o so each output is written beside its source")


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def convert_one(job: dict[str, Any]) -> dict[str, Any]:
    """Convert one PCD. Returns a result record; never raises for a bad input, so
    one unreadable file cannot take the rest of the batch with it."""
    started = time.time()
    result: dict[str, Any] = {"path": job["path"], "out": job["out"], "ok": False,
                              "label": job.get("label", os.path.basename(job["path"])),
                              "points": 0, "dropped": 0, "nonfinite": 0, "notes": [],
                              "error": ""}
    try:
        columns = parse_pcd(job["path"])
        # The fields were resolved from the header during validation; re-check them
        # against the payload, which is what a truncated or hand-edited file breaks.
        missing = [n for n in job["fields"] if n not in columns]
        if missing:
            raise SystemExit(f"payload has no {missing} field(s); present: {sorted(columns)}")
        # Select before unpacking, so `--fields rgb` names the packed field the PCD
        # carries rather than the three properties it becomes.
        picked = {n: np.asarray(columns[n]) for n in job["fields"]}
        keep = finite_mask(picked)
        if keep is not None:
            result["nonfinite"] = int((~keep).sum())
            if job["drop_nonfinite"]:
                result["dropped"] = result["nonfinite"]
                picked = {n: c[keep] for n, c in picked.items()}
        # Shared: a packed colour has to become channels before either format can
        # carry it. What happens after that is format-specific.
        picked = unpack_rgb(picked)
        provenance = f"converted from {os.path.basename(job['path'])} by pcd_convert.py"
        part = job["out"] + ".part"
        if job["format"] == "ply":
            result["points"] = write_ply(part, picked, binary=job["binary"],
                                         comments=(provenance,))
        else:
            picked, result["notes"] = to_las_columns(picked)
            result["points"] = write_las(part, picked, scale=job["las_scale"],
                                         compress=job["format"] == "laz",
                                         comment=provenance)
        # Rename only once the file is complete: the point of --force is to replace
        # an output, not to lose the old one to a half-finished write.
        os.replace(part, job["out"])
        result["ok"] = True
    except (SystemExit, OSError, ValueError, OverflowError) as exc:
        # OverflowError: laspy raises it rather than wrapping when a coordinate does
        # not fit the int32. _las_offsets checks first and says more, so reaching it
        # here means a non-coordinate dimension overflowed.
        result["error"] = str(exc) or exc.__class__.__name__
        for leftover in (job["out"] + ".part",):
            if os.path.exists(leftover):
                os.remove(leftover)
    result["seconds"] = time.time() - started
    return result


def finite_mask(columns: dict[str, np.ndarray]) -> np.ndarray | None:
    """Mask of points whose coordinates are finite, or None if all of them are.

    Only x/y/z are tested: a NaN coordinate is what breaks a viewer's bounding box
    (and every distance it computes), while a NaN intensity is just a scalar the
    viewer can be told to ignore.
    """
    bad = np.zeros(len(columns["x"]), dtype=bool)
    for axis in XYZ:
        col = columns[axis]
        if col.dtype.kind == "f":
            bad |= ~np.isfinite(col)
    return None if not bad.any() else ~bad


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="PCD files and/or directories of them")
    ap.add_argument("-o", "--out-dir", help="output directory (default: beside each input)")
    ap.add_argument("--glob", default="*.pcd", help="filename pattern for directory inputs "
                                                   "(default: *.pcd)")
    ap.add_argument("--recursive", action="store_true", help="search directories recursively")
    ap.add_argument("--format", choices=FORMATS, default="ply",
                    help="output format (default: ply -- lossless, and what the viewers read; "
                         "las/laz for build_map's PDAL loader, at the cost of quantised "
                         "coordinates and extra-byte scalars)")
    ap.add_argument("--fields", nargs="+", help="fields to keep (default: all of them); "
                                                "x y z are always written first")
    ap.add_argument("--ascii", action="store_true",
                    help="ascii PLY instead of binary_little_endian (~4x the size, much "
                         "slower; useful only to eyeball the numbers). --format ply only")
    ap.add_argument("--las-scale", type=float, default=LAS_SCALE, metavar="M",
                    help=f"LAS coordinate quantisation step in metres (default "
                         f"{LAS_SCALE}, i.e. {LAS_SCALE / 2 * 1000:g} mm worst-case error; "
                         f"PDAL's own default is a much coarser 0.01). --format las/laz only")
    ap.add_argument("--keep-nonfinite", action="store_true",
                    help="write points whose coordinates are NaN/Inf instead of dropping them "
                         "(--format ply only: LAS coordinates are integers, and NaN has no "
                         "integer to become)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel conversions (default 1; >1 only helps on an SSD -- "
                         "concurrent readers thrash a spinning disk)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild existing outputs (each is replaced only once its "
                         "replacement is complete)")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the inputs and print the plan only")
    args = ap.parse_args(argv)

    if args.jobs < 1:
        raise SystemExit(f"--jobs must be at least 1, got {args.jobs}")
    # Reject the format/flag combinations that would otherwise be accepted and then
    # quietly ignored -- `--ascii --format las` looks like it asked for something.
    if args.format == "ply":
        if args.las_scale != LAS_SCALE:
            raise SystemExit("--las-scale applies to --format las/laz; PLY stores coordinates "
                             "as floats and quantises nothing")
    else:
        if args.ascii:
            raise SystemExit(f"--ascii applies to --format ply; {args.format} is a binary format")
        if args.keep_nonfinite:
            raise SystemExit(f"--keep-nonfinite applies to --format ply: a {args.format} "
                             f"coordinate is an integer, so a NaN cannot be written as one. "
                             f"Convert to PLY to look at those points")
        if not args.las_scale > 0:
            raise SystemExit(f"--las-scale must be positive, got {args.las_scale}")
        # Up front, once: laspy missing (or LAZ without a backend) is not a property
        # of any one cloud, and a 1400-frame batch should not learn it 1400 times.
        import_laspy()
        if args.format == "laz":
            check_laz_backend()
    paths = find_inputs(args.inputs, args.glob, args.recursive)
    if not paths:
        raise SystemExit("no input files")
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None

    # Validate every input up front. Headers are cheap, and a batch over a set of
    # runs should not convert nine maps and then fail on the tenth's fields.
    label = labeller(paths)
    jobs, skipped, broken = [], [], []
    for path in paths:
        try:
            info = probe(path)
            fields = select_fields(info["fields"], args.fields, path)
        except (SystemExit, OSError, ValueError) as exc:
            broken.append((path, str(exc)))
            continue
        out = resolve_output(path, out_dir, args.format)
        if os.path.exists(out) and not args.force:
            skipped.append(path)
            continue
        jobs.append({"path": path, "out": out, "fields": fields, "binary": not args.ascii,
                     "drop_nonfinite": not args.keep_nonfinite, "points": info["points"],
                     "format": args.format, "las_scale": args.las_scale, "label": label(path),
                     "dtype": info["dtype"]})
    check_collisions(jobs)

    print(f"input    {len(paths)} PCD file(s)")
    print(f"output   {out_dir if out_dir else '<beside each input>'}/<name>.{args.format} "
          f"({describe_format(args)})")
    print(f"fields   {' '.join(args.fields) if args.fields else 'all'}")
    for line in schema_notes(jobs, args.format):
        print(f"         {line}")
    print(f"{len(jobs)} to convert, {len(skipped)} already present, {len(broken)} unusable")
    for path, why in broken:
        print(f"  SKIP {label(path)}: {why}")
    if skipped:
        print(f"  present (use --force to rebuild): "
              f"{summarize([label(p) for p in skipped])}")
    if args.dry_run:
        for job in jobs:
            print(f"  would convert {job['label']} -> {job['out']} "
                  f"({job['points']} points, {len(job['fields'])} fields)")
        return
    if not jobs:
        print("nothing to do")
        if broken:
            raise SystemExit(1)
        return
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # A per-frame scan directory is thousands of files, and one progress line each
    # buries the run in its own output. Thin them out past PROGRESS_LINES -- but
    # never a failure, and say what the interval is rather than looking stalled.
    step = -(-len(jobs) // PROGRESS_LINES)   # ceil, so it thins on the same count
                                             # as the summary below rather than at 2x
    results, started = [], time.time()
    print()
    if step > 1:
        print(f"converting {len(jobs)} clouds, reporting every {step}th "
              f"(a failure is always reported)")
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for r in pool.map(convert_one, jobs):
                results.append(r)
                report(len(results), len(jobs), r, step)
    else:
        for i, job in enumerate(jobs, 1):
            if step == 1:
                print(f"[{i}/{len(jobs)}] {job['label']} ...", flush=True)
            results.append(convert_one(job))
            report(i, len(jobs), results[-1], step)

    good = [r for r in results if r["ok"]]
    total_points = sum(r["points"] for r in good)
    total_mb = sum(os.path.getsize(r["out"]) for r in good) / 1e6
    print(f"\n{'=' * 72}\nconverted {len(good)}/{len(results)} clouds, {total_points} points, "
          f"{total_mb:.1f} MB in {time.time() - started:.1f}s\n")
    # Past PROGRESS_LINES only the clouds with something to say are listed, so the
    # ones that need a second look are not lost in a wall of identical lines.
    notable = [r for r in results if not r["ok"] or r["dropped"] or r["nonfinite"]]
    for r in (results if len(results) <= PROGRESS_LINES else notable):
        print(f"{r['label']}  {describe(r)}")
    quiet = len(results) - len(notable)
    if len(results) > PROGRESS_LINES and quiet:
        print(f"({quiet} cloud(s) converted with nothing to report)")
    if len(good) != len(results) or broken:
        raise SystemExit(1)


def schema_notes(jobs: list[dict[str, Any]], fmt: str) -> list[str]:
    """What the LAS schema will do to these fields, said once before the batch runs.

    Which fields have no native LAS dimension follows from the PCD headers, so it
    belongs here rather than in a per-cloud line repeated 1400 times -- and it is
    the difference between the two formats that a caller most needs to see *before*
    committing an hour of conversion to the wrong one.
    """
    if fmt == "ply" or not jobs:
        return []
    lines = []
    # Distinct sets, not the first job's: a --recursive batch can mix clouds from
    # different sensors, and a summary that describes one of them is worse than none.
    for extra in dict.fromkeys(tuple(las_extra_fields(j["fields"], j["dtype"])) for j in jobs):
        lines.append(f"extra bytes: {', '.join(extra)} (named scalars -- LAS has no dimension "
                     f"for them)" if extra else "extra bytes: none")
    return lines


def describe_format(args: argparse.Namespace) -> str:
    """The one-line statement of what the output format costs.

    For LAS that is the quantisation step, printed before any file is written: the
    scale is a decision, and a run should not have to be reconstructed afterwards
    to find out which one it was made with.
    """
    if args.format == "ply":
        return "ascii" if args.ascii else "binary_little_endian"
    return (f"LAS 1.4, {args.las_scale} m coordinate scale -> up to "
            f"{args.las_scale / 2 * 1000:g} mm quantisation error per axis")


def report(i: int, n: int, result: dict[str, Any], step: int) -> None:
    if result["ok"] and i % step and i != n:
        return
    print(f"[{i}/{n}] {result['label']}: {'ok' if result['ok'] else 'FAILED'} "
          f"({result['seconds']:.1f}s)", flush=True)


def describe(result: dict[str, Any]) -> str:
    if not result["ok"]:
        return f"FAILED: {result['error']}"
    size = os.path.getsize(result["out"]) / 1e6
    notes = []
    if result["dropped"]:
        notes.append(f"dropped {result['dropped']} non-finite")
    elif result["nonfinite"]:
        notes.append(f"KEPT {result['nonfinite']} non-finite (viewers mis-scale on those)")
    # What the LAS schema cost this particular cloud: which fields ended up as
    # extra bytes, and whether an intensity had to be clamped.
    notes += result.get("notes", [])
    return f"{result['points']} points, {size:.1f} MB" + "".join(f", {n}" for n in notes)


def summarize(labels: list[str], keep: int = 6) -> str:
    """Comma-joined, truncated -- a 1400-frame `--force`-less re-run should not
    print 1400 names to say it did nothing."""
    if len(labels) <= keep:
        return ", ".join(labels)
    return f"{', '.join(labels[:keep])} ... and {len(labels) - keep} more"


if __name__ == "__main__":
    main()
