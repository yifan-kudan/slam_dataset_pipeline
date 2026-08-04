#!/usr/bin/env python3
"""BAG is a ROS 1 ``.bag`` file or a ROS 2 bag directory (auto-detected).

Examples:
    python3 src/inspect_bag.py "/path/ConSLAM/seq1/Copy of recording.bag"
    python3 src/inspect_bag.py /path/raw.bag --limit-sec 10 --sample 5
    python3 src/inspect_bag.py /path/seq1_fastlio_bag --topics /velodyne_points /imu
    python3 src/inspect_bag.py /path/raw.bag --deep --json > seq1_probe.json
    python3 src/inspect_bag.py /path/odd.bag --time-fields stamp_ns --ring-fields beam
    python3 src/inspect_bag.py /path/imu_only.bag --gap-floor-sec 0.05
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.highlevel import AnyReader

# The PointField datatype table, the endian-aware strided read and the
# vendor-specific field-name / time-unit knowledge all live in pc2_layout, shared
# with rosbag1_to_ros2.py. Only the *names* there are dataset dependent, and they
# are overridable per run via --time-fields / --ring-fields.
from pc2_layout import (
    DEFAULT_RING_FIELDS, DEFAULT_TIME_FIELDS, TIME_UNIT_TO_NS,
    epoch_kind, field_view, find_field, guess_time_unit, type_name,
)


# Absolute gap tolerance for bags with no PointCloud2 topic to derive one from.
# 0.1s is one frame of a 10Hz lidar -- see ``resolve_gap_floor``.
FALLBACK_GAP_FLOOR_S = 0.1


def diffs_s(stamps_ns: list[int]) -> np.ndarray:
    """Adjacent stamp spacings, in seconds (empty for fewer than 2 stamps).

    The subtraction happens in int64: ns-epoch stamps are ~1.7e18, well past
    float64's exact-integer range, so converting before differencing would
    silently drop sub-microsecond bits.
    """
    if len(stamps_ns) < 2:
        return np.empty(0)
    return np.diff(np.asarray(stamps_ns, dtype=np.int64)) / 1e9  # convert to seconds


def median_dt_s(d: np.ndarray) -> float:
    """Nominal period, from the median of the strictly-positive spacings.

    Zeros (duplicate stamps) and negatives (backwards jumps) are excluded, so a
    partly-scrambled stream still reports the period its sensor really ran at;
    median rather than mean because dropouts are outliers that drag a mean.
    Returns 0.0 when there is no positive spacing to measure at all.
    """
    pos_dt = d[d > 0]
    return float(np.median(pos_dt)) if pos_dt.size else 0.0


def resolve_gap_floor(cloud_periods: list[float]) -> float:
    """Absolute gap tolerance: a hole shorter than one lidar frame is harmless.

    What makes a hole matter downstream is whether it can leave a scan-to-scan
    window with no IMU to preintegrate, so the tolerance is one lidar frame --
    and that has to come from this bag's measured period, not an assumed 10Hz.
    Given several clouds the fastest one starves first, so its period binds.
    """
    return min(cloud_periods) if cloud_periods else FALLBACK_GAP_FLOOR_S


def dt_stats(stamps_ns: list[int], gap_factor: float = 3.0,
             gap_floor_s: float = FALLBACK_GAP_FLOOR_S) -> dict[str, Any]:
    """Summarize spacing of a sorted-by-arrival stamp sequence."""
    if len(stamps_ns) < 2:
        return {"samples": len(stamps_ns)}
    d = diffs_s(stamps_ns)
    monotonic = bool(np.all(d >= 0))
    median = median_dt_s(d)
    # Two rules, whichever is looser. ``gap_factor`` is relative and carries the
    # low-rate topics: at 3.0 a lone dropped message (dt = 2x median) passes and
    # only 2+ consecutive losses flag. It is useless on high-rate topics, where
    # 3x median sits inside normal jitter -- hence the absolute floor, added to
    # the median so the threshold always clears the nominal period.
    gap_thresh = max(gap_factor * median, median + gap_floor_s) if median else float("inf")
    gaps = int(np.count_nonzero(d > gap_thresh))
    return {
        "samples": len(stamps_ns),
        "rate_hz": round(1.0 / median, 3) if median else None,
        "dt_min_s": round(float(d.min()), 6),
        "dt_median_s": round(median, 6),
        "dt_max_s": round(float(d.max()), 6),
        "monotonic": monotonic,
        "n_backwards": int(np.count_nonzero(d < 0)),
        "n_gaps": gaps,
        "gap_thresh_s": round(gap_thresh, 4) if median else None,
        "n_near_zero_dt": int(np.count_nonzero((d >= 0) & (d < 1e-4))),
    }


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def describe_pointcloud(msg, time_fields: tuple[str, ...] = DEFAULT_TIME_FIELDS,
                        ring_fields: tuple[str, ...] = DEFAULT_RING_FIELDS) -> dict[str, Any]:
    """Layout of one PointCloud2: field table plus what a converter config needs.

    ``time_fields`` / ``ring_fields`` are the candidate names to probe, in
    priority order -- the only dataset-dependent input here. The full field list
    is reported regardless, so a name outside the candidates is still visible.
    """
    fields = [{"name": f.name, "offset": int(f.offset),
               "datatype": type_name(f.datatype), "count": int(f.count)}
              for f in msg.fields]
    tfield = find_field(msg, time_fields)
    rfield = find_field(msg, ring_fields)
    info = {
        "fields": fields,
        "field_names": [f["name"] for f in fields],
        "point_step": int(msg.point_step),
        "width": int(msg.width),
        "height": int(msg.height),
        "n_points": int(msg.width) * int(msg.height),
        "is_dense": bool(getattr(msg, "is_dense", False)),
        "is_bigendian": bool(getattr(msg, "is_bigendian", False)),
        "has_ring": rfield is not None,
        "has_time": tfield is not None,
        "ring_field": rfield.name if rfield is not None else None,
        # Non-scalar fields break the strided single-element read below (and any
        # in-place rewrite in the converter), so surface them rather than hide it.
        "non_scalar_fields": [f["name"] for f in fields if f["count"] != 1],
    }
    # Per-point time range: tells you the unit (span ~ one revolution) and
    # whether the values are absolute Unix time or offsets from scan start.
    if tfield is not None and info["n_points"] > 0:  # min/max needs at least a point
        arr = field_view(bytes(msg.data), msg, tfield)
        tmin, tmax = float(arr.min()), float(arr.max())
        unit = guess_time_unit(tmax - tmin)
        info["point_time_field"] = tfield.name
        info["point_time_dtype"] = type_name(tfield.datatype)
        info["point_time_min"] = tmin
        info["point_time_max"] = tmax
        info["point_time_span"] = tmax - tmin
        info["point_time_unit_guess"] = unit
        if unit:
            # Reuse the header-stamp classifier: scale the raw value to ns first.
            info["point_time_epoch_kind"] = epoch_kind(tmin * TIME_UNIT_TO_NS[unit])
    return info


def inspect(bag: str, topics: list[str] | None, sample: int, deep: bool,
            limit_sec: float | None = None,
            time_fields: tuple[str, ...] = DEFAULT_TIME_FIELDS,
            ring_fields: tuple[str, ...] = DEFAULT_RING_FIELDS,
            gap_factor: float = 3.0,
            gap_floor_s: float | None = None) -> dict[str, Any]:
    path = Path(bag)
    with AnyReader([path]) as reader:
        all_conns = reader.connections
        wanted = [c for c in all_conns if topics is None or c.topic in topics]
        if topics is not None and not wanted:
            raise SystemExit(
                f"none of {topics} present. topics in bag: "
                f"{sorted({c.topic for c in all_conns})}"
            )

        # bag-level
        bagtype = "ros2" if path.is_dir() else "ros1"
        out: dict[str, Any] = {
            "bag": str(path),
            "bag_type": bagtype,
            "start_time_ns": int(reader.start_time),
            "end_time_ns": int(reader.end_time),
            "duration_s": round((reader.end_time - reader.start_time) / 1e9, 3),
            "start_epoch_kind": epoch_kind(int(reader.start_time)),
            "scanned_full": limit_sec is None,
            # Record which candidate names this run probed, so a JSON report is
            # self-describing when has_time/has_ring came out False.
            "probed_time_fields": list(time_fields),
            "probed_ring_fields": list(ring_fields),
            "topics": {},
        }

        per_topic_conns: dict[str, list] = {}
        for c in wanted:
            per_topic_conns.setdefault(c.topic, []).append(c)

        # Single pass. ``rosbags`` always yields the message blob, so a second
        # timestamp-only pass would double the (HDD-bound) IO for nothing:
        # collect bag receive-time for every message here, and in the same loop
        # deserialize up to ``sample`` messages per topic (all of them under
        # --deep) for structure + header-stamp stats.
        limit = None if deep else sample
        t0 = int(reader.start_time)
        bag_stamps: dict[str, list[int]] = {t: [] for t in per_topic_conns}
        hdr_stamps: dict[str, list[int]] = {t: [] for t in per_topic_conns}
        hdr_minus_bag: dict[str, list[float]] = {t: [] for t in per_topic_conns}
        structure: dict[str, Any] = {}
        seen: dict[str, int] = {t: 0 for t in per_topic_conns}
        for c, ts, raw in reader.messages(connections=wanted):
            if limit_sec is not None and (int(ts) - t0) / 1e9 > limit_sec:
                break
            t = c.topic
            bag_stamps[t].append(int(ts))
            if limit is not None and seen[t] >= limit:
                continue
            seen[t] += 1
            msg = reader.deserialize(raw, c.msgtype)
            hdr = getattr(msg, "header", None)
            if hdr is not None:
                sns = stamp_ns(hdr)
                hdr_stamps[t].append(sns)
                hdr_minus_bag[t].append(round((sns - int(ts)) / 1e9, 6))
                structure.setdefault(t, {})["frame_id"] = hdr.frame_id
            if t not in structure or "kind" not in structure[t]:
                st = structure.setdefault(t, {})
                if "PointCloud2" in c.msgtype:
                    st["kind"] = "pointcloud2"
                    st.update(describe_pointcloud(msg, time_fields, ring_fields))
                elif "Imu" in c.msgtype:
                    st["kind"] = "imu"
                elif hdr is not None:
                    st["kind"] = "stamped"

        # Gap detection needs a reference frame period, so measure the cloud
        # topics before scoring anything. Pure numpy over stamps already in
        # memory -- the expensive deserialize pass above still runs only once.
        if gap_floor_s is not None:
            gap_floor, floor_src = gap_floor_s, "cli"
        else:
            cloud_periods = [
                m for t, conns in per_topic_conns.items()
                if "PointCloud2" in conns[0].msgtype
                and (m := median_dt_s(diffs_s(bag_stamps[t]))) > 0
            ]
            gap_floor = resolve_gap_floor(cloud_periods)
            floor_src = "cloud_period" if cloud_periods else "fallback_10hz"
        # Echo the thresholds: an n_gaps of 0 means nothing without them.
        out["gap_factor"] = gap_factor
        out["gap_floor_s"] = round(gap_floor, 6)
        out["gap_floor_src"] = floor_src

        for t, conns in per_topic_conns.items():
            entry = {
                "msgtype": conns[0].msgtype,
                "n_connections": len(conns),
                "count": len(bag_stamps[t]),
                "bag_time": dt_stats(bag_stamps[t], gap_factor, gap_floor),
            }
            if hdr_stamps[t]:
                entry["header_time"] = dt_stats(hdr_stamps[t], gap_factor, gap_floor)
                entry["header_time"]["scope"] = "full" if deep else f"first_{len(hdr_stamps[t])}"
                entry["header_time"]["epoch_kind"] = epoch_kind(hdr_stamps[t][0])
                offs = hdr_minus_bag[t]
                entry["header_minus_bag_s"] = {
                    "min": min(offs), "max": max(offs),
                    "mean": round(sum(offs) / len(offs), 6),
                }
            entry["structure"] = structure.get(t, {})
            out["topics"][t] = entry
        return out


def print_report(r: dict[str, Any]) -> None:
    print(f"# {r['bag']}  [{r['bag_type']}]")
    print(f"  duration {r['duration_s']}s   start_epoch={r['start_epoch_kind']} "
          f"(start_time_ns={r['start_time_ns']})")
    print(f"  gap rule: dt > max({r['gap_factor']}x median, median + {r['gap_floor_s']}s)"
          f"   floor from {r['gap_floor_src']}")
    for t, e in sorted(r["topics"].items()):
        bt = e["bag_time"]
        print(f"\n  {t}   {e['msgtype']}   x{e['count']}")
        print(f"    bag_time:  rate~{bt.get('rate_hz')}Hz  "
              f"dt[min/med/max]={bt.get('dt_min_s')}/{bt.get('dt_median_s')}/{bt.get('dt_max_s')}s"
              f"  gaps={bt.get('n_gaps')}  near0dt={bt.get('n_near_zero_dt')}"
              f"  monotonic={bt.get('monotonic')}")
        if "header_time" in e:
            ht = e["header_time"]
            print(f"    hdr_time:  rate~{ht.get('rate_hz')}Hz "
                  f"dt[min/med/max]={ht.get('dt_min_s')}/{ht.get('dt_median_s')}/{ht.get('dt_max_s')}s"
                  f"  gaps={ht.get('n_gaps')}  near0dt={ht.get('n_near_zero_dt')}"
                  f"  epoch={ht.get('epoch_kind')}  ({ht.get('scope')})")
            hb = e["header_minus_bag_s"]
            print(f"    hdr-bag:   [{hb['min']}, {hb['max']}]s mean={hb['mean']}")
        st = e.get("structure", {})
        if st.get("kind") == "pointcloud2":
            print(f"    cloud:     {st['n_points']} pts  point_step={st['point_step']}  "
                  f"frame={st.get('frame_id')}")
            print(f"    fields:    {', '.join(f'{f['name']}@{f['offset']}:{f['datatype']}' for f in st['fields'])}")
            if st.get("is_bigendian"):
                print("    WARNING:   is_bigendian=True")
            if st.get("non_scalar_fields"):
                print(f"    WARNING:   count!=1 fields: {', '.join(st['non_scalar_fields'])}")
            print(f"    has_time={st['has_time']} has_ring={st['has_ring']}", end="")
            if st.get("ring_field") not in (None, "ring"):
                print(f" (ring as '{st['ring_field']}')", end="")
            if "point_time_field" in st:
                print(f"  time '{st['point_time_field']}':{st['point_time_dtype']}"
                      f" range=[{st['point_time_min']}, {st['point_time_max']}]"
                      f" span={st['point_time_span']:.6g}"
                      f" unit~{st['point_time_unit_guess']}"
                      f" {st.get('point_time_epoch_kind', '')}", end="")
            print()
        elif st.get("frame_id") is not None:
            print(f"    frame={st.get('frame_id')}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", help="ROS 1 .bag file or ROS 2 bag directory")
    p.add_argument("--topics", nargs="+", default=None,
                   help="only inspect these topics (default: all)")
    p.add_argument("--sample", type=int, default=500,
                   help="messages per topic to deserialize for structure/header (default 500)")
    p.add_argument("--deep", action="store_true",
                   help="deserialize every message so header-stamp stats cover the whole bag (slow)")
    p.add_argument("--limit-sec", type=float, default=None,
                   help="only scan the first N seconds of the bag (quick onboarding probe)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    # Vendors name the per-point time / ring fields differently; these are the
    # only dataset-dependent inputs, so they are configurable rather than hardcoded.
    p.add_argument("--time-fields", nargs="+", default=list(DEFAULT_TIME_FIELDS),
                   metavar="NAME",
                   help=f"candidate per-point time field names, priority order "
                        f"(default: {' '.join(DEFAULT_TIME_FIELDS)})")
    p.add_argument("--ring-fields", nargs="+", default=list(DEFAULT_RING_FIELDS),
                   metavar="NAME",
                   help=f"candidate ring/beam field names, priority order "
                        f"(default: {' '.join(DEFAULT_RING_FIELDS)})")
    p.add_argument("--gap-factor", type=float, default=3.0,
                   help="dt beyond this multiple of a topic's median counts as a gap "
                        "(default 3.0: tolerate one dropped message, flag 2+ in a row)")
    p.add_argument("--gap-floor-sec", type=float, default=None, metavar="SEC",
                   help="absolute gap tolerance, which dominates on high-rate topics. "
                        f"Default: measure it from the fastest PointCloud2 topic's "
                        f"period, or {FALLBACK_GAP_FLOOR_S}s if the bag has no cloud")
    args = p.parse_args(argv)
    report = inspect(args.bag, args.topics, args.sample, args.deep, args.limit_sec,
                     tuple(args.time_fields), tuple(args.ring_fields),
                     args.gap_factor, args.gap_floor_sec)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
