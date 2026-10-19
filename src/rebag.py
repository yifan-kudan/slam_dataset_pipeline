#!/usr/bin/env python3
"""Re-time an existing ROS 2 bag into another ROS 2 bag.

The third converter in the set, and the only one whose *input* is already a
finished bag. ``to_bag.py`` assembles one from loose files and
``rosbag1_to_ros2.py`` re-containers a ROS 1 recording; both read the raw
dataset. This one exists because the same recording often has to be delivered
differently to different consumers, and re-deriving it from the raw data to
change nothing but the delivery order is enormously wasteful.

The motivating case: FAST-LIO and kd_localizer want the *same* clouds with
*incompatible* bag timestamps. FAST-LIO buffers a sliding window and waits for
IMU to cover each scan (``sync_packages`` returns false until it does), so it
wants the cloud delivered at scan START. kd_localizer processes each frame the
instant rosbag2 hands it over and needs the covering IMU already buffered, so it
needs the cloud delivered at scan END or it reports "missing IMU samples" every
frame. One recording, two bags, differing only in write timestamps.

Going back to the ROS 1 source for that costs a full re-read (ConSLAM: 63 GB at
~27 MB/s off USB, plus ~20 min for rosbags to index it -- about 40 minutes).
Reading the already-converted bag instead is 3.4 GB off NVMe.

Nothing is deserialized. Messages are copied as raw CDR straight from reader to
writer, so the payload is bit-identical and the cost is I/O only. That is also
why the knobs here are limited to what can be decided from the bag timestamp
alone -- shifting, dropping, and adding ``/tf_static``. Anything that has to
read or rewrite a message body (rebasing a per-point time column, repacking a
cloud layout, re-spacing IMU bursts) belongs upstream in the converter that
produced this bag.

Using the bag timestamp as the header stamp is exact for bags this pipeline
writes: both converters move the two together, so ``hdr-bag`` is 0.0 throughout
(verify with ``inspect_bag.py`` if the input came from somewhere else).

Config is the same YAML dialect as the other two -- an ``output:`` block, a
``topics:`` list keyed by input topic name, and an optional ``tf_static:`` list:

    output:
      storage_id: sqlite3
      ros_distro: both
    topics:
      - in: /velodyne_points
        write_stamp_shift_sec: 0.11     # deliver at scan end
        warmup_after_imu_sec: 0.2       # drop scans before the IMU stream + margin
      - in: /imu
        type: imu                       # marks the stream warmup is measured from
    tf_static:
      - parent: imu_link
        child: velodyne
        translation: [0.0, 0.0, 0.028]
        rotation_xyzw: [0.0, 0.0, 1.0, 0.0]

Topics absent from ``topics:`` are dropped, so this doubles as a topic filter.
``out:`` renames a topic; omit it to keep the name.

Example:
    python3 src/rebag.py seq2_fastlio_bag -o seq2_kdamr_bag \\
        -c config/conslam/bag_conslam_kdamr.yaml
"""
import argparse
import os
from pathlib import Path
from typing import Any

from rosbags.highlevel import AnyReader

from ros2_bag_io import (
    ROS2_STORE, add_distro_arg, add_group_arg, apply_distro, atomic_bag,
    check_topics_present, check_window, filter_groups, load_yaml, normalize_topic,
    output_summary, select_groups, write_tf_static,
)


def rebag(in_bag: str, out_bag: str, cfg: dict[str, Any], limit_sec: float | None = None,
          start_sec: float | None = None, force: bool = False) -> dict[str, int]:
    topic_cfgs = {normalize_topic(t["in"]): t for t in cfg["topics"]}
    counts: dict[str, int] = {}

    with AnyReader([Path(in_bag)]) as reader:
        conns = [c for c in reader.connections if c.topic in topic_cfgs]
        check_topics_present(topic_cfgs, {c.topic for c in reader.connections}, in_bag)

        # The warmup rule needs the IMU start, which is only known after a pass
        # over the timestamps. That pass is index-only (no message bodies are
        # read), so it is far cheaper than it looks -- and cheaper than carrying
        # a "buffer clouds until the IMU shows up" state machine through the
        # write loop.
        imu_topics = [t for t, c in topic_cfgs.items() if c.get("type") == "imu"]
        first_imu_ns = None
        if any(c.get("warmup_after_imu_sec") for c in topic_cfgs.values()):
            if not imu_topics:
                raise SystemExit("warmup_after_imu_sec is set but no topic is marked "
                                 "`type: imu` -- there is no stream to measure from")
            imu_conns = [c for c in conns if c.topic in imu_topics]
            first_imu_ns = next((ts for _, ts, _ in reader.messages(connections=imu_conns)), None)
            if first_imu_ns is None:
                raise SystemExit(f"no messages on {imu_topics} -- cannot apply warmup")

        t0 = reader.start_time
        with atomic_bag(out_bag, cfg, force) as writer:
            out_conns = {}
            for c in conns:
                tc = topic_cfgs[c.topic]
                out_topic = normalize_topic(tc.get("out", c.topic))
                out_conns[c.topic] = writer.add_connection(out_topic, c.msgtype,
                                                           typestore=ROS2_STORE)
                counts[out_topic] = 0

            tf_static = cfg.get("tf_static")
            if tf_static:
                counts["/tf_static"] = write_tf_static(writer, tf_static, t0)

            for c, timestamp, raw in reader.messages(connections=conns):
                rel_sec = (timestamp - t0) / 1e9
                if start_sec is not None and rel_sec < start_sec:
                    continue
                if limit_sec is not None and rel_sec > limit_sec:
                    break
                tc = topic_cfgs[c.topic]

                warmup_sec = float(tc.get("warmup_after_imu_sec", 0.0))
                if warmup_sec > 0.0 and first_imu_ns is not None:
                    if timestamp < first_imu_ns + int(warmup_sec * 1e9):
                        continue

                # Only the delivery time moves. The header stamp inside the
                # payload is untouched -- that is the whole point: de-skew and
                # LiDAR<->IMU sync stay exactly as the upstream converter left
                # them, and only rosbag2's playback ordering changes.
                shift_ns = int(round(float(tc.get("write_stamp_shift_sec", 0.0)) * 1e9))
                out_conn = out_conns[c.topic]
                writer.write(out_conn, timestamp + shift_ns, raw)
                counts[out_conn.topic] += 1

    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("in_bag", help="input ROS 2 bag directory")
    parser.add_argument("-o", "--out", "--output", dest="output", required=True,
                        help="output ROS 2 bag directory")
    parser.add_argument("-c", "--config", required=True, help="rebag YAML config")
    parser.add_argument("--limit-sec", type=float, default=None,
                        help="stop at N seconds from bag start")
    parser.add_argument("--start-sec", type=float, default=None,
                        help="skip messages before N seconds from bag start")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing output bag (only once the new one "
                             "is complete)")
    add_distro_arg(parser)
    add_group_arg(parser)
    args = parser.parse_args(argv)
    check_window(args.start_sec, args.limit_sec)
    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"output already exists: {args.output}")
    cfg = apply_distro(load_yaml(args.config), args.ros_distro)
    groups = select_groups(cfg, args.groups)
    print(f"groups: {', '.join(groups)}")
    counts = rebag(args.in_bag, args.output, filter_groups(cfg, groups), args.limit_sec,
                   args.start_sec, args.force)
    summary = ", ".join(f"{n} {t}" for t, n in counts.items())
    print(f"done: {summary} -> {args.output} [{output_summary(cfg)}]")


if __name__ == "__main__":
    main()
