# slam_dataset_pipeline
This repository is created for Research on datasets for lifelong localization challenge in unstructured dynamic construction environments

## File structure

```
SLAM_DATASET_PIPELINE
|   README.md
└───config
|   |   paths.yaml           # every absolute path on this machine, and nowhere else
|   └───dataset_configuration_folder
|   |   |   bag_conversion_config.yaml
|   |   |   SLAM_system_config.yaml
|   |   |   ...
|   └───dds                  # Fast DDS profile + bag-play QoS overrides (committed)
|   └───hilti
|       |   platforms.yaml   # rig calibration: prism arm, extrinsics, IMU rates
|       |   sequences.yaml   # which run each sequence's map and numbers come from
|
└───src
|   |   paths.py             # resolves config/paths.yaml (`python3 src/paths.py`)
|   |   to_bag.py            # dataset folder (per-frame files + CSV) -> ROS 2 bag
|   |   batch_to_bag.py      # run to_bag.py over every sequence under a root
|   |   rosbag1_to_ros2.py   # ROS 1 .bag -> ROS 2 bag
|   |   batch_fastlio.py     # run FAST-LIO over a batch of bags
|   |   pcd_convert.py       # batch PCD -> PLY (viewers) or LAS/LAZ (PDAL/build_map)
|   |   ros2_bag_io.py       # shared writer: typestore, tf_static, output config
|   |   pc2_layout.py        # shared PointCloud2 field/layout vocabulary
|   |   pcd_io.py            # shared PCD reader / PLY + LAS writer (no ROS dependency)
|   |   resource_log.py      # CPU/memory/disk sampling during a run -> CSV + chart
|   |   inspect_bag.py       # probe a bag to work out a converter config
|   |   read_tf_extrinsic.py
|   └───hilti                # Hilti 2023 evaluation: scoring, map levelling, report
|   |   |   dataset.py       # the layout and the rig facts, from config/hilti/*.yaml
|   |   |   gcp.py           # control-point pair-distance scoring
|   |   |   eval_fastlio.py  eval_localizer.py  eval_glim.py
|   |   |   align_map.py     # the rotation that levels a FAST-LIO map frame
|   |   └───report           # prep -> figures -> tables -> assemble -> audit
|   └───kdloc                # KdLocalizer test campaigns, any dataset (see below)
|       |   make_runs.py     # campaign.yaml -> one directory of inputs per run
|       |   monitor.py  procmon.py   # recorders used during a run
|       |   logs.py          # readers for everything a run leaves behind
|       |   evaluate.py      # status, accuracy (evo), relocalization, resources
|
└───scripts
|   └───hilti                # in_container.sh, run_seq.sh, build_maps.sh, run_glim.sh
|   └───kdloc                # campaign.sh (host), run_one.sh (container)
|   └───mapping              # build_map_headless / map_dump (compiled in the container)
|   └───fastlio              # the FAST-LIO source patch this pipeline expects
```

## Machine paths (`config/paths.yaml`)

No script in this repo contains an absolute path. Dataset roots, ROS setup files,
interpreters and container mounts all live in `config/paths.yaml`, read through
`src/paths.py`:

```bash
python3 src/paths.py                       # every key, resolved, missing ones flagged
python3 src/paths.py datasets.hilti_2023.root
```

The file is gitignored like the rest of `config/`;
`config/examples/paths.example.yaml` is the committed template **and** the
fallback, so a fresh clone runs before anyone copies it. Values can reference
environment variables (`${SDP_HILTI_ROOT:-...}`), other keys
(`${datasets.hilti_2023.raw}/groundtruth_2023`) and `${SDP_REPO}` (this
checkout), which is usually enough to move to another machine without editing the
file at all. `SDP_PATHS=/somewhere/paths.yaml` overrides the search.

An unresolvable key names itself and the variable it wanted, rather than failing
somewhere downstream as a file that is not there.

Both converters run on pure-python `rosbags` (`pip install rosbags pyyaml`); no
ROS environment needs to be sourced. `psutil` and `matplotlib` are optional —
they enable the [resource log](#resource-logging-cpu--memory--disk) and its
chart, and nothing else changes without them. They share their output layer, so a bag
built from loose files and one re-containered from a ROS 1 recording are
interchangeable.

## Inspect ROS bag

Introspect a ROS 1 or ROS 2 bag: topics, message types, rates, timestamp layout
and (for PointCloud2 / Imu) the per-message structure a converter config needs.

The goal is onboarding: point this at a new dataset's raw bag and it tells you
what you must declare in a ``bag_*.yaml`` config -- which topic is LiDAR / IMU,
the PointCloud2 field layout (does it carry ``time`` / ``ring``?), whether stamps
are absolute Unix time or sensor-relative, and where the timestamps cluster or
gap. It reads with the pure-python ``rosbags`` library (no ROS environment), and
``AnyReader`` auto-detects ROS 1 vs ROS 2.

Two passes:
  * cheap pass over every message using the bag receive-time (no deserialize) for
    per-topic count / rate / dt distribution / gaps;
  * a sample pass that deserializes the first ``--sample`` messages per topic to
    read structure (PointCloud2 fields, header presence, header-vs-bag-time
    offset, per-point time range, frame_id).
With ``--deep`` the sample pass deserializes *every* message so the header-stamp
dt / monotonicity / gap stats cover the whole bag (slow on large bags).

Examples:
    python3 src/inspect_bag.py "/path/ConSLAM/seq1/Copy of recording.bag"
    python3 src/inspect_bag.py /path/seq1_fastlio_bag --topics /velodyne_points /imu
    python3 src/inspect_bag.py /path/raw.bag --deep --json > seq1_probe.json

## Bag generation

Two converters, one output standard. Both take `-o <out bag>` and `-c <config>`,
both accept `--start-sec` / `--limit-sec` to convert a window, both refuse to
overwrite an existing output, and both run on pure-python `rosbags`
(`pip install rosbags pyyaml`) — **no ROS environment needs to be sourced**. Only
standard ROS 2 messages are written (`PointCloud2`, `Imu`, `Odometry`,
`TFMessage`), so every bag is self-describing to any consumer.

| script | input | use when |
|---|---|---|
| [src/to_bag.py](src/to_bag.py) | a folder of per-frame LiDAR files + CSV IMU/GT | ConPR, and most academic dataset dumps |
| [src/rosbag1_to_ros2.py](src/rosbag1_to_ros2.py) | an existing ROS 1 `.bag` | ConSLAM, Hilti |

```bash
python3 src/to_bag.py 20230531 -o 20230531_pc2_bag -c config/conpr/bag_conpr_pc2.yaml
python3 src/rosbag1_to_ros2.py recording.bag -o out_bag -c config/conslam/bag_conslam_fastlio.yaml
```

`rosbag1_to_ros2.py` keeps only the topics named in the config, optionally
renames topics / frame ids, and for Velodyne-style clouds rebases the per-point
`time` field so it starts at 0 (with the header stamp adjusted so absolute point
times are preserved). The result plays back just like a bag built by
`to_bag.py` from loose files.

Both write bags that **Humble and Jazzy can both read** — see
[ROS 2 distro compatibility](#ros-2-distro-compatibility-humble--jazzy).

They share the whole output half — typestore, the `output:` block, the
`tf_static:` schema, per-topic knobs (`frame_id`, `min_dt_sec`,
`write_stamp_shift_sec`), the `time_field` / `time_unit` / `ring_field`
vocabulary, and the named cloud layouts — so a bag built either way is
interchangeable downstream. The shared code lives in
[src/ros2_bag_io.py](src/ros2_bag_io.py) (writer, typestore, tf_static) and
[src/pc2_layout.py](src/pc2_layout.py) (PointField datatypes, vendor field-name
tables, layout presets).

### Additional Steps for ROS conversion

The [`rosbags` CLI](https://ternaris.gitlab.io/rosbags/topics/convert.html) does the
container change (ROS 1 `.bag` -> ROS 2), and that part is **not** reimplemented here —
`rosbag1_to_ros2.py` calls the same library for reading and writing. What the CLI cannot
do is fix the *contents*, and a bag whose container is valid but whose timestamps are
wrong will run FAST-LIO to a diverged trajectory with no error message. The per-dataset
work sitting on top of the CLI:

- **Timestamp epoch differs per dataset.** Hilti 2023 stamps are absolute Unix time;
  ConSLAM stamps are sensor-relative. Which one you have decides whether a rebase is
  safe, so [src/pc2_layout.py](src/pc2_layout.py) classifies each stamp (`epoch_kind`)
  and the converter refuses to rebase an absolute column — otherwise the header would
  jump forward by ~54 years.
- **The per-point `time` column's zero point differs.** Hilti's first point carries the
  scan-*start* time (and in absolute time, so the header stamp has to be subtracted out
  to recover a relative offset); ConSLAM's zero is the *middle* of the scan. FAST-LIO
  de-skews against this column, so getting it wrong smears every scan.
- **Field layout differs.** A Hesai cloud has to be repacked into the Velodyne layout
  (`x,y,z,intensity,time,ring`) that FAST-LIO and `kd_localizer` expect — a field-by-field
  rewrite, not a container change.
- **Unit inference.** A `time` field can be s / ms / us / ns; a wrong guess scales every
  stamp by 1000x silently, so the span is checked against a plausible scan duration.

Where a dataset needs none of the above, the CLI is the right tool and this repo adds
nothing — use it directly.

### Configs

Configs live under `config/<dataset>/`:

- `config/conpr/bag_conpr_pc2.yaml`: ConPR `.bin` -> `/livox/points` PointCloud2
  (`velodyne` layout) + `/livox/imu` + tf_static. **One bag for both** FAST-LIO
  and the kd_amr_libs `kd_localizer_node` — see below
- `config/conpr/fast_lio_conpr.yaml`: FAST-LIO params for ConPR (`lidar_type: 2`)
  The `livox_ros_driver2/CustomMsg` pair (`bag_conpr_livox_custom.yaml` +
  `fast_lio_conpr_livox_custom.yaml`, `lidar_type: 1`) is gone — see below
- `config/examples/bag_pcd_example.yaml`: annotated template for a new dataset
  (per-frame PCD + CSV IMU + CSV GT) — start here
- `config/conslam/bag_conslam_fastlio.yaml`: ConSLAM ROS 1 `recording.bag` -> ROS 2 bag with
  `/velodyne_points` + `/imu` (used by `src/rosbag1_to_ros2.py`, not `to_bag.py`)
- `config/conslam/fast_lio_conslam.yaml`: FAST-LIO params for ConSLAM (Velodyne)

### ROS 2 distro compatibility (Humble / Jazzy)

The `output:` block decides which distros can open the bag. Both converters
default to the format **both** Humble and Jazzy read, so nothing needs to be set:

```yaml
output:
  storage_id: sqlite3   # sqlite3 | mcap
  ros_distro: both      # both (default) | humble | jazzy
```

| `ros_distro` | rosbag2 metadata version | opens on |
|---|---|---|
| `both` (default) | 8 | Humble **and** Jazzy |
| `humble` | 8 | same as `both` |
| `jazzy` | 9 | Jazzy and newer only |

`--ros-distro {both,humble,jazzy}` on `to_bag.py`, `rosbag1_to_ros2.py` and
`batch_to_bag.py` overrides the config, so a config pinned to one distro can be
re-targeted without editing YAML:

```bash
python3 src/to_bag.py 20230531 -o 20230531_pc2_bag \
    -c config/conpr/bag_conpr_pc2.yaml --ros-distro jazzy
```

Each run reports what it wrote, so a bag's compatibility is not a guess:

```
done: 3 /velodyne_points, 30 /imu, 2 /tf_static -> out_bag [sqlite3, rosbag2 v8 (Humble+Jazzy)]
```

**Why version 8 is the default.** Version 9 — what Jazzy's own recorder writes —
stores each topic's `offered_qos_profiles` as a YAML *sequence*. Humble's metadata
parser reads that key as a *string*, and it neither falls back nor skips the
topic: the whole bag fails to open, on `ros2 bag info` and `ros2 bag play` alike,
for sqlite3 and mcap both.

```
Exception on parsing info file: yaml-cpp: error at line 25, column 29: bad conversion
```

Version 8 stores the same QoS as a YAML string with the rmw integer enums, which
is what Humble expects and what Jazzy still parses (its decoder keeps the string
path for `version < 9`). Nothing written here needs version 9: the storage file is
byte-identical either way and only `metadata.yaml` differs, and the message
definitions involved (`PointCloud2`, `Imu`, `Odometry`, `TFMessage`) are identical
between the Humble and Jazzy typestores. `/tf_static` also stays latched
(`transient_local`) under version 8 — verified with a late subscriber on Humble,
which is what `kd_localizer` relies on to receive the extrinsics after playback
has started.

Distros older than Humble are out of reach: they need metadata version ≤ 5, which
`rosbags` cannot write. `version: 8|9` is still accepted in place of
`ros_distro:` when you want to name the format directly; setting both is an error
only when they disagree.

### `to_bag.py` config schema

One `sources:` entry per output topic — the same shape as `rosbag1_to_ros2.py`'s
`topics:` list. There is no separate `dataset:` / `bag:` / `frames:` split any
more; input and output live together per topic.

```yaml
output:  {storage_id: sqlite3, ros_distro: both}  # sqlite3 | mcap; both|humble|jazzy
sources: [ {out, type, frame_id, ...}, ... ] # type: pointcloud2|imu|odometry
tf_static: [ {parent, child, translation + rotation_xyzw | matrix}, ... ]
```

Adding a dataset should mean writing YAML, not editing Python. The two things
that used to be hardcoded are now config:

#### Point clouds

Every reader emits the same canonical per-point names (`x, y, z, intensity,
time, ring, tag, line`, where `time` is always **seconds relative to the frame
stamp**), and every output layout is written in terms of those names — so any
input format pairs with any output layout.

```yaml
dir: LiDAR
pattern: "*.bin"
format: conpr_livox_bin    # conpr_livox_bin | pcd | xyzi_bin

stamp_from: filename       # the file stem IS the timestamp
stamp_unit: ns             # s | ms | us | ns
stamp_shift_sec: -0.000705
# ...or, when filenames are frame indices (000001.pcd):
# stamp_from: file
# stamp_file: timestamps.txt

time_field: timestamp      # source per-point column names; omitted -> the
ring_field: laser_id       # vendor alias tables in pc2_layout.py are searched
intensity_field: reflectance
time_unit: s               # unit of time_field

layout: velodyne           # velodyne | ouster | livox_pc2 | xyzi
```

Layout presets (`pc2_layout.LAYOUT_PRESETS`):

| preset | fields | for |
|---|---|---|
| `velodyne` | `x,y,z,intensity,time`(f32 s rel)`,ring`(u16) | FAST-LIO, kd_localizer — **the interchange layout**, identical to what `repack: hesai_to_velodyne` emits |
| `ouster` | `x,y,z,intensity,t`(u32 ns rel)`,ring`(u16) | Ouster-native consumers |
| `livox_pc2` | `x,y,z,intensity,time,tag`(u8)`,line`(u8) | keeps the Livox per-point flags |
| `xyzi` | `x,y,z,intensity` | consumers with no de-skew |

Or spell it out. `from` is the canonical source name and `name` is what gets
written, which is also how you rename a column:

```yaml
fields:
  - {name: x, type: float32}
  - {name: y, type: float32}
  - {name: z, type: float32}
  - {name: intensity, type: float32}
  - {name: t, type: float32, from: time}   # emit `t` rather than `time`
  - {name: ring, type: uint16}
point_step: 24        # defaults to the packed extent; set only to pad
```

All eight `sensor_msgs/PointField` datatypes are available (`int8`..`float64`),
so `uint16` ring and `float64` timestamp fields are expressible.

A time column's unit follows its type: **float** columns get seconds (Velodyne
`time`), **integer** columns get nanoseconds (Ouster `t`) — otherwise 0.1 s would
truncate to 0. Override per field with `unit: s|ms|us|ns`.

Per-point time is normalized on read: **absolute** Unix per-point time (Hesai)
is differenced against the frame stamp, **relative** time (Livox, Velodyne) is
only scaled — the same rule `repack: hesai_to_velodyne` applies to ROS 1 input.

#### Corrupt per-point time

`time_outliers:` handles frames whose per-point time cannot belong to the scan
(bound with `time_max_sec`, default 2.0 s):

| | |
|---|---|
| `keep` (default) | write as-is; the span check still warns |
| `drop_frame` | discard the whole frame — **use this for ConPR** |
| `drop` | discard only the out-of-range points |
| `signed` | reinterpret an underflowed **unsigned** column as signed |

**What this is for, and why `drop_frame`.** Some ConPR `.bin` files contain *two
concatenated scan segments with different time origins*. The second segment's
`offset` is negative relative to the file's nominal stamp, and since the field is
`uint32` it wraps to ~2³² — reading as ~4.29 s inside a 0.1 s scan. Left alone,
de-skew propagates up to 40% of such a frame 4.27 s forward.

A full scan of all 12 ConPR sequences (46,235 frames) found **6 affected frames**:

| sequence | frames | bad | bad points |
|---|---:|---:|---:|
| 20230623 | 4047 | 1 | 3709 of 9173 |
| 20230627 | 3376 | 1 | 3331 of 8948 |
| 20230706 | 3732 | 1 | 2145 of 8759 |
| 20230717 | 4382 | 3 | 157 / 2092 / 940 |

Every affected sequence also has LiDAR frame intervals under 50 ms (normal is
99.84 ms) and the unaffected eight have none — the root cause is a **recording
glitch that split a frame**, not a format or precision problem. The wrap itself is
lossless and exactly reversible; what is *not* recoverable is the second segment's
true origin, which is recorded nowhere.

`drop` is not enough: removing only the out-of-range points leaves 3 of the 6
frames with a 19–83 ms backward jump in the remaining sequence (across 1255
sampled healthy frames the largest backward jump is 1.80 ms), i.e. the two time
bases are still interleaved. `drop_frame` discards the frame outright — 6 frames
out of 46,235 is 0.013%.

#### CSV columns

`columns:` names each column of the file positionally (`_` skips one), so column
order and quaternion order are config rather than code:

```yaml
# IMU  — known names: t, qx, qy, qz, qw, wx, wy, wz, ax, ay, az
columns: [t, qx, qy, qz, qw, wx, wy, wz, ax, ay, az]   # ConPR
columns: [t, wx, wy, wz, ax, ay, az]                   # EuRoC (no orientation)

# Odometry / GT — known names: t, x, y, z, qx, qy, qz, qw
columns: [t, x, y, z, qx, qy, qz, qw]                  # TUM
columns: [t, x, y, z, qw, qx, qy, qz]                  # EuRoC state (qw first!)
```

Also `delimiter`, `skip_rows`, and `stamp_unit`. For ground truth,
`orientation: nearest_imu` borrows attitude from the IMU when the file is
position-only (ConPR `pose.csv`), and `publish_at: <topic>` resamples the poses
onto another source's stamps instead of using their own rows.

#### Guardrails

A wrong unit or column order used to produce a silently wrong bag. Now:

- frame stamps that are not plausible Unix time warn and name the likely cause
  (`stamp_unit`, or index filenames needing `stamp_from: file`);
- non-numeric filenames under `stamp_from: filename` are a hard error pointing
  at `stamp_file:`;
- a per-point time span that is not scan-shaped warns and suggests the unit that
  fits (shared with `rosbag1_to_ros2.py`);
- a layout field whose source column is absent is written as zeros **with a
  warning**, and a `ring` that is present but constant warns too;
- unknown `columns:` names, layout presets, point field types, `stamp_unit`s,
  `storage_id`s and `ros_distro`s are errors that list the valid values;
- a metadata `version:` that cannot be written (anything but 8 or 9) is an error
  rather than a silent downgrade, as is a `version:`/`ros_distro:` pair that
  disagrees about the target distro.

### Point-cloud layout keys (`src/rosbag1_to_ros2.py`)

The PointCloud2 handling is split into what the ROS spec fixes and what varies by
dataset. The fixed parts — the `sensor_msgs/PointField` datatype enum (1..8) and
byte order (read from each cloud's own `is_bigendian`) — live in
[src/pc2_layout.py](src/pc2_layout.py) and never need editing. The varying parts
are per-topic config keys:

```yaml
- in: /hesai/pandar
  type: pointcloud2
  time_field: timestamp   # per-point time field name
  time_unit: s            # s | ms | us | ns  (default s)
  ring_field: ring        # ring/beam field name (repack only, default `ring`)
```

Vendors disagree on both: the time field is `time` (Velodyne), `t` (Ouster),
`timestamp` (Hesai) or `offset_time` (Livox), and the ring field is `ring`,
`channel`, `line` or `laser_id`. Units differ too — the same `float32 time` may
hold seconds or nanoseconds.

A wrong `time_unit` would be off by a power of 1000 with no error, so the
converter checks the observed per-point span against a plausible scan duration
(0.005–2 s) and warns once per topic, suggesting the unit that does fit.
`repack: hesai_to_velodyne` additionally *rejects* per-point time that is not
absolute Unix time (use `rebase_point_time` for already-relative time), and
`rebase_point_time` rejects `count != 1` fields, which its in-place rewrite
cannot handle.

To discover the right values for a new dataset, probe first —
`src/inspect_bag.py` prints the full field table, guesses the unit from the value
span, and classifies absolute vs relative:

```
fields:    x@0:float32, y@4:float32, z@8:float32, intensity@16:float32, timestamp@24:float64, ring@32:uint16
has_time=True has_ring=True  time 'timestamp':float64 range=[1674573969.0866, 1674573969.1866] span=0.0999544 unit~s unix_absolute
```

Its candidate names come from the same defaults and are overridable per run:

```bash
python3 src/inspect_bag.py /path/odd.bag --time-fields stamp_ns --ring-fields beam
```

### Choosing what goes in the bag: topic groups

One config per consumer declares everything the dataset offers — clouds, IMU,
images, `camera_info`, control-point marks — and tags each entry with `groups:`.
What actually lands in the bag is a run-time choice, so "one bag with everything"
and "just the odometry topics" are the same config with a different flag, not two
configs that drift apart:

```yaml
include_groups: [lidar, imu]     # what a run converts when --groups is absent
topics:
  - in: /hesai/pandar
    groups: [lidar]
  - in: /alphasense/imu
    groups: [imu]
  - in: /alphasense/cam0/image_raw
    groups: [camera, cam0]       # in both, so `camera` takes all five cameras
  - in: /Point_name              # UNTAGGED = unconditional, always converted
tf_static:
  - {parent: imu_link, child: velodyne}          # untagged: every bag needs it
  - {parent: velodyne, child: pp_rgb, groups: [camera]}
```

```bash
--groups all                  # every group the config declares
--groups lidar,imu            # odometry only
--groups lidar,imu,cam0       # ... plus the front camera
--groups rgb                  # images only, no clouds or IMU
--groups none                 # untagged entries only (drop every optional group)
```

Both `rosbag1_to_ros2.py` and `rebag.py` take `--groups`. Details that matter:

- **Untagged entries are unconditional.** A config with no tags behaves exactly as
  it did before groups existed, and an `imu_link->velodyne` extrinsic does not get
  dropped just because the topics beside it grew tags.
- **An unknown group name is fatal**, listing what the config declares. A typo'd
  `--groups camrea` used to be indistinguishable from "this dataset has no images".
- **A selected topic that is not in the input is fatal too.** The case that makes
  this necessary: `bag_conslam_kdamr.yaml` selects the camera group by default, but
  `rebag.py` reads an already-converted bag, so re-timing a `_fastlio_bag` written
  with `--groups lidar,imu` used to yield a valid kdamr bag with no images in it and
  no complaint. The error names the missing topics with their groups; the fix is
  either to re-convert the ROS 1 source with the group selected, or to deselect it.
- **Group membership is per entry, so a camera and its extrinsic travel together** —
  see `tf_static` above.
- The resolved set is printed (`groups: lidar, imu`) before conversion starts.

Current defaults: the `*_fastlio.yaml` configs default to `[lidar, imu]`, because
adding 40 GB of images to a bag FAST-LIO plays only costs it disk bandwidth; the
`*_kdamr.yaml` configs default to including the cameras, since that is the bag a
vision module would consume. Either is one line in the config.

### Images and `camera_info` (`src/rosbag1_to_ros2.py`)

Images need no payload handling at all: `sensor_msgs/Image` has identical field
definitions in the ROS 1 and ROS 2 typestores, so declaring the topic is enough
and the message is copied through as-is.

```yaml
- in: pp_rgb/synced2points
  out: /camera/rgb/image_raw
  type: image
  frame_id: pp_rgb
  camera_info:
    from: /path/to/calib_rgb.yaml   # relative paths resolve against the CONFIG file
    format: ros_camera_info         # or `kalibr` (see below)
    # camera: cam0                  # kalibr only: which camera in a multi-cam file
    # out: /camera/rgb/camera_info  # default: the sibling of `out:`
```

`camera_info:` builds the intrinsics **from the calibration file, not from the
recording**, and writes one message per image carrying that image's stamp and
`frame_id`. Three reasons, in order of how much they hurt:

- ConSLAM's recorded `pp_rgb/camera_info` has an **all-zero `P`**. `P` is the 3×4
  projection matrix of the rectified image, and the standard ROS path projects
  with `P`, not `K` — `image_geometry::project3dToPixel`, `image_proc`, RViz's
  Camera display all compute `P·[X Y Z 1]` and divide by the third row, so a zero
  `P` sends every point to NaN. Its `K` and `D` are correct, but they are not what
  those consumers read.
- Hilti 2023 publishes **no `camera_info` topic at all**, so there is nothing to
  forward for that dataset in the first place — its intrinsics only exist in the
  separately-downloaded kalibr calibration.
- `rosbags` names the matrices `D/K/R/P` in its ROS 1 typestore and `d/k/r/p` in
  its ROS 2 one, so copying a recorded message across the two needs a
  field-by-field rebuild regardless.

One message per image rather than one latched message, because
`image_transport`'s `CameraSubscriber` pairs image and `camera_info` by **exact**
time; a single latched message would leave those consumers waiting forever. At
352 bytes each the cost is nil next to the images.

`R` is identity and `P` is `[K | 0]` — the ROS convention for an unrectified
monocular image, which is what these bags carry. A calibration file's own
`projection_matrix` is deliberately ignored: ConSLAM's is for undistorted *and
cropped* images (fx 2333.6 against `K`'s 2351.9), so applying it to the
full-resolution raw frame would bias every projected point. A calibration whose
declared size disagrees with the images is fatal, not a warning.

`format:` selects the parser; both readers live in `CAMERA_INFO_READERS`.

| `format:` | Layout | Datasets |
|-----------|--------|----------|
| `ros_camera_info` (default) | what ROS's own `camera_calibration` writes: `image_width`, 3×3 `camera_matrix`, `distortion_coefficients`, `plumb_bob`. One camera per file | ConSLAM (`calib_rgb.yaml`, `calib_nir.yaml`) |
| `kalibr` | `intrinsics: [fx, fy, cx, cy]`, `resolution: [w, h]`, 4 `distortion_coeffs`, `distortion_model`. **All** the rig's cameras in one file, keyed `cam0`…`camN` | Hilti 2023 (`calibration_cams.yaml`), ConPR |

Because a kalibr file holds the whole rig, `camera:` picks the entry — by kalibr
key (`cam0`) or by its own `rostopic` (`/alphasense/cam0/image_raw`). Prefer the
topic: it is the spelling that cannot drift away from the `in:` topic above it,
and with five look-alike 720×540 Hilti cameras in one file, pairing cam3's
intrinsics with cam0's images is the mistake worth making impossible. There is no
default — omitting `camera:` is an error, not a guess at `cam0`. A principal point
outside the declared resolution is fatal for the same reason.

`distortion_model` maps to the CameraInfo string: kalibr `radtan` → `plumb_bob`,
padded from 4 coefficients to plumb_bob's `[k1,k2,p1,p2,k3]` with `k3 = 0`;
kalibr `equidistant` → `equidistant`, i.e. Kannala-Brandt KB4 fisheye — **which
`image_geometry` does not implement**, so a consumer projects through
`cv2.fisheye` and treats `P = [K|0]` as raw-image, not rectified. `camera_model`
must be `pinhole`: kalibr's `omni` carries a fifth intrinsic (the mirror
parameter ξ) that a CameraInfo `K` cannot express.

### Batch conversion

[src/batch_to_bag.py](src/batch_to_bag.py) runs `to_bag.py` over every sequence
folder under a dataset root:

```bash
# every ConPR sequence -> ConPR/Converted_bag/<seq>_pc2_bag
python3 src/batch_to_bag.py --root /media/kudan/seagate-2tb/datasets/ConPR \
    -c config/conpr/bag_conpr_pc2.yaml

python3 src/batch_to_bag.py --root ... -c ... --dry-run          # check inputs only
python3 src/batch_to_bag.py --root ... -c ... --limit-sec 20     # 20 s of each, to smoke-test
python3 src/batch_to_bag.py --root ... -c ... --only 20230706 --force
```

Beyond a shell loop it does four things:

- **validates every sequence before converting any of them** — a full ConPR run
  is ~25 min, so a missing `imu.csv` should surface in the first second, not the
  twentieth minute. The required files are read off the config's `sources:`;
- **injects the shared calibration**. `<root>/Calibration/calibration_lidar.yaml`
  holds both the extrinsic (`T_lidar_imu0`) and the clock offset
  (`timeshift_to_imu0`), so they come from the dataset rather than being copied
  into every config — and at full precision instead of the 9-decimal rounding.
  It prints what it overrode; `--no-calibration` keeps the config's own values;
- **collects the warnings into a summary**, since they are what decides whether
  the bags are trustworthy and a long scroll buries them;
- **logs CPU / memory / disk usage** for the run — see below.

Each sequence runs in a subprocess. That is deliberate: it keeps one sequence's
crash from taking the batch with it, and it is what lets `--jobs` run conversions
concurrently.

#### Replacing an existing bag

Both converters refuse to overwrite an existing output unless given `--force`, and
`--force` never deletes anything up front: the new bag is built in a staging
directory beside the target and swapped in with a single rename only once it is
complete. So a config error, a full disk or a Ctrl-C halfway through leaves the
previous bag intact and no half-written directory behind — which also means a
failed run can simply be re-run.

`--jobs N` parallelises, but only helps on an SSD; ConPR lives on a spinning disk
where concurrent readers thrash, hence the default of 1.

#### Resource logging (CPU / memory / disk)

`batch_to_bag.py` samples CPU, memory and disk once a second for the length of
the run and writes them to `<out-dir>/logs/resources_<timestamp>.csv`, plus a
line chart beside it:

```bash
# on by default -- nothing to pass
python3 src/batch_to_bag.py --root .../ConPR -c config/conpr/bag_conpr_pc2.yaml

python3 src/batch_to_bag.py ... --no-resource-log        # off
python3 src/batch_to_bag.py ... --no-resource-chart      # CSV only, no PNG
python3 src/batch_to_bag.py ... --resource-interval 5    # sample every 5 s
python3 src/batch_to_bag.py ... --resource-log /tmp/run.csv

# single conversions are opt-in (the batch driver already covers its subprocesses)
python3 src/to_bag.py <seq> -o <bag> -c <config> --resource-log run.csv

# re-draw the chart from a CSV kept from an earlier run
python3 src/resource_log.py Converted_bag/logs/resources_20260804-101500.csv
```

The run ends with a one-line summary:

```
resources: peak CPU 176% of 1600%, peak RSS 0.39 GB, disk read 0.85 GB,
           written 1.11 GB, 223.0 GB free
```

What is measured, and why it is measured that way:

- **the conversion, not the machine.** CPU and RSS are summed over the driver
  *and every descendant*, since `to_bag.py` runs as a subprocess and `--jobs`
  runs several at once. Whole-machine CPU and memory are logged alongside as
  context — a slow batch is often someone else's load.
- **CPU includes children the kernel has already reaped** (`children_user` /
  `children_system`). Sampling only live processes silently loses any subprocess
  that starts and finishes between two ticks: on a run whose conversions each
  took about a second it reported a 46% peak where the true figure was 102%.
- **disk two ways** — throughput (`read_mb_s` / `write_mb_s`, from the process
  tree's block-layer counters) answers "is the disk the bottleneck"; free space
  on the output volume answers "will this batch finish".
- rows are written unbuffered, so a run killed at minute 20 still leaves a
  usable log.

The chart is one panel per measure (never two y-scales on one plot), banded and
labelled per sequence in serial mode, so a spike can be attributed to the
sequence that caused it. `pip install psutil` is required for sampling and
`matplotlib` for the chart; without either the conversion runs exactly as before
and says what it skipped.

### ConSLAM (ROS 1 bag source)

ConSLAM ships each sequence as a single ROS 1 `recording.bag` (Velodyne VLP-16
LiDAR + Xsens 9-axis IMU + RGB/NIR cameras), not as per-frame files, so it uses
`rosbag1_to_ros2.py` rather than `to_bag.py`. FAST-LIO only needs LiDAR + IMU.

```bash
python3 src/rosbag1_to_ros2.py \
  "/media/kudan/seagate-2tb/datasets/ConSLAM/seq1/Copy of recording.bag" \
  -o /media/kudan/seagate-2tb/datasets/ConSLAM/seq1_fastlio_bag \
  -c config/conslam/bag_conslam_fastlio.yaml
```

Source topics -> output topics:

- `pp_points/synced2rgb` (`sensor_msgs/PointCloud2`, Velodyne VLP-16 layout
  `x,y,z,intensity,ring,time`) -> `/velodyne_points`
- `imu/data` (`sensor_msgs/Imu`, ~190 Hz) -> `/imu`
- `pp_rgb/synced2points`, `pp_nir/synced2points` (images) -> `/camera/rgb/image_raw`,
  `/camera/nir/image_raw` — declared in the same config but **not** in its default
  groups, since they are ~60 GB of the 63 GB and FAST-LIO subscribes to neither.
  Add them with `--groups all` (see below).

The converter does two data cleanups the raw bag needs:

1. **Point time rebasing** — each cloud's per-point `time` is centered on the
   header stamp (~-0.049..+0.052 s); it is rebased to start at 0 and the header
   stamp is shifted back by the same amount (absolute point times, and thus
   LiDAR<->IMU sync, are preserved), which is what FAST-LIO's Velodyne handler
   expects.
2. **IMU de-clustering** (`respace_min_dt_sec`) — the IMU is delivered in bursts
   (4-5 samples within microseconds, then a ~11 ms gap); the near-zero
   intra-burst dt breaks FAST-LIO's integration, so timestamps are re-spaced to
   be >=2 ms apart.

Then run FAST-LIO with `fast_lio_conslam.yaml` (see below).

> **Known limitation — sparse, gappy LiDAR.** `pp_points` is the RGB-*synced*
> LiDAR subset, so it is only ~4.9 Hz (not the sensor's 10 Hz) with frequent
> gaps up to ~1.5 s (the raw 10 Hz stream is not in `recording.bag`). FAST-LIO
> dead-reckons on IMU across each gap; with the sparse 16-ring VLP-16 this is
> marginally stable — it tracks cleanly for the first ~80-130 s (a good ~50 m of
> trajectory) and can then lose lock and drift. This is a property of the data,
> not the converter. If you need robust full-sequence tracking you would need
> the raw 10 Hz LiDAR (re-record from the ConSLAM sensor pipeline) or a
> LiDAR-only / loop-closing SLAM more tolerant of gaps.

> **seq2 caveat:** the seq2 `recording.bag` in this copy is a broken HTML
> download (277 bytes) and its `lidar-pcd`/poses are XYZ-only with no IMU, so
> seq2 **cannot** run in FAST-LIO until a valid `recording.bag` is re-downloaded.
> FAST-LIO is LiDAR-*inertial* and needs the IMU stream.

> **Extrinsic:** `fast_lio_conslam.yaml` uses the Velodyne->Xsens rotation from
> ConSLAM's `calib_lidar2imu.txt` (`data_calib`, 211021_seminar_room result;
> RPY -0.0251 0.0017 3.1104 — the LiDAR is mounted ~178 deg about z from the
> IMU). The calibration is rotation-only, so translation is left at 0.

#### ConSLAM images (a third bag)

For a vision / semantic-segmentation module, the images come from the same config
via `--groups` — merged into one bag, or on their own:

```bash
# clouds + IMU + RGB (+ camera_info + camera extrinsic) in one bag
python3 src/rosbag1_to_ros2.py .../seq2_recording.bag -o .../seq2_all_bag \
  -c config/conslam/bag_conslam_fastlio.yaml --groups all --limit-sec 60

# images only, e.g. to keep the odometry bag lean and merge later
python3 src/rosbag1_to_ros2.py .../seq2_recording.bag -o .../seq2_images_bag \
  -c config/conslam/bag_conslam_fastlio.yaml --groups rgb --limit-sec 60
```

Measured on seq2: `pp_rgb/synced2points` is 2064×1544 `rgb8`, **9.56 MB per
frame**, 4169 frames at 10 Hz (the LiDAR-synced subset of a 48 Hz camera) — 39.9 GB
converted, and ~96 MB/s of playback bandwidth off the same external drive FAST-LIO
reads its clouds from. That is why images are not in the `*_fastlio.yaml` default
groups: on an odometry run they compete for I/O with the estimator and buy it
nothing. `pp_nir/synced2points` (2592×1944 `mono8`, 5.04 MB/frame, 20.0 GB) is the
`nir` group, in `--groups all` but not in `--groups rgb`.

> **Play these bags with a SMALL `--read-ahead-queue-size`.** The queue counts
> *messages*, and an RGB frame here is 9.56 MB, so the value the odometry bags use
> (`batch_fastlio.py` defaults to `10000`, `run_fastlio_bags.sh` likewise) asks
> rosbag2 to buffer on the order of 95 GB. Measured on this machine (15 GB RAM) with
> `2000`: `ros2 bag play` sat in uninterruptible I/O wait (process state `D`) for 13
> minutes, having read 5 MB and published nothing, and starved every consumer. Use
> `--read-ahead-queue-size 30` (~285 MB) for image-bearing bags; pass
> `--queue-size 30` to `batch_fastlio.py` when running FAST-LIO on a `*_cam_bag`.

With the cameras selected the bag carries the three things a vision module needs:
the images, the intrinsics (one `camera_info` per frame — see
[Images and `camera_info`](#images-and-camera_info-srcrosbag1_to_ros2py)) and the
LiDAR→camera extrinsic on `/tf_static`, written `velodyne -> pp_rgb` so it composes
with `imu_link -> velodyne` into a single `imu_link -> velodyne -> pp_rgb` tree.
`bag_conslam_kdamr.yaml` includes the cameras **by default** and shifts them by the
same `write_stamp_shift_sec: 0.11` as the clouds, so their delivery relationship
survives the re-timing.

> **Extrinsic direction was verified, not assumed.** `calib_lidar2rgb.txt`
> describes its 4×4 only as "the inverse of T^{LiDAR}_{RGB}". Composing the two
> `static_transform_publisher`s in the accompanying `calib_lidar2rgb.launch`
> (through their shared `rotated_pp_rgb` frame) reproduces that matrix to 5e-9, so
> it is `T_pp_rgb_velodyne` — it maps points *from* the velodyne frame *into* the
> camera. tf wants the opposite sense, hence `matrix_is: child_to_parent`. The
> inverse puts the camera at [0.0218, -0.0067, -0.0632] m in the velodyne frame,
> ~6 cm below the LiDAR, matching the published rig photo.

> **Image↔cloud stamps shift by 48 ms.** The cloud topic gets
> `rebase_point_time`, which moves each cloud's header stamp back ~48 ms; image
> stamps are untouched. Absolute times stay correct on both sides, but raw RGB
> *leads* its cloud by ~6 ms while converted RGB *trails* the rebased cloud header
> by ~42 ms. Anything pairing image to cloud by header equality has to know.

> **Play at real time (`-r 1.0`).** Each VLP-16 scan is ~27k points; FAST-LIO
> cannot keep up faster than real time, and `ros2 bag play --clock` at higher
> rates floods it and makes the estimate diverge. `run_fastlio_bags.sh` defaults
> to `-r 1.0`; do not raise it for ConSLAM.

### ConPR: PointCloud2 rather than Livox CustomMsg

ConPR used to be converted to `livox_ros_driver2/CustomMsg` and run with
`lidar_type: 1`. It now uses a plain PointCloud2 in the `velodyne` layout with
`lidar_type: 2`, the same as ConSLAM and Hilti.

**The `CustomMsg` writer has been removed** (`type: livox_custom`,
`build_livox_custom()`, the embedded `livox_ros_driver2` message definitions and
the `offset_time_unit` / `lidar_id` / `rsvd` keys). The reasoning below is why.
Existing `*_custom_bag` bags still play and still inspect — rosbag2 embeds each
type's `.msg` text in the bag, so `AnyReader` registers `CustomMsg` from the bag
itself with nothing sourced. Only the *input* reader `format: conpr_livox_bin`
(the raw `.bin` struct) remains, and it is unrelated: it feeds the PointCloud2
path like any other reader.

FAST-LIO's two handlers (`preprocess.cpp`) differ in exactly two ways, and
neither applies to ConPR:

| | `avia_handler` (CustomMsg) | `velodyne_handler` (PointCloud2) |
|---|---|---|
| noise filter | `(tag & 0x30) == 0x10 \|\| == 0x00` | none |
| scan lines | buckets by `line` | uses `ring`, drops `ring >= scan_line` |
| per-point time | `offset_time / 1e6` (ns -> ms), always | `time * time_unit_scale` |

Measured on `20230531`: the tag filter rejects **0 of 229,632** points, and
`line` is **always 0**. So the Velodyne path is equivalent here, and it buys one
cloud layout across all datasets, no `livox_ros_driver2` dependency, and bags
that GLIM / KISS-ICP / rviz can read directly.

Two prerequisites of `velodyne_handler`, both verified on this data:

- it sets `given_offset_time` from the **last** point's `time` being > 0. If that
  were false, FAST-LIO would reconstruct per-point time from yaw assuming a
  *spinning* lidar, which is meaningless for Livox's non-repetitive pattern.
  Measured over 1799 frames sampled across all 12 sequences: the last point is
  the frame's time maximum (94.5–100.5 ms) in **1799/1799**. Note the offsets are
  *not* strictly monotonic — 38 frames (2.1%, concentrated in `20230628` and
  `20230818`) carry one local backward step of 0.1–0.5 ms — but it never lands on
  the last point, and de-skew reads each point's own `time` rather than the
  storage order, so neither the sniff nor the geometry is affected;
- it drops `ring >= scan_line`, so `scan_line` must stay >= 1 (ConPR ring is 0).

The one capability given up is `avia_handler`'s `tag` noise filter, which needs
`CustomMsg` (`velodyne_handler` ignores `tag` even when the `livox_pc2` layout
carries it). It would matter for a Livox recording with rain, dust or drag
returns — not for ConPR, and not at all for kd_localizer, which has no CustomMsg
path. If such a dataset ever turns up, restore `build_livox_custom()` from git
history rather than reinventing it.

#### One bag for FAST-LIO and kd_localizer

`bag_conpr_pc2.yaml` is built to serve both, so ConPR needs only one conversion.
`kd_amr_libs` has no CustomMsg path at all — `localizer_node` subscribes
`sensor_msgs/msg/PointCloud2` only, `ros_conversions` has a single
`LidarFrame from_msg(const PointCloud2&)`, and `kd_amr_ros/package.xml` does not
depend on `livox_ros_driver2` (even `kd_amr_handheld`'s launch runs the vendor
driver with `xfer_format: 0`, i.e. PointCloud2). So PointCloud2 is not a
preference there, it is the only option.

The two extras kd_localizer needs are inert for FAST-LIO:

| | kd_localizer | FAST-LIO |
|---|---|---|
| `tf_static` (`imu_link`->`livox_frame`) | resolves `base_frame_id`->lidar/imu from tf | never subscribes tf; uses `extrinsic_T`/`extrinsic_R` |
| `write_stamp_shift_sec: 0.11` | needs the scan's IMU delivered first, else "missing IMU samples" every frame | `sync_packages()` already blocks on `last_timestamp_imu < lidar_end_time`, so a later delivery only removes retries |

Both consumers also auto-detect the per-point time the same way: FAST-LIO wants a
float32 `time`, and kd_localizer's `detect_point_time_descriptor()` accepts
`time`/float32 as a header-relative offset (it checks the field type only, never
the values — unlike FAST-LIO's last-point sniff above). The `velodyne` layout
satisfies both.

Set `base_frame_id: imu_link` for kd_localizer: ConPR's
`Calibration/calibration_imu.yaml` has `T_i_b` = identity (base == IMU), so the
single `imu_link`->`livox_frame` transform covers base->lidar and leaves
base->imu identity. No warmup trim is needed either — `imu.csv` starts
0.207–0.647 s before the first LiDAR file in all 12 sequences.

### Adding a new dataset

1. Probe the source. For a ROS 1 bag, `python3 src/inspect_bag.py <bag>` prints
   the field table, guesses the per-point time unit, and says whether the time is
   absolute or relative. For loose files, read a PCD header or dump the first
   record.
2. Copy [config/examples/bag_pcd_example.yaml](config/examples/bag_pcd_example.yaml)
   and set: `dir`/`pattern`/`format`, how frame stamps are obtained
   (`stamp_from`), the source field names and units, `columns:` for each CSV, and
   the output topics/frames.
3. Convert a short window first — `--limit-sec 20` — and read the warnings; they
   are there to catch exactly the unit/column mistakes that otherwise show up
   later as a diverging trajectory.
4. Check the result with `src/inspect_bag.py` on the bag you just wrote.

Prefer `layout: velodyne` unless a consumer needs something else — it is the one
layout both converters emit identically, so FAST-LIO/kd_localizer configs carry
over between datasets unchanged.

## PCD -> PLY / LAS

The clouds this pipeline produces are PCDs — FAST-LIO writes one
`fastlio_map.pcd` per run, several datasets ship per-frame `.pcd` scans — but
nothing downstream reads PCD. [src/pcd_convert.py](src/pcd_convert.py) converts a
whole directory into either of the two formats that are actually wanted, chosen
with `--format`:

```bash
# one directory of maps -> <same dir>/<name>.ply
python3 src/pcd_convert.py /media/kudan/seagate-2tb/datasets/hilti_2023/maps

# every sequence's FAST-LIO map, each PLY beside its own map (12 in ~3 s)
python3 src/pcd_convert.py /media/kudan/seagate-2tb/datasets/ConPR/Converted_bag --recursive

# the same maps as LAS, for build_map's PDAL loader
python3 src/pcd_convert.py <dir> --recursive --format las

# per-frame scans into one directory, coordinates + intensity only
python3 src/pcd_convert.py "<ConSLAM>/seq1/lidar-pcd" -o /tmp/ply --fields x y z intensity

python3 src/pcd_convert.py <dir> --dry-run     # fields + point counts only
python3 src/pcd_convert.py <dir> --ascii       # to eyeball the numbers
```

Note the second form: the per-run maps are all called `fastlio_map.pcd`, so adding
`-o <one dir>` is *refused* rather than letting one of them overwrite the rest —
without `-o` each output lands next to its own map, which keeps the sequence name
in the path.

### Which format

**PLY (default) to look at a cloud.** CloudCompare, MeshLab, Open3D and Blender
all read it, and it is schema-free: any property name, any of eight scalar types,
written verbatim. The conversion is a lossless re-container.

**LAS/LAZ to feed PDAL** — which is the only point-cloud reader `build_map` has
(its other loader is a *mesh* loader, and a face-less PLY segfaults it). LAS is a
survey format, and it charges for that in two ways the tool makes explicit rather
than silent:

- **coordinates are quantised.** LAS stores `int32 * scale + offset`, so the header
  scale *is* the resolution of the file. `--las-scale` defaults to `0.001` (0.5 mm
  worst-case error), not PDAL's own `0.01` — a 1 cm grid is coarser than the voxel
  sizes these maps get built and compared at, and checking a map should not add a
  layer of error on top of the one being measured. The error is printed before any
  file is written. The offset is per-axis `floor(min)`, which is what keeps a
  georeferenced cloud inside the int32 at 1 mm steps: 2147 km of range around the
  offset, but only 2.1 km around zero. A cloud too large for the scale is refused
  with the scale that *would* fit, rather than laspy's bare `OverflowError`;
- **the dimension list is fixed.** x/y/z, intensity and RGB have a home in the
  point record. The `normal_x`, `ring`, `t`, `curvature` and `label` fields these
  clouds actually carry become **ExtraBytes** dimensions — kept under their own
  name and exact numpy type (extra bytes even carry an `int64` that PLY has to
  widen to a lossy `double`), but a plain scalar rather than something a viewer
  reads as a normal or a colour. Which fields those are is reported once, up
  front, and shows in `--dry-run`. Intensity becomes a `uint16`, saturating rather
  than wrapping (`astype` would turn 70000 into 4464), and colour is widened from
  PCD's 8 bits to LAS's 16.

The header is LAS 1.4 with legacy point format 0 (or 2 with colour), and both
halves of that are forced by PDAL 2.6.2: it will not read the ExtraBytes VLR out
of a 1.2 file — a `ring` column silently disappears — and it refuses point formats
6–10 outright unless the global-encoding WKT bit is set, which these clouds have
no CRS to justify. LAS output needs `laspy` (`pip install '.[las]'`); `--format
laz` additionally needs a compression backend (`'.[laz]'`), checked at startup
rather than on the first of 1400 files.

### Either way

It needs no PCL or PDAL install and shares its PCD reader with `to_bag.py` via
[src/pcd_io.py](src/pcd_io.py), so a cloud is parsed the same way whether it ends
up in a bag or in a file. Beyond a loop over `pcl_pcd2ply` / `pdal translate`:

- **all inputs are checked before any is converted.** PCD headers are cheap to
  read, so a missing field, a `binary_compressed` payload, or two inputs whose
  names would collide in one output directory surface in the first second —
  `--recursive` over a set of runs, every one of which holds a `fastlio_map.pcd`,
  is refused rather than silently letting the last one win;
- **names are translated, not dumped verbatim.** PCL's `normal_x` is PLY's `nx`,
  and a packed `rgb` field becomes red/green/blue, so a viewer shows normals and
  colour instead of anonymous scalars. For PLY, fields keep the width the header
  declares — a `U1` intensity stays one byte;
- **NaN/Inf coordinates are dropped** and the count reported. `--keep-nonfinite`
  keeps them in a PLY (one such point ruins a viewer's bounding box and every
  distance computed from it, so it is worth seeing); for LAS it is refused,
  because an integer coordinate has nothing a NaN can become;
- **outputs are replaced only once complete**, as with the bag tools: existing
  files need `--force`, and each file is written beside its target and renamed
  into place, so a Ctrl-C on a 10-million-point map leaves the old file usable;
- **flags that do not apply to the format are refused**, not ignored — `--ascii
  --format las` looks like it asked for something.

PLY output is `binary_little_endian` by default — ~4x smaller and much faster to
write and load than ascii. `binary_compressed` PCD input is not supported; convert
it first with `pcl_convert_pcd_ascii_binary`.

## Running FAST-LIO

`src/batch_fastlio.py` takes `--config-file` / `--config-path`. `--config-path`
defaults to this repo's `config/` dir, and `--config-file` is relative to it, so
just name the per-dataset file:

```bash
python3 src/batch_fastlio.py \
  --dataset conpr \
  --config-file conpr/fast_lio_conpr.yaml \
  --glob '*_pc2_bag'
```

(`run_fastlio_bags.sh` was the original and is retired — the Python runner has the
same per-bag sequence and outputs, plus everything below.)

`--dataset NAME` takes the root from `datasets.<NAME>.root` in `config/paths.yaml`;
`--root DIR` still overrides it for a one-off.

### The transport, and why a run can now fail on frame count

Every run exports `FASTRTPS_DEFAULT_PROFILES_FILE=config/dds/fastdds_shm_big.xml`
into the environment the node **and** `ros2 bag play` inherit. Without it this host
delivers about half the clouds: Fast DDS carries intra-host payload over shared
memory, the default segment cannot hold a stream of 1.37 MB samples at 10 Hz, and
the **writer** drops rather than waits — kernel UDP counters stay at zero
throughout, and `SensorDataQoS` on the subscriber side is not the cause. Measured
on `site2_handheld_5`: 2049/2053 clouds with the profile, 1087/2053 without.

Because that loss is silent — the map just thins out and the trajectory still
looks plausible — every bag is checked afterwards: `pos_log.txt` rows against the
bag's own cloud count, and **the bag fails** below `--min-frame-pct` (default 95).

```
frames 926/929 (99.7%) of /velodyne_points                     # ok
frames 457/929 (49.2%) of /velodyne_points -- BELOW 95%: ...   # FAILED
```

`--dds-profile none` runs on the stock transport (expect the loss),
`--dds-profile <xml>` picks another one, `--min-frame-pct 0` disables the check,
and `--play-qos-overrides auto` additionally hands the player
`config/dds/qos_lidar.yaml` (`keep_all`, so it blocks instead of overwriting).
The profile only applies to Fast DDS; with another RMW the runner warns and
carries on.

### Keeping a sweep out of the canonical outputs

`--run-name NAME` writes that run's map, logs and odometry under
`<bag>/runs/NAME/` instead of `<bag>/`, with the map named
`<sequence>_fastlio_map.pcd`:

```bash
python3 src/batch_fastlio.py --dataset hilti_2023 --glob '*_fastlio_bag' \
  --config-file hilti/fast_lio_hilti.yaml --only site2_handheld_5 \
  --run-name qosfix_r0.3 --no-rviz
```

Without it a second run silently overwrites the map and the `pos_log.txt` an
evaluation or a report was built from. `config/hilti/sequences.yaml` records which
run each Hilti sequence's numbers come from, and the evaluators read it, so a map
and its trajectory cannot drift apart.

`--root` is the **dataset** root — the directory holding the `<sequence>/`
folders, because that is where `pose.csv` is looked up for `--eval-odom`. Bags are
searched for in `<root>/Converted_bag` first (where `batch_to_bag.py` writes them),
falling back to `<root>` itself for the older layout where bags sit next to the
sequences. `--bag-root` overrides both. Getting this wrong used to be silent:
`--root <dataset>` found no bags at all, and `--root <dataset>/Converted_bag`
found them but then skipped every evaluation.

The Python runner is the one to prefer for long batches:

- a bag that fails (bad playback, `/map_save` timeout, node crash at startup)
  is reported and the batch continues; the run ends with a pass/fail table and a
  non-zero exit status. The shell script runs under `set -e`, so the first
  failure aborts every sequence still queued;
- `--start-from NAME` / `--only NAME ...` resume or subset a batch by bag or
  sequence name (`run_fastlio_bags.sh` only has `--start-from`);
- `--dry-run` prints the bag list and the resolved output paths;
- ROS overlays are options (`--ros-setup`, repeatable `--overlay`, both
  defaulting to `config/paths.yaml`) and a missing one is a warning, not a hard
  stop — `ws_livox` was only ever needed for `CustomMsg` bags, never for a bag
  these converters write now;
- `/map_save` is called directly rather than gated on `ros2 service list`. The
  enumeration APIs are not dependable: on this machine `ros2 service list` returns
  nothing at all — not even the CLI node's own services — while `service call
  /map_save` succeeds and `ros2 bag record <topic>` subscribes in 0.2 s. The old
  pre-check turned that into `/map_save never appeared` for every bag in a batch;
- `--eval-odom` uses this repo's `src/export_odometry_tum.py` /
  `src/evaluate_pose_csv_tum.py` (override with `--eval-dir`), rather than the
  shell script's `<root>/process/` copies. They are run with `--python`
  (default `ros.python` from `config/paths.yaml`) because they import `rclpy` /
  `rosbag2_py`.

Bags always run one at a time: FAST-LIO is a single ROS graph driven by
`/clock`, so two at once would interleave into one map.

(The ConPR default is `conpr/fast_lio_conpr.yaml`.)

For ConSLAM, the bag `seq1_fastlio_bag` sits at the dataset root, so it is
auto-discovered by the `*_fastlio_bag` glob.

Each map is written next to the bag it came from, as
`<bag>/maps/fastlio_map.pcd`. The `map_file_path` in the per-dataset configs is a
placeholder: the runner stages a temp copy of the config per bag with that key
rewritten, because FAST-LIO reads it once at startup and `mapping.launch.py`
takes no parameter overrides. Pass `--map-dir DIR` to also collect every map into
one directory as `<sequence>_fastlio_map.pcd`.

`--glob` is a glob matched against directory basenames one level under `--root`,
so it needs the wildcard (`'*_custom_bag'`, not `'_custom_bag'`) and does not
descend into nested layouts. Naming bags as positional arguments skips discovery
entirely. The Python runner also derives each sequence name (the sibling folder
holding `pose.csv`) by stripping the glob's suffix, so `--glob '*_fastlio_bag'`
maps `seq1_fastlio_bag` to `seq1`.

### Per-bag outputs

Everything a run produces is collected next to the bag:

```
<bag>/maps/fastlio_map.pcd   map, written directly by FAST-LIO
<bag>/fastlio_log/           pos_log.txt, mat_pre/mat_out/dbg/imu.txt, fast_lio_time_log.csv
<bag>/ros_logs/              ROS 2 launch and node logs
<bag>/results/               recorded odometry bag, .tum export, eval report
```

`results/` only appears with `--record-odom` / `--eval-odom`. `--odom-dir` and
`--ros-log-dir` override their defaults to collect every bag into one shared
directory instead.

`fastlio_log/` is a copy: FAST-LIO's `ROOT_DIR` is baked in at compile time
(`-DROOT_DIR=<source dir>/`), so it always writes to `Log/` in its own source
tree and each bag overwrites the last. The runner clears those files before a run
and copies them out after, so a batch keeps one set per bag. They are only
populated when `runtime_pos_log_enable: true`.

Because the time log is written after `rclcpp::spin()` returns and the rest is
flushed in the node destructor, the runner shuts FAST-LIO down with SIGINT and
waits up to 10s before escalating. Killing the batch harder than that loses the
current bag's logs.

Note that `pcd_save.pcd_save_en` in the configs is a dead setting: the block that
would accumulate scans and write `PCD/scans.pcd` is commented out in
`laserMapping.cpp`, so the only point cloud produced is the map from the
`/map_save` service.

## Hilti 2023 evaluation (`src/hilti/`)

The Hilti SLAM Challenge 2023 sequences are scored differently from the other
datasets here, because they ship no dense ground truth: 2–8 **surveyed control
points** per sequence, and nothing else. So the metric is the distance between
pairs of control points, which needs no trajectory alignment and is therefore
independent of whatever frame an estimate lives in (`src/hilti/gcp.py`).

Everything that is not code is config:

| file | what it holds |
| --- | --- |
| `config/paths.yaml` | `datasets.hilti_2023.{root,raw,groundtruth,kdm,results}` |
| `config/hilti/platforms.yaml` | rig calibration: prism lever arm, LiDAR↔IMU extrinsic, IMU rate, the `imu_link → velodyne` transform for the bags rebagged without `/tf_static` |
| `config/hilti/sequences.yaml` | the 15 sequences, their platform and site, whether a `.kdm` exists, and **which run** each one's map and numbers come from |
| `config/hilti/fast_lio_hilti*.yaml` | FAST-LIO configs (incl. the `dense` / `bacc` / `probe_*` sweeps) |
| `config/glim/hilti_headless*` | GLIM configs for the `glim_rosbag` comparison runs |

`python3 src/hilti/dataset.py` prints the resolved layout and every sequence's
run; add a sequence name for that one's paths.

### One sequence, end to end

```bash
# 1. FAST-LIO, into its own run directory (see --run-name above)
python3 src/batch_fastlio.py --dataset hilti_2023 --glob '*_fastlio_bag' \
  --config-file hilti/fast_lio_hilti.yaml --only site2_robot_1 \
  --run-name r0.3 -r 0.3 --no-rviz

# 2. score it: control points + health gates + frame rate
python3 src/hilti/eval_fastlio.py site2_robot_1 --json auto

# 3. measure how far that map frame is from vertical, and emit R_align
python3 src/hilti/align_map.py site2_robot_1

# 4. localizer config + node params for the sequence
python3 src/hilti/gen_cfg.py site2_robot_1

# 5. cloud -> levelled LAS -> .kdm, in the mapping container
SERVICE=mapping scripts/hilti/in_container.sh \
  /opt/sdp/scripts/hilti/build_maps.sh site2_robot_1

# 6. replay the bag against that map, in the release container
scripts/hilti/in_container.sh /opt/sdp/scripts/hilti/run_seq.sh site2_robot_1

# 7. score the localizer run against the survey and against FAST-LIO
python3 src/hilti/eval_localizer.py site2_robot_1 --json auto
```

Steps 5–6 need the Humble release container (the release binaries cannot run on a
Jazzy host). `scripts/hilti/in_container.sh` composes the `docker compose run`
command from `config/paths.yaml` and mounts this checkout at `$SDP`, the results
directory at `$W` and the bag tree at its host path, so the in-container scripts
have no absolute paths either.

### Why the map has to be levelled first

kd_localizer seeds its ESKF gravity at the **map frame's** −Z
(`kd_amr_libs/localization/src/motion_model/imu.cpp:139`). A FAST-LIO map frame is
the IMU frame at the first processed scan, so it inherits the rig's mounting: 88–92°
off vertical on the Hilti robot (XSens mounted x-up) and 177–180° on the handheld rig
(Alphasense z-down). At 180° the 2-DOF sphere parameterisation is at its antipode and
the correction gradient vanishes — the pose diverges in a perfectly good map with a
0.87 ray inlier ratio. `align_map.py` measures the tilt from the filter's own gravity
state, asserts the Euler round-trip, and writes both the PDAL matrix `build_maps.sh`
rotates the cloud with and the matching `initial_pose.rotation` for the config.

### Diagnostics and the report

```bash
python3 src/hilti/diag_gcp.py site2_handheld_4      # split a pair error into vertical/horizontal
python3 src/hilti/grav_drift.py <pos_log.txt> ...   # how far the gravity state and |ba| wandered
python3 src/hilti/eval_glim.py site2_handheld_5     # the same three checks for a GLIM run
./src/hilti/report/build.sh                         # data -> figures -> tables -> page -> audit
```

`build.sh` runs the whole chain in dependency order and ends with `audit.py`, which
re-checks every number quoted in the hand-written prose (`src/hilti/report/prose/`)
against the regenerated `data.json`. A `CHECK` line means the prose and the data
disagree and one of them has to change.

## KdLocalizer test campaigns (`src/kdloc/`, `scripts/kdloc/`)

A campaign replays recorded bags through `kd_localizer_node` and `kd_relocalizer_node`
against a prior map, one run at a time, and scores what the localizer reported. The
machinery around the localizer is the same for every dataset; this part of the pipeline
holds it, so a new dataset only supplies its inputs:

| the dataset supplies | what it is |
| --- | --- |
| `<campaign dir>/campaign.yaml` | topics, frames, static transforms, maps, bags, initial poses and the groups of runs. Annotated template: [config/examples/kdloc_campaign.example.yaml](config/examples/kdloc_campaign.example.yaml) |
| `<campaign dir>/compose.yaml` | the localizer container. The campaign directory and this checkout must be mounted at the same paths as on the host |
| one reference trajectory per sequence | TUM, in the map frame, each row stamped at the instant it describes on the clock of the localizer's poses (the bag's header stamps). Only needed for accuracy figures |

```bash
# 1. on the localizer machine: one directory of inputs per run
python3 $SDP/src/kdloc/make_runs.py <campaign dir>          # --dry-run lists the runs

# 2. run them, strictly one at a time (a short smoke run first)
LIMIT=60 $SDP/scripts/kdloc/campaign.sh <campaign dir> warm_s2
nohup $SDP/scripts/kdloc/campaign.sh <campaign dir> >> <campaign dir>/logs/campaign.log 2>&1 &

# 3. on the evaluation machine, after copying runs/ back
python3 src/kdloc/evaluate.py <campaign dir> --reference <dir of TUM files>
```

Each run gets `runs/<name>/` with its inputs (`run.env`, `params.yaml`, `cfg.yaml`,
`init.yaml`, `static_tf.args`) and, once run, `out/`: the pose log of every status message
(`pose.csv`), the node, relocalizer and bag-play logs, the saved trajectory, per-node CPU and
memory (`proc.csv`), tegrastats on a Jetson, and a copy of the bag's `metadata.yaml`.
`runs/<name>/done` marks a run that finished with a pose log and nothing invalid (a Fast DDS
fallback to UDP or a bag-read error makes a run invalid); `make_runs.py` refuses to change the
inputs of a run that is done. A run started with `LIMIT` is a smoke test and is never marked
done.

`campaign.sh` waits until a run's bag and map exist, holds `<campaign dir>/.machine.lock`
during each run (other jobs can take the same lock to stay out of measurements), and repeats
runs that did not finish in up to three passes. Two optional executable hooks in
`<campaign dir>/hooks/` cover what differs between machines: `ready <name>` decides (and may
prepare) whether a run's inputs are in place, and `contended <start> <end>` prints any other
work that used the machine during a run, which keeps that run from being marked done.

`evaluate.py` writes `eval/<name>.json` per run and `eval/summary.csv` (also printed):

- **status**: success (tracked to the end of the bag and never LOST), LOST events with their
  reasons, time in odometry mode, when tracking started;
- **accuracy**, computed with [evo](https://github.com/MichaelGrupp/evo)
  (`pip install '.[kdloc]'`): the reference is paired with the TRACKING poses within
  `--t-max-diff` (0.05 s) and interpolated at their stamps. Pairs whose interpolation would
  bridge a gap longer than `--max-gap` (0.3 s, a LOST stretch) or reach past the end of a
  trajectory are dropped. Reported: mean, median, 95th percentile, maximum and RMSE of the 3D
  position error, its horizontal and vertical parts, the mean split by odometry mode, the time
  spent more than 1 m off, and the figures before the first LOST. High accuracy means a
  maximum below 10 cm, as in the team template;
- **relocalization**: every attempt and its outcome; each published pose hint compared with
  the reference at its scan, correct within 1 m and 10 degrees (the heading check is skipped
  for a position-only reference);
- **processing and resources**: the localizer's own per-update timing, updates against scans
  played, per-node CPU and memory, GPU load and power.

`eval/<name>.est.tum` holds the TRACKING poses, so evo's own tools work on any run directly,
e.g. `evo_ape tum <reference>.tum eval/<name>.est.tum --sync_method interpolation -p`.

Re-scoring the 72 runs of the LUH campaign (2026-10-15) this way reproduces its report:
success, LOST times, odometry-mode share and every relocalization verdict are identical; mean
errors of the successful runs agree within 0.06 cm except one run that tracked metres off.
The figures differ slightly because evo scores at the localizer's poses, where the LUH script
scored at the ground-truth rows; two S1 runs whose maximum sat at the 10 cm limit (10.06 and
10.01 cm) read 9.66 and 9.89 cm.
