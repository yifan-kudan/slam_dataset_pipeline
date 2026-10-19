"""Tests for the PCD -> PLY/LAS converter and the shared PCD/PLY/LAS io layer.

The cases are the ones that would produce a *plausible but wrong* file rather than
an error: a narrow field wrapped on the way through, a packed colour written as a
meaningless float, an output replaced by a half-written one, two inputs quietly
overwriting each other's PLY.

The LAS cases add the two ways that format loses information -- a quantised
coordinate and a fixed dimension list -- plus the pair of header choices that
decide whether PDAL can read the result at all (see ``pcd_io.LAS_VERSION``).
"""
import os
import shutil
import struct
import subprocess

import numpy as np
import pytest

import pcd_io
import pcd_convert


# --------------------------------------------------------------------------
# Fixtures: PCDs written by hand, so the header is exactly what is under test
# --------------------------------------------------------------------------
def write_pcd(path, fields, types, sizes, rows, data="ascii"):
    header = (f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
              f"FIELDS {' '.join(fields)}\nSIZE {' '.join(str(s) for s in sizes)}\n"
              f"TYPE {' '.join(types)}\nCOUNT {' '.join('1' for _ in fields)}\n"
              f"WIDTH {len(rows)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              f"POINTS {len(rows)}\nDATA {data}\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if data == "ascii":
            f.write(("\n".join(" ".join(repr(v) for v in row) for row in rows) + "\n").encode())
        else:
            codes = {("F", 4): "f", ("F", 8): "d", ("U", 1): "B", ("U", 2): "H",
                     ("U", 4): "I", ("U", 8): "Q", ("I", 2): "h", ("I", 4): "i",
                     ("I", 8): "q"}
            fmt = "<" + "".join(codes[(t, s)] for t, s in zip(types, sizes, strict=True))
            for row in rows:
                f.write(struct.pack(fmt, *row))
    return path


def xyzi_pcd(tmp_path, name="cloud.pcd", rows=((1.0, 2.0, 3.0, 7), (4.0, 5.0, 6.0, 8)),
             data="ascii"):
    return write_pcd(tmp_path / name, ["x", "y", "z", "intensity"], ["F", "F", "F", "U"],
                     [4, 4, 4, 1], rows, data)


def read_ply(path):
    """Minimal PLY reader: (list of (name, type), structured array)."""
    with open(path, "rb") as f:
        props, fmt = [], None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("property"):
                _, typ, name = line.split()
                props.append((name, typ))
            elif line == "end_header":
                break
        codes = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
                 "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8"}
        if fmt == "ascii":
            return props, np.loadtxt(f, ndmin=2)
        dtype = np.dtype([(n, codes[t]) for n, t in props])
        return props, np.frombuffer(f.read(), dtype=dtype)


# --------------------------------------------------------------------------
# PCD reading
# --------------------------------------------------------------------------
@pytest.mark.parametrize("data", ["ascii", "binary"])
def test_roundtrip_values(tmp_path, data):
    cols = pcd_io.parse_pcd(str(xyzi_pcd(tmp_path, data=data)))
    assert cols["x"].tolist() == [1.0, 4.0]
    assert cols["intensity"].tolist() == [7, 8]


def test_ascii_keeps_declared_types(tmp_path):
    """An intensity declared U1 must not come back as an 8-byte double, or the PLY
    grows 8x and claims a precision the sensor does not have."""
    cols = pcd_io.parse_pcd(str(xyzi_pcd(tmp_path)))
    assert cols["intensity"].dtype == np.uint8
    assert cols["x"].dtype == np.float32


def test_ascii_out_of_range_is_not_wrapped(tmp_path):
    """`astype(np.uint8)` turns 300 into 44 silently; the column has to stay float."""
    path = xyzi_pcd(tmp_path, rows=((1.0, 2.0, 3.0, 300.0),))
    cols = pcd_io.parse_pcd(str(path))
    assert cols["intensity"].tolist() == [300.0]


def test_truncated_binary_payload_is_fatal(tmp_path):
    path = xyzi_pcd(tmp_path, data="binary")
    with open(path, "rb") as f:
        raw = f.read()
    path.write_bytes(raw[:-4])
    with pytest.raises(SystemExit, match="truncated"):
        pcd_io.parse_pcd(str(path))


def test_ascii_column_count_mismatch_is_fatal(tmp_path):
    path = write_pcd(tmp_path / "short.pcd", ["x", "y", "z", "intensity"],
                     ["F", "F", "F", "F"], [4, 4, 4, 4], [(1.0, 2.0, 3.0)])
    with pytest.raises(SystemExit, match="4 fields"):
        pcd_io.parse_pcd(str(path))


def test_compressed_payload_is_reported(tmp_path):
    path = write_pcd(tmp_path / "z.pcd", ["x", "y", "z"], ["F", "F", "F"], [4, 4, 4],
                     [], data="binary_compressed")
    with pytest.raises(SystemExit, match="binary_compressed"):
        pcd_io.parse_pcd(str(path))


# --------------------------------------------------------------------------
# PLY writing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("binary", [True, False])
def test_write_ply_values_and_types(tmp_path, binary):
    out = str(tmp_path / "out.ply")
    cols = {"x": np.array([1.5, -2.5], np.float32), "y": np.zeros(2, np.float32),
            "z": np.ones(2, np.float32), "intensity": np.array([7, 250], np.uint8)}
    assert pcd_io.write_ply(out, cols, binary=binary) == 2
    props, arr = read_ply(out)
    assert props == [("x", "float"), ("y", "float"), ("z", "float"), ("intensity", "uchar")]
    xs = arr["x"] if binary else arr[:, 0]
    inten = arr["intensity"] if binary else arr[:, 3]
    assert xs.tolist() == [1.5, -2.5]
    assert inten.tolist() == [7, 250]


def test_normals_get_ply_names(tmp_path):
    """CloudCompare reads normals from nx/ny/nz; PCL's normal_x is just a scalar."""
    out = str(tmp_path / "n.ply")
    cols = {a: np.ones(1, np.float32) for a in ("x", "y", "z")}
    cols |= {f"normal_{a}": np.zeros(1, np.float32) for a in "xyz"}
    pcd_io.write_ply(out, cols)
    assert [n for n, _ in read_ply(out)[0]] == ["x", "y", "z", "nx", "ny", "nz"]


def test_packed_rgb_is_unpacked(tmp_path):
    """PCL packs colour into one 4-byte field; written through verbatim a viewer
    shows a nonsense scalar instead of colour."""
    packed = np.array([(0x11 << 16) | (0x22 << 8) | 0x33], np.uint32)
    cols = pcd_io.unpack_rgb({"x": np.ones(1, np.float32), "rgb": packed})
    assert (cols["red"][0], cols["green"][0], cols["blue"][0]) == (0x11, 0x22, 0x33)
    assert "rgb" not in cols


def test_float_packed_rgb_is_reinterpreted_not_converted(tmp_path):
    """The same bytes are often typed F4; casting rather than viewing them would
    turn the bit pattern into a huge float and lose the colour."""
    packed = np.array([(0x11 << 16) | (0x22 << 8) | 0x33], np.uint32).view(np.float32)
    cols = pcd_io.unpack_rgb({"rgb": packed})
    assert (cols["red"][0], cols["green"][0], cols["blue"][0]) == (0x11, 0x22, 0x33)


def test_int64_field_warns_and_widens(tmp_path, capsys):
    assert pcd_io.ply_type(np.dtype("<i8"), "timestamp") == "double"
    assert "WARNING" in capsys.readouterr().out


def test_mismatched_lengths_are_fatal(tmp_path):
    with pytest.raises(SystemExit, match="point count"):
        pcd_io.write_ply(str(tmp_path / "bad.ply"),
                         {"x": np.ones(3, np.float32), "y": np.ones(2, np.float32)})


# --------------------------------------------------------------------------
# Batch behaviour
# --------------------------------------------------------------------------
def test_converts_a_directory(tmp_path, capsys):
    src = tmp_path / "in"
    src.mkdir()
    for name in ("a.pcd", "b.pcd"):
        xyzi_pcd(src, name)
    pcd_convert.main([str(src), "-o", str(tmp_path / "out")])
    assert sorted(os.listdir(tmp_path / "out")) == ["a.ply", "b.ply"]
    assert "converted 2/2" in capsys.readouterr().out


def test_writes_beside_the_input_without_out_dir(tmp_path):
    xyzi_pcd(tmp_path, "map.pcd")
    pcd_convert.main([str(tmp_path / "map.pcd")])
    assert os.path.isfile(tmp_path / "map.ply")


def test_existing_output_is_kept_without_force(tmp_path):
    xyzi_pcd(tmp_path, "map.pcd")
    (tmp_path / "map.ply").write_bytes(b"previous")
    pcd_convert.main([str(tmp_path / "map.pcd")])
    assert (tmp_path / "map.ply").read_bytes() == b"previous"
    pcd_convert.main([str(tmp_path / "map.pcd"), "--force"])
    assert (tmp_path / "map.ply").read_bytes() != b"previous"


def test_failed_conversion_leaves_the_old_output_intact(tmp_path, monkeypatch):
    """--force must not cost the previous PLY when the new one cannot be written."""
    xyzi_pcd(tmp_path, "map.pcd")
    (tmp_path / "map.ply").write_bytes(b"previous")

    def boom(*_args, **_kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(pcd_convert, "write_ply", boom)
    with pytest.raises(SystemExit):
        pcd_convert.main([str(tmp_path / "map.pcd"), "--force"])
    assert (tmp_path / "map.ply").read_bytes() == b"previous"
    assert not os.path.exists(str(tmp_path / "map.ply.part"))


def test_colliding_outputs_are_refused(tmp_path):
    """--recursive over per-run map directories: a dozen fastlio_map.pcd would all
    want the same fastlio_map.ply, and the last one would silently win."""
    for run in ("run1", "run2"):
        (tmp_path / run).mkdir()
        xyzi_pcd(tmp_path / run, "fastlio_map.pcd")
    with pytest.raises(SystemExit, match="more than one"):
        pcd_convert.main([str(tmp_path), "--recursive", "-o", str(tmp_path / "out")])


def test_summary_tells_same_named_inputs_apart(tmp_path, capsys):
    """A FAST-LIO batch is a dozen fastlio_map.pcd under a dozen sequences; a
    summary of a dozen identical basenames says nothing about which is which."""
    for run in ("run1", "run2"):
        (tmp_path / run).mkdir()
        xyzi_pcd(tmp_path / run, "fastlio_map.pcd")
    pcd_convert.main([str(tmp_path), "--recursive"])
    out = capsys.readouterr().out
    assert f"run1{os.sep}fastlio_map.pcd" in out
    assert f"run2{os.sep}fastlio_map.pcd" in out


def test_long_batch_thins_its_own_output(tmp_path, capsys):
    """A per-frame scan directory is thousands of files -- two lines each buries
    the run, and the thinning has to say it is thinning rather than look stalled."""
    src = tmp_path / "scans"
    src.mkdir()
    for i in range(pcd_convert.PROGRESS_LINES + 5):
        xyzi_pcd(src, f"{i:06d}.pcd")
    pcd_convert.main([str(src), "-o", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "reporting every" in out
    assert f"{pcd_convert.PROGRESS_LINES + 5} cloud(s) converted with nothing to report" in out
    assert len(os.listdir(tmp_path / "out")) == pcd_convert.PROGRESS_LINES + 5


def test_field_selection(tmp_path):
    xyzi_pcd(tmp_path, "map.pcd")
    pcd_convert.main([str(tmp_path / "map.pcd"), "--fields", "intensity", "x", "y", "z"])
    props, _ = read_ply(tmp_path / "map.ply")
    assert [n for n, _ in props] == ["x", "y", "z", "intensity"]   # geometry first


def test_unknown_field_is_reported_and_nothing_is_written(tmp_path):
    xyzi_pcd(tmp_path, "map.pcd")
    with pytest.raises(SystemExit):
        pcd_convert.main([str(tmp_path / "map.pcd"), "--fields", "ring"])
    assert not os.path.exists(tmp_path / "map.ply")


def test_missing_xyz_is_reported(tmp_path):
    write_pcd(tmp_path / "flat.pcd", ["x", "y"], ["F", "F"], [4, 4], [(1.0, 2.0)])
    with pytest.raises(SystemExit):
        pcd_convert.main([str(tmp_path / "flat.pcd")])


def test_nonfinite_points_are_dropped_by_default(tmp_path, capsys):
    xyzi_pcd(tmp_path, "map.pcd", rows=((1.0, 2.0, 3.0, 1), (float("nan"), 0.0, 0.0, 2)))
    pcd_convert.main([str(tmp_path / "map.pcd")])
    _, arr = read_ply(tmp_path / "map.ply")
    assert arr["x"].tolist() == [1.0]
    assert "dropped 1 non-finite" in capsys.readouterr().out


def test_nonfinite_points_can_be_kept(tmp_path, capsys):
    xyzi_pcd(tmp_path, "map.pcd", rows=((1.0, 2.0, 3.0, 1), (float("nan"), 0.0, 0.0, 2)))
    pcd_convert.main([str(tmp_path / "map.pcd"), "--keep-nonfinite"])
    _, arr = read_ply(tmp_path / "map.ply")
    assert len(arr) == 2
    assert "KEPT 1 non-finite" in capsys.readouterr().out


def test_one_bad_input_does_not_stop_the_batch(tmp_path, capsys):
    src = tmp_path / "in"
    src.mkdir()
    xyzi_pcd(src, "good.pcd")
    (src / "bad.pcd").write_bytes(b"VERSION 0.7\n")   # no DATA line
    with pytest.raises(SystemExit):
        pcd_convert.main([str(src), "-o", str(tmp_path / "out")])
    assert os.listdir(tmp_path / "out") == ["good.ply"]
    assert "SKIP" in capsys.readouterr().out


# --------------------------------------------------------------------------
# LAS: the quantisation, and the fixed dimension list
# --------------------------------------------------------------------------
laspy = pytest.importorskip("laspy", reason="LAS output is an optional extra")


def rich_pcd(tmp_path, name="rich.pcd", rows=None):
    """A cloud with one field of every kind LAS has to make a decision about:
    coordinates, an intensity that does not fit uint16, a packed colour, and three
    fields with no LAS dimension at all (one of them int64)."""
    rows = rows if rows is not None else [
        (1.0, 2.0, 3.0, 7.4, 3, 2**60, 0.5, (0x11 << 16) | (0x22 << 8) | 0x33),
        (4.0, 5.0, 6.0, 70000.0, 4, 2**60 + 1, -0.5, (0xFF << 16)),
    ]
    return write_pcd(tmp_path / name,
                     ["x", "y", "z", "intensity", "ring", "t", "normal_x", "rgb"],
                     ["F", "F", "F", "F", "U", "U", "F", "U"],
                     [4, 4, 4, 4, 2, 8, 4, 4], rows, data="binary")


def test_las_header_is_one_pdal_can_read(tmp_path):
    """The two choices in pcd_io.LAS_VERSION, asserted rather than trusted.

    LAS 1.2 would drop every extra dimension on the way through PDAL, and point
    formats 6-10 make PDAL refuse the file outright for want of a WKT flag. Both
    failures are silent-ish and neither is visible from laspy's side, so they are
    pinned here.
    """
    rich_pcd(tmp_path)
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las"])
    las = laspy.read(str(tmp_path / "rich.las"))
    assert str(las.header.version) == "1.4"
    assert las.header.point_format.id in (0, 2)


@pytest.mark.skipif(shutil.which("pdal") is None, reason="pdal not installed")
def test_pdal_reads_the_las_and_sees_the_extra_dimensions(tmp_path):
    """The end-to-end version of the above: build_map's loader is PDAL's."""
    rich_pcd(tmp_path)
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las"])
    out = subprocess.run(["pdal", "info", "--summary", str(tmp_path / "rich.las")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ring" in out.stdout and '"num_points": 2' in out.stdout.replace("  ", " ")


def test_las_quantises_coordinates_to_the_declared_scale(tmp_path):
    """The scale is the resolution of the file, so the default has to be the fine
    one: at PDAL's own 1 cm a map is rounded 10x more coarsely than it is built."""
    rich_pcd(tmp_path, rows=[(1.0004, 2.0, 3.0, 1, 0, 0, 0.0, 0)])
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las"])
    fine = laspy.read(str(tmp_path / "rich.las"))
    assert fine.header.scales.tolist() == [pcd_io.LAS_SCALE] * 3
    assert abs(float(np.asarray(fine.x)[0]) - 1.0004) <= pcd_io.LAS_SCALE / 2

    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las",
                      "--las-scale", "0.01", "--force"])
    coarse = laspy.read(str(tmp_path / "rich.las"))
    assert float(np.asarray(coarse.x)[0]) == 1.0     # 4 tenths of a mm -> gone


def test_las_reports_the_quantisation_error_before_converting(tmp_path, capsys):
    """A run should not have to be reconstructed afterwards to find out what
    resolution its LAS files are at."""
    rich_pcd(tmp_path)
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las", "--dry-run"])
    assert "0.5 mm quantisation error" in capsys.readouterr().out


def test_las_offset_keeps_a_georeferenced_cloud_inside_int32(tmp_path):
    """1 mm steps in an int32 reach 2147 km from the offset but only 2.1 km from
    zero, so a UTM easting needs the per-file offset to survive at all.

    Written as F8: at that magnitude a float32 PCD is itself quantised to ~6 cm, so
    a float32 fixture would measure the input's error rather than the writer's.
    """
    write_pcd(tmp_path / "utm.pcd", ["x", "y", "z"], ["F", "F", "F"], [8, 8, 8],
              [(500000.123, 5000000.456, 3.0)], data="binary")
    pcd_convert.main([str(tmp_path / "utm.pcd"), "--format", "las"])
    las = laspy.read(str(tmp_path / "utm.las"))
    assert float(np.asarray(las.x)[0]) == pytest.approx(500000.123, abs=pcd_io.LAS_SCALE)
    assert float(np.asarray(las.y)[0]) == pytest.approx(5000000.456, abs=pcd_io.LAS_SCALE)


def test_las_span_beyond_int32_is_refused_with_a_way_out(tmp_path, capsys):
    """laspy's own OverflowError says nothing about scales or axes."""
    rich_pcd(tmp_path, rows=[(0.0, 0.0, 0.0, 1, 0, 0, 0.0, 0),
                             (3e6, 0.0, 0.0, 1, 0, 0, 0.0, 0)])
    with pytest.raises(SystemExit):
        pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las"])
    out = capsys.readouterr().out
    assert "spans 3000000.0 m in x" in out and "--las-scale" in out
    assert not os.path.exists(tmp_path / "rich.las")
    assert not os.path.exists(tmp_path / "rich.las.part")


def test_las_extra_dimensions_keep_their_name_and_exact_type(tmp_path):
    """LAS cannot express `ring` or `t`, but its extra bytes can carry them exactly
    -- including the int64 that the PLY path has to widen to a lossy double."""
    rich_pcd(tmp_path)
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las"])
    las = laspy.read(str(tmp_path / "rich.las"))
    assert [d.name for d in las.point_format.extra_dimensions] == ["ring", "t", "normal_x"]
    assert np.asarray(las.ring).tolist() == [3, 4]
    assert np.asarray(las.t).tolist() == [2**60, 2**60 + 1]      # exact, not 2**60 twice
    assert np.asarray(las.normal_x).tolist() == [0.5, -0.5]


def test_las_says_which_fields_became_extra_bytes(tmp_path, capsys):
    """Known from the PCD header, so it is said once, up front, and in --dry-run --
    not on each of 1400 identical frames."""
    rich_pcd(tmp_path)
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las", "--dry-run"])
    assert "extra bytes: ring, t, normal_x" in capsys.readouterr().out


def test_ply_keeps_the_pcd_name_for_normals_and_las_does_not_rename_it(tmp_path):
    """The rename is a PLY convention (CloudCompare keys off nx/ny/nz). LAS has no
    convention to rename *to*, so the field keeps the name it had."""
    rich_pcd(tmp_path)
    pcd_convert.main([str(tmp_path / "rich.pcd")])
    assert "nx" in [n for n, _ in read_ply(tmp_path / "rich.ply")[0]]
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las"])
    las = laspy.read(str(tmp_path / "rich.las"))
    assert "normal_x" in [d.name for d in las.point_format.extra_dimensions]


def test_las_intensity_is_clamped_not_wrapped(tmp_path, capsys):
    """`astype(uint16)` turns 70000 into 4464 -- a value a viewer would happily
    render as if the sensor had reported it."""
    rich_pcd(tmp_path)
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las"])
    las = laspy.read(str(tmp_path / "rich.las"))
    assert np.asarray(las.intensity).tolist() == [7, 65535]
    assert "clamped 1 intensity value(s)" in capsys.readouterr().out


def test_las_colour_is_widened_to_16_bit(tmp_path):
    """LAS declares 16 bits per channel. Writing PCD's 0-255 into it unchanged makes
    a colour cloud render as near-black in anything that trusts the declaration."""
    rich_pcd(tmp_path)
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las"])
    las = laspy.read(str(tmp_path / "rich.las"))
    assert las.header.point_format.id == pcd_io.LAS_POINT_FORMAT_RGB
    assert np.asarray(las.red).tolist() == [0x11 * 257, 0xFF * 257]
    assert np.asarray(las.blue).tolist() == [0x33 * 257, 0]


def test_field_named_like_an_unwritten_las_dimension_is_refused(tmp_path):
    """laspy accepts `classification` as an extra dimension and then writes the
    point record's own one instead, leaving the extra bytes zeroed."""
    write_pcd(tmp_path / "seg.pcd", ["x", "y", "z", "classification"],
              ["F", "F", "F", "U"], [4, 4, 4, 1], [(1.0, 2.0, 3.0, 5)])
    with pytest.raises(SystemExit):
        pcd_convert.main([str(tmp_path / "seg.pcd"), "--format", "las"])
    assert not os.path.exists(tmp_path / "seg.las")


def test_nonfinite_points_are_dropped_for_las_too(tmp_path):
    """A NaN coordinate has no integer to become; unlike PLY it cannot be kept."""
    rich_pcd(tmp_path, rows=[(1.0, 2.0, 3.0, 1, 0, 0, 0.0, 0),
                             (float("nan"), 0.0, 0.0, 1, 0, 0, 0.0, 0)])
    pcd_convert.main([str(tmp_path / "rich.pcd"), "--format", "las"])
    assert np.asarray(laspy.read(str(tmp_path / "rich.las")).x).tolist() == [1.0]


@pytest.mark.parametrize("argv, why", [
    (["--format", "las", "--ascii"], "--ascii"),
    (["--format", "las", "--keep-nonfinite"], "--keep-nonfinite"),
    (["--las-scale", "0.01"], "--las-scale"),
    (["--format", "las", "--las-scale", "-1"], "positive"),
])
def test_flags_that_do_not_apply_to_the_format_are_refused(tmp_path, argv, why):
    """Accepting `--ascii --format las` and ignoring it looks like it worked."""
    xyzi_pcd(tmp_path, "map.pcd")
    with pytest.raises(SystemExit, match=why):
        pcd_convert.main([str(tmp_path / "map.pcd"), *argv])


def test_laz_either_compresses_or_says_what_is_missing(tmp_path):
    """laspy imports fine without a compression backend and only fails on the first
    write, which in a 1400-frame batch means failing 1400 times."""
    xyzi_pcd(tmp_path, "map.pcd")
    argv = [str(tmp_path / "map.pcd"), "--format", "laz"]
    if any(b.is_available() for b in laspy.LazBackend):
        pcd_convert.main(argv)
        assert len(laspy.read(str(tmp_path / "map.laz")).points) == 2
    else:
        with pytest.raises(SystemExit, match="lazrs"):
            pcd_convert.main(argv)
        assert not os.path.exists(tmp_path / "map.laz")


def test_las_and_ply_outputs_do_not_collide(tmp_path):
    """The extension comes from --format, so the same batch can produce both."""
    xyzi_pcd(tmp_path, "map.pcd")
    pcd_convert.main([str(tmp_path / "map.pcd")])
    pcd_convert.main([str(tmp_path / "map.pcd"), "--format", "las"])
    assert (tmp_path / "map.ply").exists() and (tmp_path / "map.las").exists()
