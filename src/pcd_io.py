"""Shared PCD reading / PLY + LAS writing layer for the point-file tools.

Splits, like ``pc2_layout.py`` does for PointCloud2, the part that is **fixed by
the file formats** from the part that **varies per dataset**:

fixed
  * ``PCD_TYPE_TO_NUMPY`` -- the PCD ``TYPE``/``SIZE`` pairs (PCL writes
    little-endian, and the format has no byte-order field);
  * ``NUMPY_TO_PLY`` -- the PLY scalar type names, which are the eight from the
    original specification. There is no 64-bit integer among them, so an ``I8``
    /``U8`` PCD column cannot be carried across losslessly (see ``ply_type``);
  * ``LAS_SCALE`` / ``LAS_XYZI`` -- LAS's fixed schema. Its coordinates are
    ``int32 * scale + offset``, so the scale is chosen here rather than inherited
    from a writer's default, and its dimension list is closed, so anything the
    cloud carries beyond x/y/z/intensity/colour becomes an ExtraBytes dimension.

varies per dataset
  * which fields a cloud carries and what they are called. PCL emits
    ``normal_x``, the PLY convention is ``nx``, and colour is a *packed* float in
    PCD but three ``uchar`` properties in PLY -- ``PLY_RENAMES`` / ``unpack_rgb``
    translate, so a viewer sees normals and colours rather than anonymous scalars.

The two output formats are not interchangeable, and the difference is the reason
both exist here (``pcd_convert.py --format`` picks one):

PLY
  a schema-free container. Any property name, any of eight scalar types, written
  verbatim -- so a cloud round-trips exactly and a viewer shows every field under
  the name it had. This is the format to check a map in.

LAS
  the ASPRS survey standard, and the only point-cloud format ``build_map``'s
  PDAL loader reads. Coordinates are **quantised** to the header scale and the
  dimension list is fixed. Both cost something, so both are explicit: see
  ``LAS_SCALE`` and ``to_las_columns``.

Used by ``to_bag.py`` (the ``pcd`` reader) and ``pcd_convert.py``. Deliberately
free of any ROS/``rosbags`` dependency: converting a map should not need the bag
stack installed. ``laspy`` is imported only when LAS output is asked for, so PLY
output needs nothing beyond numpy either.
"""
import os
from typing import Any, BinaryIO

import numpy as np

from pc2_layout import warn_once

# PCD (TYPE, SIZE) -> numpy code. PCL writes little-endian and the header carries
# no byte-order field, so the codes are explicitly little-endian rather than
# native -- otherwise a big-endian host would read every multi-byte field wrong.
PCD_TYPE_TO_NUMPY: dict[tuple[str, int], str] = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
}

# numpy code -> PLY scalar type name. These eight are the whole PLY type set;
# int64/uint64 have no equivalent, which ply_type() reports.
NUMPY_TO_PLY: dict[str, str] = {
    "i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
    "i4": "int", "u4": "uint", "f4": "float", "f8": "double",
}

# PCL field name -> the name a PLY consumer recognises. Normals are `nx/ny/nz`
# in the PLY specification; CloudCompare and MeshLab both key off those, and
# treat `normal_x` as an unnamed scalar field instead.
PLY_RENAMES: dict[str, str] = {
    "normal_x": "nx", "normal_y": "ny", "normal_z": "nz",
}

# Packed-colour field names. PCL stores RGB as 3 bytes inside a 4-byte field
# typed either F4 or U4 (both occur in the wild); PLY wants three uchar
# properties, so unpack_rgb() expands these.
RGB_FIELDS: tuple[str, ...] = ("rgb", "rgba")

# --- LAS ------------------------------------------------------------------
# LAS stores each coordinate as an int32 times this scale plus a per-file offset,
# so the scale is not a compression knob: it *is* the resolution of the file, and
# writing one means accepting a quantisation error of up to half of it.
#
# 1 mm, not PDAL's 0.01 default: a cm grid is coarser than the voxel sizes these
# maps are built and compared at, and checking a map should not add a layer of
# error on top of the one being measured. It costs nothing -- the point record is
# a fixed 4 bytes per axis either way -- so the only reason for a coarser scale is
# a cloud whose extent does not fit (see write_las).
LAS_SCALE = 0.001

# The width of a stored coordinate. With LAS_SCALE that is a 2147 km span around
# the offset -- ample for any of these maps -- but only because the offset is
# per-file: at offset 0 a UTM easting runs out of int32 after 2.1 km.
LAS_INT32_MAX = 2**31 - 1

# LAS 1.4, and legacy point formats 0/2 within it. Both halves matter:
#   * 1.2 cannot be used -- the ExtraBytes VLR is a 1.4 feature, and while laspy
#     will write one into a 1.2 file, PDAL 2.6.2 does not read it back, so a
#     `ring`/`t` column silently disappears between writer and reader;
#   * formats 6-10 cannot be used -- PDAL refuses them outright unless the
#     global-encoding WKT bit is set, and a FAST-LIO map has no CRS to put in a
#     WKT VLR. Format 0 (20-byte record) and 2 (+RGB) are read by everything.
LAS_VERSION = "1.4"
LAS_POINT_FORMAT_PLAIN = 0
LAS_POINT_FORMAT_RGB = 2

# The only PCD fields with a real home in the LAS point record. Everything else a
# cloud carries -- normal_x, ring, t, curvature, label -- becomes an ExtraBytes
# dimension: still named, still exactly typed (LAS extra bytes even carry the
# int64 that PLY has to widen to double), but a scalar rather than something a
# viewer knows how to interpret. See to_las_columns for the conversions.
LAS_XYZI: tuple[str, ...] = ("x", "y", "z", "intensity")
RGB_CHANNELS: tuple[str, ...] = ("red", "green", "blue")

# LAS colour is 16-bit per channel while PCD packs 8. 0xFFFF // 0xFF: expands the
# range properly, so 255 becomes full white rather than 1/257th of it -- which is
# what a viewer that trusts the declared depth would otherwise show.
RGB_8_TO_16 = 257

# ExtraBytes VLR field widths, from the specification.
LAS_EXTRA_NAME_MAX = 32
LAS_EXTRA_DESC_MAX = 32


def _name(f: BinaryIO) -> str:
    return getattr(f, "name", "<pcd>")


def _point_count(path: str, columns: dict[str, np.ndarray]) -> int:
    """The shared length of the columns, or a fatal error naming the odd one out."""
    if not columns:
        raise SystemExit(f"{path}: no fields to write")
    n = len(next(iter(columns.values())))
    bad = {k: len(v) for k, v in columns.items() if len(v) != n}
    if bad:
        raise SystemExit(f"{path}: fields disagree on point count; expected {n}, got {bad}")
    return n


# --------------------------------------------------------------------------
# PCD
# --------------------------------------------------------------------------
def read_pcd_header(f: BinaryIO) -> dict[str, Any]:
    """Parse the header of an open PCD, leaving ``f`` on the first data byte.

    Separate from the data read so a caller can list a cloud's fields and point
    count without paying for the payload -- which is what makes ``pcd_convert.py``
    able to validate a whole batch up front.
    """
    header_lines = []
    while True:
        line = f.readline()
        if not line:
            raise SystemExit(f"PCD header missing DATA line: {_name(f)}")
        text = line.decode("ascii", errors="replace").strip()
        header_lines.append(text)
        if text.startswith("DATA"):
            parts = text.split()
            if len(parts) < 2:
                raise SystemExit(f"PCD DATA line names no storage mode: {_name(f)}")
            data_kind = parts[1].lower()
            break

    header: dict[str, list[str]] = {}
    for line in header_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header[parts[0].upper()] = parts[1:]
    for key in ("FIELDS", "SIZE", "TYPE"):
        if key not in header:
            raise SystemExit(f"PCD header has no {key} line: {_name(f)}")

    fields = header["FIELDS"]
    try:
        sizes = [int(x) for x in header["SIZE"]]
    except ValueError:
        raise SystemExit(f"PCD SIZE line is not numeric ({header['SIZE']}): {_name(f)}") from None
    types = header["TYPE"]
    counts = [int(x) for x in header.get("COUNT", ["1"] * len(fields))]
    width = int(header.get("WIDTH", ["0"])[0])
    height = int(header.get("HEIGHT", ["1"])[0])
    points = int(header.get("POINTS", [str(width * height)])[0])
    if any(c != 1 for c in counts):
        raise SystemExit(f"PCD fields with COUNT != 1 are not supported yet: {_name(f)}")
    if not len(fields) == len(sizes) == len(types):
        raise SystemExit(f"PCD header lists {len(fields)} FIELDS, {len(sizes)} SIZE and "
                         f"{len(types)} TYPE entries; they must agree: {_name(f)}")
    if len(set(fields)) != len(fields):
        raise SystemExit(f"PCD header repeats a field name ({fields}): {_name(f)}")

    dtype_fields = []
    for name, size, typ in zip(fields, sizes, types, strict=True):
        code = PCD_TYPE_TO_NUMPY.get((typ.upper(), size))
        if code is None:
            raise SystemExit(f"unsupported PCD field type {typ}{size} in {_name(f)}")
        dtype_fields.append((name, code))
    return {
        "fields": fields, "dtype": np.dtype(dtype_fields), "data": data_kind,
        "width": width, "height": height, "points": points,
    }


def parse_pcd(path: str) -> dict[str, np.ndarray]:
    """Any PCD, ascii or binary, as {field name: 1-D array}.

    Arrays come back in the type the header declares, so a ``U1`` intensity stays
    one byte instead of becoming an 8-byte double on the way through.
    """
    with open(path, "rb") as f:
        header = read_pcd_header(f)
        return read_pcd_data(f, header)


def read_pcd_data(f: BinaryIO, header: dict[str, Any]) -> dict[str, np.ndarray]:
    """Read the payload of a PCD whose header has just been consumed."""
    fields, dtype, points = header["fields"], header["dtype"], header["points"]
    if header["data"] == "ascii":
        arr = np.loadtxt(f, dtype=np.float64, ndmin=2)
        if arr.shape[1] != len(fields):
            raise SystemExit(f"PCD header declares {len(fields)} fields {fields} but the ascii "
                             f"payload has {arr.shape[1]} columns: {_name(f)}")
        return {name: _cast_ascii(arr[:, i], dtype[name], name, _name(f))
                for i, name in enumerate(fields)}
    if header["data"] == "binary":
        raw = f.read(dtype.itemsize * points)
        if len(raw) < dtype.itemsize * points:
            raise SystemExit(f"PCD payload is truncated: header claims {points} points "
                             f"({dtype.itemsize * points} bytes), file holds {len(raw)}: "
                             f"{_name(f)}")
        arr = np.frombuffer(raw, dtype=dtype, count=points)
        return {name: arr[name] for name in fields}
    raise SystemExit(f"unsupported PCD DATA mode {header['data']}: {_name(f)}; "
                     f"convert binary_compressed with pcl_convert_pcd_ascii_binary")


def _cast_ascii(col: np.ndarray, dtype: np.dtype, field: str, path: str) -> np.ndarray:
    """Narrow one ascii column to its declared type -- but only if it fits.

    ``np.loadtxt`` can only give float64, and ``astype`` to a narrow integer
    *wraps* out-of-range values, so a corrupt column would come back as plausible
    small numbers. Values that do not fit keep the float64 they were read as: the
    consumer's own range check (``to_bag.clamp_to_type``, ``ply_type``) then sees
    them and says so.
    """
    if dtype.kind == "f":
        return col.astype(dtype)
    info = np.iinfo(dtype)
    if col.size and (not np.isfinite(col).all()
                     or col.min() < info.min or col.max() > info.max):
        warn_once(f"{os.path.basename(path)}:{field}", (
            f"ascii field {field!r} holds values outside its declared "
            f"{dtype.name} range [{info.min}, {info.max}] (or NaN/Inf); kept as float64 "
            f"rather than wrapped"))
        return col
    return col.astype(dtype)


# --------------------------------------------------------------------------
# PLY
# --------------------------------------------------------------------------
def ply_type(dtype: np.dtype, field: str) -> str:
    """PLY scalar type name for a numpy dtype.

    PLY has no 64-bit integer, so ``i8``/``u8`` is widened to ``double``, which is
    exact only up to 2**53 -- worth a warning, because a nanosecond timestamp
    column is above that and would land on a neighbouring value.
    """
    code = dtype.str.lstrip("<>|=")
    name = NUMPY_TO_PLY.get(code)
    if name is None:
        if dtype.itemsize == 8 and dtype.kind in "iu":
            warn_once(f"ply:{field}", (
                f"field {field!r} is {dtype.name}, which PLY cannot express; written as "
                f"double -- exact only below 2**53, so a nanosecond timestamp loses its "
                f"last digits. Drop it with --fields if that matters"))
            return "double"
        raise SystemExit(f"field {field!r} has dtype {dtype.name}, which has no PLY equivalent")
    return name


def _le_code(dtype: np.dtype) -> str:
    """Explicitly little-endian numpy code. A ``binary_little_endian`` PLY is not
    native order on a big-endian host, so the byte order is stated rather than
    inherited."""
    code = dtype.str.lstrip("<>|=")
    return code if dtype.itemsize == 1 else f"<{code}"


def unpack_rgb(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Expand a packed PCD ``rgb``/``rgba`` field into PLY red/green/blue[/alpha].

    Written through unchanged if the cloud has none. PCL packs the three bytes
    into a 4-byte field that may be typed either float or uint; both are just
    reinterpreted as uint32 here, never converted, since the float bit pattern is
    not a number in any meaningful sense.
    """
    out: dict[str, np.ndarray] = {}
    for name, col in columns.items():
        if name not in RGB_FIELDS or col.dtype.itemsize != 4:
            out[name] = col
            continue
        packed = np.ascontiguousarray(col).view(np.uint32)
        out["red"] = ((packed >> 16) & 0xFF).astype(np.uint8)
        out["green"] = ((packed >> 8) & 0xFF).astype(np.uint8)
        out["blue"] = (packed & 0xFF).astype(np.uint8)
        if name == "rgba":
            out["alpha"] = ((packed >> 24) & 0xFF).astype(np.uint8)
    return out


def write_ply(path: str, columns: dict[str, np.ndarray], binary: bool = True,
              comments: tuple[str, ...] = ()) -> int:
    """Write {field name: 1-D array} as a PLY vertex list; returns the point count.

    ``binary`` selects ``binary_little_endian`` (the default: a 300k-point map is
    ~10 MB binary against ~40 MB and a 20x slower write in ascii). Field names are
    passed through ``PLY_RENAMES`` so normals arrive as normals; the caller has
    already chosen and ordered the fields.
    """
    n = _point_count(path, columns)
    props = [(PLY_RENAMES.get(name, name), col, ply_type(col.dtype, name))
             for name, col in columns.items()]
    header = ["ply", f"format {'binary_little_endian' if binary else 'ascii'} 1.0"]
    header += [f"comment {c}" for c in comments]
    header += [f"element vertex {n}"]
    header += [f"property {typ} {name}" for name, _, typ in props]
    header += ["end_header", ""]

    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        if not n:
            return 0
        if binary:
            dtype = np.dtype([(name, _le_code(col.dtype)) for name, col, _ in props])
            out = np.empty(n, dtype=dtype)
            for name, col, _ in props:
                out[name] = col
            f.write(out.tobytes())
        else:
            # "%.9g" round-trips float32 exactly and keeps a float64 coordinate to
            # sub-millimetre; "%d" for integers, so a uchar intensity is not "17.0".
            fmt = " ".join("%d" if col.dtype.kind in "iub" else "%.9g"
                           for _, col, _ in props)
            np.savetxt(f, np.column_stack([col for _, col, _ in props]), fmt=fmt)
    return n


# --------------------------------------------------------------------------
# LAS
# --------------------------------------------------------------------------
def import_laspy(what: str = "LAS output"):
    """Import laspy, or explain how to get it.

    Deferred rather than imported at module scope so that reading a PCD, writing a
    PLY and building a bag keep working on numpy alone -- laspy is an extra that
    only the LAS path needs.
    """
    try:
        import laspy
    except ImportError:
        raise SystemExit(f"{what} needs laspy: pip install 'laspy>=2.0' "
                         f"(or pip install '.[las]')") from None
    return laspy


def check_laz_backend() -> None:
    """Fail before a batch starts if LAZ compression has no backend.

    laspy itself imports fine without one and only raises when the first file is
    written, which in a 1400-frame run means failing 1400 times over.
    """
    laspy = import_laspy("LAZ output")
    if not any(b.is_available() for b in laspy.LazBackend):
        raise SystemExit("LAZ output needs a compression backend: "
                         "pip install lazrs (or pip install '.[laz]'). "
                         "Use --format las for uncompressed output")


def las_mapped(columns: dict[str, np.ndarray]) -> tuple[str, ...]:
    """Which of ``columns`` go into the LAS point record proper.

    Colour only counts when all three channels are there: the point format is
    chosen from this, and a cloud carrying a lone ``red`` gets the plain record
    plus an extra byte rather than a colour record with two empty channels.
    """
    rgb = all(c in columns for c in RGB_CHANNELS)
    return LAS_XYZI + (RGB_CHANNELS if rgb else ())


def las_extra_fields(fields: list[str], dtype: np.dtype) -> list[str]:
    """Which fields will become ExtraBytes dimensions, from the PCD header alone.

    Runs the real conversion over zero-length columns rather than re-deriving the
    rule: whether a packed ``rgb`` expands into native colour channels depends on
    its declared width, and one place should decide that.

    Known before any payload is read, so ``--dry-run`` can say it, and it is a
    property of the batch rather than of a cloud -- reported once by the caller
    instead of on every one of 1400 identical frames.
    """
    empty = {name: np.empty(0, dtype=dtype[name]) for name in fields}
    columns, _ = to_las_columns(unpack_rgb(empty))
    mapped = las_mapped(columns)
    return [name for name in columns if name not in mapped]


def to_las_columns(columns: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Fit PCD columns to LAS's fixed schema; returns (columns, notes).

    Unlike the PLY path, this is **not** a lossless re-container. ``notes`` is what
    it cost *this* cloud -- it ends up in the per-cloud summary rather than being
    left for the reader to discover. What it cost the *batch* -- which fields have
    no native LAS home -- is ``las_extra_fields``, reported once.

      * ``intensity`` is a LAS ``uint16``. A float reflectivity is rounded, and
        anything outside [0, 65535] is clamped rather than wrapped (``astype``
        would turn 70000 into 4464, a plausible-looking value);
      * colour is widened from PCD's 8 bits to LAS's 16 (``RGB_8_TO_16``);
      * everything else is left exactly as it is, to be written as an ExtraBytes
        dimension under its own name -- so ``normal_x`` stays ``normal_x`` here,
        where the PLY path renames it to ``nx``. There is no LAS convention to
        rename it *to*, and a viewer shows it as a named scalar either way.

    Run after ``unpack_rgb``, which is shared with the PLY path: a packed colour
    has to become three channels before either format can carry it.
    """
    out: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for name, col in columns.items():
        if name == "intensity":
            out[name], clamped = _clamp_to_uint16(col)
            if clamped:
                notes.append(f"clamped {clamped} intensity value(s) into LAS uint16")
        elif name in RGB_CHANNELS and col.dtype == np.uint8:
            out[name] = col.astype(np.uint16) * RGB_8_TO_16
        else:
            # alpha included: LAS has no alpha channel, so it stays a uint8 extra
            # byte, which carries it exactly.
            out[name] = col
    return out, notes


def _clamp_to_uint16(col: np.ndarray) -> tuple[np.ndarray, int]:
    """One column as uint16, saturating instead of wrapping; returns (column, n clamped)."""
    if col.dtype == np.uint16:
        return col, 0
    info = np.iinfo(np.uint16)
    if col.dtype.kind == "f":
        # Non-finite first: it compares false against both bounds, so it would
        # survive the clip and then cast to an arbitrary integer.
        col = np.where(np.isfinite(col), col, 0.0)
        col = np.rint(col)
    over = int(np.count_nonzero((col < info.min) | (col > info.max)))
    return np.clip(col, info.min, info.max).astype(np.uint16), over


def write_las(path: str, columns: dict[str, np.ndarray], scale: float = LAS_SCALE,
              compress: bool = False, comment: str = "") -> int:
    """Write {field name: 1-D array} as a LAS/LAZ point cloud; returns the count.

    ``scale`` is the coordinate quantisation step (see ``LAS_SCALE``) and is
    applied to all three axes, so a map is not finer in x than in z. The offset is
    per-axis and per-file: ``floor(min)``, which keeps the stored integers small
    and, more to the point, keeps a georeferenced cloud inside int32 -- with an
    offset of 0 a UTM easting overflows a 1 mm scale at 2.1 km.

    ``compress`` writes LAZ. The extension is not consulted, because the caller
    writes to ``<name>.las.part`` first and renames on success.

    Columns must already have been through ``to_las_columns``.
    """
    laspy = import_laspy()
    n = _point_count(path, columns)
    missing = [a for a in ("x", "y", "z") if a not in columns]
    if missing:
        raise SystemExit(f"{path}: no {missing} field(s); a LAS file needs coordinates")
    if not scale > 0:
        raise SystemExit(f"{path}: LAS scale must be positive, got {scale}")

    mapped = las_mapped(columns)
    header = laspy.LasHeader(
        version=LAS_VERSION,
        point_format=(LAS_POINT_FORMAT_RGB if RGB_CHANNELS[0] in mapped
                      else LAS_POINT_FORMAT_PLAIN))
    header.scales = [scale] * 3
    header.offsets = _las_offsets(path, columns, scale, n)
    header.generating_software = "pcd_convert.py"
    if comment:
        # LAS has no comment line. A description VLR is where a reader (and
        # `pdal info --metadata`) can still find where the file came from.
        header.vlrs.append(laspy.VLR(user_id="SDPIPE", record_id=1,
                                     description="source"[:LAS_EXTRA_DESC_MAX],
                                     record_data=comment.encode("utf-8")))

    native = set(header.point_format.dimension_names)
    for name, col in columns.items():
        if name in mapped:
            continue
        if len(name) > LAS_EXTRA_NAME_MAX:
            raise SystemExit(f"{path}: field {name!r} is {len(name)} characters; a LAS extra "
                             f"dimension name is limited to {LAS_EXTRA_NAME_MAX}")
        if name in native:
            # laspy accepts the clash and then silently writes the *native*
            # dimension instead, leaving the extra bytes zeroed.
            raise SystemExit(f"{path}: field {name!r} collides with the LAS point record's own "
                             f"{name!r} dimension, which this writer does not populate; drop or "
                             f"rename it (--fields) to write this cloud as LAS")
        header.add_extra_dim(laspy.ExtraBytesParams(
            name=name, type=col.dtype, description=f"PCD {name}"[:LAS_EXTRA_DESC_MAX]))

    las = laspy.LasData(header)
    for axis in ("x", "y", "z"):
        # float64 before laspy divides by the scale: the subtraction of a large
        # offset from a float32 coordinate is where the precision would go.
        setattr(las, axis, np.asarray(columns[axis], dtype=np.float64))
    for name, col in columns.items():
        if name not in ("x", "y", "z"):
            setattr(las, name, col)
    with open(path, "wb") as f:
        las.write(f, do_compress=compress)
    return n


def _las_offsets(path: str, columns: dict[str, np.ndarray], scale: float,
                 n: int) -> list[float]:
    """Per-axis LAS offset, checked against the int32 the coordinates go into.

    laspy raises ``OverflowError`` on a value that does not fit, which reads as a
    numpy problem rather than a format one; this says which axis, how far it is
    over, and what to do about it.
    """
    if not n:
        return [0.0, 0.0, 0.0]
    offsets = []
    for axis in ("x", "y", "z"):
        col = np.asarray(columns[axis], dtype=np.float64)
        lo = float(np.floor(np.min(col)))
        span = (float(np.max(col)) - lo) / scale
        if span > LAS_INT32_MAX:
            raise SystemExit(
                f"{path}: the cloud spans {float(np.max(col)) - lo:.1f} m in {axis}, which at a "
                f"{scale} m scale needs {span:.3g} > int32 steps. Write it with a coarser "
                f"--las-scale (at least {(float(np.max(col)) - lo) / LAS_INT32_MAX:.2g}), or as "
                f"PLY, which stores coordinates as floats and has no such limit")
        offsets.append(lo)
    return offsets
