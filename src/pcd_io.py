"""Shared PCD reading / PLY writing layer for the point-file tools.

Splits, like ``pc2_layout.py`` does for PointCloud2, the part that is **fixed by
the file formats** from the part that **varies per dataset**:

fixed
  * ``PCD_TYPE_TO_NUMPY`` -- the PCD ``TYPE``/``SIZE`` pairs (PCL writes
    little-endian, and the format has no byte-order field);
  * ``NUMPY_TO_PLY`` -- the PLY scalar type names, which are the eight from the
    original specification. There is no 64-bit integer among them, so an ``I8``
    /``U8`` PCD column cannot be carried across losslessly (see ``ply_type``).

varies per dataset
  * which fields a cloud carries and what they are called. PCL emits
    ``normal_x``, the PLY convention is ``nx``, and colour is a *packed* float in
    PCD but three ``uchar`` properties in PLY -- ``PLY_RENAMES`` / ``unpack_rgb``
    translate, so a viewer sees normals and colours rather than anonymous scalars.

Used by ``to_bag.py`` (the ``pcd`` reader) and ``pcd_to_ply.py``. Deliberately
free of any ROS/``rosbags`` dependency: converting a map to PLY should not need
the bag stack installed.
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


def _name(f: BinaryIO) -> str:
    return getattr(f, "name", "<pcd>")


# --------------------------------------------------------------------------
# PCD
# --------------------------------------------------------------------------
def read_pcd_header(f: BinaryIO) -> dict[str, Any]:
    """Parse the header of an open PCD, leaving ``f`` on the first data byte.

    Separate from the data read so a caller can list a cloud's fields and point
    count without paying for the payload -- which is what makes ``pcd_to_ply.py``
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
    if not columns:
        raise SystemExit(f"{path}: no fields to write")
    n = len(next(iter(columns.values())))
    bad = {k: len(v) for k, v in columns.items() if len(v) != n}
    if bad:
        raise SystemExit(f"{path}: fields disagree on point count; expected {n}, got {bad}")

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
