#!/usr/bin/env python3
"""Sample CPU / memory / disk usage while a conversion runs -> CSV + line chart.

A full ConPR batch is ~25 minutes of solid I/O, and the question afterwards is
always the same: was it CPU-bound or disk-bound, how much RAM did it take, and
did the output volume have room. This samples all three on a background thread
and writes one CSV row per tick, so the answer is measured rather than guessed.

What is measured:

  * **the conversion, not the machine.** CPU and RSS are summed over this
    process *and every descendant* (``to_bag.py`` runs as a subprocess, and
    ``--jobs`` runs several at once), so the numbers describe the conversion.
    System-wide CPU/memory are logged alongside as context -- a machine that is
    busy with something else explains a slow batch.
  * **disk two ways.** Throughput (``read_mb_s`` / ``write_mb_s``, from the
    process tree's block-layer counters) answers "is the disk the bottleneck";
    free space on the output volume answers "will this batch finish".

Three details that make the numbers trustworthy rather than merely plausible:

  * **CPU includes children the kernel has already reaped.** ``cpu_times()``
    carries ``children_user``/``children_system`` -- the CPU of descendants that
    have been waited for -- so summing ``user + system + children_*`` over the
    live tree counts a subprocess that started and finished entirely between two
    ticks. Sampling live processes alone loses those: it reported 46% peak on a
    run whose conversions each lasted about a second. Nothing is double-counted,
    because a child's time only moves into its parent's ``children_*`` once it
    is reaped and it has left the tree by then.
  * **I/O is accumulated per process, keeping the last value seen for ones that
    have exited.** ``/proc/<pid>/io`` has no equivalent of ``children_*``, so
    summing only live processes would erase a sequence's I/O the moment its
    subprocess finished. (The one thing this cannot recover is a process that
    lived entirely between two ticks -- at the default 1 s period against
    minutes-long conversions, that is nothing; shorten ``--resource-interval``
    if a run's conversions are very fast.)
  * **rows are written with ``os.write`` to an unbuffered fd.** The sampler
    thread runs while ``ProcessPoolExecutor`` forks workers; a buffered Python
    file object holds a lock that a fork can duplicate mid-hold, and the log has
    to survive a Ctrl-C anyway. O_CLOEXEC keeps the fd out of the subprocesses.

Sizes are 1024-based: ``_mb`` is MiB, ``_gb`` is GiB.

Standalone use -- re-draw the chart from a CSV kept from an earlier run::

    python3 src/resource_log.py Converted_bag/logs/resources_20260804-101500.csv
"""
import contextlib
import csv
import io
import os
import threading
import time
from collections.abc import Generator
from typing import Any

try:
    import psutil
except ImportError:                                  # optional dependency
    psutil = None                                    # type: ignore[assignment]

MB = 1024.0 * 1024.0
GB = MB * 1024.0

COLUMNS = (
    "elapsed_sec",          # seconds since sampling started
    "timestamp",            # local wall clock, ISO-ish
    "stage",                # what the batch was doing (sequence name), "" if unset
    "cpu_percent",          # conversion process tree, top-style (100 = one core)
    "cpu_count",            # cores on this machine, so cpu_percent is interpretable
    "cpu_percent_system",   # whole machine, 0-100
    "rss_mb",               # conversion process tree, resident
    "mem_used_mb",          # whole machine
    "mem_total_mb",
    "read_mb_s",            # conversion process tree, block-layer
    "write_mb_s",
    "read_total_mb",        # cumulative since sampling started
    "write_total_mb",
    "disk_free_gb",         # output volume
    "disk_used_gb",
    "procs",                # live processes in the tree at this tick
)

# Reference data-viz palette, light mode: categorical slots 1 and 2, plus the
# surface / ink / grid tokens. Never more than two series on one axis here, so
# slots 1-2 are the whole palette -- they are the pair the palette documents as
# passing every colour-vision gate.
_S1, _S2 = "#2a78d6", "#eb6834"
_SURFACE, _INK, _INK_MUTED, _GRID, _BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#f0efec"


def available() -> bool:
    return psutil is not None


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
class ResourceLog:
    """Background sampler writing one CSV row per ``interval``.

    Never raises into the caller: a conversion must not fail because a process
    vanished between listing it and reading it. A sampling error is counted and
    reported once, and the batch carries on.
    """

    def __init__(self, csv_path: str, interval: float = 1.0, disk_path: str | None = None):
        if interval <= 0:
            raise SystemExit(f"--resource-interval must be > 0, got {interval}")
        self.csv_path = os.path.abspath(csv_path)
        self.interval = float(interval)
        self.disk_path = os.path.abspath(disk_path) if disk_path else os.path.dirname(self.csv_path)
        self.rows: list[dict[str, Any]] = []
        self.errors = 0

        self._stage = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd = -1
        self._root = None
        self._io_seen: dict[tuple[int, float], tuple[int, int]] = {}
        self._t0 = 0.0
        self._prev_t = 0.0
        self._prev_cpu = 0.0
        self._prev_io = (0, 0)
        self._base_io = (0, 0)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "ResourceLog":
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        # O_CLOEXEC: to_bag.py subprocesses have no business inheriting this.
        self._fd = os.open(self.csv_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o644)
        self._write_row(COLUMNS)

        self._root = psutil.Process()
        psutil.cpu_percent(interval=None)      # prime the system-wide counter
        self._t0 = self._prev_t = time.monotonic()
        self._prev_cpu, _, self._prev_io, _ = self._walk()
        self._base_io = self._prev_io
        self._thread = threading.Thread(target=self._loop, name="resource-log", daemon=True)
        self._thread.start()
        return self

    def stage(self, name: str) -> None:
        """Label the rows from here on (the sequence currently being converted)."""
        self._stage = name or ""

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 5.0)
            self._thread = None
        if self._fd >= 0:
            self._sample()          # a final row, so a fast run is not an empty file
            os.close(self._fd)
            self._fd = -1

    # -- the thread --------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _walk(self) -> tuple[float, float, tuple[int, int], int]:
        """(cpu_seconds, rss_bytes, (read, write) bytes, live process count).

        CPU and I/O are cumulative totals for the whole tree; RSS is
        instantaneous and therefore only over the live processes.
        """
        procs = []
        try:
            procs = [self._root, *self._root.children(recursive=True)]
        except psutil.Error:
            pass

        cpu, rss, alive = 0.0, 0.0, 0
        for proc in procs:
            try:
                with proc.oneshot():
                    times = proc.cpu_times()
                    # `children_*` is this process's already-reaped descendants.
                    # Counting it for every live process attributes each reaped
                    # subtree to exactly one reaper -- a pool worker's own
                    # to_bag.py children land in the worker's counter, and move
                    # up to ours only once the worker itself is reaped.
                    cpu += (float(times.user) + float(times.system)
                            + float(getattr(times, "children_user", 0.0))
                            + float(getattr(times, "children_system", 0.0)))
                    rss += float(proc.memory_info().rss)
                    alive += 1
                    try:
                        counters = proc.io_counters()
                        # (pid, create_time) rather than pid alone: a recycled
                        # pid would inherit the dead process's byte counts and
                        # make the next delta negative.
                        self._io_seen[(proc.pid, proc.create_time())] = (
                            int(counters.read_bytes), int(counters.write_bytes))
                    except (psutil.AccessDenied, NotImplementedError, AttributeError):
                        pass
            except psutil.Error:        # exited between listing and reading
                continue

        read = sum(v[0] for v in self._io_seen.values())
        write = sum(v[1] for v in self._io_seen.values())
        return cpu, rss, (read, write), alive

    def _sample(self) -> None:
        try:
            now = time.monotonic()
            dt = max(now - self._prev_t, 1e-6)
            cpu, rss, io_now, alive = self._walk()
            vmem = psutil.virtual_memory()
            try:
                disk = psutil.disk_usage(self.disk_path)
                free_gb, used_gb = disk.free / GB, disk.used / GB
            except OSError:             # the volume went away mid-run
                free_gb = used_gb = float("nan")

            row = {
                "elapsed_sec": round(now - self._t0, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "stage": self._stage,
                "cpu_percent": round(max(cpu - self._prev_cpu, 0.0) / dt * 100.0, 1),
                "cpu_count": os.cpu_count() or 1,
                "cpu_percent_system": round(psutil.cpu_percent(interval=None), 1),
                "rss_mb": round(rss / MB, 1),
                "mem_used_mb": round(vmem.used / MB, 1),
                "mem_total_mb": round(vmem.total / MB, 1),
                "read_mb_s": round(max(io_now[0] - self._prev_io[0], 0) / MB / dt, 3),
                "write_mb_s": round(max(io_now[1] - self._prev_io[1], 0) / MB / dt, 3),
                "read_total_mb": round(max(io_now[0] - self._base_io[0], 0) / MB, 1),
                "write_total_mb": round(max(io_now[1] - self._base_io[1], 0) / MB, 1),
                "disk_free_gb": round(free_gb, 2),
                "disk_used_gb": round(used_gb, 2),
                "procs": alive,
            }
            self._prev_t, self._prev_cpu, self._prev_io = now, cpu, io_now
            self.rows.append(row)
            self._write_row([row[c] for c in COLUMNS])
        except Exception:       # a monitor must never take the conversion down
            self.errors += 1

    def _write_row(self, values) -> None:
        # Formatted with csv (so a stage name with a comma still round-trips),
        # then handed to os.write: an unbuffered fd holds no lock for a fork to
        # duplicate, and leaves nothing unflushed if the run is killed.
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow(values)
        os.write(self._fd, buf.getvalue().encode("utf-8"))

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str:
        """One line for the run summary, or "" if nothing was sampled."""
        if not self.rows:
            return ""
        peak_cpu = max(r["cpu_percent"] for r in self.rows)
        peak_rss = max(r["rss_mb"] for r in self.rows)
        read = self.rows[-1]["read_total_mb"]
        written = self.rows[-1]["write_total_mb"]
        free = self.rows[-1]["disk_free_gb"]
        return (f"peak CPU {peak_cpu:.0f}% of {self.rows[-1]['cpu_count']*100}%, "
                f"peak RSS {peak_rss/1024:.2f} GB, disk read {read/1024:.2f} GB, "
                f"written {written/1024:.2f} GB, {free:.1f} GB free")


@contextlib.contextmanager
def record(csv_path: str | None, interval: float = 1.0, disk_path: str | None = None,
           chart: bool = True) -> Generator[ResourceLog | None, None, None]:
    """Sample for the duration of the block, then write the chart.

    Yields None when logging is off or psutil is missing, so the caller's code
    path is the same either way (``if log: log.stage(...)``).
    """
    if not csv_path:
        yield None
        return
    if psutil is None:
        print("resource log: psutil is not installed, skipping "
              "(pip install psutil to enable CPU/memory/disk logging)")
        yield None
        return

    log = ResourceLog(csv_path, interval, disk_path).start()
    try:
        yield log
    finally:
        log.close()
        line = log.summary()
        if line:
            print(f"\nresources: {line}")
        print(f"  csv   {log.csv_path}")
        if log.errors:
            print(f"  note  {log.errors} sample(s) failed and were skipped")
        if chart and log.rows:
            png = plot(log.csv_path)
            if png:
                print(f"  chart {png}")


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def read_csv(csv_path: str) -> list[dict[str, Any]]:
    """Load a resource CSV back into rows of floats (strings kept as strings)."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for raw in rows:
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if key is None:                     # more fields than the header
                continue
            if key in ("timestamp", "stage"):
                row[key] = value or ""
            elif value in (None, ""):
                # A killed run leaves its last line half-written, so a missing
                # cell is normal. NaN, not "": the chart does arithmetic on these.
                row[key] = float("nan")
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = float("nan")
        out.append(row)
    return out


def _finite(rows: list[dict[str, Any]], name: str, default: float) -> float:
    """The last usable value of a column that is constant over a run
    (``cpu_count``, ``mem_total_mb``), ignoring cells a truncated row left NaN."""
    for row in reversed(rows):
        value = row.get(name)
        if isinstance(value, float) and value == value and value > 0:
            return value
    return default


def _stage_spans(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """Contiguous runs of the `stage` column as (start_sec, end_sec, name)."""
    spans: list[tuple[float, float, str]] = []
    for row in rows:
        name, t = row.get("stage", ""), row["elapsed_sec"]
        if spans and spans[-1][2] == name:
            spans[-1] = (spans[-1][0], t, name)
        else:
            spans.append((t, t, name))
    return [s for s in spans if s[2]]


def _panel(ax, x, series: list[tuple[list[float], str, str]], ylabel: str, title: str,
           zero_base: bool = True, extreme: str = "max", title_pad: float = 8.0) -> None:
    """One small multiple: at most two series, always in the same unit.

    Never a second y-scale. Two units on one plot align two arbitrary scales and
    invent a correlation that is not in the data, so anything measured
    differently -- or on a wildly different scale, which is just as unreadable --
    gets its own panel instead.
    """
    for values, label, colour in series:
        ax.plot(x, values, color=colour, linewidth=2.0, solid_joinstyle="round",
                solid_capstyle="round", label=label)
    ax.set_title(title, fontsize=10, color=_INK, loc="left", pad=title_pad)
    ax.set_ylabel(ylabel, fontsize=8, color=_INK_MUTED)
    ax.grid(True, color=_GRID, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    # Headroom for the peak label: the extreme is by definition at the top of the
    # panel, so without padding its annotation is drawn over the legend. It has
    # to be a fraction of the *drawn* range, not of the data's spread: pinning
    # the floor to 0 stretches a near-constant series (a flat 71 MB RSS) across
    # the whole panel, and a margin sized to its 0.1 MB spread vanishes.
    ax.margins(y=0.18)
    if zero_base:
        top = max((v for values, _, _ in series for v in values if v == v), default=0.0)
        ax.set_ylim(0.0, top * 1.15 if top > 0 else 1.0)
    # Plain tick labels: matplotlib otherwise factors a near-constant axis into
    # an offset ("+2.229e2" above the ticks), which is unreadable as free space.
    ax.ticklabel_format(useOffset=False, axis="y", style="plain")
    if len(series) > 1:     # a single series is named by the panel title already
        # Above the axes, not inside them: an in-plot legend lands on whichever
        # series happens to run high, and this chart's shape is not known ahead.
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8,
                  frameon=False, ncols=len(series), labelcolor=_INK_MUTED,
                  handlelength=1.6, columnspacing=1.4, borderaxespad=0.2)

    # One label per panel, on the series that actually has the extreme -- a value
    # on every point is noise, and "peak 0.0" on the flat series is worse than
    # no label at all.
    pick = max if extreme == "max" else min
    candidates = []
    for values, _, colour in series:
        finite = [(v, t) for v, t in zip(values, x, strict=True) if v == v]
        if finite:
            value, at = pick(finite)
            candidates.append((value, at, colour))
    if candidates:
        value, at, colour = pick(candidates, key=lambda c: c[0])
        ax.plot([at], [value], marker="o", markersize=8, color=colour,
                markeredgecolor=_SURFACE, markeredgewidth=2.0, zorder=5)
        # Decimals from the spread, not the magnitude: free space is a big number
        # whose interesting movement is in the second decimal ("low 222.9" hides
        # exactly what the panel is there to show), while 210 MB/s wants none.
        spread = max((v for values, _, _ in series for v in values if v == v), default=0.0) - \
            min((v for values, _, _ in series for v in values if v == v), default=0.0)
        decimals = 0 if spread >= 100 else 1 if spread >= 10 else 2 if spread >= 1 else 3
        # Flip the label inward near the right edge so it cannot run off the axes.
        late = x and at > x[0] + 0.75 * (x[-1] - x[0])
        ax.annotate(f"{'peak' if extreme == 'max' else 'low'} {value:,.{decimals}f}", (at, value),
                    textcoords="offset points", xytext=(-8 if late else 8, 6),
                    ha="right" if late else "left", fontsize=8, color=_INK)


def plot(csv_path: str, png_path: str | None = None, title: str | None = None) -> str | None:
    """Draw the CSV as stacked line panels. Returns the PNG path, or None if
    matplotlib is missing or the log is empty (the CSV is still the record)."""
    try:
        import matplotlib
        matplotlib.use("Agg")           # no display on a conversion box
        import matplotlib.pyplot as plt
    except ImportError:
        print("resource log: matplotlib is not installed, so no chart was drawn "
              "(pip install matplotlib, or re-plot later with "
              f"`python3 src/resource_log.py {csv_path}`)")
        return None

    # A row with no usable elapsed_sec cannot be placed on the x axis at all.
    rows = [r for r in read_csv(csv_path) if r.get("elapsed_sec") == r.get("elapsed_sec")]
    if len(rows) < 2:
        return None
    png_path = png_path or os.path.splitext(csv_path)[0] + ".png"

    def col(name: str) -> list[float]:
        return [r.get(name, float("nan")) for r in rows]

    total_sec = col("elapsed_sec")[-1]
    # Minutes past a few minutes: a 25-minute batch on a seconds axis reads as
    # four-digit tick labels.
    in_minutes = total_sec > 180.0
    x = [v / 60.0 for v in col("elapsed_sec")] if in_minutes else col("elapsed_sec")
    cores = _finite(rows, "cpu_count", 1.0)
    # Both CPU series as "% of the whole machine" so they can share one axis
    # honestly: the tree total is top-style (1600% on 16 cores), the system
    # figure is already 0-100.
    cpu_tree = [v / cores for v in col("cpu_percent")]
    mem_total = _finite(rows, "mem_total_mb", 1.0)
    mem_pct = [v / mem_total * 100.0 for v in col("mem_used_mb")]

    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True,
                             gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor(_SURFACE)
    for ax in axes:
        ax.set_facecolor(_SURFACE)

    _panel(axes[0], x, [(cpu_tree, "conversion", _S1),
                        (col("cpu_percent_system"), "whole machine", _S2)],
           f"% of {cores:.0f} cores", "CPU", title_pad=22.0)
    # Conversion RSS and machine-wide memory are both MB but differ by ~50x, so
    # sharing an axis would flatten the conversion's line onto the floor. Two
    # panels, and the machine's is a percentage because "how full" is the
    # question there.
    _panel(axes[1], x, [(col("rss_mb"), "conversion", _S1)], "MB", "Memory — conversion (RSS)")
    _panel(axes[2], x, [(mem_pct, "machine", _S1)], "% of RAM",
           f"Memory — whole machine (of {mem_total/1024:.1f} GB)")
    axes[2].set_ylim(0, 100)
    _panel(axes[3], x, [(col("read_mb_s"), "read", _S1), (col("write_mb_s"), "write", _S2)],
           "MB/s", "Disk throughput — conversion")
    _panel(axes[4], x, [(col("disk_free_gb"), "free", _S1)], "GB",
           "Free space on the output volume", zero_base=False, extreme="min")
    axes[4].set_xlabel("elapsed (min)" if in_minutes else "elapsed (s)",
                       fontsize=9, color=_INK_MUTED)

    # Alternate bands mark which sequence was converting -- being able to point
    # at a spike and name it is the reason a chart beats the CSV.
    spans = _stage_spans(rows)
    if in_minutes:
        spans = [(a / 60.0, b / 60.0, n) for a, b, n in spans]
    if spans and len(spans) <= 24:
        rotation = 30 if len(spans) > 6 else 0
        for i, (start, end, name) in enumerate(spans):
            if i % 2:
                for ax in axes:
                    ax.axvspan(start, end, color=_BAND, zorder=0, linewidth=0)
            # Above the top panel's title row, so the band names never land on
            # the legend or on the data.
            axes[0].annotate(name, (start, 1.0), xycoords=("data", "axes fraction"),
                             textcoords="offset points", xytext=(2, 16), fontsize=7,
                             color=_INK_MUTED, rotation=rotation,
                             ha="left", va="bottom", clip_on=False, annotation_clip=False)

    shown = f"{total_sec/60:.1f} min" if in_minutes else f"{total_sec:.0f} s"
    fig.suptitle(title or f"Conversion resource usage — {shown}, {os.path.basename(csv_path)}",
                 fontsize=12, color=_INK, x=0.02, ha="left", y=0.995)
    fig.savefig(png_path, dpi=140, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Re-draw a resource log CSV as a line chart.")
    ap.add_argument("csv", help="resource log written by batch_to_bag.py / to_bag.py")
    ap.add_argument("-o", "--out", help="output PNG (default: alongside the CSV)")
    args = ap.parse_args(argv)
    png = plot(args.csv, args.out)
    print(png if png else "nothing to plot")


if __name__ == "__main__":
    main()
