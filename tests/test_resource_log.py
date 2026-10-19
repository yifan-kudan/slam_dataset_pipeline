"""Tests for src/resource_log.py -- the CPU/memory/disk sampler.

The one that matters most is `test_counts_a_subprocess_that_never_got_sampled`:
sampling only the *live* process tree looked right in review and was wrong in
practice, under-reporting a real batch by half because each conversion
subprocess started and finished between two ticks.
"""
import os
import subprocess
import sys

import pytest

import resource_log
from resource_log import COLUMNS, ResourceLog, read_csv

pytestmark = pytest.mark.skipif(not resource_log.available(), reason="psutil not installed")

# ~0.3 s of CPU in a child process, with no I/O and no imports to skew the timing.
BURN = "sum(i * i for i in range(3_000_000))"


def burn(seconds_of_work: str = BURN) -> None:
    subprocess.run([sys.executable, "-c", seconds_of_work], check=True)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def test_writes_a_csv_with_the_documented_columns(tmp_path):
    csv_path = str(tmp_path / "res.csv")
    log = ResourceLog(csv_path, interval=0.05).start()
    burn()
    log.close()

    rows = read_csv(csv_path)
    assert rows, "no samples were written"
    assert list(rows[0]) == list(COLUMNS)
    assert rows[-1]["elapsed_sec"] > 0
    assert rows[-1]["cpu_count"] >= 1
    assert rows[-1]["mem_total_mb"] > 0
    # elapsed_sec must be non-decreasing, or the chart's x axis is meaningless
    assert all(b["elapsed_sec"] >= a["elapsed_sec"]
               for a, b in zip(rows, rows[1:], strict=False))


def test_counts_a_subprocess_that_never_got_sampled(tmp_path):
    """A child that lives entirely between two ticks still has to be counted.

    The interval is longer than the whole test, so the sampler thread never runs
    while the child is alive: everything charged here comes from the kernel's
    children_user/children_system on the parent, read by close()'s final sample.
    """
    csv_path = str(tmp_path / "res.csv")
    log = ResourceLog(csv_path, interval=30.0).start()
    burn()
    log.close()

    rows = read_csv(csv_path)
    assert len(rows) == 1, "the sampler thread should not have ticked at all"
    assert rows[0]["cpu_percent"] > 10.0, (
        f"the subprocess's CPU was lost: {rows[0]['cpu_percent']}%")


def test_cpu_percent_is_not_wildly_over_capacity(tmp_path):
    """Sanity bound: a single busy child cannot exceed a few cores' worth.

    Guards the double-counting failure mode of adding children_* while also
    summing the live children it refers to.
    """
    csv_path = str(tmp_path / "res.csv")
    log = ResourceLog(csv_path, interval=0.05).start()
    burn()
    log.close()

    peak = max(r["cpu_percent"] for r in read_csv(csv_path))
    assert peak <= 300.0, f"{peak}% for one busy subprocess means CPU is counted twice"


def test_stage_labels_survive_a_comma(tmp_path):
    csv_path = str(tmp_path / "res.csv")
    log = ResourceLog(csv_path, interval=0.05).start()
    log.stage("12 sequences, 4 jobs")
    burn()
    log.close()

    rows = read_csv(csv_path)
    assert any(r["stage"] == "12 sequences, 4 jobs" for r in rows)
    assert all(list(r) == list(COLUMNS) for r in rows), "a stage name shifted the columns"


def test_totals_are_monotonic(tmp_path):
    """Cumulative columns must never go backwards -- they do if a process that
    exits takes its byte counters out of the running total with it."""
    csv_path = str(tmp_path / "res.csv")
    log = ResourceLog(csv_path, interval=0.05).start()
    for _ in range(3):
        burn()
    log.close()

    rows = read_csv(csv_path)
    for name in ("read_total_mb", "write_total_mb"):
        values = [r[name] for r in rows]
        assert values == sorted(values), f"{name} went backwards: {values}"


def test_rejects_a_non_positive_interval(tmp_path):
    with pytest.raises(SystemExit):
        ResourceLog(str(tmp_path / "res.csv"), interval=0)


def test_summary_is_empty_without_samples(tmp_path):
    assert ResourceLog(str(tmp_path / "res.csv")).summary() == ""


def test_record_is_a_no_op_without_a_path(tmp_path):
    with resource_log.record(None) as log:
        assert log is None


def test_record_writes_csv_and_chart(tmp_path):
    csv_path = str(tmp_path / "logs" / "res.csv")
    with resource_log.record(csv_path, interval=0.05) as log:
        assert log is not None
        log.stage("20230531")
        burn()
    assert os.path.isfile(csv_path)
    assert os.path.isfile(os.path.splitext(csv_path)[0] + ".png")


def test_record_can_skip_the_chart(tmp_path):
    csv_path = str(tmp_path / "res.csv")
    with resource_log.record(csv_path, interval=0.05, chart=False):
        burn()
    assert os.path.isfile(csv_path)
    assert not os.path.exists(os.path.splitext(csv_path)[0] + ".png")


def test_a_sampling_failure_does_not_escape(tmp_path, monkeypatch):
    """A monitor must never take the conversion down with it."""
    log = ResourceLog(str(tmp_path / "res.csv"), interval=0.05).start()
    monkeypatch.setattr(log, "_walk", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    log._sample()
    log.close()
    assert log.errors >= 1


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, 0)) for c in COLUMNS) + "\n")


def sample_row(t, stage="", **over):
    row = {c: 0 for c in COLUMNS}
    row.update(elapsed_sec=t, timestamp="2026-08-04T12:00:00", stage=stage, cpu_count=16,
               cpu_percent=100 + t, cpu_percent_system=20, rss_mb=120, mem_used_mb=9000,
               mem_total_mb=16000, read_mb_s=1.5, write_mb_s=200, read_total_mb=t,
               write_total_mb=t * 10, disk_free_gb=200 - t, disk_used_gb=100 + t, procs=2)
    row.update(over)
    return row


def test_stage_spans_groups_contiguous_rows():
    rows = [sample_row(0, "a"), sample_row(1, "a"), sample_row(2, "b"), sample_row(3, "")]
    assert resource_log._stage_spans(rows) == [(0, 1, "a"), (2, 2, "b")]


def test_stage_spans_reopens_a_repeated_name():
    """a -> b -> a is three bands, not two: the same sequence converted twice
    (a --force rerun) must not swallow the one in between."""
    rows = [sample_row(0, "a"), sample_row(1, "b"), sample_row(2, "a")]
    assert [s[2] for s in resource_log._stage_spans(rows)] == ["a", "b", "a"]


def test_plot_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    csv_path = tmp_path / "res.csv"
    write_csv(csv_path, [sample_row(t, "20230531" if t < 5 else "20230609") for t in range(10)])
    png = resource_log.plot(str(csv_path))
    assert png and os.path.isfile(png)
    assert os.path.getsize(png) > 1000


def test_plot_handles_a_long_run(tmp_path):
    """Past three minutes the x axis switches to minutes; the stage bands have to
    be converted with it or they land off the plot."""
    pytest.importorskip("matplotlib")
    csv_path = tmp_path / "res.csv"
    write_csv(csv_path, [sample_row(t * 10, f"seq{t // 30}") for t in range(120)])
    assert resource_log.plot(str(csv_path), str(tmp_path / "out.png"))


def test_plot_declines_an_empty_log(tmp_path):
    csv_path = tmp_path / "res.csv"
    write_csv(csv_path, [sample_row(0)])
    assert resource_log.plot(str(csv_path)) is None


def test_read_csv_tolerates_a_truncated_row(tmp_path):
    """A killed run leaves a partial last line; the chart should still draw."""
    csv_path = tmp_path / "res.csv"
    write_csv(csv_path, [sample_row(t) for t in range(4)])
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write("4.0,2026-08-04T12:00:00,,12")
    rows = read_csv(str(csv_path))
    assert len(rows) == 5
    assert rows[-1]["cpu_percent"] == 12
    missing = rows[-1]["disk_free_gb"]
    assert missing != missing, "a missing cell must be NaN, not a string the chart cannot plot"


def test_plot_survives_a_truncated_row(tmp_path):
    pytest.importorskip("matplotlib")
    csv_path = tmp_path / "res.csv"
    write_csv(csv_path, [sample_row(t) for t in range(4)])
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write("4.0,2026-08-04T12:00:00,,12")
    assert resource_log.plot(str(csv_path), str(tmp_path / "out.png"))
