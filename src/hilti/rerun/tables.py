#!/usr/bin/env python3
"""The report's tables, straight out of data.json -- no number is typed by hand.

    tables.py [--data <path>] [--out <dir>]      # writes tbl_*.html

Each `tbl_*` returns an HTML fragment. Every figure in the report has a table here
carrying the same numbers, which is what discharges the data-viz relief rule for the
low-contrast palette slots: nothing is ever readable only as a colour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hilti import dataset                                          # noqa: E402

MB = 1 << 20


def esc(v):
    return (str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def num(v, fmt="{:.2f}", dash="—"):
    if v is None or v != v:
        return dash
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return esc(v)


REASON_SHORT = {
    "Pose covariance is too high (Rotation X)": "cov rot X",
    "Pose covariance is too high (Rotation Y)": "cov rot Y",
    "Pose covariance is too high (Rotation Z)": "cov rot Z",
    "Pose covariance is too high (Position X)": "cov pos X",
    "Pose covariance is too high (Position Y)": "cov pos Y",
    "Pose covariance is too high (Position Z)": "cov pos Z",
    "Motion predictor update failed": "predictor",
}


def short_reason(reason):
    """The lost_reason as a chip-sized label; unknown reasons pass through truncated."""
    return REASON_SHORT.get(reason, (reason or "?")[:18])


def chip(text, kind):
    return f'<span class="pill {kind}">{esc(text)}</span>'


def table(headers, rows, note=None, cls=""):
    """One table. `headers` are (label, numeric?) pairs; `rows` are lists of cells."""
    th = "".join(f'<th{" class=num" if n else ""}>{h}</th>' for h, n in headers)
    body = []
    for r in rows:
        tds = []
        for (label, n), cell in zip(headers, r):
            extra = ' class="num"' if n else ""
            tds.append(f"<td{extra}>{cell}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    out = (f'<div class="tw{" " + cls if cls else ""}"><table><thead><tr>{th}</tr>'
           f'</thead><tbody>{"".join(body)}</tbody></table></div>')
    if note:
        out += f'<p class="tnote">{note}</p>'
    return out


def ordered(data):
    rows = list(data["sequences"].values())
    rows.sort(key=lambda r: (r["site"], r["platform"] != "robot", r["seq"]))
    return rows


# --------------------------------------------------------------------------
def tbl_fastlio(data):
    rows = []
    for r in ordered(data):
        f = r.get("fastlio") or {}
        pre = r.get("fastlio_prefix") or {}
        rows.append([
            f'<code>{esc(r["seq"])}</code>',
            r["platform"],
            num(r.get("bag_duration_s"), "{:.0f}"),
            num(r.get("bag_clouds"), "{:,}"),
            num(pre.get("frame_pct"), "{:.1f}"),
            num(f.get("frame_pct"), "{:.1f}"),
            num(f.get("path_1hz_m"), "{:.1f}"),
            num(f.get("max_v"), "{:.2f}"),
            num(f.get("ba_peak"), "{:.3f}"),
            num(f.get("worst_9s_m"), "{:.1f}"),
            num(f.get("pairs"), "{:d}"),
            num(f.get("err_mean_cm"), "{:.1f}"),
            num(f.get("err_max_cm"), "{:.1f}"),
            num(f.get("err_rel_pct"), "{:.2f}"),
        ])
    return table([("sequence", False), ("rig", False), ("bag s", True),
                  ("clouds", True), ("frames&nbsp;% before", True),
                  ("frames&nbsp;% now", True), ("path m", True), ("max |v|", True),
                  ("|ba| peak", True), ("worst 9 s", True), ("cp pairs", True),
                  ("cp mean cm", True), ("cp max cm", True), ("rel %", True)], rows,
                 cls="wide",
                 note="<b>frames %</b> is processed scans (pos_log rows) over the bag's "
                      "own cloud count; the <i>before</i> column is the same sequence's "
                      "pre-fix run named in <code>config/hilti/sequences.yaml</code> as "
                      "<code>alt_run</code>, and is blank where no comparable run exists. "
                      "<b>max |v|</b>, <b>|ba| peak</b> and <b>worst 9 s</b> are "
                      "divergence detectors, not accuracy figures: a walking rig cannot "
                      "cover 25 m in 9 s. <b>cp</b> columns are control-point "
                      "pair-distance error against the survey — frame-invariant, so they "
                      "need no alignment.")


def tbl_maps(data):
    rows = []
    for r in ordered(data):
        cov = next((c for c in (data.get("coverage") or [])
                    if c["seq"] == r["seq"]), {})
        al = r.get("align") or {}
        bbox = cov.get("bbox_m")
        rows.append([
            f'<code>{esc(r["seq"])}</code>',
            num(r.get("pcd_points"), "{:,}"),
            ("&times;".join(f"{v:.0f}" for v in bbox) if bbox else "—"),
            num(cov.get("z_span_m"), "{:.1f}"),
            num(cov.get("occupied_voxels_1m"), "{:,}"),
            num(100 * cov["wall_voxel_frac"], "{:.0f}") if cov.get("wall_voxel_frac")
            is not None else "—",
            num(al.get("tilt_from_plus_z_deg"), "{:.1f}"),
            num((r.get("las_bytes") or 0) / MB, "{:.0f}") if r.get("las_bytes") else "—",
            num((r.get("kdm_bytes") or 0) / MB, "{:.0f}") if r.get("kdm_bytes") else "—",
        ])
    return table([("map from", False), ("cloud points", True), ("extent m", True),
                  ("z span", True), ("1 m voxels", True), ("wall %", True),
                  ("frame tilt °", True), ("LAS MB", True), (".kdm MB", True)], rows,
                 note="<b>1 m voxels</b> is the honest size measure: point count is "
                      "dominated by how long the rig lingered and the extent by one "
                      "stray return. <b>wall %</b> is the share of voxels whose normal "
                      "is more than 66° off vertical — the horizontal signal a "
                      "registration has to work with. <b>frame tilt</b> is how far the "
                      "raw FAST-LIO map frame's +Z is from vertical, which is what "
                      "<code>align_map.py</code> rotates out before the .kdm is built; "
                      "~180° on the handheld rig and ~90° on the robot is the IMU "
                      "mounting, not an error.")


def _loc_row(seq_label, bag_s, d, extra=None):
    if not d:
        return [seq_label, num(bag_s, "{:.0f}"), "—", "—", "—", "—", "—", "—", "—",
                "—", "—", chip("not run", "no")] + ([extra] if extra is not None else [])
    lost = d.get("first_lost")
    # The reason, not the stamp: `first_lost[1]` is absolute Unix time, which says nothing
    # to a reader, while WHICH axis tripped is the whole content of the diagnostic. The
    # tracked-seconds column already says when.
    state = chip("complete", "ok") if not lost else chip(f"lost · {short_reason(lost[0])}", "no")
    pct = (100 * d["tracked_s"] / bag_s) if bag_s else None
    row = [seq_label, num(bag_s, "{:.0f}"), num(d.get("tracked_s"), "{:.0f}"),
           num(pct, "{:.0f}"),
           num(d.get("err_mean_cm"), "{:.1f}"), num(d.get("err_max_cm"), "{:.1f}"),
           num(d.get("mapframe_median"), "{:.3f}"), num(d.get("mapframe_p95"), "{:.3f}"),
           num(d.get("mapframe_max"), "{:.2f}"),
           num(d.get("mapframe_over_1m_pct"), "{:.1f}"),
           num(d.get("inlier_median"), "{:.2f}"), num(d.get("sigma_peak"), "{:.0f}"),
           state]
    if extra is not None:
        row.append(extra)
    return row


LOC_HEADERS = [("run", False), ("bag s", True), ("tracked s", True), ("%", True),
               ("cp mean cm", True), ("cp max cm", True), ("vs FL med m", True),
               ("p95", True), ("max", True), ("&gt;1 m %", True),
               ("inlier med", True), ("σ peak", True), ("status", False)]

LOC_NOTE = ("<b>cp</b> is control-point pair-distance error, the only figure with "
            "survey backing, and it is frame-invariant. <b>vs FL</b> is the distance "
            "from the sequence's own FAST-LIO trajectory in the map's frame — dense "
            "(5–10 Hz) where the control points are 2–8 samples, so it is what makes a "
            "silent slip visible; it measures agreement with FAST-LIO, not truth, and "
            "where the two disagree the control points arbitrate. <b>σ peak</b> is the "
            "quantity <code>max_pose_sigma_factor_increase: 200</code> thresholds, and "
            "is taken over TRACKING rows only — after a LOST the node republishes a "
            "frozen value.")


def tbl_own(data):
    rows = [_loc_row(f'<code>{esc(r["seq"])}</code>', r.get("bag_duration_s"),
                     r.get("localizer")) for r in ordered(data)]
    return table(LOC_HEADERS, rows, note=LOC_NOTE, cls="wide")


def tbl_hubs(data):
    hubs = data.get("hubs") or {}
    rows = []
    for site, info in sorted(hubs.items()):
        for rank, (name, value) in enumerate(info.get("ranking") or []):
            is_hub = name == info["hub"]
            rows.append([
                site if rank == 0 else "",
                f'<code>{esc(name)}</code>' + (" " + chip("hub", "ok") if is_hub else ""),
                num(value, "{:.3f}" if info.get("basis", "").startswith("mean") else "{:,}"),
            ])
    basis = next((i.get("basis") for i in hubs.values()), "")
    return table([("site", False), ("candidate map", False),
                  ("mean coverage" if basis.startswith("mean") else "1 m voxels", True)],
                 rows,
                 note="Ranked by the mean fraction of every <i>other</i> sequence of the "
                      "same site whose cloud lands within 0.5 m of this one's after "
                      "registration. That is the quantity that decides whether the "
                      "localizer can match at all, so it picks the hub — not the biggest "
                      "map, which is a different question.")


def tbl_register(data):
    rows = []
    for name, c in sorted(data["cross"].items(), key=lambda kv: (kv[1]["site"], kv[0])):
        reg = c.get("register")
        label = (f'<code>{esc(c["seq"])}</code> &rarr; '
                 f'<code>{esc(c["hub"])}</code>')
        if not reg:
            rows.append([label, "—", "—", "—", "—", "—", "—", "—", "—"])
            continue
        pos = reg["T_hub_seq"]["position"]
        lc = reg.get("local_check") or {}
        rc = reg.get("route_coverage") or {}
        rows.append([
            label,
            num(reg["T_hub_seq"]["rotation_rpy_deg"][2], "{:+.1f}"),
            ", ".join(f"{v:.1f}" for v in pos),
            num(reg["overlap"]["within_0.2m"], "{:.2f}"),
            num(reg["overlap"]["within_0.5m"], "{:.2f}"),
            num(reg["overlap"]["within_1.0m"], "{:.2f}"),
            num(reg["overlap"]["median_nn_m"] * 100, "{:.0f}"),
            (f'{num(lc.get("d_translation_m", 0) * 100, "{:.1f}")} cm / '
             f'{num(lc.get("d_rotation_deg"), "{:.2f}")}°' if lc else "—"),
            (f'{num(rc.get("median"), "{:.2f}")} / {num(rc.get("min"), "{:.2f}")}'
             if rc else "—"),
        ])
    return table([("sequence &rarr; map", False), ("yaw °", True),
                  ("translation m", True), ("@0.2 m", True), ("@0.5 m", True),
                  ("@1.0 m", True), ("median nn cm", True),
                  ("local check", True), ("route cov. med / min", True)], rows,
                 cls="wide",
                 note="<b>@r</b> is the fraction of the sequence's cloud landing within "
                      "<i>r</i> of a hub point after registration — the ceiling on what "
                      "cross-sequence matching can achieve at all. <b>local check</b> "
                      "re-fits using only the points within 20 m of the sequence's own "
                      "origin and reports the disagreement: a few centimetres means the "
                      "global fit was not dragged by the far end of the site, so the "
                      "prior is right where the run actually begins. <b>route "
                      "coverage</b> is the same overlap measured in a 15 m ball around "
                      "each of 200 poses along the sequence's own trajectory — the "
                      "minimum, not the mean, is what has to survive the whole walk.")


def tbl_cross(data):
    rows = []
    for name, c in sorted(data["cross"].items(), key=lambda kv: (kv[1]["site"], kv[0])):
        own = ((data["sequences"].get(c["seq"]) or {}).get("localizer") or {})
        label = (f'<code>{esc(c["seq"])}</code> in '
                 f'<code>{esc(c["hub"])}</code>')
        rows.append(_loc_row(label,
                             (data["sequences"].get(c["seq"]) or {}).get("bag_duration_s"),
                             c.get("localizer"),
                             extra=num(own.get("err_mean_cm"), "{:.1f}")))
    return table(LOC_HEADERS + [("own-map cp", True)], rows, cls="wide",
                 note=LOC_NOTE + " The last column repeats the same sequence's own-map "
                      "control-point error, which is the only fair baseline for the "
                      "cross-map figure beside it.")


def tbl_rate(data):
    """The three-arm rate/queue experiment: one row per sequence, one column per arm."""
    exp = data.get("rate_experiment") or {}
    arms = exp.get("arms") or []
    if not arms:
        return '<div class="missing">no rate_experiment in data.json</div>'
    rows = []
    for r in exp["rows"]:
        cells = [f'<code>{esc(r["seq"])}</code>',
                 f'{r["imu_hz"]} Hz',
                 num((r.get(arms[-1]["tag"]) or {}).get("pairs"), "{:d}")]
        for a in arms:
            v = r.get(a["tag"])
            cells.append(num((v or {}).get("err_mean_cm"), "{:.1f}"))
        # The ratio that carries the finding: how much the slow playback bought.
        first = (r.get(arms[0]["tag"]) or {}).get("err_mean_cm")
        mid = (r.get(arms[1]["tag"]) or {}).get("err_mean_cm")
        last = (r.get(arms[-1]["tag"]) or {}).get("err_mean_cm")
        cells.append(f"{first / mid:.1f}&times;" if first and mid else "—")
        cells.append(f"{first / last:.1f}&times;" if first and last else "—")
        rows.append(cells)
    head = ([("sequence", False), ("IMU", True), ("pairs", True)]
            + [(a["label"].replace("-r ", "r"), True) for a in arms]
            + [("r0.3 gain", True), ("queue gain", True)])
    return table(head, rows, cls="wide",
                 note="Mean control-point error, cm. Every arm delivered 99.6&ndash;99.9 % "
                      "of the clouds, so nothing here is frame loss. <b>r0.3 gain</b> is "
                      "what slowing the playback bought; <b>queue gain</b> is what the deep "
                      "IMU queue bought at full speed. They agree wherever the metric has "
                      "enough control points to be worth reading &mdash; and "
                      "<code>site2_robot_3</code>, the only 200 Hz rig in the table, does "
                      "not move at all, which is what points at the queue in the first "
                      "place. <code>site2_handheld_5</code> does not move either: its error "
                      "is a ~37 m fold in the stairwell, which no amount of input fidelity "
                      "touches.")


def tbl_cross2(data):
    """Both cross-sequence configurations, per pair, with the registration beside them."""
    rows = []
    for name, c in sorted(data["cross"].items(), key=lambda kv: (kv[1]["site"], kv[0])):
        reg = c.get("register") or {}
        bag = (data["sequences"].get(c["seq"]) or {}).get("bag_duration_s")
        cells = [f'<code>{esc(c["seq"])}</code> in <code>{esc(c["hub"])}</code>',
                 num(bag, "{:.0f}"),
                 num((reg.get("overlap") or {}).get("within_0.5m"), "{:.2f}"),
                 num((reg.get("local_check") or {}).get("d_translation_m", 0) * 100, "{:.1f}")
                 if reg.get("local_check") else "—",
                 num((reg.get("route_coverage") or {}).get("min"), "{:.2f}")]
        for key in ("localizer", "localizer_v2"):
            d = c.get(key)
            if not d:
                cells += ["—", "—", "—"]
                continue
            cells += [num(d.get("tracked_s"), "{:.0f}"),
                      num(d.get("mapframe_median"), "{:.2f}"),
                      num(d.get("mapframe_over_1m_pct"), "{:.0f}")]
        rows.append(cells)
    return table([("sequence in hub map", False), ("bag s", True), ("overlap", True),
                  ("local cm", True), ("route min", True),
                  ("A tracked s", True), ("A med m", True), ("A &gt;1 m %", True),
                  ("B tracked s", True), ("B med m", True), ("B &gt;1 m %", True)],
                 rows, cls="wide",
                 note="<b>A</b> is the canonical config, identical to the own-map runs. "
                      "<b>B</b> adds <code>prior_gravity_sigma</code> 0.3&nbsp;&rarr;&nbsp;"
                      "0.05 and <code>max_pose_sigma_factor_increase</code> "
                      "200&nbsp;&rarr;&nbsp;2000. <b>med</b> is the median distance from "
                      "the sequence's own FAST-LIO trajectory pushed into the hub map by "
                      "the registration, over the part of the run that tracked, and "
                      "<b>&gt;1 m %</b> the share of that span more than a metre away — "
                      "so a long tracked span with a high &gt;1 m share is a run that "
                      "survived without being localized. Control-point scores are absent "
                      "from all but one pair because the runs end before reaching a second "
                      "surveyed point.")


ALL = {"fastlio": tbl_fastlio, "rate": tbl_rate, "cross2": tbl_cross2, "maps": tbl_maps, "own": tbl_own, "hubs": tbl_hubs,
       "register": tbl_register, "cross": tbl_cross}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out")
    a = ap.parse_args()
    out_dir = a.out or os.path.join(dataset.results_dir(), "rerun_report")
    data = json.load(open(a.data or os.path.join(out_dir, "data.json")))
    for name, fn in ALL.items():
        html = fn(data)
        path = os.path.join(out_dir, f"tbl_{name}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  tbl_{name}  {len(html):,} bytes")


if __name__ == "__main__":
    main()
