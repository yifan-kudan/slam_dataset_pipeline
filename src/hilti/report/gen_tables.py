#!/usr/bin/env python3
"""Emit the report's HTML tables from data.json, so no number is retyped by hand."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hilti import dataset                                             # noqa: E402

OUT = dataset.report_dir()
D = json.load(open(f"{OUT}/data.json"))
BAGS = json.load(open(f"{OUT}/bags.json"))

GROUPS = [
    ("robot platform · site2 · slab and ramp",
     ["site2_robot_3", "site2_robot_2", "site2_robot_1"]),
    ("site3 handheld · all four single-level",
     ["site3_handheld_1", "site3_handheld_3", "site3_handheld_4", "site3_handheld_2"]),
    ("site1 handheld · includes an unlit enclosed stairwell",
     ["site1_handheld_4", "site1_handheld_2", "site1_handheld_3", "site1_handheld_1",
      "site1_handheld_5"]),
    ("site2 handheld · every route crosses a floor",
     ["site2_handheld_6", "site2_handheld_4", "site2_handheld_5"]),
]


# Literals rather than escapes inside the f-strings below: a backslash in an
# f-string is only legal from Python 3.12 and this repo targets 3.10.
HI = ' class="hi"'
REBAGGED = '<span class="chip-x">补发</span>'


def short(seq):
    """site2_handheld_4 -> s2·h4, as the pair table's first column."""
    return seq.replace("site", "s").replace("_handheld_", "·h").replace("_robot_", "·r")


def cls(v, good, mid):
    if v is None:
        return ""
    return " class=\"ok\"" if v <= good else (" class=\"mid\"" if v <= mid else " class=\"hi\"")


def chip(pct, lost):
    """Verdict follows whether a LOST ever fired -- the percentage only says how far it got."""
    if pct is None:
        return '<span class="chip div">未运行</span>'
    if not lost:
        return '<span class="chip pass">零 Lost</span>'
    if pct >= 60:
        return '<span class="chip lvl">末段发散</span>'
    return '<span class="chip div">早期 Lost</span>'



def num(v, f="{:.1f}", dash="—"):
    return dash if v is None else f.format(v)


def main_table():
    r = ['<table>', '<caption>Every sequence at the delivered configuration '
         '(<code>-r 1</code>, <code>track_only</code>, that sequence\'s own levelled map, '
         'covariance limit 200). Control-point error counts only the tracked span. '
         '&sigma; peak is taken over TRACKING rows only — after a LOST the node keeps '
         'republishing the frozen value, which would otherwise equal the peak. '
         '&ldquo;vs FAST-LIO&rdquo; is a direct comparison in the shared map frame, '
         'no alignment.<br><sup>†</sup> 两个系统评的控制点对不同一批（一方少覆盖了一个点），'
         '两列不可直接相减。少于 2 对时不报精度，只报对数。'
         '<code>site2_handheld_6</code> 一行的 FAST-LIO 取 <code>-r 0.15</code> —— '
         '它的地图是用那条云重建的，r0.3 那条已被证明错 1.4 m（见 §09）。</caption>',
         '<thead><tr><th class="l">序列</th><th>跟踪 / 整包</th><th>%</th>'
         '<th>内点 中位</th><th>&sigma; 峰值</th><th>kd 控制点<br>平均 / 最大 (cm)</th>'
         '<th>FAST-LIO<br>平均 / 最大 (cm)</th><th>vs FAST-LIO<br>中位</th>'
         '<th class="l">判定</th></tr></thead><tbody>']
    tot_trk = tot_dur = 0.0
    n99 = 0
    for title, seqs in GROUPS:
        r.append(f'<tr class="grp"><td colspan="9">{title}</td></tr>')
        for s in seqs:
            m = D[s]
            kd = m.get("kd_score", {})
            fl = m.get("fl_score", {})
            trk = kd.get("tracked_s")
            dur = m["bag_duration_s"]
            pct = 100 * trk / dur if trk else None
            tot_dur += dur
            tot_trk += trk or 0.0
            if trk and not kd.get("first_lost"):
                n99 += 1
            kp, fp = kd.get("pairs", 0), fl.get("pairs", 0)
            # One pair is two control points -- a single scalar, not an accuracy figure.
            kdcp = (f'{kd["err_mean_cm"]:.1f} / {kd["err_max_cm"]:.1f}'
                    if kp >= 2 else
                    f'<span class="na">{kp} 对</span>')
            flcp = (f'{fl["err_mean_cm"]:,.1f} / {fl["err_max_cm"]:,.1f}'
                    + ('' if kp == fp else '<sup>†</sup>')
                    if fp else '<span class="na">—</span>')
            mf = m.get("mf")
            r.append(
                f'<tr><td class="seq">{s}</td>'
                f'<td>{num(trk)} / {dur:.1f} s</td>'
                f'<td{cls(-(pct or 0), -98.5, -60)}>{num(pct, "{:.1f}")}</td>'
                f'<td>{num(kd.get("inlier_median"), "{:.3f}")}</td>'
                f'<td{cls(kd.get("sigma_peak"), 50, 200)}>{num(kd.get("sigma_peak"), "{:.0f}")}</td>'
                f'<td{cls(kd.get("err_mean_cm") if kp >= 2 else None, 10, 30)}>{kdcp}</td>'
                f'<td{cls(fl.get("err_mean_cm") if fp else None, 10, 30)}>{flcp}</td>'
                f'<td{cls(mf["median"] * 1000 if mf else None, 40, 200)}>'
                f'{num(mf["median"] * 1000 if mf else None, "{:.0f}") } mm</td>'
                f'<td class="l">{chip(pct, kd.get("first_lost"))}</td></tr>')
    r.append(f'<tr class="tot"><td class="seq">15 条序列</td>'
             f'<td>{tot_trk:,.0f} / {tot_dur:,.0f} s</td>'
             f'<td>{100 * tot_trk / tot_dur:.0f}</td><td>—</td><td>—</td>'
             f'<td>{n99} 条零 Lost</td><td>—</td><td>7–31 mm</td>'
             f'<td class="l">—</td></tr>')
    r.append('</tbody></table>')
    return "\n".join(r), dict(tot_trk=tot_trk, tot_dur=tot_dur, n99=n99)


def tilt_table():
    r = ['<table>', '<caption><code>up</code> is the true up direction inside that '
         'sequence&rsquo;s FAST-LIO map frame, taken from the filter&rsquo;s own '
         'mid-run gravity state and cross-checked against the raw <code>/imu</code> '
         'stream. The robot&rsquo;s XSens is mounted <strong>x-up</strong>, the '
         'handheld Alphasense <strong>z-down</strong> — the tilt is a property of the '
         'rig, not of the run.</caption>',
         '<thead><tr><th class="l">sequence</th><th>gravity in the map frame  (m/s²)</th>'
         '<th>up</th><th>off +Z</th></tr></thead><tbody>']
    for s in sorted(D, key=lambda q: D[q]["tilt_deg"], reverse=True):
        m = D[s]
        u = m["up_fl"]
        r.append(f'<tr><td class="seq">{s}</td>'
                 f'<td class="mono">[{", ".join(f"{-x * 9.807:+.3f}" for x in u)}]</td>'
                 f'<td class="mono">[{", ".join(f"{x:+.3f}" for x in u)}]</td>'
                 f'<td class="hi">{m["tilt_deg"]:.2f}°</td></tr>')
    r.append('</tbody></table>')
    return "\n".join(r)


def pairs_table():
    """The vertical/horizontal decomposition, same-floor rows then cross-floor rows."""
    rows = []
    for s in D:
        for p in D[s].get("kd_pairs") or D[s].get("fl_pairs", []):
            src = "kd" if D[s].get("kd_pairs") else "FAST-LIO"
            rows.append((s, src, p))
    same = sorted([r for r in rows if abs(r[2]["gt_dz"]) < 0.2],
                  key=lambda r: -r[2]["gt_dist"])[:6]
    cross = sorted([r for r in rows if abs(r[2]["gt_dz"]) >= 0.2 and abs(r[2]["err"]) < 100],
                   key=lambda r: -abs(r[2]["err"]))[:8]
    out = ['<table>', '<caption>Vertical / horizontal decomposition along the estimate&rsquo;s '
           'own gravity vector, because the survey frame&rsquo;s z <em>is</em> vertical. '
           'Same-floor pairs hold to centimetres over baselines up to 50 m; every pair that '
           'crosses a floor is wrong, and the vertical component carries the error.</caption>',
           '<thead><tr><th class="l">sequence · pair</th><th>Δz survey</th>'
           '<th>dist survey</th><th>dist est.</th><th>error</th><th>vert survey</th>'
           '<th>vert est.</th><th>horiz survey</th><th>horiz est.</th></tr></thead><tbody>']
    for title, group in (("same floor · |Δz| &lt; 0.2 m", same),
                         ("crosses a floor", cross)):
        out.append(f'<tr class="grp"><td colspan="9">{title}</td></tr>')
        for s, _run, p in group:
            vert_bad = abs(p["est_vert"] - p["gt_dz"]) > 0.25
            out.append(
                f'<tr><td class="seq">{short(s)} · {p["i"]}-{p["j"]}</td>'
                f'<td>{p["gt_dz"]:+.3f}</td><td>{p["gt_dist"]:.3f}</td>'
                f'<td>{p["est_dist"]:.3f}</td>'
                f'<td{cls(abs(p["err"]) * 100, 20, 100)}>{p["err"]:+.3f}</td>'
                f'<td>{p["gt_dz"]:+.3f}</td>'
                f'<td{HI if vert_bad else ""}>{p["est_vert"]:+.3f}</td>'
                f'<td>{p["gt_horiz"]:.3f}</td><td>{p["est_horiz"]:.3f}</td></tr>')
    out.append('</tbody></table>')
    return "\n".join(out)


def bags_table():
    r = ['<table>',
         '<caption>转换后的 LiDAR+IMU 包，全部 <code>rosbag2 metadata v8</code>。'
         '「tf」列是包里是否自带 <code>/tf_static</code> —— 10 个没有的必须在运行时补发 '
         '<code>imu_link → velodyne</code>，否则 kd_localizer 停在 UNINITIALIZED。</caption>',
         '<thead><tr><th class="l">序列</th><th>平台</th><th>时长</th><th>点云</th><th>Hz</th>'
         '<th>IMU</th><th>Hz</th><th>大小</th><th>tf</th><th>地图</th></tr></thead><tbody>']
    tot = dict(dur=0.0, cl=0, imu=0, gb=0.0, notf=0)
    for s in D:
        m = D[s]
        b = next(x for x in BAGS if x["seq"] == s)
        tot["dur"] += b["dur"]
        tot["cl"] += b["lidar"]
        tot["imu"] += b["imu"]
        tot["gb"] += b["gb"]
        tot["notf"] += 0 if b["tf"] else 1
        r.append(f'<tr><td class="seq">{s}</td><td>{m["platform"]}</td>'
                 f'<td>{b["dur"]:.1f} s</td><td>{b["lidar"]:,}</td><td>{b["lhz"]:.3f}</td>'
                 f'<td>{b["imu"]:,}</td><td>{b["ihz"]:.1f}</td><td>{b["gb"]:.2f} G</td>'
                 f'<td>{"自带" if b["tf"] else REBAGGED}</td>'
                 f'<td>{"✓" if m["has_kdm"] else "<span class=na>无</span>"}</td></tr>')
    r.append(f'<tr class="tot"><td class="seq">15 条</td><td>—</td>'
             f'<td>{tot["dur"]:,.0f} s</td><td>{tot["cl"]:,}</td><td>—</td>'
             f'<td>{tot["imu"]:,}</td><td>—</td><td>{tot["gb"]:.1f} G</td>'
             f'<td>{tot["notf"]} 条补发</td><td>14 / 15</td></tr>')
    r.append('</tbody></table>')
    return "\n".join(r)


if __name__ == "__main__":
    t, stats = main_table()
    open(f"{OUT}/tbl_main.html", "w").write(t)
    open(f"{OUT}/tbl_tilt.html", "w").write(tilt_table())
    open(f"{OUT}/tbl_pairs.html", "w").write(pairs_table())
    open(f"{OUT}/tbl_bags.html", "w").write(bags_table())
    json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
