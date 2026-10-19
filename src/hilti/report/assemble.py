#!/usr/bin/env python3
"""Splice the hand-written prose, the generated tables and the figures into one page.

Figures are inlined as base64 data URIs: the artifact CSP blocks every external host, so a
<img src> pointing anywhere else renders as a broken image.
"""
import base64
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hilti import dataset                                             # noqa: E402

# The prose is source, so it lives with the code: prose/head.html + body1..6.html,
# hand-written, with <!--TBL:x--> / <!--FIG:x--> / <!--SEQBLOCKS--> placeholders that
# this script fills from the generated artefacts. (It used to live in a scratch
# directory, which is why it nearly did not survive the session that wrote it.)
SRC = str(Path(__file__).resolve().parent / "prose")
OUT = dataset.report_dir()
DEST = os.environ.get("HILTI_REPORT_HTML", f"{OUT}/hilti_report.html")

ALT = {
    "fig_coverage": "全部 15 条序列的跟踪覆盖率与控制点精度对照",
    "fig_tilt": "15 条序列的重力倾角条形图，机器人约 90 度、手持约 178 度",
    "fig_vertical": "控制点误差分别对垂直落差与水平基线的散点图",
    "fig_sigma": "各序列位姿协方差因子的中位数到峰值区间，与 20 / 200 两条上限",
    "fig_agreement": "14 条序列的 kd 与 FAST-LIO 逐帧差值随时间变化",
    "fig_maps": "14 张 kdm 平面地图的平面图，按法向着色",
    "fig_incident": "site2_robot_1 的高度曲线与两系统差值，显示 t+632 秒的虚假爬升",
    "fig_divergence": "site2_handheld_5 的速度与离起点距离，显示 t+113 秒起发散",
}


def fig_alt(name):
    if name in ALT:
        return ALT[name]
    seq = name.replace("fig_", "")
    return f"{seq} 的地图平面图与剖面图、两条轨迹比对、内点率与协方差诊断曲线"


def inline_fig(name):
    p = f"{OUT}/{name}.png"
    if not os.path.exists(p):
        raise SystemExit(f"missing figure: {p}")
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return (f'<img src="data:image/png;base64,{b64}" alt="{fig_alt(name)}" '
            f'loading="lazy">')


def read(p):
    return open(p, encoding="utf8").read()


TABLES = {
    "main": f"{OUT}/tbl_main.html",
    "tilt": f"{OUT}/tbl_tilt.html",
    "pairs": f"{OUT}/tbl_pairs.html",
    "bags": f"{OUT}/tbl_bags.html",
}

parts = [read(f"{SRC}/head.html")]
body = "\n".join(read(f"{SRC}/{f}") for f in
                 ("body1.html", "body2.html", "body3.html", "body4.html",
                  "body5.html", "body6.html"))

body = body.replace("<!--SEQBLOCKS-->", read(f"{OUT}/seqblocks.html"))
body = re.sub(r"<!--TBL:(\w+)-->", lambda m: read(TABLES[m.group(1)]), body)
body = re.sub(r"<!--FIG:(\w+)-->", lambda m: inline_fig(m.group(1)), body)
parts.append(body)

html = "\n".join(parts)
open(DEST, "w", encoding="utf8").write(html)

left = re.findall(r"<!--(?:TBL|FIG):\w+-->", html)
size = len(html.encode())
print(f"wrote {DEST}  {size / 1e6:.2f} MB"
      f"  figures={html.count('data:image/png')}  unresolved={left}")
if size > 15.5e6:
    print("WARNING: approaching the 16 MB artifact limit")
