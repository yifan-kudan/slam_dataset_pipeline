#!/usr/bin/env python3
"""Emit the per-sequence blocks for section 07.

The prose per sequence is written by hand; every number in the stat line is read from
data.json so the two can never drift apart.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hilti import dataset                                             # noqa: E402

OUT = dataset.report_dir()
D = json.load(open(f"{OUT}/data.json"))

# ordered as the report reads them: cleanest platform first
ORDER = [
    ("机器人平台 · site2 · 只有 robot_1 跨楼层", ["site2_robot_3", "site2_robot_2",
                                              "site2_robot_1"]),
    ("site3 手持 · 四条全部单层", ["site3_handheld_1", "site3_handheld_3",
                              "site3_handheld_4", "site3_handheld_2"]),
    ("site1 手持 · 含无照明封闭楼梯井", ["site1_handheld_4", "site1_handheld_2",
                                 "site1_handheld_3", "site1_handheld_1",
                                 "site1_handheld_5"]),
    ("site2 手持 · 每条路线都跨楼层", ["site2_handheld_6", "site2_handheld_4",
                                "site2_handheld_5"]),
]

NOTE = {
 "site2_robot_3":
   "本数据集最干净的一条。全程 100%、零 Lost，内点率中位 0.912，协方差峰值只有 13.3 —— "
   "库默认上限 20 都够用。与 FAST-LIO 一致到 <strong>7 mm 中位 / 145 mm 最大</strong>，"
   "是 14 条里最紧的。控制点 10 对全部在 20 cm 以下 —— 而它的 5 个控制点垂直只跨 "
   "<strong>0.18 m</strong>，即整条路线在同一层楼面上，这正是它没有跨层误差的原因。",
 "site2_robot_2":
   "同样零 Lost。它的控制点垂直只跨 0.05 m —— 三条机器人序列里最平的一条。"
   "只有 3 对控制点，所以 5.2 cm 这个数字的统计量不大，"
   "但它与 FAST-LIO 的 4.9 cm 几乎相同，两个独立系统在同一批点上一致。"
   "地图是 14 张里最偏斜的（84% 水平 / 13% 垂直），"
   "而它照样跟通 —— 楼板足够约束一个在楼板上行驶的平台。",
 "site2_robot_1":
   "最长的一条（700 s），也是<strong>三条机器人序列里唯一真正跨楼层的</strong>"
   "（控制点垂直跨 5.66 m），因此也是唯一有跨层误差的一条。"
   "它同时是「定位器比里程计更准」有多个控制点支撑的一条："
   "kd 20.6 / 81.0 cm 对 FAST-LIO 28.9 / 102.3 cm，21 对控制点。"
   "原因不是它更聪明，而是它<em class=\"i\">拒绝跟随</em>里程计在 t+632 s 的一次虚假爬升 —— "
   "详见 <span class=\"mono\">§08</span>。协方差峰值 41.1 出现在位姿健康的时刻，"
   "是「上限不能用默认 20」的直接证据。",
 "site3_handheld_1":
   "全部 15 条里精度最好的：<strong>1.7 / 3.1 cm</strong>，6 对控制点，99% 跟通。"
   "高度曲线上三次明显的抬升是操作者把杆举起又放下，四个控制点都落在同一层楼面 —— "
   "survey 的四个点高度几乎相同，这是「单层路线」最直观的样子。",
 "site3_handheld_3":
   "28 对控制点，全部 15 条里控制点最多的一条，因此 3.5 / 12.0 cm 这个数字的统计量最强。"
   "100% 跟通，协方差峰值 10.7 —— 全部序列里最低。"
   "它的 8 个控制点垂直只跨 0.04 m，全部同层；12 cm 的最大值出现在一对 23.6 m 基线的点对上，"
   "而更长的 60.8 m 基线点对反而更准 —— 再次说明误差与水平距离无关。",
 "site3_handheld_4":
   "99% 跟通，3.9 / 8.2 cm。与 FAST-LIO 的一致性是 14 条里最松的（31 mm 中位），"
   "但绝对值仍在厘米级，且两个系统对控制点给出几乎相同的答案（3.9 对 3.8 cm）——"
   "说明 31 mm 是两条轨迹的细节差异，不是谁错了。",
 "site3_handheld_2":
   "site3 四条里最差但依然合格：6.9 / 14.0 cm，15 对控制点，99% 跟通。"
   "内点率中位 0.600 是四条里最低的 —— 平面图能看出原因，"
   "它走的是一条极窄的长走廊（extent 30 × 85 m），可用的墙面几何在任一时刻都很少。",
 "site1_handheld_4":
   "site1 五条里跟得最久的（90%）。跟踪期间精度 <strong>1.2 / 2.6 cm，比 FAST-LIO 的 "
   "2.4 / 3.7 cm 更准</strong> —— 而且是同一批 3 对控制点，可以直接相减。"
   "诊断曲线的形状是 site1 这一组的典型："
   "内点率一路健康，协方差在最后 10% 才拔起并撞上 200。",
 "site1_handheld_2":
   "81% 跟通，跟踪期间 1.9 / 4.5 cm（6 对）。注意它比 FAST-LIO 多覆盖了一个控制点，"
   "所以两者的 6 对与 3 对<em class=\"i\">不是同一批点</em>，不能直接相减 —— "
   "这是本次重跑相对上一份报告变化最大的一个数字（那次只有 3 对，报 1.1 cm）。"
   "一致性图上它在竖线处从厘米级直接跳走，"
   "是「末段真发散、此前完全正确」最干净的一个例子。",
 "site1_handheld_3":
   "内点率中位 0.300，全部 15 条里最低 —— 它的路线在几何最退化的一段。"
   "尽管如此跟踪期间仍有 4.1 / 8.9 cm。这条序列的 kd 运行比 FAST-LIO 多覆盖了一个控制点"
   "（6 对 vs 3 对），因为 FAST-LIO 的首帧输出比包头晚了一点，错过了第一个点 —— "
   "两列因此不可直接相减。",
 "site1_handheld_1":
   "68% 跟通，2.4 / 6.6 cm，与 FAST-LIO 的 2.5 / 6.5 cm 在同一批 6 对点上几乎相同。"
   "它的 Lost 时刻内点率只有 <strong>0.33</strong> —— "
   "与 <code>site2_handheld_6</code> 的 0.98 形成对照：这一条是真的失去支撑，"
   "不是误报。失败位置落在上游报告记录过的 t≈167–186 s <strong>无照明封闭楼梯井</strong>里，"
   "FAST-LIO 在同一段出了 10.5° 偏航 —— 两个系统在同一处几何退化上失败，不是两回事。",
 "site1_handheld_5":
   "五条里最差：36% 跟通。它要爬三层楼。跟踪窗口内只够到 2 个控制点，即<strong>只有 1 对</strong>，"
   "那一对读到 22.9 cm —— 按本报告口径这不作精度数字，只说明它没有整体跑飞。"
   "对照 FAST-LIO 自己在这条上错到 68.0 / 104.3 cm（3 对），"
   "它是 <span class=\"mono\">§04</span> 那张表里跨层误差最大的老序列。"
   "另外它在上限 1000 下反而更短（22.8 s）—— 撞上了 ConSLAM 记录过的间歇性 ESKF 启动失败。",
 "site2_handheld_6":
   "<strong>确证的误报。</strong>t+43.1 s 判 <code>Pose covariance is too high (Rotation Y)</code>，"
   "而那一刻内点率 <strong>0.98</strong>，且整个跟踪段与地图对应的 FAST-LIO 轨迹一致到 "
   "<strong>16 mm 中位 / 152 mm 最大</strong> —— 位姿是对的。"
   "注意这里用的地图是从它的 <code>-r 0.15</code> 云重建的那张（那次轨迹 15.7 cm），"
   "不是已被证明错 1.4 m 的 r0.3 版本。把上限抬到 2000 也没跟通，"
   "而是在 t+56.5 s 让协方差涨到中位数的 500 倍才失败 —— 那才是真发散。",
 "site2_handheld_4":
   "18% 跟通就判 Lost（<code>Rotation Z</code>，协方差峰值 190.3）。"
   "这个数字衡量的<em class=\"i\">不是</em>定位器精度，而是「能不能在一张错的地图里定位」："
   "它的地图 extent 149.7 m 而路径只有 63.2 m。高度面板上看得清楚 —— "
   "FAST-LIO 在 t+20 s 后一路爬到 +5 m 并停在那里，而 cp1 的测绘高度是 +5.5 m，"
   "两者差的那 1 m 就是 170.8 cm 那个误差的主要来源。",
 "site2_handheld_5":
   "<strong>15 条里唯一没有 kd_localizer 运行的一条</strong> —— 它的 FAST-LIO 彻底发散，"
   "没有可用的点云可以建图，因此也就没有可以定位的地图。"
   "发散的形状很明确，单独见 <span class=\"mono\">§08</span>。",
}


def block(seq):
    m = D[seq]
    kd = m.get("kd_score", {})
    fl = m.get("fl_score", {})
    trk = kd.get("tracked_s")
    dur = m["bag_duration_s"]
    bits = [f'{dur:.0f} s / {m["bag_clouds"]:,} 帧']
    if trk:
        bits.append(f'跟踪 {trk:.1f} s（{100 * trk / dur:.0f}%）')
    else:
        bits.append('无运行')
    if kd.get("pairs"):
        bits.append(f'控制点 {kd["err_mean_cm"]:.1f} / {kd["err_max_cm"]:.1f} cm'
                    f'（{kd["pairs"]} 对）')
    elif fl.get("pairs"):
        bits.append(f'FAST-LIO 控制点 {fl["err_mean_cm"]:,.1f} cm')
    if m.get("mf"):
        bits.append(f'vs FAST-LIO {m["mf"]["median"] * 1000:.0f} mm 中位')
    if kd.get("first_lost"):
        bits.append(f'Lost：{kd["first_lost"][0].replace("Pose covariance is too high", "协方差过高")}')
    chip = ('<span class="chip pass">全程</span>' if trk and 100 * trk / dur >= 99 else
            '<span class="chip lvl">末段发散</span>' if trk and 100 * trk / dur >= 60 else
            '<span class="chip div">早期 Lost</span>' if trk else
            '<span class="chip div">无地图</span>')
    fig = f"<!--FIG:fig_{seq}-->" if seq != "site2_handheld_5" else ""
    figblock = f'<div class="figmat">{fig}</div>' if fig else ""
    return (f'  <div class="seqblock">\n'
            f'    <div class="seqhead"><h4>{seq}</h4>{chip}</div>\n'
            f'    <div class="col"><p style="font-family:var(--mono);font-size:12.5px;'
            f'color:var(--muted)">{" · ".join(bits)}</p>\n'
            f'    <p>{NOTE[seq]}</p></div>\n'
            + (f'    <figure>{figblock}</figure>\n' if fig else "")
            + '  </div>')


parts = []
for title, seqs in ORDER:
    parts.append(f'  <h3>{title}</h3>')
    parts += [block(s) for s in seqs]
open(f"{OUT}/seqblocks.html", "w").write("\n".join(parts))
print(f"wrote seqblocks.html with {sum(len(s) for _, s in ORDER)} sequences")
