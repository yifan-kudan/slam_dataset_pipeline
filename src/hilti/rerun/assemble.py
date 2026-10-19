#!/usr/bin/env python3
"""Assemble the re-run report: prose + generated tables + dual-mode figures -> one HTML.

    assemble.py [--out <dir>] [-o report.html]

The page is written against a token system, never raw hex, so the light and dark
palettes swap in one place -- and every token is defined on bare `:root` first,
because the common case is a viewer on the OS default with no `data-theme` stamp at
all. Figures are PNGs and cannot adapt, so each one is rendered twice by
figures.py and the pair is swapped by the same three-state rule.

Prose lives in prose/*.html so the writing is editable without touching this file;
{{tbl_x}} and {{fig_x}} placeholders in it are substituted here. A missing table or
figure becomes a visible note rather than a silent hole.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hilti import dataset                                          # noqa: E402

HERE = Path(__file__).resolve().parent

HEAD = """<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Archivo:wght@500;600;700&\
family=IBM+Plex+Mono:wght@400;500&\
family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
/* ---- tokens -------------------------------------------------------------
   Cool concrete neutrals, because the subject is a construction site measured
   with a total station, and a hi-vis accent taken off the site itself. The
   bright accent is a FILL and a RULE only: on the light ground it is far below
   text contrast, so text uses the darker olive step of the same hue.
   Every token is defined here on bare :root -- the default viewer state stamps
   no data-theme at all, so a token whose only home is a media block or an
   attribute scope would simply never apply. */
:root {
  color-scheme: light;
  --ground:      #e9ecee;
  --surface:     #fbfcfd;
  --surface-2:   #f0f2f4;
  --ink:         #0f151a;
  --ink-2:       #3f4d57;
  --muted:       #6f7f8a;
  --rule:        #d2d8dd;
  --rule-firm:   #b6bfc7;
  --accent-ink:  #4a6f05;
  --accent:      #a8d63a;
  --accent-wash: #eef6dc;
  --link:        #1d5f8a;
  --good:        #0ca30c;
  --warning:     #fab219;
  --critical:    #d03b3b;
  --fig-shade:   rgba(15,21,26,.07);

  --display: "Archivo", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --body: "Source Serif 4", "Iowan Old Style", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground:      #0e1215;
    --surface:     #161c21;
    --surface-2:   #1d242a;
    --ink:         #e9eef2;
    --ink-2:       #bdc8d0;
    --muted:       #8595a0;
    --rule:        #252d34;
    --rule-firm:   #38434c;
    --accent-ink:  #b7e04a;
    --accent:      #8ab52c;
    --accent-wash: #1c2415;
    --link:        #79b4dd;
    --fig-shade:   rgba(255,255,255,.07);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground:      #0e1215;
  --surface:     #161c21;
  --surface-2:   #1d242a;
  --ink:         #e9eef2;
  --ink-2:       #bdc8d0;
  --muted:       #8595a0;
  --rule:        #252d34;
  --rule-firm:   #38434c;
  --accent-ink:  #b7e04a;
  --accent:      #8ab52c;
  --accent-wash: #1c2415;
  --link:        #79b4dd;
  --fig-shade:   rgba(255,255,255,.07);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--body);
  font-size: 17px;
  line-height: 1.66;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1140px; margin: 0 auto; padding: 0 24px 110px; }
.col { max-width: 70ch; }

h1, h2, h3, .stat .v, .eyebrow, th, .pill, .stat .k {
  font-family: var(--display);
}
h1 {
  font-size: clamp(2.05rem, 1.2rem + 3vw, 3.35rem);
  line-height: 1.02; font-weight: 700; letter-spacing: -.028em;
  margin: 0 0 .4em; text-wrap: balance;
}
h2 {
  font-size: 1.58rem; line-height: 1.18; font-weight: 650; letter-spacing: -.017em;
  margin: 0 0 .45em; text-wrap: balance;
}
h3 {
  font-size: 1.05rem; font-weight: 650; letter-spacing: -.005em;
  margin: 2.1em 0 .3em; text-wrap: balance;
}
p { margin: 0 0 1.05em; }
a { color: var(--link); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:focus-visible, summary:focus-visible {
  outline: 2px solid var(--accent-ink); outline-offset: 3px; border-radius: 2px;
}
em { font-style: italic; }
b, strong { font-weight: 600; }

/* ---- the datum rule ----------------------------------------------------
   Every section opens on one. The whole report is about levels -- floors,
   storeys, the vertical component of the error -- so the section marker is a
   levelling datum: a short hi-vis tick sitting on a hairline that runs the
   column. It encodes the same thing the eyebrow says, and nothing else. */
section { margin: 0 0 68px; scroll-margin-top: 20px; }
.datum {
  display: flex; align-items: center; gap: 14px;
  border-top: 1px solid var(--rule-firm);
  margin: 0 0 22px; padding-top: 0;
}
.datum::before {
  content: ""; width: 46px; height: 4px; background: var(--accent);
  margin-top: -2px; flex: none;
}
.eyebrow {
  font-family: var(--mono); font-size: .68rem; font-weight: 500;
  letter-spacing: .15em; text-transform: uppercase; color: var(--muted);
  margin: 0; padding-top: 8px;
}
.eyebrow .n { color: var(--accent-ink); }

header.masthead { padding: 84px 0 30px; }
header.masthead .datum { margin-bottom: 30px; }
.lede {
  font-size: 1.2rem; line-height: 1.5; color: var(--ink-2); max-width: 60ch;
  margin: 0 0 1.5em;
}
.meta {
  display: flex; flex-wrap: wrap; gap: 6px 30px;
  font-family: var(--mono); font-size: .76rem; color: var(--muted);
  border-top: 1px solid var(--rule); padding-top: 16px;
}
.meta b { color: var(--ink-2); font-weight: 500; }

/* ---- stat strip ---- */
.stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(166px, 1fr));
  gap: 1px; background: var(--rule); border: 1px solid var(--rule);
  margin: 34px 0 10px;
}
.stat { background: var(--surface); padding: 17px 19px; }
.stat .k {
  font-family: var(--mono); font-size: .64rem; font-weight: 500; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted); display: block; margin-bottom: 8px;
}
.stat .v {
  font-size: 1.95rem; font-weight: 700; letter-spacing: -.028em; line-height: 1;
  display: block;
}
.stat .v .u { font-size: .78rem; font-weight: 500; color: var(--muted); margin-left: 4px; }
.stat .n {
  font-family: var(--body); font-size: .84rem; color: var(--muted); display: block;
  margin-top: 7px; line-height: 1.4;
}

/* ---- tables ---- */
.tw {
  overflow-x: auto; margin: 24px 0 6px; border: 1px solid var(--rule);
  background: var(--surface);
}
table {
  border-collapse: collapse; width: 100%;
  font-family: var(--mono); font-size: .78rem; font-variant-numeric: tabular-nums;
}
th {
  text-align: left; font-size: .63rem; font-weight: 600; letter-spacing: .07em;
  text-transform: uppercase; color: var(--muted); padding: 11px 12px;
  border-bottom: 1px solid var(--rule-firm); white-space: nowrap;
  background: var(--surface-2);
}
td {
  padding: 8px 12px; border-bottom: 1px solid var(--rule); white-space: nowrap;
  color: var(--ink-2);
}
/* The run tables carry 13-14 columns. Tightening them fits the column measure on a
   desktop; the .tw container still scrolls on anything narrower. */
.tw.wide table { font-size: .715rem; }
.tw.wide th { padding: 10px 7px; font-size: .58rem; letter-spacing: .05em; }
.tw.wide td { padding: 7px 7px; }
tbody tr:last-child td { border-bottom: 0; }
th.num, td.num { text-align: right; }
td code { font-size: 1em; background: none; padding: 0; color: var(--ink); }
.tnote {
  font-family: var(--body); font-size: .88rem; color: var(--muted);
  max-width: 84ch; margin: 10px 0 0;
}
.tnote b { color: var(--ink-2); }

code {
  font-family: var(--mono); font-size: .84em; background: var(--surface-2);
  padding: 1px 5px; color: var(--ink-2);
}
pre {
  font-family: var(--mono); font-size: .76rem; line-height: 1.62;
  background: var(--surface); border: 1px solid var(--rule);
  border-left: 3px solid var(--accent); padding: 14px 16px;
  overflow-x: auto; margin: 20px 0; color: var(--ink-2);
}
pre code { background: none; padding: 0; font-size: 1em; }
.cmt { color: var(--muted); }
.add { color: var(--good); }
.del { color: var(--critical); }

.pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: var(--mono); font-size: .64rem; font-weight: 500; letter-spacing: .06em;
  text-transform: uppercase; padding: 2px 7px; white-space: nowrap;
  border: 1px solid currentColor;
}
.pill::before {
  content: ""; width: 5px; height: 5px; border-radius: 50%; background: currentColor;
}
.pill.ok { color: var(--good); }
.pill.warn { color: var(--warning); }
.pill.no { color: var(--critical); }

/* ---- figures: two PNGs, one per theme, swapped on the same three states ---- */
figure { margin: 30px 0 6px; }
figure .fw { overflow-x: auto; border: 1px solid var(--rule); background: var(--surface); }
figure img { display: block; width: 100%; height: auto; min-width: 560px; }
img.fig-dark { display: none; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) img.fig-light { display: none; }
  :root:not([data-theme="light"]) img.fig-dark { display: block; }
}
:root[data-theme="dark"] img.fig-light { display: none; }
:root[data-theme="dark"] img.fig-dark { display: block; }
figcaption {
  font-family: var(--body); font-size: .88rem; color: var(--muted); margin-top: 11px;
  max-width: 84ch;
}
figcaption b { color: var(--ink-2); }
.missing {
  font-family: var(--mono); font-size: .78rem; color: var(--critical);
  border: 1px dashed var(--critical); padding: 12px 14px; margin: 24px 0;
}

.note {
  border-left: 3px solid var(--accent); background: var(--accent-wash);
  padding: 14px 17px; margin: 24px 0; font-size: .97rem; max-width: 78ch;
}
.note .eyebrow { padding-top: 0; margin-bottom: .5em; color: var(--accent-ink); }
.note p:last-child { margin-bottom: 0; }

ul, ol { margin: 0 0 1.05em; padding-left: 1.3em; max-width: 70ch; }
li { margin-bottom: .45em; }
li::marker { color: var(--muted); }

/* ---- the pipeline: a real sequence, so it is numbered ---- */
.steps { display: grid; gap: 0; margin: 26px 0 8px; border-top: 1px solid var(--rule); }
.step {
  display: grid; grid-template-columns: 3.2rem 1fr; gap: 0 20px;
  padding: 18px 0; border-bottom: 1px solid var(--rule);
}
.step .n {
  font-family: var(--mono); font-size: .8rem; font-weight: 500; color: var(--accent-ink);
  padding-top: 2px;
}
.step h3 { margin: 0 0 .25em; font-size: 1rem; }
.step p { margin: 0 0 .5em; font-size: .95rem; }
.step p:last-child { margin-bottom: 0; }
.step code { font-size: .8rem; }

.foot {
  font-family: var(--mono); font-size: .72rem; color: var(--muted); line-height: 1.7;
  border-top: 1px solid var(--rule-firm); margin-top: 70px; padding-top: 20px;
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""


def embed_figure(out_dir, name, caption):
    """The light/dark pair as one <figure>, or a visible note if it was not drawn."""
    parts = []
    for mode in ("light", "dark"):
        path = os.path.join(out_dir, f"{name}_{mode}.png")
        if not os.path.isfile(path):
            return (f'<div class="missing">figure {name} ({mode}) is missing — '
                    f'run <code>figures.py --only {name}</code></div>')
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        parts.append(f'<img class="fig-{mode}" alt="{caption[:120]}" '
                     f'src="data:image/png;base64,{b64}">')
    return (f'<figure><div class="fw">{"".join(parts)}</div>'
            f'<figcaption>{caption}</figcaption></figure>')


CAPTIONS = {
    "fig_frames": "Clouds FAST-LIO processed, as a share of the count in the bag's own "
                  "metadata. <b>Orange</b> is the same sequence's pre-fix run, <b>blue</b> "
                  "this one. Nothing about FAST-LIO changed between them: the loss was "
                  "Fast DDS discarding 1.37 MB samples at the writer because the default "
                  "shared-memory segment could not hold them, and the fix is an XML "
                  "profile exported to the node and the bag player alike.",
    "fig_cp": "Mean control-point pair-distance error, log axis. Three configurations per "
              "sequence where all three exist: FAST-LIO's own trajectory, kd_localizer in "
              "a map built from that same sequence, and kd_localizer in its site's hub "
              "map. Pair distances are frame-invariant, so no alignment enters any of "
              "these numbers.",
    "fig_matrix": "Row = the sequence a map was built from; column = the sequence placed "
                  "in it. The cell is the fraction of the column's cloud landing within "
                  "0.5 m of a row point after registration. The matrix is asymmetric on "
                  "purpose — a small map inside a large one scores high going one way and "
                  "low going the other, and it is the row mean that says which map can "
                  "serve as the site's hub.",
    "fig_sites": "Each site's hub map as a top-down occupancy histogram, with every "
                 "sequence's own FAST-LIO route pushed through its registration onto it. "
                 "Where a route runs off the shaded area, the map does not cover that "
                 "part of the walk — which is the geometric limit on what any localizer "
                 "could do there.",
    "fig_cross": "Distance between kd_localizer's pose in the hub map and the same "
                 "sequence's own FAST-LIO trajectory, transformed into that map by the "
                 "registration. Log axis; the dashed line is 1 m. This is the measurement "
                 "the control points cannot make — 2–8 surveyed points per sequence "
                 "against 5–10 Hz here — and it is how a silent slip becomes visible.",
    "fig_own": "The same measurement for the own-map runs, where the .kdm was built from "
               "the sequence's own cloud and shares its frame exactly. These curves are "
               "the floor: whatever a cross-map run costs, it is on top of this.",
    "fig_health": "The two quantities the diagnostics threshold, over the own-map runs. "
                  "Left: the ray inlier ratio, 5th percentile to median — the reason "
                  "<code>min_ray_inlier_ratio</code> is disabled on this dataset. Right: "
                  "the pose covariance factor, median to peak, against the limit of 200. "
                  "Both are taken over TRACKING rows only.",
}


def stat(k, v, unit="", note=""):
    u = f'<span class="u">{unit}</span>' if unit else ""
    n = f'<span class="n">{note}</span>' if note else ""
    return f'<div class="stat"><span class="k">{k}</span><span class="v">{v}{u}</span>{n}</div>'


def stats_strip(data):
    """The headline numbers, computed here so the strip cannot go stale.

    Everything is derived from data.json -- the same file the tables and figures
    read -- so a figure and the number above it cannot disagree.
    """
    seqs = list(data.get("sequences", {}).values())
    fl = [r["fastlio"] for r in seqs if r.get("fastlio")]
    kd = [r["localizer"] for r in seqs if r.get("localizer")]
    cross = [c["localizer"] for c in data.get("cross", {}).values() if c.get("localizer")]
    bag_total = sum(r.get("bag_duration_s") or 0 for r in seqs)

    cells = []
    if fl:
        worst = min(f["frame_pct"] for f in fl)
        cells.append(stat("frame rate", f"{worst:.1f}", "%",
                          f"worst of {len(fl)} sequences; every pre-fix run was 45–53 %"))
    if kd:
        # 0.97 and not 0.99: the monitor comes up a beat after the bag starts, so a run
        # that reaches the end still reports a span a second or two short of the duration.
        full = sum(1 for r in seqs if r.get("localizer")
                   and r["localizer"]["tracked_s"] >= 0.97 * (r.get("bag_duration_s") or 1e9))
        cells.append(stat("own-map, full bag", f"{full}", f" / {len(kd)}",
                          "runs reach the end with no LOST"))
        # Agreement with the sequence's own odometry: the one figure every run produces,
        # and dense enough (5-10 Hz) to mean something. Control-point scores are missing
        # wherever a run ends before its second surveyed point, so a median over them
        # would silently be a median over a different set per configuration.
        mf = sorted(k["mapframe_median"] for k in kd if k.get("mapframe_median") is not None)
        if mf:
            cells.append(stat("own-map agreement", f"{100 * mf[len(mf) // 2]:.1f}", " cm",
                              f"median distance from its own FAST-LIO trajectory, "
                              f"worst {100 * mf[-1]:.1f} cm"))
    if cross:
        n_pairs = len(data.get("cross", {}))
        full = sum(1 for c in data["cross"].values()
                   for d in (c.get("localizer"), c.get("localizer_v2")) if d
                   and d["tracked_s"] >= 0.97 * ((data["sequences"].get(c["seq"]) or {})
                                                 .get("bag_duration_s") or 1e9))
        cells.append(stat("cross-sequence", f"{full}", f" / {n_pairs}",
                          "pairs track their full bag, in either configuration"))
    if not cells:
        return '<div class="missing">no results in data.json yet</div>'
    return f'<div class="stats">{"".join(cells)}</div>'


def build(out_dir, out_html, title):
    data_path = os.path.join(out_dir, "data.json")
    data = json.load(open(data_path)) if os.path.isfile(data_path) else {}
    body = []
    prose_dir = HERE / "prose"
    files = sorted(prose_dir.glob("*.html"))
    if not files:
        raise SystemExit(f"no prose in {prose_dir}")
    for f in files:
        body.append(f.read_text(encoding="utf-8"))
    html = "\n".join(body)

    def sub_tbl(m):
        path = os.path.join(out_dir, f"tbl_{m.group(1)}.html")
        if not os.path.isfile(path):
            return (f'<div class="missing">table {m.group(1)} is missing — '
                    f'run <code>tables.py</code></div>')
        return open(path, encoding="utf-8").read()

    html = html.replace("{{stats}}", stats_strip(data))
    html = re.sub(r"\{\{tbl_([a-z_0-9]+)\}\}", sub_tbl, html)
    html = re.sub(r"\{\{(fig_[a-z_0-9]+)\}\}",
                  lambda m: embed_figure(out_dir, m.group(1),
                                         CAPTIONS.get(m.group(1), "")), html)
    # Numbers the prose quotes from data.json rather than hard-coding.
    for key, value in (data.get("headline") or {}).items():
        html = html.replace(f"{{{{n_{key}}}}}", str(value))
    left = re.findall(r"\{\{[a-z_0-9]+\}\}", html)
    if left:
        print(f"warning: unresolved placeholders {sorted(set(left))}", file=sys.stderr)

    # A plain replace, not str.format: HEAD is mostly CSS and every brace in it would
    # have to be doubled.
    page = HEAD.replace("__TITLE__", title) + '<div class="wrap">\n' + html + "\n</div>\n"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out_html}  ({os.path.getsize(out_html) / 1e6:.2f} MB)")
    return out_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="report directory (default: <results>/rerun_report)")
    ap.add_argument("-o", "--html")
    ap.add_argument("--title", default="Fifteen Sequences, Three Hub Maps")
    a = ap.parse_args()
    out_dir = a.out or os.path.join(dataset.results_dir(), "rerun_report")
    build(out_dir, a.html or os.path.join(out_dir, "report.html"), a.title)


if __name__ == "__main__":
    main()
