#!/usr/bin/env python3
"""Per-node CPU / RSS / threads for kd_localizer and kd_relocalizer, sampled from the HOST's
/proc (container processes are visible there), once a second, until killed.

    procmon.py out.csv

CPU is the utime+stime delta over the tick in cores (1.0 = one core busy), so the localizer's
alignment threads can read above 1. Reads /proc directly because the Jetson host has no psutil.
"""
import os
import sys
import time

NAMES = ("kd_localizer", "kd_relocalizer")
HZ = os.sysconf("SC_CLK_TCK")


def sample():
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                st = f.read()
            comm = st[st.index("(") + 1:st.rindex(")")]
            if comm not in NAMES:
                continue
            fields = st[st.rindex(")") + 2:].split()
            ticks = int(fields[11]) + int(fields[12])        # utime + stime
            threads = int(fields[17])
            rss_kb = 0
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
            prev = out.get(comm)
            # Several processes of one name (launch wrapper + node) are summed.
            out[comm] = (ticks + (prev[0] if prev else 0), rss_kb + (prev[1] if prev else 0),
                         threads + (prev[2] if prev else 0))
        except (OSError, ValueError, IndexError):
            continue
    return out


def main():
    with open(sys.argv[1], "w", buffering=1) as f:
        f.write("t_s," + ",".join(f"{n}_cpu_cores,{n}_rss_mb,{n}_threads" for n in NAMES) + "\n")
        t0, last, last_t = time.time(), {}, time.time()
        while True:
            time.sleep(1.0)
            now, cur = time.time(), sample()
            row = [f"{now - t0:.1f}"]
            for n in NAMES:
                if n in cur:
                    d = (cur[n][0] - last[n][0]) / HZ / (now - last_t) if n in last else 0.0
                    row += [f"{max(d, 0.0):.3f}", f"{cur[n][1] / 1024:.1f}", str(cur[n][2])]
                else:
                    row += ["", "", ""]
            f.write(",".join(row) + "\n")
            last, last_t = cur, now


if __name__ == "__main__":
    main()
