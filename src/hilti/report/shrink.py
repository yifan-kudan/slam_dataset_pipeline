#!/usr/bin/env python3
"""Palette-quantise the report PNGs.

These are line art and scatter plots over a flat ground -- a few hundred distinct colours at
most -- so an adaptive palette is visually lossless here while cutting the page weight that
has to be inlined as base64 by roughly two thirds.
"""
import glob
import os
import sys

from PIL import Image

total_before = total_after = 0
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "*.png")):
    before = os.path.getsize(f)
    im = Image.open(f).convert("RGB")
    q = im.quantize(colors=192, method=Image.MEDIANCUT, dither=Image.NONE)
    q.save(f, optimize=True)
    after = os.path.getsize(f)
    total_before += before
    total_after += after
    print(f"{f:34} {before / 1024:8.0f} KB -> {after / 1024:7.0f} KB")
print(f"\ntotal {total_before / 1e6:.2f} MB -> {total_after / 1e6:.2f} MB "
      f"(base64 ~{total_after * 1.37 / 1e6:.2f} MB)")
