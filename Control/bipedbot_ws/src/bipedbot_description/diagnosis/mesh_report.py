#!/usr/bin/env python3
# pip install trimesh numpy
import sys
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from pathlib import Path

urdf, meshdir = sys.argv[1], Path(sys.argv[2])
for link in ET.parse(urdf).getroot().findall("link"):
    vis = link.find("visual")
    if vis is None:
        continue
    org = np.array([float(v) for v in vis.find("origin").get("xyz").split()])
    name = Path(vis.find("geometry/mesh").get("filename")).name
    m = trimesh.load(meshdir / name, force="mesh")
    m.apply_scale(0.001)                      # mm -> m, same as your scale tag
    lo, hi = m.bounds
    c = (lo + hi) / 2 + org                   # centre in the link frame
    s = hi - lo
    print(f"{link.get('name'):18s} faces={len(m.faces):7d}")
    if "tyre" in name or "wheel" in name:
        r = max(s[1], s[2]) / 2
        print(f'   cylinder radius="{r:.4f}" length="{s[0]:.4f}"  origin xyz="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" rpy="0 1.5708 0"')
    else:
        print(f'   box size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}"  origin xyz="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"')