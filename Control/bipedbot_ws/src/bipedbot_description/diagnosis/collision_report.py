#!/usr/bin/env python3
"""Suggest collision shapes that follow each link's real geometry.

Usage (from the package folder):
    python collision_report.py urdf/bipedbot.urdf.xacro meshes
    python collision_report.py urdf/bipedbot.urdf.xacro meshes --hull meshes_collision

Needs: pip install numpy trimesh fast-simplification
For every link with a <visual> mesh it prints
  * tyres/wheels: a cylinder (axle along x)
  * everything else: a tight ORIENTED box (rotated to fit slanted legs)
and, with --hull, also exports a low-poly convex hull per link.
All values are in the link frame, in metres, ready to paste into <collision>.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh
from trimesh import transformations as tf

ap = argparse.ArgumentParser()
ap.add_argument("urdf")
ap.add_argument("meshdir")
ap.add_argument("--hull", help="folder to export convex-hull STLs into")
ap.add_argument("--faces", type=int, default=100, help="target faces per hull")
a = ap.parse_args()

meshdir = Path(a.meshdir)
if a.hull:
    Path(a.hull).mkdir(exist_ok=True)


def vec(s):
    return [float(v) for v in s.split()] if s else [0.0, 0.0, 0.0]


def fmt(v):
    return " ".join(f"{x:.4f}" for x in v)


for link in ET.parse(a.urdf).getroot().findall("link"):
    vis = link.find("visual")
    if vis is None:
        continue
    mesh_el = vis.find("geometry/mesh")
    org = vis.find("origin")
    xyz = vec(org.get("xyz")) if org is not None else [0.0] * 3
    rpy = vec(org.get("rpy")) if org is not None else [0.0] * 3
    name = Path(mesh_el.get("filename")).name

    m = trimesh.load(meshdir / name, force="mesh")
    m.apply_scale(vec(mesh_el.get("scale", "1 1 1")))
    T = tf.euler_matrix(*rpy, axes="sxyz")
    T[:3, 3] = xyz
    m.apply_transform(T)          # mesh is now in the link frame, metres

    lname = link.get("name")
    print(f"\n{lname}  ({len(m.faces)} faces)")

    if "tyre" in lname or "wheel" in lname:
        lo, hi = m.bounds
        c, s = (lo + hi) / 2, hi - lo
        print(f'  <collision><origin xyz="{fmt(c)}" rpy="0 1.5708 0"/>'
              f'<geometry><cylinder radius="{max(s[1], s[2]) / 2:.4f}" '
              f'length="{s[0]:.4f}"/></geometry></collision>')
    else:
        box = m.bounding_box_oriented
        Tb = box.primitive.transform
        r = tf.euler_from_matrix(Tb, axes="sxyz")
        print(f'  <collision><origin xyz="{fmt(Tb[:3, 3])}" rpy="{fmt(r)}"/>'
              f'<geometry><box size="{fmt(box.primitive.extents)}"/></geometry></collision>')

    if a.hull:
        h = m.convex_hull
        if len(h.faces) > a.faces:
            try:
                h = h.simplify_quadric_decimation(face_count=a.faces)
            except Exception as e:
                print(f"  (could not decimate hull: {e})")
        out = Path(a.hull) / f"{Path(name).stem}_col.stl"
        h.export(out)
        print(f"  hull -> {out} ({len(h.faces)} faces); use with NO origin and NO scale:")
        print(f'  <collision><geometry><mesh filename="package://bipedbot_description/'
              f'{Path(a.hull).name}/{out.name}"/></geometry></collision>')
