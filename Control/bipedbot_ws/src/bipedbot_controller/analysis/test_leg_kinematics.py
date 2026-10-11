"""Independent 3-D check of leg_height_controller's joint mapping, using the URDF axes.
Usage: python3 test_leg_kinematics.py path/to/bipedbot.urdf.xacro [dir containing leg_height_controller.py]"""
import sys, types, math
import numpy as np
for m in ("rclpy", "rclpy.node", "std_msgs", "std_msgs.msg"):          # stub ROS for the import
    sys.modules[m] = types.ModuleType(m)
sys.modules["rclpy.node"].Node = object
sys.modules["std_msgs.msg"].Float64 = object
sys.modules["std_msgs.msg"].Float64MultiArray = object
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else ".")
import leg_height_controller as L
import xml.etree.ElementTree as ET

def rodrigues(axis, q):
    a = np.array(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K

root = ET.parse(sys.argv[1]).getroot()
J = {}
for j in root.findall("joint"):
    ax = j.find("axis")
    J[j.get("name")] = (np.array([float(v) for v in j.find("origin").get("xyz").split()]),
                        np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None)

def axle_pos(side, q_hip, q_knee):
    """Wheel-axle position in base_link from the URDF joint chain (no planar assumptions)."""
    p = np.zeros(3); R = np.eye(3)
    for jn, q in ((f"{side}_top_leg_joint", q_hip), (f"{side}_bottom_leg_joint", q_knee),
                  (f"{side}_motor_joint", 0.0)):
        o, ax = J[jn]
        p = p + R @ o
        if ax is not None:
            R = R @ rodrigues(ax, q)
    return p

geo = L.LegGeometry(J["right_bottom_leg_joint"][0][1:], J["right_motor_joint"][0][1:])
signs = [-1.0, -1.0, -1.0, 1.0]
hip_z = J["right_top_leg_joint"][0][2]; hip_y = J["right_top_leg_joint"][0][1]
print(f"L1={geo.L1:.4f} L2={geo.L2:.4f} nominal H={geo.h_nominal:.5f}")
worst = 0.0
for H in np.arange(0.09, 0.2101, 0.005):
    ph, pk = geo.ik(H)
    q = [signs[0]*ph, signs[1]*pk, signs[2]*ph, signs[3]*pk]       # [L hip, L knee, R hip, R knee]
    pl = axle_pos("left", q[0], q[1]); pr = axle_pos("right", q[2], q[3])
    want_y, want_z = hip_y + geo.py0, hip_z - H
    err = max(abs(pl[1]-want_y), abs(pl[2]-want_z), abs(pr[1]-want_y), abs(pr[2]-want_z), abs(pl[1]-pr[1]), abs(pl[2]-pr[2]))
    worst = max(worst, err)
print("worst axle position error over H=0.09..0.21 m: %.2e m" % worst)
assert worst < 1e-9
# q = 0 must be the nominal pose
q0 = geo.ik(geo.h_nominal); print("IK(nominal) =", [round(v, 12) for v in q0]); assert max(abs(v) for v in q0) < 1e-9
# the *wrong* signs must be detected by the same test
bad = [-1.0, +1.0, -1.0, +1.0]; ph, pk = geo.ik(0.14)
pl = axle_pos("left", bad[0]*ph, bad[1]*pk)
print("with a wrong left-knee sign the axle error would be %.3f m (test is discriminating)" % abs(pl[2] - (hip_z - 0.14)))
# joint ranges over the useful range
for lo, hi in ((0.10, 0.20), (0.11, 0.165)):
    qs = np.array([[signs[0]*geo.ik(H)[0], signs[1]*geo.ik(H)[1]] for H in np.linspace(lo, hi, 50)])
    print(f"H in [{lo},{hi}]: hip q in [{qs[:,0].min():+.3f},{qs[:,0].max():+.3f}] rad, knee q (left) in [{qs[:,1].min():+.3f},{qs[:,1].max():+.3f}] rad  (limits 0.70 / 0.85)")
print("ALL KINEMATIC CHECKS PASSED")
