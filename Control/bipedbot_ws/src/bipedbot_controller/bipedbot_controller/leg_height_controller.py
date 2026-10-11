#!/usr/bin/env python3
"""
Leg height controller for the bipedbot.

Input    : /leg_height_cmd   std_msgs/Float64   desired hip->axle height H in metres
Output   : /leg_position_controller/commands  Float64MultiArray
                 [left_hip, left_knee, right_hip, right_knee]  joint angles (rad)
           /balance/setpoint_offset           Float64
                 predicted change of the balance angle, theta0(H) - theta0(H_nominal), rad

Both legs are driven symmetrically in the balancing plane (y-z). The command is clamped to
[h_min, h_max] and rate limited, because the leg motion shifts the centre of mass and the
balance angle (analysis/leg_model.py). The balance node adds the offset to its setpoint, so
the slow outer loop does not have to absorb the shift by rolling the robot away.

Geometry comes from the URDF: thigh = joint offset of *_bottom_leg_joint, shank = joint
offset of *_motor_joint (both in the y-z plane).
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray


class LegGeometry:
    """Two-link planar leg: hip -> knee (thigh) -> wheel axle (shank)."""

    def __init__(self, thigh, shank):
        self.t0 = tuple(thigh)
        self.s0 = tuple(shank)
        self.L1 = math.hypot(*self.t0)
        self.L2 = math.hypot(*self.s0)
        ang = lambda v: math.atan2(v[1], v[0])
        self.a_t0 = ang(self.t0)
        self.psi0 = ang(self.s0) - ang(self.t0)       # angle thigh->shank at q = 0
        self.py0 = self.t0[0] + self.s0[0]            # nominal axle y offset from the hip
        self.h_nominal = -(self.t0[1] + self.s0[1])   # nominal hip->axle height

    @staticmethod
    def rot(phi, v):
        c, s = math.cos(phi), math.sin(phi)
        return (c * v[0] - s * v[1], s * v[0] + c * v[1])

    def fk(self, ph, pk):
        """Axle position relative to the hip for planar angles (rotation about +x)."""
        knee = self.rot(ph, self.t0)
        s = self.rot(ph + pk, self.s0)
        return (knee[0] + s[0], knee[1] + s[1])

    def ik(self, h):
        """Planar angles (phi_hip, phi_knee) that put the axle h below the hip, keeping the
        axle at its nominal fore-aft offset (knee behind the leg line)."""
        py = self.py0
        D = math.hypot(py, h)
        c = (D * D - self.L1 ** 2 - self.L2 ** 2) / (2.0 * self.L1 * self.L2)
        c = max(-1.0, min(1.0, c))
        psi = math.acos(c)
        pk = psi - self.psi0
        ph = (math.atan2(-h, py) - self.a_t0
              - math.atan2(self.L2 * math.sin(psi), self.L1 + self.L2 * c))
        return ph, pk


class LegHeightController(Node):
    def __init__(self):
        super().__init__("leg_height_controller")

        self.declare_parameter("height_topic", "/leg_height_cmd")
        self.declare_parameter("output_topic", "/leg_position_controller/commands")
        self.declare_parameter("offset_topic", "/balance/setpoint_offset")

        # Geometry from the URDF (y, z): *_bottom_leg_joint origin, *_motor_joint origin.
        self.declare_parameter("thigh_vec", [-0.079867, -0.055924])
        self.declare_parameter("shank_vec", [0.074565, -0.106489])

        # Joint angle = sign * planar angle. Derived from the URDF axes:
        #   hips: both -x   -> -1;  knees: right +x -> +1, left -x -> -1.
        # Order: [left_hip, left_knee, right_hip, right_knee]
        self.declare_parameter("joint_signs", [-1.0, -1.0, -1.0, 1.0])

        # Operating range and speed. Extending above nominal reduces the delay margin of the
        # current outer-loop gains (analysis/leg_model.py), so h_max stays close to nominal.
        self.declare_parameter("h_min", 0.11)
        self.declare_parameter("h_max", 0.165)
        self.declare_parameter("max_rate", 0.01)        # m/s
        self.declare_parameter("rate_hz", 50.0)

        # Balance-angle table theta0(H) from analysis/leg_model.py (URDF inertials).
        self.declare_parameter("ff_enable", True)
        self.declare_parameter("table_h", [0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
                                           0.16, 0.17, 0.18, 0.19, 0.20, 0.21])
        self.declare_parameter("table_theta0", [-0.04096, -0.03696, -0.03216, -0.02698,
                                                -0.02162, -0.01617, -0.01066, -0.00505,
                                                0.00071, 0.00672, 0.01316, 0.02029,
                                                0.02872])

        self.geo = LegGeometry(self.get_parameter("thigh_vec").value,
                               self.get_parameter("shank_vec").value)
        self.signs = list(self.get_parameter("joint_signs").value)
        self.h_min = float(self.get_parameter("h_min").value)
        self.h_max = float(self.get_parameter("h_max").value)
        self.max_rate = float(self.get_parameter("max_rate").value)
        self.ff_enable = bool(self.get_parameter("ff_enable").value)
        self.tab_h = np.array(self.get_parameter("table_h").value, float)
        self.tab_th = np.array(self.get_parameter("table_theta0").value, float)
        self.theta0_nom = float(np.interp(self.geo.h_nominal, self.tab_h, self.tab_th))

        # The robot spawns at q = 0, which is the nominal pose.
        self.h = self.geo.h_nominal
        self.h_target = self.h

        self.create_subscription(Float64, self.get_parameter("height_topic").value,
                                 self.height_cb, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("output_topic").value, 10)
        self.off_pub = self.create_publisher(
            Float64, self.get_parameter("offset_topic").value, 10)

        period = 1.0 / float(self.get_parameter("rate_hz").value)
        self.dt = period
        self.create_timer(period, self.tick)
        self.get_logger().info(
            f"Leg height controller: nominal H={self.geo.h_nominal:.4f} m, "
            f"range [{self.h_min:.3f}, {self.h_max:.3f}] m, max rate {self.max_rate} m/s, "
            f"L1={self.geo.L1:.4f} L2={self.geo.L2:.4f}")

    def height_cb(self, msg: Float64):
        h = min(max(float(msg.data), self.h_min), self.h_max)
        if abs(h - msg.data) > 1e-9:
            self.get_logger().warn(f"height {msg.data:.3f} m clamped to {h:.3f} m")
        self.h_target = h

    def joint_angles(self, h):
        ph, pk = self.geo.ik(h)
        s = self.signs
        return [s[0] * ph, s[1] * pk, s[2] * ph, s[3] * pk]

    def tick(self):
        step = self.max_rate * self.dt
        self.h += max(-step, min(step, self.h_target - self.h))

        cmd = Float64MultiArray()
        cmd.data = self.joint_angles(self.h)
        self.cmd_pub.publish(cmd)

        off = Float64()
        off.data = (float(np.interp(self.h, self.tab_h, self.tab_th)) - self.theta0_nom
                    if self.ff_enable else 0.0)
        self.off_pub.publish(off)


def main():
    rclpy.init()
    node = LegHeightController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
