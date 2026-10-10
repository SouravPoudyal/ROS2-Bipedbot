#!/usr/bin/env python3
"""
Balancing node, stage 2: PID output is wheel ACCELERATION, integrated into the
wheel speed command (the wheels take a velocity command).

Debug (/balance/debug): [angle, setpoint, error, wheel_speed, pid_accel]
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray


def quat_to_roll_pitch(x, y, z, w):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(s)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class BalancePID(Node):
    def __init__(self):
        super().__init__("balance_pid")

        self.declare_parameter("imu_topic", "/imu/out")
        self.declare_parameter("output_mode", "wheel_velocity")  # or "twist"
        self.declare_parameter("output_topic", "/simple_velocity_controller/commands")
        # Mirrored wheel axes in the URDF (left -x, right +x): left is negated.
        self.declare_parameter("left_sign", -1.0)
        self.declare_parameter("right_sign", 1.0)

        self.declare_parameter("tilt_axis", "roll")
        self.declare_parameter("angle_sign", 1.0)
        self.declare_parameter("setpoint", -0.03)
        self.declare_parameter("kp", 400.0)            # rad/s^2 per rad
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 20.0)             # rad/s^2 per rad/s
        self.declare_parameter("i_limit", 1.0)
        self.declare_parameter("output_limit", 20.0)   # rad/s, wheel speed clamp
        self.declare_parameter("fall_angle", 0.7)
        self.declare_parameter("enabled", True)
        self.declare_parameter("integrate_output", True)

        self.integral = 0.0
        self.speed = 0.0
        self.last_stamp = None

        imu_topic = self.get_parameter("imu_topic").value
        self.mode = self.get_parameter("output_mode").value
        out_topic = self.get_parameter("output_topic").value

        self.create_subscription(Imu, imu_topic, self.imu_cb, 10)
        if self.mode == "twist":
            self.cmd_pub = self.create_publisher(Twist, out_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Float64MultiArray, out_topic, 10)
        self.dbg_pub = self.create_publisher(Float64MultiArray, "/balance/debug", 10)

        self.get_logger().info(
            f"Balancing on {imu_topic}, mode={self.mode}, publishing to {out_topic}"
        )

    def p(self, name):
        return self.get_parameter(name).value

    def publish_cmd(self, u):
        if self.mode == "twist":
            msg = Twist()
            msg.linear.x = u
        else:
            msg = Float64MultiArray()
            msg.data = [self.p("left_sign") * u, self.p("right_sign") * u]
        self.cmd_pub.publish(msg)

    def imu_cb(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp is None:
            self.last_stamp = stamp
            return
        dt = stamp - self.last_stamp
        self.last_stamp = stamp
        if dt <= 0.0 or dt > 0.1:
            return

        q = msg.orientation
        roll, pitch = quat_to_roll_pitch(q.x, q.y, q.z, q.w)
        if self.p("tilt_axis") == "roll":
            angle, rate = roll, msg.angular_velocity.x
        else:
            angle, rate = pitch, msg.angular_velocity.y

        sign = self.p("angle_sign")
        angle *= sign
        rate *= sign
        error = angle - self.p("setpoint")

        if not self.p("enabled") or abs(error) > self.p("fall_angle"):
            self.integral = 0.0
            self.speed = 0.0
            self.publish_cmd(0.0)
            self.publish_debug(angle, error, 0.0, 0.0)
            return

        i_lim = self.p("i_limit")
        self.integral = clamp(self.integral + error * dt, -i_lim, i_lim)
        pid = self.p("kp") * error + self.p("ki") * self.integral + self.p("kd") * rate

        limit = self.p("output_limit")
        if self.p("integrate_output"):
            self.speed = clamp(self.speed + pid * dt, -limit, limit)  # clamp = anti-windup
            u = self.speed
        else:
            u = clamp(pid, -limit, limit)

        self.publish_cmd(u)
        self.publish_debug(angle, error, u, pid)

    def publish_debug(self, angle, error, u, pid):
        d = Float64MultiArray()
        d.data = [angle, self.p("setpoint"), error, u, pid]
        self.dbg_pub.publish(d)


def main():
    rclpy.init()
    node = BalancePID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
