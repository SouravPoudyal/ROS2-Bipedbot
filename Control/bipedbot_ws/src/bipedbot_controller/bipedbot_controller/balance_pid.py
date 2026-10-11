#!/usr/bin/env python3
"""
PID self-balancing node for bipedbot (wheels commanded in velocity).

Subscribes : sensor_msgs/Imu            (default /imu/out)
Publishes  : wheel command              (Float64MultiArray [left, right], or Twist)
             /balance/debug             (Float64MultiArray:
                [angle, effective_setpoint, error, wheel_speed, pid_accel, lean_offset])

Drive      : geometry_msgs/Twist on /cmd_vel (linear.x = forward speed in m/s,
             angular.z = yaw rate in rad/s). Forward speed is passed through the outer
             loop (the robot leans to reach it); yaw is a differential on the two wheels.
             Commands time out to zero. Only active when integrate_output is True.

Inner loop : PID on tilt angle -> wheel ACCELERATION, integrated into the speed command
             (a plain PD on speed cannot stabilise an inverted pendulum).
Outer loop : wheel speed shifts the lean target (speed_kp, speed_ki) so the robot does
             not run away at the speed limit and finds its own balance angle.

Gains are re-read every cycle:   ros2 param set /balance_pid kp 500.0
"""
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray


def quat_to_roll_pitch(x, y, z, w):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)
    return roll, pitch


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class BalancePID(Node):
    def __init__(self):
        super().__init__("balance_pid")

        # --- I/O ---
        self.declare_parameter("imu_topic", "/imu/out")
        self.declare_parameter("output_mode", "wheel_velocity")  # or "twist"
        self.declare_parameter("output_topic", "/simple_velocity_controller/commands")
        # Mirrored wheel axes in the URDF (left -x, right +x): left is negated.
        self.declare_parameter("left_sign", -1.0)
        self.declare_parameter("right_sign", 1.0)

        # --- Inner loop (angle) ---
        self.declare_parameter("tilt_axis", "roll")    # "roll" or "pitch"
        self.declare_parameter("angle_sign", 1.0)
        self.declare_parameter("setpoint", -0.005)     # rad, nominal upright angle
        self.declare_parameter("kp", 400.0)            # rad/s^2 per rad (integrate mode)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 20.0)             # rad/s^2 per rad/s
        self.declare_parameter("i_limit", 1.0)
        self.declare_parameter("output_limit", 20.0)   # rad/s, wheel speed clamp
        self.declare_parameter("fall_angle", 0.7)      # rad
        self.declare_parameter("enabled", True)
        self.declare_parameter("integrate_output", True)

        # --- Outer loop (speed -> lean target) ---
        self.declare_parameter("speed_kp", 0.003)      # rad of lean per rad/s of wheel speed
        self.declare_parameter("speed_ki", 0.0005)     # rad of lean per rad/s per second
        self.declare_parameter("lean_limit", 0.2)      # rad, max total lean offset

        # --- Driving (cmd_vel) --- NEW
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("max_speed", 0.3)        # m/s, clamp on linear.x
        self.declare_parameter("max_yaw_rate", 1.0)     # rad/s, clamp on angular.z
        self.declare_parameter("cmd_accel", 0.3)        # m/s^2, slew limit on the speed command
        self.declare_parameter("cmd_timeout", 0.5)      # s, commands older than this count as zero
        self.declare_parameter("forward_sign", -1.0)     # flip to -1.0 if +linear.x drives backwards
        self.declare_parameter("yaw_sign", 1.0)         # flip to -1.0 if +angular.z turns the wrong way
        self.declare_parameter("wheel_radius", 0.037)   # m
        self.declare_parameter("wheel_separation", 0.12)  # m

        self.integral = 0.0
        self.speed = 0.0
        self.trim = 0.0
        self.last_stamp = None
        self.v_cmd = 0.0        # NEW: latest commanded forward speed [m/s]
        self.wz_cmd = 0.0       # NEW: latest commanded yaw rate [rad/s]
        self.w_cmd_f = 0.0      # NEW: slew-limited wheel-speed target [rad/s]
        self.last_cmd_t = None  # NEW: wall-clock time of the last command

        imu_topic = self.get_parameter("imu_topic").value
        self.mode = self.get_parameter("output_mode").value
        out_topic = self.get_parameter("output_topic").value

        self.create_subscription(Imu, imu_topic, self.imu_cb, 10)
        self.create_subscription(Twist, self.get_parameter("cmd_vel_topic").value,
                                 self.cmd_cb, 10)
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

    def cmd_cb(self, msg: Twist):
        vmax = self.p("max_speed")
        wmax = self.p("max_yaw_rate")
        self.v_cmd = clamp(msg.linear.x, -vmax, vmax)
        self.wz_cmd = clamp(msg.angular.z, -wmax, wmax)
        self.last_cmd_t = time.monotonic()

    def publish_cmd(self, u, dlt=0.0):
        if self.mode == "twist":
            msg = Twist()
            msg.linear.x = u
        else:
            lim = self.p("output_limit")
            msg = Float64MultiArray()
            msg.data = [
                clamp(self.p("left_sign") * (u + dlt), -lim, lim),
                clamp(self.p("right_sign") * (u - dlt), -lim, lim),
            ]
        self.cmd_pub.publish(msg)

    def reset(self):
        self.integral = 0.0
        self.speed = 0.0
        self.trim = 0.0
        self.w_cmd_f = 0.0

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
            angle = roll
            rate = msg.angular_velocity.x
        else:
            angle = pitch
            rate = msg.angular_velocity.y

        sign = self.p("angle_sign")
        angle *= sign
        rate *= sign

        base_error = angle - self.p("setpoint")

        # Fallen over or disabled: stop and reset
        if not self.p("enabled") or abs(base_error) > self.p("fall_angle"):
            self.reset()
            self.publish_cmd(0.0)
            self.publish_debug(angle, self.p("setpoint"), base_error, 0.0, 0.0, 0.0)
            return

        integrate = self.p("integrate_output")

        # NEW: drive command. Commands that stop arriving count as zero.
        r = self.p("wheel_radius")
        if self.last_cmd_t is None or time.monotonic() - self.last_cmd_t > self.p("cmd_timeout"):
            v_cmd, wz_cmd = 0.0, 0.0
        else:
            v_cmd, wz_cmd = self.v_cmd, self.wz_cmd
        w_target = self.p("forward_sign") * v_cmd / r            # rad/s
        if self.p("cmd_accel") > 0.0:
            step = self.p("cmd_accel") / r * dt                  # slew limit per cycle
            self.w_cmd_f += clamp(w_target - self.w_cmd_f, -step, step)
        else:
            self.w_cmd_f = w_target
        if not integrate:
            self.w_cmd_f = 0.0
            wz_cmd = 0.0

        # Outer loop: wheel speed (relative to the commanded speed) shifts the lean target.
        # Speed below the command -> lean further forward, so the robot speeds up.
        lean = 0.0
        if integrate:
            tl = self.p("lean_limit")
            speed_err = self.speed - self.w_cmd_f
            self.trim = clamp(self.trim - self.p("speed_ki") * speed_err * dt, -tl, tl)
            lean = clamp(self.trim - self.p("speed_kp") * speed_err, -tl, tl)

        sp_eff = self.p("setpoint") + lean
        error = angle - sp_eff

        i_lim = self.p("i_limit")
        self.integral = clamp(self.integral + error * dt, -i_lim, i_lim)

        pid = self.p("kp") * error + self.p("ki") * self.integral + self.p("kd") * rate

        limit = self.p("output_limit")
        if integrate:
            self.speed = clamp(self.speed + pid * dt, -limit, limit)
            u = self.speed
        else:
            u = clamp(pid, -limit, limit)

        # NEW: yaw as a differential on the two physical wheel speeds
        # (omega_z = -2 r delta / d, so delta = -d * omega_z / (2 r)).
        dlt = self.p("yaw_sign") * (-self.p("wheel_separation") * wz_cmd / (2.0 * r))

        self.publish_cmd(u, dlt)
        self.publish_debug(angle, sp_eff, error, u, pid, lean, self.w_cmd_f)

    def publish_debug(self, angle, sp_eff, error, u, pid, lean, w_cmd=0.0):
        d = Float64MultiArray()
        # index 6 (w_cmd) is new; indices 0-5 are unchanged
        d.data = [angle, sp_eff, error, u, pid, lean, w_cmd]
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