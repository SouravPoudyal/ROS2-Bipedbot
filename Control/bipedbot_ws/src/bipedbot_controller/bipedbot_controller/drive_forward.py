#!/usr/bin/env python3
"""
Send a smooth forward-speed command to the balancing robot on /cmd_vel.

The balance node (balance_pid.py) turns /cmd_vel into a lean; this script only
publishes the command, with a ramp up, a hold and a ramp down, then stops.
Ctrl+C also sends a zero command.

Examples (ROS environment and workspace sourced):
  python3 drive_forward.py
  python3 drive_forward.py --ros-args -p speed:=0.1 -p duration:=4.0
  python3 drive_forward.py --ros-args -p speed:=0.1 -p shuttle:=true
  python3 drive_forward.py --ros-args -p speed:=-0.05 -p duration:=3.0   # backwards

Parameters
  topic     /cmd_vel   Twist topic read by balance_pid
  speed     0.1        m/s, signed (negative = backwards)
  duration  3.0        s, length of one move including both ramps
  ramp      1.0        s, ramp-up and ramp-down time (shortened if duration is small)
  yaw       0.0        rad/s, optional constant turn rate during the move
  shuttle   false      forward, pause, then the same move backwards
  settle    1.0        s, zero command before/between/after moves
  rate      20.0       Hz, publish rate (keep above 1 / cmd_timeout of balance_pid)
  max_speed 0.3        m/s, safety clamp (matches balance_pid default)
"""
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def speed_at(t, speed, duration, ramp):
    """Trapezoid profile: 0 -> speed -> 0 over `duration` seconds."""
    if t < 0.0 or t > duration:
        return 0.0
    ramp = max(1e-6, min(ramp, duration / 2.0))
    return speed * min(1.0, t / ramp, (duration - t) / ramp)


class DriveForward(Node):
    def __init__(self):
        super().__init__("drive_forward")
        self.declare_parameter("topic", "/cmd_vel")
        self.declare_parameter("speed", 0.1)
        self.declare_parameter("duration", 3.0)
        self.declare_parameter("ramp", 1.0)
        self.declare_parameter("yaw", 0.0)
        self.declare_parameter("shuttle", False)
        self.declare_parameter("settle", 1.0)
        self.declare_parameter("rate", 20.0)
        self.declare_parameter("max_speed", 0.3)

        p = lambda n: self.get_parameter(n).value
        self.max_speed = abs(p("max_speed"))
        self.speed = max(-self.max_speed, min(self.max_speed, float(p("speed"))))
        if self.speed != float(p("speed")):
            self.get_logger().warn(f"speed clamped to {self.speed} m/s (max_speed)")
        self.duration = max(0.1, float(p("duration")))
        self.ramp = float(p("ramp"))
        self.yaw = float(p("yaw"))
        self.settle = max(0.0, float(p("settle")))
        self.shuttle = bool(p("shuttle"))

        # (start time, speed) of each move, measured from the first command
        self.moves = [(self.settle, self.speed)]
        end = self.settle + self.duration
        if self.shuttle:
            self.moves.append((end + self.settle, -self.speed))
            end += self.settle + self.duration
        self.t_end = end + self.settle

        self.pub = self.create_publisher(Twist, p("topic"), 10)
        self.t0 = time.monotonic()
        self.done = False
        self.create_timer(1.0 / float(p("rate")), self.tick)
        self.get_logger().info(
            f"Driving {self.speed:+.3f} m/s for {self.duration:.1f} s"
            f"{' and back' if self.shuttle else ''} on {p('topic')}")

    def command(self, v, wz):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(wz)
        self.pub.publish(msg)

    def tick(self):
        t = time.monotonic() - self.t0
        if t >= self.t_end:
            self.command(0.0, 0.0)
            self.done = True
            return
        v = 0.0
        for start, spd in self.moves:
            v += speed_at(t - start, spd, self.duration, self.ramp)
        moving = abs(v) > 1e-9
        self.command(v, self.yaw if moving else 0.0)

    def stop(self):
        for _ in range(3):
            self.command(0.0, 0.0)


def main():
    rclpy.init()
    node = DriveForward()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
            node.get_logger().info("Stopped (zero command sent).")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
