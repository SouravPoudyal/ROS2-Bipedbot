#!/usr/bin/env python3
"""
Scripted test scenarios for the bipedbot: leg height (/leg_height_cmd) combined with
driving (/cmd_vel). Logs /balance/debug and the leg joint angles to a CSV and prints
per-phase metrics at the end.

Scenarios (-p scenario:=NAME)
  sweep     stand, then hold a series of leg heights. The 'lean_end' metric of each phase is
            the residual the outer loop still has to supply, i.e. the error of the theta0(H)
            table (feedforward) at that height.
  drive     at three heights: forward, pause, backward.
  yaw       at two heights: +yaw, pause, -yaw. Use small yaw first; the yaw sign is untested.
  combined  constant slow drive while the legs go down and up again.

Examples (ROS environment and workspace sourced, simulation, balance_pid and
leg_height_controller running):
  ros2 run bipedbot_controller scenario_runner.py --ros-args -p scenario:=sweep
  ros2 run bipedbot_controller scenario_runner.py --ros-args -p scenario:=drive -p speed:=0.05
  ros2 run bipedbot_controller scenario_runner.py --ros-args -p scenario:=yaw -p yaw:=0.2

Parameters
  scenario    sweep   see above
  hold        8.0     s, hold time after a height change (before the next phase)
  speed       0.05    m/s, drive speed (balance_pid clamps it and slews it by cmd_accel)
  yaw         0.2     rad/s, yaw rate for the yaw scenario
  drive_time  4.0     s, length of each forward / backward phase
  h_start     0.1624  m, height the legs are at when the scenario starts (used to time the
                      travel; the leg node rate limits at max_rate)
  max_rate    0.01    m/s, must match leg_height_controller
  abort_error 0.35    rad, stop the scenario if |tilt error| exceeds this
  out         ''      CSV path (default /tmp/scenario_<name>_<time>.csv)

The scenario ends with a zero command. It never changes balance_pid or leg parameters.
"""
import csv
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

H_NOM = 0.162413
LEG_JOINTS = ["left_top_leg_joint", "left_bottom_leg_joint",
              "right_top_leg_joint", "right_bottom_leg_joint"]
DBG_COLS = ["angle", "sp_eff", "error", "u", "pid", "lean", "w_cmd"]


def build_steps(name, h0, rate, hold, speed, yaw, drive_time):
    """Return a list of phases: dict(label, dur, h, v, wz). h is None to keep the height."""
    steps = []
    cur = [h0]

    def go(label, h=None, v=0.0, wz=0.0, t=None):
        t = hold if t is None else t
        dur = t
        if h is not None:
            dur += abs(h - cur[0]) / rate
            cur[0] = h
        steps.append(dict(label=label, dur=dur, h=h, v=v, wz=wz))

    go("start", t=5.0)
    if name == "sweep":
        for h in [0.14, 0.12, 0.11, 0.14, H_NOM, 0.165, H_NOM]:
            go(f"H={h:.3f}", h=h)
    elif name == "drive":
        for h in [H_NOM, 0.14, 0.12]:
            go(f"H={h:.3f}", h=h)
            go(f"H={h:.3f} fwd", v=speed, t=drive_time)
            go(f"H={h:.3f} pause", t=4.0)
            go(f"H={h:.3f} back", v=-speed, t=drive_time)
            go(f"H={h:.3f} pause", t=4.0)
        go(f"H={H_NOM:.3f}", h=H_NOM)
    elif name == "yaw":
        for h in [H_NOM, 0.12]:
            go(f"H={h:.3f}", h=h)
            go(f"H={h:.3f} yaw+", wz=yaw, t=drive_time)
            go(f"H={h:.3f} pause", t=4.0)
            go(f"H={h:.3f} yaw-", wz=-yaw, t=drive_time)
            go(f"H={h:.3f} pause", t=4.0)
        go(f"H={H_NOM:.3f}", h=H_NOM)
    elif name == "combined":
        go("drive", v=speed, t=6.0)
        go("drive, legs down", h=0.12, v=speed)
        go("drive, legs up", h=H_NOM, v=speed)
        go("stop", t=6.0)
    else:
        raise ValueError(f"unknown scenario '{name}' (sweep, drive, yaw, combined)")
    return steps


def phase_metrics(rows, n_phases, labels):
    """Per-phase metrics from rows: dict(t, phase, <DBG_COLS>)."""
    out = []
    for i in range(n_phases):
        r = [x for x in rows if x["phase"] == i]
        if not r:
            out.append((labels[i], None))
            continue
        err = [x["error"] for x in r]
        t_end = r[-1]["t"]
        tail = [x["lean"] for x in r if x["t"] >= t_end - 2.0] or [r[-1]["lean"]]
        out.append((labels[i], dict(
            rms_err=math.sqrt(sum(e * e for e in err) / len(err)),
            max_err=max(abs(e) for e in err),
            max_lean=max(abs(x["lean"]) for x in r),
            max_u=max(abs(x["u"]) for x in r),
            lean_end=sum(tail) / len(tail))))
    return out


class ScenarioRunner(Node):
    def __init__(self):
        super().__init__("scenario_runner")
        gp = lambda n: self.get_parameter(n).value
        self.declare_parameter("scenario", "sweep")
        self.declare_parameter("hold", 8.0)
        self.declare_parameter("speed", 0.05)
        self.declare_parameter("yaw", 0.2)
        self.declare_parameter("drive_time", 4.0)
        self.declare_parameter("h_start", H_NOM)
        self.declare_parameter("max_rate", 0.01)
        self.declare_parameter("abort_error", 0.35)
        self.declare_parameter("out", "")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("height_topic", "/leg_height_cmd")
        self.declare_parameter("rate", 20.0)

        self.name = str(gp("scenario"))
        self.steps = build_steps(self.name, float(gp("h_start")), float(gp("max_rate")),
                                 float(gp("hold")), float(gp("speed")), float(gp("yaw")),
                                 float(gp("drive_time")))
        self.abort_error = float(gp("abort_error"))
        out = str(gp("out"))
        self.out = out or time.strftime(f"/tmp/scenario_{self.name}_%Y%m%d_%H%M%S.csv")

        self.cmd_pub = self.create_publisher(Twist, gp("cmd_topic"), 10)
        self.h_pub = self.create_publisher(Float64, gp("height_topic"), 10)
        self.create_subscription(Float64MultiArray, "/balance/debug", self.dbg_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)

        self.q = {}
        self.rows = []
        self.t0 = None            # set when the first /balance/debug arrives
        self.idx = -1
        self.step_t = 0.0
        self.last_h_pub = 0.0
        self.done = False
        self.aborted = False
        self.create_timer(1.0 / float(gp("rate")), self.tick)

        total = sum(s["dur"] for s in self.steps)
        self.get_logger().info(
            f"Scenario '{self.name}': {len(self.steps)} phases, about {total:.0f} s. "
            f"Waiting for /balance/debug ...")

    def js_cb(self, m: JointState):
        for n, p in zip(m.name, m.position):
            if n in LEG_JOINTS:
                self.q[n] = p

    def dbg_cb(self, m: Float64MultiArray):
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        if self.done or self.idx < 0 or len(m.data) < 6:
            return
        d = list(m.data) + [0.0] * (7 - len(m.data))
        row = dict(t=now - self.t0, phase=self.idx)
        row.update({k: d[i] for i, k in enumerate(DBG_COLS)})
        self.rows.append(row)
        if abs(d[2]) > self.abort_error and not self.aborted:
            self.aborted = True
            self.get_logger().error(
                f"Tilt error {d[2]:+.2f} rad in phase '{self.steps[self.idx]['label']}': "
                f"aborting scenario (robot probably fell).")

    def command(self, v, wz):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def tick(self):
        if self.done:
            return
        if self.t0 is None:
            return
        now = time.monotonic() - self.t0
        if self.aborted:
            self.command(0.0, 0.0)
            self.finish()
            return

        if self.idx < 0 or now - self.step_t >= self.steps[self.idx]["dur"]:
            nxt = self.idx + 1
            if nxt >= len(self.steps):
                self.command(0.0, 0.0)
                self.finish()
                return
            self.idx = nxt
            self.step_t = now
            s = self.steps[self.idx]
            self.get_logger().info(
                f"[{now:6.1f}s] phase {self.idx + 1}/{len(self.steps)}: {s['label']} "
                f"(v={s['v']:+.3f}, wz={s['wz']:+.2f}, {s['dur']:.0f} s)")
            if s["h"] is not None:
                self.h_pub.publish(Float64(data=float(s["h"])))
                self.last_h_pub = now

        s = self.steps[self.idx]
        self.command(s["v"], s["wz"])

    def finish(self):
        self.done = True
        labels = [s["label"] for s in self.steps]
        try:
            with open(self.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "phase", "label"] + DBG_COLS + LEG_JOINTS)
                for r in self.rows:
                    w.writerow([f"{r['t']:.3f}", r["phase"], labels[r["phase"]]]
                               + [f"{r[k]:.5f}" for k in DBG_COLS]
                               + [f"{self.q.get(j, float('nan')):.4f}" for j in LEG_JOINTS])
            self.get_logger().info(f"Log written to {self.out}")
        except OSError as e:
            self.get_logger().error(f"Could not write {self.out}: {e}")

        print(f"\nScenario '{self.name}'" + ("  *** ABORTED ***" if self.aborted else ""))
        print(f"{'phase':<26}{'rms err':>9}{'max err':>9}{'max lean':>10}"
              f"{'max u':>8}{'lean_end':>10}")
        for lab, m in phase_metrics(self.rows, len(self.steps), labels):
            if m is None:
                print(f"{lab:<26}{'(not reached)':>46}")
            else:
                print(f"{lab:<26}{m['rms_err']:9.4f}{m['max_err']:9.4f}"
                      f"{m['max_lean']:10.4f}{m['max_u']:8.2f}{m['lean_end']:10.4f}")
        print("rad for tilt/lean, rad/s for u (wheel speed); lean_end = mean lean in the "
              "last 2 s of the phase\n")


def main():
    rclpy.init()
    node = ScenarioRunner()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            for _ in range(3):
                node.command(0.0, 0.0)
            if not node.done:
                node.get_logger().info("Interrupted: zero command sent.")
                node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()