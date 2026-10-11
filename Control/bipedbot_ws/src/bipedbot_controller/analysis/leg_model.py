#!/usr/bin/env python3
"""
leg_model.py - planar model of the bipedbot legs and of the balance dynamics
as a function of leg height.

Reads masses, centres of mass, inertias and joint offsets from the URDF, builds the
two-link leg (hip -> knee -> wheel axle) in the y-z plane (the balancing plane), and
for each hip height H computes:

  * IK joint angles (hip, knee)
  * effective pendulum length l = I / (m d) and balance angle theta0   (report Eq. 6)
  * stability of the cascade (report Eq. 20, with the Appendix D actuator-lag model)
  * rough static joint torques

Usage:  python3 leg_model.py path/to/bipedbot.urdf.xacro [--gains kp kd kvp kvi] [--simulate]
Needs numpy only.
"""
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np

G = 9.81
R_WHEEL = 0.037
GAINS = dict(kp=400.0, kd=20.0, kvp=0.003, kvi=5e-4)       # final gains (stage 3b)


# --------------------------------------------------------------------------- URDF
def load_urdf(path):
    root = ET.parse(path).getroot()
    links, joints = {}, {}
    for l in root.findall("link"):
        inert = l.find("inertial")
        if inert is None:
            continue
        xyz = [float(v) for v in inert.find("origin").get("xyz").split()]
        links[l.get("name")] = dict(
            m=float(inert.find("mass").get("value")),
            c=(xyz[1], xyz[2]),                       # CoM (y, z) in the link frame
            ixx=float(inert.find("inertia").get("ixx")),
        )
    for j in root.findall("joint"):
        xyz = [float(v) for v in j.find("origin").get("xyz").split()]
        joints[j.get("name")] = dict(o=(xyz[1], xyz[2]))
    return links, joints


def rot(phi, v):
    """Rotate the planar vector v=(y,z) by phi about +x (right-handed)."""
    c, s = math.cos(phi), math.sin(phi)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


class Robot:
    def __init__(self, path):
        self.links, self.joints = load_urdf(path)
        j = self.joints
        self.hip = j["right_top_leg_joint"]["o"]       # in base_link
        self.t0 = j["right_bottom_leg_joint"]["o"]      # knee, in thigh frame
        self.s0 = j["right_motor_joint"]["o"]           # axle, in shank frame
        for side in ("left",):                          # planar model assumes symmetry
            assert j[f"{side}_top_leg_joint"]["o"] == self.hip
            assert j[f"{side}_bottom_leg_joint"]["o"] == self.t0
            assert j[f"{side}_motor_joint"]["o"] == self.s0
        self.L1 = math.hypot(*self.t0)
        self.L2 = math.hypot(*self.s0)
        ang = lambda v: math.atan2(v[1], v[0])
        self.a_t0, self.psi0 = ang(self.t0), ang(self.s0) - ang(self.t0)
        self.p0 = add(self.t0, self.s0)                 # hip -> axle at q = 0
        self.y_imu = j["imu_joint"]["o"]

    # ---- kinematics -------------------------------------------------------
    def fk(self, ph, pk):
        knee = add(self.hip, rot(ph, self.t0))
        axle = add(knee, rot(ph + pk, self.s0))
        return knee, axle

    def ik(self, H, py=None):
        """Joint angles (phi_h, phi_k) for the axle H below the hip, y offset py."""
        py = self.p0[0] if py is None else py
        p = (py, -H)
        D = math.hypot(*p)
        c = (D * D - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        if not -1.0 <= c <= 1.0:
            raise ValueError(f"H={H:.3f} m is outside the leg's reach")
        psi = math.acos(c)                              # knee-behind branch
        pk = psi - self.psi0
        ph = (math.atan2(p[1], p[0]) - self.a_t0
              - math.atan2(self.L2 * math.sin(psi), self.L1 + self.L2 * c))
        return ph, pk

    # ---- mass properties --------------------------------------------------
    def body(self, ph, pk):
        """Mass, CoM offset from the axle (dy, dz), inertia about the axle.
        Wheels and tyres are excluded (report Section 4.2)."""
        L, j = self.links, self.joints
        hip = self.hip
        knee, axle = self.fk(ph, pk)
        pts = []   # (mass, com_y, com_z, ixx)

        def put(name, origin, angle):
            l = L[name]
            c = add(origin, rot(angle, l["c"]))
            pts.append((l["m"], c[0], c[1], l["ixx"]))

        put("base_link", (0.0, 0.0), 0.0)
        put("imu_link_1", j["imu_joint"]["o"], 0.0)
        for side in ("right", "left"):
            put(f"{side}_top_leg_1", hip, ph)
            put(f"{side}_bottom_leg_1", knee, ph + pk)
            put(f"{side}_motor_1", axle, ph + pk)
        m = sum(p[0] for p in pts)
        cy = sum(p[0] * p[1] for p in pts) / m
        cz = sum(p[0] * p[2] for p in pts) / m
        I = sum(p[3] + p[0] * ((p[1] - axle[0]) ** 2 + (p[2] - axle[1]) ** 2) for p in pts)
        dy, dz = cy - axle[0], cz - axle[1]
        d = math.hypot(dy, dz)
        return dict(m=m, dy=dy, dz=dz, d=d, I=I, l=I / (m * d),
                    theta0=math.atan2(dy, dz), axle=axle, knee=knee)

    def wheel_mass(self):
        L = self.links
        return sum(L[n]["m"] for n in ("left_wheel_1", "right_wheel_1",
                                       "left_tyre_1", "right_tyre_1"))


# --------------------------------------------------------------------------- dynamics
def cascade_A(l, kp, kd, kvp, kvi, r=R_WHEEL, Ta=None):
    """Linear cascade model. Ta=None: report Eq. (20) (4 states [th, thd, w, tau]).
    Ta given: report Appendix D (5 states [th, thd, wc, ww, tau])."""
    if Ta is None:
        return np.array([
            [0, 1, 0, 0],
            [(G - r * kp) / l, -r * kd / l, -r * kp * kvp / l, r * kp / l],
            [kp, kd, kp * kvp, -kp],
            [0, 0, -kvi, 0]], float)
    return np.array([
        [0, 1, 0, 0, 0],
        [G / l, 0, -r / (l * Ta), r / (l * Ta), 0],
        [kp, kd, kp * kvp, 0, -kp],
        [0, 0, 1 / Ta, -1 / Ta, 0],
        [0, 0, -kvi, 0, 0]], float)


def max_re(l, gains, Ta=None):
    return float(np.max(np.linalg.eigvals(cascade_A(l, Ta=Ta, **gains)).real))


def critical_lag(l, gains, lo=1e-4, hi=0.2):
    """Smallest actuator lag Ta (s) that makes the cascade unstable; 0 if unstable at
    Ta -> 0, inf if stable up to `hi`."""
    if max_re(l, gains) > 0:
        return 0.0
    if max_re(l, gains, Ta=hi) <= 0:
        return float("inf")
    a, b = lo, hi
    for _ in range(60):
        mid = 0.5 * (a + b)
        if max_re(l, gains, Ta=mid) > 0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)



# --------------------------------------------------------------------------- simulation
def simulate(rb, H0, H1, rate, gains, ff=True, t_end=40.0, dt=2e-3):
    """Linear cascade (report Eq. 4, 5, 10, 15) with a leg-height ramp H0 -> H1 at `rate` m/s.
    The balance angle theta0 and length l follow the leg pose. ff=True feeds the predicted
    change of theta0 into the setpoint (what leg_height_controller publishes).
    Returns peak tilt error, peak wheel speed, final base displacement, final lean trim."""
    Hs = np.arange(0.09, 0.2101, 0.005)
    tab = [(H, rb.body(*rb.ik(H))) for H in Hs]
    th0_tab = np.array([b["theta0"] for _, b in tab])
    l_tab = np.array([b["l"] for _, b in tab])
    th0_nom = float(np.interp(H0, Hs, th0_tab))
    kp, kd, kvp, kvi = gains["kp"], gains["kd"], gains["kvp"], gains["kvi"]
    r = R_WHEEL

    def H_of(t):
        step = rate * t
        return H0 + math.copysign(min(abs(H1 - H0), step), H1 - H0)

    def f(t, x):
        th, thd, w, tau, s = x
        H = H_of(t)
        th0 = float(np.interp(H, Hs, th0_tab)); l = float(np.interp(H, Hs, l_tab))
        ffv = (th0 - th0_nom) if ff else 0.0
        e = th - th0_nom - ffv + kvp * w - tau
        wd = kp * e + kd * thd
        thdd = (G / l) * (th - th0) - (r / l) * wd
        return np.array([thd, thdd, wd, -kvi * w, r * w])

    x = np.array([th0_nom, 0.0, 0.0, 0.0, 0.0]); t = 0.0
    peak_err = peak_w = 0.0
    for _ in range(int(t_end / dt)):
        k1 = f(t, x); k2 = f(t + dt/2, x + dt/2*k1); k3 = f(t + dt/2, x + dt/2*k2); k4 = f(t + dt, x + dt*k3)
        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4); t += dt
        H = H_of(t)
        peak_err = max(peak_err, abs(x[0] - float(np.interp(H, Hs, th0_tab))))
        peak_w = max(peak_w, abs(x[2]))
    return peak_err, peak_w, x[4], x[3]


def run_simulations(rb, gains):
    H0 = -rb.p0[1]
    print("\nTime-domain prediction (linear model, ideal actuator, 40 s, no saturation):")
    print(f"{'move':>26} {'rate':>7} {'ff':>4} {'peak tilt err':>14} {'peak speed':>11} {'final base x':>13} {'final trim':>11}")
    for dH, rate in ((-0.03, 0.01), (-0.06, 0.01), (+0.01, 0.01)):
        for ff in (False, True):
            pe, pw, sx, tr = simulate(rb, H0, H0 + dH, rate, gains, ff)
            print(f"{'H %.4f -> %.4f' % (H0, H0 + dH):>26} {rate:7.3f} {'on' if ff else 'off':>4} "
                  f"{pe:11.4f} rad {pw:7.2f} rad/s {sx:10.3f} m {tr:+10.4f}")


# --------------------------------------------------------------------------- report
def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    gains = dict(GAINS)
    if "--gains" in sys.argv:
        i = sys.argv.index("--gains")
        gains = dict(zip(("kp", "kd", "kvp", "kvi"), map(float, sys.argv[i + 1:i + 5])))
    rb = Robot(sys.argv[1])

    print(f"thigh L1 = {rb.L1:.4f} m, shank L2 = {rb.L2:.4f} m, "
          f"angle between them at q=0: {math.degrees(rb.psi0):.1f} deg")
    H0 = -rb.p0[1]
    print(f"nominal hip->axle: y = {rb.p0[0]:+.4f}, H0 = {H0:.4f} m; "
          f"reach {abs(rb.L2 - rb.L1):.4f} .. {rb.L1 + rb.L2:.4f} m")
    b0 = rb.body(0.0, 0.0)
    print(f"\n[check q=0] m={b0['m']:.3f} kg  d={b0['d']:.4f} m  I={b0['I']:.5f}  "
          f"l={b0['l']:.4f} m  theta0={b0['theta0']:+.4f} rad")
    print("            report Table 2: m=1.103, h=0.139, I=0.0259, l=0.170, theta0=-0.0037")

    ev = np.linalg.eigvals(cascade_A(b0["l"], **gains))
    print("[check] eigenvalues at q=0:", np.round(sorted(ev, key=lambda z: z.real), 2))
    print("        report (3b, l=0.17): -0.31+-5.20j, -0.18, -2.36 ; critical lag 9.4 ms")
    print(f"        this model critical lag: {1e3 * critical_lag(b0['l'], gains):.1f} ms\n")

    M_total = b0["m"] + rb.wheel_mass()
    F = M_total * G / 2.0                          # vertical wheel force per leg
    print(f"gains kp={gains['kp']} kd={gains['kd']} kvp={gains['kvp']} kvi={gains['kvi']}")
    print(f"static wheel load per leg F = {F:.2f} N\n")
    hdr = (f"{'H(m)':>6} {'dH(cm)':>7} {'hip(deg)':>9} {'knee(deg)':>10} {'l(m)':>6} "
           f"{'theta0':>8} {'maxRe':>7} {'crit.lag':>9} {'tau_hip':>8} {'tau_knee':>9}")
    print(hdr)
    print("-" * len(hdr))
    for H in np.arange(0.09, 0.2251, 0.01):
        try:
            ph, pk = rb.ik(H)
        except ValueError:
            continue
        b = rb.body(ph, pk)
        knee, axle = b["knee"], b["axle"]
        tau_hip = F * abs(axle[0] - rb.hip[0])
        tau_knee = F * abs(axle[0] - knee[0])
        cl = critical_lag(b["l"], gains)
        cl_s = "unstable" if cl == 0 else ("   >200" if cl == float("inf") else f"{1e3*cl:7.1f}ms")
        print(f"{H:6.3f} {100*(H-H0):+7.1f} {math.degrees(ph):+9.1f} {math.degrees(pk):+10.1f} "
              f"{b['l']:6.3f} {b['theta0']:+8.4f} {max_re(b['l'], gains):+7.2f} {cl_s:>9} "
              f"{tau_hip:8.3f} {tau_knee:9.3f}")
    if "--simulate" in sys.argv:
        run_simulations(rb, gains)


if __name__ == "__main__":
    main()
