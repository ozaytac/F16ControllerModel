"""
fly_demo.py — F-16 6-DOF flight demonstration (simplified)
===========================================================
Altitude : 6 000 m
Speed    : Mach 0.6  (~190 m/s at 6 km)

Maneuver sequence
-----------------
  0 –  30 s   Straight & level cruise
 30 –  60 s   Left banked turn  (φ_cmd = −20°)
 60 –  90 s   Roll-out and cruise

Sign convention
---------------
  Positive elevator de → Cm decreases → nose DOWN (NASA/MATLAB tables).

  Altitude hold (proportional only):
    de = de_trim + 0.002 * (alt − alt_cmd)
    Too high (alt > alt_cmd) → positive error → increase de → nose down ✓

Run
---
    python fly_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from f16_aerodata   import get_f16_aerodata
from f16_enginedata import DEFAULT_ENGINE_DATA
from f16_engine     import tgear
from f16_model      import f16_model, f16_trim_cost

# ── Mission parameters ────────────────────────────────────────────────────────
ALT_M  = 6_000.0          # target altitude [m]
MACH   = 0.6
# ISA at 6 000 m: T ≈ 249.15 K → a ≈ 316.5 m/s
VT_MS  = MACH * 316.5     # ≈ 189.9 m/s
DT     = 0.01             # 10 ms integration step
T_END  = 90.0

# Controller gains  (proportional only — keep it simple)
# Roll PD (critically damped for F-16 roll τ ≈ 0.5 s at Mach 0.6)
KP_PHI = 0.15             # proportional  [rad/rad]
KD_PHI = 0.08             # derivative    [rad/(rad/s)]
DA_LIM = np.radians(15.0) # aileron authority limit

# ── Load aero tables ──────────────────────────────────────────────────────────
print("Loading aerodynamic tables …")
aero = get_f16_aerodata()

# ── Trim search ───────────────────────────────────────────────────────────────
print("Searching for trim condition …")
trim_s0       = np.zeros(14)
trim_s0[2]    = -ALT_M
trim_s0[3]    =  VT_MS
# UX = [alpha/theta, beta, da, de, dr, throttle]
UX0 = np.array([np.radians(4.0), 0.0, 0.0, np.radians(-2.0), 0.0, 0.50])
res = minimize(
    f16_trim_cost, UX0,
    args=(trim_s0, DT, None, aero, DEFAULT_ENGINE_DATA),
    method="Nelder-Mead",
    options=dict(xatol=1e-8, fatol=1e-12, maxiter=80_000),
)
alpha_t = res.x[0]
de_t    = res.x[3]
dp_t    = float(np.clip(res.x[5], 0.0, 1.0))
power_t = tgear(dp_t)
print(f"  Trim cost : {res.fun:.3e}")
print(f"  AoA       : {np.degrees(alpha_t):.2f} °")
print(f"  Elevator  : {np.degrees(de_t):.2f} °")
print(f"  Throttle  : {dp_t:.3f}")

# ── Build initial simulation state ────────────────────────────────────────────
s0     = np.zeros(14)
s0[2]  = -ALT_M
s0[3]  =  VT_MS
s0[4]  =  alpha_t
s0[6]  =  np.cos(alpha_t / 2.0)   # q0 — pure pitch quaternion
s0[8]  =  np.sin(alpha_t / 2.0)   # q2
s0[13] =  power_t

# ── Quaternion → roll & pitch ─────────────────────────────────────────────────
def _euler(st):
    q0, q1, q2, q3 = st[6:10]
    phi   = np.arctan2(2*(q0*q1 + q2*q3), 1.0 - 2*(q1**2 + q2**2))
    theta = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
    return phi, theta

# ── Maneuver schedule ────────────────────────────────────────────────────────
def _phi_cmd(t):
    if t < 20.0:  return 0.0
    if t < 60.0:  return -np.radians(15.0) * min((t - 20.0) / 5.0, 1.0)
    return 0.0

# ── PD Controller ─────────────────────────────────────────────────────────────
def get_action(t, state):
    phi, theta = _euler(state)
    p_body  = state[10]              # roll rate  [rad/s]
    q_body  = state[11]              # pitch rate [rad/s]
    vt      = state[3]               # true airspeed [m/s]
    alt     = -state[2]              # altitude [m]

    # Throttle: speed only (decoupled from altitude)
    dp = float(np.clip(dp_t + 1.0 * (VT_MS - vt) / VT_MS, 0.0, 1.0))

    # Elevator: altitude hold + pitch damping + climb rate + bank lift comp
    # positive de = nose DOWN → signs opposite to intuition
    cos_phi    = max(np.cos(phi), 0.1)
    alt_err    = float(np.clip(alt - ALT_M, -100.0, 100.0))
    climb_rate = vt * np.sin(theta - state[4])   # flight-path angle, zero at trim
    de = de_t
    de += 0.0005 * alt_err                          # ±100 m → ±0.05 rad
    de += 0.20   * q_body                           # pitch rate damping
    de += 0.005  * climb_rate                       # phugoid damper: climb → nose down
    de -= 0.01   * (1.0 / cos_phi - 1.0)           # bank lift compensation
    de  = float(np.clip(de, np.radians(-25.0), np.radians(25.0)))

    # Roll PD  (positive da → left roll in this model, so signs are reversed)
    da = float(np.clip(KP_PHI * (phi - _phi_cmd(t)) + KD_PHI * p_body,
                       -DA_LIM, DA_LIM))

    return np.array([da, de, 0.0, dp])

# ── Simulate ──────────────────────────────────────────────────────────────────
n_steps = int(T_END / DT)
t_arr   = np.linspace(0.0, T_END, n_steps + 1)
st_arr  = np.zeros((n_steps + 1, 14))
st_arr[0] = s0

print(f"Simulating {T_END:.0f} s  ({n_steps} steps, dt = {DT*1000:.0f} ms) …")
state = s0.copy()
last  = n_steps
for i in range(n_steps):
    action = get_action(i * DT, state)
    state  = f16_model(state, action, DT, aero_data=aero, engine_data=DEFAULT_ENGINE_DATA)

    # Renormalise quaternion to suppress drift
    qnorm = np.linalg.norm(state[6:10])
    if qnorm > 1e-6:
        state[6:10] /= qnorm

    if np.any(~np.isfinite(state)):
        print(f"  ⚠  Non-finite state at t = {i*DT:.2f} s — stopping early.")
        last = i
        break
    st_arr[i + 1] = state

t_arr  = t_arr[:last + 1]
st_arr = st_arr[:last + 1]
print(f"Simulation complete  ({t_arr[-1]:.1f} s).")

# ── Derived quantities ────────────────────────────────────────────────────────
x_km   =  st_arr[:, 0] / 1_000.0
y_km   =  st_arr[:, 1] / 1_000.0
alt_km = -st_arr[:, 2] / 1_000.0
vt_kmh =  st_arr[:, 3] * 3.6
phi_d  =  np.degrees([_euler(s)[0] for s in st_arr])
alp_d  =  np.degrees(st_arr[:, 4])

# ── Plot ──────────────────────────────────────────────────────────────────────
BG, DARK = "#0e1117", "#1a1d27"

PHASES = [
    (  0, 20, "Cruise",     "#4CAF50"),
    ( 20, 60, "Left turn",  "#2196F3"),
    ( 60, 90, "Roll-out",   "#9C27B0"),
]

fig = plt.figure(figsize=(15, 9), facecolor=BG)
fig.suptitle(
    f"F-16 6-DOF  ·  Alt = {ALT_M/1000:.0f} km  ·  Mach {MACH}  ·  {t_arr[-1]:.0f} s",
    fontsize=13, fontweight="bold", color="white", y=0.98,
)
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38,
                      top=0.92, bottom=0.12, left=0.06, right=0.97)
ax3 = fig.add_subplot(gs[0, :], projection="3d")
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])

def _style(ax, title):
    ax.set_facecolor(DARK)
    for sp in ax.spines.values(): sp.set_edgecolor("#555")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.xaxis.label.set_color("#aaa"); ax.yaxis.label.set_color("#aaa")
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.grid(alpha=0.18, color="gray")

def _bands(ax):
    for t0, t1, _, col in PHASES:
        if t0 >= t_arr[-1]: break
        ax.axvspan(t0, min(t1, t_arr[-1]), alpha=0.09, color=col, lw=0)

# 3-D trajectory
ax3.set_facecolor(DARK)
sc = ax3.scatter(x_km, y_km, alt_km, c=t_arr, cmap="plasma", s=3, linewidths=0)
ax3.plot(x_km, y_km, alt_km, color="#5c9bd6", lw=0.6, alpha=0.3)
ax3.scatter(x_km[0],  y_km[0],  alt_km[0],  color="lime", s=80, zorder=6, label="Start")
ax3.scatter(x_km[-1], y_km[-1], alt_km[-1], color="red",  s=80, zorder=6, label="End")
ax3.set_xlabel("X – north (km)", color="lightgray", fontsize=8)
ax3.set_ylabel("Y – east  (km)", color="lightgray", fontsize=8)
ax3.set_zlabel("Altitude  (km)", color="lightgray", fontsize=8)
ax3.tick_params(colors="lightgray", labelsize=7)
ax3.set_title("3-D Trajectory  (colour = elapsed time)", color="white", fontsize=10)
ax3.view_init(elev=25, azim=-55)
ax3.legend(fontsize=8, facecolor=DARK, edgecolor="#555", labelcolor="white")
cb = plt.colorbar(sc, ax=ax3, label="Time (s)", shrink=0.52, pad=0.08)
cb.ax.yaxis.set_tick_params(color="lightgray", labelcolor="lightgray")
cb.set_label("Time (s)", color="lightgray")

# Altitude
_style(ax1, "Altitude")
ax1.plot(t_arr, alt_km, color="#5c9bd6", lw=1.5)
ax1.axhline(ALT_M / 1000, color="lime", lw=0.8, ls="--", alpha=0.6, label="Target")
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("km")
ax1.legend(fontsize=7, facecolor=DARK, edgecolor="#555", labelcolor="white")
_bands(ax1)

# Airspeed
_style(ax2, "Airspeed")
ax2.plot(t_arr, vt_kmh, color="#e05c5c", lw=1.5)
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("km/h")
_bands(ax2)

# Bank angle + AoA
_style(ax4, "Bank Angle  &  AoA")
ax4.plot(t_arr, phi_d, color="#80cbc4", lw=1.4, label="φ  (°)")
ax4.plot(t_arr, alp_d, color="#f0c040", lw=1.2, label="α  (°)")
ax4.set_xlabel("Time (s)")
ax4.legend(fontsize=7, facecolor=DARK, edgecolor="#555", labelcolor="white")
_bands(ax4)

# Phase legend strip
handles = [plt.Rectangle((0,0),1,1, facecolor=c, alpha=0.7, label=lbl)
           for _, _, lbl, c in PHASES]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
           facecolor=DARK, edgecolor="#555", labelcolor="white",
           bbox_to_anchor=(0.5, 0.01))

out = "trajectory.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved  →  {out}")
