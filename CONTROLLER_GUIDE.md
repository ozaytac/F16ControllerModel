# F-16 6-DOF Flight Controller - Development Guide

> **Why this document exists:** Getting a stable flight demo from the F-16 6-DOF model
> took ~30 iterations of debugging. This guide documents every pitfall so you don't
> repeat them.

## Source & Attribution

The original MATLAB F-16 model (6-DOF aerodynamics, engine model, trim solver) was
authored by **[vvrn1](https://github.com/vvrn1)** and is available at:

> 🔗 **<https://github.com/vvrn1/F16model>**

This Python implementation is a direct port of that codebase. The aerodynamic lookup
tables (`F16aerodata.m`), engine model (`F16enginedata.m`), ISA atmosphere, and
6-DOF equations of motion are all derived from the original MATLAB source.
All credit for the underlying flight model goes to the original author.

If you use this code, please star and credit the original repository.

---

## Files

| File | Description |
|------|-------------|
| `fly_demo.py` | **Main entry point.** 90 s flight demo: cruise → banked turn → rollout. Saves `trajectory.png`. |
| `f16_model.py` | 6-DOF step function and trim cost function |
| `f16_aerodata.py` | Parses `F16aerodata.m` at runtime and returns lookup tables |
| `F16aerodata.m` | **Required at runtime.** Must be in the same directory as the Python files. |
| `f16_enginedata.py` | Engine thrust tables (Idle / Mil / Max vs Mach & altitude) |
| `f16_engine.py` | Throttle mapping, power rate, thrust interpolation |
| `f16_atmosphere.py` | ISA atmosphere - Mach, dynamic pressure, density, gravity |
| `f16_parameters.py` | Aircraft mass, inertia, geometry |
| `test_f16_model.py` | pytest suite - 42 tests (mock + integration) |

---

## Requirements

```
numpy
scipy
matplotlib
pytest      # for running tests only
```

Install with:

```bash
pip install numpy scipy matplotlib pytest
```

Python 3.10+, tested on Google Colab (A100) and local CPython.

---

## Running the Demo

```bash
python fly_demo.py
```

Expected output:

```
Loading aerodynamic tables …
Searching for trim condition …
  Trim cost : 1.494e-16
  AoA       : 3.79 °
  Elevator  : -1.45 °
  Throttle  : 0.229
Simulating 90 s  (9000 steps, dt = 10 ms) …
Simulation complete  (90.0 s).
Saved  →  trajectory.png
```

The trim search takes a few seconds. The simulation itself runs in under a second.
Output plot is saved as `trajectory.png` in the working directory.

**Validated results at Mach 0.6 / 6000 m:**

| Metric | Value |
|--------|-------|
| Altitude deviation | ±2 m |
| Bank angle at t=90 s | 0.00° |
| Speed deviation | −1 km/h |
| Rollout time (−15° → 0°) | < 5 s |

---

## Model Overview

- **State vector (14):** `[X, Y, Z, Vt, α, β, q0, q1, q2, q3, p, q, r, power]`
- **Action vector (4):** `[da, de, dr, dp]` - aileron, elevator, rudder (rad), throttle [0–1]
- **Trim point used:** Mach 0.6, Alt = 6000 m
- **Trim cost achieved:** `1.494e-16` (machine precision)

---

## Critical Sign Convention

**The most common source of bugs.** Always verify before writing any controller.

```python
de > 0  →  nose DOWN   (trailing edge up,   Cm decreases)
de < 0  →  nose UP     (trailing edge down,  Cm increases)

da > 0  →  LEFT roll   (phi decreases)   ← counter-intuitive
da < 0  →  RIGHT roll  (phi increases)

p = state[10]   # roll rate  [rad/s]
q = state[11]   # pitch rate [rad/s]
r = state[12]   # yaw rate   [rad/s]
```

> **Verified by step test:** applying `da = +1°` for 1 s produces `p → −13°/s`
> (left roll). The sign is opposite to the SAE convention used in some other F-16
> references. Confirm with a step test before writing any roll controller.

Wrong sign on any feedback term → **positive feedback → divergence within seconds.**

---

## State Indices (Quick Reference)

| Index | Symbol | Unit | Description |
|-------|--------|------|-------------|
| 0 | X | m | North position |
| 1 | Y | m | East position |
| 2 | Z | m | Down position (altitude = −Z) |
| 3 | Vt | m/s | True airspeed |
| 4 | α | rad | Angle of attack |
| 5 | β | rad | Sideslip angle |
| 6–9 | q0–q3 | - | Quaternion attitude |
| 10 | p | rad/s | Roll rate |
| 11 | q | rad/s | Pitch rate |
| 12 | r | rad/s | Yaw rate |
| 13 | power | % | Engine power state |

---

## Working Controller (Final)

Tested for 90 s at Mach 0.6 / 6000 m. Deviations: alt ±2 m, bank 0.00°, speed −1 km/h.

```python
import numpy as np

# --- Unpack state ---
alt    = -state[2]
Vt     = state[3]
alpha  = state[4]
p_body = state[10]   # roll rate
q_body = state[11]   # pitch rate

# Euler angles from quaternion
q0, q1, q2, q3 = state[6], state[7], state[8], state[9]
phi   = np.arctan2(2*(q0*q1 + q2*q3), 1.0 - 2*(q1**2 + q2**2))
theta = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))

# --- Altitude hold ---
alt_err    = alt - alt_target
climb_rate = Vt * np.sin(theta - alpha)   # ← IMPORTANT: use (theta - alpha), not theta

de  = de_trim
de += 0.0005 * np.clip(alt_err, -100, 100)   # proportional - max ±2.9°
de += 0.20   * q_body                         # pitch rate damping
de -= 0.01   * (1.0 / np.cos(phi) - 1.0)     # lift compensation in bank
de += 0.005  * climb_rate                     # phugoid damper
de  = np.clip(de, -25*np.pi/180, 25*np.pi/180)

# --- Roll PD controller ---
# da > 0 = LEFT roll in this model → error = (phi - phi_cmd), not (phi_cmd - phi)
phi_cmd = desired_bank_angle   # rad
da  = 0.15 * (phi - phi_cmd) + 0.08 * p_body
da  = np.clip(da, -15*np.pi/180, 15*np.pi/180)

# --- Speed hold (throttle only - decoupled from altitude) ---
dp = np.clip(dp_trim + 1.0 * (Vt_target - Vt) / Vt_target, 0.0, 1.0)

# --- Rudder ---
dr = 0.0   # sufficient for coordinated turn at low bank angles
```

---

## Maneuver Schedule

```python
# Ramp phi_cmd for smooth roll-in (avoids inertia overshoot)
if t < t_turn:
    phi_cmd = 0.0
elif t < t_turn + 5.0:
    phi_cmd = -15.0 * np.pi/180 * (t - t_turn) / 5.0   # 5s linear ramp
elif t < t_rollout:
    phi_cmd = -15.0 * np.pi/180
else:
    phi_cmd = 0.0
```

**Why −15° and not −25°:** At Mach 0.6, `cos(25°) = 0.906` → 10% lift deficit → altitude
controller saturates → speed bleed → spiral. At 15°, `cos(15°) = 0.966` → trivially
compensated by the `1/cos(φ)` term.

---

## Pitfalls - Ordered by Severity

### 1. Aileron sign convention (causes immediate divergence)

This model uses a **non-standard aileron sign**: `da > 0` produces **left roll**.
Most references and intuition assume the opposite.

```python
# WRONG - positive error (phi > phi_cmd) → positive da → MORE left roll → diverges
da = 0.15 * (phi_cmd - phi) - 0.08 * p_body

# CORRECT - positive error (phi > phi_cmd) → positive da → corrects left
da = 0.15 * (phi - phi_cmd) + 0.08 * p_body
```

Verify with a 1 s step test before writing any roll controller:

```python
action = np.array([np.radians(1.0), de_trim, 0.0, dp_trim])
# After 1 s: state[10] (p) should be negative → left roll ✓
```

---

### 2. Climb rate formula (silent constant bias)

```python
# WRONG - introduces constant bias at trim
climb_rate = Vt * np.sin(theta)

# CORRECT - zero at trim (theta ≈ alpha at level flight)
climb_rate = Vt * np.sin(theta - alpha)
```

At trim with Vt=190 m/s and theta=3.79°, the wrong formula outputs +12.5 m/s permanently.
With gain 0.005, that's a 0.063 rad (3.6°) constant nose-down bias → steady descent of −2.4 m/s.

---

### 3. Altitude hold gain too high → drag spiral

```
gain × alt_error = elevator deflection
0.002 × 118 m   = 0.236 rad = 13.5°  ← nearly full deflection
```

13.5° nose-up elevator → massive drag → speed drops → more nose-up needed → spiral.

**Rule of thumb:** Keep max elevator from altitude hold below **3°**.
With gain 0.0005 and clip ±100 m: max = 0.0005 × 100 = 0.05 rad = 2.9°. ✓

---

### 4. Roll: open-loop pulse doesn't work

F-16 roll rate at Mach 0.6 with 3° aileron ≈ **30°/s**.
A 5 s pulse rolls the aircraft **150°**, not 20°.

Threshold-based cutoff (stop at target bank) also fails due to roll inertia -
at cutoff the aircraft is already spinning at ~190°/s and overshoots by 100°+.

**Solution:** PD controller with rate damping:

```python
da = np.clip(0.15*(phi - phi_cmd) + 0.08*p_body, -da_max, +da_max)
```

The `+0.08*p_body` term arrests the roll rate before target is reached.

---

### 5. Coupling throttle to altitude → phugoid amplification

Energy-based throttle `dp ∝ (E_target − E_actual)` sounds physically correct but
**adds thrust at exactly the wrong phase** of a phugoid oscillation:

- Aircraft zooms high + slows down → energy deficit → throttle increases
- Extra thrust accelerates the aircraft → next zoom is larger → diverges

**Solution:** Decouple throttle from altitude. Track **speed only**:

```python
dp = np.clip(dp_trim + 1.0*(Vt_target - Vt)/Vt_target, 0, 1)
```

---

### 6. Asymmetric altitude clip

The altitude error clip must be asymmetric based on your dominant failure mode:

| Clip | Behavior |
|------|----------|
| `clip(err, -50, 150)` | Strong nose-down, weak nose-up → use when zoom is the problem |
| `clip(err, -150, 50)` | Weak nose-down, strong nose-up → use when sink is the problem |
| `clip(err, -100, 100)` | Symmetric → general use after phugoid is damped |

---

### 7. Pitch damping sign

```python
# WRONG - positive feedback, diverges in ~10s
de -= 0.10 * q_body   # pitch up → removes nose-down → amplifies

# CORRECT - negative feedback
de += 0.10 * q_body   # pitch up (q>0) → adds nose-down → damps
```

---

### 8. Elevator gain with airspeed scaling

`0.1 × Vt × sin(θ−α)` as a damping term introduces an implicit gain of `0.1 × Vt ≈ 19`
at Mach 0.6. That's two orders of magnitude too large. Use `q_body` (pitch rate) directly -
it's already normalized and doesn't scale with airspeed.

---

## Stability Envelope

This controller was validated **only** at:

- Mach 0.6, Alt 6000 m
- Bank angles up to ±15°
- 90 s simulation

**Known limitations:**

| Condition | Status |
|-----------|--------|
| Mach > 0.9 | Untested - F-16 is aerodynamically unstable, requires FBW |
| Bank > 45° | Lift deficit exceeds controller authority |
| Alt < 1000 m | Ground effect not modeled |
| Aggressive maneuvers | No structural/AoA limits enforced |

---

## Trim Solver Notes

The MATLAB→Python conversion introduced two bugs in the trim solver:

```python
# Bug 1: DLEF unit error in aerodata
# MATLAB clips in degrees but value was in radians
np.clip(DLEF, 0, 25*np.pi/180)  # WRONG - clips to 0.44 rad
np.clip(DLEF, 0, 25.0)           # CORRECT - clips to 25 deg

# Bug 2: scipy RGI returns array, not scalar
float(rgi([[alpha, beta, de]]))     # WRONG - shape mismatch
float(rgi([[alpha, beta, de]])[0])  # CORRECT
```

After fixes, trim cost at Mach 0.6 / 6000 m: `1.494e-16` (machine precision).

---

## Iteration History (Condensed)

| Attempt | Change | Alt dev | Bank range | Issue |
|---------|--------|---------|------------|-------|
| 1 | Open-loop Mach 1.14 | 5000 m | ±180° | Aerodynamically unstable regime |
| 2 | Open-loop Mach 0.6 | 1200 m | ±180° | Aileron gain × 100 too high |
| 3 | Open-loop pulse | 1800 m | ±157° | Roll inertia overshoots |
| 4 | Threshold cutoff | 2000 m | ±150° | Inertia carries past threshold |
| 5 | PD roll + wrong alt | 2286 m | ±43° | Alt gain 0.002 → drag spiral |
| 6 | PD roll + alt clipped | 1025 m | ±40° | Speed bleed, no throttle |
| 7 | + Speed hold | 800 m | ±35° | Phugoid appears post-rollout |
| 8 | + Climb rate damper | 237 m | ±5° | Wrong climb_rate formula → sink |
| 9 | Fix climb_rate = Vt·sin(θ−α) | **±2 m** | **±0.1°** | ✅ **Final** |

---

## Quick Diagnostic Checklist

If the simulation diverges, check in this order:

1. **Aileron sign** - run 1 s step test, confirm `da=+1° → p < 0` (left roll)
2. **Signs first** - print de, da at t=1s, verify direction matches intent
3. **Trim cost** - should be < 1e-6; if not, retrim before flying
4. **Gains × max error** - compute max elevator/aileron deflection analytically
5. **climb_rate formula** - print at t=0, should be ~0 at trim
6. **Throttle coupling** - is dp responding to altitude or speed?
7. **Roll rate** - print p_body during maneuver; if > 50 deg/s, Kd too small
