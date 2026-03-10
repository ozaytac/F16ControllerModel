# F-16 6-DOF Nonlinear Flight Model - Python

Python translation of the F-16 6-DOF nonlinear flight dynamics model originally written in MATLAB by E. van Oort & L. Sonneveldt (2006), based on NASA report 1538.

## Credits

Original MATLAB source by **[vvrn1](https://github.com/vvrn1)**:

- Repository: [https://github.com/vvrn1/F16model](https://github.com/vvrn1/F16model)

All aerodynamic lookup tables, engine model, ISA atmosphere, and 6-DOF equations of motion are derived from that work. If you use this code, please star and credit the original repository.

## Files

| File | Translates | Description |
|---|---|---|
| `f16_parameters.py` | `F16parameter.m` | Aircraft mass, inertia, geometry; derived inertia constants |
| `f16_atmosphere.py` | `ISA_atmos.m` | ISA atmosphere - Mach, dynamic pressure, density, gravity |
| `f16_enginedata.py` | `F16enginedata.m` | Engine thrust tables (Idle / Mil / Max vs Mach & altitude) |
| `f16_engine.py` | `tgear.m`, `power_dot.m`, `engine_power.m` | Throttle mapping, power rate, thrust interpolation |
| `f16_aerodata.py` | `F16aerodata.m` | Runtime parser - reads `F16aerodata.m` and returns lookup tables |
| `f16_model.py` | `F16model.m`, `F16modeltrim.m` | 6-DOF step function and trim cost function |
| `F16aerodata.m` | - | Original MATLAB data file (required at runtime by `f16_aerodata.py`) |
| `test_f16_model.py` | - | pytest suite (39 mock tests + 3 integration tests) |

## Requirements

```
numpy
scipy
pytest   # for running tests only
```

Install with:

```bash
pip install numpy scipy pytest
```

## Quick Start

### Run one simulation step

```python
import numpy as np
from f16_aerodata import get_f16_aerodata
from f16_enginedata import DEFAULT_ENGINE_DATA
from f16_model import f16_model

# Load aerodynamic tables (parsed from F16aerodata.m once)
aero = get_f16_aerodata()

# State vector (14 elements)
# [X_earth, Y_earth, Z_earth, Vt, alpha, beta, q0, q1, q2, q3, p, q, r, power]
state = np.zeros(14)
state[2]  = -5000.0   # Z_earth (negative = 5000 m altitude)
state[3]  =  200.0    # airspeed [m/s]
state[6]  =  1.0      # q0 - unit quaternion (wings level, zero attitude)
state[13] =  50.0     # engine power [0–100]

# Action vector (4 elements): [aileron (rad), elevator (rad), rudder (rad), throttle (0–1)]
action = np.array([0.0, np.radians(-2.0), 0.0, 0.6])

dt = 0.01   # time step [s]

next_state = f16_model(state, action, dt, aero_data=aero, engine_data=DEFAULT_ENGINE_DATA)
print(next_state)
```

### State vector layout

| Index | Symbol | Unit | Description |
|---|---|---|---|
| 0 | X_earth | m | Position north (NED frame) |
| 1 | Y_earth | m | Position east |
| 2 | Z_earth | m | Position down (negative = above ground) |
| 3 | Vt | m/s | True airspeed |
| 4 | alpha | rad | Angle of attack |
| 5 | beta | rad | Sideslip angle |
| 6 | q0 | - | Quaternion scalar part |
| 7 | q1 | - | Quaternion vector x |
| 8 | q2 | - | Quaternion vector y |
| 9 | q3 | - | Quaternion vector z |
| 10 | p | rad/s | Roll rate (body) |
| 11 | q | rad/s | Pitch rate (body) |
| 12 | r | rad/s | Yaw rate (body) |
| 13 | power | - | Engine power level [0–100] |

### Action vector layout

| Index | Symbol | Limits | Description |
|---|---|---|---|
| 0 | aileron | ±21.5° | Aileron deflection [rad] |
| 1 | elevator | −25° to +25° | Elevator deflection [rad] |
| 2 | rudder | ±30° | Rudder deflection [rad] |
| 3 | throttle | 0–1 | Throttle position |

### Trim cost function

`f16_trim_cost` evaluates how far a candidate control / attitude vector is from a trimmed (steady-state) flight condition. Minimise it with `scipy.optimize.minimize` to find a trim point.

```python
from scipy.optimize import minimize
from f16_model import f16_trim_cost
from f16_aerodata import get_f16_aerodata
from f16_enginedata import DEFAULT_ENGINE_DATA
import numpy as np

aero = get_f16_aerodata()

state0 = np.zeros(14)
state0[2]  = -5000.0
state0[3]  =  200.0
state0[6]  =  1.0
state0[13] =  50.0

# UX = [alpha, phi, theta, elevator, rudder, throttle]
UX0 = np.array([np.radians(5.0), 0.0, 0.0, np.radians(-2.0), 0.0, 0.6])

result = minimize(
    f16_trim_cost, UX0,
    args=(state0, 0.01, aero, DEFAULT_ENGINE_DATA),
    method="Nelder-Mead",
)
print("Trim cost:", result.fun)
print("Trim UX  :", result.x)
```

### Mock mode (no data files needed)

Pass `aero_data=None` and `engine_data=None` to use lightweight mocks. All aerodynamic coefficients are zero and thrust scales linearly with power. Useful for unit testing control logic without the full tables.

```python
next_state = f16_model(state, action, dt, aero_data=None, engine_data=None)
```

## Running Tests

```bash
# Fast mock-only tests (no data files required)
pytest test_f16_model.py -m "not integration"

# Full suite including real aero/engine data
pytest test_f16_model.py
```

Expected output: **42 passed**.
