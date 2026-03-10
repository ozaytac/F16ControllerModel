"""F-16 6-DOF nonlinear flight model (Euler quaternion formulation).

Translates two MATLAB files:
  * F16model.m       — one Euler-integration step
  * F16modeltrim.m   — trim cost function for use with an optimiser

The *aero_data* / *engine_data* keyword arguments let callers inject custom
data objects (or ``None`` for lightweight mock data).  This supports the
``model=None`` pytest pattern used throughout the project.

State vector (14 elements, indices 0–13)
-----------------------------------------
0  X_earth   – inertial x position [m]
1  Y_earth   – inertial y position [m]
2  Z_earth   – inertial z position [m]   (negative = above ground)
3  Vt        – true airspeed [m/s]
4  alpha     – angle of attack [rad]
5  beta      – sideslip angle [rad]
6  q0        – quaternion scalar part
7  q1        – quaternion vector part (x)
8  q2        – quaternion vector part (y)
9  q3        – quaternion vector part (z)
10 p_body    – roll rate [rad/s]
11 q_body    – pitch rate [rad/s]
12 r_body    – yaw rate [rad/s]
13 power     – engine power level [0–100]

Action vector (4 elements)
---------------------------
0  da  – aileron deflection [rad]
1  de  – elevator deflection [rad]
2  dr  – rudder deflection [rad]
3  dp  – throttle [0, 1]

References
----------
Nguyen, L. T. et al., "Simulator study of stall/post-stall characteristics
of a fighter airplane with relaxed longitudinal static stability", NASA TP-1538,
1979.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator

from f16_parameters import F16Parameters, DEFAULT_PARAMS, RTD
from f16_atmosphere import isa_atmos
from f16_engine import tgear, power_dot, engine_power
from f16_enginedata import F16EngineData, DEFAULT_ENGINE_DATA


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
State = NDArray[np.float64]    # shape (14,)
Action = NDArray[np.float64]   # shape (4,)


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _interp1(xp: NDArray, fp: NDArray, x: float) -> float:
    """1-D linear interpolation with clamp extrapolation (MATLAB interp1).

    Parameters
    ----------
    xp : array_like
        Monotonically increasing x-coordinates of data points.
    fp : array_like
        y-values at *xp*.
    x : float
        Query point.
    """
    return float(np.interp(x, xp, fp))


def _interp2(
    alpha_pts: NDArray,
    beta_pts: NDArray,
    data: NDArray,
    alpha_q: float,
    beta_q: float,
) -> float:
    """2-D bilinear interpolation (mirrors MATLAB ``interp2(beta,alpha,V,BETA,ALPHA)``).

    Parameters
    ----------
    alpha_pts : NDArray
        Alpha breakpoints (row axis).
    beta_pts : NDArray
        Beta breakpoints (column axis).
    data : NDArray
        Table, shape ``(len(alpha_pts), len(beta_pts))``.
    alpha_q, beta_q : float
        Query point in (alpha, beta) space (degrees when called from model).
    """
    rgi = RegularGridInterpolator(
        (alpha_pts, beta_pts),
        data,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return float(rgi([[alpha_q, beta_q]])[0])


def _interp3(
    alpha_pts: NDArray,
    beta_pts: NDArray,
    de_pts: NDArray,
    data: NDArray,
    alpha_q: float,
    beta_q: float,
    de_q: float,
) -> float:
    """3-D trilinear interpolation (mirrors MATLAB ``interp3(beta,alpha,de,V,BETA,ALPHA,DE)``).

    In MATLAB ``interp3(X,Y,Z,V,Xq,Yq,Zq)``: X → columns (axis 1),
    Y → rows (axis 0), Z → pages (axis 2).  The data array must have
    shape ``(len(alpha_pts), len(beta_pts), len(de_pts))``.
    """
    rgi = RegularGridInterpolator(
        (alpha_pts, beta_pts, de_pts),
        data,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return float(rgi([[alpha_q, beta_q, de_q]])[0])


# ---------------------------------------------------------------------------
# Aerodynamic coefficient computation
# ---------------------------------------------------------------------------

def _compute_aero_coeffs(
    ALPHA: float,
    BETA: float,
    DE: float,
    DA: float,
    DR: float,
    da_norm: float,
    dr_norm: float,
    dlef_norm: float,
    alpha_lef: float,
    q_body: float,
    p_body: float,
    r_body: float,
    Vt: float,
    beta: float,
    p: F16Parameters,
    d: object,          # aero data namespace / object
) -> tuple[float, float, float, float, float, float]:
    """Compute the six total aerodynamic force/moment coefficients.

    Translated from F16model.m lines 116–243.

    Parameters
    ----------
    ALPHA, BETA, DE, DA, DR : float
        Aerodynamic angles and control surface deflections [deg].
    da_norm, dr_norm, dlef_norm : float
        Normalised control deflections.
    alpha_lef : float
        Alpha clipped to LEF table limit [deg].
    q_body, p_body, r_body : float
        Body-axis angular rates [rad/s].
    Vt : float
        True airspeed [m/s].
    beta : float
        Sideslip [rad] (used in Cl_tot, Cn_tot normalisation).
    p : F16Parameters
        Aircraft parameters.
    d : object
        Aerodynamic data with attributes matching ``f16_aerodata.py``.

    Returns
    -------
    CX_tot, CY_tot, CZ_tot, Cl_tot, Cm_tot, Cn_tot : float
    """

    # ------------------------------------------------------------------
    # CX — axial force coefficient (F16model.m lines 117–120)
    # ------------------------------------------------------------------
    CX0 = _interp3(d.alpha1, d.beta, d.de1, d.CX, ALPHA, BETA, DE)
    delta_CX_lef = (
        _interp2(d.alpha2, d.beta, d.CX_lef, alpha_lef, BETA)
        - _interp3(d.alpha1, d.beta, d.de1, d.CX, ALPHA, BETA, 0.0)
    )
    CXq      = _interp1(d.alpha1, d.CXq,      ALPHA)
    dCXq_lef = _interp1(d.alpha2, d.dCXq_lef, alpha_lef)

    # ------------------------------------------------------------------
    # CZ — normal force coefficient (F16model.m lines 122–128)
    # ------------------------------------------------------------------
    CZ0 = _interp3(d.alpha1, d.beta, d.de1, d.CZ, ALPHA, BETA, DE)
    delta_CZ_lef = (
        _interp2(d.alpha2, d.beta, d.CZ_lef, alpha_lef, BETA)
        - _interp3(d.alpha1, d.beta, d.de1, d.CZ, ALPHA, BETA, 0.0)
    )
    CZq      = _interp1(d.alpha1, d.CZq,      ALPHA)
    dCZq_lef = _interp1(d.alpha2, d.dCZq_lef, alpha_lef)

    # ------------------------------------------------------------------
    # Cm — pitching moment coefficient (F16model.m lines 130–141)
    # ------------------------------------------------------------------
    Cm0 = _interp3(d.alpha1, d.beta, d.de1, d.Cm, ALPHA, BETA, DE)
    delta_Cm_lef = (
        _interp2(d.alpha2, d.beta, d.Cm_lef, alpha_lef, BETA)
        - _interp3(d.alpha1, d.beta, d.de1, d.Cm, ALPHA, BETA, 0.0)
    )
    Cmq      = _interp1(d.alpha1, d.Cmq,      ALPHA)
    dCmq_lef = _interp1(d.alpha2, d.dCmq_lef, alpha_lef)
    dCm      = _interp1(d.alpha1, d.dCm,      ALPHA)
    # F16model.m line 141: interp2(de3, alpha1, dCm_ds, DE, ALPHA)
    # axes: alpha1 rows, de3 cols → RegularGridInterpolator((alpha1, de3), data)
    dCm_ds   = _interp2(d.alpha1, d.de3, d.dCm_ds, ALPHA, DE)

    # ------------------------------------------------------------------
    # CY — side force coefficient (F16model.m lines 143–161)
    # ------------------------------------------------------------------
    CY0 = _interp2(d.alpha1, d.beta, d.CY, ALPHA, BETA)
    delta_CY_lef = (
        _interp2(d.alpha2, d.beta, d.CY_lef, alpha_lef, BETA) - CY0
    )
    delta_CY_da20 = (
        _interp2(d.alpha1, d.beta, d.CY_da20, ALPHA, BETA) - CY0
    )
    delta_CY_da20lef = (
        _interp2(d.alpha2, d.beta, d.CY_da20lef, alpha_lef, BETA)
        - _interp2(d.alpha2, d.beta, d.CY_lef,    alpha_lef, BETA)
        - delta_CY_da20
    )
    delta_CY_dr30 = (
        _interp2(d.alpha1, d.beta, d.CY_dr30, ALPHA, BETA) - CY0
    )
    CYr      = _interp1(d.alpha1, d.CYr,      ALPHA)
    dCYr_lef = _interp1(d.alpha2, d.dCYr_lef, alpha_lef)
    CYp      = _interp1(d.alpha1, d.CYp,      ALPHA)
    dCYp_lef = _interp1(d.alpha2, d.dCYp_lef, alpha_lef)

    # ------------------------------------------------------------------
    # Cn — yawing moment coefficient (F16model.m lines 163–187)
    # ------------------------------------------------------------------
    Cn0 = _interp3(d.alpha1, d.beta, d.de2, d.Cn, ALPHA, BETA, DE)
    delta_Cn_lef = (
        _interp2(d.alpha2, d.beta, d.Cn_lef, alpha_lef, BETA)
        - _interp3(d.alpha1, d.beta, d.de2, d.Cn, ALPHA, BETA, 0.0)
    )
    delta_Cn_da20 = (
        _interp2(d.alpha1, d.beta, d.Cn_da20, ALPHA, BETA)
        - _interp3(d.alpha1, d.beta, d.de2, d.Cn, ALPHA, BETA, 0.0)
    )
    delta_Cn_da20lef = (
        _interp2(d.alpha2, d.beta, d.Cn_da20lef, alpha_lef, BETA)
        - _interp2(d.alpha2, d.beta, d.Cn_lef,    alpha_lef, BETA)
        - delta_Cn_da20
    )
    delta_Cn_dr30 = (
        _interp2(d.alpha1, d.beta, d.Cn_dr30, ALPHA, BETA)
        - _interp3(d.alpha1, d.beta, d.de2, d.Cn, ALPHA, BETA, 0.0)
    )
    Cnr      = _interp1(d.alpha1, d.Cnr,      ALPHA)
    dCnbeta  = _interp1(d.alpha1, d.dCnbeta,  ALPHA)
    dCnr_lef = _interp1(d.alpha2, d.dCnr_lef, alpha_lef)
    Cnp      = _interp1(d.alpha1, d.Cnp,      ALPHA)
    dCnp_lef = _interp1(d.alpha2, d.dCnp_lef, alpha_lef)

    # ------------------------------------------------------------------
    # Cl — rolling moment coefficient (F16model.m lines 188–212)
    # ------------------------------------------------------------------
    Cl0 = _interp3(d.alpha1, d.beta, d.de2, d.Cl, ALPHA, BETA, DE)
    delta_Cl_lef = (
        _interp2(d.alpha2, d.beta, d.Cl_lef, alpha_lef, BETA)
        - _interp3(d.alpha1, d.beta, d.de2, d.Cl, ALPHA, BETA, 0.0)
    )
    delta_Cl_da20 = (
        _interp2(d.alpha1, d.beta, d.Cl_da20, ALPHA, BETA)
        - _interp3(d.alpha1, d.beta, d.de2, d.Cl, ALPHA, BETA, 0.0)
    )
    delta_Cl_da20lef = (
        _interp2(d.alpha2, d.beta, d.Cl_da20lef, alpha_lef, BETA)
        - _interp2(d.alpha2, d.beta, d.Cl_lef,    alpha_lef, BETA)
        - delta_Cl_da20
    )
    delta_Cl_dr30 = (
        _interp2(d.alpha1, d.beta, d.Cl_dr30, ALPHA, BETA)
        - _interp3(d.alpha1, d.beta, d.de2, d.Cl, ALPHA, BETA, 0.0)
    )
    Clr      = _interp1(d.alpha1, d.Clr,      ALPHA)
    dClbeta  = _interp1(d.alpha1, d.dClbeta,  ALPHA)
    dClr_lef = _interp1(d.alpha2, d.dClr_lef, alpha_lef)
    Clp      = _interp1(d.alpha1, d.Clp,      ALPHA)
    dClp_lef = _interp1(d.alpha2, d.dClp_lef, alpha_lef)

    # ------------------------------------------------------------------
    # Total coefficients (F16model.m lines 214–243)
    # ------------------------------------------------------------------

    # F16model.m lines 215–216
    CX_tot: float = (
        CX0
        + delta_CX_lef * dlef_norm
        + (p.cref / (2.0 * Vt)) * (CXq + dCXq_lef * dlef_norm) * q_body
    )

    # F16model.m lines 218–222
    CY_tot: float = (
        CY0
        + delta_CY_lef * dlef_norm
        + (delta_CY_da20 + delta_CY_da20lef * dlef_norm) * da_norm
        + delta_CY_dr30 * dr_norm
        + (p.bref / (2.0 * Vt)) * (CYr + dCYr_lef * dlef_norm) * r_body
        + (p.bref / (2.0 * Vt)) * (CYp + dCYp_lef * dlef_norm) * p_body
    )

    # F16model.m lines 224–225
    CZ_tot: float = (
        CZ0
        + delta_CZ_lef * dlef_norm
        + (p.cref / (2.0 * Vt)) * (CZq + dCZq_lef * dlef_norm) * q_body
    )

    # F16model.m lines 227–232
    Cl_tot: float = (
        Cl0
        + delta_Cl_lef * dlef_norm
        + (delta_Cl_da20 + delta_Cl_da20lef * dlef_norm) * da_norm
        + delta_Cl_dr30 * dr_norm
        + (p.bref / (2.0 * Vt)) * (Clr + dClr_lef * dlef_norm) * r_body
        + (p.bref / (2.0 * Vt)) * (Clp + dClp_lef * dlef_norm) * p_body
        + dClbeta * BETA
    )

    # F16model.m lines 234–236
    Cm_tot: float = (
        Cm0 * 1.0
        + CZ_tot * (p.xcgr - p.xcg)
        + delta_Cm_lef * dlef_norm
        + (p.cref / (2.0 * Vt)) * (Cmq + dCmq_lef * dlef_norm) * q_body
        + dCm
        + dCm_ds
    )

    # F16model.m lines 238–243
    Cn_tot: float = (
        Cn0
        + delta_Cn_lef * dlef_norm
        - CY_tot * (p.xcgr - p.xcg) * (p.cref / p.bref)
        + (delta_Cn_da20 + delta_Cn_da20lef * dlef_norm) * da_norm
        + (p.bref / (2.0 * Vt)) * (Cnr + dCnr_lef * dlef_norm) * r_body
        + (p.bref / (2.0 * Vt)) * (Cnp + dCnp_lef * dlef_norm) * p_body
        + delta_Cn_dr30 * dr_norm
        + dCnbeta * beta * RTD   # F16model.m line 243: beta * rtd (deg)
    )

    return CX_tot, CY_tot, CZ_tot, Cl_tot, Cm_tot, Cn_tot


# ---------------------------------------------------------------------------
# State saturation helper
# ---------------------------------------------------------------------------

def _saturate_state_action(
    alpha: float,
    beta: float,
    da: float,
    de: float,
    dr: float,
    dp: float,
) -> tuple[float, float, float, float, float, float]:
    """Apply physical saturation limits to angles and control deflections.

    Translated from F16model.m lines 31–73.

    All angles are in radians; throttle is dimensionless.
    """

    # F16model.m lines 31–35 — sideslip limits ±30°
    beta = np.clip(beta, -30.0 * np.pi / 180.0, 30.0 * np.pi / 180.0)

    # F16model.m lines 38–42 — elevator limits ±25°
    de = np.clip(de, -25.0 * np.pi / 180.0, 25.0 * np.pi / 180.0)

    # F16model.m lines 45–49 — AoA limits −20°…+90°
    alpha = np.clip(alpha, -20.0 * np.pi / 180.0, 90.0 * np.pi / 180.0)

    # F16model.m lines 53–57 — aileron limits ±21.5°
    da = np.clip(da, -21.5 * np.pi / 180.0, 21.5 * np.pi / 180.0)

    # F16model.m lines 60–64 — rudder limits ±30°
    dr = np.clip(dr, -30.0 * np.pi / 180.0, 30.0 * np.pi / 180.0)

    # F16model.m lines 69–73 — throttle limits [0, 1]
    dp = np.clip(dp, 0.0, 1.0)

    return float(alpha), float(beta), float(da), float(de), float(dr), float(dp)


# ---------------------------------------------------------------------------
# Leading-edge flap schedule
# ---------------------------------------------------------------------------

def _compute_lef(
    ALPHA: float,
    qbar: float,
    rho: float,
) -> tuple[float, float, float]:
    """Compute leading-edge flap schedule.

    Translated from F16model.m lines 91–103.

    Parameters
    ----------
    ALPHA : float
        Angle of attack [deg].
    qbar : float
        Dynamic pressure [Pa].
    rho : float
        Air density [kg/m³].

    Returns
    -------
    DLEF : float
        LEF deflection [deg].
    alpha_lef : float
        Alpha clamped to LEF table limit (≤ 45°) [deg].
    dlef_norm : float
        Normalised LEF term used in coefficient build-up.
    """
    # F16model.m line 91
    DLEF: float = 1.38 * ALPHA - 9.05 * qbar / (101325.0 * rho / 1.225) + 1.45

    # F16model.m lines 93–97 — LEF deflection saturation [0, 25°]
    DLEF = float(np.clip(DLEF, 0.0, 25.0 * np.pi / 180.0))

    # F16model.m lines 99–103 — LEF table alpha limit
    alpha_lef: float = min(ALPHA, 45.0)

    # F16model.m line 114 — normalised LEF term
    dlef_norm: float = 1.0 - DLEF / 25.0

    return DLEF, alpha_lef, dlef_norm


# ---------------------------------------------------------------------------
# Mock aerodynamic data (used when aero_data=None)
# ---------------------------------------------------------------------------

class _MockAeroData:
    """Minimal stand-in for F16AeroData used in tests (returns zeros).

    Every table look-up evaluates to 0.0, giving a trivially integrated
    trajectory suitable for checking state-vector bookkeeping.
    """

    # Grid axes — must match the real breakpoint counts exactly
    alpha1 = np.linspace(-20.0, 90.0, 20)   # 20 pts
    alpha2 = np.linspace(-20.0, 45.0, 14)   # 14 pts
    beta   = np.linspace(-30.0, 30.0, 19)   # 19 pts
    de1    = np.array([-25.0, -10.0,  0.0, 10.0, 25.0])       # 5 pts
    de2    = np.array([-25.0,  0.0,  25.0])                    # 3 pts
    de3    = np.array([-25.0, -10.0,  0.0, 10.0, 15.0, 20.0, 25.0])  # 7 pts

    # 1-D tables keyed by alpha2 (14 pts)
    _alpha2_1d = {
        "dCXq_lef", "dCZq_lef", "dCmq_lef",
        "dCYr_lef", "dCYp_lef",
        "dCnr_lef", "dCnp_lef",
        "dClr_lef", "dClp_lef",
    }
    # 1-D tables keyed by alpha1 (20 pts)
    _alpha1_1d = {
        "CXq", "CZq", "Cmq",
        "CYr", "CYp", "Cnr", "Cnp", "Clr", "Clp",
        "dCnbeta", "dClbeta", "dCm",
    }

    def __getattr__(self, name: str) -> NDArray:
        # 1-D alpha2 tables
        if name in self._alpha2_1d:
            return np.zeros(14)
        # 1-D alpha1 tables
        if name in self._alpha1_1d:
            return np.zeros(20)
        # 3-D tables (alpha1 × beta × de1)
        if name in ("CX", "CZ", "Cm"):
            return np.zeros((20, 19, 5))
        # 3-D tables (alpha1 × beta × de2)
        if name in ("Cn", "Cl"):
            return np.zeros((20, 19, 3))
        # 2-D table (alpha1 × de3) — deep-stall Cm correction
        if name == "dCm_ds":
            return np.zeros((20, 7))
        # 2-D tables keyed by alpha2 × beta
        if name in ("CX_lef", "CZ_lef", "Cm_lef",
                    "CY_lef", "CY_da20lef",
                    "Cn_lef", "Cn_da20lef",
                    "Cl_lef", "Cl_da20lef"):
            return np.zeros((14, 19))
        # 2-D tables keyed by alpha1 × beta
        return np.zeros((20, 19))


_MOCK_AERO = _MockAeroData()


# ---------------------------------------------------------------------------
# Main 6-DOF integration step
# ---------------------------------------------------------------------------

def f16_model(
    state: ArrayLike,
    action: ArrayLike,
    dt: float,
    params: Optional[F16Parameters] = None,
    aero_data: Optional[object] = None,
    engine_data: Optional[F16EngineData] = None,
) -> NDArray[np.float64]:
    """Advance the F-16 state by one Euler-integration step of duration *dt*.

    Translated from F16model.m (lines 1–339).

    Parameters
    ----------
    state : array_like, shape (14,)
        Current state vector (see module docstring).
    action : array_like, shape (4,)
        Control action vector [da, de, dr, dp].
    dt : float
        Integration time step [s].
    params : F16Parameters or None, optional
        Aircraft parameters.  Defaults to :data:`f16_parameters.DEFAULT_PARAMS`.
    aero_data : object or None, optional
        Aerodynamic lookup tables (must expose the same attributes as the
        object returned by :func:`f16_aerodata.get_f16_aerodata`).
        Pass ``None`` to use zero-valued mock tables for unit tests.
    engine_data : F16EngineData or None, optional
        Engine thrust tables.  Defaults to :data:`f16_enginedata.DEFAULT_ENGINE_DATA`.
        Pass ``None`` for mock engine (linear thrust).

    Returns
    -------
    NDArray[np.float64], shape (14,)
        Next state vector after one integration step.

    Examples
    --------
    >>> import numpy as np
    >>> state = np.zeros(14); state[3] = 200.0  # Vt = 200 m/s
    >>> state[6] = 1.0                           # unit quaternion
    >>> action = np.array([0.0, 0.0, 0.0, 0.5])
    >>> next_s = f16_model(state, action, 0.01, aero_data=None, engine_data=None)
    >>> next_s.shape
    (14,)
    """

    state  = np.asarray(state,  dtype=float)
    action = np.asarray(action, dtype=float)

    p_obj  = params if params is not None else DEFAULT_PARAMS
    d      = aero_data if aero_data is not None else _MOCK_AERO

    # ------------------------------------------------------------------
    # Unpack state  (F16model.m lines 8–25)
    # ------------------------------------------------------------------
    X_earth = state[0]   # F16model.m line 8
    Y_earth = state[1]   # F16model.m line 9
    Z_earth = state[2]   # F16model.m line 10
    Vt      = state[3]   # F16model.m line 11
    alpha   = state[4]   # F16model.m line 12
    beta    = state[5]   # F16model.m line 13
    q0      = state[6]   # F16model.m line 14
    q1      = state[7]   # F16model.m line 15
    q2      = state[8]   # F16model.m line 16
    q3      = state[9]   # F16model.m line 17
    p_body  = state[10]  # F16model.m line 18
    q_body  = state[11]  # F16model.m line 19 (pitch rate)
    r_body  = state[12]  # F16model.m line 20
    power   = state[13]  # F16model.m line 25

    # Unpack action  (F16model.m lines 26–29)
    da = action[0]   # aileron  [rad]
    de = action[1]   # elevator [rad]
    dr = action[2]   # rudder   [rad]
    dp = action[3]   # throttle [0, 1]

    # ------------------------------------------------------------------
    # Saturate state and action  (F16model.m lines 31–73)
    # ------------------------------------------------------------------
    alpha, beta, da, de, dr, dp = _saturate_state_action(
        alpha, beta, da, de, dr, dp
    )

    # ------------------------------------------------------------------
    # Atmosphere / engine  (F16model.m lines 76–79)
    # ------------------------------------------------------------------
    mach, qbar, rho, g = isa_atmos(-Z_earth, Vt)   # line 76
    pdot   = power_dot(dp, power)                   # line 78
    thrust = engine_power(mach, -Z_earth, power, engine_data)  # line 79

    # ------------------------------------------------------------------
    # Body-axis velocity components  (F16model.m lines 81–83)
    # ------------------------------------------------------------------
    u_body = Vt * np.cos(alpha) * np.cos(beta)   # line 81
    v_body = Vt * np.sin(beta)                    # line 82
    w_body = Vt * np.sin(alpha) * np.cos(beta)   # line 83

    # ------------------------------------------------------------------
    # Convert to degrees for table lookups  (F16model.m lines 86–114)
    # ------------------------------------------------------------------
    ALPHA = alpha * RTD   # line 86
    BETA  = beta  * RTD   # line 87
    DE    = de    * RTD   # line 88
    DA    = da    * RTD   # line 89
    DR    = dr    * RTD   # line 90

    # Elevator clamp for tables  (F16model.m lines 105–109)
    DE = float(np.clip(DE, -25.0, 25.0))

    # Leading-edge flap  (F16model.m lines 91–114)
    _DLEF, alpha_lef, dlef_norm = _compute_lef(ALPHA, qbar, rho)

    # Normalised control deflections  (F16model.m lines 112–113)
    da_norm: float = DA / 21.5    # line 112
    dr_norm: float = DR / 30.0   # line 113

    # ------------------------------------------------------------------
    # Aerodynamic coefficients  (F16model.m lines 116–251)
    # ------------------------------------------------------------------
    CX_tot, CY_tot, CZ_tot, Cl_tot, Cm_tot, Cn_tot = _compute_aero_coeffs(
        ALPHA, BETA, DE, DA, DR,
        da_norm, dr_norm, dlef_norm, alpha_lef,
        q_body, p_body, r_body, Vt, beta,
        p_obj, d,
    )

    # ------------------------------------------------------------------
    # Total forces and moments  (F16model.m lines 245–251)
    # ------------------------------------------------------------------
    Xbar = qbar * p_obj.Sref * CX_tot   # line 245 [N]
    Ybar = qbar * p_obj.Sref * CY_tot   # line 246
    Zbar = qbar * p_obj.Sref * CZ_tot   # line 247
    Lbar = Cl_tot * qbar * p_obj.Sref * p_obj.bref   # line 249 [N·m]
    Mbar = Cm_tot * qbar * p_obj.Sref * p_obj.cref   # line 250
    Nbar = Cn_tot * qbar * p_obj.Sref * p_obj.bref   # line 251

    # ------------------------------------------------------------------
    # Translational accelerations in body frame  (F16model.m lines 258–263)
    # ------------------------------------------------------------------
    u_body_dot = (
        r_body * v_body - q_body * w_body
        + (Xbar + thrust) / p_obj.mass
        + 2.0 * (q1 * q3 - q0 * q2) * g
    )   # line 258–259
    v_body_dot = (
        p_body * w_body - r_body * u_body
        + Ybar / p_obj.mass
        + 2.0 * (q2 * q3 + q0 * q1) * g
    )   # line 260–261
    w_body_dot = (
        q_body * u_body - p_body * v_body
        + Zbar / p_obj.mass
        + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * g
    )   # line 262–263

    # ------------------------------------------------------------------
    # Airspeed / AoA / sideslip derivatives  (F16model.m lines 265–270)
    # ------------------------------------------------------------------
    Vt_dot    = (
        u_body * u_body_dot + v_body * v_body_dot + w_body * w_body_dot
    ) / Vt   # line 265–266
    beta_dot  = (
        v_body_dot * Vt - v_body * Vt_dot
    ) / (Vt * Vt * np.cos(beta))   # line 267–268
    alpha_dot = (
        u_body * w_body_dot - w_body * u_body_dot
    ) / (u_body * u_body + w_body * w_body)   # line 269–270

    # ------------------------------------------------------------------
    # Quaternion kinematics  (F16model.m lines 272–283)
    # ------------------------------------------------------------------
    q0_dot = 0.5 * (-p_body * q1 - q_body * q2 - r_body * q3)   # line 272
    q1_dot = 0.5 * ( p_body * q0 + r_body * q2 - q_body * q3)   # line 273
    q2_dot = 0.5 * ( q_body * q0 - r_body * q1 + p_body * q3)   # line 274
    q3_dot = 0.5 * ( r_body * q0 + q_body * q1 - p_body * q2)   # line 275

    # Quaternion normalisation correction  (F16model.m lines 278–283)
    dq = q0 * q0_dot + q1 * q1_dot + q2 * q2_dot + q3 * q3_dot   # line 278
    q0_dot -= dq * q0   # line 280
    q1_dot -= dq * q1   # line 281
    q2_dot -= dq * q2   # line 282
    q3_dot -= dq * q3   # line 283

    # ------------------------------------------------------------------
    # Angular rate derivatives  (F16model.m lines 285–291)
    # ------------------------------------------------------------------
    p_body_dot = (
        (p_obj.C1 * r_body + p_obj.C2 * p_body) * q_body
        + p_obj.C3 * Lbar
        + p_obj.C4 * (Nbar + q_body * p_obj.heng)
    )   # line 285–286
    q_body_dot = (
        p_obj.C5 * p_body * r_body
        - p_obj.C6 * (p_body * p_body - r_body * r_body)
        + p_obj.C7 * (Mbar - p_obj.heng * r_body)
    )   # line 287–289 (note: sign on heng term per original)
    r_body_dot = (
        (p_obj.C8 * p_body - p_obj.C2 * r_body) * q_body
        + p_obj.C4 * Lbar
        + p_obj.C9 * (Nbar + q_body * p_obj.heng)
    )   # line 290–291

    # ------------------------------------------------------------------
    # Navigation equations (DCM from quaternion)  (F16model.m lines 293–301)
    # ------------------------------------------------------------------
    x_earth_dot = (
        (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * u_body
        + 2.0 * (q1 * q2 - q0 * q3) * v_body
        + 2.0 * (q1 * q3 + q0 * q2) * w_body
    )   # lines 293–295
    y_earth_dot = (
        2.0 * (q1 * q2 + q0 * q3) * u_body
        + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * v_body
        + 2.0 * (q2 * q3 - q0 * q1) * w_body
    )   # lines 296–298
    z_earth_dot = (
        2.0 * (q1 * q3 - q0 * q2) * u_body
        + 2.0 * (q2 * q3 + q0 * q1) * v_body
        + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * w_body
    )   # lines 299–301

    # ------------------------------------------------------------------
    # Euler integration  (F16model.m lines 303–316)
    # ------------------------------------------------------------------
    X_earth += x_earth_dot * dt   # line 303
    Y_earth += y_earth_dot * dt   # line 304
    Z_earth += z_earth_dot * dt   # line 305
    Vt      += Vt_dot      * dt   # line 306
    alpha   += alpha_dot   * dt   # line 307
    beta    += beta_dot    * dt   # line 308
    q0      += q0_dot      * dt   # line 309
    q1      += q1_dot      * dt   # line 310
    q2      += q2_dot      * dt   # line 311
    q3      += q3_dot      * dt   # line 312
    p_body  += p_body_dot  * dt   # line 313
    q_body  += q_body_dot  * dt   # line 314
    r_body  += r_body_dot  * dt   # line 315
    power   += pdot        * dt   # line 316

    # ------------------------------------------------------------------
    # Pack next state  (F16model.m lines 321–338)
    # ------------------------------------------------------------------
    next_state = np.empty(14, dtype=float)
    next_state[0]  = X_earth   # line 321
    next_state[1]  = Y_earth   # line 322
    next_state[2]  = Z_earth   # line 323
    next_state[3]  = Vt        # line 324
    next_state[4]  = alpha     # line 325
    next_state[5]  = beta      # line 326
    next_state[6]  = q0        # line 327
    next_state[7]  = q1        # line 328
    next_state[8]  = q2        # line 329
    next_state[9]  = q3        # line 330
    next_state[10] = p_body    # line 331
    next_state[11] = q_body    # line 332
    next_state[12] = r_body    # line 333
    next_state[13] = power     # line 338

    return next_state


# ---------------------------------------------------------------------------
# Trim cost function
# ---------------------------------------------------------------------------

# Trim weight vector (F16modeltrim.m lines 355–369)
_TRIM_WEIGHTS = np.array([
    2.0,   # Vt_dot     — line 356
    10.0,  # beta_dot   — line 357
    100.0, # alpha_dot  — line 358
    10.0,  # q0_dot     — line 359
    10.0,  # q1_dot     — line 360
    10.0,  # q2_dot     — line 361
    10.0,  # q3_dot     — line 362
    10.0,  # p_dot      — line 363
    10.0,  # q_dot      — line 364
    10.0,  # r_dot      — line 365
    0.0,   # x_dot      — line 366
    0.0,   # y_dot      — line 367
    5.0,   # z_dot      — line 368
    50.0,  # pow_dot    — line 369
])


def f16_trim_cost(
    UX: ArrayLike,
    state0: ArrayLike,
    dt: float,
    params: Optional[F16Parameters] = None,
    aero_data: Optional[object] = None,
    engine_data: Optional[F16EngineData] = None,
) -> float:
    """Weighted squared-norm cost function for F-16 trim optimisation.

    Translated from F16modeltrim.m (lines 1–371).

    Parameters
    ----------
    UX : array_like, shape (6,)
        Optimisation variables: [theta/alpha (rad), beta (rad), da (rad),
        de (rad), dr (rad), dp (throttle)].
        Note: ``UX[0]`` serves as both *theta* (pitch angle) and *alpha* (AoA)
        in the original code (F16modeltrim.m lines 17–19).
    state0 : array_like, shape (14,)
        Reference state (positions, velocity, angular rates).
    dt : float
        Integration time step [s].
    params : F16Parameters or None, optional
    aero_data : object or None, optional
    engine_data : F16EngineData or None, optional

    Returns
    -------
    float
        Weighted sum of squared state derivatives at the query point.
        A trim solution corresponds to cost ≈ 0.

    Notes
    -----
    The cost is evaluated over *one* integration step and uses the same
    state-propagation logic as :func:`f16_model`.  The quaternion is
    recomputed from Euler angles derived from the optimisation variables
    (F16modeltrim.m lines 25–28).
    """

    UX     = np.asarray(UX,     dtype=float)
    state0 = np.asarray(state0, dtype=float)

    p_obj = params if params is not None else DEFAULT_PARAMS

    # ------------------------------------------------------------------
    # Unpack reference state  (F16modeltrim.m lines 12–31)
    # ------------------------------------------------------------------
    X_earth = state0[0]
    Y_earth = state0[1]
    Z_earth = state0[2]
    Vt      = state0[3]
    phi     = state0[6]    # F16modeltrim.m line 16 (stored in slot 7 = index 6)
    psi     = state0[8]    # F16modeltrim.m line 18 (stored in slot 9 = index 8)
    p_body  = state0[9]    # F16modeltrim.m line 29 (slot 10 = index 9)
    q_body  = state0[10]   # line 30
    r_body  = state0[11]   # line 31

    # Optimisation variables  (F16modeltrim.m lines 17–40)
    theta = UX[0]   # pitch angle = AoA for wings-level trim
    alpha = UX[0]   # F16modeltrim.m lines 17, 19 — same variable
    beta  = UX[1]
    da    = UX[2]
    de    = UX[3]
    dr    = UX[4]
    dp    = UX[5]

    # Recompute quaternion from Euler angles  (F16modeltrim.m lines 25–28)
    q0 = (np.cos(psi / 2.0) * np.cos(theta / 2.0) * np.cos(phi / 2.0)
          + np.sin(psi / 2.0) * np.sin(theta / 2.0) * np.sin(phi / 2.0))
    q1 = (np.cos(psi / 2.0) * np.cos(theta / 2.0) * np.sin(phi / 2.0)
          - np.sin(psi / 2.0) * np.sin(theta / 2.0) * np.cos(phi / 2.0))
    q2 = (np.cos(psi / 2.0) * np.sin(theta / 2.0) * np.cos(phi / 2.0)
          + np.sin(psi / 2.0) * np.cos(theta / 2.0) * np.sin(phi / 2.0))
    q3 = (-np.cos(psi / 2.0) * np.sin(theta / 2.0) * np.sin(phi / 2.0)
          + np.sin(psi / 2.0) * np.cos(theta / 2.0) * np.cos(phi / 2.0))

    # Engine power from throttle  (F16modeltrim.m line 41)
    power = tgear(dp)

    # Assemble state for the step  (mirroring F16modeltrim.m implicit usage)
    trim_state  = np.array([X_earth, Y_earth, Z_earth, Vt,
                             alpha, beta,
                             q0, q1, q2, q3,
                             p_body, q_body, r_body,
                             power])
    trim_action = np.array([da, de, dr, dp])

    # ------------------------------------------------------------------
    # Saturate  (F16modeltrim.m lines 43–85 — same limits as F16model.m)
    # ------------------------------------------------------------------
    alpha, beta, da, de, dr, dp = _saturate_state_action(
        alpha, beta, da, de, dr, dp
    )

    # ------------------------------------------------------------------
    # Atmosphere / engine  (F16modeltrim.m lines 91–94)
    # ------------------------------------------------------------------
    mach, qbar, rho, g = isa_atmos(-Z_earth, Vt)
    pdot   = power_dot(dp, power)
    thrust = engine_power(mach, -Z_earth, power, engine_data)

    # ------------------------------------------------------------------
    # Body-axis velocities  (F16modeltrim.m lines 96–98)
    # ------------------------------------------------------------------
    u_body = Vt * np.cos(alpha) * np.cos(beta)
    v_body = Vt * np.sin(beta)
    w_body = Vt * np.sin(alpha) * np.cos(beta)

    # ------------------------------------------------------------------
    # Degrees / LEF / normalised controls  (F16modeltrim.m lines 101–129)
    # ------------------------------------------------------------------
    ALPHA = alpha * RTD
    BETA  = beta  * RTD
    DE    = float(np.clip(de * RTD, -25.0, 25.0))
    DA    = da    * RTD
    DR    = dr    * RTD
    _DLEF, alpha_lef, dlef_norm = _compute_lef(ALPHA, qbar, rho)
    da_norm = DA / 21.5
    dr_norm = DR / 30.0

    d = aero_data if aero_data is not None else _MOCK_AERO

    # ------------------------------------------------------------------
    # Coefficients / forces / moments  (F16modeltrim.m lines 131–266)
    # ------------------------------------------------------------------
    CX_tot, CY_tot, CZ_tot, Cl_tot, Cm_tot, Cn_tot = _compute_aero_coeffs(
        ALPHA, BETA, DE, DA, DR,
        da_norm, dr_norm, dlef_norm, alpha_lef,
        q_body, p_body, r_body, Vt, beta,
        p_obj, d,
    )

    Xbar = qbar * p_obj.Sref * CX_tot
    Ybar = qbar * p_obj.Sref * CY_tot
    Zbar = qbar * p_obj.Sref * CZ_tot
    Lbar = Cl_tot * qbar * p_obj.Sref * p_obj.bref
    Mbar = Cm_tot * qbar * p_obj.Sref * p_obj.cref
    Nbar = Cn_tot * qbar * p_obj.Sref * p_obj.bref

    # ------------------------------------------------------------------
    # Derivatives  (F16modeltrim.m lines 273–316)
    # ------------------------------------------------------------------
    u_body_dot = (
        r_body * v_body - q_body * w_body
        + (Xbar + thrust) / p_obj.mass
        + 2.0 * (q1 * q3 - q0 * q2) * g
    )
    v_body_dot = (
        p_body * w_body - r_body * u_body
        + Ybar / p_obj.mass
        + 2.0 * (q2 * q3 + q0 * q1) * g
    )
    w_body_dot = (
        q_body * u_body - p_body * v_body
        + Zbar / p_obj.mass
        + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * g
    )

    Vt_dot    = (u_body * u_body_dot + v_body * v_body_dot
                 + w_body * w_body_dot) / Vt
    beta_dot  = (v_body_dot * Vt - v_body * Vt_dot) / (Vt * Vt * np.cos(beta))
    alpha_dot = ((u_body * w_body_dot - w_body * u_body_dot)
                 / (u_body * u_body + w_body * w_body))

    q0_dot = 0.5 * (-p_body * q1 - q_body * q2 - r_body * q3)
    q1_dot = 0.5 * ( p_body * q0 + r_body * q2 - q_body * q3)
    q2_dot = 0.5 * ( q_body * q0 - r_body * q1 + p_body * q3)
    q3_dot = 0.5 * ( r_body * q0 + q_body * q1 - p_body * q2)

    dq = q0 * q0_dot + q1 * q1_dot + q2 * q2_dot + q3 * q3_dot
    q0_dot -= dq * q0
    q1_dot -= dq * q1
    q2_dot -= dq * q2
    q3_dot -= dq * q3

    p_body_dot = (
        (p_obj.C1 * r_body + p_obj.C2 * p_body) * q_body
        + p_obj.C3 * Lbar
        + p_obj.C4 * (Nbar + q_body * p_obj.heng)
    )
    q_body_dot = (
        p_obj.C5 * p_body * r_body
        - p_obj.C6 * (p_body * p_body - r_body * r_body)
        + p_obj.C7 * (Mbar - p_obj.heng * r_body)
    )
    r_body_dot = (
        (p_obj.C8 * p_body - p_obj.C2 * r_body) * q_body
        + p_obj.C4 * Lbar
        + p_obj.C9 * (Nbar + q_body * p_obj.heng)
    )

    x_earth_dot = (
        (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * u_body
        + 2.0 * (q1 * q2 - q0 * q3) * v_body
        + 2.0 * (q1 * q3 + q0 * q2) * w_body
    )
    y_earth_dot = (
        2.0 * (q1 * q2 + q0 * q3) * u_body
        + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * v_body
        + 2.0 * (q2 * q3 - q0 * q1) * w_body
    )
    z_earth_dot = (
        2.0 * (q1 * q3 - q0 * q2) * u_body
        + 2.0 * (q2 * q3 + q0 * q1) * v_body
        + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * w_body
    )

    # ------------------------------------------------------------------
    # Cost  (F16modeltrim.m lines 370–371)
    # ------------------------------------------------------------------
    Xdot = np.array([
        Vt_dot, beta_dot, alpha_dot,
        q0_dot, q1_dot, q2_dot, q3_dot,
        p_body_dot, q_body_dot, r_body_dot,
        x_earth_dot, y_earth_dot, z_earth_dot,
        pdot,
    ])
    cost: float = float(_TRIM_WEIGHTS @ (Xdot * Xdot))
    return cost
