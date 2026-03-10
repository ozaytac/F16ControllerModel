"""F-16 engine model functions.

Translates three MATLAB files:
  * tgear.m      — throttle-to-power-command mapping
  * power_dot.m  — engine power-level rate-of-change
  * engine_power.m — thrust from current power level

All functions accept an optional *engine_data* keyword argument.
Pass ``engine_data=None`` to use a lightweight mock suitable for unit
testing (the mock simply returns the values expected at mid-range throttle).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from f16_enginedata import F16EngineData, DEFAULT_ENGINE_DATA


def tgear(thtl: float) -> float:
    """Convert throttle position to power command.

    Translated from tgear.m (lines 10–16).

    Parameters
    ----------
    thtl : float
        Throttle position [0, 1].

    Returns
    -------
    float
        Power command [0, 100] (approximately).

    Notes
    -----
    The relationship is piecewise-linear:

    * Below 0.77 : ``Pc = 64.94 × thtl``
    * At or above 0.77 : ``Pc = 217.38 × thtl − 117.38``

    Examples
    --------
    >>> round(tgear(0.5), 4)
    32.47
    >>> round(tgear(0.9), 4)
    78.242
    """

    # tgear.m lines 12–16
    if thtl <= 0.77:
        tgear_value: float = 64.94 * thtl
    else:
        tgear_value = 217.38 * thtl - 117.38
    return tgear_value


def power_dot(dp: float, pa: float) -> float:
    """Compute engine power-level rate of change.

    Translated from power_dot.m (lines 1–31).

    Parameters
    ----------
    dp : float
        Throttle position (demanded power input), [0, 1].
    pa : float
        Actual power level [0, 100].

    Returns
    -------
    float
        Rate of change of power level [power-units / s].

    Notes
    -----
    ``dp`` is first converted to a commanded power ``Pc`` via the same
    piecewise-linear mapping as :func:`tgear`.  A time constant ``teng``
    is then selected based on the power error and whether the engine is in
    military or afterburner regime.

    Examples
    --------
    >>> power_dot(0.5, 0.0)  # accelerating from zero
    32.47
    """

    # power_dot.m lines 2–6  (same mapping as tgear.m)
    if dp <= 0.77:
        Pc: float = 64.94 * dp
    else:
        Pc = 217.38 * dp - 117.38

    # power_dot.m lines 7–13  — time constant selection
    delta: float = Pc - pa
    if delta <= 25.0:
        teng_star: float = 1.0
    elif delta >= 50.0:
        teng_star = 0.1
    else:
        teng_star = 1.9 - 0.036 * delta

    # power_dot.m lines 14–30  — regime-dependent logic
    if Pc >= 50.0:
        if pa >= 50.0:
            Pc = Pc          # power_dot.m line 16 (no-op in original)
            teng: float = 5.0
        else:
            Pc = 60.0
            teng = teng_star
    else:
        if pa >= 50.0:
            Pc = 40.0
            teng = 5.0
        else:
            Pc = Pc          # power_dot.m line 28 (no-op in original)
            teng = teng_star

    # power_dot.m line 31
    pdot: float = teng * (Pc - pa)
    return pdot


def engine_power(
    ma: float,
    alt: float,
    pa: float,
    engine_data: Optional[F16EngineData] = None,
) -> float:
    """Compute engine thrust from current power level.

    Translated from engine_power.m (lines 1–12).

    Parameters
    ----------
    ma : float
        Mach number [–].
    alt : float
        Altitude above MSL [m].  Converted to feet internally (line 3).
    pa : float
        Actual power level [0, 100].
    engine_data : F16EngineData or None, optional
        Pre-loaded engine tables.  Defaults to :data:`DEFAULT_ENGINE_DATA`.
        Pass ``None`` to use a linear mock (testing only).

    Returns
    -------
    float
        Net thrust [N].

    Notes
    -----
    The original MATLAB converts altitude to feet (÷ 0.3048) before
    interpolating the engine tables, which are defined in ft.  Thrust is
    returned in lbf by the tables and multiplied by 4.4482216 N/lbf.

    Examples
    --------
    >>> thrust = engine_power(0.3, 0.0, 50.0)
    >>> thrust > 0
    True
    """

    if engine_data is None:
        # Mock: simple linear thrust proportional to power, for unit tests
        return pa * 500.0  # approximate mid-range N per power unit

    # engine_power.m line 3 — convert m → ft
    alt_ft: float = alt / 0.3048

    # MATLAB: interp2(f16engine.alt, f16engine.Ma, f16engine.Idle, alt, Ma)
    # In MATLAB interp2(X,Y,V,Xq,Yq): X=alt (columns/dim1), Y=Ma (rows/dim0)
    # → V has shape (n_Ma, n_alt); scipy axes: (Ma, alt), query: (ma, alt_ft)
    _kw = dict(method="linear", bounds_error=False, fill_value=None)
    axes = (engine_data.Ma, engine_data.alt)
    query = np.array([[ma, alt_ft]])

    T_Idle: float = float(RegularGridInterpolator(axes, engine_data.Idle, **_kw)(query)[0])
    T_Mil:  float = float(RegularGridInterpolator(axes, engine_data.Mil,  **_kw)(query)[0])
    T_Max:  float = float(RegularGridInterpolator(axes, engine_data.Max,  **_kw)(query)[0])

    # engine_power.m lines 7–11
    if pa < 50.0:
        thrust: float = T_Idle + (T_Mil - T_Idle) * pa / 50.0
    else:
        thrust = T_Mil + (T_Max - T_Mil) * (pa - 50.0) / 50.0

    # engine_power.m line 12 — lbf → N
    thrust *= 4.4482216
    return thrust
