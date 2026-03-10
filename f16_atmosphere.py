"""International Standard Atmosphere (ISA) model.

Translated from ISA_atmos.m.

Supports a simple two-layer model: troposphere (alt < 11 000 m) with a
standard lapse rate and stratosphere base (alt ≥ 11 000 m) with an
isothermal layer at 216.65 K.
"""

from __future__ import annotations

import numpy as np


def isa_atmos(alt: float, vt: float) -> tuple[float, float, float, float]:
    """Compute ISA atmospheric properties at a given altitude and airspeed.

    Translated from ISA_atmos.m (lines 1–16).

    Parameters
    ----------
    alt : float
        Geometric altitude above mean sea level [m].  Must be ≥ 0.
    vt : float
        True airspeed [m/s].

    Returns
    -------
    mach : float
        Mach number [–].
    qbar : float
        Dynamic pressure [Pa].
    rho : float
        Air density [kg/m³].
    grav : float
        Local gravitational acceleration [m/s²].

    Notes
    -----
    The gravity model uses an inverse-square relationship with Earth's mean
    radius (Re = 6 371 000 m).  The density model is an exponential
    approximation derived from hydrostatic equilibrium.

    Examples
    --------
    >>> mach, qbar, rho, g = isa_atmos(0.0, 300.0)
    >>> round(rho, 3)
    1.225
    """

    # ISA_atmos.m line 2
    rho0: float = 1.225          # sea-level density [kg/m³]
    # ISA_atmos.m line 3
    Re: float = 6_371_000.0     # Earth mean radius [m]
    # ISA_atmos.m line 4
    R: float = 287.05           # specific gas constant for dry air [J/(kg·K)]
    # ISA_atmos.m line 5
    T0: float = 288.15          # sea-level standard temperature [K]
    # ISA_atmos.m line 6
    g0: float = 9.80665         # standard gravity [m/s²]
    # ISA_atmos.m line 7
    gamma: float = 1.4          # ratio of specific heats

    # ISA_atmos.m lines 8–12
    if alt >= 11_000.0:
        temp: float = 216.65    # isothermal stratosphere base [K]
    else:
        temp = T0 - 0.0065 * alt   # tropospheric temperature [K]

    # ISA_atmos.m line 13 — exponential density approximation
    rho: float = rho0 * np.exp((-g0 / (R * temp)) * alt)

    # ISA_atmos.m line 14 — Mach number
    mach: float = vt / np.sqrt(gamma * R * temp)

    # ISA_atmos.m line 15 — dynamic pressure [Pa]
    qbar: float = 0.5 * rho * vt * vt

    # ISA_atmos.m line 16 — local gravity (inverse-square law)
    grav: float = g0 * (Re * Re / ((Re + alt) * (Re + alt)))

    return mach, qbar, rho, grav
