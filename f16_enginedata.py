"""F-16 engine thrust data tables.

Translated from F16enginedata.m (original MATLAB).

The tables contain thrust [lbf] indexed by altitude (ft) and Mach number.
``get_f16_enginedata()`` converts them to SI on load.

References
----------
NASA Technical Paper 1538, Appendix A.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class F16EngineData:
    """Engine thrust lookup tables.

    Attributes
    ----------
    Ma : np.ndarray
        Mach number breakpoints, shape (6,). F16enginedata.m line 3.
    alt : np.ndarray
        Altitude breakpoints [ft], shape (6,). F16enginedata.m line 4.
    Idle : np.ndarray
        Idle thrust table [lbf], shape (6, 6) indexed [alt_idx, Ma_idx].
        F16enginedata.m lines 5–10.
    Mil : np.ndarray
        Military (max dry) thrust table [lbf], shape (6, 6).
        F16enginedata.m lines 11–16.
    Max : np.ndarray
        Maximum (afterburner) thrust table [lbf], shape (6, 6).
        F16enginedata.m lines 17–22.
    """

    Ma: np.ndarray
    alt: np.ndarray
    Idle: np.ndarray
    Mil: np.ndarray
    Max: np.ndarray


def get_f16_enginedata() -> F16EngineData:
    """Return F-16 engine data tables (thrust in lbf, altitude in ft).

    Returns
    -------
    F16EngineData
        Dataclass holding the three thrust tables and their grid axes.

    Notes
    -----
    Thrust values are kept in lbf to match the original MATLAB source.
    ``engine_power()`` in ``f16_engine.py`` applies the lbf → N conversion
    (×4.4482216) before returning thrust. Altitude is in feet as required
    by the engine interpolation (engine_power.m line 3 converts m → ft).
    """

    # F16enginedata.m line 3
    Ma = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # F16enginedata.m line 4  (altitude in ft)
    alt = np.array([0.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0])

    # F16enginedata.m lines 5–10
    # Rows → altitude index, columns → Mach index
    Idle = np.array([
        [1060.0,  670.0,  880.0, 1140.0, 1500.0, 1860.0],
        [ 635.0,  425.0,  690.0, 1010.0, 1330.0, 1700.0],
        [  60.0,   25.0,  345.0,  755.0, 1130.0, 1525.0],
        [-1020.0, -710.0, -300.0,  350.0,  910.0, 1360.0],
        [-2700.0, -1900.0, -1300.0, -247.0,  600.0, 1100.0],
        [-3600.0, -1400.0, -595.0, -342.0, -200.0,  700.0],
    ])

    # F16enginedata.m lines 11–16
    Mil = np.array([
        [12680.0,  9150.0,  6200.0,  3950.0,  2450.0,  1400.0],
        [12680.0,  9150.0,  6313.0,  4040.0,  2470.0,  1400.0],
        [12610.0,  9312.0,  6610.0,  4290.0,  2600.0,  1560.0],
        [12640.0,  9839.0,  7090.0,  4660.0,  2840.0,  1660.0],
        [12390.0, 10176.0,  7750.0,  5320.0,  3250.0,  1930.0],
        [11680.0,  9848.0,  8050.0,  6100.0,  3800.0,  2310.0],
    ])

    # F16enginedata.m lines 17–22
    Max = np.array([
        [20000.0, 15000.0, 10800.0,  7000.0,  4000.0,  2500.0],
        [21420.0, 15700.0, 11225.0,  7323.0,  4435.0,  2600.0],
        [22700.0, 16860.0, 12250.0,  8154.0,  5000.0,  2835.0],
        [24240.0, 18910.0, 13760.0,  9285.0,  5700.0,  3215.0],
        [26070.0, 21075.0, 15975.0, 11115.0,  6860.0,  3950.0],
        [28886.0, 23319.0, 18300.0, 13484.0,  8642.0,  5057.0],
    ])

    return F16EngineData(Ma=Ma, alt=alt, Idle=Idle, Mil=Mil, Max=Max)


DEFAULT_ENGINE_DATA = get_f16_enginedata()
