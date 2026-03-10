"""F-16 aircraft physical parameters.

Translated from F16parameter.m (original MATLAB source by Nguyen et al.,
NASA Technical Paper 1538, 1979).

All units are SI (kg, m, N, N·m) unless otherwise noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# F16parameter.m line 14
RTD: float = 57.29577951       # rad-to-deg conversion factor
DTR: float = 0.017453293       # deg-to-rad conversion factor
PI: float = 3.141592654        # π


@dataclass
class F16Parameters:
    """Aggregated F-16 mass, geometry, and inertia parameters.

    Parameters
    ----------
    mass : float
        Aircraft mass [kg]. F16parameter.m line 1.
    Ixx : float
        Moment of inertia about x (roll) axis [kg·m²]. Line 2.
    Iyy : float
        Moment of inertia about y (pitch) axis [kg·m²]. Line 3.
    Izz : float
        Moment of inertia about z (yaw) axis [kg·m²]. Line 4.
    Ixz : float
        Product of inertia [kg·m²]. Line 5.
    Sref : float
        Wing reference area [m²]. Line 6.
    bref : float
        Wing span (reference length for lateral/directional) [m]. Line 7.
    cref : float
        Mean aerodynamic chord (reference length for pitch) [m]. Line 8.
    xcg : float
        Centre-of-gravity location as fraction of MAC. Line 9.
    xcgr : float
        Reference CG location as fraction of MAC. Line 10.
    heng : float
        Engine angular momentum (gyroscopic term) [kg·m²/s]. Line 11.
    Gamma : float
        Inertia coupling constant Ixx·Izz - Ixz². Derived. Line 17.
    C1–C9 : float
        Pre-computed inertia coefficients. Lines 18–25.
    """

    # F16parameter.m line 1
    mass: float = 9295.44
    # F16parameter.m line 2
    Ixx: float = 12874.8
    # F16parameter.m line 3
    Iyy: float = 75673.6
    # F16parameter.m line 4
    Izz: float = 85552.1
    # F16parameter.m line 5
    Ixz: float = 1331.4
    # F16parameter.m line 6
    Sref: float = 27.87
    # F16parameter.m line 7
    bref: float = 9.144
    # F16parameter.m line 8
    cref: float = 3.45
    # F16parameter.m line 9
    xcg: float = 0.3
    # F16parameter.m line 10
    xcgr: float = 0.35
    # F16parameter.m line 11
    heng: float = 216.9

    # Derived inertia constants — initialised post-init (F16parameter.m lines 17–25)
    Gamma: float = field(init=False)
    C1: float = field(init=False)
    C2: float = field(init=False)
    C3: float = field(init=False)
    C4: float = field(init=False)
    C5: float = field(init=False)
    C6: float = field(init=False)
    C7: float = field(init=False)
    C8: float = field(init=False)
    C9: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived inertia coefficients (F16parameter.m lines 17–25)."""
        # F16parameter.m line 17
        self.Gamma = self.Ixx * self.Izz - self.Ixz ** 2
        # F16parameter.m line 18
        self.C1 = ((self.Iyy - self.Izz) * self.Izz - self.Ixz ** 2) / self.Gamma
        # F16parameter.m line 19
        self.C2 = ((self.Ixx - self.Iyy + self.Izz) * self.Ixz) / self.Gamma
        # F16parameter.m line 20
        self.C3 = self.Izz / self.Gamma
        # F16parameter.m line 21
        self.C4 = self.Ixz / self.Gamma
        # F16parameter.m line 22
        self.C5 = (self.Izz - self.Ixx) / self.Iyy
        # F16parameter.m line 23
        self.C6 = self.Ixz / self.Iyy
        # F16parameter.m line 24
        self.C7 = 1.0 / self.Iyy
        # F16parameter.m line 25
        self.C8 = (self.Ixx * (self.Ixx - self.Iyy) + self.Ixz ** 2) / self.Gamma
        # F16parameter.m line 26 (implied)
        self.C9 = self.Ixx / self.Gamma


# Module-level singleton (equivalent to running F16parameter.m as a script)
DEFAULT_PARAMS = F16Parameters()
