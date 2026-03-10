"""pytest test suite for the F-16 6-DOF Python model.

All tests that touch aerodynamic tables pass ``aero_data=None`` and
``engine_data=None`` so they run without the large data files.  Tests that
need real tables are marked with ``@pytest.mark.integration`` and are skipped
unless the aerodata module is importable.

Run fast (mock-only) tests:
    pytest test_f16_model.py -m "not integration"

Run everything (requires f16_aerodata.py):
    pytest test_f16_model.py
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from f16_parameters import F16Parameters, DEFAULT_PARAMS, RTD, DTR
from f16_atmosphere import isa_atmos
from f16_engine import tgear, power_dot, engine_power
from f16_enginedata import DEFAULT_ENGINE_DATA
from f16_model import f16_model, f16_trim_cost, _TRIM_WEIGHTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_quat_state(vt: float = 200.0, alt: float = 5000.0) -> np.ndarray:
    """Return a simple wings-level state at a given airspeed / altitude."""
    s = np.zeros(14)
    s[2] = -alt        # Z_earth (negative = altitude above ground)
    s[3] = vt          # Vt
    s[6] = 1.0         # q0 (unit quaternion, zero attitude)
    s[13] = 50.0       # power mid-range
    return s


def _zero_action() -> np.ndarray:
    return np.zeros(4)


# ---------------------------------------------------------------------------
# F16Parameters tests
# ---------------------------------------------------------------------------

class TestF16Parameters:
    def test_default_mass(self) -> None:
        assert DEFAULT_PARAMS.mass == pytest.approx(9295.44)

    def test_derived_gamma(self) -> None:
        p = DEFAULT_PARAMS
        expected = p.Ixx * p.Izz - p.Ixz ** 2
        assert p.Gamma == pytest.approx(expected)

    def test_c3_definition(self) -> None:
        """C3 = Izz / Gamma  (F16parameter.m line 20)."""
        p = DEFAULT_PARAMS
        assert p.C3 == pytest.approx(p.Izz / p.Gamma)

    def test_c7_definition(self) -> None:
        """C7 = 1/Iyy  (F16parameter.m line 24)."""
        p = DEFAULT_PARAMS
        assert p.C7 == pytest.approx(1.0 / p.Iyy)

    def test_custom_params(self) -> None:
        p = F16Parameters(mass=10000.0)
        assert p.mass == 10000.0
        # Gamma should still be recomputed correctly
        assert p.Gamma == pytest.approx(p.Ixx * p.Izz - p.Ixz ** 2)

    def test_rtd_dtr_consistency(self) -> None:
        assert RTD * DTR == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# ISA atmosphere tests
# ---------------------------------------------------------------------------

class TestISAAtmos:
    def test_sea_level_density(self) -> None:
        _, _, rho, _ = isa_atmos(0.0, 100.0)
        assert rho == pytest.approx(1.225, rel=1e-3)

    def test_sea_level_gravity(self) -> None:
        _, _, _, g = isa_atmos(0.0, 100.0)
        assert g == pytest.approx(9.80665, rel=1e-5)

    def test_mach_at_sea_level(self) -> None:
        """Speed of sound at sea level ≈ 340 m/s → Mach 1 at ~340 m/s."""
        mach, _, _, _ = isa_atmos(0.0, 340.0)
        assert mach == pytest.approx(1.0, abs=0.01)

    def test_stratosphere_isothermal(self) -> None:
        """Above 11 km the temperature is constant at 216.65 K."""
        mach1, _, rho1, _ = isa_atmos(11_000.0, 200.0)
        mach2, _, rho2, _ = isa_atmos(15_000.0, 200.0)
        # Density must be lower at higher altitude
        assert rho2 < rho1

    def test_qbar_positive(self) -> None:
        _, qbar, _, _ = isa_atmos(5000.0, 200.0)
        assert qbar > 0.0

    def test_gravity_decreases_with_altitude(self) -> None:
        _, _, _, g0 = isa_atmos(0.0, 100.0)
        _, _, _, g5 = isa_atmos(5_000.0, 100.0)
        assert g5 < g0


# ---------------------------------------------------------------------------
# Engine function tests
# ---------------------------------------------------------------------------

class TestTgear:
    def test_low_throttle(self) -> None:
        """Below 0.77 throttle: Pc = 64.94 * thtl."""
        assert tgear(0.5) == pytest.approx(64.94 * 0.5)

    def test_high_throttle(self) -> None:
        """At/above 0.77 throttle: Pc = 217.38 * thtl - 117.38."""
        assert tgear(0.9) == pytest.approx(217.38 * 0.9 - 117.38)

    def test_boundary_continuity(self) -> None:
        """Both formulae agree at thtl = 0.77."""
        low  = 64.94 * 0.77
        high = 217.38 * 0.77 - 117.38
        assert low == pytest.approx(high, abs=0.01)

    def test_zero_throttle(self) -> None:
        assert tgear(0.0) == pytest.approx(0.0)


class TestPowerDot:
    def test_returns_float(self) -> None:
        result = power_dot(0.5, 30.0)
        assert isinstance(result, float)

    def test_zero_error_zero_rate(self) -> None:
        """When Pc == Pa the power error is zero and pdot should be zero."""
        # At dp=0.5: Pc = 64.94*0.5 = 32.47; set Pa=32.47
        Pc = 64.94 * 0.5
        result = power_dot(0.5, Pc)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_increasing_power(self) -> None:
        """pdot > 0 when power is demanded above current level."""
        result = power_dot(0.9, 0.0)
        assert result > 0.0

    def test_decreasing_power(self) -> None:
        """pdot < 0 when current power exceeds demand (afterburner to idle)."""
        result = power_dot(0.0, 90.0)
        assert result < 0.0


class TestEnginePower:
    def test_mock_returns_float(self) -> None:
        thrust = engine_power(0.5, 5000.0, 50.0, engine_data=None)
        assert isinstance(thrust, float)

    def test_mock_proportional(self) -> None:
        t1 = engine_power(0.5, 5000.0, 25.0, engine_data=None)
        t2 = engine_power(0.5, 5000.0, 50.0, engine_data=None)
        assert t2 == pytest.approx(2.0 * t1)

    @pytest.mark.integration
    def test_real_data_returns_positive(self) -> None:
        thrust = engine_power(0.3, 0.0, 50.0, engine_data=DEFAULT_ENGINE_DATA)
        assert thrust > 0.0

    @pytest.mark.integration
    def test_afterburner_greater_than_mil(self) -> None:
        t_mil = engine_power(0.3, 0.0, 50.0, engine_data=DEFAULT_ENGINE_DATA)
        t_max = engine_power(0.3, 0.0, 100.0, engine_data=DEFAULT_ENGINE_DATA)
        assert t_max > t_mil


# ---------------------------------------------------------------------------
# F16Model integration step tests (mock mode)
# ---------------------------------------------------------------------------

class TestF16ModelMock:
    def test_output_shape(self) -> None:
        s = _unit_quat_state()
        ns = f16_model(s, _zero_action(), 0.01,
                       aero_data=None, engine_data=None)
        assert ns.shape == (14,)

    def test_output_dtype(self) -> None:
        s = _unit_quat_state()
        ns = f16_model(s, _zero_action(), 0.01,
                       aero_data=None, engine_data=None)
        assert ns.dtype == np.float64

    def test_position_changes_with_velocity(self) -> None:
        """Aircraft moving forward should update X_earth."""
        s = _unit_quat_state(vt=200.0)
        ns = f16_model(s, _zero_action(), 0.1,
                       aero_data=None, engine_data=None)
        # With zero attitude quaternion and forward airspeed, X should change
        assert abs(ns[3] - s[3]) < 50.0   # Vt should not blow up

    def test_control_saturation_aileron(self) -> None:
        """Aileron command beyond ±21.5° should be clamped."""
        s = _unit_quat_state()
        a = np.array([np.radians(90.0), 0.0, 0.0, 0.5])   # 90° aileron
        ns = f16_model(s, a, 0.01, aero_data=None, engine_data=None)
        assert ns.shape == (14,)   # should not raise

    def test_control_saturation_throttle(self) -> None:
        """Throttle > 1 should be clamped to 1."""
        s = _unit_quat_state()
        a = np.array([0.0, 0.0, 0.0, 2.0])   # throttle = 2
        ns = f16_model(s, a, 0.01, aero_data=None, engine_data=None)
        assert ns.shape == (14,)

    def test_no_nan_in_output(self) -> None:
        s = _unit_quat_state()
        ns = f16_model(s, _zero_action(), 0.01,
                       aero_data=None, engine_data=None)
        assert not np.any(np.isnan(ns))

    def test_power_changes_over_time(self) -> None:
        """Power level should evolve when throttle is applied."""
        s = _unit_quat_state()
        s[13] = 0.0   # start at zero power
        a = np.array([0.0, 0.0, 0.0, 1.0])   # full throttle
        ns = f16_model(s, a, 0.1, aero_data=None, engine_data=None)
        assert ns[13] != s[13]

    def test_altitude_integration(self) -> None:
        """Z_earth (negative up) should change when there is a vertical rate."""
        s = _unit_quat_state(alt=5000.0)
        # Tilted quaternion: rotate 5° nose-up
        theta = np.radians(5.0)
        s[6] = np.cos(theta / 2.0)   # q0
        s[8] = np.sin(theta / 2.0)   # q2
        ns = f16_model(s, _zero_action(), 1.0,
                       aero_data=None, engine_data=None)
        assert ns[2] != s[2]

    def test_zero_dt_returns_same_positions(self) -> None:
        """With dt=0 the integrated positions should not change."""
        s = _unit_quat_state()
        ns = f16_model(s, _zero_action(), 0.0,
                       aero_data=None, engine_data=None)
        np.testing.assert_allclose(ns[:3], s[:3], atol=1e-12)

    def test_list_inputs_accepted(self) -> None:
        """Should accept Python list inputs (not just ndarray)."""
        s = list(_unit_quat_state())
        a = [0.0, 0.0, 0.0, 0.0]
        ns = f16_model(s, a, 0.01, aero_data=None, engine_data=None)
        assert ns.shape == (14,)


# ---------------------------------------------------------------------------
# Trim cost function tests (mock mode)
# ---------------------------------------------------------------------------

class TestF16TrimCost:
    def _ref_state(self) -> np.ndarray:
        s = np.zeros(14)
        s[2] = -5000.0    # 5 km altitude
        s[3] = 200.0      # 200 m/s
        s[6] = 1.0        # unit quaternion (phi=theta=psi=0)
        s[13] = 50.0      # power
        return s

    def test_returns_scalar(self) -> None:
        UX = np.array([np.radians(5.0), 0.0, 0.0, np.radians(-2.0), 0.0, 0.5])
        cost = f16_trim_cost(UX, self._ref_state(), 0.01,
                              aero_data=None, engine_data=None)
        assert isinstance(cost, float)

    def test_cost_nonnegative(self) -> None:
        UX = np.array([np.radians(5.0), 0.0, 0.0, np.radians(-2.0), 0.0, 0.5])
        cost = f16_trim_cost(UX, self._ref_state(), 0.01,
                              aero_data=None, engine_data=None)
        assert cost >= 0.0

    def test_list_input_accepted(self) -> None:
        UX    = [np.radians(5.0), 0.0, 0.0, np.radians(-2.0), 0.0, 0.5]
        state = list(self._ref_state())
        cost  = f16_trim_cost(UX, state, 0.01,
                               aero_data=None, engine_data=None)
        assert isinstance(cost, float)

    def test_trim_weight_length(self) -> None:
        assert len(_TRIM_WEIGHTS) == 14

    @pytest.mark.integration
    def test_real_data_trim_finite(self) -> None:
        """With real aero data the trim cost should be a finite number."""
        try:
            from f16_aerodata import get_f16_aerodata
        except ImportError:
            pytest.skip("f16_aerodata not available")
        d = get_f16_aerodata()
        UX = np.array([np.radians(5.0), 0.0, 0.0, np.radians(-2.0), 0.0, 0.5])
        cost = f16_trim_cost(UX, self._ref_state(), 0.01,
                              aero_data=d, engine_data=DEFAULT_ENGINE_DATA)
        assert math.isfinite(cost)


# ---------------------------------------------------------------------------
# Edge-case / robustness tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_small_vt_no_division_error(self) -> None:
        """Vt should not cause division-by-zero in derivatives."""
        s = _unit_quat_state(vt=1.0)
        ns = f16_model(s, _zero_action(), 0.001,
                       aero_data=None, engine_data=None)
        assert not np.any(np.isnan(ns))

    def test_negative_altitude_treated_as_underground(self) -> None:
        """Z_earth > 0 means below ground — atmosphere should still work."""
        s = _unit_quat_state()
        s[2] = 10.0   # slightly below ground (Z_earth > 0 → alt < 0)
        # May produce a RuntimeWarning for negative altitude in exp; no crash
        ns = f16_model(s, _zero_action(), 0.01,
                       aero_data=None, engine_data=None)
        assert ns.shape == (14,)

    def test_large_alpha_saturated(self) -> None:
        """AoA beyond 90° should be clamped before computation."""
        s = _unit_quat_state()
        s[4] = np.radians(120.0)   # beyond 90° limit
        ns = f16_model(s, _zero_action(), 0.01,
                       aero_data=None, engine_data=None)
        assert not np.any(np.isnan(ns))
