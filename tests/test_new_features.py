"""Tests for the three new features added to ALPSS:

1. Shock stress  (src/alpss/analysis/shock_stress.py)
2. Spall DNS qualifier  (additions to src/alpss/analysis/spall.py)
3. HEL RDP + Linear Hybrid  (additions to src/alpss/analysis/hel.py)
"""

import pytest
import numpy as np

# ============================================================
# 1. Shock stress
# ============================================================

from alpss.analysis.shock_stress import (
    shock_stress_acoustic,
    shock_stress_hugoniot,
    shock_stress_hugoniot_uncertainty,
    get_hugoniot_S,
    calculate_shock_stress,
)


class TestGetHugonotS:
    def test_copper(self):
        assert get_hugoniot_S("Cu") == pytest.approx(1.49)

    def test_copper_aliases(self):
        for alias in ("cu", "Cu", "copper", "COPPER"):
            assert get_hugoniot_S(alias) == pytest.approx(1.49)

    def test_aluminum(self):
        assert get_hugoniot_S("Al") == pytest.approx(1.34)

    def test_unknown_material_returns_default(self):
        assert get_hugoniot_S("unobtainium") == pytest.approx(1.49)

    def test_custom_default(self):
        assert get_hugoniot_S("mystery", default=1.0) == pytest.approx(1.0)


class TestShockStressAcoustic:
    def test_basic(self):
        # σ = 0.5 * 8960 * 3950 * 300  ≈ 5.3076e9 Pa
        sigma = shock_stress_acoustic(density=8960, C0=3950, peak_velocity=300.0)
        assert sigma == pytest.approx(0.5 * 8960 * 3950 * 300)

    def test_zero_velocity(self):
        assert shock_stress_acoustic(8960, 3950, 0.0) == pytest.approx(0.0)


class TestShockStressHugoniot:
    def test_copper_300ms(self):
        rho, C0, S, v = 8960, 3950, 1.49, 300.0
        u_p = v / 2
        U_s = C0 + S * u_p
        expected = rho * U_s * u_p
        assert shock_stress_hugoniot(rho, C0, v, S) == pytest.approx(expected)

    def test_larger_than_acoustic_for_positive_S(self):
        # EOS stress > acoustic for any positive S and v > 0
        rho, C0, S, v = 8960, 3950, 1.49, 200.0
        assert shock_stress_hugoniot(rho, C0, v, S) > shock_stress_acoustic(rho, C0, v)

    def test_equals_acoustic_when_S_zero(self):
        rho, C0, v = 8960, 3950, 200.0
        # When S=0: σ = ρ * C0 * u_p = ρ * C0 * v/2 = acoustic formula
        assert shock_stress_hugoniot(rho, C0, v, S=0.0) == pytest.approx(
            shock_stress_acoustic(rho, C0, v)
        )

    def test_gpa_value_copper(self):
        # Cu, v_peak ≈ 300 m/s → σ ≈ few GPa
        sigma_pa = shock_stress_hugoniot(8960, 3950, 300.0, 1.49)
        sigma_gpa = sigma_pa * 1e-9
        assert 1.0 < sigma_gpa < 10.0


class TestShockStressUncertainty:
    def test_positive_for_positive_unc(self):
        unc = shock_stress_hugoniot_uncertainty(8960, 3950, 300.0, 10.0, 1.49)
        assert unc > 0

    def test_scales_linearly_with_velocity_unc(self):
        unc1 = shock_stress_hugoniot_uncertainty(8960, 3950, 300.0, 10.0, 1.49)
        unc2 = shock_stress_hugoniot_uncertainty(8960, 3950, 300.0, 20.0, 1.49)
        assert unc2 == pytest.approx(2 * unc1, rel=1e-9)

    def test_zero_unc_gives_zero(self):
        assert shock_stress_hugoniot_uncertainty(8960, 3950, 300.0, 0.0, 1.49) == pytest.approx(0.0)


class TestCalculateShockStress:
    def test_hugoniot_method(self):
        r = calculate_shock_stress(8960, 3950, 300.0, method="hugoniot")
        assert r["method"] == "hugoniot"
        assert r["shock_stress_gpa"] == pytest.approx(r["shock_stress_pa"] * 1e-9)
        assert r["shock_stress_gpa"] > 0

    def test_acoustic_method(self):
        r = calculate_shock_stress(8960, 3950, 300.0, method="acoustic")
        assert r["method"] == "acoustic"
        assert np.isnan(r["S"])

    def test_material_lookup(self):
        r_cu = calculate_shock_stress(8960, 3950, 300.0, material="Cu")
        assert r_cu["S"] == pytest.approx(1.49)

    def test_explicit_S_overrides_material(self):
        r = calculate_shock_stress(8960, 3950, 300.0, material="Cu", S=1.0)
        assert r["S"] == pytest.approx(1.0)

    def test_uncertainty_propagated(self):
        r = calculate_shock_stress(8960, 3950, 300.0, peak_velocity_unc=10.0)
        assert r["shock_stress_unc_gpa"] > 0

    def test_gpa_pa_consistency(self):
        r = calculate_shock_stress(8960, 3950, 300.0, peak_velocity_unc=5.0)
        assert r["shock_stress_pa"] == pytest.approx(r["shock_stress_gpa"] * 1e9)
        assert r["shock_stress_unc_pa"] == pytest.approx(r["shock_stress_unc_gpa"] * 1e9)


# ============================================================
# 2. Spall DNS qualifier
# ============================================================

from alpss.analysis.spall import (
    _rdp_simplify,
    _detect_spall_topology,
    spall_analysis_with_dns,
    SpallResult,
)


class TestRDPSimplify:
    def test_straight_line_collapses_to_endpoints(self):
        t = np.linspace(0, 10, 100)
        v = 3.0 * t + 1.0  # perfect line
        pts = np.column_stack((t, v))
        idx = _rdp_simplify(pts, epsilon=0.01)
        assert len(idx) == 2  # only endpoints needed

    def test_fewer_than_3_points_returned_as_is(self):
        pts = np.array([[0, 0], [1, 1]])
        idx = _rdp_simplify(pts, epsilon=1.0)
        assert len(idx) == 2

    def test_checkmark_shape_preserves_key_vertices(self):
        # Build plateau → drop → rebound
        t = np.linspace(0, 50, 500)
        v = np.zeros_like(t)
        v[t <= 10] = 500.0                                       # plateau
        v[(t > 10) & (t <= 25)] = 500 - 25 * (t[(t > 10) & (t <= 25)] - 10)  # pullback
        v[t > 25] = 125 + 5 * (t[t > 25] - 25)                 # rebound
        pts = np.column_stack((t, v))
        idx = _rdp_simplify(pts, epsilon=5.0)
        assert len(idx) >= 3  # at least peak, valley, rebound


class TestDetectSpallTopology:
    @pytest.fixture
    def valid_checkmark(self):
        """Synthetic velocity trace with a clear checkmark."""
        t = np.linspace(0, 80, 800)
        v = np.zeros_like(t)
        v[t <= 15] = 500.0                                          # plateau
        v[(t > 15) & (t <= 35)] = 500 - 20 * (t[(t > 15) & (t <= 35)] - 15)   # pullback
        v[t > 35] = 100 + 8 * (t[t > 35] - 35)                    # rebound
        return t, v

    def test_valid_spall_detected(self, valid_checkmark):
        t, v = valid_checkmark
        ok, reason, keys, pts = _detect_spall_topology(t, v, rdp_epsilon=5.0,
                                                        min_pullback_velocity=30.0,
                                                        min_recomp_ratio=0.01,
                                                        min_recomp_velocity_ratio=1.01,
                                                        min_recomp_time_ns=1.0)
        assert ok is True
        assert keys is not None
        assert keys["pullback_depth"] > 30

    def test_returns_false_for_flat_signal(self):
        t = np.linspace(0, 50, 500)
        v = np.ones(500) * 200.0
        ok, reason, keys, _ = _detect_spall_topology(t, v, rdp_epsilon=2.0)
        assert ok is False

    def test_returns_false_when_pullback_too_small(self, valid_checkmark):
        t, v = valid_checkmark
        ok, reason, keys, _ = _detect_spall_topology(
            t, v, rdp_epsilon=5.0, min_pullback_velocity=9999.0
        )
        assert ok is False
        assert "pullback" in reason.lower() or "minimum" in reason.lower()

    def test_rdp_points_returned_on_failure(self):
        t = np.linspace(0, 50, 100)
        v = np.ones(100) * 100.0
        ok, reason, keys, rdp_pts = _detect_spall_topology(t, v)
        assert rdp_pts is not None


class TestSpallAnalysisWithDNS:
    @pytest.fixture
    def vc_iua(self):
        """Minimal vc_out / iua_out mimicking ALPSS output for a valid spall trace."""
        np.random.seed(0)
        t = np.linspace(0, 80e-9, 800)   # seconds
        v = np.zeros(800)
        # plateau at 500 m/s for 15 ns, pullback to 100 m/s, rebound
        v[t <= 15e-9] = 500.0
        drop_mask = (t > 15e-9) & (t <= 35e-9)
        v[drop_mask] = 500 - 20 * (t[drop_mask] - 15e-9) / 1e-9
        v[t > 35e-9] = 100 + 8 * (t[t > 35e-9] - 35e-9) / 1e-9
        v += np.random.normal(0, 2, 800)

        vc = {"time_f": t, "velocity_f_smooth": v}
        iua = {"vel_uncert": np.ones(800) * 5.0, "freq_uncert": np.ones(800) * 1e6}
        return vc, iua

    def test_dns_disabled_returns_early(self):
        vc = {"time_f": np.zeros(10), "velocity_f_smooth": np.zeros(10)}
        iua = {"vel_uncert": np.zeros(10), "freq_uncert": np.zeros(10)}
        r = spall_analysis_with_dns(vc, iua, spall_calculation="no",
                                    C0=3950, density=8960,
                                    pb_neighbors=3, pb_idx_correction=0)
        assert r.ok is False
        assert "disabled" in r.dns_classification

    def test_valid_spall_classified(self, vc_iua):
        vc, iua = vc_iua
        r = spall_analysis_with_dns(
            vc, iua,
            spall_detection_method="max_min",
            spall_calculation="yes",
            C0=3950, density=8960,
            pb_neighbors=5, pb_idx_correction=0,
        )
        assert isinstance(r, SpallResult)
        assert r.ok is True
        assert r.dns_classification == "Valid Spall"
        assert r.spall_strength_pa > 0
        assert np.isfinite(r.strain_rate)

    def test_rdp_method_runs(self, vc_iua):
        vc, iua = vc_iua
        r = spall_analysis_with_dns(
            vc, iua,
            spall_detection_method="rdp",
            spall_calculation="yes",
            C0=3950, density=8960,
            pb_neighbors=5, pb_idx_correction=0,
            rdp_epsilon=5.0,
            min_pullback_velocity=10.0,
        )
        assert isinstance(r, SpallResult)
        # Should detect valid spall or give informative DNS reason
        assert isinstance(r.dns_classification, str)

    def test_spall_result_fields_populated(self, vc_iua):
        vc, iua = vc_iua
        r = spall_analysis_with_dns(
            vc, iua,
            spall_detection_method="max_min",
            spall_calculation="yes",
            C0=3950, density=8960,
            pb_neighbors=5, pb_idx_correction=0,
        )
        assert np.isfinite(r.v_peak)
        assert np.isfinite(r.v_pullback)
        assert np.isfinite(r.t_peak)
        assert np.isfinite(r.t_pullback)


# ============================================================
# 3. HEL RDP + Linear Hybrid
# ============================================================

from alpss.analysis.hel import hel_detection, hel_detection_rdp_hybrid, HELResult


@pytest.fixture
def synthetic_hel_signal():
    """Synthetic PDV trace with a sharp HEL knee at ~4 ns."""
    np.random.seed(7)
    t = np.linspace(-2, 25, 200)
    v = np.zeros_like(t)

    # Baseline noise before t=0
    v[t < 0] = np.random.normal(0, 0.5, (t < 0).sum())

    # Steep rise to HEL plateau (0–4 ns)
    rise_mask = (t >= 0) & (t < 4)
    v[rise_mask] = 160 * t[rise_mask] / 4 + np.random.normal(0, 0.3, rise_mask.sum())

    # HEL plateau (4–12 ns) at ~160 m/s
    plat_mask = (t >= 4) & (t < 12)
    v[plat_mask] = 160 + np.random.normal(0, 0.5, plat_mask.sum())

    # Ramp to peak (12–25 ns)
    ramp_mask = t >= 12
    v[ramp_mask] = 160 + 30 * (t[ramp_mask] - 12) + np.random.normal(0, 1, ramp_mask.sum())

    unc = np.ones_like(v) * 5.0
    return t, v, unc


class TestHELRDPHybrid:
    def test_detects_hel_in_synthetic(self, synthetic_hel_signal):
        t, v, u = synthetic_hel_signal
        result = hel_detection_rdp_hybrid(
            t, v, u,
            hel_start_ns=0.0, hel_end_ns=15.0,
            min_velocity=50.0,
            density=8960, acoustic_velocity=3950, C_L=4700,
            rdp_epsilon=2.0,
            slope_drop_ratio=0.85,
            min_plateau_duration_ns=0.5,
            min_points=5,
        )
        assert result.ok is True
        assert result.method in ("rdp_linear", "gradient_fallback")
        assert result.strength_gpa > 0
        assert result.free_surface_velocity == pytest.approx(160, abs=15)

    def test_strain_rate_is_positive(self, synthetic_hel_signal):
        t, v, u = synthetic_hel_signal
        result = hel_detection_rdp_hybrid(
            t, v, u,
            hel_start_ns=0.0, hel_end_ns=15.0,
            min_velocity=50.0, C_L=4700,
            rdp_epsilon=2.0,
        )
        if result.ok:
            assert result.strain_rate > 0

    def test_returns_helresult_always(self, synthetic_hel_signal):
        t, v, u = synthetic_hel_signal
        result = hel_detection_rdp_hybrid(t, v, u, min_velocity=9999.0)
        assert isinstance(result, HELResult)

    def test_fallback_on_flat_signal(self):
        t = np.linspace(0, 20, 200)
        v = np.ones(200) * 100.0
        u = np.ones(200) * 2.0
        result = hel_detection_rdp_hybrid(t, v, u, min_velocity=50.0)
        assert isinstance(result, HELResult)
        assert result.method in ("rdp_linear", "gradient", "gradient_fallback")

    def test_rdp_points_stored_when_ok(self, synthetic_hel_signal):
        t, v, u = synthetic_hel_signal
        result = hel_detection_rdp_hybrid(
            t, v, u, hel_start_ns=0.0, hel_end_ns=15.0,
            min_velocity=50.0, rdp_epsilon=2.0,
        )
        if result.ok and result.method == "rdp_linear":
            assert result.rdp_points is not None
            assert result.rdp_points.shape[1] == 2

    def test_hel_detection_dispatches_to_hybrid(self, synthetic_hel_signal):
        """hel_detection(method='rdp_linear') should delegate correctly."""
        t, v, u = synthetic_hel_signal
        result = hel_detection(
            t, v, u,
            hel_start_ns=0.0, hel_end_ns=15.0,
            min_velocity=50.0, method="rdp_linear",
            density=8960, acoustic_velocity=3950,
            hel_rdp_epsilon=2.0,
        )
        assert isinstance(result, HELResult)
        assert result.method in ("rdp_linear", "gradient_fallback", "gradient")

    def test_hel_detection_gradient_still_works(self, synthetic_hel_signal):
        """Original gradient method must be unaffected."""
        t, v, u = synthetic_hel_signal
        result = hel_detection(
            t, v, u,
            hel_start_ns=0.0, hel_end_ns=15.0,
            min_velocity=50.0, method="gradient",
            density=8960, acoustic_velocity=3950,
        )
        assert isinstance(result, HELResult)
        assert result.method == "gradient"
