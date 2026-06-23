"""Tests for closed-form microstrip synthesis/analysis (rfx.microstrip).

Reference points are taken from the standard Hammerstad-Jensen quasi-static
model (Pozar, *Microwave Engineering*, 4th ed., §3.8). The model is ~1%
accurate, so tolerances are a few percent.
"""

import math

import pytest

from rfx.microstrip import (
    microstrip_impedance,
    microstrip_eps_eff,
    microstrip_width,
)


# ---------------------------------------------------------------------------
# Analysis vs published reference points
# ---------------------------------------------------------------------------

class TestAnalysisReference:
    def test_fr4_50ohm_geometry(self):
        # 50 ohm microstrip on FR4 (eps_r=4.4, h=1.6 mm): standard textbook
        # result is w ~ 3.0-3.1 mm and eps_eff ~ 3.3.
        w = 3.06e-3
        z0, eps_eff = microstrip_impedance(w, 1.6e-3, 4.4)
        assert z0 == pytest.approx(50.0, abs=1.0)  # within ~2%
        assert eps_eff == pytest.approx(3.3, abs=0.1)

    def test_ro4003c_50ohm_geometry(self):
        # Rogers RO4003C (eps_r=3.38), h=0.508 mm (20 mil): a 50 ohm line is
        # ~1.15 mm wide with eps_eff ~ 2.65 (Rogers app notes / TX-line tools).
        w = 1.17e-3
        z0, eps_eff = microstrip_impedance(w, 0.508e-3, 3.38)
        assert z0 == pytest.approx(50.0, abs=1.5)  # within ~3%
        assert eps_eff == pytest.approx(2.67, abs=0.1)

    def test_eps_eff_bounds(self):
        # eps_eff is always between 1 and eps_r.
        eps_r = 4.4
        eps_eff = microstrip_eps_eff(3.0e-3, 1.6e-3, eps_r)
        assert 1.0 < eps_eff < eps_r

    def test_eps_eff_matches_impedance_call(self):
        # microstrip_impedance must return the same eps_eff as the standalone.
        _, eps_eff_from_z = microstrip_impedance(2.0e-3, 1.0e-3, 3.55)
        eps_eff_direct = microstrip_eps_eff(2.0e-3, 1.0e-3, 3.55)
        assert eps_eff_from_z == pytest.approx(eps_eff_direct, rel=1e-12)

    def test_wider_trace_lowers_impedance(self):
        # Monotonicity sanity: wider trace -> lower Z0.
        z_narrow, _ = microstrip_impedance(0.5e-3, 1.6e-3, 4.4)
        z_wide, _ = microstrip_impedance(5.0e-3, 1.6e-3, 4.4)
        assert z_narrow > z_wide


# ---------------------------------------------------------------------------
# Synthesis <-> analysis roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    @pytest.mark.parametrize("eps_r", [4.4, 3.38, 2.2, 9.8])
    @pytest.mark.parametrize("z0_target", [50.0, 75.0])
    def test_roundtrip_recovers_z0(self, eps_r, z0_target):
        h = 1.6e-3
        w = microstrip_width(z0_target, h, eps_r)
        assert w > 0.0
        z0_back, _ = microstrip_impedance(w, h, eps_r)
        # Synthesis and analysis are different closed forms; HJ accuracy ~1%.
        assert z0_back == pytest.approx(z0_target, rel=0.03)

    def test_roundtrip_straddles_wh_boundary(self):
        # Low eps_r + high Z0 lands in the narrow (w/h < 1) regime; high
        # eps_r + low Z0 lands in the wide (w/h > 1) regime. Exercise both.
        h = 1.0e-3
        # Narrow regime case (high-Z line on a low-eps_r substrate).
        w_narrow = microstrip_width(120.0, h, 2.2)
        assert w_narrow / h < 1.0
        z_narrow, _ = microstrip_impedance(w_narrow, h, 2.2)
        assert z_narrow == pytest.approx(120.0, rel=0.03)
        # Wide regime case.
        w_wide = microstrip_width(40.0, h, 9.8)
        assert w_wide / h > 1.0
        z_wide, _ = microstrip_impedance(w_wide, h, 9.8)
        assert z_wide == pytest.approx(40.0, rel=0.03)

    def test_fr4_50ohm_width_value(self):
        # The headline number: 50 ohm on FR4 (h=1.6 mm) -> ~3.0-3.1 mm.
        w = microstrip_width(50.0, 1.6e-3, 4.4)
        assert 3.0e-3 <= w <= 3.1e-3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_impedance_rejects_nonpositive_width(self):
        with pytest.raises(ValueError):
            microstrip_impedance(0.0, 1.6e-3, 4.4)
        with pytest.raises(ValueError):
            microstrip_impedance(-1e-3, 1.6e-3, 4.4)

    def test_impedance_rejects_nonpositive_height(self):
        with pytest.raises(ValueError):
            microstrip_impedance(3.0e-3, 0.0, 4.4)

    def test_impedance_rejects_eps_below_one(self):
        with pytest.raises(ValueError):
            microstrip_impedance(3.0e-3, 1.6e-3, 0.9)

    def test_eps_eff_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            microstrip_eps_eff(-1.0, 1.6e-3, 4.4)
        with pytest.raises(ValueError):
            microstrip_eps_eff(3.0e-3, 1.6e-3, 0.5)

    def test_width_rejects_nonpositive_z0(self):
        with pytest.raises(ValueError):
            microstrip_width(0.0, 1.6e-3, 4.4)
        with pytest.raises(ValueError):
            microstrip_width(-50.0, 1.6e-3, 4.4)

    def test_width_rejects_bad_substrate(self):
        with pytest.raises(ValueError):
            microstrip_width(50.0, -1.6e-3, 4.4)
        with pytest.raises(ValueError):
            microstrip_width(50.0, 1.6e-3, 0.5)


# ---------------------------------------------------------------------------
# Substrate presets (added in this change)
# ---------------------------------------------------------------------------

class TestSubstratePresets:
    def test_new_presets_in_library(self):
        from rfx import MATERIAL_LIBRARY
        assert "rogers4350b" in MATERIAL_LIBRARY
        assert "rt_duroid_5880" in MATERIAL_LIBRARY

    def test_ro4350b_eps_r(self):
        from rfx import MATERIAL_LIBRARY
        assert MATERIAL_LIBRARY["rogers4350b"]["eps_r"] == pytest.approx(3.48, abs=0.02)
        assert MATERIAL_LIBRARY["rogers4350b"]["sigma"] > 0.0  # lossy dielectric

    def test_rt_duroid_5880_eps_r(self):
        from rfx import MATERIAL_LIBRARY
        assert MATERIAL_LIBRARY["rt_duroid_5880"]["eps_r"] == pytest.approx(2.20, abs=0.02)
        assert MATERIAL_LIBRARY["rt_duroid_5880"]["sigma"] > 0.0

    def test_presets_resolvable_via_real_api(self):
        # Resolve through the actual Simulation material-resolution path.
        from rfx import Simulation
        sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.01))
        for name, expected in (("rogers4350b", 3.48), ("rt_duroid_5880", 2.20)):
            spec = sim._resolve_material(name)
            assert spec.eps_r == pytest.approx(expected, abs=0.02)

    def test_pcb_aliases_resolve(self):
        # Natural-name aliases map to canonical library keys.
        from rfx.pcb import resolve_pcb_material
        from rfx import MATERIAL_LIBRARY
        assert resolve_pcb_material("ro4350b") == "rogers4350b"
        assert resolve_pcb_material("rt5880") == "rt_duroid_5880"
        assert resolve_pcb_material("duroid5880") == "rt_duroid_5880"
        # Aliased canonical names exist in the library.
        assert resolve_pcb_material("ro4350b") in MATERIAL_LIBRARY
        assert resolve_pcb_material("rt5880") in MATERIAL_LIBRARY


class TestSynthesisRobustness:
    def test_high_z_high_eps_no_crash_and_warns(self):
        # A high-impedance line on a high-eps_r substrate drives the wide-form
        # argument b <= 1 (its logs would be complex). The synthesis must fall
        # through to the narrow branch and return a finite width — not raise a
        # bare math-domain error — while warning that w/h is extrapolated.
        with pytest.warns(UserWarning, match="validated range"):
            w = microstrip_width(200.0, 1.6e-3, 9.8)
        assert math.isfinite(w) and w > 0.0
