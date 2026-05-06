"""Tests for inverse-design preflight checks (issue #30).

Covers the four new checks:
  1. Tightened minimum-resolution thresholds (5 cells PEC / 10 cells dielectric)
  2. NTFF ↔ PEC overlap (hard error)
  3. NTFF near-field gap < λ/4 (warning)
  4. AD memory estimate via ``sim.estimate_ad_memory`` / ``check_ad_memory``
"""

from __future__ import annotations

import warnings

import pytest

from rfx.api import Simulation, AD_MemoryEstimate
from rfx.geometry.csg import Box


C = 2.998e8


def _sane_sim(freq_max=5e9, domain=(0.04, 0.04, 0.04)):
    sim = Simulation(freq_max=freq_max, domain=domain, boundary="cpml",
                     cpml_layers=8, dx=C / freq_max / 20.0)
    sim.add_port((domain[0] / 2, domain[1] / 2, domain[2] / 2), "ez")
    sim.add_probe((domain[0] / 2 + 2e-3, domain[1] / 2, domain[2] / 2), "ez")
    return sim


def test_ntff_pec_overlap_rejects():
    """NTFF face crossing a PEC bbox must raise an error."""
    sim = _sane_sim()
    # PEC waveguide-like wall spanning z = 10..40 mm, large xy extent
    sim.add(Box((0.005, 0.005, 0.010), (0.035, 0.035, 0.040)),
                     material="pec")
    # NTFF z_lo=24mm sits strictly inside the PEC z-range (10..40mm)
    sim.add_ntff_box(
        corner_lo=(0.010, 0.010, 0.024),
        corner_hi=(0.030, 0.030, 0.035),
        n_freqs=4,
    )
    # Either raises on strict=True or shows an ERROR entry in the report
    with pytest.raises(ValueError, match="NTFF face"):
        sim.preflight(strict=True)

    issues = sim.preflight(strict=False)
    assert any("NTFF face" in s and "PEC" in s for s in issues), issues


def test_ntff_near_field_gap_warns():
    """NTFF box 2 cells above a source should warn about λ/4 gap."""
    freq_max = 5e9
    dx = C / freq_max / 20.0  # 3 mm
    sim = Simulation(freq_max=freq_max, domain=(0.04, 0.04, 0.04),
                     boundary="cpml", cpml_layers=8, dx=dx)
    src_z = 0.020
    sim.add_port((0.020, 0.020, src_z), "ez")
    sim.add_probe((0.025, 0.020, src_z), "ez")
    # NTFF z_lo only 2 cells above the source — well below λ/4
    sim.add_ntff_box(
        corner_lo=(0.010, 0.010, src_z + 2 * dx),
        corner_hi=(0.030, 0.030, src_z + 3 * dx),
        n_freqs=4,
    )
    issues = sim.preflight(strict=False)
    assert any("λ/4" in s or "lambda/4" in s.lower() or "near-field" in s
               for s in issues), (
        "Expected a λ/4 near-field warning, got: " + "\n".join(issues)
    )


def test_ad_memory_estimate_reasonable():
    """40×40×40 @ 3GHz: estimate must be > 0 and < 10GB (checkpointed).
    At n_steps=10000 the non-checkpointed value must be much larger.
    """
    freq_max = 3e9
    dx = C / freq_max / 20.0
    # Use a small physical domain to land at ~40 cells
    extent = 40 * dx
    sim = Simulation(freq_max=freq_max, domain=(extent, extent, extent),
                     boundary="cpml", cpml_layers=6, dx=dx)
    sim.add_port((extent / 2, extent / 2, extent / 2), "ez")
    sim.add_probe((extent / 2 + dx, extent / 2, extent / 2), "ez")

    est_small = sim.estimate_ad_memory(500, available_memory_gb=40.0)
    assert isinstance(est_small, AD_MemoryEstimate)
    assert est_small.forward_gb > 0
    assert 0 < est_small.ad_checkpointed_gb < 10.0
    # checkpoint ≪ full when n_steps is large
    est_big = sim.estimate_ad_memory(10000, available_memory_gb=40.0)
    assert est_big.ad_full_gb > 5 * est_big.ad_checkpointed_gb


def test_preflight_passes_on_clean_setup():
    """A sane patch-antenna-ish sim triggers no ERROR / NTFF warnings.

    The original 5 GHz / 60 mm-cube version pre-2026-05-07 was not
    actually sane: the GP was sub-cell (2 mm at dx≈3 mm = 0.7 cells),
    the NTFF box reached into the 24 mm-thick x-CPML, and after the
    NTFF λ/2 near-field guardrail landed (NTFF face must be ≥ λ/2
    from any source), the 60 mm-cube domain at 5 GHz left no room
    for a sane NTFF box at all (λ/2 = 30 mm, full domain = 60 mm).
    Bumped to 10 GHz / 60 mm cube so λ/2 = 15 mm and a real
    1-cell-margin NTFF box fits clear of CPML and clear of the
    near field on every face.
    """
    freq_max = 10e9
    dx = C / freq_max / 20.0   # 1.5 mm
    extent = 0.060             # 60 mm cube; CPML 8·dx = 12 mm, so the
                                # CPML-free interior is [12, 48] mm.
    sim = Simulation(freq_max=freq_max, domain=(extent, extent, extent),
                     boundary="cpml", cpml_layers=8, dx=dx)
    # Ground plane well resolved (xy = 24 cells, z = 8 cells PEC volume).
    sim.add(Box((0.012, 0.012, 0.012), (0.048, 0.048, 0.024)),
                     material="pec")
    # Source 2 cells above GP, clear of CPML on every face.
    sim.add_port((0.030, 0.030, 0.027), "ez")
    sim.add_probe((0.032, 0.030, 0.027), "ez")
    # NTFF box: λ_min = 30 mm at 10 GHz → λ/2 = 15 mm.  Every face
    # sits ≥ 17 mm from the source, well above both λ/4 and λ/2.
    sim.add_ntff_box(
        corner_lo=(0.013, 0.013, 0.044),
        corner_hi=(0.047, 0.047, 0.047),
        n_freqs=4,
    )
    issues = sim.preflight(strict=False)
    # Filter to errors/NTFF/resolution issues that this test guards.
    bad = [s for s in issues
           if s.startswith("ERROR:")
           or "NTFF face" in s
           or "λ/4" in s
           or "λ/2" in s]
    assert not bad, "Clean setup should not trip new checks: " + "\n".join(bad)


def test_waveguide_resolution_warns():
    """Dielectric block whose ``cells_per_λ_eff`` is below the P1.5
    threshold (10) should warn — the rule fires on per-axis effective
    wavelength, not on per-dimension cell count.

    Original WR-90 setup (eps_r=2.2 / freq_max=10 GHz / dx=2 mm) gave
    ~10.1 cells/λ_eff, marginally above the threshold; the warning was
    silent and the assertion never tested the rule. Bumped eps_r to 3.0
    so cells_per_λ_eff ≈ 8.7 and P1.5 actually fires.
    """
    freq_max = 10e9
    dx = 2e-3
    sim = Simulation(freq_max=freq_max, domain=(0.05, 0.05, 0.05),
                     boundary="cpml", cpml_layers=6, dx=dx)
    sim.add_material("diel", eps_r=3.0)
    sim.add(Box((0.010, 0.010, 0.010),
                         (0.010 + 0.01016, 0.010 + 0.02286, 0.040)),
                     material="diel")
    sim.add_port((0.015, 0.020, 0.025), "ez")
    sim.add_probe((0.018, 0.020, 0.025), "ez")
    issues = sim.preflight(strict=False)
    # P1.5 wording was reworded from "under-resolved" to "Need ≥10 for
    # phase-accurate propagation"; pin the substantive content
    # (dielectric + cells/λ_eff threshold) rather than the old wording.
    assert any(
        "dielectric" in s
        and ("cells per λ_eff" in s or "phase-accurate" in s)
        for s in issues
    ), "Expected a dielectric resolution warning: " + "\n".join(issues)
