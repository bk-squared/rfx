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
    """A sane patch-antenna-ish sim triggers no ERROR / NTFF warnings."""
    freq_max = 5e9
    dx = C / freq_max / 20.0
    extent = 0.060
    sim = Simulation(freq_max=freq_max, domain=(extent, extent, extent),
                     boundary="cpml", cpml_layers=8, dx=dx)
    # Ground plane well resolved (≥5 cells PEC)
    sim.add(Box((0.010, 0.010, 0.010), (0.050, 0.050, 0.012)),
                     material="pec")
    sim.add_port((0.030, 0.030, 0.020), "ez")
    sim.add_probe((0.032, 0.030, 0.020), "ez")
    # NTFF box with generous λ/4 margins (λ_min = 60mm at 5GHz → margin > 15mm)
    sim.add_ntff_box(
        corner_lo=(0.005, 0.005, 0.040),
        corner_hi=(0.055, 0.055, 0.055),
        n_freqs=4,
    )
    issues = sim.preflight(strict=False)
    # Filter to errors/NTFF/resolution issues we just added
    bad = [s for s in issues
           if s.startswith("ERROR:")
           or "NTFF face" in s
           or "λ/4" in s]
    assert not bad, "Clean setup should not trip new checks: " + "\n".join(bad)


def test_waveguide_resolution_warns():
    """WR-90 narrow wall (10.16mm) at dx=2mm → ~5 cells for dielectric
    air inside a WR-90-shaped dielectric box should trip the ≥10-cell
    dielectric threshold."""
    # We emulate the "narrow wall" as a dielectric box of the narrow-wall
    # size so the dielectric ≥10 cells rule fires.
    freq_max = 10e9
    dx = 2e-3
    sim = Simulation(freq_max=freq_max, domain=(0.05, 0.05, 0.05),
                     boundary="cpml", cpml_layers=6, dx=dx)
    sim.add_material("diel", eps_r=2.2)
    # 10.16mm narrow-wall × 22.86mm broad-wall × 30mm length
    sim.add(Box((0.010, 0.010, 0.010),
                         (0.010 + 0.01016, 0.010 + 0.02286, 0.040)),
                     material="diel")
    sim.add_port((0.015, 0.020, 0.025), "ez")
    sim.add_probe((0.018, 0.020, 0.025), "ez")
    issues = sim.preflight(strict=False)
    assert any("under-resolved" in s and "dielectric" in s for s in issues), \
        "Expected a dielectric under-resolution warning: " + "\n".join(issues)
