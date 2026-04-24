"""Issue #37: preflight thresholds must be physics-based, not cell-count.

Validated configurations (e.g. 05_patch_antenna) should produce no false
under-resolved warnings. Only genuine under-resolution should warn.
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation, Box


def _issues(sim):
    return sim.preflight()


def _has(issues, substring):
    return any(substring in i for i in issues)


def test_thin_pec_sheet_is_silent():
    """1-cell PEC on dx=0.5mm (half-wavelength-fraction) should not warn."""
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
                     cpml_layers=4)
    sim.add_source((0.005, 0.005, 0.002), "ez")
    sim.add_probe((0.005, 0.005, 0.005), "ez")
    sim.add(Box((0.003, 0.003, 0.005), (0.007, 0.007, 0.0055)), material="pec")
    issues = _issues(sim)
    assert not _has(issues, "PEC volume"), (
        f"1-cell PEC should not trigger a volume under-resolved warning; "
        f"issues: {issues!r}"
    )


def test_partial_pec_volume_warns():
    """3-cell PEC extent is the partial-volume case — should warn."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     cpml_layers=4)
    sim.add_source((0.01, 0.01, 0.002), "ez")
    sim.add(Box((0.005, 0.005, 0.005), (0.010, 0.010, 0.008)),
            material="pec")
    issues = _issues(sim)
    assert _has(issues, "PEC volume"), (
        f"3-cell PEC volume should warn; issues: {issues!r}"
    )


def test_fine_dielectric_is_silent():
    """Dielectric with ≥10 cells per λ_eff should not warn."""
    sim = Simulation(freq_max=2.4e9, domain=(0.08, 0.08, 0.04), dx=1e-3,
                     cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    # 60x60x1.5mm substrate at dx=1mm: λ_eff/dx ≈ 60mm/1mm = 60 → silent.
    sim.add(Box((0.010, 0.010, 0.012),
                (0.070, 0.070, 0.0135)), material="fr4")
    sim.add_source((0.04, 0.04, 0.013), "ez")
    issues = _issues(sim)
    assert not _has(issues, "cells per λ_eff"), (
        f"FR4 at 2.4 GHz with dx=1mm should be silent; issues: {issues!r}"
    )


def test_coarse_dielectric_warns():
    """Dielectric with dx near λ_eff should warn."""
    sim = Simulation(freq_max=30e9, domain=(0.02, 0.02, 0.02), dx=2e-3,
                     cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    sim.add(Box((0.005, 0.005, 0.005), (0.015, 0.015, 0.015)),
            material="fr4")
    sim.add_source((0.010, 0.010, 0.003), "ez")
    issues = _issues(sim)
    assert _has(issues, "cells per λ_eff"), (
        f"Coarse FR4 should warn; issues: {issues!r}"
    )


def test_dielectric_near_old_threshold_now_warns():
    """~12 cells/λ_eff — above old threshold (10) but below new (15).

    rfx's Yee update without subpixel smoothing degrades to 1st-order
    at ε discontinuities (Meep ships subpixel ON to stay 2nd-order).
    The pre-2026-04-24 threshold of 10 cells/λ_eff was borrowed from
    subpixel-smoothed codes and is too loose for raw Yee. Raised to 15.
    """
    # εr=2, f_max=17.5 GHz, dx=1mm → λ_eff=12.1mm → 12.1 cells/λ_eff.
    sim = Simulation(freq_max=17.5e9, domain=(0.04, 0.02, 0.02), dx=1e-3,
                     cpml_layers=4)
    sim.add_material("eps2", eps_r=2.0)
    sim.add(Box((0.010, 0.005, 0.005), (0.030, 0.015, 0.015)),
            material="eps2")
    sim.add_source((0.020, 0.010, 0.005), "ez")
    issues = _issues(sim)
    assert _has(issues, "cells per λ_eff"), (
        f"12 cells/λ_eff should warn under the tightened threshold; "
        f"issues: {issues!r}"
    )


def _build_wr90_slab_nu_sim(dx_fine):
    """Shared WR-90 εr=2 slab with a refined interior band along x."""
    import numpy as _np
    import jax.numpy as _jnp
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary
    from rfx.geometry.csg import Box as _Box
    from rfx.auto_config import smooth_grading

    a_wg, b_wg = 0.02286, 0.01016
    dom_x = 0.200
    slab_lo, slab_hi = 0.095, 0.105
    dx_coarse = 1e-3
    n_pre = int(round(slab_lo / dx_coarse))
    n_slab = int(round((slab_hi - slab_lo) / dx_fine))
    n_post = int(round((dom_x - slab_hi) / dx_coarse))
    raw = _np.concatenate([
        _np.full(n_pre, dx_coarse),
        _np.full(n_slab, dx_fine),
        _np.full(n_post, dx_coarse),
    ])
    dx_profile = smooth_grading(raw, max_ratio=1.3)

    sim = Simulation(
        freq_max=12e9, domain=(float(_np.sum(dx_profile)), a_wg, b_wg),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=20, dx=dx_coarse, dx_profile=dx_profile,
    )
    sim.add_material("slab", eps_r=2.0)
    sim.add(_Box((slab_lo, 0, 0), (slab_hi, a_wg, b_wg)),
            material="slab")
    port_freqs = _jnp.linspace(8.2e9, 12.4e9, 5)
    sim.add_waveguide_port(
        0.040, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=10.3e9, bandwidth=0.5,
        reference_plane=0.050, name="left",
    )
    sim.add_waveguide_port(
        0.160, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=10.3e9, bandwidth=0.5,
        reference_plane=0.150, name="right",
    )
    return sim


def test_compute_waveguide_s_matrix_rejects_unnormalized_nu():
    """``normalize=False`` on a NU mesh must raise — the dispersion-
    cancellation two-run is the only validated NU lane today.
    """
    sim = _build_wr90_slab_nu_sim(dx_fine=0.25e-3)
    try:
        sim.compute_waveguide_s_matrix(num_periods=2, normalize=False)
    except NotImplementedError as exc:
        assert "non-uniform" in str(exc).lower()
        assert "normalize=true" in str(exc).lower()
    else:
        raise AssertionError(
            "compute_waveguide_s_matrix on a dx_profile grid with "
            "normalize=False should raise NotImplementedError."
        )


def test_compute_waveguide_s_matrix_dispatches_nu_when_normalized():
    """With ``normalize=True`` and single-mode ports, the NU lane runs
    end-to-end (CPML-on-PEC-axis fix lands).  This is the regression
    that locks in the dispatch wiring; numeric accuracy is exercised
    by ``scripts/nu_vs_uniform_slab_cost_accuracy.py``.
    """
    sim = _build_wr90_slab_nu_sim(dx_fine=0.5e-3)
    res = sim.compute_waveguide_s_matrix(num_periods=2, normalize=True)
    assert res.s_params.shape[0] == 2  # two ports
    assert res.s_params.shape[1] == 2
    assert res.s_params.shape[2] == 5  # five freqs


def test_dielectric_sparam_active_raises_threshold_to_20():
    """17 cells/λ_eff silent without S-param extraction; warns with one.

    S-parameter extraction (waveguide port or flux monitor) amplifies
    ε-interface phase error into |S| magnitude error — the WR-90 εr=2
    case at dx=1mm, f_max=12 GHz sits at 17.7 cells/λ_eff and shows
    ~5% |S21| deficit at Fabry-Perot peaks vs analytic Airy. The
    preflight tightens to 20 cells/λ_eff when any port or flux
    monitor is present.
    """
    # εr=2, f_max=12 GHz, dx=1mm → λ_eff=17.7mm → 17.7 cells/λ_eff.
    common = dict(freq_max=12e9, domain=(0.05, 0.02286, 0.01016),
                  dx=1e-3, cpml_layers=4)

    # Case A: same resolution, NO S-param — should be silent (17.7 > 15).
    sim_a = Simulation(**common)
    sim_a.add_material("eps2", eps_r=2.0)
    sim_a.add(Box((0.020, 0.0, 0.0), (0.030, 0.02286, 0.01016)),
              material="eps2")
    sim_a.add_source((0.025, 0.01143, 0.00508), "ez")
    issues_a = _issues(sim_a)
    assert not _has(issues_a, "cells per λ_eff"), (
        f"17.7 cells/λ_eff without S-param should stay silent; "
        f"issues: {issues_a!r}"
    )

    # Case B: same resolution + waveguide port — should WARN (17.7 < 20).
    sim_b = Simulation(**common)
    sim_b.add_material("eps2", eps_r=2.0)
    sim_b.add(Box((0.020, 0.0, 0.0), (0.030, 0.02286, 0.01016)),
              material="eps2")
    sim_b.add_waveguide_port(
        0.005, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=np.linspace(8e9, 12e9, 11), f0=10e9, bandwidth=0.4,
    )
    issues_b = _issues(sim_b)
    assert _has(issues_b, "cells per λ_eff"), (
        f"17.7 cells/λ_eff WITH waveguide port should warn (threshold "
        f"raised to 20 when S-param extraction is active); "
        f"issues: {issues_b!r}"
    )
    # Make sure the stronger hint is present.
    assert any("S-parameter extraction" in i for i in issues_b), (
        f"S-param-specific suffix missing; issues: {issues_b!r}"
    )
