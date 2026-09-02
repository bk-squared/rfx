"""Preflight — configuration guards and thresholds: physics-based mesh
thresholds, false-positive refinements, structured records + setup guards,
TFSF + lumped RLC.

One file per preflight stage (tier 3b of the 2026-09 test-corpus
reorganisation, see ``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``).
This file replaces three entries of the ``guards-and-preflight`` PR lane
(``.github/workflows/pr-tests.yml``). Sections, each formerly its own file:

1. **Issue #37: thresholds must be physics-based, not cell-count** — was
   ``test_preflight_physics_thresholds.py``. Validated configurations (e.g.
   05_patch_antenna) must produce no false under-resolved warnings; only
   genuine under-resolution warns (thin PEC sheet silent, partial PEC
   volume warns, dielectric cells-per-lambda_eff thresholds 15 / 20 when
   S-parameter extraction is active, NU dispatch of
   ``compute_waveguide_s_matrix``, waveguide-port evanescent margin on the
   REALIZED guide, #738).
2. **False-positive refinements (2026-05-06)** — was
   ``test_preflight_false_positives.py``. The Y2 MSL stub-notch demo exposed
   three checks firing on canonical transmission-line geometry: FP1 the
   PEC-volume warning on thin microstrip strips, FP3 the CPML-extension
   warning on full-domain substrates (issue #61), FP4 the inside-PEC warning
   on H-component probes at a 1-cell trace centre. Both halves of each
   refinement are pinned (the FP case goes silent, the original footgun
   still warns), plus the R2-STOP lock on ``pec_boundary_open`` (NTFF is
   the sole radiation-intent signal; valid closed structures must stay
   silent).
3. **Tier-2 structured preflight records + two setup guards** — was
   ``test_preflight_structured_and_guards.py``: (a) ``preflight()`` returns
   ``PreflightIssue`` (a str subclass carrying ``.severity`` / ``.code``) in
   a ``PreflightReport`` list with the canonical report API, codes set at
   the check site; (b) conformal-PEC + fine dx (<=2mm) surfaces at setup;
   (c) all-lossless dielectric in an open domain warns of the artificial-Q
   trap; plus the Phase A meta-coverage (every emitted issue coded),
   strict aggregate raise, ``raise_for_failure``, validator crashes
   propagate, run() advisory tier vs forward() error tier, the issue #166
   2D collapsed-z and unit-adaptive formatting locks.
4. **TFSF plane-wave + lumped RLC guard** — was
   ``test_preflight_tfsf_lumped.py``: a bare ``add_lumped_rlc`` driven by a
   TFSF plane wave has no defined series circuit and diverges (~1e35 by
   ~250 steps); the advisory warns on that pairing and not on TFSF alone or
   on the validated port-fed lane.

Every assertion, tolerance, fixture value and parametrisation of the
absorbed files is kept verbatim (the identical ``_issues`` / ``_has`` /
``_codes`` helpers are defined once).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from rfx import Simulation, Box, GaussianPulse
from rfx.api._preflight import (
    _PreflightMixin,
    PreflightErrorWarning,
    PreflightIssue,
    PreflightReport,
    PreflightWarning,
)


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_guards.py
# ===========================================================================

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


def test_wg_port_evanescent_no_warning_below_threshold():
    """freqs up to 6.5 GHz in a 40×20 mm guide: fc_TE20=7.5 GHz,
    threshold=0.90×7.5=6.75 GHz → 6.5 < 6.75, no evanescent warning.

    dx=0.002 (not the original 0.003): issue #738 made the 0.90×fc_next
    margin heuristic read the REALIZED guide, and dx=0.003 divides
    neither 40 mm nor 20 mm. Measured on this fixture at dx=0.003 (both
    y and z closed by PEC domain faces, so guide_source=domain_faces):
    the guide rasterizes to 42.0000 × 21.0000 mm, the next cutoff lands
    at 7.138 GHz (TE20 on 42 mm and TE01 on 21 mm are degenerate there)
    and the threshold at 6.424 GHz — which 6.5 GHz then exceeds. So the
    docstring above described a guide the fixture did not have. This is
    a fixture repair, not a gate move: at dx=0.002 the same measurement
    gives 40.0000 × 20.0000 mm with declared == rasterized on both axes,
    and the checker reproduces the original 7.5/6.75 GHz verdict
    unchanged (measured: preflight emits no findings at all).
    """
    import jax.numpy as jnp
    from rfx.boundaries.spec import BoundarySpec, Boundary

    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        dx=0.002,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        0.020, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=jnp.linspace(5e9, 6.5e9, 4), f0=6e9, bandwidth=0.5,
    )
    issues = _issues(sim)
    assert not any(getattr(i, "code", None) == "port_evanescent" for i in issues), (
        f"6.5 GHz < 0.90×fc_TE20=6.75 GHz should not warn; issues: {issues!r}"
    )


def test_wg_port_evanescent_warns_above_threshold():
    """freqs up to 7.0 GHz in a 40×20 mm guide: fc_TE20=7.5 GHz,
    threshold=0.90×7.5=6.75 GHz → 7.0 > 6.75, evanescent warning must fire.

    dx=0.002, see the sibling test above for the measurement (issue #738
    fixture repair, not a gate move)."""
    import jax.numpy as jnp
    from rfx.boundaries.spec import BoundarySpec, Boundary

    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        dx=0.002,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        0.020, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=jnp.linspace(5e9, 7.0e9, 4), f0=6e9, bandwidth=0.5,
    )
    issues = _issues(sim)
    # Issue #738 review: `_has(issues, "contamination")` (a prose
    # substring match) is the same fragile-substring failure mode this
    # issue exists to catch — `_has(issues, "evanescent")` silently could
    # not match "Evanescent {label} contamination" (case-sensitive E vs
    # e). Assert on the structured code instead.
    assert any(getattr(i, "code", None) == "port_evanescent" for i in issues), (
        f"7.0 GHz > 0.90×fc_next must trigger a port_evanescent finding; "
        f"issues: {issues!r}"
    )


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_guards.py
# ===========================================================================

def _codes(sim):
    return {getattr(i, "code", None) for i in sim.preflight()}


# ---------------------------------------------------------------------------
# FP1 — thin-sheet PEC strip should not trigger PEC-volume warning
# ---------------------------------------------------------------------------
def test_thin_pec_strip_with_4_cell_y_silent_on_volume_warning():
    """Strip-shape PEC with z = 1 cell is a thin sheet, not a volume.
    The 4-cells-along-y signal must not fire the volume warning."""
    DX = 0.5e-3
    LX, LY, LZ = 0.030, 0.005, 0.002
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                     cpml_layers=4)
    sim.add_source((LX/2, LY/2, 0.0005), "ez")
    # 30mm × 2mm × 0.5mm = 60 × 4 × 1 cells (y-axis is the FP zone)
    sim.add(Box((0.0, 0.0015, 0.001), (LX, 0.0035, 0.0015)),
            material="pec")
    issues = _issues(sim)
    assert not _has(issues, "PEC volume"), (
        f"thin PEC strip (1 cell in z) must not fire volume warning; "
        f"issues: {issues!r}"
    )


def test_pec_volume_partial_in_all_axes_still_warns():
    """A PEC that is 3-5 cells in EVERY axis is the original target of
    the volume warning — must still fire."""
    DX = 1e-3
    sim = Simulation(freq_max=10e9, domain=(0.020, 0.020, 0.020), dx=DX,
                     cpml_layers=4)
    sim.add_source((0.010, 0.010, 0.002), "ez")
    # 4mm × 4mm × 4mm = 4 × 4 × 4 cells: every axis in [3, 5).
    sim.add(Box((0.005, 0.005, 0.005), (0.009, 0.009, 0.009)),
            material="pec")
    issues = _issues(sim)
    assert _has(issues, "PEC volume"), (
        f"true 4-cell PEC volume must still warn; issues: {issues!r}"
    )


# ---------------------------------------------------------------------------
# FP3 — explicit full-domain Box edge is not a CPML-extension footgun
# ---------------------------------------------------------------------------
def test_full_domain_dielectric_silent_on_cpml_extension():
    """Box((0, 0, 0), (LX, LY, ...)) is the canonical MSL substrate
    pattern — must not trigger the issue #61 CPML-extension warning."""
    LX, LY, LZ = 0.030, 0.005, 0.002
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=0.5e-3,
                     cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    sim.add(Box((0, 0, 0), (LX, LY, 0.0005)), material="fr4")
    sim.add_source((LX/2, LY/2, 0.0002), "ez")
    issues = _issues(sim)
    assert not _has(issues, "extends into CPML"), (
        f"full-domain Box must not fire CPML-extension warning; "
        f"issues: {issues!r}"
    )


def test_inset_box_leaking_into_cpml_still_warns():
    """A Box that genuinely crosses the domain edge into the exterior
    absorber must still warn — this is the original issue #61
    leak-into-absorber case.

    Issue #500: CPML pads EXTERIOR to the requested domain, so
    ``[0, LZ]`` is absorber-free by construction — a Box inset short of
    the edge but still nominally within ``[0, LZ]`` (the pre-#500
    fixture here) can never touch the absorber, no matter how close to
    the edge it sits. The only genuine leak is a bounding-box coordinate
    that is literally negative (or past ``domain_extent``) — the box
    below is inset in x (unaffected, no leak there) but straddles z=0.
    """
    LX, LY, LZ = 0.030, 0.005, 0.002
    DX = 0.5e-3
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                     cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    # CPML thickness = 4 x 0.5mm = 2mm. Box inset 0.5mm in x (no leak
    # there) but c1[2]=-0.0003 is genuinely in the exterior z_lo absorber.
    sim.add(Box((0.0005, 0, -0.0003), (LX - 0.0005, LY, 0.0002)),
            material="fr4")
    sim.add_source((LX/2, LY/2, 0.0002), "ez")
    issues = _issues(sim)
    assert _has(issues, "extends into CPML"), (
        f"Box straddling the z=0 edge and leaking into CPML must still warn; "
        f"issues: {issues!r}"
    )


# ---------------------------------------------------------------------------
# FP4 — H-component probe at thin-PEC-sheet centre is valid
# ---------------------------------------------------------------------------
def _msl_sim_with_probe(component: str) -> Simulation:
    """Tiny MSL geometry with one diagnostic probe at the centre of the
    1-cell trace PEC.  Used to test FP4 component-aware exemption."""
    EPS_R = 3.66
    H_SUB = 254e-6
    W_TRACE = 600e-6
    DX = 127e-6
    LX, LY, LZ = 0.010, 0.005, H_SUB + 1.0e-3
    sim = Simulation(freq_max=9e9, domain=(LX, LY, LZ), dx=DX,
                     cpml_layers=4)
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_trace = LY / 2.0
    sim.add(
        Box((0, y_trace - W_TRACE / 2, H_SUB),
            (LX, y_trace + W_TRACE / 2, H_SUB + DX)),
        material="pec",
    )
    sim.add_source((0.5e-3, y_trace, 0.5 * H_SUB), "ez")
    # Probe at trace cell centre: z = H_SUB + 0.5·dx, inside the PEC
    # trace bbox by construction.
    sim.add_probe((LX / 2, y_trace, H_SUB + 0.5 * DX), component)
    return sim


def test_hy_probe_at_thin_trace_pec_silent_on_inside_pec():
    """An Hy diagnostic probe placed at the centre of a 1-cell trace
    PEC measures tangential H — physically non-zero — and must not
    trigger the inside-PEC warning."""
    sim = _msl_sim_with_probe("hy")
    issues = _issues(sim)
    assert not _has(issues, "is inside PEC geometry"), (
        f"Hy probe at thin-trace PEC centre must not warn; "
        f"issues: {issues!r}"
    )


def test_ez_probe_at_thin_trace_pec_still_warns():
    """An Ez probe at the same position is killed by the PEC update —
    the warning must still fire (only H components are exempt)."""
    sim = _msl_sim_with_probe("ez")
    issues = _issues(sim)
    assert _has(issues, "is inside PEC geometry"), (
        f"Ez probe inside thin PEC must still warn; issues: {issues!r}"
    )


def test_hy_probe_inside_thick_pec_volume_still_warns():
    """H decays to zero deep inside a thick PEC volume.  An Hy probe
    placed at the centre of a 5-cell PEC cube must still warn — the
    thin-sheet exemption applies only to ≤ 1.5·dx-thick PEC."""
    DX = 1.0e-3
    LX, LY, LZ = 0.020, 0.020, 0.020
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                     cpml_layers=4)
    sim.add_source((0.002, 0.002, 0.002), "ez")
    # 5 × 5 × 5 mm = 5 × 5 × 5 cells PEC volume.
    sim.add(Box((0.005, 0.005, 0.005), (0.010, 0.010, 0.010)),
            material="pec")
    sim.add_probe((0.0075, 0.0075, 0.0075), "hy")  # cell centre, deep
    issues = _issues(sim)
    assert _has(issues, "is inside PEC geometry"), (
        f"Hy probe in thick PEC volume must still warn; "
        f"issues: {issues!r}"
    )


# ---------------------------------------------------------------------------
# Item #3 (LLM-naive-usage audit) — pec_boundary_open advisory: R2-STOP lock.
#
# The audit asked whether the ``pec_boundary_open`` advisory
# (``_validate_cfg_pec_boundary_open_structure``) should be UNGATED from its
# ``self._ntff is not None`` condition so an open radiator read via a near-field
# probe / S11 alone (no NTFF box) also warns. Investigation (2026-07-09)
# R2-STOPPED that ungating: NTFF is the SOLE radiation-intent signal on the
# Simulation config (there is no directivity / far-field / "radiate" flag), so
# a source (and/or a finite PEC object) inside a ``boundary="pec"`` domain is
# config-IDENTICAL between an open radiator that mistakenly used PEC and a
# legitimate closed cavity / internal-PEC numerics test. The committed suite is
# full of the latter — e.g. ``test_adi.py::
# test_simulation_adi_internal_pec_geometry_masks_ez`` (PEC Box in a pec box +
# source), ``test_extract_s_matrix_pec_mask.py``, ``test_conformal.py::
# test_api_conformal_flag`` (PEC cylinder in a pec box + port). Any broadening
# that catches the footgun would false-alarm all of them, and a false-alarming
# preflight erodes trust worse than the silent gap. These tests LOCK that
# decision: the NTFF-declared open radiator still warns; the valid closed
# structures must stay silent so a future well-meaning ungating cannot regress
# them unnoticed.
# ---------------------------------------------------------------------------
def test_pec_boundary_open_still_warns_when_ntff_declared():
    """Radiation intent (an NTFF box) + boundary='pec' must still warn —
    the existing, principled gate is preserved by the R2-STOP."""
    sim = Simulation(freq_max=10e9, domain=(0.06, 0.06, 0.06), dx=2e-3,
                     boundary="pec")
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add(Box((0.028, 0.028, 0.020), (0.032, 0.032, 0.024)), material="pec")
    sim.add_ntff_box((0.01, 0.01, 0.01), (0.05, 0.05, 0.05))
    assert "pec_boundary_open" in _codes(sim), (
        "NTFF-declared open radiator on a PEC boundary must still warn"
    )


def test_pec_cavity_with_internal_pec_object_stays_silent():
    """FALSE-POSITIVE lock: a source + finite PEC object inside a pec box
    (the ``test_adi`` internal-PEC-masks-Ez / ``test_conformal`` patterns) is a
    VALID closed structure and must NOT emit pec_boundary_open. This is the
    population that any ntff-ungating would false-alarm — the reason #3 was
    R2-STOPPED."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     boundary="pec")
    sim.add(Box((0.008, 0.008, 0.0), (0.012, 0.012, 0.01)), material="pec")
    sim.add_source((0.01, 0.01, 0.0), "ez")
    sim.add_probe((0.01, 0.01, 0.0), "ez")
    assert "pec_boundary_open" not in _codes(sim), (
        "internal-PEC-object closed cavity must not warn (R2-STOP rationale)"
    )


def test_pec_empty_cavity_with_source_stays_silent():
    """FALSE-POSITIVE lock: a bare source in a pec box (empty resonant cavity)
    is config-identical to an open radiator and must NOT warn — there is no
    discriminator, hence the R2-STOP."""
    sim = Simulation(freq_max=12e9, domain=(0.03, 0.03, 0.03), dx=1.5e-3,
                     boundary="pec")
    sim.add_source((0.015, 0.015, 0.015), "ez")
    assert "pec_boundary_open" not in _codes(sim), (
        "empty PEC cavity with a source must not warn (R2-STOP rationale)"
    )


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_guards.py
# ===========================================================================

# ---------------------------------------------------------------- (a) records
def test_preflight_returns_back_compatible_structured_issues():
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.0225), component="ez")  # exterior CPML: nz=47, pad=16/16 -> interior idx 16..30 (last interior z~=20.99mm); 0.0225 rounds to idx 31, first exterior node (#500 L7)
    sim.add_probe((0.01, 0.01, 0.022), component="ez")
    report = sim.preflight()
    assert report, "expected a preflight finding for a source/probe in CPML"
    for issue in report:
        assert isinstance(issue, PreflightIssue)
        assert isinstance(issue, str)          # back-compat: still a string
        assert issue.severity in ("warning", "error")
        assert isinstance(issue.code, str) and issue.code
    # Back-compat operations the old list[str] supported still work.
    assert isinstance("\n".join(report), str)


def test_preflight_issue_is_a_real_string():
    pi = PreflightIssue("ERROR: x", severity="error", code="conformal_nan")
    assert pi == "ERROR: x" and pi.startswith("ERROR") and pi.severity == "error"


def test_preflight_report_is_a_list_with_canonical_api():
    """PreflightReport IS a list (back-compat) AND mirrors the in-repo report
    idiom (.issues/.errors/.warnings/.ok/.format()/.to_dict()/.to_json())."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.0225), component="ez")  # exterior CPML: nz=47, pad=16/16 -> interior idx 16..30 (last interior z~=20.99mm); 0.0225 rounds to idx 31, first exterior node (#500 L7)
    sim.add_probe((0.01, 0.01, 0.022), component="ez")
    report = sim.preflight()
    assert isinstance(report, PreflightReport) and isinstance(report, list)
    # list[str] ops the 65 legacy call sites rely on
    assert isinstance("\n".join(report), str)
    assert len(report) == len(list(report)) and bool(report)
    # canonical report API
    assert report.issues == list(report)
    assert report.ok == (not report.errors)
    assert set(report.warnings) | set(report.errors) == set(report)
    assert "preflight:" in report.format()


def test_codes_set_at_check_site():
    """Codes come from the checks themselves (PreflightWarning instance /
    PreflightConfigError), NOT from text inference of the message.

    Confirms the deleted ``_preflight_code_for`` is not relied on: a probe in
    CPML carries the absorber_overlap slug set in
    ``_validate_cfg_absorber_placement`` and the source object identifies the
    emitting check.
    """
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.0225), component="ez")  # exterior CPML: nz=47, pad=16/16 -> interior idx 16..30 (last interior z~=20.99mm); 0.0225 rounds to idx 31, first exterior node (#500 L7)
    sim.add_probe((0.01, 0.01, 0.022), component="ez")
    report = sim.preflight()
    absorber = report.by_code("absorber_overlap")
    assert absorber, f"expected absorber_overlap code, got {[i.code for i in report]}"
    for issue in absorber:
        assert issue.code == "absorber_overlap"
        assert issue.source == "_validate_cfg_absorber_placement"


def test_error_severity_mapping_end_to_end():
    """A PreflightErrorWarning emitted by any validator must surface as a
    severity='error' PreflightIssue (not masking other checks)."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim._validate_ntff_inverse_design = lambda **kw: warnings.warn(
        "forced known-bad config", PreflightErrorWarning
    )
    report = sim.preflight()
    errs = [i for i in report if i.severity == "error"]
    assert any("forced known-bad" in i for i in errs)


# --------------------------------------------------------- (b) conformal guard
def _fake_conformal(dx, dy=None, dz=None, faces=("z_lo", "z_hi")):
    spec = SimpleNamespace(conformal_faces=lambda: set(faces))
    return SimpleNamespace(_boundary_spec=spec, _dx=dx, _dy=dy, _dz=dz)


def test_conformal_fine_dx_warns():
    # WARNING severity (not error/forbid): conformal-fine-dx is a known,
    # development-coupled bug; convergence tests must still RUN it, so it must
    # not hard-fail. Agents gate on the code, not a hard-stop.
    fake = _fake_conformal(1e-3)
    with pytest.warns(UserWarning, match="KNOWN"):
        _PreflightMixin._validate_cfg_conformal_fine_dx(fake, 1e-3)


def test_conformal_coarse_dx_silent():
    fake = _fake_conformal(3e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _PreflightMixin._validate_cfg_conformal_fine_dx(fake, 3e-3)


def test_no_conformal_silent():
    fake = _fake_conformal(1e-3, faces=())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _PreflightMixin._validate_cfg_conformal_fine_dx(fake, 1e-3)


# --------------------------------------------------- (c) lossless-resonator
def _box_sim(material):
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add(Box((0.005,) * 3, (0.015,) * 3), material=material)
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.012), component="ez")
    return sim


def _lossless_warnings(sim):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _PreflightMixin._validate_cfg_lossless_resonator_in_absorber(sim, warnings)
    return [w for w in rec if "artificially" in str(w.message).lower()]


def test_lossless_dielectric_in_cpml_warns():
    # alumina: eps_r 9.8, sigma 0 (built-in, resolved via MATERIAL_LIBRARY).
    assert len(_lossless_warnings(_box_sim("alumina"))) == 1


def test_lossy_dielectric_silent():
    # fr4: sigma 0.025 => not the artificial-Q trap, no false positive.
    assert len(_lossless_warnings(_box_sim("fr4"))) == 0


# ------------------------------------------------ (d) Phase A meta-coverage
def _bad_sim_probe_in_cpml():
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.0225), component="ez")  # exterior CPML: nz=47, pad=16/16 -> interior idx 16..30 (last interior z~=20.99mm); 0.0225 rounds to idx 31, first exterior node (#500 L7)
    sim.add_probe((0.01, 0.01, 0.022), component="ez")
    return sim


def _bad_sim_no_sources():
    # no_sources advisory
    return Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="pec")


def _bad_sim_lossless_q():
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add(Box((0.005,) * 3, (0.015,) * 3), material="alumina")  # lossless_q
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.012), component="ez")
    return sim


def _bad_sim_under_resolved_dielectric():
    # mesh_resolution: a fat high-eps slab under-resolved at coarse dx.
    sim = Simulation(domain=(0.04, 0.04, 0.04), freq_max=20e9, dx=2e-3,
                     boundary="pec")
    sim.add_material("hi", eps_r=12.0)
    sim.add(Box((0.005, 0.005, 0.005), (0.035, 0.035, 0.035)), material="hi")
    sim.add_source((0.02, 0.02, 0.02), component="ez")
    sim.add_probe((0.025, 0.025, 0.025), component="ez")
    return sim


def _bad_sim_upml_refinement():
    # upml_refinement: structurally-impossible config (error severity).
    from rfx import GaussianPulse
    sim = Simulation(freq_max=6e9, domain=(0.04, 0.04, 0.02),
                     boundary="upml", cpml_layers=6, dx=0.002)
    sim.add_refinement((0.004, 0.008), ratio=2)
    sim.add_source((0.01, 0.02, 0.01), "ez",
                   waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.026, 0.02, 0.01), "ez")
    return sim


def _all_emitted_issues(report):
    return list(report)


def test_every_emitted_issue_carries_a_check_site_code():
    """Phase A meta-test: across a battery of deliberately-bad sims, EVERY
    emitted preflight issue must carry a non-empty code != 'uncoded' (codes are
    set at the check site, not inferred)."""
    builders = (
        _bad_sim_probe_in_cpml,
        _bad_sim_no_sources,
        _bad_sim_lossless_q,
        _bad_sim_under_resolved_dielectric,
        _bad_sim_upml_refinement,
    )
    total = 0
    for build in builders:
        report = build().preflight()
        for issue in _all_emitted_issues(report):
            total += 1
            assert isinstance(issue, PreflightIssue)
            assert issue.code and issue.code != "uncoded", (
                f"{build.__name__}: uncoded issue {str(issue)!r}"
            )
            assert issue.severity in ("warning", "error")
    assert total > 0, "battery emitted no issues — meta-test is vacuous"


def test_error_severity_config_issue_is_coded():
    """The structurally-impossible upml+refinement raise surfaces as an
    error-severity issue with its check-site slug (not 'uncoded')."""
    report = _bad_sim_upml_refinement().preflight()
    errs = report.errors
    assert errs, "expected an error-severity issue for upml+refinement"
    assert any(i.code == "upml_refinement" for i in errs), (
        f"codes: {[i.code for i in errs]}"
    )
    assert not report.ok


def test_to_dict_and_to_json_roundtrip_carry_code_and_severity():
    """Real serialization: to_dict()/to_json() carry code + severity per issue
    (the str subclass alone is NOT json-dumpable with its attrs)."""
    import json

    report = _bad_sim_probe_in_cpml().preflight()
    assert report, "expected at least one issue to serialize"
    d = report.to_dict()
    assert d["n_issues"] == len(report)
    for src, rec in zip(report, d["issues"]):
        assert rec["code"] == src.code
        assert rec["severity"] == src.severity
        assert rec["message"] == str(src)
    back = json.loads(report.to_json())
    assert back["issues"][0]["code"] == report[0].code
    assert back["issues"][0]["severity"] == report[0].severity


# ------------------------------------------------ Phase C-full: aggregate strict
def test_strict_aggregates_all_issues_in_one_raise():
    """strict=True escalates EVERY finding in ONE ValueError (aggregate-then-
    raise), not fail-on-first — preserving the historical 'strict escalates any
    issue' contract while reporting all problems at once."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.0225), component="ez")  # exterior CPML: nz=47, pad=16/16 -> interior idx 16..30 (last interior z~=20.99mm); 0.0225 rounds to idx 31, first exterior node (#500 L7)
    sim.add_probe((0.01, 0.01, 0.022), component="ez")     # in exterior CPML (#500)
    # strict=False shows >= 2 findings...
    report = sim.preflight()
    assert len(report) >= 2, f"need a multi-issue config; got {list(report)}"
    # ...and strict raises ONCE listing all of them.
    with pytest.raises(ValueError) as exc:
        sim.preflight(strict=True)
    text = str(exc.value)
    assert text.count("\n  - ") >= 2, f"expected aggregated list, got: {text}"


def test_raise_for_failure_is_errors_only_gate():
    """report.raise_for_failure() is the SOFTER pre-launch gate: it raises only
    on error-severity, letting advisory warnings through (unlike strict=True)."""
    report = _bad_sim_probe_in_cpml().preflight()   # warnings only, no errors
    assert report and report.ok          # ok == no error-severity issues
    report.raise_for_failure()           # must NOT raise on warning-only report


# ------------------------------------------------ Phase D: validator crash is loud
def test_validator_crash_propagates_not_swallowed():
    """A validator raising a NON-ValueError is a bug, not a finding — it must
    propagate (loud), not degrade to a soft advisory that hides it."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.012), component="ez")

    def _boom():
        raise RuntimeError("validator bug")
    # Target a validator run() actually invokes (run() uses check_ntff=False, so
    # the NTFF inverse-design check is NOT on its path).
    sim._validate_simulation_config = _boom

    # Via the auto-preflight path (run) the bug must surface, not be swallowed.
    with pytest.raises(RuntimeError, match="validator bug"):
        sim.run(n_steps=5)


# ----------------------------------------- run() error-severity + NTFF surface
def test_run_uses_ntff_advisory_tier_but_forward_gets_the_error():
    """run() must NOT hard-fail on the NTFF PEC-overlap error, while
    forward()/optimize (the inverse-design entry points) still do.

    Issue #303 changed the MECHANISM (run() now uses check_ntff="advisory"
    — the validator IS invoked, with include_pec_overlap_error=False — so
    λ/4 advisories reach run() users) but the CONTRACT locked here is
    unchanged: the error tier stays off run() and on forward(). The mock
    respects the kwarg the way the real validator does."""
    def _sim():
        s = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
        s.add_source((0.01, 0.01, 0.01), component="ez")
        s.add_probe((0.01, 0.01, 0.012), component="ez")

        # Stand in for an NTFF-box-crosses-PEC error-severity finding that
        # honors the error-tier switch like the real validator.
        def _fake_ntff(*, include_pec_overlap_error: bool = True):
            if include_pec_overlap_error:
                raise ValueError("NTFF box face crosses PEC")

        s._validate_ntff_inverse_design = _fake_ntff
        return s

    # run(): advisory tier -> validator invoked WITHOUT the error tier
    _sim().run(n_steps=5)
    # forward(): full tier -> error-severity -> re-raised
    with pytest.raises(ValueError, match="NTFF box face crosses PEC"):
        _sim().forward(n_steps=5)


def test_run_hard_fails_on_error_severity_and_skip_bypasses():
    """run() re-raises on a structurally-impossible (error-severity) config, and
    skip_preflight=True is the documented escape hatch."""
    sim = Simulation(domain=(0.02,) * 3, freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.012), component="ez")
    sim._validate_simulation_config = lambda: (_ for _ in ()).throw(
        ValueError("structurally impossible config")
    )
    with pytest.raises(ValueError, match="structurally impossible config"):
        sim.run(n_steps=5)
    # escape hatch: skip_preflight bypasses the preflight (and its re-raise)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")  # no preflight warning/raise should fire
        sim.run(n_steps=5, skip_preflight=True)


# ------------------------------------------------- issue #166: 2D z + units
def test_absorber_overlap_no_false_positive_on_2d_collapsed_z():
    """2D modes collapse z to a single cell with NO absorber (Grid strips z
    from cpml_axes and sets pad_z=0). Every 2D source/probe necessarily sits
    at z=0, so the z-axis proximity check must not fire (issue #166: cv03
    emitted one absorber_overlap line per line-source point)."""
    sim = Simulation(domain=(16e-6, 9e-6, 1e-7), freq_max=7.5e13, dx=1e-7,
                     boundary="upml", cpml_layers=20, mode="2d_tmz")
    # mid-domain in x and y — only the collapsed z coordinate (0) is "near"
    # the phantom z absorber the old mirror logic assumed.
    sim.add_source((8e-6, 4.5e-6, 0), component="ez")
    sim.add_probe((9e-6, 4.5e-6, 0), component="ez")
    report = sim.preflight()
    overlap = report.by_code("absorber_overlap")
    assert not overlap, f"false positive on collapsed z axis: {list(overlap)}"


def test_absorber_overlap_still_fires_on_2d_xy():
    """The 2D-z exemption must not silence real x/y absorber overlap.

    Issue #500: CPML pads EXTERIOR to the requested domain (see
    rfx-known-issues.md #500), so a source has to sit at a genuinely
    negative x (past the x=0 domain edge, inside the 2e-6m-thick x_lo
    absorber: cpml_layers=20 * dx=1e-7) to be "in" it — x=1e-7 (interior,
    just past the edge) used to false-fire under the pre-#500
    interior-frame bug but is not actually in the absorber.
    """
    sim = Simulation(domain=(16e-6, 9e-6, 1e-7), freq_max=7.5e13, dx=1e-7,
                     boundary="upml", cpml_layers=20, mode="2d_tmz")
    sim.add_source((-5e-7, 4.5e-6, 0), component="ez")   # genuinely in x_lo absorber
    report = sim.preflight()
    overlap = report.by_code("absorber_overlap")
    assert overlap, "expected absorber_overlap for a source in the x_lo absorber"
    assert any("x-thickness" in str(i) for i in overlap)


def test_unit_adaptive_formatting_helpers():
    """_fmt_len/_fmt_freq pick units that keep digits visible at any scale
    (issue #166: fixed mm/GHz rendered 0.1µm as 0.000mm and 74.95THz as
    74950.00GHz)."""
    from rfx.api._preflight import _fmt_len, _fmt_freq
    assert _fmt_len(1e-7) == "100nm"
    assert _fmt_len(2e-6) == "2µm"
    assert _fmt_len(0.002) == "2mm"
    assert _fmt_len(0.02286) == "22.86mm"
    assert _fmt_len(1.5) == "1.5m"
    assert _fmt_len(5e-10) == "0.5nm"
    assert _fmt_len(0.0) == "0mm"
    assert _fmt_freq(7.495e13) == "74.95THz"
    assert _fmt_freq(10e9) == "10GHz"
    assert _fmt_freq(9.322e9) == "9.322GHz"
    assert _fmt_freq(2.45e6) == "2.45MHz"


def test_mesh_warning_uses_adaptive_units_at_optical_scale():
    """The cv03-class mesh-resolution warning must print THz/µm, not
    0.000mm / five-digit GHz, at optical scale."""
    sim = Simulation(domain=(16e-6, 9e-6, 1e-7), freq_max=7.495e13, dx=1e-7,
                     boundary="upml", cpml_layers=20, mode="2d_tmz")
    sim.add_material("wg", eps_r=12.0)
    sim.add(Box((0, 4e-6, 0), (16e-6, 5e-6, 1e-7)), material="wg")
    sim.add_source((8e-6, 4.5e-6, 0), component="ez")
    report = sim.preflight()
    mesh = [str(i) for i in report.by_code("mesh_resolution")
            if "cells per λ_eff" in str(i)]
    assert mesh, "expected the cells-per-λ_eff warning at 11.5 cells/λ_eff"
    assert any("74.95THz" in m for m in mesh), mesh
    assert any("100nm" in m for m in mesh), mesh
    assert not any("0.000mm" in m for m in mesh), mesh


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_guards.py
# ===========================================================================

_CODE = "tfsf_lumped_rlc_unstable"


def test_tfsf_plus_lumped_rlc_warns():
    sim = Simulation(freq_max=16e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 20,
                     boundary="cpml", cpml_layers=8, mode="3d")
    sim.add_tfsf_source(f0=8e9, bandwidth=0.6, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    sim.add_lumped_rlc(position=(0.010, 0.010, 0.010), component="ez",
                       R=50.0, C=0.20e-12, topology="series")
    assert _CODE in _codes(sim), "TFSF + lumped RLC should warn about the unstable pairing"


def test_tfsf_alone_no_warning():
    """No false positive: a TFSF plane wave with no lumped element must NOT warn."""
    sim = Simulation(freq_max=16e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 20,
                     boundary="cpml", cpml_layers=8, mode="3d")
    sim.add_tfsf_source(f0=8e9, bandwidth=0.6, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    assert _CODE not in _codes(sim)


def test_lumped_rlc_with_port_no_warning():
    """No false positive: the validated PORT-fed varactor lane must NOT warn."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 15,
                     boundary="cpml", cpml_layers=6)
    sim.add_port(position=(0.0093, 0.0093, 0.0093), component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=0.9))
    sim.add_lumped_rlc(position=(0.0093, 0.0093, 0.0093), component="ez",
                       R=50.0, C=0.20e-12, topology="series")
    assert _CODE not in _codes(sim)
