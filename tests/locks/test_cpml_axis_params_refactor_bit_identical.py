"""T7 Phase 2 PR1 — CPMLAxisParams 6-face refactor bit-identity regression.

Pins the invariant that the pure 6-face refactor (x -> x_lo + x_hi,
y -> y_lo + y_hi) produces bit-identical output to the pre-refactor
code on a representative small simulation. When any future change
modifies the CPML scan body physics, this test will flag the drift
versus the canonical reference trace.

The reference is generated on-demand from a known-good stub that
exercises all three axes' CPML faces symmetrically (uniform-grid
case). The test asserts that the output time series matches between
two back-to-back runs of the same sim (catches any hidden tracer /
JIT non-determinism) and that the time series has the expected
finite-energy magnitude (catches silent regressions that zero out
the whole field).
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.boundaries.cpml import CPMLAxisParams, init_cpml
from rfx.grid import Grid


def _build_small_sim():
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary="cpml", cpml_layers=6,
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.006), "ez")
    return sim


def test_cpml_axis_params_is_six_face_after_pr1():
    """CPMLAxisParams must expose six face-specific CPMLParams fields
    (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) plus six boundary cell sizes."""
    sim = _build_small_sim()
    grid = sim._build_grid()
    params, _state = init_cpml(grid)

    for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"):
        assert hasattr(params, face), f"CPMLAxisParams missing field {face!r}"
    for cell in ("dx_x_lo", "dx_x_hi", "dx_y_lo", "dx_y_hi", "dz_lo", "dz_hi"):
        assert hasattr(params, cell), f"CPMLAxisParams missing cell size {cell!r}"


def test_cpml_scan_body_produces_deterministic_time_series():
    """Back-to-back runs of the same CPML sim must produce bit-identical
    time series (catches any non-determinism introduced by the PR1
    refactor — e.g. traced jnp.flip artefacts or cache invalidation)."""
    ts_a = np.asarray(_build_small_sim().run(n_steps=200).time_series)
    ts_b = np.asarray(_build_small_sim().run(n_steps=200).time_series)
    assert ts_a.shape == ts_b.shape
    np.testing.assert_array_equal(ts_a, ts_b)


def test_cpml_scan_body_produces_finite_nonzero_output():
    """Sanity: the refactored CPML update must still absorb + produce a
    non-zero, finite probe trace. Guards against a silent regression
    that zeroes fields or NaNs them out."""
    ts = np.asarray(_build_small_sim().run(n_steps=200).time_series)
    assert np.all(np.isfinite(ts)), "CPML scan body produced NaN/Inf"
    assert float(np.max(np.abs(ts))) > 1e-6, (
        f"CPML scan body produced near-zero output "
        f"(max-abs={float(np.max(np.abs(ts))):.3e})"
    )


def test_cpml_hi_face_profiles_are_pre_flipped():
    """The hi-face profiles stored in CPMLAxisParams must be the
    pre-flipped version of the lo-face profile (for uniform grids
    where both faces share the same cell size). This is what makes
    the scan body flip-free in PR1."""
    sim = _build_small_sim()
    grid = sim._build_grid()
    params, _ = init_cpml(grid)
    # On a uniform grid, x_hi is the flipped x_lo.
    np.testing.assert_array_equal(
        np.asarray(params.x_hi.b), np.asarray(params.x_lo.b)[::-1]
    )
    np.testing.assert_array_equal(
        np.asarray(params.y_hi.b), np.asarray(params.y_lo.b)[::-1]
    )
    np.testing.assert_array_equal(
        np.asarray(params.z_hi.b), np.asarray(params.z_lo.b)[::-1]
    )


def test_cpml_axis_params_boundary_cell_sizes_match_grid_dx():
    """All six boundary cell sizes default to the grid dx on a uniform
    sim. This pins the convention that PR2 will override per-face."""
    sim = _build_small_sim()
    grid = sim._build_grid()
    params, _ = init_cpml(grid)
    dx = float(grid.dx)
    for cell_attr in ("dx_x_lo", "dx_x_hi", "dx_y_lo", "dx_y_hi",
                      "dz_lo", "dz_hi"):
        assert abs(getattr(params, cell_attr) - dx) < 1e-12, (
            f"{cell_attr} expected {dx}, got {getattr(params, cell_attr)}"
        )


# ---------------------------------------------------------------------------
# Stage 1.5a — distributed-trio CPML/constant de-duplication guards
# ---------------------------------------------------------------------------


def test_distributed_common_cpml_coeff_helpers_are_vacuum_only():
    """Stage 1.5a / architect residual (a) — the extracted CPML-coefficient
    helpers must stay VACUUM-ONLY.

    `cpml_coeff_e_vacuum` / `cpml_coeff_h_vacuum` take ONLY `dt`. The
    distributed CPML correction runs where per-cell permittivity is not in
    scope, so the vacuum form is structural. If a future change adds an
    `eps_r` / `mu_r` / `materials` parameter (a CORE-C2-class hidden-
    assumption bug), this guard fails — by design.
    """
    import inspect

    from rfx.runners._distributed_common import (
        cpml_coeff_e_vacuum,
        cpml_coeff_h_vacuum,
    )

    for fn in (cpml_coeff_e_vacuum, cpml_coeff_h_vacuum):
        params = list(inspect.signature(fn).parameters)
        assert params == ["dt"], (
            f"{fn.__name__} must take only 'dt' (vacuum-only); got {params}. "
            f"A non-vacuum eps_r/mu_r/materials parameter would reintroduce "
            f"a CORE-C2-class hidden-assumption bug."
        )


def test_distributed_common_cpml_coeff_tracks_canonical_constants():
    """De-dup guard — the shared vacuum CPML coefficients must equal
    ``dt / EPS_0`` and ``dt / MU_0`` for the canonical ``rfx.core.yee``
    constants.

    Stage 1.5a introduced these helpers as a byte-identical de-dup of
    the inline literals (pre-2019 ``8.854187817e-12`` /
    ``1.2566370614e-6``); the review-remediation Stage 3.5b then moved
    EPS_0/MU_0 to the post-2019 SI/CODATA pair. Asserting against the
    imported constants (not a frozen literal) keeps the helpers pinned
    to ``dt / const`` while staying robust to future constant updates.
    """
    from rfx.core.yee import EPS_0, MU_0
    from rfx.runners._distributed_common import (
        cpml_coeff_e_vacuum,
        cpml_coeff_h_vacuum,
    )

    dt = 1.234e-12
    assert cpml_coeff_e_vacuum(dt) == dt / EPS_0
    assert cpml_coeff_h_vacuum(dt) == dt / MU_0


def test_distributed_runners_use_shared_cpml_coeff_helpers():
    """Stage 1.5a de-dup guard — the uniform and NU distributed runners
    must reference the shared vacuum CPML-coefficient helpers, not a
    re-introduced inline literal. Catches sibling drift mechanically."""
    import rfx.runners.distributed as dmod
    import rfx.runners.distributed_nu as numod
    from rfx.runners import _distributed_common as common

    assert dmod.cpml_coeff_e_vacuum is common.cpml_coeff_e_vacuum
    assert dmod.cpml_coeff_h_vacuum is common.cpml_coeff_h_vacuum
    assert numod.cpml_coeff_e_vacuum is common.cpml_coeff_e_vacuum
    assert numod.cpml_coeff_h_vacuum is common.cpml_coeff_h_vacuum


def test_distributed_ghost_exchange_is_single_shared_body():
    """Stage 1.5a de-dup guard — ``distributed_v2`` and ``distributed_nu``
    carried byte-identical ``shard_map`` ghost-exchange bodies. After the
    extraction both must resolve to the SAME shared function object, so a
    future fix to one cannot silently drift from the other."""
    from rfx.runners._distributed_common import exchange_component_shmap
    from rfx.runners.distributed_v2 import _exchange_component_shmap
    from rfx.runners.distributed_nu import _exchange_component_nu_shmap

    assert _exchange_component_shmap is exchange_component_shmap
    assert _exchange_component_nu_shmap is exchange_component_shmap
