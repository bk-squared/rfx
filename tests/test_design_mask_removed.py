"""Issue #625: design_mask was REMOVED from every public surface, not
deprecated.

PI decision (2026-08-11): the masking mechanism (`jnp.where(mask, x,
stop_gradient(x))` applied to field state each step) was measured to save
ZERO reverse-mode AD memory (partial-eval residuals have whole-array
granularity, so confining the design variable to a fraction of cells does
not shrink the tape) while CORRUPTING the gradient in every configuration
tested (an observable outside the design region reads back exactly 0.0; one
co-located with the mask read back 52.8-69.9% error against a reference
matching central finite differences to 1e-5). The masked gradient was a
function of mask geometry alone, not a derivative of anything. See
CHANGELOG.md for the full write-up and `docs/agent-memory/` for the durable
record.

This module is the falsifier the removal itself asks for: every public
entry point that used to accept `design_mask=` must now reject it with a
plain `TypeError` (unrecognised keyword argument) so a future
reintroduction is a deliberate, reviewable act rather than a silent
re-add. Do not weaken these to `pytest.raises(Exception)` — the point is
that Python's own signature machinery is the only guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation


def _build_uniform_sim() -> Simulation:
    nx, ny, nz, dx = 12, 10, 10, 5e-3
    sim = Simulation(
        freq_max=5e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        boundary="pec",
    )
    sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
    sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
    return sim


def _build_nonuniform_sim() -> Simulation:
    nx, ny, nz, dx = 12, 10, 10, 5e-3
    sim = Simulation(
        freq_max=5e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        boundary="pec",
        dx_profile=np.full(nx, dx),
    )
    sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
    sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
    return sim


def test_forward_rejects_design_mask_kwarg():
    """The uniform-lane fence used to be a NotImplementedError; there is no
    longer a fence to raise it, because there is no longer a parameter."""
    sim = _build_uniform_sim()
    mask = np.zeros(sim._build_grid().shape, dtype=bool)
    with pytest.raises(TypeError, match="design_mask"):
        sim.forward(n_steps=2, design_mask=mask, skip_preflight=True)


def test_forward_nonuniform_rejects_design_mask_kwarg():
    """The non-uniform lane used to be design_mask's only SUPPORTED lane;
    it must now reject the kwarg exactly like the uniform lane."""
    sim = _build_nonuniform_sim()
    grid = sim._build_nonuniform_grid()
    mask = np.zeros(grid.shape, dtype=bool)
    with pytest.raises(TypeError, match="design_mask"):
        sim.forward(n_steps=2, design_mask=mask, skip_preflight=True)


def test_optimize_rejects_design_mask_kwarg():
    from rfx.optimize import DesignRegion, optimize

    sim = _build_nonuniform_sim()
    region = DesignRegion(
        corner_lo=(0.01, 0.01, 0.01),
        corner_hi=(0.02, 0.02, 0.02),
        eps_range=(1.0, 4.0),
    )
    mask = np.zeros(sim._build_nonuniform_grid().shape, dtype=bool)

    def obj(result):
        return (result.time_series[:, 0] ** 2).sum()

    with pytest.raises(TypeError, match="design_mask"):
        optimize(
            sim, region, obj,
            n_iters=1, n_steps=4, verbose=False, skip_preflight=True,
            design_mask=mask,
        )


def test_estimate_ad_memory_rejects_design_mask_kwarg():
    sim = _build_uniform_sim()
    with pytest.raises(TypeError, match="design_mask"):
        sim.estimate_ad_memory(n_steps=100, design_mask=np.zeros((2, 2, 2), dtype=bool))


def test_plan_ad_memory_rejects_design_mask_kwarg():
    sim = _build_uniform_sim()
    with pytest.raises(TypeError, match="design_mask"):
        sim.plan_ad_memory(
            n_steps=100, available_memory_gb=1.0,
            design_mask=np.zeros((2, 2, 2), dtype=bool),
        )


def test_mesh_intelligence_report_rejects_design_mask_kwarg():
    sim = _build_uniform_sim()
    with pytest.raises(TypeError, match="design_mask"):
        sim.mesh_intelligence_report(
            n_steps=100, design_mask=np.zeros((2, 2, 2), dtype=bool),
        )


def test_ad_memory_estimate_has_no_active_design_fraction_field():
    """The reporting field that existed only to surface design_mask's
    active-cell fraction (`ad_active_design_fraction`) goes with the
    parameter -- it could never be anything but 1.0 or None once
    design_mask cannot be constructed."""
    sim = _build_uniform_sim()
    estimate = sim.estimate_ad_memory(n_steps=100)
    assert not hasattr(estimate, "ad_active_design_fraction")
    assert "ad_active_design_fraction" not in estimate.to_dict()
