"""N-1 guard and introspection tests for the guarded subgrid core.

This branch carries ``add_refinement`` and ``validate_subgrid`` plus the
pre-existing ``overlap_z_slab`` SBP-SAT runner inherited from main. The
``stage2_disjoint_3d`` centered/two-interface runner is NOT in this branch,
so ``run()`` on a refinement with ``topology="stage2_disjoint_3d"`` must fail
loudly with ``NotImplementedError`` rather than silently dispatching to the
wrong (overlap) runner. Refined runs on the default ``overlap_z_slab``
topology dispatch normally and must NOT trip the guard.
"""

from __future__ import annotations

import inspect

import pytest

from rfx import GaussianPulse, Simulation


def _refined_simulation(topology: str = "overlap_z_slab") -> Simulation:
    """Return a small PEC ``Simulation`` with one z-slab refinement."""
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    sim.add_refinement((0.004, 0.012), ratio=4, topology=topology)
    sim.add_source(
        (0.02, 0.02, 0.008),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.02, 0.02, 0.01), "ez")
    return sim


def test_run_on_disjoint_topology_raises_not_implemented():
    """run() with topology='stage2_disjoint_3d' raises NotImplementedError."""
    sim = _refined_simulation(topology="stage2_disjoint_3d")
    with pytest.raises(NotImplementedError, match="stage2_disjoint_3d"):
        sim.run(n_steps=20, compute_s_params=False)


def test_run_on_overlap_topology_does_not_trip_the_n1_guard():
    """A refined run on the inherited overlap_z_slab runner must not trip N-1."""
    sim = _refined_simulation(topology="overlap_z_slab")
    try:
        sim.run(n_steps=2, compute_s_params=False)
    except NotImplementedError as exc:  # pragma: no cover - guard regression
        if "stage2_disjoint_3d" in str(exc):
            pytest.fail("N-1 guard wrongly fired for the overlap_z_slab topology")
        raise


def test_add_refinement_signature_matches_documented_kwargs():
    """add_refinement exposes exactly the documented parameter set."""
    params = set(inspect.signature(Simulation.add_refinement).parameters)
    params.discard("self")
    assert params == {"z_range", "ratio", "xy_margin", "tau", "validation", "topology"}
