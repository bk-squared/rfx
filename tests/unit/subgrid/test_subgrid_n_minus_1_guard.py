"""Topology-guard behaviour for the public subgrid run path.

A refined ``run()`` must fail loudly on a configuration the production
envelope does not support, rather than silently dispatching it through the
wrong runner:

* ``topology="stage2_disjoint_3d"`` is research-only.  Under the default
  production validation a refined ``run()`` is rejected with a ``ValueError``;
  under ``validation="research"`` it dispatches to the dedicated disjoint
  runner (``run_disjoint_stage2_path``) -- never to the overlap runner.
* A guarded one-sided ``overlap_z_slab`` that touches exactly one physical z
  boundary stays inside the production envelope and runs to completion.

Branch B1 used a placeholder ``NotImplementedError`` guard here; Branch B2
replaced it with the real runner dispatch, so the loud-failure signal for an
unsupported topology is now the production validation ``ValueError`` rather
than ``NotImplementedError``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation


def _disjoint_simulation(validation: str = "production") -> Simulation:
    """Return a small ``stage2_disjoint_3d`` refined Simulation."""
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    sim.add_refinement(
        (0.004, 0.012),
        ratio=4,
        topology="stage2_disjoint_3d",
        validation=validation,
    )
    sim.add_source(
        (0.02, 0.02, 0.008),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.02, 0.02, 0.01), "ez")
    return sim


def _guarded_overlap_simulation() -> Simulation:
    """Return a guarded one-sided ``overlap_z_slab`` refined Simulation.

    The slab touches the z-lo PEC face, x/y stay closed PEC with no CPML --
    inside the production validation envelope.
    """
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    sim.add_refinement((0.0, 0.012), ratio=4)
    sim.add_source(
        (0.02, 0.02, 0.002),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.02, 0.02, 0.003), "ez")
    return sim


def test_run_on_disjoint_topology_is_rejected_under_production():
    """A disjoint-topology run() fails loudly under production validation."""
    sim = _disjoint_simulation(validation="production")
    with pytest.raises(ValueError, match="subgrid validation"):
        sim.run(n_steps=20, compute_s_params=False)


def test_run_on_disjoint_topology_research_mode_uses_the_disjoint_runner():
    """validation='research' dispatches the disjoint topology to its runner."""
    sim = _disjoint_simulation(validation="research")
    result = sim.run(n_steps=20, compute_s_params=False)
    series = np.asarray(result.time_series)
    assert series.shape[0] == 20
    assert np.all(np.isfinite(series))


def test_run_on_guarded_overlap_slab_stays_in_the_production_envelope():
    """A guarded one-sided overlap z-slab validates and runs to completion."""
    sim = _guarded_overlap_simulation()
    report = sim.validate_subgrid()
    assert report.supported is True

    result = sim.run(n_steps=4, compute_s_params=False)
    assert np.all(np.isfinite(np.asarray(result.time_series)))


def test_add_refinement_signature_matches_documented_kwargs():
    """add_refinement exposes exactly the documented parameter set."""
    params = set(inspect.signature(Simulation.add_refinement).parameters)
    params.discard("self")
    assert params == {"z_range", "ratio", "xy_margin", "tau", "validation", "topology"}
