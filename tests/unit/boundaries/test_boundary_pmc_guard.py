"""T7-E runtime tests (T7 Phase 2, 2026-04).

Phase 1 (T7-C/D/E) rejected ``'pmc'`` tokens at Simulation construction
via ``_check_pmc_phase1``. Phase 2 PR3 ships
``rfx/boundaries/pmc.py::apply_pmc_faces`` wired into the uniform scan
body; the guard is removed. These tests now assert the runtime
accepts PMC faces and produces finite output (the module-level
physics tests live in ``test_boundary_pmc_runtime.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


class TestSpecOnly:
    def test_spec_accepts_pmc_on_single_face(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="cpml"),
        )
        assert spec.pmc_faces() == {"z_lo"}

    def test_spec_accepts_pmc_on_both_faces_same_axis(self):
        spec = BoundarySpec(x="cpml", y="cpml", z="pmc")
        assert spec.pmc_faces() == {"z_lo", "z_hi"}

    def test_spec_accepts_pmc_mixed_with_pec(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="pec"),
        )
        assert spec.pmc_faces() == {"z_lo"}
        assert spec.pec_faces() == {"z_hi"}


class TestSimulationAcceptsPMC:
    def _sim(self, spec):
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
            boundary=spec,
        )
        sim.add_source((0.005, 0.005, 0.005), "ez")
        sim.add_probe((0.005, 0.005, 0.006), "ez")
        return sim

    def test_single_pmc_face_builds_and_runs(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="cpml"),
        )
        sim = self._sim(spec)
        ts = np.asarray(sim.run(n_steps=20).time_series)
        assert ts.shape[0] == 20
        assert np.all(np.isfinite(ts))

    def test_both_pmc_faces_build_and_run(self):
        spec = BoundarySpec(x="cpml", y="cpml", z="pmc")
        sim = self._sim(spec)
        ts = np.asarray(sim.run(n_steps=20).time_series)
        assert ts.shape[0] == 20
        assert np.all(np.isfinite(ts))

    def test_pmc_plus_pec_builds_and_runs(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="pec"),
        )
        sim = self._sim(spec)
        ts = np.asarray(sim.run(n_steps=20).time_series)
        assert ts.shape[0] == 20
        assert np.all(np.isfinite(ts))

    def test_grid_pmc_faces_set_from_boundary_spec(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="cpml"),
        )
        sim = self._sim(spec)
        grid = sim._build_grid()
        assert grid.pmc_faces == {"z_lo"}
