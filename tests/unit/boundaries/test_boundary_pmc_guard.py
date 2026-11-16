"""PMC boundary specifications and grid face propagation.

Finite, nonzero runtime traces are covered in test_boundary_pmc_runtime.py.
"""

from __future__ import annotations

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


    def test_grid_pmc_faces_set_from_boundary_spec(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="cpml"),
        )
        sim = self._sim(spec)
        grid = sim._build_grid()
        assert grid.pmc_faces == {"z_lo"}
