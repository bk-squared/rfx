"""T7-C Phase 1: per-face CPML thickness spec + runtime guard.

The ``Boundary`` dataclass carries ``lo_thickness`` / ``hi_thickness``
attributes; non-absorbing faces reject thickness at construction time.
The ``Simulation.__init__`` guard raises ``NotImplementedError`` when
the spec asks for asymmetric thickness that the existing CPMLAxisParams
engine cannot honour. The CPMLAxisParams refactor (Phase 2) lands in a
follow-up.
"""

from __future__ import annotations

import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


class TestBoundaryThicknessAttributes:
    def test_symmetric_thickness_accepted(self):
        b = Boundary(lo="cpml", hi="cpml", lo_thickness=8, hi_thickness=8)
        assert b.lo_thickness == 8
        assert b.hi_thickness == 8

    def test_default_thickness_is_none(self):
        b = Boundary(lo="cpml", hi="cpml")
        assert b.lo_thickness is None
        assert b.hi_thickness is None

    def test_resolved_thickness_falls_back_to_default(self):
        b = Boundary(lo="cpml", hi="cpml", lo_thickness=12)
        assert b.resolved_lo_thickness(default=8) == 12
        assert b.resolved_hi_thickness(default=8) == 8

    def test_thickness_on_pec_face_raises(self):
        with pytest.raises(ValueError, match="absorbing faces"):
            Boundary(lo="pec", hi="cpml", lo_thickness=4)
        with pytest.raises(ValueError, match="absorbing faces"):
            Boundary(lo="cpml", hi="pec", hi_thickness=4)

    def test_thickness_on_pmc_face_raises(self):
        with pytest.raises(ValueError, match="absorbing faces"):
            Boundary(lo="pmc", hi="cpml", lo_thickness=4)

    def test_thickness_on_periodic_face_raises(self):
        with pytest.raises(ValueError, match="absorbing faces"):
            # Periodic must be symmetric anyway, but thickness must also
            # reject it. The periodic-mismatch check fires first if we
            # try to put thickness on a single face, so test with
            # symmetric periodic.
            Boundary(lo="periodic", hi="periodic", lo_thickness=4)

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Boundary(lo="cpml", hi="cpml", lo_thickness=-1)

    def test_dict_round_trip_preserves_thickness(self):
        b = Boundary(lo="cpml", hi="cpml", lo_thickness=12, hi_thickness=4)
        assert Boundary.from_dict(b.to_dict()) == b

    def test_dict_round_trip_omits_none_thickness(self):
        b = Boundary(lo="cpml", hi="cpml")
        d = b.to_dict()
        assert "lo_thickness" not in d and "hi_thickness" not in d


class TestSimulationThicknessPhase1Guard:
    def test_uniform_scalar_thickness_works(self):
        """Legacy cpml_layers=N with no per-face thickness: no error."""
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            cpml_layers=8,
        )
        assert sim._boundary_spec.absorber_type == "cpml"

    def test_symmetric_per_axis_thickness_matches_default_works(self):
        """Per-face thickness equal to the default scalar: passes and
        the resolved thickness everywhere is the scalar."""
        spec = BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml", lo_thickness=8, hi_thickness=8),
            y="cpml",
            z="cpml",
        )
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            cpml_layers=8, boundary=spec,
        )
        # spec carries the explicit thickness
        assert sim._boundary_spec.x.lo_thickness == 8
        assert sim._boundary_spec.x.hi_thickness == 8
        # and the runtime scalar (engine authority in Phase 1) is still 8
        assert sim._cpml_layers == 8

    def test_asymmetric_per_face_thickness_raises(self):
        spec = BoundarySpec(
            x="cpml",
            y="cpml",
            z=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=16),
        )
        with pytest.raises(NotImplementedError, match="asymmetric per-face"):
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                cpml_layers=8, boundary=spec,
            )

    def test_asymmetric_per_axis_thickness_raises(self):
        """Different thickness between two axes (all symmetric per-axis)."""
        spec = BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml", lo_thickness=8, hi_thickness=8),
            y=Boundary(lo="cpml", hi="cpml", lo_thickness=16, hi_thickness=16),
            z="cpml",
        )
        with pytest.raises(NotImplementedError, match="asymmetric per-face"):
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                cpml_layers=8, boundary=spec,
            )

    def test_pec_face_with_cpml_axis_no_thickness_conflict(self):
        """PEC face doesn't carry thickness and must not trigger the guard."""
        spec = BoundarySpec(
            x="cpml",
            y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        )
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            cpml_layers=8, boundary=spec,
        )
        assert "z_lo" in sim._pec_faces
