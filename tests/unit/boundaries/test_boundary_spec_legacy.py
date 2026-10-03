"""BoundarySpec declarations and retained scalar-boundary compatibility.

Removed pec_faces calls must name the BoundarySpec replacement.
"""

from __future__ import annotations

import warnings

import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


# ---------------------------------------------------------------------------
# Explicit BoundarySpec path
# ---------------------------------------------------------------------------

class TestExplicitBoundarySpec:
    def test_boundary_spec_passed_directly(self):
        spec = BoundarySpec(x="cpml", y="periodic",
                            z=Boundary(lo="pec", hi="cpml"))
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            boundary=spec,
        )
        assert sim._boundary_spec == spec
        # Derived legacy views stay in sync.
        assert sim._periodic_axes == "y"
        assert "z_lo" in sim._pec_faces

    def test_boundary_spec_plus_pec_faces_conflict(self):
        spec = BoundarySpec.uniform("cpml")
        with pytest.raises(TypeError, match="BoundarySpec PEC faces"):
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                boundary=spec, pec_faces={"z_lo"},
            )

    def test_boundary_spec_no_deprecation_warning(self):
        spec = BoundarySpec.uniform("cpml")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                boundary=spec,
            )


# ---------------------------------------------------------------------------
# Legacy scalar path
# ---------------------------------------------------------------------------

class TestLegacyScalarPath:
    def test_scalar_cpml_round_trip(self):
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            boundary="cpml",
        )
        assert sim._boundary_spec == BoundarySpec.uniform("cpml")

    def test_scalar_cpml_no_warning_for_simple_use(self):
        """Scalar boundary shorthand remains supported without deprecation."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                boundary="cpml",
            )

    def test_scalar_pec_round_trip(self):
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            boundary="pec", cpml_layers=0,
        )
        assert sim._boundary_spec == BoundarySpec.uniform("pec")
