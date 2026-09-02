"""T7-B: legacy-API → ``BoundarySpec`` normalization shim.

Covers the round-trip between the scalar ``boundary=`` + ``pec_faces`` +
``set_periodic_axes()`` triad and the canonical ``BoundarySpec`` carried
on ``Simulation._boundary_spec``. Also verifies DeprecationWarnings are
emitted at the right entry points without false-positives on the common
scalar path.
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
        with pytest.raises(ValueError, match="pec_faces.*cannot be combined"):
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
        """Plain boundary='cpml' without pec_faces / set_periodic_axes
        must NOT emit DeprecationWarning (keeps 76 legacy test sites
        quiet; the loud warning kicks in only for the advanced triad)."""
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


# ---------------------------------------------------------------------------
# Legacy pec_faces path (DeprecationWarning expected)
# ---------------------------------------------------------------------------

class TestPecFacesDeprecation:
    def test_pec_faces_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="pec_faces.* is deprecated"):
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                boundary="cpml", pec_faces={"z_lo"},
            )

    def test_pec_faces_round_trip_to_spec(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sim = Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                boundary="cpml", pec_faces={"z_lo", "z_hi"},
            )
        assert sim._boundary_spec.z == Boundary(lo="pec", hi="pec")
        assert sim._boundary_spec.x == Boundary(lo="cpml", hi="cpml")


# ---------------------------------------------------------------------------
# Legacy set_periodic_axes path (DeprecationWarning expected)
# ---------------------------------------------------------------------------

class TestSetPeriodicAxesDeprecation:
    def test_set_periodic_axes_emits_deprecation_warning(self):
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            boundary="cpml",
        )
        with pytest.warns(DeprecationWarning,
                          match="set_periodic_axes.* is deprecated"):
            sim.set_periodic_axes("xy")

    def test_set_periodic_axes_updates_spec(self):
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            boundary="cpml",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sim.set_periodic_axes("xz")
        assert sim._boundary_spec.x == Boundary(lo="periodic", hi="periodic")
        assert sim._boundary_spec.z == Boundary(lo="periodic", hi="periodic")
        assert sim._boundary_spec.y == Boundary(lo="cpml", hi="cpml")
        assert sim._periodic_axes == "xz"
