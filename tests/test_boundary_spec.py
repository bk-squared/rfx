"""Unit tests for ``rfx.boundaries.spec`` (T7-A).

Covers the type surface, construction-time validation, and
serialisation round-trips for the per-axis, per-face boundary
specification introduced in T7-A (2026-04).
"""

from __future__ import annotations

import pytest

from rfx.boundaries.spec import (
    Boundary,
    BoundarySpec,
    normalize_boundary,
)


# ---------------------------------------------------------------------------
# Boundary — per-face, single-axis
# ---------------------------------------------------------------------------

class TestBoundary:
    def test_from_string_shorthand_symmetric(self):
        """``Boundary.from_string('cpml')`` matches explicit symmetric."""
        explicit = Boundary(lo="cpml", hi="cpml")
        shorthand = Boundary.from_string("cpml")
        assert explicit == shorthand
        assert explicit.to_dict() == {"lo": "cpml", "hi": "cpml"}

    def test_token_case_insensitive(self):
        assert Boundary(lo="CPML", hi="cpml") == Boundary(lo="cpml", hi="cpml")

    def test_valid_tokens_all_accepted(self):
        for tok in ("cpml", "upml", "pec", "pmc", "periodic"):
            # same-token symmetric must always be legal
            b = Boundary.from_string(tok)
            assert b.lo == tok and b.hi == tok

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="unknown boundary token"):
            Boundary.from_string("mur")

    def test_periodic_asymmetry_raises(self):
        with pytest.raises(ValueError, match="periodic must be symmetric"):
            Boundary(lo="periodic", hi="cpml")
        with pytest.raises(ValueError, match="periodic must be symmetric"):
            Boundary(lo="pec", hi="periodic")

    def test_asymmetric_non_periodic_is_fine(self):
        """PEC / CPML asymmetry is legitimate (e.g. patch antenna)."""
        b = Boundary(lo="pec", hi="cpml")
        assert b.lo == "pec" and b.hi == "cpml"

    def test_dict_round_trip(self):
        b = Boundary(lo="pec", hi="cpml")
        assert Boundary.from_dict(b.to_dict()) == b


# ---------------------------------------------------------------------------
# BoundarySpec — three axes, six faces total
# ---------------------------------------------------------------------------

class TestBoundarySpec:
    def test_shorthand_axis_strings(self):
        spec = BoundarySpec(x="cpml", y="periodic", z="pec")
        assert spec.x == Boundary(lo="cpml", hi="cpml")
        assert spec.y == Boundary(lo="periodic", hi="periodic")
        assert spec.z == Boundary(lo="pec", hi="pec")

    def test_mixed_shorthand_and_boundary(self):
        spec = BoundarySpec(
            x="cpml",
            y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        )
        assert spec.z.lo == "pec"
        assert spec.z.hi == "cpml"

    def test_uniform_all_six_faces(self):
        spec = BoundarySpec.uniform("cpml")
        for _axis, _side, tok in spec.faces():
            assert tok == "cpml"

    def test_cpml_upml_mix_raises(self):
        with pytest.raises(ValueError, match="cannot mix CPML and UPML"):
            BoundarySpec(x="cpml", y="upml", z="pec")
        with pytest.raises(ValueError, match="cannot mix CPML and UPML"):
            BoundarySpec(
                x="cpml",
                y="cpml",
                z=Boundary(lo="pec", hi="upml"),
            )

    def test_unknown_token_in_spec_raises(self):
        with pytest.raises(ValueError, match="unknown boundary token"):
            BoundarySpec(x="mur", y="cpml", z="cpml")

    def test_periodic_mismatch_raises_through_spec(self):
        # Still rejected at the axis-level Boundary validation even when
        # routed through the spec.
        with pytest.raises(ValueError, match="periodic must be symmetric"):
            BoundarySpec(
                x=Boundary(lo="periodic", hi="cpml"),
                y="cpml",
                z="cpml",
            )

    def test_absorber_type_single_cpml(self):
        spec = BoundarySpec(x="cpml", y="periodic", z=Boundary(lo="pec", hi="cpml"))
        assert spec.absorber_type == "cpml"

    def test_absorber_type_none_when_no_absorber(self):
        spec = BoundarySpec(x="pec", y="periodic", z="pec")
        assert spec.absorber_type is None

    def test_periodic_axes_view(self):
        spec = BoundarySpec(x="periodic", y="cpml", z="periodic")
        # z cannot be 'periodic' while x is — actually both are symmetric
        # here so this is legitimate. Match behaviour of set_periodic_axes.
        assert spec.periodic_axes() == "xz"

    def test_pec_faces_view(self):
        spec = BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"))
        assert spec.pec_faces() == {"z_lo"}

    def test_pmc_faces_view(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="cpml"),
        )
        assert spec.pmc_faces() == {"z_lo"}

    def test_dict_round_trip(self):
        spec = BoundarySpec(x="cpml", y="periodic",
                            z=Boundary(lo="pec", hi="cpml"))
        d = spec.to_dict()
        assert BoundarySpec.from_dict(d) == spec

    def test_equality_and_repr(self):
        a = BoundarySpec.uniform("cpml")
        b = BoundarySpec(x="cpml", y="cpml", z="cpml")
        assert a == b
        # frozen dataclass repr should at least name the class
        assert "BoundarySpec" in repr(a)


# ---------------------------------------------------------------------------
# normalize_boundary — entry-point helper
# ---------------------------------------------------------------------------

class TestNormalizeBoundary:
    def test_none_returns_default_uniform(self):
        assert normalize_boundary(None) == BoundarySpec.uniform("cpml")
        assert normalize_boundary(None, default="pec") == BoundarySpec.uniform("pec")

    def test_string_returns_uniform(self):
        assert normalize_boundary("cpml") == BoundarySpec.uniform("cpml")

    def test_boundaryspec_returned_as_is(self):
        spec = BoundarySpec(x="cpml", y="periodic", z="pec")
        assert normalize_boundary(spec) is spec

    def test_dict_converts(self):
        spec = BoundarySpec.uniform("cpml")
        assert normalize_boundary(spec.to_dict()) == spec

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="cannot normalize boundary"):
            normalize_boundary(42)
