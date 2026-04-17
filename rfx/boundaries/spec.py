"""Per-axis, per-face boundary specification (T7, v1.7.0).

The ``BoundarySpec`` / ``Boundary`` types replace the asymmetric
``boundary=<scalar> + pec_faces + set_periodic_axes()`` triad with a
single canonical object. A ``BoundarySpec`` holds one ``Boundary`` per
axis, and each ``Boundary`` names the ``lo`` and ``hi`` face.

The shape is modelled on Tidy3D's ``BoundarySpec`` (axis-grouped with
per-face fields). The naming convention uses ``lo`` / ``hi`` to match
the existing rfx ``pec_faces`` convention (``x_lo`` / ``x_hi`` / ...).

T7-A scope — this module is types + construction-time validation only.
Per-face CPML thickness (T7-C), preflight per-face refactoring (T7-D),
and PMC runtime (T7-E) layer onto this foundation in later stories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

BOUNDARY_TOKENS = ("cpml", "upml", "pec", "pmc", "periodic")
ABSORBING_TOKENS = ("cpml", "upml")


def _normalise_token(value: str | "Boundary") -> str:
    if isinstance(value, Boundary):
        raise TypeError(
            "expected a string token, got Boundary; pass the Boundary "
            "object directly to the axis kwarg instead of to from_string."
        )
    if not isinstance(value, str):
        raise TypeError(
            f"boundary token must be a string, got {type(value).__name__}"
        )
    token = value.strip().lower()
    if token not in BOUNDARY_TOKENS:
        raise ValueError(
            f"unknown boundary token {value!r}; valid tokens are "
            f"{BOUNDARY_TOKENS}"
        )
    return token


@dataclass(frozen=True)
class Boundary:
    """Per-face boundary specification for one axis.

    ``lo`` and ``hi`` carry one of the tokens ``cpml``, ``upml``, ``pec``,
    ``pmc``, ``periodic``.  A ``periodic`` axis must be symmetric: both
    faces must be ``periodic`` or neither.

    Optional ``lo_thickness`` / ``hi_thickness`` (CPML/UPML only) override
    the global ``cpml_layers`` scalar for that face. A ``None`` value
    (default) defers to the scalar.
    """

    lo: str
    hi: str
    lo_thickness: int | None = None
    hi_thickness: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "lo", _normalise_token(self.lo))
        object.__setattr__(self, "hi", _normalise_token(self.hi))
        if (self.lo == "periodic") != (self.hi == "periodic"):
            raise ValueError(
                f"periodic must be symmetric on an axis; got "
                f"lo={self.lo!r} hi={self.hi!r}. Use Boundary(lo='periodic', "
                f"hi='periodic') or rename the asymmetric side."
            )
        # thickness is meaningful only on absorbing faces.
        if self.lo_thickness is not None and self.lo not in ABSORBING_TOKENS:
            raise ValueError(
                f"lo_thickness is only meaningful for absorbing faces "
                f"(cpml, upml); got lo={self.lo!r} with thickness="
                f"{self.lo_thickness}."
            )
        if self.hi_thickness is not None and self.hi not in ABSORBING_TOKENS:
            raise ValueError(
                f"hi_thickness is only meaningful for absorbing faces "
                f"(cpml, upml); got hi={self.hi!r} with thickness="
                f"{self.hi_thickness}."
            )
        for name, val in (("lo_thickness", self.lo_thickness),
                          ("hi_thickness", self.hi_thickness)):
            if val is not None and (not isinstance(val, int) or val < 0):
                raise ValueError(
                    f"{name} must be a non-negative int, got {val!r}."
                )

    @classmethod
    def from_string(cls, token: str) -> "Boundary":
        """Axis-shorthand constructor: symmetric Boundary(lo=t, hi=t)."""
        t = _normalise_token(token)
        return cls(lo=t, hi=t)

    def to_dict(self) -> dict:
        out = {"lo": self.lo, "hi": self.hi}
        if self.lo_thickness is not None:
            out["lo_thickness"] = self.lo_thickness
        if self.hi_thickness is not None:
            out["hi_thickness"] = self.hi_thickness
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Boundary":
        return cls(
            lo=d["lo"], hi=d["hi"],
            lo_thickness=d.get("lo_thickness"),
            hi_thickness=d.get("hi_thickness"),
        )

    def resolved_lo_thickness(self, default: int) -> int:
        """Per-face layer count, falling back to the scalar default."""
        return self.lo_thickness if self.lo_thickness is not None else default

    def resolved_hi_thickness(self, default: int) -> int:
        return self.hi_thickness if self.hi_thickness is not None else default


_BoundaryInput = Union[str, Boundary, dict]


def _coerce_axis(value: _BoundaryInput) -> Boundary:
    """Accept string shorthand, dict, or Boundary — return Boundary."""
    if isinstance(value, Boundary):
        return value
    if isinstance(value, dict):
        return Boundary.from_dict(value)
    if isinstance(value, str):
        return Boundary.from_string(value)
    raise TypeError(
        f"axis boundary must be a string, Boundary, or dict — got "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class BoundarySpec:
    """Canonical per-axis, per-face boundary description for Simulation.

    Each of ``x``, ``y``, ``z`` is a ``Boundary``. Shorthand string
    inputs on any axis expand to symmetric ``Boundary(lo=s, hi=s)``.
    """

    x: Boundary
    y: Boundary
    z: Boundary

    def __init__(self, x: _BoundaryInput, y: _BoundaryInput,
                 z: _BoundaryInput):
        object.__setattr__(self, "x", _coerce_axis(x))
        object.__setattr__(self, "y", _coerce_axis(y))
        object.__setattr__(self, "z", _coerce_axis(z))
        self._validate_absorber_consistency()

    def _validate_absorber_consistency(self) -> None:
        tokens = {
            self.x.lo, self.x.hi,
            self.y.lo, self.y.hi,
            self.z.lo, self.z.hi,
        }
        absorbing = tokens & set(ABSORBING_TOKENS)
        if len(absorbing) > 1:
            raise ValueError(
                "cannot mix CPML and UPML on different faces of the same "
                "simulation; the update stencils differ and no absorber "
                f"interface layer is implemented. Got absorbing tokens: "
                f"{sorted(absorbing)}."
            )

    @classmethod
    def uniform(cls, token: str) -> "BoundarySpec":
        """All six faces use the same boundary token."""
        t = _normalise_token(token)
        b = Boundary(lo=t, hi=t)
        return cls(x=b, y=b, z=b)

    def faces(self):
        """Iterate (axis_name, 'lo'|'hi', token) tuples over the 6 faces."""
        for axis_name, boundary in (("x", self.x), ("y", self.y),
                                    ("z", self.z)):
            yield axis_name, "lo", boundary.lo
            yield axis_name, "hi", boundary.hi

    @property
    def absorber_type(self) -> str | None:
        """Return the unique absorbing-face token ('cpml' or 'upml'), or
        None if no face uses an absorbing boundary. Used by the runners
        and preflight to pick the absorber update path."""
        for _axis, _side, tok in self.faces():
            if tok in ABSORBING_TOKENS:
                return tok
        return None

    def periodic_axes(self) -> str:
        """Return the concatenated string of axis names with
        ``periodic`` on both faces (e.g. ``'xy'``)."""
        out = []
        for axis_name, boundary in (("x", self.x), ("y", self.y),
                                    ("z", self.z)):
            if boundary.lo == "periodic":
                # symmetry already enforced in Boundary.__post_init__
                out.append(axis_name)
        return "".join(out)

    def _faces_with_token(self, token: str) -> set[str]:
        return {
            f"{axis_name}_{side}"
            for axis_name, side, tok in self.faces()
            if tok == token
        }

    def pec_faces(self) -> set[str]:
        """Face labels (``"z_lo"`` …) set to ``pec``. Legacy view."""
        return self._faces_with_token("pec")

    def pmc_faces(self) -> set[str]:
        """Face labels set to ``pmc``."""
        return self._faces_with_token("pmc")

    def to_dict(self) -> dict:
        return {
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
            "z": self.z.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoundarySpec":
        return cls(x=d["x"], y=d["y"], z=d["z"])


def normalize_boundary(obj: _BoundaryInput | BoundarySpec | None,
                       default: str = "cpml") -> BoundarySpec:
    """Accept any legal boundary input and return a canonical
    ``BoundarySpec``.

    ``None`` -> ``BoundarySpec.uniform(default)``.
    ``str``  -> ``BoundarySpec.uniform(str)``.
    ``BoundarySpec`` -> returned as-is.
    ``dict`` -> ``BoundarySpec.from_dict``.
    """
    if obj is None:
        return BoundarySpec.uniform(default)
    if isinstance(obj, BoundarySpec):
        return obj
    if isinstance(obj, str):
        return BoundarySpec.uniform(obj)
    if isinstance(obj, dict):
        return BoundarySpec.from_dict(obj)
    raise TypeError(
        f"cannot normalize boundary input of type {type(obj).__name__}; "
        f"expected BoundarySpec, str, dict, or None."
    )
