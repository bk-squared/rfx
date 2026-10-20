"""Declared face kinds and planes for the boundary-model B1 record.

Kernels do not consume this descriptor in B1. It states what each face should be
and where; ``tests/contracts/test_realized_boundary.py`` compares every entry
point's fields with it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from inspect import signature

import jax

from rfx.boundaries import cpml, upml
from rfx.boundaries.spec import normalize_boundary
from rfx.boundaries.depths import FACES, Kind, resolve_face_depths, grid_face_depths


class _DefaultBoundary(str):
    pass


DEFAULT_BOUNDARY = _DefaultBoundary("cpml")


@dataclass(frozen=True)
class Requirement:
    """One feature's requested face kinds, collected without applying them."""

    feature: str
    faces: tuple[str, ...]
    admissible: tuple[Kind, ...]
    bloch: bool = False
    predicate: str | None = None


@dataclass(frozen=True)
class Features:
    """Declaration metadata and requirements supplied to ``resolve_kinds``."""

    requirements: tuple[Requirement, ...] = ()
    layers: int = 16
    absorber_parameters: tuple[tuple[str, float], ...] = ()
    explicit_faces: bool = True
    origin: str = "declared"
    face_origins: tuple[tuple[str, str], ...] = ()
    bloch_axes: tuple[str, ...] = ()
    domain: tuple[float, float, float] | None = None

    def __post_init__(self):
        if self.domain is not None:
            object.__setattr__(self, "domain", tuple(self.domain))


@dataclass(frozen=True)
class Face:
    name: str
    kind: Kind
    origin: str
    layers: int = 0
    conformal: bool = False


@dataclass(frozen=True)
class Axis:
    name: str
    invariant: bool
    pairing: tuple[str, str] | None
    bloch: bool


@dataclass(frozen=True)
class BoundaryModel:
    """Hashable declaration; ``k_t`` is reserved and unconsumed in B1."""

    faces: tuple[Face, ...]
    axes: tuple[Axis, ...]
    absorber_type: str | None
    absorber_parameters: tuple[tuple[str, float], ...]
    requirements: tuple[Requirement, ...]
    departures: tuple[str, ...]
    mode: str
    declaration: Features
    k_t: None = None

    def face(self, name: str) -> Face:
        return self.faces[FACES.index(name)]


@dataclass(frozen=True)
class FacePlane:
    name: str
    kind: Kind
    plane_m: float | None
    terminal_m: float | None = None
    backing: Kind | None = None


@dataclass(frozen=True)
class Realization:
    """Required planes in metres and periodic lengths in metres."""

    faces: tuple[FacePlane, ...]
    periods: tuple[tuple[str, float], ...]


def _read_absorber_parameters():
    cpml_defaults = signature(cpml._cpml_profile).parameters
    cpml_parameters = {name: cpml_defaults[name].default
                       for name in ("order", "R_asymptotic", "kappa_max")}
    # Alpha is set in the profile body, not in its signature.
    with jax.ensure_compile_time_eval():
        cpml_parameters["alpha_max"] = float(cpml._cpml_profile(2, 0.0, 1.0).alpha.max())
    upml_defaults = signature(upml._sigma_profile_1d).parameters
    return {
        "cpml": tuple(sorted(cpml_parameters.items())),
        "upml": tuple((name, upml_defaults[name].default) for name in ("order", "R_asymptotic")),
    }


_ABSORBER_PARAMETERS = _read_absorber_parameters()


def resolve_kinds(spec, *, mode: str, features: Features) -> BoundaryModel:
    """Read the declaration and collect requirements without rewriting faces.

    The z axis of a 2-D mode is INVARIANT. Its stored face kind is the
    equivalent PEC (TMz) or PMC (TEz); conflicting explicit faces are recorded.
    """
    if features.origin not in ("declared", "default", "feature"):
        raise ValueError(f"unknown origin {features.origin!r}")
    boundary = normalize_boundary(spec)
    depths = {record.name: record for record in resolve_face_depths(
        boundary, budget=features.layers, mode=mode, validate=False)}
    faces, axes, departures = [], [], []
    for axis in "xyz":
        invariant = axis == "z" and mode in ("2d_tmz", "2d_tez")
        declaration = getattr(boundary, axis)
        paired = declaration.lo == "periodic" and not invariant
        axes.append(Axis(axis, invariant,
                         (f"{axis}_lo", f"{axis}_hi") if paired else None,
                         paired and axis in features.bloch_axes))
        for side in ("lo", "hi"):
            name = f"{axis}_{side}"
            kind = depths[name].kind
            origin = dict(features.face_origins).get(name, features.origin)
            if invariant:
                equivalent = Kind.PEC if mode == "2d_tmz" else Kind.PMC
                token = getattr(declaration, side)
                declared_kind = Kind.ABSORBER if token in ("cpml", "upml") else Kind(token.upper())
                if features.explicit_faces and declared_kind != equivalent:
                    departures.append(f"{name}: declared {declared_kind.value}; {mode} equivalent {equivalent.value}; fixed in B5")
                kind = equivalent
                if not features.explicit_faces:
                    origin = "feature"
            layers = depths[name].realized
            faces.append(Face(name, kind, origin, layers, declaration.conformal))
    absorber = boundary.absorber_type if any(f.kind == Kind.ABSORBER for f in faces) else None
    metadata = replace(features, layers=features.layers if absorber else 0, face_origins=(),
                       explicit_faces=features.explicit_faces if mode != "3d" else True)
    parameters = dict(_ABSORBER_PARAMETERS.get(absorber, ()))
    if absorber:
        parameters.update(features.absorber_parameters)
    return BoundaryModel(tuple(faces), tuple(axes), absorber,
                         tuple(sorted(parameters.items())),
                         tuple(sorted(features.requirements, key=repr)), tuple(departures),
                         mode, metadata)


def electric_faces(model: BoundaryModel) -> frozenset[str]:
    """PEC faces and the electric terminal backings of absorber faces."""
    return frozenset(f.name for f in model.faces if f.kind in (Kind.PEC, Kind.ABSORBER))


def magnetic_faces(model: BoundaryModel) -> frozenset[str]:
    """Faces with kind PMC."""
    return frozenset(f.name for f in model.faces if f.kind == Kind.PMC)


def realize(model: BoundaryModel, grid) -> Realization:
    """Place walls on declared faces and absorber backings at the pad ends.

    ``grid.cells`` supplies the pad metrics. Declared lengths come from the
    model metadata or ``grid.domain``; stored node counts do not set a period.
    """
    domain = model.declaration.domain
    if domain is None:
        domain = grid.domain
    records = {face.name: face for face in grid_face_depths(grid)}
    planes, periods = [], []
    for ax, axis in enumerate(model.axes):
        length = domain[ax]
        if axis.pairing:
            periods.append((axis.name, length))
        for side in ("lo", "hi"):
            name = f"{axis.name}_{side}"
            # PR3a relabels only the full-guide default faces. Other feature
            # departures (including Floquet) retain their declared kinds.
            guide_faces = getattr(grid, "_waveguide_admission", ((),))[0]
            face = records[name] if name in guide_faces else model.face(name)
            plane = 0.0 if side == "lo" else length
            if axis.invariant:
                planes.append(FacePlane(face.name, face.kind, None))
            elif face.kind == Kind.ABSORBER:
                cells = grid.cells(ax)
                pad = records[name].realized
                # Last stored width supplies a bounding node, not a pad cell.
                widths = cells[:pad] if side == "lo" else cells[len(cells) - 1 - pad:len(cells) - 1]
                terminal = plane + (-1 if side == "lo" else 1) * sum(widths)
                planes.append(FacePlane(face.name, face.kind, plane, terminal, Kind.PEC))
            else:
                planes.append(FacePlane(face.name, face.kind, plane))
    return Realization(tuple(planes), tuple(periods))


def collect_requirements(sim, *, rcs: bool = False) -> tuple[Requirement, ...]:
    """Collect TFSF, waveguide, Floquet and optional RCS requirements."""
    requirements = []
    tfsf = sim._tfsf
    if tfsf is not None:
        requirements.append(Requirement("TFSF", ("x_lo", "x_hi"), (Kind.ABSORBER,)))
        electric_axis = tfsf.polarization[-1]
        magnetic_axis = "y" if electric_axis == "z" else "z"
        oblique = tfsf.angle_deg != 0
        for axis, wall in ((electric_axis, Kind.PEC), (magnetic_axis, Kind.PMC)):
            if getattr(tfsf, "closed_box", False):
                kinds = (Kind.ABSORBER,)
            elif oblique and tfsf.method == "methodB" and axis == magnetic_axis:
                kinds = (Kind.ABSORBER,)
            elif oblique or tfsf.method == "methodB":
                kinds = (Kind.PERIODIC,)
            else:
                kinds = (Kind.PERIODIC, wall)
            requirements.append(Requirement("TFSF", (f"{axis}_lo", f"{axis}_hi"),
                                            kinds, oblique and tfsf.method == "bloch"))
    for port in sim._waveguide_ports:
        axis = port.direction[-1]
        for a in "xyz":
            requirements.append(Requirement(f"waveguide:{port.name}", (f"{a}_lo", f"{a}_hi"),
                                            (Kind.ABSORBER,) if a == axis else (Kind.PEC,),
                                            predicate="realized aperture equals cross section"))
    for port in sim._floquet_ports:
        for axis in "xyz":
            requirements.append(Requirement(f"Floquet:{port.name}", (f"{axis}_lo", f"{axis}_hi"),
                                            (Kind.ABSORBER,) if axis == port.axis else (Kind.PERIODIC,),
                                            axis != port.axis and port.scan_theta != 0))
    if rcs:
        requirements.append(Requirement("RCS", FACES, (Kind.ABSORBER,)))
    return tuple(sorted(requirements, key=repr))


def with_requirements(model: BoundaryModel, requirements: tuple[Requirement, ...]) -> BoundaryModel:
    """Replace only the collected requirements of an existing declaration."""
    if model.requirements == requirements:
        return model
    return replace(model, requirements=requirements)


def require_floquet_lateral_faces(sim, axis: str) -> str:
    """Admit a port against the current declaration, before periodic auto-fill.

    Scalar absorbers mean absorb along the port axis and auto-fill transverse
    periodicity. Their original spec identity distinguishes that shorthand
    from a later explicit face declaration, even one with identical kinds.
    """
    transverse = "".join(a for a in "xyz" if a != axis)
    explicit = sim._boundary_spec is not getattr(sim._boundary, "spec", None)
    conflicts = [
        f"{a}_{side}={kind!r}"
        for a, side, kind in sim._boundary_spec.faces()
        if a in transverse and kind != "periodic" and explicit
    ]
    if sim._periodic_axes:
        conflicts.extend(f"{a}_{side}: missing from periodic_axes={sim._periodic_axes!r}"
                         for a in transverse if a not in sim._periodic_axes
                         for side in ("lo", "hi"))
    if conflicts:
        raise ValueError(
            f"Floquet port on axis={axis!r} requires periodic lateral faces; "
            f"declared {', '.join(conflicts)}. Declare periodic on "
            f"{' and '.join(transverse)}, or omit those faces; "
            "a Floquet port's modes need periodic lateral faces.")
    return transverse


def refresh_boundary_model(sim, *, rcs: bool = False) -> BoundaryModel:
    """Store and return ``sim``'s face declaration with its current feature requirements."""
    for port in sim._floquet_ports:
        require_floquet_lateral_faces(sim, port.axis)
    sim._boundary_model = with_requirements(
        sim._boundary_model, collect_requirements(sim, rcs=rcs))
    return sim._boundary_model
