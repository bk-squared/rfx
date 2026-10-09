"""One declaration-to-pad rule, independent of grids and coefficient builders."""

from dataclasses import dataclass
from enum import Enum

from rfx.boundaries.spec import normalize_boundary


FACES = tuple(f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi"))


class Kind(str, Enum):
    PEC = "PEC"
    PMC = "PMC"
    ABSORBER = "ABSORBER"
    PERIODIC = "PERIODIC"


@dataclass(frozen=True)
class FaceDepth:
    name: str
    kind: Kind
    declared: int
    realized: int


def resolve_face_depths(spec=None, *, budget: int, absorbing_axes="xyz",
                        pec_faces=(), pmc_faces=(), periodic_axes="",
                        mode="3d", face_layers=None,
                        validate=True) -> tuple[FaceDepth, ...]:
    """Resolve six face kinds and depths; validate even suppressed faces.

    ``validate=False`` is for the declaration-time model, which records the
    declared depth and leaves the budget check to the grid build (as before).

    ``declared`` includes the scalar fallback on walls, retaining the legacy
    Grid.face_layers view. ``realized`` counts exterior pad cells. Terminal
    planes continue to be placed by model.realize using the grid's metrics.
    Legacy builders supply their normalized face mapping instead of a spec.
    """
    boundary = normalize_boundary(spec) if spec is not None else None
    layers = face_layers or {}
    records = []
    for name in FACES:
        axis, side = name.split("_")
        declaration = getattr(boundary, axis) if boundary is not None else None
        token = getattr(declaration, side) if declaration is not None else "cpml"
        thickness = getattr(declaration, f"{side}_thickness") if declaration is not None else None
        depth = layers.get(name, budget if thickness is None else thickness)
        try:
            valid = int(depth) == depth and 0 <= depth <= budget
        except (TypeError, ValueError, OverflowError):
            valid = False
        if validate and not valid:
            raise ValueError(
                f"face_layers[{name!r}]={depth} must be an integer between "
                f"0 and cpml_layers={budget} (the allocation budget)."
            )
        kind = Kind.ABSORBER if token in ("cpml", "upml") else Kind(token.upper())
        if name in pec_faces:
            kind = Kind.PEC
        elif name in pmc_faces:
            kind = Kind.PMC
        elif axis in periodic_axes:
            kind = Kind.PERIODIC
        invariant = axis == "z" and mode in ("2d_tmz", "2d_tez")
        if invariant:
            kind = Kind.PEC if mode == "2d_tmz" else Kind.PMC
        realized = int(depth) if kind == Kind.ABSORBER and axis in absorbing_axes and not invariant else 0
        records.append(FaceDepth(name, kind, int(depth), realized))
    return tuple(records)


def grid_face_depths(grid, *, spec=None, pec_faces=None, pmc_faces=None, budget=None):
    """Read a grid's record, or resolve the legacy grid-like declaration."""
    records = getattr(grid, "boundary_depths", None)
    if (records is not None
            and (budget is None or budget == grid.cpml_layers)
            and (pec_faces is None or set(pec_faces) == set(getattr(grid, "pec_faces", ()) or ()))
            and (pmc_faces is None or set(pmc_faces) == set(getattr(grid, "pmc_faces", ()) or ()))):
        return records
    # A runner disabling CPML via all-wall overrides still has the grid's
    # allocation budget; validate its declarations against that allocation.
    allocation = getattr(grid, "cpml_layers", 0)
    return resolve_face_depths(
        spec, budget=allocation if budget is None or budget == 0 else budget,
        absorbing_axes=getattr(grid, "cpml_axes", "xyz"),
        pec_faces=(getattr(grid, "pec_faces", ()) or ()) if pec_faces is None else pec_faces,
        pmc_faces=(getattr(grid, "pmc_faces", ()) or ()) if pmc_faces is None else pmc_faces,
        periodic_axes=getattr(grid, "periodic_axes", ""),
        mode=getattr(grid, "mode", "3d"),
        face_layers=getattr(grid, "face_layers", None),
    )


def simulation_face_depths(sim):
    """Realized pad depths before allocating metrics, including waveguide axes.

    §7 Addendum 3: uniform waveguide grids pad only their port-normal axes;
    diagnostics must describe those actual pads. The NU builder does not make
    that transverse rewrite and keeps its declared absorbing axes.
    """
    if sim._waveguide_ports and not sim._uses_nonuniform_mesh:
        return sim._build_grid().boundary_depths
    return resolve_face_depths(
        sim._boundary_spec, budget=sim._cpml_layers,
        periodic_axes="".join(a for a, yes in zip("xyz", sim._periodic_flags()) if yes), mode=sim._mode,
    )


def has_positive_cpml_faces(spec, budget):
    """The coax lane requires CPML, not UPML, on all six nonzero faces."""
    return (all(token == "cpml" for _, _, token in spec.faces())
            and all(face.realized > 0 for face in resolve_face_depths(spec, budget=budget)))


def adi_uniform_faces(spec, budget):
    """ADI's scalar-only absorber cannot carry a wall or a depth override."""
    return all(face.kind == Kind.ABSORBER and face.declared == budget
               for face in resolve_face_depths(spec, budget=budget)) and all(
                   token == "cpml" for _, _, token in spec.faces())


def distributed_electric_walls(grid):
    """§7 Addendum 4: a zero-pad absorber has its PEC backing at the domain face."""
    return frozenset(face.name for face in grid_face_depths(grid)
                     if face.kind == Kind.PEC or (face.kind == Kind.ABSORBER and face.realized == 0))
