"""Feature admission for the unchanged PR3a boundary operators.

The declaration remains separate from the realized, legacy grid. In particular,
small waveguide apertures retain the legacy rewrite, PR3b (boundary-model
predeclaration, Addendum 5). Admission never changes mesh metrics.
"""
from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import warnings

from rfx.boundaries.depths import Kind, grid_face_depths, resolve_face_depths


class WaveguideBoundaryWarning(UserWarning):
    """An omitted boundary defaults to the waveguide's electric walls."""


def boundary_was_explicit(sim):
    return getattr(sim, "_boundary_explicit",
                   sim._boundary_model.declaration.origin != "default")


def _guide_axes(sim, extra=""):
    axes = {entry.direction[1] for entry in sim._waveguide_ports}
    axes.update(extra)
    return "".join(axis for axis in "xyz" if axis in axes) or "x"


def full_aperture(sim, entry, grid):
    """Compare the production port's snapped node bounds with the cross section."""
    for index, axis in enumerate("xyz"):
        if axis == entry.direction[1]:
            continue
        requested = getattr(entry, f"{axis}_range")
        if sim._uses_nonuniform_mesh:
            from rfx.runners.nonuniform import _range_to_slice_nu
            span, _ = _range_to_slice_nu(grid, axis, requested)
        elif requested is None:
            pad = getattr(grid, f"pad_{axis}_lo")
            span = (pad, grid.shape[index] - pad)
        else:
            # Like _range_to_slice, position_to_index retains the input
            # scalar division dtype and uses nearest_uniform_index.
            lo, hi = [0., 0., 0.], [0., 0., 0.]
            lo[index], hi[index] = requested
            span = (grid.position_to_index(lo)[index], grid.position_to_index(hi)[index] + 1)
        bounds = (getattr(grid, f"pad_{axis}_lo"),
                  grid.shape[index] - getattr(grid, f"pad_{axis}_hi"))
        if span != bounds:
            return False
    return True


def guide_faces(sim, grid):
    # Addendum 5c(1): only a zero-depth absorbing face was rewritten on main.
    zero_depth = {face.name for face in grid_face_depths(grid) if face.realized == 0}
    return frozenset(f"{axis}_{side}"
                     for entry in sim._waveguide_ports if full_aperture(sim, entry, grid)
                     for axis in "xyz" if axis != entry.direction[1]
                     for side in ("lo", "hi") if f"{axis}_{side}" in zero_depth)


def _guide_declaration(sim, faces):
    declaration = {}
    for axis in "xyz":
        if any(face.startswith(axis + "_") for face in faces):
            declaration[axis] = "pec"
        else:
            value = getattr(sim._boundary_spec, axis).to_dict()
            declaration[axis] = (value["lo"] if len(value) == 2 and value["lo"] == value["hi"]
                                 else value)
    return declaration


def _record_defaults(sim, grid):
    if sim._waveguide_ports:
        faces = guide_faces(sim, grid)
        explicit = boundary_was_explicit(sim)
        incompatible = sorted(face.name for face in grid.boundary_depths
                              if face.name in faces and face.kind != Kind.PEC)
        state = None
        if not explicit:
            state = getattr(sim, "_boundary_default_warned", None)
            if not isinstance(state, list) or len(state) != 2 or state[1] != id(sim):
                state = [False, id(sim)]
                sim._boundary_default_warned = state
        grid._waveguide_admission = (incompatible, _guide_declaration(sim, faces), explicit, state)
    if sim._waveguide_ports and not explicit:
        # The uniform legacy grid already has these electric backing walls.
        # Do not change pads, allocation axes, or the NU operator (PR3b).
        grid.boundary_depths = tuple(
            replace(face, kind=Kind.PEC)
            if face.name in faces and face.realized == 0 and face.kind == Kind.ABSORBER else face
            for face in grid.boundary_depths)
    return grid


def build_grid(sim, *, extra_waveguide_axes=""):
    """Build the uniform legacy lattice, then record its full-guide defaults."""
    from rfx.grid import Grid, _periodic_resolution, _wall_closed_axes
    sim._require_uniform_mesh("uniform grid construction")
    periodic = "".join(a for a, yes in zip("xyz", sim._periodic_flags()) if yes)
    dx = sim._dx
    if dx is not None and sim._declared_mesh["_dx"] is None:
        physical = periodic.replace("z", "") if sim._mode.startswith("2d") else periodic
        dx = _periodic_resolution(
            sim._domain, physical, dx, automatic=True,
            wall_axes=_wall_closed_axes(sim._boundary_spec.pec_faces(),
                                       sim._boundary_spec.pmc_faces(),
                                       is_2d=sim._mode.startswith("2d")))
    # Smaller apertures retain the legacy rewrite, PR3b; Addendum 5.
    axes = _guide_axes(sim, extra_waveguide_axes) if sim._waveguide_ports or extra_waveguide_axes else "xyz"
    grid = Grid(
        freq_max=sim._freq_max, domain=sim._domain, dx=dx,
        cpml_layers=sim._cpml_layers,
        cpml_axes="".join(axis for axis in axes if axis not in periodic),
        mode=sim._mode, kappa_max=sim._cpml_kappa_max,
        pec_faces=sim._boundary_spec.pec_faces(),
        pmc_faces=sim._boundary_spec.pmc_faces(),
        face_layers=sim._resolve_face_layers(),
        conformal_faces=sim._boundary_spec.conformal_faces(), periodic_axes=periodic)
    return _record_defaults(sim, grid)


def _warn_caller(message):
    root = Path(__file__).resolve().parents[1]
    frame = inspect.currentframe().f_back
    try:
        while frame is not None and Path(frame.f_code.co_filename).resolve().is_relative_to(root):
            frame = frame.f_back
        if frame is None:
            warnings.warn(message, WaveguideBoundaryWarning, stacklevel=2)
        else:
            warnings.warn_explicit(message, WaveguideBoundaryWarning,
                                   frame.f_code.co_filename, frame.f_lineno)
    finally:
        del frame


def admit_waveguide(sim, *, lane="dispatch"):
    """Refuse undeclared guide models without changing a realized grid."""
    if not sim._waveguide_ports:
        return
    if sim._uses_nonuniform_mesh:
        _admit_graded_guide(sim, lane)
        return
    grid = sim._build_realized_grid()
    faces = guide_faces(sim, grid)
    periodic = sorted(face for face in faces if sim._periodic_flags()["xyz".index(face[0])])
    if periodic:
        raise ValueError("Full-aperture waveguide requires PEC on " + ", ".join(periodic)
                         + "; the active periodic feature requires PERIODIC on those faces.")
    declared = resolve_face_depths(sim._boundary_spec, budget=sim._cpml_layers)
    incompatible = sorted(face.name for face in declared
                          if face.name in faces and face.kind != Kind.PEC)
    if incompatible and boundary_was_explicit(sim):
        _refuse_guide(incompatible, _guide_declaration(sim, faces), lane)
    if incompatible and (any(getattr(grid, f"pad_{face}") for face in incompatible)
                         or any(sim._periodic_flags()["xyz".index(face[0])] for face in incompatible)):
        raise ValueError(
            f"{lane}: full-aperture waveguide defaults on {', '.join(incompatible)} "
            "would change realized transverse pads or an active periodic feature; this requires PR3b.")
    state = getattr(sim, "_boundary_default_warned", [False])
    if incompatible and not state[0]:
        _warn_caller("Waveguide boundary default: " + ", ".join(incompatible)
                     + " are solved as PEC electric walls.")
        state[0] = True


def _admit_graded_guide(sim, lane):
    # Addendum 5d: real transverse pads are retained only by an explicit
    # declaration. Unknown imported provenance keeps main's behavior (5c(6)).
    if boundary_was_explicit(sim) is not False:
        return
    grid = sim._build_realized_grid()
    faces = sorted({f"{axis}_{side}"
                    for entry in sim._waveguide_ports if full_aperture(sim, entry, grid)
                    for axis in "xyz" if axis != entry.direction[1]
                    for side in ("lo", "hi")})
    if faces:
        raise ValueError(
            f"{lane}: full-aperture waveguide boundary was not declared on {', '.join(faces)}; "
            "the graded mesh keeps absorbers there, so the port would not be in a guide; "
            f"declare boundary={_guide_declaration(sim, faces)!r} (absorber on the port axis).")


def _refuse_guide(faces, declaration, lane):
    raise ValueError(
        f"{lane}: full-aperture waveguide requires PEC on {', '.join(faces)}; "
        f"declare boundary={declaration!r} (absorber on the port axis).")


def admit_grid_waveguide(grid, ports):
    """Carry a Simulation's admission through direct low-level grid calls.

    A standalone Grid has no feature-selected allocation to rewrite. A grid
    built by Simulation retains its admission even if that Simulation dies.
    """
    record = getattr(grid, "_waveguide_admission", None)
    if record is None:
        return
    faces, declaration, explicit, state = record
    if faces and explicit:
        _refuse_guide(faces, declaration, "low-level dispatch")
    elif faces and any(getattr(grid, f"pad_{face}") or face[0] in grid.periodic_axes for face in faces):
        raise ValueError("Full-aperture waveguide defaults would change realized transverse "
                         "pads or an active periodic feature; this requires PR3b.")
    elif faces and not state[0]:
        _warn_caller("Waveguide boundary default: " + ", ".join(faces)
                     + " are solved as PEC electric walls.")
        state[0] = True
