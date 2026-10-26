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
import weakref

from rfx.boundaries.depths import Kind, grid_face_depths, resolve_face_depths, simulation_face_depths


_WALL_AXIS_WARNED = weakref.WeakKeyDictionary()


def share_wall_warning_history(sim, diagnostic_copy):
    """An internal preflight reader represents the same Simulation warning scope."""
    _WALL_AXIS_WARNED[diagnostic_copy] = _WALL_AXIS_WARNED.setdefault(sim, set())


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
    """Recognize declared full ranges before comparing snapped interior bounds."""
    for index, axis in enumerate("xyz"):
        if axis == entry.direction[1]:
            continue
        requested = getattr(entry, f"{axis}_range")
        if requested is None or (requested[0] <= 1e-9
                                  and requested[1] >= sim._domain[index] - 1e-9):
            continue
        if sim._uses_nonuniform_mesh:
            from rfx.runners.nonuniform import _range_to_slice_nu
            span, _ = _range_to_slice_nu(grid, axis, requested)
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
    from rfx.grid import C0, Grid, _periodic_resolution, axis_kinds
    sim._require_uniform_mesh("uniform grid construction")
    periodic = "".join(a for a, yes in zip("xyz", sim._periodic_flags()) if yes)
    dx = sim._dx if sim._dx is not None else C0 / sim._freq_max / 20.0
    axes = _guide_axes(sim, extra_waveguide_axes) if sim._waveguide_ports or extra_waveguide_axes else "xyz"
    kwargs = dict(
        freq_max=sim._freq_max, domain=sim._domain,
        cpml_layers=sim._cpml_layers,
        cpml_axes="".join(axis for axis in axes if axis not in periodic),
        mode=sim._mode, kappa_max=sim._cpml_kappa_max,
        pec_faces=sim._boundary_spec.pec_faces(),
        pmc_faces=sim._boundary_spec.pmc_faces(),
        face_layers=sim._resolve_face_layers(),
        conformal_faces=sim._boundary_spec.conformal_faces())
    depths = resolve_face_depths(
        budget=sim._cpml_layers, absorbing_axes=kwargs["cpml_axes"],
        pec_faces=kwargs["pec_faces"],
        pmc_faces=kwargs["pmc_faces"],
        periodic_axes=periodic, mode=sim._mode, face_layers=kwargs["face_layers"])
    kinds = axis_kinds(depths)
    physical = periodic.replace("z", "") if sim._mode.startswith("2d") else periodic
    dx = _periodic_resolution(
        sim._domain, physical, dx, automatic=sim._declared_mesh["_dx"] is None,
        wall_axes="".join(a for a in "xyz" if kinds[a] == "wall"
                          and not (a == "z" and sim._mode.startswith("2d"))))
    grid = Grid(**kwargs, dx=dx, periodic_axes=periodic, boundary_depths=depths)
    judge_wall_axes(sim, grid)
    return _record_defaults(sim, grid)


def judge_wall_axes(sim, grid, *, boundary_depths=None):
    """Judge the built unprofiled wall lengths where Simulation owns snap.

    Low-level grid builders have no declaration acceptance policy. Declared
    warnings are emitted once per axis/declared/solved/dx tuple on this sim.
    """
    from rfx.grid import axis_kinds, realized_cells
    depths = grid_face_depths(grid) if boundary_depths is None else boundary_depths
    messages = []
    findings = []
    for index, (axis, kind) in enumerate(axis_kinds(depths).items()):
        if kind != "wall" or (axis == "z" and sim._mode.startswith("2d")):
            continue
        # The lattice counts a periodic axis without its closing node, whatever
        # the face table says (a PEC box with a Floquet port, refused later).
        if axis in getattr(grid, "periodic_axes", ""):
            continue
        if getattr(sim, f"_d{axis}_profile") is not None:
            continue
        length = float(sim._domain[index])
        if hasattr(grid, "dx_arr_f64"):
            from rfx.model.graded_grid import interior_cells
            widths = (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)[index]
            cells = interior_cells(widths, getattr(grid, f"pad_{axis}_lo"),
                                   getattr(grid, f"pad_{axis}_hi"))
            n, solved = len(cells), float(sum(cells))
            cell = float(cells[0])  # an unprofiled axis has one interior width
        else:
            span = grid.interior[index]
            n = span.stop - span.start - 1
            cell = float(grid.cells(axis)[span.start])
            solved = n * cell
        rel = solved / length - 1
        if abs(rel) <= .01:
            continue
        nearest_n = realized_cells(length, cell, "wall")
        adjacent = min((length / count for count in (nearest_n - 1, nearest_n + 1) if count >= 1),
                       key=lambda spacing: abs(spacing - cell))
        messages.append(
            f"wall axis {axis!r}: declared {length * 1e3:.12g} mm, solved {solved * 1e3:.12g} mm, "
            f"rel {rel * 100:+.12g} %, {n} cells, dx={cell * 1e3:.12g} mm; "
            f"nearest dx={length / nearest_n * 1e3:.12g} mm, next dx={adjacent * 1e3:.12g} mm "
            "(both pass 1 %)")
        findings.append((axis, length, solved, cell, messages[-1]))
    if messages:
        message = "; ".join(messages) + "; set snap='declared' to accept."
        if sim._snap == "declared":
            from rfx.preflight._common import PreflightWarning
            warned = _WALL_AXIS_WARNED.setdefault(sim, set())
            for axis, length, solved, dx, text in findings:
                key = (axis, length, solved, dx)
                if key in warned:
                    continue
                warnings.warn(PreflightWarning(
                    text + "; accepted with snap='declared'.",
                    code="wall_axis_off_node", loc=axis, source="judge_wall_axes"),
                    stacklevel=3)
                warned.add(key)
        else:
            raise ValueError(message)


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


def admit_ntff(sim, *, lane):
    """Refuse NTFF on reflecting domain faces; periodic axes are undecided."""
    if sim._ntff is None:
        return
    refuse_ntff_reflecting_faces(
        simulation_face_depths(sim), lane=lane)


def refuse_ntff_reflecting_faces(faces, *, lane):
    """Refuse reflecting faces from the declaration or realized run record."""
    incompatible = {}
    for face in faces:
        if face.kind in (Kind.PEC, Kind.PMC):
            incompatible[face.name] = face.kind.value.lower()
        elif face.kind == Kind.ABSORBER and face.realized == 0:
            incompatible[face.name] = "zero-depth absorber"
    if incompatible:
        descriptions = [f"{face} is {kind}" for face, kind in sorted(incompatible.items())]
        raise ValueError(
            f"{lane}: an NTFF box radiates its surface currents into free space, but "
            + "; ".join(descriptions)
            + ": the far field would miss their images (a symmetry or ground plane is not "
            "reconstructed). Every face of the realized model must be absorbing: draw a "
            "finite ground plane as geometry, and note that a waveguide port on the uniform mesh "
            "solves its transverse faces as PEC walls; a symmetry plane is not supported "
            "with NTFF yet.")


def admit_waveguide(sim, *, lane="dispatch"):
    """Refuse undeclared guide models without changing a realized grid."""
    if not sim._waveguide_ports:
        return
    if sim._uses_nonuniform_mesh:
        _admit_graded_guide(sim, lane)
        return
    grid = sim._build_realized_grid()
    faces = guide_faces(sim, grid)
    _refuse_periodic_guide(sim, faces)
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
    # Addendum 5f: full guides require explicitly declared transverse PEC.
    grid = sim._build_realized_grid()
    faces = sorted({f"{axis}_{side}"
                    for entry in sim._waveguide_ports if full_aperture(sim, entry, grid)
                    for axis in "xyz" if axis != entry.direction[1]
                    for side in ("lo", "hi")})
    _refuse_periodic_guide(sim, faces)
    explicit = boundary_was_explicit(sim)
    if explicit is True:
        declared = resolve_face_depths(sim._boundary_spec, budget=sim._cpml_layers)
        incompatible = sorted(face.name for face in declared
                              if face.name in faces and face.kind != Kind.PEC)
        if incompatible:
            _refuse_guide(incompatible, _guide_declaration(sim, faces), lane)
    if faces and explicit is not True:
        raise ValueError(
            f"{lane}: full-aperture waveguide boundary was not declared on {', '.join(faces)}; "
            "the graded mesh keeps absorbers there, so the port would not be in a guide; "
            f"declare boundary={_guide_declaration(sim, faces)!r} (absorber on the port axis).")


def _refuse_periodic_guide(sim, faces):
    periodic = sorted(face for face in faces if sim._periodic_flags()["xyz".index(face[0])])
    if periodic:
        raise ValueError("Full-aperture waveguide requires PEC on " + ", ".join(periodic)
                         + "; the active periodic feature requires PERIODIC on those faces.")


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


def waveguide_segment_cells(cfg, grid):
    """Half-open xyz bounds from H through the reference plane, excluding probes.

    H acts at x_index-1 for a positive launch and x_index for a negative
    launch. A declared reference includes cell centres strictly on its port
    side, also when behind the source. S extraction reads reference waves
    only; the reference shift uses the section's beta. Transverse bounds are the
    compiled aperture cell spans.
    """
    import numpy as np
    from rfx.geometry.rasterize_grid import _axis_node_positions
    axis = "xyz".index(cfg.normal_axis)
    h = cfg.x_index - 1 if cfg.direction.startswith("+") else cfg.x_index
    endpoints = [h, cfg.x_index, cfg.ref_x]
    if cfg.segment_planes:
        nodes = np.asarray(_axis_node_positions(
            grid.cells(cfg.normal_axis), getattr(grid, f"pad_{cfg.normal_axis}_lo")))
        centres = nodes + np.asarray(grid.cells(cfg.normal_axis)) / 2
        endpoints.extend(int(np.searchsorted(centres, p, side="left") - 1)
                         if p >= nodes[cfg.x_index] else
                         int(np.searchsorted(centres, p, side="right"))
                         for p in cfg.segment_planes)
    bounds = [(cfg.u_lo, cfg.u_hi), (cfg.v_lo, cfg.v_hi)]
    bounds.insert(axis, (max(0, min(endpoints)), min(grid.shape[axis], max(endpoints) + 1)))
    return tuple(bounds)


def resolve_waveguide_section_medium(waveguide_ports, grid, materials, updates,
                                     localized, design_box=None):
    """Resolve one scalar lossless medium over each corrected/reference section.

    Material conductivity is checked, never the CPML profile. Geometric PEC
    edges at the aperture boundary do not constitute a PEC volume. A probe is
    judged separately: only consumers of its waves refuse a different medium.
    """
    if not waveguide_ports:
        return waveguide_ports
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rfx._traced_check import traced_check
    from rfx.model.operands import _leaves, realized_operands
    from rfx.model.design_box import design_box_window
    from rfx.model.waveguide_modes import PortMedium, with_port_medium
    from rfx.boundaries.pec import wall_edge_masks, resolve_wall_faces

    wall_masks = wall_edge_masks(grid.shape, resolve_wall_faces(
        grid, tuple(a in getattr(grid, "periodic_axes", "") for a in "xyz"))[0])

    # Scalar cell media and PEC volume from the one realized operand list (#1961):
    # sheets, films, lumped elements and wall-derived tensors are not judged here.
    scalar = ("materials.eps_r", "materials.sigma", "materials.mu_r")
    updates_scalar = ("debye", "lorentz", "kerr_chi3", "pec_mask", "pec_occupancy")
    # The shared list does not classify per-component material leaves: enumerate a
    # scalar view and judge those components below (sigma) or refuse them (eps_r, mu_r).
    split = tuple(f for f in ("eps_r", "sigma", "mu_r")
                  if isinstance(getattr(materials, f), (tuple, list)))
    view = materials._replace(**{f: jnp.zeros(()) for f in split})
    operands = [(r.name, r.array, r.vacuum, r.label)
                for r in realized_operands(grid, view, updates, localized)
                if (r.name in scalar and r.name.split(".", 1)[1] not in split)
                or (r.source is not None and r.name.split(".", 1)[0] in updates_scalar)]
    if isinstance(materials.sigma, (tuple, list)):
        operands.extend((name, array, 0, "conductivity")
                        for name, array in _leaves(materials.sigma, "materials.sigma"))
    # Explicit component overrides must describe this same scalar medium.
    # Scalar-cell enumeration deliberately excludes wall-derived component means.
    for key in ("aniso_eps", "aniso_inv_eps"):
        value = updates.get(key)
        if value is not None:
            for name, array in _leaves(value, key):
                component = name.rsplit(".", 1)[-1]
                weights = updates.get("conformal_weights")
                if (key == "aniso_eps" and weights is not None
                        and ".nodes." not in name and component in ("0", "1", "2")):
                    # finish_tensor_operands applies eps_edge / safe_w for
                    # conformal PEC. That known geometric factor is not a
                    # material anisotropy; any residual mismatch still refuses.
                    weight = weights[int(component)]
                    array = array * jnp.where(weight > 0, weight, 1.0)
                operands.append((name, array, None, "anisotropic permittivity"))
    reason = ("waveguide sections require one homogeneous, isotropic, lossless, "
              "non-dispersive medium (rfx#1994)")

    def section(array, bounds):
        trailing = 0
        start = array.ndim - 3 - trailing
        slices = [slice(None)] * array.ndim
        offsets = [0] * array.ndim
        if start >= 0:
            for axis, (lo, hi) in enumerate(bounds):
                offsets[start+axis] = lo
                if array.shape[start+axis] != 1:
                    slices[start+axis] = slice(lo, hi)
        return array[tuple(slices)], start, offsets

    def resolve(cfg, multimode):
        bounds = waveguide_segment_cells(cfg, grid)
        prefix = f"Waveguide port {cfg.port_name!r}: section cells {bounds}: "
        for field in ("eps_r", "mu_r"):
            if isinstance(getattr(materials, field), (tuple, list)):
                raise ValueError(prefix + f"per-component {field} is unsupported; " + reason)
        if design_box is not None:
            write, _, _, _ = design_box_window(design_box.bounds, grid.shape)
            if all(max(lo, write[2*a]) < min(hi, write[2*a+1])
                   for a, (lo, hi) in enumerate(bounds)):
                raise ValueError(prefix + f"design_box update window {write} intersects; " + reason)
        portions = [section(array, bounds) for _, array, _, _ in operands]
        arrays = [p[0] for p in portions]
        traced = any(isinstance(x, jax.core.Tracer) for x in arrays)

        def judge(*values):
            scalars = {}
            for (name, _, vacuum, label), value, (_, start, offsets) in zip(operands, values, portions):
                host = np.asarray(value)
                if not host.size:
                    continue
                if name in ("materials.eps_r", "materials.mu_r"):
                    target = float(host.flat[0])
                    scalars[name] = target
                    bad = (host != target) | ~np.isfinite(host) | (host <= 0)
                elif vacuum is not None:
                    bad = host != vacuum
                else:
                    eps = scalars.get("materials.eps_r", 1.0)
                    target = 1 / eps if name.startswith("aniso_inv_eps") else eps
                    # Node smoothing owns off-diagonal and PEC keep records; compare
                    # explicit diagonal components here, not its geometric metadata.
                    if ".nodes." in name:
                        continue
                    bad = ~np.isclose(host, target, rtol=2e-7, atol=0)
                    component = name.rsplit(".", 1)[-1]
                    if component in ("0", "1", "2") and host.ndim == 3:
                        # Wall smoothing averages with exterior vacuum only on
                        # tangential edges that the PEC update pins to zero.
                        pinned = np.asarray(section(wall_masks[int(component)], bounds)[0])
                        bad &= ~pinned
                if np.any(bad):
                    first = tuple(np.argwhere(bad)[0])
                    index = (tuple(int(first[start+a] + offsets[start+a]) for a in range(3))
                             if start >= 0 else tuple(lo for lo, _ in bounds))
                    raise ValueError(prefix + f"{label} ({name})={host[first]!r} at cell {index}; " + reason)
            medium = PortMedium(scalars.get("materials.eps_r", 1.), scalars.get("materials.mu_r", 1.))
            if traced and medium != cfg.medium:
                raise ValueError(prefix + "traced override over the segment cannot be judged before the run; " + reason)
            if (multimode or cfg.multimode) and medium != PortMedium():
                raise ValueError(prefix + f"eps_r={medium.eps_r}, mu_r={medium.mu_r}: multimode filled ports are unsupported; " + reason)
            return medium

        probe_bounds = list(bounds)
        probe_bounds["xyz".index(cfg.normal_axis)] = (cfg.probe_x, cfg.probe_x + 1)
        probe_arrays = [section(array, probe_bounds)[0] for _, array, _, _ in operands]

        def probe_medium(medium, *values):
            matches, descriptions = True, []
            for (name, _, vacuum, label), value in zip(operands, values):
                value = np.asarray(value)
                if name == "materials.eps_r":
                    target = medium.eps_r
                elif name == "materials.mu_r":
                    target = medium.mu_r
                elif vacuum is not None:
                    target = vacuum
                else:
                    if ".nodes." in name:
                        continue
                    target = 1 / medium.eps_r if name.startswith("aniso_inv_eps") else medium.eps_r
                bad = ~np.isclose(value, target, rtol=2e-7, atol=0)
                component = name.rsplit(".", 1)[-1]
                if vacuum is None and component in ("0", "1", "2") and value.ndim == 3:
                    bad &= ~np.asarray(section(wall_masks[int(component)], probe_bounds)[0])
                if value.size == 0 or np.any(bad):
                    matches = False
                    descriptions.append(f"{label} ({name})={np.unique(value).tolist()}")
            return matches, "; ".join(descriptions)

        if traced:
            # The probe-plane medium cannot mark the config inside a trace, so a mismatch refuses the run.
            def judge_traced(*values):
                medium = judge(*values[:len(arrays)])
                matches, description = probe_medium(medium, *values[len(arrays):])
                if not matches:
                    raise ValueError(prefix + "traced override puts another medium at the probe plane ("
                                     + description + "); its S21 cannot be split; " + reason)
            traced_check("waveguide port", "section medium", judge_traced, *arrays, *probe_arrays, ordered=True)
            return cfg
        medium = judge(*arrays)
        cfg = with_port_medium(cfg, medium)
        matches, description = probe_medium(medium, *probe_arrays)
        return cfg._replace(probe_medium_matches=matches, probe_medium_description=description)

    return [[resolve(cfg, True) for cfg in group] if isinstance(group, list)
            else resolve(group, False) for group in waveguide_ports]
