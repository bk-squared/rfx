"""Admission of TF/SF transverse boundary replacements (Addendum 5).

Only the realized update arrays can establish invariance. Declared geometry
and read-only monitors are not used as proxies for those arrays.
"""
from dataclasses import fields, is_dataclass

import numpy as np

from rfx.boundaries.depths import Kind, grid_face_depths


def _leaves(value, name):
    if value is None:
        return
    if hasattr(value, "shape"):
        yield name, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _leaves(getattr(value, field.name), f"{name}.{field.name}")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        names = getattr(value, "_fields", range(len(value)))
        for key, child in zip(names, value):
            yield from _leaves(child, f"{name}.{key}")


def invariant(value, axis, shape):
    """Exact equality over each full spatial array, pads included.

    Unknown traced inputs cannot establish this predicate. A caller using a
    traced transverse structure must explicitly declare its periodic axis.
    """
    for name, array in _leaves(value, "update"):
        dims = tuple(array.shape)
        spatial = next((offset for offset in range(len(dims) - 2)
                        if dims[offset:offset + 3] == tuple(shape)), None)
        if spatial is None:
            continue
        # JVP/linearization carries a concrete primal when the design value
        # is known. Inspect that realized value without changing its trace.
        while hasattr(array, "primal"):
            array = array.primal
        try:
            host = np.asarray(array)
        except Exception as exc:
            return False, f"{name} ({type(exc).__name__}: invariance unavailable)"
        index = spatial + axis
        first = np.take(host, [0], axis=index)
        if not np.array_equal(host, np.broadcast_to(first, host.shape)):
            return False, name
    return True, ""


def _refuse(axis, feature):
    raise ValueError(
        f"TF/SF {axis}_lo, {axis}_hi: {feature} prevents an invariant periodic "
        f"replacement; declare boundary with {axis}='periodic' to solve an array, "
        "or use closed_box=True for a finite scatterer.")


def replacement_axes(cfg, grid, *, requirements=(), records=None):
    """Validate admissible face pairs and return absorber axes to check."""
    if getattr(cfg, "closed_box", False):
        return ()
    electric = getattr(cfg, "electric_component", getattr(cfg, "polarization", "ez"))[-1]
    magnetic = "y" if electric == "z" else "z"
    records = {face.name: face for face in (grid_face_depths(grid) if records is None else records)}
    for side in ("lo", "hi"):
        face = records[f"x_{side}"]
        if face.kind != Kind.ABSORBER or face.realized == 0:
            raise ValueError(f"TF/SF x_{side} requires {{ABSORBER}} with positive realized depth.")
    result = []
    for axis, wall in ((magnetic, Kind.PMC), (electric, Kind.PEC)):
        # Legacy rewrite, PR3b: the oblique transverse wavevector is not
        # invariant. Addendum 5 leaves that operator unchanged.
        if getattr(cfg, "angle_deg", 0) != 0 and axis == magnetic:
            continue
        if grid.shape["xyz".index(axis)] == 1:
            continue  # the existing invariant 2-D lane is PR3b
        pair = tuple(records[f"{axis}_{side}"].kind for side in ("lo", "hi"))
        allowed = (Kind.PERIODIC, wall)
        if pair == (Kind.ABSORBER, Kind.ABSORBER):
            for requirement in requirements:
                if any(face.startswith(axis + "_") for face in requirement.faces):
                    if Kind.PERIODIC not in requirement.admissible:
                        _refuse(axis, requirement.feature)
            result.append(axis)
        elif pair not in ((Kind.PERIODIC,) * 2, (wall,) * 2):
            raise ValueError(
                f"TF/SF {axis}_lo, {axis}_hi must be a pair in "
                f"{{{', '.join(kind.value for kind in allowed)}}}; got "
                f"{pair[0].value}, {pair[1].value}.")
    return tuple(result)


def check_arrays(cfg, grid, values, *, localized=(), requirements=(), records=None, active_axes=None):
    axes = replacement_axes(cfg, grid, requirements=requirements, records=records)
    if active_axes is not None:
        axes = tuple(axis for axis in axes if axis in active_axes)
    for axis in axes:
        if isinstance(values, dict):
            for name in ("design_box", "design_occupancy"):
                box = values.get(name)
                if box is not None:
                    index = "xyz".index(axis)
                    if tuple(box.bounds[2*index:2*index+2]) != (0, grid.shape[index]):
                        _refuse(axis, name)
                    shape = tuple(box.bounds[2*i+1] - box.bounds[2*i] for i in range(3))
                    okay, feature = invariant(box, index, shape)
                    if not okay:
                        _refuse(axis, feature)
        for name, updates in localized:
            if updates is not None and len(updates):
                _refuse(axis, name)
        okay, feature = invariant(values, "xyz".index(axis), grid.shape)
        if not okay:
            _refuse(axis, feature)
    return axes


def admit_simulation(sim, *, root=None):
    """Dispatch and preflight use the production rasterization, never shapes."""
    if sim._tfsf is None or sim._tfsf.closed_box:
        return ()
    import jax
    from rfx.boundaries.model import collect_requirements
    from rfx.model.conductors import realized_conductors
    with jax.ensure_compile_time_eval():
        if root is None:
            root = realized_conductors(sim, sim._build_realized_grid(),
                                      nonuniform=sim._uses_nonuniform_mesh)
        return check_arrays(sim._tfsf, root.grid,
                            (root.assembly, root.pec_edges, root.sheet_impedance),
                            localized=tuple((name, getattr(sim, name, ())) for name in
                                            ("_ports", "_lumped_rlc", "_waveguide_ports",
                                             "_msl_ports", "_coaxial_ports", "_floquet_ports")),
                            requirements=collect_requirements(sim),
                            records=grid_face_depths(root.grid, spec=sim._boundary_spec))


def report(sim, issues, *, root=None):
    from rfx.preflight._common import PreflightIssue
    axes = admit_simulation(sim, root=root)
    if axes and not sim._uses_nonuniform_mesh:
        faces = ", ".join(f"{axis}_{side}" for axis in axes for side in ("lo", "hi"))
        issues.append(PreflightIssue(
            f"TF/SF invariant faces {faces} are solved periodic, including their pads.",
            severity="warning", code="tfsf_transverse_periodic"))


def boundary_flags(cfg, grid=None):
    from rfx.sources.tfsf import is_tfsf_methodB
    if getattr(cfg, "closed_box", False):
        return (False, False, False), "xyz"
    flags = [False, not is_tfsf_methodB(cfg), True]
    if grid is not None:
        records = grid_face_depths(grid)
        for axis in (1, 2):
            # Legacy rewrite, PR3b (Addendum 5): preserve the oblique
            # transverse-wavevector operator, including its padded wrap.
            if getattr(cfg, "angle_deg", 0) != 0 and "xyz"[axis] == getattr(cfg, "transverse_axis", "y"):
                continue
            if all(face.kind in (Kind.PEC, Kind.PMC) for face in records[2*axis:2*axis+2]):
                flags[axis] = False
    return tuple(flags), "xy" if is_tfsf_methodB(cfg) else "x"


def admit_setup(values, *, nonuniform=False):
    """Last check on actual low-level inputs, including AD material overrides."""
    from rfx.boundaries.features import admit_grid_waveguide
    admit_grid_waveguide(values["grid"], values.get("waveguide_ports"))
    if values.get("tfsf") is None:
        return
    cfg = values["tfsf"][0]
    if values["use_ntff"]:
        from rfx.farfield import require_box_encloses_injected_region
        from rfx.sources.tfsf import tfsf_injection_planes
        require_box_encloses_injected_region(
            values["ntff_box" if nonuniform else "ntff"],
            tfsf_injection_planes(cfg), shape=values["grid"].shape)
    if getattr(cfg, "closed_box", False):
        return
    arrays = {name: values.get(name) for name in (
        "materials", "debye", "lorentz", "aniso_eps", "aniso_inv_eps", "pec_mask",
        "pec_edge_masks", "pec_occupancy", "conformal_weights", "sheet_impedance",
        "kerr_chi3", "design_box", "design_occupancy")}
    from rfx.boundaries.pec import realized_pec_edge_masks
    if values.get("pec_sheets") or values.get("pec_wires"):
        arrays["conductor_edges"] = realized_pec_edge_masks(
            values.get("pec_mask"), sheets=values.get("pec_sheets") or (),
            wires=values.get("pec_wires") or (),
            periodic=values.get("periodic", (False, False, False)))
    check_arrays(cfg, values["grid"], arrays,
                 # The low-level RCS path does not install the slab wrap.
                 # Its open transverse operator is unchanged in PR3a.
                 active_axes=None if nonuniform else tuple(
                     axis for axis, wraps in zip("xyz", values["periodic"]) if wraps),
                 records=grid_face_depths(values["grid"], pec_faces=values.get("pec_faces"),
                                          pmc_faces=values.get("pmc_faces")),
                 localized=tuple((name, values.get(name)) for name in (
                     "sources", "mag_sources", "waveguide_ports", "wire_ports", "wire_port_sparams",
                     "lumped_port_sparams", "lumped_rlc", "rlc_metas")))
