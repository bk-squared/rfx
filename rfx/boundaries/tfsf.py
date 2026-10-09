"""Admission of TF/SF transverse boundary replacements (Addendum 5).

Only the realized update arrays can establish invariance. Declared geometry
and read-only monitors are not used as proxies for those arrays.
"""
from dataclasses import fields, is_dataclass
import warnings

import numpy as np

from rfx.boundaries.depths import Kind, grid_face_depths


def validate_source_boundary(sim, *, closed_box):
    """Addendum 5c does not admit the previously unsupported UPML source."""
    if any(token == "upml" for _, _, token in sim._boundary_spec.faces()):
        raise ValueError("TFSF plane-wave source refuses boundary='upml'")
    if closed_box and sim._boundary != "cpml":
        raise ValueError("Closed-box TFSF requires boundary='cpml'")


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


class UnjudgedTFSFInvarianceWarning(UserWarning):
    """The legacy operator is retained for a material unavailable at tracing."""


def _without_duplicated_end_edge(name, host, axis, index):
    """Ignore only main's empty final edge in that mask component's direction.

    Addendum 5c(3): the arrays sent to the kernel remain untouched. This is
    not an exemption for slots, other components, materials, or pad rows.
    """
    mask = any(name.endswith(f"{family}.{axis}") for family in
               ("pec_edge_masks", "conductor_edges"))
    if mask and host.dtype == np.bool_ and not np.any(np.take(host, -1, axis=index)):
        return np.take(host, np.arange(host.shape[index] - 1), axis=index)
    return host


def invariant(value, axis, shape):
    """Exact equality over each full spatial array, pads included.

    Unknown traced inputs return None; Addendum 5c(5) preserves their legacy
    operator and reports that invariance was not judged.
    """
    unjudged = ""
    for name, array in _leaves(value, "update"):
        dims = tuple(array.shape)
        # ADE coefficients and states store poles before the three spatial
        # dimensions. A pole count equal to nx must not be mistaken for x.
        # Both material arrays and pole coefficients can broadcast singleton
        # spatial dimensions. Non-singleton dimensions must remain invariant.
        spatial = (len(dims) - 3 if len(dims) >= 3
                   and all(n in (1, size) for n, size in zip(dims[-3:], shape)) else None)
        if spatial is None:
            continue
        index = spatial + axis
        if dims[index] == 1:
            continue  # broadcasting is constant even when its value is traced
        # JVP/linearization carries a concrete primal when the design value
        # is known. Inspect that realized value without changing its trace.
        while hasattr(array, "primal"):
            array = array.primal
        try:
            host = np.asarray(array)
        except Exception as exc:
            from jax.errors import TracerArrayConversionError
            if isinstance(exc, TracerArrayConversionError):
                unjudged = name
                continue
            return False, f"{name} ({type(exc).__name__}: invariance unavailable)"
        host = _without_duplicated_end_edge(name, host, axis, index)
        first = np.take(host, [0], axis=index)
        if not np.array_equal(host, np.broadcast_to(first, host.shape)):
            return False, name
    return (None, unjudged) if unjudged else (True, "")


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
    # Propagation-axis walls keep main's operator (Addendum 5c(5)).
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
        if pair in ((Kind.ABSORBER,) * 2, (wall,) * 2):
            for requirement in requirements:
                if any(face.startswith(axis + "_") for face in requirement.faces):
                    if Kind.PERIODIC not in requirement.admissible:
                        _refuse(axis, requirement.feature)
            result.append(axis)
        elif pair != (Kind.PERIODIC,) * 2:
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
                    if okay is False:
                        _refuse(axis, feature)
        for name, updates in localized:
            if updates is not None and len(updates):
                _refuse(axis, name)
        okay, feature = invariant(values, "xyz".index(axis), grid.shape)
        if okay is None:
            warnings.warn(
                f"TF/SF {axis}_lo, {axis}_hi: invariance was not judged for traced "
                f"{feature}; the legacy periodic operator is retained (Addendum 5c).",
                UnjudgedTFSFInvarianceWarning, stacklevel=2)
        elif not okay:
            _refuse(axis, feature)
    return axes


def admit_simulation(sim, *, root=None, materials=None, material_overrides=()):
    """Dispatch and preflight use the production rasterization, never shapes."""
    # Addendum 5e: the graded operator retains absorbers, with no wrap to admit.
    if sim._tfsf is None or sim._tfsf.closed_box or sim._uses_nonuniform_mesh:
        return ()
    import jax
    from rfx.boundaries.model import collect_requirements
    from rfx.model.conductors import realized_conductors
    with jax.ensure_compile_time_eval():
        if root is None:
            root = realized_conductors(sim, sim._build_realized_grid(),
                                      nonuniform=sim._uses_nonuniform_mesh)
        names = ("materials", "debye", "lorentz", "pec_mask", "pec_shapes",
                 "boundary_pec_shapes", "kerr_chi3")
        values = {names[i] if i < len(names) else f"assembly[{i}]": value
                  for i, value in enumerate(root.assembly)}
        if materials is not None:
            values["materials"] = materials
        replacements = {name: value for name, value in
                        zip(("eps_r", "sigma", "mu_r"), material_overrides)
                        if value is not None}
        # Match forward's replacement semantics, including removal of folded
        # material views. Unoverridden components and conductors still count.
        if "eps_r" in replacements:
            replacements["eps_r_lumped"] = None
        if "sigma" in replacements:
            replacements["sigma_lumped"] = None
        if replacements:
            values["materials"] = values["materials"]._replace(**replacements)
        values.update(pec_edge_masks=root.pec_edges, sheet_impedance=root.sheet_impedance)
        return check_arrays(sim._tfsf, root.grid, values,
                            localized=tuple((name, getattr(sim, name, ())) for name in
                                            ("_ports", "_lumped_rlc", "_waveguide_ports",
                                             "_msl_ports", "_coaxial_ports", "_floquet_ports")),
                            requirements=collect_requirements(sim),
                            records=grid_face_depths(root.grid, spec=sim._boundary_spec))


def report(sim, issues, *, root=None, material_overrides=()):
    from rfx.preflight._common import PreflightIssue
    axes = admit_simulation(sim, root=root, material_overrides=material_overrides)
    note = ""
    if (not sim._uses_nonuniform_mesh and sim._tfsf is not None
            and not sim._tfsf.closed_box and sim._tfsf.angle_deg != 0):
        grid = root.grid if root is not None else sim._build_realized_grid()
        # The registration entry, not an initialized config: open oblique
        # Method B absorbs on y and wraps z only (sources/tfsf.py).
        wrapped = (False, sim._tfsf.method != "methodB", True)
        axes = tuple(axis for axis, wraps, size in
                     zip("xyz", wrapped, grid.shape) if wraps and size > 1)
        tilt = "y" if sim._tfsf.polarization == "ez" else "z"
        if tilt in axes:
            note = f" Structure not judged for invariance along {tilt} at oblique incidence."
    if axes and not sim._uses_nonuniform_mesh:
        faces = ", ".join(f"{axis}_{side}" for axis in axes for side in ("lo", "hi"))
        issues.append(PreflightIssue(
            f"TF/SF faces {faces} are solved periodic, including their pads." + note,
            severity="warning", code="tfsf_transverse_periodic"))


def boundary_flags(cfg, grid=None):
    from rfx.sources.tfsf import is_tfsf_methodB
    if getattr(cfg, "closed_box", False):
        return (False, False, False), "xyz"
    # Addendum 5c(2): admitted declared walls retain main's wrap. Finite
    # transverse structure is refused before this operator is selected.
    return (False, not is_tfsf_methodB(cfg), True), "xy" if is_tfsf_methodB(cfg) else "x"


def admit_setup(*, grid, tfsf, materials, waveguide_ports, ntff, updates,
                localized, periodic, pec_faces=None, pmc_faces=None,
                feature_owned=False, nonuniform=False):
    """Last check on actual low-level inputs, including AD material overrides."""
    from rfx.boundaries.features import admit_grid_waveguide
    admit_grid_waveguide(grid, waveguide_ports)
    if tfsf is None:
        return
    cfg = tfsf[0]
    if ntff is not None:
        from rfx.farfield import require_box_encloses_injected_region
        from rfx.sources.tfsf import tfsf_injection_planes
        require_box_encloses_injected_region(
            ntff, tfsf_injection_planes(cfg), shape=grid.shape)
    if getattr(cfg, "closed_box", False) or not feature_owned or nonuniform:
        # Explicit low-level periodic flags are the caller's declaration,
        # not a feature rewrite (Addendum 5c(5)).
        # The graded operator installs no periodic wrap (Addendum 5e).
        return
    check_arrays(cfg, grid, dict(materials=materials, **updates),
                 # The low-level RCS path does not install the slab wrap.
                 # Its open transverse operator is unchanged in PR3a.
                 active_axes=tuple(
                     axis for axis, wraps in zip("xyz", periodic) if wraps),
                 records=grid_face_depths(grid, pec_faces=pec_faces, pmc_faces=pmc_faces),
                 localized=localized)
