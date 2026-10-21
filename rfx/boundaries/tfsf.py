"""Admission of TF/SF transverse boundary replacements (Addendum 5).

Only the realized update arrays can establish invariance. Declared geometry
and read-only monitors are not used as proxies for those arrays.
"""
from dataclasses import fields, is_dataclass
from fnmatch import fnmatchcase
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


def _refuse(axis, feature, *, nonuniform=False):
    if nonuniform:
        raise ValueError(
            f"TF/SF {axis}_lo, {axis}_hi: {feature} prevents an invariant periodic "
            "replacement; on the graded mesh a finite scatterer or an array under "
            "a plane wave is not available: run it on the uniform mesh "
            f"(closed_box=True, or {axis}='periodic')")
    raise ValueError(
        f"TF/SF {axis}_lo, {axis}_hi: {feature} prevents an invariant periodic "
        f"replacement; declare boundary with {axis}='periodic' to solve an array, "
        "or use closed_box=True for a finite scatterer.")


def replacement_axes(cfg, grid, *, requirements=(), records=None, nonuniform=False):
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
                        _refuse(axis, requirement.feature, nonuniform=nonuniform)
            result.append(axis)
        elif pair != (Kind.PERIODIC,) * 2:
            raise ValueError(
                f"TF/SF {axis}_lo, {axis}_hi must be a pair in "
                f"{{{', '.join(kind.value for kind in allowed)}}}; got "
                f"{pair[0].value}, {pair[1].value}.")
    return tuple(result)


def check_arrays(cfg, grid, values, *, localized=(), requirements=(), records=None,
                 active_axes=None, nonuniform=False):
    axes = replacement_axes(cfg, grid, requirements=requirements, records=records,
                            nonuniform=nonuniform)
    if active_axes is not None:
        axes = tuple(axis for axis in axes if axis in active_axes)
    for axis in axes:
        if isinstance(values, dict):
            for name in ("design_box", "design_occupancy"):
                box = values.get(name)
                if box is not None:
                    index = "xyz".index(axis)
                    if tuple(box.bounds[2*index:2*index+2]) != (0, grid.shape[index]):
                        _refuse(axis, name, nonuniform=nonuniform)
                    shape = tuple(box.bounds[2*i+1] - box.bounds[2*i] for i in range(3))
                    okay, feature = invariant(box, index, shape)
                    if okay is False:
                        _refuse(axis, feature, nonuniform=nonuniform)
        for name, updates in localized:
            if updates is not None and len(updates):
                _refuse(axis, name, nonuniform=nonuniform)
        okay, feature = invariant(values, "xyz".index(axis), grid.shape)
        if okay is None:
            operator = "existing graded" if nonuniform else "legacy periodic"
            warnings.warn(
                f"TF/SF {axis}_lo, {axis}_hi: invariance was not judged for traced "
                f"{feature}; the {operator} operator is retained (Addendum 5c).",
                UnjudgedTFSFInvarianceWarning, stacklevel=2)
        elif not okay:
            materials = values.get('materials') if isinstance(values, dict) else None
            tensor = getattr(getattr(materials, 'components', None), 'tensor', None)
            if (getattr(tensor, 'node_smoothing', None) is not None
                    and ('tensor.eps.' in feature or 'aniso_eps.' in feature)
                    and getattr(cfg, 'angle_deg', 0) != 0
                    and any(getattr(grid, f'pad_{axis}_{side}') for side in ('lo', 'hi'))):
                raise ValueError(
                    "full_tensor's diagonal is not invariant along a padded periodic axis "
                    f"({axis}); use closed_box=True for a finite scatterer")
            _refuse(axis, feature, nonuniform=nonuniform)
    return axes


def admit_simulation(sim, *, root=None, materials=None, material_overrides=()):
    """Dispatch and preflight use the production rasterization, never shapes."""
    # Both meshes require invariance; the graded operator retains its absorbers.
    if sim._tfsf is None or sim._tfsf.closed_box:
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
                            nonuniform=sim._uses_nonuniform_mesh,
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


# (vacuum value / adapter, user-facing label). None marks configuration or
# observation, never an exemption inferred from an unrecognized parameter.
INJECTION_OPERANDS = {
    **{name: (None, "configuration/source/observation") for name in (
        "grid", "n_steps", "boundary", "cpml_axes", "pec_axes", "periodic",
        "tfsf", "_feature_tfsf", "sources", "mag_sources", "probes", "dft_planes",
        "flux_monitors", "waveguide_ports", "ntff", "ntff_box", "ntff_data",
        "current_moments", "field_dtype", "stencil_order", "aniso_inv_eps_smooth",
        "wire_port_sparams", "lumped_port_sparams", "wire_refplane_sparams",
        "wire_ports", "s_param_freqs", "pec_faces", "pmc_faces", "emit_time_series",
        "rlc_states")},
    "materials": ("leaves", "material"),
    "debye": ("poles", "Debye pole medium"),
    "lorentz": ("poles", "Drude/Lorentz pole medium"),
    "kerr_chi3": (0, "Kerr (chi3) medium"),
    "pec_mask": (0, "PEC volume"),
    "pec_edge_masks": (0, "PEC edge (volume, sheet, wire or pinned sheet)"),
    "pec_sheets": ("realized", "PEC edges are checked through pec_edge_masks"),
    "pec_wires": ("realized", "PEC edges are checked through pec_edge_masks"),
    "pec_occupancy": (0, "PEC occupancy"),
    "aniso_eps": (1, "anisotropic permittivity"),
    "aniso_inv_eps": (1, "inverse permittivity"),
    "conformal_weights": (1, "conformal PEC"),
    "sheet_impedance": ("sheet", "surface-impedance sheet"),
    "design_box": ("window", "design window"),
    "design_occupancy": ("window", "design occupancy window"),
    "lumped_rlc": ("cells", "lumped RLC element"),
    "rlc_metas": ("cells", "lumped RLC element"),
    "pinned_sheet_layout": ("layout", "pinned sheet layout"),
    "nonvacuum_masks": (0, "caller-supplied non-vacuum mask"),
    **{f"materials.{name}": (value, label) for name, value, label in (
        ("eps_r", 1, "permittivity"), ("sigma", 0, "conductivity"),
        ("mu_r", 1, "permeability"), ("sigma_lumped.*", 0, "lumped conductivity"),
        ("eps_r_lumped.*", 0, "lumped permittivity"),
        ("mu_r_wire.*", 0, "wire permeability"),
        ("edge_keep.*", 1, "PEC edge keep"), ("sigma_film.*", 0, "film conductivity"))},
    **{f"materials.components.{name}.*": (value, label) for name, value, label in (
        ("eps", 1, "component permittivity"), ("sigma", 0, "component conductivity"),
        ("mu", 1, "component permeability"), ("eps_lumped", 0, "lumped permittivity"),
        ("sigma_lumped", 0, "lumped conductivity"), ("mu_wire", 0, "wire permeability"),
        ("eps_update", 1, "component permittivity"),
        ("sigma_update", 0, "component conductivity"),
        ("mu_update", 1, "component permeability"),
        ("upml_eps", 1, "component permittivity"),
        ("upml_sigma", 0, "component conductivity"))},
    **{f"materials.components.{kind}.{field}.*": (0, label)
       for kind, label in (("debye", "Debye pole medium"),
                           ("lorentz", "Drude/Lorentz pole medium"))
       for field in ("coefficients", "weights")},
    **{f"materials.components.tensor.{name}.*": (1, "tensor permittivity")
       for name in ("eps", "inv_eps", "conformal_weights")},
    **{f"materials.components.tensor.node_smoothing.{name}*": (value, "node tensor")
       for name, value in (("xi", "symmetric_identity"), ("triplets", "identity"),
                           ("bulk", 1), ("active", 0), ("cross", 0), ("pec_keep", 1),
                           ("guard_count", None))},
}


def injection_material_rule(name):
    """Classify every spatial material leaf, including declaration-time views."""
    name = name.replace(".components.cells_view", "")
    for pattern, row in INJECTION_OPERANDS.items():
        if fnmatchcase(name, pattern):
            return row
    raise ValueError(f"Unclassified TFSF material operand: {name}")


# Values are (E/H family, component axis). These indices never name a pole, tensor entry,
# triplet or spatial axis. Builders: model/materials.py::realize_components,
# model/tensor_operands.py, geometry/node_tensor.py, model/node_pec.py,
# boundaries/pec.py, model/occupancy.py, sources/{sources,wire_radius}.py,
# model/thin_conductors.py, materials/thin_conductor.py and lumped.py.
_INJECTION_COMPONENTS = {
    **{f"{family}.{c}": ("E", axis) for c, axis in enumerate("xyz") for family in (
        "pec_edge_masks", "aniso_eps", "aniso_inv_eps", "conformal_weights",
        *(f"materials.{name}" for name in (
            "sigma_lumped", "eps_r_lumped", "edge_keep", "sigma_film")),
        *(f"materials.components.{name}" for name in (
            "eps", "sigma", "eps_update", "sigma_update",
            "eps_lumped", "sigma_lumped", "upml_eps", "upml_sigma")),
        *(f"materials.components.tensor.{name}" for name in (
            "eps", "inv_eps", "conformal_weights")),
        *(f"materials.components.tensor.node_smoothing.{name}" for name in (
            "bulk", "active", "pec_keep")))},
    **{f"{family}.{c}": ("H", axis) for c, axis in enumerate("xyz") for family in (
        "materials.components.mu", "materials.components.mu_update",
        "materials.components.mu_wire", "materials.mu_r_wire")},
    **{f"sheet_impedance.mask_e{axis}": ("E", axis) for axis in "xyz"},
    **{f"{family}.e{axis}": ("E", axis) for family in ("lumped_rlc", "rlc_metas") for axis in "xyz"},
}


def _injection_arrays(grid, materials, updates, localized):
    """Yield actual spatial operands with their vacuum values and labels."""
    for name, array in _leaves(materials, "materials"):
        if array.ndim < 3 and name not in ("materials.eps_r", "materials.sigma", "materials.mu_r"):
            continue
        vacuum, label = injection_material_rule(name)
        if vacuum == "symmetric_identity":
            # Symmetric tensor order: xx, yy, zz, xy, xz, yz.
            vacuum = int(int(name.rsplit(".", 1)[1]) < 3)
        if vacuum is not None:
            yield name, array, vacuum, label
    for name, value in (*updates.items(), *localized):
        vacuum, label = INJECTION_OPERANDS[name]
        if value is None or vacuum is None or vacuum == "realized":
            continue
        if vacuum == "poles":
            coeffs = value[0]
            # Setups receive ADE coefficients, whose masked recurrence terms
            # carry the realized pole reach (not the initial, zero P state).
            names = ("alpha", "beta") if name == "debye" else ("a", "b", "c")
            value = {key: getattr(coeffs, key) for key in names}
            vacuum = 0
        elif vacuum == "sheet":
            value = {f"mask_e{a}": getattr(value, f"mask_e{a}") for a in "xyz"}
            vacuum = 0
        elif vacuum == "cells":
            import jax.numpy as jnp
            masks = {}
            for meta in value:
                mask = masks.get(meta.component, jnp.zeros(grid.shape, dtype=bool))
                masks[meta.component] = mask.at[meta.i, meta.j, meta.k].set(True)
            value, vacuum = masks, 0
        elif vacuum in ("window", "layout"):
            import jax.numpy as jnp
            mask = jnp.zeros(grid.shape, dtype=bool)
            if vacuum == "window":
                mask = mask.at[tuple(slice(*value.bounds[2*a:2*a+2]) for a in range(3))].set(True)
            else:
                from rfx.model.occupancy import layout_edge_indices
                for _, sheet, pair in value:
                    for tangent, values in enumerate(pair):
                        _, indices = layout_edge_indices(grid, sheet, tangent)
                        mask = mask.at[indices].max(values != 0)
            value, vacuum = mask, 0
        for operand, array in _leaves(value, name):
            yield operand, array, vacuum, label


def refuse_injection_plane_operands(cfg, grid, materials, updates, localized):
    """Refuse any non-vacuum update on a corrected TF/SF plane, also in AD."""
    from rfx._traced_check import traced_check
    from rfx.sources.tfsf import is_tfsf_methodB, tfsf_injection_planes
    from types import SimpleNamespace
    if grid is None:
        grid = SimpleNamespace(shape=materials.eps_r.shape)
    bounds = tfsf_injection_planes(cfg)
    closed = getattr(cfg, "closed_box", False)
    method_b = is_tfsf_methodB(cfg)
    if closed:
        feature, check_name = "closed_box TFSF", "vacuum faces"
        prefix = "closed_box TFSF requires vacuum on and adjacent to all six faces; non-vacuum material found at "
    elif method_b:
        feature, check_name = "Method-B oblique TFSF", "vacuum boundaries"
        prefix = ("Method-B oblique TFSF requires vacuum on and adjacent to all four transverse "
                  "TFSF boundary planes; non-vacuum material found at ")
    else:
        feature, check_name = "TFSF slab", "vacuum boundaries"
        prefix = ("TFSF plane-wave source requires vacuum on and adjacent to the TFSF x boundaries; "
                  "non-vacuum material found at ")
    operands = tuple(_injection_arrays(grid, materials, updates, localized))
    arrays, metadata = [], []
    for axis, (lo, hi) in bounds.items():
        for index, suffix in zip((lo - 1, lo, hi, hi + 1), ("lo-1", "lo", "hi", "hi+1")):
            plane = f"{axis}={index}" if closed else f"{axis}_{suffix}"
            spatial_slice = [slice(a - 1, b + 2) for a, b in bounds.values()] if closed else [slice(None)] * 3
            spatial_slice["xyz".index(axis)] = slice(index, index + 1)
            for name, array, vacuum, label in operands:
                field, component = _INJECTION_COMPONENTS.get(
                    name.replace(".components.cells_view", ""), (None, None))
                if suffix in ("lo", "hi") and (
                        component == axis or (field, suffix) in (("E", "hi"), ("H", "lo"))):
                    continue  # Uncorrected total-field component; adjacent planes still count.
                # Most leaves end in xyz; node triplets end in xyz,3,3.
                trailing = 2 if vacuum == "identity" else 0
                start = array.ndim - 3 - trailing
                if start < 0:
                    cut = array  # homogeneous scalar material
                else:
                    slices = [slice(None)] * array.ndim
                    for a, sl in enumerate(spatial_slice):
                        slices[start + a] = slice(None) if array.shape[start + a] == 1 else sl
                    cut = array[tuple(slices)]
                arrays.append(cut)
                metadata.append((plane, name, vacuum, label))

    def check(*values):
        for value, (plane, name, vacuum, label) in zip(values, metadata):
            expected = np.eye(3) if vacuum == "identity" else vacuum
            if not np.all(np.asarray(value) == expected):
                raise ValueError(f"{prefix}{plane}: {label} ({name})")

    traced_check(feature, check_name, check, *arrays, ordered=True)


def admit_setup(*, grid, tfsf, materials, waveguide_ports, ntff, updates,
                localized, periodic, pec_faces=None, pmc_faces=None,
                feature_owned=False, nonuniform=False):
    """Last check on actual low-level inputs, including AD material overrides."""
    from rfx.boundaries.features import admit_grid_waveguide
    admit_grid_waveguide(grid, waveguide_ports)
    if tfsf is None:
        return
    cfg = tfsf[0]
    refuse_injection_plane_operands(cfg, grid, materials, updates, localized)
    if ntff is not None:
        from rfx.farfield import require_box_encloses_injected_region
        from rfx.sources.tfsf import tfsf_injection_planes
        require_box_encloses_injected_region(
            ntff, tfsf_injection_planes(cfg), shape=grid.shape)
    if getattr(cfg, "closed_box", False) or not feature_owned:
        # Explicit low-level periodic flags are the caller's declaration,
        # not a feature rewrite (Addendum 5c(5)).
        return
    check_arrays(cfg, grid, dict(materials=materials, **updates),
                 # The low-level RCS path does not install the slab wrap.
                 # Its open transverse operator is unchanged in PR3a. The
                 # graded mesh keeps its transverse absorbers (a periodic
                 # axis is refused at admission), so it judges the axes
                 # admit_simulation judges.
                 active_axes=None if nonuniform else tuple(
                     axis for axis, wraps in zip("xyz", periodic) if wraps),
                 records=grid_face_depths(grid, pec_faces=pec_faces, pmc_faces=pmc_faces),
                 localized=localized, nonuniform=nonuniform)
