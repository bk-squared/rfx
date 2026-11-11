"""Reference metadata for inputs whose live geometry cannot be read on host."""
from dataclasses import fields, is_dataclass
from weakref import WeakKeyDictionary, ref


# Host metadata follows simulation lifetime without becoming solve arguments.
_kernel_views = WeakKeyDictionary()


def forget_kernel(sim):
    _kernel_views.pop(sim, None)


class _DeclaredDomain(tuple):
    """Solved extents with the caller's original extents retained for reporting."""


def retain_declared_domain(domain, declared):
    if domain is declared:
        return domain
    result = _DeclaredDomain(domain)
    result.declared = tuple(declared)
    return result


from jax.tree_util import register_pytree_node
register_pytree_node(_DeclaredDomain, lambda domain: (tuple(domain), domain.declared),
                    lambda declared, values: retain_declared_domain(tuple(values), declared))


def remember_kernel(sim, root, record):
    """Keep compact host rows and lend the live kernel owner without owning it."""
    from rfx._declarations import declaration_key
    from rfx.model.realization_integrals import quantities
    object.__setattr__(record, "_conductors", ref(root))
    if record.lane.startswith("fwd_") or any(row.traced_inputs for row in record.realization_table.rows):
        return
    store = quantities(root.grid)
    _kernel_views[sim] = record, store.get("smoothing", 0), declaration_key(sim), store.get("conformal")


def kernel_view(sim):
    """Prepare requested arrays afresh and verify the solved table's rows."""
    from rfx._declarations import declaration_key
    cached = _kernel_views.get(sim)
    if cached is None:
        return None
    record, smoothing, key, conformal = cached
    if key != declaration_key(sim):
        forget_kernel(sim)
        return None
    from rfx.model.realization_report import rebuild_view
    return rebuild_view(sim, record, smoothing, conformal)


def table_traced_inputs(sim, grid):
    """Mesh and geometry-shape tracers: no row of the table can be read on host."""
    import jax
    from rfx.core.jax_utils import is_tracer
    result = []
    for axis, name in zip("xyz", ("dx_arr", "dy_arr", "dz")):
        if is_tracer(getattr(grid, name, None)):
            result.append(f"mesh.{axis}")
    for i, entry in enumerate(sim._geometry):
        shape = entry.shape
        for f in fields(shape) if is_dataclass(shape) else ():
            if any(is_tracer(v) for v in jax.tree.leaves(getattr(shape, f.name))):
                result.append(f"geometry[{i}].{f.name}")
    return tuple(result)


def f0_traced_fields(entry, spec):
    """The f0 width regime is unavailable when either declaration or G is live."""
    from rfx.core.jax_utils import is_tracer
    return tuple(name for name, value in (
        ("surface_impedance_f0", entry.surface_impedance_f0),
        ("sigma_bulk", entry.sigma_bulk),
        ("g_sheet", getattr(spec, "g_sheet", None))) if is_tracer(value))


def sheet_traced_inputs(sim, conductors):
    """``thin_conductor`` index -> its traced f0 material inputs (that row only)."""
    specs = iter(getattr(conductors, "sheet_impedance", ()))
    result = {}
    for i, entry in enumerate(sim._thin_conductors):
        if entry.surface_impedance_f0 is None:
            continue
        names = f0_traced_fields(entry, next(specs, None))
        if names:
            result[i] = tuple(f"thin_conductor[{i}].{name}" for name in names)
    return result


def table_wide(rows):
    """Whether the whole table is unavailable, not only some f0 sheets' rows."""
    return any(row.traced_inputs for row in rows if not row.label.startswith("thin_conductor["))


def traced_inputs(sim, grid, conductors=None):
    """Every traced input: table-wide ones, then each f0 sheet's own."""
    sheets = sheet_traced_inputs(sim, conductors)
    return table_traced_inputs(sim, grid) + tuple(n for i in sorted(sheets) for n in sheets[i])


def nominal_record(sim, conductors, lane, *, compact):
    """Resolve an explicit nominal mesh through the existing family producers."""
    import copy
    import jax
    import warnings
    from rfx.core.jax_utils import is_tracer
    from rfx.errors import RfxRefusal
    from rfx.model.conductors import realized_conductors, preview_port_stages
    from rfx.realized_geometry import record_from_conductors
    inputs = traced_inputs(sim, conductors.grid, conductors)
    nominal = getattr(conductors.grid, "nominal_cells", None) or {}
    axes = [name[-1] for name in inputs if name.startswith("mesh.")]
    if not axes or any(not name.startswith("mesh.") for name in inputs):
        return None
    if any(axis not in nominal for axis in axes):
        return None
    reference = copy.copy(sim)
    for axis in axes:
        setattr(reference, f"_d{axis}_profile", nominal[axis])
    reference._nominal_profiles = None
    if is_tracer(getattr(reference, "_dt_pin", None)):
        return None
    # The nominal declaration is a reference point, not another live solve.
    with warnings.catch_warnings(), jax.ensure_compile_time_eval():
        warnings.simplefilter("ignore")
        grid = reference._build_nonuniform_grid()
        try:
            root = realized_conductors(reference, grid, nonuniform=True)
            root = preview_port_stages(reference, root)
        except RfxRefusal:
            return None
        from rfx.model.realization_integrals import quantities
        mode = quantities(conductors.grid).get("smoothing", 0)
        if mode:
            from rfx.geometry.smoothing import smoothed_shape_pairs, compute_smoothed_eps_nonuniform
            from rfx.model.realization_integrals import begin
            pairs, _ = smoothed_shape_pairs(reference, grid)
            begin(reference, grid, shapes=pairs)
            if mode == 2:
                from rfx.model.tensor_operands import full_tensor_eps
                full_tensor_eps(reference, grid, root.materials, pairs)
            else:
                from rfx.model.tensor_operands import smoothing_nominal_h_max
                compute_smoothed_eps_nonuniform(
                    grid, pairs, nominal_h_max=smoothing_nominal_h_max(reference, grid))
        record = record_from_conductors(reference, root, lane=lane, compact=compact)
        if record is not None:
            object.__setattr__(record, "_quantity_grid", grid)
        return record
