"""Reference metadata for inputs whose live geometry cannot be read on host."""
from dataclasses import fields, is_dataclass
from weakref import proxy


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
    """Lend the concrete execution to subsequent diagnostic readers."""
    from rfx.api._declarations import declaration_key
    from rfx.model.realization_table import PreparedRealization
    if any(row.traced_inputs for row in record.realization_table.rows):
        return
    if "distributed" in record.lane:
        from rfx.model.realization_snapshot import snapshot
        value = snapshot(sim, root, record)
    else:
        value = PreparedRealization(proxy(sim), root, record.lane, record)
    sim._kernel_realization = declaration_key(sim), value


def kernel_view(sim):
    """Expand legacy diagnostic arrays without assembling or rebuilding the table."""
    from dataclasses import replace
    from rfx.api._declarations import declaration_key
    from rfx.model.realization_table import PreparedRealization
    from rfx.realized_geometry import record_from_conductors, _conductor_view
    cached = getattr(sim, "_kernel_realization", None)
    if cached is None or cached[0] != declaration_key(sim):
        return None
    key, value = cached
    from rfx.model.realization_snapshot import DistributedSnapshot
    if isinstance(value, DistributedSnapshot):
        value = value.restore()
        sim._kernel_realization = key, value
    if isinstance(value, PreparedRealization):
        record = record_from_conductors(sim, value.conductors, lane=value.lane, compact=False)
        if record is None:
            return None
        value = _conductor_view(replace(record, realization_table=value.record.realization_table),
                                value.conductors)
        sim._kernel_realization = key, value
    return value


def traced_inputs(sim, grid):
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


def nominal_record(sim, conductors, lane, *, compact):
    """Resolve an explicit nominal mesh through the existing family producers."""
    import copy
    import jax
    import warnings
    from rfx.core.jax_utils import is_tracer
    from rfx.errors import RfxRefusal
    from rfx.model.conductors import realized_conductors, preview_port_stages
    from rfx.realized_geometry import record_from_conductors
    inputs = traced_inputs(sim, conductors.grid)
    nominal = getattr(conductors.grid, "nominal_cells", None) or {}
    axes = [name[-1] for name in inputs if name.startswith("mesh.")]
    if not axes or any(name.startswith("geometry[") for name in inputs):
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
        mode = (getattr(conductors.grid, "realization_quantities", None) or {}).get("smoothing", 0)
        if mode:
            from rfx.geometry.smoothing import smoothed_shape_pairs, compute_smoothed_eps_nonuniform
            from rfx.model.realization_integrals import begin
            pairs, _ = smoothed_shape_pairs(reference, grid)
            begin(reference, grid, shapes=pairs)
            if mode == 2:
                from rfx.model.tensor_operands import full_tensor_eps
                full_tensor_eps(reference, grid, root.materials, pairs)
            else:
                compute_smoothed_eps_nonuniform(grid, pairs)
        record = record_from_conductors(reference, root, lane=lane, compact=compact)
        if record is not None:
            object.__setattr__(record, "_quantity_grid", grid)
        return record
