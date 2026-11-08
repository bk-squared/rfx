"""Declaration identities when a refused or traced model has no readable box."""
from dataclasses import replace


def identity_record(sim, ctx, reason):
    """Retain the table producer's labels without inventing geometric evidence."""
    from rfx.core.jax_utils import is_tracer
    from rfx.model.realization_table import PreparedRealization, realization_table
    from rfx.realized_geometry import RealizedGeometry, _node_arrays

    lane = ctx.lane or "traced"
    table = realization_table(PreparedRealization(sim, ctx, lane, None))
    inputs = tuple(f"mesh.{axis}" for axis in "xyz"
                   if is_tracer(getattr(sim, f"_d{axis}_profile")))
    if inputs:
        table = replace(table, rows=tuple(
            row if row.status == "not represented" else
            replace(row, status="not evaluated: traced", traced_inputs=inputs)
            for row in table.rows))
    sizes, nodes = ((), ()) if inputs or ctx.grid is None else _node_arrays(
        sim, ctx.grid, ctx.lane == "nonuniform")
    return RealizedGeometry((), (), (), tuple(nodes), tuple(sizes), (), None, (), (), None,
                            tuple(ctx.periodic), lane=lane,
                            limitations=(reason,), realization_table=table)
