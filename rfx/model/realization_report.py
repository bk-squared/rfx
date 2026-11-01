"""Build requested diagnostic arrays from current declarations, never a snapshot."""
from dataclasses import replace


def rebuild_view(sim, solved, smoothing):
    import copy
    from rfx.model.conductors import PreparedSolve, preview_port_stages
    from rfx.model.realization_table import finish_table
    from rfx.realized_geometry import record_from_conductors

    # A fresh preparation uses the solve assembler without replacing its store.
    reader = copy.copy(sim)
    reader._pf_campaign_ctx = None
    reader._realized_geometry_record = None
    reader._coax_geometry = None
    root = PreparedSolve(reader).take()
    if solved.lane == "run_subgridded":
        from rfx.runners.subgridded import _run_subgridded_once
        root.grid._prepared_conductors = root
        root = _run_subgridded_once(reader, root.grid, root.materials,
            root.pec_cells, 0, _warn_experimental=False, _prepare_only=True)
    else:
        root = preview_port_stages(reader, root)
    _integrals(reader, root, solved.realization_table, smoothing)
    rebuilt = finish_table(reader, root, lane=solved.lane)
    table = solved.realization_table
    if rebuilt.realization_table.rows != table.rows:
        raise ValueError(
            "Realization rows differ from the solved table; current declarations "
            "or reporter preparation cannot reproduce this solve. Run again "
            "with the current declarations before requesting diagnostic arrays.")
    if root.kernel_context is not None:
        reader._geometry, reader._thin_conductors = [], []
    record = record_from_conductors(reader, root, lane=solved.lane, compact=False)
    record = replace(record, realization_table=table, dt=solved.dt)
    object.__setattr__(root, "realization_table", table)
    object.__setattr__(record, "_conductors", root)
    object.__setattr__(record, "_report_grid", root.grid)
    return record


def _integrals(sim, root, table, smoothing):
    """Use the solver's producers for optional fill and open-area quantities."""
    conformal = any(row.open_area_m2 is not None for row in table.rows)
    if not smoothing and not conformal:
        return
    mode = {0: False, 1: True, 2: "full_tensor", 3: "kottke_pec"}[smoothing]
    if root.lane == "nonuniform":
        from rfx.geometry.smoothing import smoothed_shape_pairs, compute_smoothed_eps_nonuniform
        from rfx.model.realization_integrals import begin
        from rfx.model.tensor_operands import full_tensor_eps
        pairs, _ = smoothed_shape_pairs(sim, root.grid)
        begin(sim, root.grid, shapes=pairs, smoothing=mode)
        if mode == "full_tensor":
            full_tensor_eps(sim, root.grid, root.materials, pairs)
        else:
            compute_smoothed_eps_nonuniform(root.grid, pairs)
    else:
        from rfx.model.tensor_operands import form_tensor_operands
        shapes = root.assembly[4]
        form_tensor_operands(sim, root.grid, root.materials,
            subpixel_smoothing=mode, conformal=conformal,
            conformal_min_weight=.1, pec_shapes=shapes)
