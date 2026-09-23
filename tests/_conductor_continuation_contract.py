"""Read the arrays assembled for a declared model; never invoke continuation."""
import numpy as np


def assembled_arrays(sim, *, include_smoothed=True):
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.geometry.rasterize_grid import (
        coords_from_uniform_grid, coords_from_nonuniform_grid)
    from rfx.geometry.smoothing import (
        compute_inv_eps_tensor_diag, smoothed_shape_pairs)
    nu = any(getattr(sim, name, None) is not None
             for name in ("_dx_profile", "_dy_profile", "_dz_profile"))
    grid = sim._build_nonuniform_grid() if nu else sim._build_grid()
    sheets, wires, impedances = [], [], []
    if nu:
        from rfx.runners.nonuniform import assemble_materials_nu
        mats, debye, lorentz, cells = assemble_materials_nu(
            sim, grid, pec_sheets=sheets, pec_wires=wires, sheet_specs=impedances)
        pec_shapes = []
    else:
        mats, debye, lorentz, cells, pec_shapes, _, _ = sim._assemble_materials(
            grid, pec_sheets=sheets, pec_wires=wires, sheet_specs=impedances)
    arrays = {name: np.asarray(getattr(mats, name))
              for name in ("eps_r", "sigma", "mu_r")}
    arrays["pec_mask"] = (np.zeros(grid.shape, bool) if cells is None
                          else np.asarray(cells))
    edges = realized_pec_edge_masks(arrays["pec_mask"], sheets=sheets, wires=wires,
                                   periodic=sim._periodic_flags() if not nu else (False,)*3)
    for axis, arr in zip("xyz", edges):
        arrays["edge_"+axis] = np.asarray(arr)
    for i, sheet in enumerate(sheets):
        arrays[f"sheet_{i}"] = np.asarray(sheet.footprint)
    for i, sheet in enumerate(impedances):
        arrays[f"impedance_{i}"] = np.asarray(sheet.mask)
    # Build the same tensor as the uniform kottke_pec runner, from the
    # assembler's shapes and production smoothed dielectric pairs.
    if not nu and include_smoothed:
        pairs, _ = smoothed_shape_pairs(sim, grid)
        tensor = compute_inv_eps_tensor_diag(grid, dielectric_shapes=pairs,
                                             pec_shapes=pec_shapes, background_eps=1.)
        for axis, arr in zip("xyz", tensor):
            arrays["tensor_"+axis] = np.asarray(arr)
        from rfx.boundaries.pec import kottke_fenced_edge_masks
        fenced = kottke_fenced_edge_masks(edges, tensor, sheets=sheets, wires=wires,
                                          periodic=sim._periodic_flags())
        for axis, arr in zip("xyz", fenced):
            arrays["fenced_"+axis] = np.asarray(arr)
    if nu and include_smoothed and debye is None and lorentz is None:
        from rfx.geometry.smoothing import compute_smoothed_eps_nonuniform
        pairs, _ = smoothed_shape_pairs(sim, grid)
        if pairs:
            tensor = compute_smoothed_eps_nonuniform(grid, pairs, background_eps=1.)
            for axis, arr in zip("xyz", tensor):
                arrays["smoothed_"+axis] = np.asarray(arr)
    poles = []
    for spec in (debye, lorentz):
        if spec is not None:
            poles.extend(np.asarray(mask) for mask in spec[1])
    coords = coords_from_nonuniform_grid(grid) if nu else coords_from_uniform_grid(grid)
    return grid, arrays, poles, (coords.x, coords.y, coords.z)


def violations(sim, grid, arrays, poles, nodes, *, held_faces=()):
    """Compare valid lattice layers, reporting every unexplained difference.

    Exceptions are constructed from declarations, independently of the
    geometry continuation code: pole columns, literal held-face rows, and
    unsupported conducting shapes' rasterized transverse support.
    """
    from rfx.geometry.csg import Box, Cylinder, declared_bounds
    findings, exceptions = [], []
    entries = [(i, e.shape, e.material_name) for i, e in enumerate(sim._geometry)
               if sim._resolve_material(e.material_name).sigma >= sim._PEC_SIGMA_THRESHOLD]
    entries += [(len(sim._geometry)+i, tc.shape, f"thin[{i}]") for i, tc in enumerate(sim._thin_conductors)]
    for axis in range(3):
        for side in (0, 1):
            pad = getattr(grid, "pad_"+"xyz"[axis]+("_hi" if side else "_lo"))
            if not pad:
                continue
            others = [a for a in range(3) if a != axis]
            shape2 = tuple(grid.shape[a]-1 for a in others)
            exempt = np.zeros(shape2, bool)
            for i, pole in enumerate(poles):
                columns = pole.any(axis=axis)[:-1, :-1]
                exempt |= columns
                exceptions.append((axis, side, f"pole[{i}]", int(columns.sum())))
            per_array = {name: exempt.copy() for name in arrays}
            for index, shape, name in entries:
                bounds = declared_bounds(shape)
                if bounds is None:
                    continue
                lo, hi = bounds
                face = 0. if side == 0 else sim._unresolved_domain[axis]
                reached = lo[axis] <= face if side == 0 else hi[axis] >= face
                reason = None
                if reached and not isinstance(shape, Box):
                    if not isinstance(shape, Cylinder) or "xyz".index(shape.axis) != axis:
                        reason = "unsupported:"+name
                if (index, "xyz"[axis]+("-hi" if side else "-lo")) in held_faces:
                    reason = "port-terminal:"+name
                if reason is None:
                    continue
                # The exception belongs to the ENTRY. Read its declared
                # cell/sheet/wire lattice directly, without continuation.
                # Sheets can snap off their declared z coordinate, and their
                # closed footprints need not share the Box node sampler.
                from rfx.geometry.rasterize_grid import (
                    GridCoords, classify_pec_entry, cell_centres_from_nodes,
                    cell_sizes_from_uniform_grid, cell_sizes_from_nonuniform_grid)
                from rfx.boundaries.pec import realized_pec_edge_masks
                coords = GridCoords(*nodes, grid.shape)
                sizes = (cell_sizes_from_nonuniform_grid(grid) if hasattr(grid, "dx_arr")
                         else cell_sizes_from_uniform_grid(grid))
                centres = cell_centres_from_nodes(coords, sizes)
                cells, sheet, wire = classify_pec_entry(shape, coords, centres, sizes)
                cell_array = np.zeros(grid.shape, bool) if cells is None else np.asarray(cells)
                edges = realized_pec_edge_masks(cell_array,
                    sheets=[sheet] if sheet is not None else [],
                    wires=[wire] if wire is not None else [])
                tensor = None
                if any(key.startswith("tensor_") for key in arrays):
                    from rfx.geometry.smoothing import compute_inv_eps_tensor_diag
                    tensor = compute_inv_eps_tensor_diag(grid, dielectric_shapes=[],
                                                        pec_shapes=[shape], background_eps=1.)
                smoothed = None
                if any(key.startswith("smoothed_") for key in arrays):
                    # The NU smoothed lane carries a conducting entry as an
                    # eps_r = 1 shape; its influence footprint is where that
                    # shape alone moves a background that is not 1.
                    from rfx.geometry.smoothing import compute_smoothed_eps_nonuniform
                    smoothed = compute_smoothed_eps_nonuniform(
                        grid, [(shape, 1.0)], background_eps=2.)
                counts = {}
                for key in arrays:
                    support = None
                    if key == "pec_mask":
                        support = cell_array
                    elif key.startswith(("edge_", "fenced_")):
                        support = np.asarray(edges["xyz".index(key[-1])])
                    elif key.startswith(("sheet_", "impedance_")) and sheet is not None:
                        support = np.asarray(sheet.footprint)
                    elif key.startswith("tensor_") and tensor is not None:
                        support = np.asarray(tensor["xyz".index(key[-1])]) != 1.
                    elif key.startswith("smoothed_") and smoothed is not None:
                        support = np.asarray(smoothed["xyz".index(key[-1])]) != 2.
                    elif name.startswith("thin[") and key in ("eps_r", "sigma"):
                        tc = sim._thin_conductors[int(name[5:-1])]
                        if not tc.is_pec:
                            support = np.asarray(shape.mask_on_coords(*nodes))
                    if support is not None:
                        window = support.any(axis=axis)[:-1, :-1]
                        per_array[key] |= window
                        counts[key] = int(window.sum())
                exceptions.append((axis, side, reason, counts))
            for name, raw in arrays.items():
                arr = raw[:-1, :-1, :-1]
                # PEC cells and the component-normal E edge are cell-indexed
                # on this axis. Other arrays sample nodes.
                cell_axis = name == "pec_mask" or name in (
                    "edge_"+"xyz"[axis], "fenced_"+"xyz"[axis],
                    "tensor_"+"xyz"[axis], "smoothed_"+"xyz"[axis])
                n = grid.shape[axis]
                face_index = pad if side == 0 else n-pad-1-int(cell_axis)
                layers = range(pad) if side == 0 else range(n-pad-1, n-1)
                reference = np.take(arr, face_index, axis=axis)
                count = 0
                for layer in layers:
                    diff = np.take(arr, layer, axis=axis) != reference
                    count += int(np.count_nonzero(diff & ~per_array[name]))
                if count:
                    findings.append(dict(array=name, axis=axis, side=side, differences=count))
    return findings, exceptions
