"""Reduce the fill fractions and open areas already formed by the solver.

Equal-material unions are partitioned by their minimum signed-distance owner.
The reduction never evaluates a shape or introduces another sampling rule.
"""
import numpy as np


_stores = {}


def quantities(grid, *, create=False):
    import weakref
    owner = getattr(grid, "dx_arr", grid)
    key = id(owner)
    found = _stores.get(key)
    if found is not None and found[0]() is owner:
        return found[1]
    if not create:
        return {}
    store = {}
    _stores[key] = weakref.ref(owner, lambda ref: _stores.pop(key, None)), store
    return store


def begin(sim, grid, *, shapes=(), conformal_shapes=(), smoothing=None, conformal=None):
    store = quantities(grid, create=True)
    if conformal is not None:
        store["conformal"] = conformal
    if smoothing is not None:
        store["smoothing"] = (2 if smoothing == "full_tensor" else 3 if smoothing == "kottke_pec"
                              else int(bool(smoothing)))
    owners = {id(entry.shape): i for i, entry in enumerate(sim._geometry)}
    store.setdefault("owners", {}).update({id(shape): owners[id(shape)]
        for shape in (*[s for s, _ in shapes], *conformal_shapes) if id(shape) in owners})


def _weights(grid, component, *, face=False):
    """Clip the primal/dual integration cells to the physical domain."""
    result = []
    for a, axis in enumerate("xyz"):
        d = np.asarray(grid.cells(a), dtype=float)
        lo, hi = getattr(grid, f"pad_{axis}_lo"), getattr(grid, f"pad_{axis}_hi")
        periodic = axis in getattr(grid, "periodic_axes", "")
        end = len(d) - hi - (not periodic)
        w = np.zeros_like(d)
        w[lo:end] = d[lo:end]
        dual = (a != component) if not face else a == component
        if dual:
            from rfx._internal._grid_metric import dual_spacings_from_cells
            w = dual_spacings_from_cells(np.r_[w[-1] if periodic else 0., w])[1:]
        if face and a == component:
            w = (w > 0).astype(float)
        result.append(w.reshape(tuple(len(w) if b == a else 1 for b in range(3))))
    return result[0] * result[1] * result[2]


def host_owner(sdf, index=0, owner=None, union=None):
    """Partition concrete samples without adding device operands or tracing."""
    from rfx.core.jax_utils import is_tracer
    if is_tracer(sdf) or is_tracer(union):
        return None
    if union is None:
        return np.zeros(np.shape(sdf), dtype=np.int32)
    if owner is not None:
        np.copyto(owner, index, where=np.asarray(sdf) < np.asarray(union))
    return owner


def capture(grid, shapes, fraction, owner, component, *, face=False):
    from rfx.core.jax_utils import is_tracer
    store = quantities(grid)
    if owner is None or not store or is_tracer(fraction) or any(is_tracer(grid.cells(a)) for a in range(3)):
        return
    ids = store.get("owners", {})
    if not any(id(s) in ids for s in shapes):
        return
    group = tuple(ids[id(s)] for s in shapes if id(s) in ids)
    for key in group:
        store.setdefault("groups", {})[key] = group
    weights = _weights(grid, component, face=face)
    values = np.asarray(fraction) * weights
    reference = None
    if face:
        from rfx.model.realization_reference_area import drawn_open_fractions
        reference = drawn_open_fractions(grid, shapes, component)
    for index, shape in enumerate(shapes):
        if id(shape) not in ids:
            continue
        selected = np.asarray(owner) == index
        total = float(np.sum(np.where(selected, values, 0.)))
        key = ids[id(shape)]
        store.setdefault("open" if face else "filled", {}).setdefault(key, {})[component] = total
        if face and reference is not None:
            store.setdefault("drawn_open", {}).setdefault(key, {})[component] = float(
                np.sum(np.where(selected, reference * weights, 0.)))
            deviation = float(np.max(np.where(selected & (weights > 0),
                np.abs(np.asarray(fraction)-reference), 0.)))
            store.setdefault("owner_deviation", {})[key] = max(
                store.get("owner_deviation", {}).get(key, 0.), deviation)
        if not face:
            occupied = selected & (np.asarray(fraction) > 0)
            faces = store.setdefault("smoothed_continued", {}).setdefault(key, {})
            for a, axis in enumerate("xyz"):
                for side, word in enumerate(("lo", "hi")):
                    depth = getattr(grid, f"pad_{axis}_{word}")
                    if depth:
                        region = [slice(None)] * 3
                        region[a] = slice(-depth, None) if side else slice(0, depth)
                        if occupied[tuple(region)].any():
                            faces[2*a+side] = True


def read_rows(rows, grid, sim):
    from dataclasses import replace
    store = quantities(grid) or {}
    result = []
    for row in rows:
        if row.label.startswith("geometry["):
            index = int(row.label[9:-1])
            if row.family == "material" and "continued" in store:
                row = replace(row, continued_faces=tuple(
                    f"{'xyz'[face//2]}-{'hi' if face%2 else 'lo'}"
                    for face in store["continued"].get(index, ())))
            volume = store.get("filled", {}).get(index)
            area = store.get("open", {}).get(index)
            group = store.get("groups", {}).get(index, (index,))
            if volume:
                filled = volume.get(3, float(np.mean(list(volume.values()))))
                drawn = _drawn_volume(sim._geometry[index].shape)
                row = replace(row, filled_volume_m3=filled, declared_volume_m3=drawn,
                              volume_equivalent_length_m=float(np.cbrt(filled)),
                              volume_miss_percent=None if not drawn else 100*(float(np.cbrt(filled/drawn))-1),
                              model="smoothed fill; union owner partition")
                reference = _drawn_union_volume([sim._geometry[k].shape for k in group])
                if reference is not None and reference > 0:
                    total = sum(values.get(3, float(np.mean(list(values.values()))))
                                for k in group if (values := store.get("filled", {}).get(k)))
                    row = replace(row, integral_comparison=("filled volume", reference, total, 3))
                else:
                    row = replace(row, status="not evaluated: analytic filled-volume reference unavailable")
                row = replace(row, continued_faces=tuple(
                    f"{'xyz'[face//2]}-{'hi' if face%2 else 'lo'}" for face in
                    sorted(store.get("smoothed_continued", {}).get(index, ()))))
            if area:
                row = replace(row, open_area_m2=tuple(area.get(a, 0.) for a in range(3)),
                              model="conformal open faces; union owner partition")
                drawn = store.get("drawn_open", {}).get(index)
                if drawn:
                    declared, realized = sum(drawn.values()), sum(area.values())
                    cell = float(np.mean([a.cell_size_m for a in row.axes]))
                    row = replace(row, declared_area_m2=declared, realized_area_m2=realized,
                        declared_area_equivalent_length_m=float(np.sqrt(declared)),
                        area_equivalent_length_m=float(np.sqrt(realized)),
                        area_miss_percent=100*(float(np.sqrt(realized/declared))-1),
                        area_miss_cells=(float(np.sqrt(realized))-float(np.sqrt(declared)))/cell,
                        max_owner_deviation=store["owner_deviation"][index])
                    group_declared = sum(sum(store["drawn_open"][k].values()) for k in group)
                    group_realized = sum(sum(store["open"][k].values()) for k in group)
                    row = replace(row, integral_comparison=(
                        "summed open area", group_declared, group_realized, 2))
                else:
                    row = replace(row, status="not evaluated: analytic open-area reference unavailable")
        result.append(row)
    return tuple(result)


def _drawn_volume(shape):
    from rfx.geometry.csg import Box, Sphere, Cylinder
    if isinstance(shape, Box):
        return float(np.prod(np.asarray(shape.corner_hi)-np.asarray(shape.corner_lo)))
    if isinstance(shape, Sphere):
        return float(4*np.pi*shape.radius**3/3)
    if isinstance(shape, Cylinder):
        return float(np.pi*shape.radius**2*shape.height)
    return None


def _drawn_union_volume(shapes):
    """Closed-form union of Boxes, or disjoint analytic primitive volumes."""
    from rfx.geometry.csg import Box
    if all(isinstance(s, Box) for s in shapes):
        nodes = [np.unique([edge[a] for s in shapes for edge in (s.corner_lo, s.corner_hi)])
                 for a in range(3)]
        centres = np.meshgrid(*[(n[:-1]+n[1:])/2 for n in nodes], indexing="ij", sparse=True)
        occupied = np.zeros(tuple(len(n)-1 for n in nodes), dtype=bool)
        for s in shapes:
            occupied |= ((centres[0] >= s.corner_lo[0]) & (centres[0] < s.corner_hi[0]) &
                         (centres[1] >= s.corner_lo[1]) & (centres[1] < s.corner_hi[1]) &
                         (centres[2] >= s.corner_lo[2]) & (centres[2] < s.corner_hi[2]))
        weights = np.diff(nodes[0])[:, None, None]*np.diff(nodes[1])[None, :, None]*np.diff(nodes[2])[None, None, :]
        return float(np.sum(weights*occupied))
    volumes = [_drawn_volume(s) for s in shapes]
    if any(v is None for v in volumes):
        return None
    for i, first in enumerate(shapes):
        lo, hi = first.bounding_box()
        for second in shapes[i+1:]:
            other_lo, other_hi = second.bounding_box()
            if all(min(hi[a], other_hi[a]) > max(lo[a], other_lo[a]) for a in range(3)):
                return None
    return sum(volumes)
