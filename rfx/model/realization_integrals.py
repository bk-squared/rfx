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


def begin(sim, grid, *, shapes=(), conformal_shapes=(), smoothing=None):
    store = quantities(grid, create=True)
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
            from rfx._grid_metric import dual_spacings_from_cells
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
    weights = _weights(grid, component, face=face)
    values = np.asarray(fraction) * weights
    for index, shape in enumerate(shapes):
        if id(shape) not in ids:
            continue
        selected = np.asarray(owner) == index
        total = float(np.sum(np.where(selected, values, 0.)))
        key = ids[id(shape)]
        store.setdefault("open" if face else "filled", {}).setdefault(key, {})[component] = total
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
            if volume:
                filled = volume.get(3, float(np.mean(list(volume.values()))))
                drawn = _drawn_volume(sim._geometry[index].shape)
                row = replace(row, filled_volume_m3=filled, declared_volume_m3=drawn,
                              volume_equivalent_length_m=float(np.cbrt(filled)),
                              volume_miss_percent=None if not drawn else 100*(float(np.cbrt(filled/drawn))-1),
                              model="smoothed fill; union owner partition")
                row = replace(row, continued_faces=tuple(
                    f"{'xyz'[face//2]}-{'hi' if face%2 else 'lo'}" for face in
                    sorted(store.get("smoothed_continued", {}).get(index, ()))))
            if area:
                row = replace(row, open_area_m2=tuple(area.get(a, 0.) for a in range(3)),
                              model="conformal open faces; union owner partition")
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
