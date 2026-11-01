"""Host-only ownership metadata following the absorber's material selection."""
import numpy as np


def retain_continuation(sim, grid, geometry_masks, eps_r, sigma, mu_r, pole_mask):
    from rfx.core.jax_utils import is_tracer
    pads = [(getattr(grid, f"pad_{a}_lo"), getattr(grid, f"pad_{a}_hi")) for a in "xyz"]
    masks = dict(geometry_masks or ())
    if not masks or not any(lo or hi for lo, hi in pads):
        return
    if any(is_tracer(a) for a in (eps_r, sigma, mu_r, pole_mask, *masks.values())):
        return
    owner = np.full(grid.shape, -1, dtype=np.int32)
    for index, entry in enumerate(sim._geometry):
        if id(entry) in masks:
            np.copyto(owner, index, where=np.asarray(masks[id(entry)]))
    # Only vacuum/pole predicates affect source selection. Carry those compact
    # predicates through x, y, z in the same order as the material pad copies.
    vacuum = (np.asarray(eps_r) == 1) & (np.asarray(sigma) == 0) & (np.asarray(mu_r) == 1)
    pole = None if pole_mask is None else np.array(pole_mask, dtype=bool, copy=True)
    arrays = [owner, vacuum] + ([] if pole is None else [pole])
    for axis, (lo, hi) in enumerate(pads):
        def slab(start, end):
            region = [slice(None)] * 3
            region[axis] = slice(start, end)
            return tuple(region)

        if lo:
            for array in arrays:
                array[slab(0, lo)] = array[slab(lo, lo + 1)]
        if hi:
            n = grid.shape[axis]
            outer, inner = slab(n-hi-1, n-hi), slab(n-hi-2, n-hi-1)
            use_inner = None
            if n-lo-hi >= 2:
                use_inner = vacuum[outer] & ~vacuum[inner]
                if pole is not None:
                    use_inner &= ~pole[inner]
            for array in arrays:
                source = array[outer]
                if use_inner is not None:
                    source = np.where(use_inner, array[inner], source)
                    array[outer] = source
                array[slab(n-hi, n)] = source
    continued = {}
    for axis, (lo, hi) in enumerate(pads):
        for side, depth in enumerate((lo, hi)):
            if depth:
                region = [slice(None)] * 3
                region[axis] = slice(-depth, None) if side else slice(0, depth)
                for index in np.unique(owner[tuple(region)]):
                    if index >= 0:
                        continued.setdefault(int(index), []).append(2*axis+side)
    from rfx.model.realization_integrals import quantities
    quantities(grid, create=True)["continued"] = continued
