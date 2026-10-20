"""Static polarization support; coefficient arrays retain their native layout."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PoleBox:
    """Half-open Python bounds, carried as pytree metadata, never array data."""

    bounds: tuple
    domain_shape: tuple

    def tree_flatten(self):
        return (), (self.bounds, self.domain_shape)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*metadata)

    @property
    def slices(self):
        return tuple(slice(lo, hi) for lo, hi in self.bounds)

    @property
    def shape(self):
        return tuple(hi - lo for lo, hi in self.bounds)

    def full_x(self, nx):
        return PoleBox(((0, int(nx)),) + self.bounds[1:], (int(nx),) + self.domain_shape[1:])


def full_box(shape):
    return PoleBox(tuple((0, int(n)) for n in shape), tuple(shape))


def support_box(shape, fractions, *, full_x=False):
    """Bounding box of the realized pole reach; tracers/everywhere stay full."""
    if fractions is None or any(f is None for f in fractions):
        return full_box(shape)
    if any(isinstance(a, jax.core.Tracer) for f in fractions for a in f):
        return full_box(shape)
    reach = np.zeros(shape, dtype=bool)
    for components in fractions:
        for a in components:
            reach |= np.asarray(a) > 0
    indices = np.nonzero(reach)
    bounds = tuple((int(i.min()), int(i.max()) + 1) if i.size else (0, 0)
                   for i in indices)
    box = PoleBox(bounds, tuple(shape))
    return box.full_x(shape[0]) if full_x else box


def mask_support(shape, masks, n_poles, periodic=(False,) * 3, *, full_x=False):
    """Realize concrete mask fractions once, before a mapped slab builder."""
    from rfx.materials.debye import pole_edge_fractions
    masks = masks if isinstance(masks, (tuple, list)) else [masks] * n_poles
    if any(m is None or isinstance(m, jax.core.Tracer) for m in masks):
        return full_box(shape)
    with jax.ensure_compile_time_eval():
        fractions = tuple(pole_edge_fractions(m, periodic) for m in masks)
    return support_box(shape, fractions, full_x=full_x)


def state_box(state, shape):
    return state.box if state.box is not None else full_box(shape)


def rebox_state(state, box, shape):
    """Rebase a polarization record, zero extending only newly covered cells."""
    old = state_box(state, shape)
    if old == box:
        return state._replace(box=box)
    overlap = tuple((max(a, c), min(b, d))
                    for (a, b), (c, d) in zip(old.bounds, box.bounds))
    def move(a):
        out = jnp.zeros(a.shape[:-3] + box.shape, dtype=a.dtype)
        if any(hi <= lo for lo, hi in overlap):
            return out
        source = tuple(slice(lo - start, hi - start)
                       for (lo, hi), (start, _) in zip(overlap, old.bounds))
        target = tuple(slice(lo - start, hi - start)
                       for (lo, hi), (start, _) in zip(overlap, box.bounds))
        return out.at[(Ellipsis,) + target].set(a[(Ellipsis,) + source])
    values = {name: move(getattr(state, name)) for name in state._fields if name != 'box'}
    return state._replace(**values, box=box)


def align_pole_states(debye, lorentz, shape):
    """Mixed media share one union support before their carries enter a scan."""
    if debye is None or lorentz is None:
        return debye, lorentz
    d, l = state_box(debye[1], shape), state_box(lorentz[1], shape)
    box = (l if 0 in d.shape else d if 0 in l.shape else
           PoleBox(tuple((min(a, c), max(b, d))
                         for (a, b), (c, d) in zip(d.bounds, l.bounds)), d.domain_shape))
    return ((debye[0], rebox_state(debye[1], box, shape)),
            (lorentz[0], rebox_state(lorentz[1], box, shape)))


def slices(box):
    return (slice(None),) * 3 if box is None else box.slices


def pole_slice(a, box):
    return a[(slice(None),) + slices(box)]


def write_box(base, value, box):
    if box is None:
        return value
    return jax.lax.dynamic_update_slice(base, value, tuple(lo for lo, _ in box.bounds))
