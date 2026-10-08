"""Whole-domain x arrays to owned slabs and neighbour ghosts.

This module owns placement, not field updates, pole packing, CPML state,
source ownership, or gathering. A mesh result merges rank and local x;
without a mesh the leading axes are (rank, local x), for legacy callers.
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.runners._rank import mesh_ranks, rank_shard_map


class Fill(NamedTuple):
    alignment: object
    physical: object


FILL = {
    'eps_r': Fill(1., 1.),  # Vacuum permittivity; avoid division by zero in Yee updates.
    'mu_r': Fill(1., 1.),  # Vacuum permeability; magnetic face means keep their own low-face view.
    'sigma': Fill(0., 0.),  # No conductive material outside the real cells.
    'lumped': Fill(0., 0.),  # No component-owned lumped stamp outside the model (#1236).
    'state': Fill(0., 0.),  # Initial field/state split has zero exterior fields.
    'pec_edge': Fill(True, False),  # Alignment is metal; True physical ghosts short x faces (#689/#931).
    'pec_cell': Fill(True, False),  # Legacy incident-cell PEC rule uses the same ghost-False convention.
    'pec_occupancy': Fill(0., 0.),  # No soft conductor outside the domain (#689/#931).
    'inv_dx': Fill(0., 1.),  # Inert alignment (#622); historical E inverse-spacing physical ghost is 1.
    'inv_dx_h': Fill(0., 0.),  # Inert alignment and H ghosts; retain the real high-boundary H zero (#622).
    'width': Fill('edge', 'edge'),  # Primal lengths replicate endpoints, retaining true seam neighbours.
}


@dataclass(frozen=True)
class Slab:
    """Static x ownership, constructed before tracing a time-step scan."""
    nx: int
    n_devices: int
    ghost_width: int = 1

    def __post_init__(self):
        if self.nx < 1 or self.n_devices < 1 or self.ghost_width < 0:
            raise ValueError('slab requires positive nx/device count and nonnegative ghosts')

    @property
    def nx_per_rank(self):
        return (self.nx + self.n_devices - 1) // self.n_devices

    @property
    def nx_padded(self):
        return self.n_devices * self.nx_per_rank

    @property
    def pad_x(self):
        return self.nx_padded - self.nx

    @property
    def nx_local(self):
        return self.nx_per_rank + 2 * self.ghost_width

    @property
    def owned(self):
        return tuple((min(r * self.nx_per_rank, self.nx),
                      min((r + 1) * self.nx_per_rank, self.nx))
                     for r in range(self.n_devices))

    @property
    def faces(self):
        return tuple((r == 0, r == self.n_devices - 1) for r in range(self.n_devices))

    @classmethod
    def from_grid(cls, grid):
        layout = getattr(grid, 'layout', None)
        return layout if layout is not None else cls(grid.nx, grid.n_devices, grid.ghost_width)


def forward_sharding(arr):
    """Read primal placement through AD/batching, including JAX 0.4 tracers."""
    if isinstance(arr, jax.core.Tracer):
        if hasattr(arr, 'primal'):
            return forward_sharding(arr.primal)
        if hasattr(arr, 'val') and hasattr(arr, 'batch_dim'):
            sharding = forward_sharding(arr.val)
            if sharding is not None and arr.batch_dim is not None:
                spec = list(sharding.spec) + [None] * (arr.val.ndim - len(sharding.spec))
                del spec[arr.batch_dim]
                return NamedSharding(sharding.mesh, P(*spec))
            return sharding
    sharding = getattr(arr, 'sharding', None)
    return sharding if isinstance(sharding, NamedSharding) else None


def is_sharded(arr, layout, mesh):
    """Validate x placement without fetching any design values."""
    sharding = forward_sharding(arr)
    if sharding is None:
        return False
    spec = tuple(sharding.spec) + (None,) * (arr.ndim - len(sharding.spec))
    if spec != ('x',) + (None,) * (arr.ndim - 1):
        return False
    expected = (layout.nx_padded, *arr.shape[1:])
    if (arr.shape != expected or sharding.mesh.devices.shape != mesh.devices.shape
            or not np.array_equal(sharding.mesh.devices, mesh.devices)):
        raise ValueError(f"sharded forward override must have shape {expected}, P('x'), "
                         'and the same ordered devices as forward(devices=...).')
    return True


def _halo(arr, layout, mesh, fill):
    """Keep the existing ppermute transpose: ghosts return cotangents to owners."""
    if layout.ghost_width != 1:
        raise NotImplementedError('sharded forward overrides require exchange_interval=1')
    right = [(i, i + 1) for i in range(layout.n_devices - 1)]
    left = [(i + 1, i) for i in range(layout.n_devices - 1)]

    def halo(local, *, rank):
        lo = lax.ppermute(local[-1:], 'x', right)
        hi = lax.ppermute(local[:1], 'x', left)
        slab = jnp.concatenate((lo, local, hi), axis=0)
        indices = rank * layout.nx_per_rank + jnp.arange(layout.nx_local) - 1
        real = (indices >= 0) & (indices < layout.nx)
        shape = (-1,) + (1,) * (arr.ndim - 1)
        if fill.alignment == fill.physical and fill.physical != 'edge':
            return jnp.where(real.reshape(shape), slab, jnp.asarray(fill.physical, arr.dtype))
        padded = (indices >= layout.nx) & (indices < layout.nx_padded)
        if fill.physical == 'edge':
            # Widths are geometry, but retain a differentiable endpoint policy.
            low = lax.psum(jnp.where(rank == 0, local[0], 0), 'x')
            owner = (layout.nx - 1) // layout.nx_per_rank
            high = lax.psum(jnp.where(rank == owner, local[(layout.nx - 1) % layout.nx_per_rank], 0), 'x')
            outside = jnp.where((indices < 0).reshape(shape), low, high)
        else:
            outside = jnp.where(padded.reshape(shape),
                                jnp.asarray(fill.alignment, arr.dtype),
                                jnp.asarray(fill.physical, arr.dtype))
        return jnp.where(real.reshape(shape), slab, outside)

    return jax.jit(rank_shard_map(halo, mesh=mesh, in_specs=P('x'),
        out_specs=P('x'), check_rep=False))(mesh_ranks(mesh), arr)


def _pad(data, lo, hi, value, xp):
    if not lo and not hi:
        return data
    widths = [(lo, hi)] + [(0, 0)] * (data.ndim - 1)
    return xp.pad(data, widths, mode='edge') if value == 'edge' else xp.pad(
        data, widths, mode='constant', constant_values=value)


def _slice(arr, layout, rank, fill, xp):
    """One global-index convention for host, device and traced local arrays."""
    want_lo = rank * layout.nx_per_rank - layout.ghost_width
    want_hi = (rank + 1) * layout.nx_per_rank + layout.ghost_width
    lo, hi = max(0, min(layout.nx, want_lo)), min(layout.nx, want_hi)
    data = arr[lo:hi]
    alignment = max(0, min(want_hi, layout.nx_padded) - max(want_lo, layout.nx))
    if alignment and fill.alignment == 'edge' and not data.shape[0]:
        data = xp.repeat(arr[layout.nx - 1:layout.nx], alignment, axis=0)
    else:
        data = _pad(data, 0, alignment, fill.alignment, xp)
    return _pad(data, max(0, -want_lo), max(0, want_hi - layout.nx_padded), fill.physical, xp)


def cut(array, layout, kind, *, mesh=None):
    """Place array leaves with the declared alignment and ghost policy.

    ``kind`` is a FILL key. ``Fill`` is also accepted for legacy coefficient
    adapters whose dt-dependent values are outside the spatial-array table.
    Tuples (including optional per-component records) map over their leaves.
    A field-to-kind mapping cuts a named record and leaves omitted fields at
    their record defaults (for legacy material bundles).
    Concrete placement constructs only addressable shards. Local tracers keep
    slice/pad/stack on the tape; x-sharded designs exchange their owned halos.
    """
    fill = FILL[kind] if isinstance(kind, str) else kind
    if array is None:
        return None
    if isinstance(kind, dict):
        return type(array)(**{name: cut(getattr(array, name), layout, tag, mesh=mesh)
                              for name, tag in kind.items()})
    if isinstance(array, (tuple, list, dict)):
        return jax.tree.map(lambda a: cut(a, layout, kind, mesh=mesh), array)
    if array.shape[0] not in (layout.nx, layout.nx_padded):
        raise ValueError(f'cut needs {layout.nx} real or {layout.nx_padded} padded x rows; got {array.shape}')
    sharding = mesh if isinstance(mesh, NamedSharding) else None
    mesh = sharding.mesh if sharding is not None else mesh
    if mesh is not None and layout.ghost_width and is_sharded(array, layout, mesh):
        return _halo(array, layout, mesh, fill)
    if mesh is not None and not isinstance(array, jax.core.Tracer):
        xp = np if isinstance(array, np.ndarray) else jnp
        shape = (layout.n_devices * layout.nx_local, *array.shape[1:])
        def slab(index):
            # Closed-over concrete arrays may be staged while a caller traces.
            with jax.ensure_compile_time_eval():
                return _slice(array, layout, (index[0].start or 0) // layout.nx_local, fill, xp)
        return jax.make_array_from_callback(shape, sharding or NamedSharding(mesh, P('x')),
                                           slab)
    # Match the historical local AD program: align first, then slice and stack.
    arr = array[:layout.nx] if array.shape[0] != layout.nx else array
    arr = _pad(arr, 0, layout.pad_x, fill.alignment, jnp)
    aligned = Slab(layout.nx_padded, layout.n_devices, layout.ghost_width)
    slabs = jnp.stack([_slice(arr, aligned, r, fill, jnp) for r in range(layout.n_devices)])
    if mesh is None:
        return slabs
    return jax.device_put(slabs.reshape((layout.n_devices * layout.nx_local, *array.shape[1:])),
                          NamedSharding(mesh, P('x')))
