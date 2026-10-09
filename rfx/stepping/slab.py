"""Whole-domain x arrays to owned slabs and neighbour ghosts.

This module owns placement, not field updates, CPML state,
source ownership, or gathering. A mesh result merges rank and local x;
without a mesh the leading axes are (rank, local x), for legacy callers.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.stepping.rank import mesh_ranks, rank_shard_map


class Fill(NamedTuple):
    alignment: object
    physical: object


FILL = {
    'debye_ca': Fill(1., 1.),  # Vacuum E decay, from init_debye (#1302).
    'debye_cb': Fill('dt/eps0', 'dt/eps0'),  # Vacuum curl coupling at the caller's dt (#1302).
    'debye_cc': Fill('1/eps0', '1/eps0'),  # Vacuum polarization coupling, with initializer rounding.
    'debye_alpha': Fill(0., 0.),  # No pole reaches an exterior cell (#1302).
    'debye_beta': Fill(0., 0.),  # No exterior E-to-polarization coupling (#1302).
    'debye_state': Fill(0., 0.),  # Initial/exterior Debye polarization is zero.
    'lorentz_ca': Fill(1., 1.),  # Vacuum E decay, from init_lorentz (#1302).
    'lorentz_cb': Fill('dt/eps0', 'dt/eps0'),  # Vacuum curl coupling at the caller's dt (#1302).
    'lorentz_cc': Fill('1/eps0', '1/eps0'),  # 1/cc must stay finite in mixed ADE backward; zero caused NaN eps gradients (#1227).
    'lorentz_a': Fill(0., 0.),  # No exterior pole recurrence (#1302).
    'lorentz_b': Fill(0., 0.),  # No exterior previous-polarization recurrence (#1302).
    'lorentz_c': Fill(0., 0.),  # No exterior E-to-polarization coupling (#1302).
    'lorentz_state': Fill(0., 0.),  # Initial/exterior Lorentz polarization and history are zero.
    'pole_mask': Fill(False, False),  # No Debye/Lorentz pole outside the real cells (#1302).
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


def cut(array, layout, kind, *, mesh=None, dt=None, host=False):
    """Place array leaves with the declared alignment and ghost policy.

    ``kind`` is a FILL key; pole curl coupling also requires ``dt``.
    ``Fill`` remains accepted by compatibility adapters. ``host=True`` keeps
    concrete no-mesh output in NumPy, preserving the input dtype.
    Tuples (including optional per-component records) map over their leaves.
    A field-to-kind mapping cuts a named record and leaves omitted fields at
    their record defaults (for legacy material bundles).
    Concrete placement constructs only addressable shards. Local tracers keep
    slice/pad/stack on the tape; x-sharded designs exchange their owned halos.
    """
    fill = pole_fill(kind, dt) if isinstance(kind, str) else kind
    if array is None:
        return None
    if isinstance(kind, dict):
        return type(array)(**{name: cut(getattr(array, name), layout, tag, mesh=mesh, dt=dt, host=host)
                              for name, tag in kind.items()})
    if isinstance(array, (tuple, list, dict)):
        return jax.tree.map(lambda a: cut(a, layout, kind, mesh=mesh, dt=dt, host=host), array)
    if array.shape[0] not in (layout.nx, layout.nx_padded):
        raise ValueError(f'cut needs {layout.nx} real or {layout.nx_padded} padded x rows; got {array.shape}')
    if host:
        if mesh is not None or isinstance(array, jax.core.Tracer):
            raise ValueError('host slab output requires a concrete array without a mesh')
        return np.stack([_slice(np.asarray(array), layout, r, fill, np)
                         for r in range(layout.n_devices)])
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
    merged = slabs.reshape((layout.n_devices * layout.nx_local, *array.shape[1:]))
    tracer = merged
    while isinstance(tracer, jax.core.Tracer):
        if getattr(tracer, 'batch_dim', None) is not None:
            # JIT's batching rule inserts an unpartitioned dimension at each
            # batch axis. device_put's batching rule keeps P('x') on axis 0.
            # The logical x axis therefore stays partitioned under vmap;
            # unbatched calls retain their original device_put program.
            return jax.jit(lambda a: a, out_shardings=NamedSharding(mesh, P('x')))(merged)
        tracer = getattr(tracer, 'primal', None)
    return jax.device_put(merged, NamedSharding(mesh, P('x')))


@lru_cache(maxsize=None)
def _vacuum_dispersion_values(kind, n_poles, dt):
    """Per field, the coefficient ``init_debye`` / ``init_lorentz`` give ONE
    vacuum cell (eps_r 1, sigma 0, no pole reaching it) at this ``dt``
    (#1302): Debye ``ca = 1, cb = dt/eps_0, cc = 1/eps_0, alpha = beta = 0``;
    Lorentz ``ca = 1, cb = dt/eps_0, cc = 1/eps_0, a = b = c = 0``.

    No pole reaches the cell (all-False masks), so the pole parameters do not
    enter; placeholder poles only fix the pole count. Returned as Python
    floats (one per field, equal across components and poles, checked).
    """
    from rfx.core.yee import MaterialArrays
    from rfx.materials.debye import DebyePole, init_debye
    from rfx.materials.lorentz import LorentzPole, init_lorentz
    with jax.ensure_compile_time_eval():
        one = jnp.ones((1, 1, 1))
        vacuum = MaterialArrays(eps_r=one, sigma=jnp.zeros((1, 1, 1)), mu_r=one)
        masks = [jnp.zeros((1, 1, 1), dtype=bool)] * n_poles
        if kind == "debye":
            coeffs, _ = init_debye([DebyePole(1.0, 1e-11)] * n_poles, vacuum, dt,
                                   mask=masks)
        else:
            coeffs, _ = init_lorentz([LorentzPole(1.0, 1.0, 1.0)] * n_poles, vacuum,
                                     dt, mask=masks)
    values = {}
    for name, field in zip(coeffs._fields, coeffs):
        leaves = [float(v) for leaf in jax.tree.leaves(field)
                  for v in np.asarray(leaf).ravel()]
        if len(set(leaves)) != 1:
            raise AssertionError(f"vacuum {kind} {name} is not one value: {leaves}")
        values[name] = leaves[0]
    return values


def vacuum_dispersion_values(coeffs, dt):
    """:func:`_vacuum_dispersion_values` for a coefficient bundle's type and
    pole count (a per-pole field's leading axis)."""
    if hasattr(coeffs, "alpha"):
        kind, n_poles = "debye", coeffs.alpha.shape[0]
    else:
        kind, n_poles = "lorentz", coeffs.a.shape[0]
    return _vacuum_dispersion_values(kind, int(n_poles), float(dt))


def shard_stacked(arr, shd):
    """Merge the device axis into x, then shard.

    ``(n_devices, nx_local, ny, nz) -> (n_devices*nx_local, ny, nz)``.

    """
    n_dev = arr.shape[0]
    rest = arr.shape[1:]
    return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)


def shard_stacked_poles(arr, shd):
    """Merge the device axis into the pole axis, then shard.

    ``(n_devices, n_poles, nx_local, ny, nz) ->``
    ``(n_devices*n_poles, nx_local, ny, nz)``, so ``P("x")`` hands each
    device ``(n_poles, nx_local, ny, nz)``.

    """
    n_dev, n_poles, nx_loc, ny_a, nz_a = arr.shape
    return jax.device_put(
        arr.reshape(n_dev * n_poles, nx_loc, ny_a, nz_a), shd,
    )


def pole_fill(kind, dt=None, n_poles=1):
    """Resolve time-dependent vacuum values with the existing initializer order."""
    fill = FILL[kind]
    if fill.physical in ('dt/eps0', '1/eps0'):
        family, field = kind.split('_', 1)
        if dt is None and field == 'cb':
            raise ValueError(f'{kind} requires dt for its vacuum fill')
        # Keep the initializer's rounding (including its configured precision).
        value = _vacuum_dispersion_values(family, n_poles, (0. if dt is None else float(dt)))[field]
        return Fill(value, value)
    return fill


def cut_poles(array, layout, kind, *, mesh=None, dt=None):
    """Pack (pole, x, ...) as (rank * pole, local x, ...) on a mesh.

    Without a mesh retain the explicit (rank, pole, local x, ...) axes.
    Component tuples and state records map over their leaves.
    """
    if array is None:
        return None
    if isinstance(array, (tuple, list, dict)):
        return jax.tree.map(lambda a: cut_poles(a, layout, kind, mesh=mesh, dt=dt), array)
    fill = pole_fill(kind, dt, array.shape[0])
    slabs = jnp.stack([cut(a, layout, fill) for a in array], axis=1)
    if mesh is None:
        return slabs
    sharding = mesh if isinstance(mesh, NamedSharding) else NamedSharding(mesh, P('x'))
    return shard_stacked_poles(slabs, sharding)


def cut_pole_coeffs(coeffs, layout, family, *, dt, mesh=None):
    """Apply the field's named policy and its spatial or pole-axis layout."""
    if coeffs is None:
        return None
    spatial = ('ca', 'cb') if family == 'debye' else ('ca', 'cb', 'cc')
    n_poles = coeffs.alpha.shape[0] if family == 'debye' else coeffs.a.shape[0]
    def field(name, value):
        kind = family + '_' + name
        if name in spatial:
            return cut(value, layout, pole_fill(kind, dt, n_poles), mesh=mesh)
        return cut_poles(value, layout, kind, dt=dt, mesh=mesh)
    return type(coeffs)(*(field(name, value) for name, value in zip(coeffs._fields, coeffs)))


def fill_pole_rows(coeffs, dt, nx_per, nx, *, rank):
    """Reset physical and alignment rows of coefficients formed on a slab."""
    if rank is None:
        raise ValueError('slab rank must be supplied as data')
    family = 'debye' if hasattr(coeffs, 'alpha') else 'lorentz'
    n_poles = coeffs.alpha.shape[0] if family == 'debye' else coeffs.a.shape[0]
    rows = rank * nx_per + jnp.arange(nx_per + 2) - 1
    real = ((rows >= 0) & (rows < nx))[:, None, None]
    return type(coeffs)(*(
        jax.tree.map(lambda a: jnp.where(real, a, pole_fill(family + '_' + name, dt, n_poles).physical), field)
        for name, field in zip(coeffs._fields, coeffs)))


def zero_pole_state(state_type, n_poles, layout, shape, *, mesh, dtype, kind):
    """Allocate the packed polarization carry directly on its owning devices."""
    fill = FILL[kind]
    assert fill == Fill(0., 0.)
    zeros = jnp.zeros((layout.n_devices * n_poles, layout.nx_local) + tuple(shape),
                      dtype=dtype, device=NamedSharding(mesh, P('x')))
    return state_type(*(zeros for _ in state_type._fields))


def stage_poles(build_coeffs, masks, layout, shape, *, mesh, n_poles, state_type, dtype, kind):
    """Place locally built pole factors and their zero state in rank/pole order.

    The callback owns pole arithmetic on local masks; this module owns its
    partition spec, rank data and the matching packed state layout.
    """
    local = rank_shard_map(build_coeffs, mesh=mesh, in_specs=P('x'),
                           out_specs=P('x'), check_rep=False)
    coeffs = jax.jit(local)(mesh_ranks(mesh), masks)
    state = zero_pole_state(state_type, n_poles, layout, shape, mesh=mesh,
                            dtype=dtype, kind=kind + '_state')
    return coeffs, state
