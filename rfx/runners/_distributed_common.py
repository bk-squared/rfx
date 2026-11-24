"""Shared scaffolding for the distributed FDTD runners.

This module holds shared helpers for ``distributed_v2.py`` and
``distributed_nu.py``. It also contains the uniform CPML helpers moved here
when the pmap runner in ``distributed.py`` was retired. The non-uniform
CPML kernel retains its per-axis, per-face spacing implementation.
"""

from __future__ import annotations

from rfx.stepping.slab import (
    Slab, Fill, cut, cut_poles, cut_pole_coeffs, fill_pole_rows, stage_poles,
    _vacuum_dispersion_values, vacuum_dispersion_values, shard_stacked, shard_stacked_poles,
)

from functools import partial

from rfx._internal import _realized

import jax
import numpy as np
import jax.numpy as jnp

from rfx.core.drives import drive_layout, inject_drives
from jax import lax
from rfx.stepping.rank import mesh_ranks, rank_pair_exchange, rank_shard_map
from jax.sharding import Mesh, PartitionSpec as P

from rfx.model.conductors import distributed_mask_edges, distributed_mask_spec
from rfx.core.yee import (
    EPS_0,
    MU_0,
    FDTDState,
    MaterialArrays,
    ade_state_dtype,
    cell_owned_component_materials,
    component_e_materials,
    component_h_materials,
    edge_mean_components,
    e_update_coeffs,
    map_lumped,
    _shift_fwd,
    _shift_bwd,
)
from rfx.materials.debye import (
    DebyeCoeffs, DebyePole, DebyeState, debye_e_component, debye_e_coeffs,
    debye_pole_coeffs, init_debye, per_component,
)
from rfx.materials.lorentz import (
    LorentzCoeffs, LorentzPole, LorentzState, init_lorentz, lorentz_e_coeffs,
    lorentz_pole_coeffs, lorentz_e_component,
    lorentz_p_component, mixed_e_component_coeffs,
)

__all__ = [
    "split_array_x",
    "gather_array_x",
    "split_poles_x",
    "unstack_and_gather",
    "exchange_component_shmap",
    "apply_pec_face_shmap",
    "apply_pec_mask_shmap",
    "shard_stacked",
    "shard_stacked_poles",
    "inject_sources_shmap",
    "sample_probes_shmap",
    "_update_h_local_nu",
    "_update_e_local_nu",
    "slab_e_component_materials",
    "slab_e_materials_shmap",
    "update_h_nu_shmap",
    "update_e_nu_shmap",
    "_split_state",
    "_split_materials",
    "_split_debye_coeffs",
    "_split_debye_state",
    "_split_lorentz_coeffs",
    "_split_lorentz_state",
    "_update_h_local",
    "_update_e_local",
    "_update_e_debye_local",
    "_update_e_lorentz_local",
    "_update_e_local_with_dispersion",
    "_distributed_boundary_layers",
]




def split_array_x(arr, n_devices, ghost=1, pad_value=0.0):
    if arr.shape[0] % n_devices:
        raise ValueError('split_array_x requires x length divisible by n_devices')
    return cut(arr, Slab(arr.shape[0], n_devices, ghost), Fill(pad_value, pad_value), mesh=None)


def shard_x_slabs(arr, n_devices, nx_per, ghost, pad_value, sharding):
    if arr.ndim != 3:
        raise ValueError('shard_x_slabs requires a 3-D array')
    return cut(arr, Slab(n_devices * nx_per, n_devices, ghost), Fill(pad_value, pad_value), mesh=sharding)


def stage_dispersion_slabs(materials, dt, debye_spec, lorentz_spec,
                           n_devices, nx_per, ghost, sharding, *, nx, high_wall):
    """Place pole masks slab by slab, then build only the fixed ADE terms.

    E coefficients are deferred to the loop, where they read the same slab
    component materials as the plain E update and CPML. No whole-domain
    coefficient array or device-axis stack is constructed.
    """
    out = []
    for kind, spec in zip(("debye", "lorentz"), (debye_spec, lorentz_spec)):
        if spec is None:
            out.append(None)
            continue
        poles, masks = spec
        masks = jax.tree.map(
            lambda a: cut(a, Slab(nx, n_devices, ghost), "pole_mask", mesh=sharding),
            masks)
        out.append(stage_slab_pole_coeffs(
            poles, masks, dt, kind, sharding.mesh, nx_per, nx,
            (n_devices * (nx_per + 2 * ghost),) + materials.eps_r.shape[1:], high_wall=high_wall))
    return tuple(out)


from rfx.model.electric_metrics import slab_cell_sizes, slab_metric_kwargs


def stage_slab_pole_coeffs(poles, masks, dt, kind, mesh, nx_per, nx, shape, *, high_wall, grid=None):
    """Fixed ADE terms and zero carry; E coefficient slots stay ``None``.

    The compiled setup contains only pole-mask means and their fixed ADE
    factors, never differentiable epsilon/sigma coefficients. In particular,
    forward does not execute a large eager shard_map on every call.
    """
    def local(local_masks, *, rank):
        shape_local = (nx_per + 2,) + shape[1:]
        fractions = [None] * len(poles) if local_masks is None else [
            slab_pole_fractions(m, nx_per, nx, rank=rank, high_wall=high_wall, cell_sizes=slab_cell_sizes(grid, rank)) for m in
            (local_masks if isinstance(local_masks, (tuple, list)) else [local_masks] * len(poles))]
        if len(fractions) != len(poles):
            raise ValueError(f"Expected {len(poles)} {kind} masks, got {len(fractions)}")
        if kind == "debye":
            alpha, beta = debye_pole_coeffs(poles, dt, shape_local, fractions)
            coeffs = DebyeCoeffs(None, None, None, alpha, beta)
        else:
            damping, restoring, c = lorentz_pole_coeffs(poles, dt, shape_local, fractions)
            coeffs = LorentzCoeffs(ca=None, cb=None, cc=None, damping=damping, restoring=restoring, c=c)
        return _pad_slab_dispersion_coeffs(coeffs, dt, nx_per, nx, rank=rank)

    state_type = DebyeState if kind == "debye" else LorentzState
    return stage_poles(local, masks, Slab(nx, mesh.size), shape[1:], mesh=mesh,
                       n_poles=len(poles), state_type=state_type,
                       dtype=ade_state_dtype(), kind=kind)



from rfx.stepping.slab_update import _pad_slab_dispersion_coeffs


from rfx.stepping.slab_update import slab_dispersion_coeffs


def gather_array_x(slabs, ghost=1):
    """Gather slabs back into a single array, stripping ghost cells.

    Parameters
    ----------
    slabs : ndarray, shape (n_devices, nx_local + 2*ghost, ny, nz)
    ghost : int

    Returns
    -------
    arr : ndarray, shape (nx, ny, nz)
    """
    inner = slabs[:, ghost:-ghost, :, :]
    n_devices = inner.shape[0]
    nx_per = inner.shape[1]
    ny = inner.shape[2]
    nz = inner.shape[3]
    return inner.reshape(n_devices * nx_per, ny, nz)


def split_poles_x(arr, n_poles, n_devices, ghost):
    return cut_poles(arr[:n_poles], Slab(arr.shape[1], n_devices, ghost), "state")


def unstack_and_gather(sharded_arr, n_devices, nx_local, ghost, pad_x, nx):
    """Flatten a sharded x-slab array back to one full-domain array.

    The shard_map runners hold fields as ``(n_devices * nx_local, ny, nz)``
    -- rank slabs concatenated along x, each still carrying its ghost cells.
    This reshapes that into ``(n_devices, nx_local, ny, nz)``, hands it to
    :func:`gather_array_x` (which strips the ghosts and merges the rank and x
    axes), and trims the high-x PEC padding cells that were added to make
    ``nx`` divisible by ``n_devices``.

    Parameters
    ----------
    sharded_arr : jax array, shape ``(n_devices * nx_local, ny, nz)``
    n_devices, nx_local, ghost, pad_x : int
        The slab decomposition the array was built with.
    nx : int
        Trim bound: the UNPADDED global x cell count. ``distributed_v2``
        passes its local ``nx`` (``grid.shape[0]``); ``distributed_nu``
        passes ``sharded_grid.nx``. Those are the same quantity -- both
        equal ``n_devices * (nx_local - 2 * ghost) - pad_x``, i.e. the
        gathered length less the padding. It is an explicit parameter.

    Must remain pure JAX: callers wrap the runners in ``jax.grad`` to drive
    an objective from the gathered ``final_state``, and an earlier
    ``np.array(sharded_arr)`` host-pull here raised
    ``TracerArrayConversionError``. Guarded by
    tests/unit/runners/test_distributed_v2_gather_traceable.py.
    """
    total_x = sharded_arr.shape[0]
    assert total_x == n_devices * nx_local, (
        f"unstack: total_x={total_x} != n_devices*nx_local={n_devices * nx_local}"
    )
    stacked = jnp.reshape(
        sharded_arr,
        (n_devices, nx_local) + tuple(sharded_arr.shape[1:]),
    )
    gathered = gather_array_x(stacked, ghost)
    if pad_x > 0:
        gathered = gathered[:nx]
    return gathered














from rfx.stepping.slab_exchange import exchange_component_shmap


from rfx.stepping.slab_exchange import exchange_h_yee_shmap


from rfx.stepping.slab_exchange import exchange_e_yee_shmap



from rfx.stepping.slab_walls import apply_pec_face_shmap




from rfx.stepping.slab_walls import apply_pec_mask_shmap








from rfx.stepping.slab_update import inject_sources_shmap


from rfx.stepping.slab_update import sample_probes_shmap



from rfx.stepping.slab_update import _update_h_local_nu


from rfx.model.electric_metrics import _slab_x_lo_view, _slab_model_rows


from rfx.model.materials import slab_pole_fractions


from rfx.model.electric_metrics import slab_e_component_materials


from rfx.stepping.slab_update import component_e_coeffs


from rfx.stepping.slab_update import slab_e_coeffs


from rfx.stepping.slab_update import slab_e_materials_shmap


from rfx.stepping.slab_update import _update_e_local_nu



from rfx.stepping.slab_update import update_h_nu_shmap


from rfx.stepping.slab_update import update_e_nu_shmap




def _split_state(state, n_devices, ghost=1):
    layout = Slab(state.ex.shape[0], n_devices, ghost)
    return FDTDState(*(cut(field, layout, "state", mesh=None) for field in state[:6]),
                     step=jnp.broadcast_to(state.step, (n_devices,)))


from rfx.stepping.slab_update import magnetic_low_ghost


def _split_materials(materials, n_devices, ghost=1):
    """Legacy stacked material record; radius support is still refused."""
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(materials, lane="distributed material split", unsupported=True)
    return cut(materials, Slab(materials.eps_r.shape[0], n_devices, ghost),
               dict(eps_r="eps_r", sigma="sigma", mu_r="mu_r",
                    sigma_lumped="lumped", eps_r_lumped="lumped", sigma_film="lumped"), mesh=None)




def _split_debye_coeffs(coeffs: DebyeCoeffs, n_devices, ghost=1, *, dt):
    return cut_pole_coeffs(coeffs, Slab(jax.tree.leaves(coeffs.ca)[0].shape[0], n_devices, ghost),
                           "debye", dt=dt)


def _split_debye_state(state: DebyeState, n_devices, ghost=1):
    return cut_poles(state, Slab(state.px.shape[1], n_devices, ghost), "debye_state")


def _split_lorentz_coeffs(coeffs: LorentzCoeffs, n_devices, ghost=1, *, dt):
    return cut_pole_coeffs(coeffs, Slab(jax.tree.leaves(coeffs.ca)[0].shape[0], n_devices, ghost),
                           "lorentz", dt=dt)


def _split_lorentz_state(state: LorentzState, n_devices, ghost=1):
    return cut_poles(state, Slab(state.px.shape[1], n_devices, ghost), "lorentz_state")



from rfx.stepping.slab_update import _update_h_local


from rfx.stepping.slab_update import _update_e_local



from rfx.stepping.slab_update import _update_e_debye_local


from rfx.stepping.slab_update import _update_e_lorentz_local


from rfx.stepping.slab_update import _update_e_local_with_dispersion






def _distributed_boundary_layers(grid, n_devices, *, pec_faces=None,
                                 pmc_faces=None, cpml_layers=None):
    """Keep absorbers and walls on owned rows, including boxes without CPML.

    PEC writes the end E row: one owned row. The PMC image reads the
    adjacent interior H row and holds the high ghost with a zero update
    coefficient. Two owned rows retain the image stencil on its owner.
    An L-layer absorber needs L owned rows and
    the exchanged curl neighbour. Return the active depths for sizing.
    """
    pec_faces = set(getattr(grid, "pec_faces", None) or ()) if pec_faces is None else set(pec_faces)
    pmc_faces = set(getattr(grid, "pmc_faces", None) or ()) if pmc_faces is None else set(pmc_faces)
    from rfx.boundaries.depths import grid_face_depths

    layers = {face.name: face.realized for face in grid_face_depths(
        grid, pec_faces=pec_faces, pmc_faces=pmc_faces, budget=cpml_layers)}
    pad_x = (-grid.nx) % n_devices
    nx_per = (grid.nx + pad_x) // n_devices
    for face, rank, width in (("x_lo", 0, nx_per),
                              ("x_hi", n_devices - 1, max(0, nx_per - pad_x))):
        depth = layers[face]
        required = 2 if face in pmc_faces else 1
        if depth > 0 and width < depth:
            raise ValueError(
                f"distributed CPML face {face}: boundary rank {rank} has "
                f"{width} physical x cells, but {depth} absorber layers require "
                f"at least {depth} physical x cells plus 1 exchanged ghost cell "
                f"(nx={grid.nx}, devices={n_devices}, pad_x={pad_x}). "
                "Reduce devices or cpml_layers (or that face's "
                "lo_thickness/hi_thickness), or enlarge the x domain."
            )
        elif width < required:
            kind = "PMC" if face in pmc_faces else "PEC" if face in pec_faces else "boundary"
            raise ValueError(
                f"distributed {kind} face {face}: boundary rank {rank} has "
                f"{width} physical x cells, but {kind} requires at least "
                f"{required} physical x cells "
                f"(nx={grid.nx}, devices={n_devices}, pad_x={pad_x}). "
                "Reduce devices or enlarge the x domain."
            )
    return layers







def init_cpml_slabs(grid, nx_local, n_devices, *, mesh=None, **profile_options):
    """Build global profiles and the single-device psi layout on every rank.

    x faces are (s_x, Ny, Nz), y faces (L, s_y, Nz), and z faces
    (L, Ny, s_z). Merge the rank dimension into x when placing on a mesh.
    """
    from rfx.boundaries.cpml import CPMLState, init_cpml, _axis_buffer_depths
    from jax.sharding import NamedSharding

    _distributed_boundary_layers(grid, n_devices)
    params, _ = init_cpml(grid, **profile_options)
    depths = _axis_buffer_depths(grid, grid.cpml_layers)
    extents = (nx_local, grid.ny, grid.nz)
    state = {}
    for name in CPMLState._fields:
        axis = "xyz".index(name.split("_")[-1][0])
        shape = tuple(depths[i] if i == axis else extents[i] for i in range(3))
        if mesh is None:
            state[name] = jnp.zeros((n_devices, *shape), jnp.float32)
        else:
            state[name] = jnp.zeros((n_devices * shape[0], *shape[1:]), jnp.float32,
                                   device=NamedSharding(mesh, P("x")))
    return params, CPMLState(**state)


from rfx.stepping.slab_update import cpml_h_materials


from rfx.stepping.slab_update import cpml_slab_frame


from rfx.stepping.slab_update import cpml_absorbing_faces


from rfx.stepping.slab_update import _h_backward


from rfx.stepping.slab_update import _slab_curl_kwargs
