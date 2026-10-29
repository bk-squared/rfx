from __future__ import annotations

from typing import NamedTuple
import jax.numpy as jnp
from rfx.core.yee import FDTDState, init_materials
from rfx.stepping import update_e, update_h
from rfx.boundaries.pec import apply_pec


class DisjointSubgridState3D(NamedTuple):
    """Coarse exterior plus fine block field state.

    Coarse arrays retain full rectangular storage for implementation
    simplicity, but cells inside ``fine_region`` are inactive and must stay
    zero.  Energy accounting ignores those inactive cells.
    """

    ex_c: jnp.ndarray
    ey_c: jnp.ndarray
    ez_c: jnp.ndarray
    hx_c: jnp.ndarray
    hy_c: jnp.ndarray
    hz_c: jnp.ndarray
    ex_f: jnp.ndarray
    ey_f: jnp.ndarray
    ez_f: jnp.ndarray
    hx_f: jnp.ndarray
    hy_f: jnp.ndarray
    hz_f: jnp.ndarray
    step: int


def _yee_step(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    dt: float,
    dx: float,
    *,
    apply_outer_pec: bool,
):
    shape = ex.shape
    mats = init_materials(shape)
    fdtd = FDTDState(
        ex=ex, ey=ey, ez=ez,
        hx=hx, hy=hy, hz=hz,
        step=jnp.array(0, dtype=jnp.int32),
    )
    fdtd = update_h(fdtd, mats, dt, dx, high_wall=(False, False, False))
    fdtd = update_e(fdtd, mats, dt, dx, high_wall=(False, False, False))
    if apply_outer_pec:
        fdtd = apply_pec(fdtd)
    return fdtd.ex, fdtd.ey, fdtd.ez, fdtd.hx, fdtd.hy, fdtd.hz


def _yee_h_step(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    dt: float,
    dx: float,
):
    shape = ex.shape
    mats = init_materials(shape)
    fdtd = FDTDState(
        ex=ex, ey=ey, ez=ez,
        hx=hx, hy=hy, hz=hz,
        step=jnp.array(0, dtype=jnp.int32),
    )
    fdtd = update_h(fdtd, mats, dt, dx, high_wall=(False, False, False))
    return fdtd.hx, fdtd.hy, fdtd.hz


def _yee_e_step(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    dt: float,
    dx: float,
    *,
    apply_outer_pec: bool,
):
    shape = ex.shape
    mats = init_materials(shape)
    fdtd = FDTDState(
        ex=ex, ey=ey, ez=ez,
        hx=hx, hy=hy, hz=hz,
        step=jnp.array(0, dtype=jnp.int32),
    )
    fdtd = update_e(fdtd, mats, dt, dx, high_wall=(False, False, False))
    if apply_outer_pec:
        fdtd = apply_pec(fdtd)
    return fdtd.ex, fdtd.ey, fdtd.ez


def _apply_fine_pec_axes(
    state: DisjointSubgridState3D,
    axes: str,
) -> DisjointSubgridState3D:
    """Apply physical PEC axes to the fine block only."""
    fine = FDTDState(
        ex=state.ex_f,
        ey=state.ey_f,
        ez=state.ez_f,
        hx=state.hx_f,
        hy=state.hy_f,
        hz=state.hz_f,
        step=jnp.asarray(state.step, dtype=jnp.int32),
    )
    fine = apply_pec(fine, axes=axes)
    return state._replace(
        ex_f=fine.ex,
        ey_f=fine.ey,
        ez_f=fine.ez,
        hx_f=fine.hx,
        hy_f=fine.hy,
        hz_f=fine.hz,
    )


