"""Disjoint-domain 3D subgrid prototype scaffolding.

This module is the Stage-2 conservative restart point for true 3-D
subgridding.  It deliberately does **not** reuse the older overlay
``sbp_sat_3d`` topology.  The coarse grid owns only the exterior of a
rectangular hole; the fine grid owns the hole volume.  Interface SAT
coupling is added in later slices, after the topology and norm-compatible
face operators are pinned by tests.

Current slice:
- disjoint coarse-hole topology,
- norm-compatible 2-D face restriction/prolongation operators,
- six-face Maxwell cross-coupled interface SAT prototype,
- CFL-scaled SAT coefficient,
- disjoint energy accounting.

This is still a standalone research prototype, not a promoted
``Simulation.add_refinement`` support lane.  It provides physical
evidence for the conservative disjoint topology before any public API
integration.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0, FDTDState, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.subgridding.material_sat import interface_pair_deltas
from rfx.subgridding.jit_runner import (
    _prolong_node_aligned_2d,
    _restrict_node_aligned_2d,
)

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)
Z0 = np.sqrt(MU_0 / EPS_0)


class DisjointSubgridConfig3D(NamedTuple):
    """Configuration for a disjoint coarse-hole + fine-block domain."""

    shape_c: tuple[int, int, int]
    fine_region: tuple[int, int, int, int, int, int]
    shape_f: tuple[int, int, int]
    dx_c: float
    dx_f: float
    dt: float
    ratio: int
    sat_strength: float  # dimensionless SAT sigma; alpha = sigma * c0*dt/dx_f
    coarse_active_mask: jnp.ndarray
    face_projection: str
    shape_convention: str


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


def build_coarse_active_mask(
    shape_c: tuple[int, int, int],
    fine_region: tuple[int, int, int, int, int, int],
) -> jnp.ndarray:
    """Return a boolean mask that is false inside the fine-grid hole."""
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = fine_region
    if not (0 <= fi_lo < fi_hi <= shape_c[0]):
        raise ValueError(f"invalid i-range in fine_region={fine_region!r}")
    if not (0 <= fj_lo < fj_hi <= shape_c[1]):
        raise ValueError(f"invalid j-range in fine_region={fine_region!r}")
    if not (0 <= fk_lo < fk_hi <= shape_c[2]):
        raise ValueError(f"invalid k-range in fine_region={fine_region!r}")
    mask = jnp.ones(shape_c, dtype=jnp.bool_)
    return mask.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo:fk_hi].set(False)


def restrict_face_mean(fine_face: jnp.ndarray, ratio: int) -> jnp.ndarray:
    """Restrict a fine 2-D face to a coarse face by block averaging."""
    if fine_face.ndim != 2:
        raise ValueError(f"fine_face must be 2-D, got shape {fine_face.shape}")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    ny_f, nz_f = fine_face.shape
    if ny_f % ratio or nz_f % ratio:
        raise ValueError(
            f"fine face shape {fine_face.shape} must be divisible by ratio={ratio}"
        )
    return jnp.mean(
        fine_face.reshape(ny_f // ratio, ratio, nz_f // ratio, ratio),
        axis=(1, 3),
    )


def prolong_face_repeat(coarse_face: jnp.ndarray, ratio: int) -> jnp.ndarray:
    """Prolong a coarse 2-D face to a fine face by constant repeat."""
    if coarse_face.ndim != 2:
        raise ValueError(f"coarse_face must be 2-D, got shape {coarse_face.shape}")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    return jnp.repeat(jnp.repeat(coarse_face, ratio, axis=0), ratio, axis=1)


def prolong_face_linear(coarse_face: jnp.ndarray, ratio: int) -> jnp.ndarray:
    """Prolong a coarse 2-D face with separable linear interpolation.

    This diagnostic projection is not the default norm-adjoint repeat/mean
    pair.  It samples the coarse face at integer fine coordinates and linearly
    interpolates in between, preserving coarse values at fine indices that are
    integer multiples of ``ratio`` while reducing piecewise-constant trace
    jumps.
    """
    if coarse_face.ndim != 2:
        raise ValueError(f"coarse_face must be 2-D, got shape {coarse_face.shape}")
    if ratio <= 0:
        raise ValueError("ratio must be positive")

    def interp_axis(arr: jnp.ndarray, axis: int) -> jnp.ndarray:
        n = arr.shape[axis]
        coord = jnp.arange(n * ratio, dtype=arr.dtype) / ratio
        lo = jnp.floor(coord).astype(jnp.int32)
        hi = jnp.minimum(lo + 1, n - 1)
        weight = (coord - lo.astype(arr.dtype))
        if axis == 0:
            return (
                (1.0 - weight)[:, None] * jnp.take(arr, lo, axis=0)
                + weight[:, None] * jnp.take(arr, hi, axis=0)
            )
        return (
            (1.0 - weight)[None, :] * jnp.take(arr, lo, axis=1)
            + weight[None, :] * jnp.take(arr, hi, axis=1)
        )

    return interp_axis(interp_axis(coarse_face, 0), 1)


def _prolong_face_for_config(
    coarse_face: jnp.ndarray,
    ratio: int,
    projection: str,
    fine_shape: tuple[int, int],
) -> jnp.ndarray:
    if projection == "repeat_mean":
        return prolong_face_repeat(coarse_face, ratio)
    if projection == "linear":
        return prolong_face_linear(coarse_face, ratio)
    if projection == "node_adjoint":
        return _prolong_node_aligned_2d(
            coarse_face,
            fine_shape[0],
            fine_shape[1],
            ratio,
        )
    raise ValueError(
        f"unknown disjoint face_projection={projection!r}; "
        "expected 'repeat_mean', 'linear', or 'node_adjoint'"
    )


def _restrict_face_for_config(
    fine_face: jnp.ndarray,
    coarse_shape: tuple[int, int],
    ratio: int,
    projection: str,
) -> jnp.ndarray:
    if projection in {"repeat_mean", "linear"}:
        return restrict_face_mean(fine_face, ratio)
    if projection == "node_adjoint":
        return _restrict_node_aligned_2d(
            fine_face,
            coarse_shape[0],
            coarse_shape[1],
            ratio,
        )
    raise ValueError(
        f"unknown disjoint face_projection={projection!r}; "
        "expected 'repeat_mean', 'linear', or 'node_adjoint'"
    )


def init_disjoint_subgrid_3d(
    shape_c: tuple[int, int, int] = (20, 20, 20),
    fine_region: tuple[int, int, int, int, int, int] = (8, 12, 8, 12, 8, 12),
    *,
    dx_c: float = 0.003,
    ratio: int = 2,
    courant: float = 0.45,
    sat_strength: float = 0.05,
    face_projection: str = "repeat_mean",
    shape_convention: str = "cell_extent",
) -> tuple[DisjointSubgridConfig3D, DisjointSubgridState3D]:
    """Initialize the disjoint-domain Stage-2 prototype."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if shape_convention not in {"cell_extent", "endpoint_node"}:
        raise ValueError("shape_convention must be 'cell_extent' or 'endpoint_node'")
    if face_projection not in {"repeat_mean", "linear", "node_adjoint"}:
        raise ValueError(
            "face_projection must be 'repeat_mean', 'linear', or 'node_adjoint'"
        )
    if shape_convention == "endpoint_node" and face_projection != "node_adjoint":
        raise ValueError("endpoint_node shape requires face_projection='node_adjoint'")
    if shape_convention == "cell_extent" and face_projection == "node_adjoint":
        raise ValueError("node_adjoint projection requires endpoint_node shape")
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = fine_region
    active = build_coarse_active_mask(shape_c, fine_region)
    dx_f = dx_c / ratio
    dt = courant * dx_f / (C0 * np.sqrt(3.0))
    if shape_convention == "endpoint_node":
        shape_f = (
            (fi_hi - fi_lo - 1) * ratio + 1,
            (fj_hi - fj_lo - 1) * ratio + 1,
            (fk_hi - fk_lo - 1) * ratio + 1,
        )
    else:
        shape_f = (
            (fi_hi - fi_lo) * ratio,
            (fj_hi - fj_lo) * ratio,
            (fk_hi - fk_lo) * ratio,
        )
    config = DisjointSubgridConfig3D(
        shape_c=shape_c,
        fine_region=fine_region,
        shape_f=shape_f,
        dx_c=float(dx_c),
        dx_f=float(dx_f),
        dt=float(dt),
        ratio=int(ratio),
        sat_strength=float(sat_strength),
        coarse_active_mask=active,
        face_projection=str(face_projection),
        shape_convention=str(shape_convention),
    )
    def zc():
        return jnp.zeros(shape_c, dtype=jnp.float32)

    def zf():
        return jnp.zeros(shape_f, dtype=jnp.float32)
    state = DisjointSubgridState3D(
        ex_c=zc(), ey_c=zc(), ez_c=zc(),
        hx_c=zc(), hy_c=zc(), hz_c=zc(),
        ex_f=zf(), ey_f=zf(), ez_f=zf(),
        hx_f=zf(), hy_f=zf(), hz_f=zf(),
        step=0,
    )
    return config, state


def _zero_inactive(arr: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(mask, arr, jnp.zeros_like(arr))


def zero_coarse_hole(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Force all coarse fields in the fine-grid hole to zero."""
    m = config.coarse_active_mask
    return state._replace(
        ex_c=_zero_inactive(state.ex_c, m),
        ey_c=_zero_inactive(state.ey_c, m),
        ez_c=_zero_inactive(state.ez_c, m),
        hx_c=_zero_inactive(state.hx_c, m),
        hy_c=_zero_inactive(state.hy_c, m),
        hz_c=_zero_inactive(state.hz_c, m),
    )


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
    fdtd = update_h(fdtd, mats, dt, dx)
    fdtd = update_e(fdtd, mats, dt, dx)
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
    fdtd = update_h(fdtd, mats, dt, dx)
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
    fdtd = update_e(fdtd, mats, dt, dx)
    if apply_outer_pec:
        fdtd = apply_pec(fdtd)
    return fdtd.ex, fdtd.ey, fdtd.ez


def step_disjoint_topology_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Topology-only disjoint update.

    Coarse and fine blocks are advanced independently with PEC outer
    walls, and the coarse hole is zeroed after H and E updates so the
    coarse grid cannot evolve inside the fine-owned volume.

    This function intentionally has no interface coupling yet.
    """
    state = zero_coarse_hole(state, config)
    ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _yee_step(
        state.ex_c, state.ey_c, state.ez_c,
        state.hx_c, state.hy_c, state.hz_c,
        config.dt, config.dx_c,
        apply_outer_pec=True,
    )
    coarse = state._replace(
        ex_c=ex_c, ey_c=ey_c, ez_c=ez_c,
        hx_c=hx_c, hy_c=hy_c, hz_c=hz_c,
    )
    coarse = zero_coarse_hole(coarse, config)

    # The fine block's six faces are artificial coarse/fine interfaces, not
    # physical PEC walls.  Leaving them un-zeroed lets the SAT terms carry
    # tangential fields across the interface instead of first clamping them to a
    # conductor boundary.  The prototype still uses zero-padded curl stencils at
    # those faces until the paper-derived interface closure replaces this slice.
    ex_f, ey_f, ez_f, hx_f, hy_f, hz_f = _yee_step(
        state.ex_f, state.ey_f, state.ez_f,
        state.hx_f, state.hy_f, state.hz_f,
        config.dt, config.dx_f,
        apply_outer_pec=False,
    )
    return coarse._replace(
        ex_f=ex_f, ey_f=ey_f, ez_f=ez_f,
        hx_f=hx_f, hy_f=hy_f, hz_f=hz_f,
        step=state.step + 1,
    )


def _levi_civita(i: int, j: int, k: int) -> int:
    if len({i, j, k}) < 3:
        return 0
    if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        return 1
    return -1


_E_NAMES = ("ex", "ey", "ez")
_H_NAMES = ("hx", "hy", "hz")


def _sat_alpha(config: DisjointSubgridConfig3D) -> float:
    """CFL/SBP-scaled dimensionless SAT coefficient.

    Semi-discrete SAT penalties have the form ``(dt/eps0/dn) * H`` for
    E updates and ``(dt/mu0/dn) * E`` for H updates.  Using
    ``1/eps0 = Z0*c0`` and ``1/mu0 = c0/Z0`` gives the impedance-balanced
    discrete scaling used below:

    ``alpha = sigma * c0 * dt / dx_f``.

    ``sigma`` is kept small in this prototype until the six-face block is
    matched against a full paper-derived SBP closure.
    """
    return float(config.sat_strength * C0 * config.dt / config.dx_f)


def _face_slices(
    config: DisjointSubgridConfig3D,
    face: str,
) -> tuple[int, int, tuple, tuple]:
    """Return ``(axis, normal_sign, coarse_slice, fine_slice)`` for a face."""
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    if face == "x_lo":
        if fi_lo <= 0:
            raise ValueError("x_lo SAT needs an active coarse cell before the hole")
        return 0, +1, (fi_lo - 1, slice(fj_lo, fj_hi), slice(fk_lo, fk_hi)), (0, slice(None), slice(None))
    if face == "x_hi":
        if fi_hi >= config.shape_c[0]:
            raise ValueError("x_hi SAT needs an active coarse cell after the hole")
        return 0, -1, (fi_hi, slice(fj_lo, fj_hi), slice(fk_lo, fk_hi)), (-1, slice(None), slice(None))
    if face == "y_lo":
        if fj_lo <= 0:
            raise ValueError("y_lo SAT needs an active coarse cell before the hole")
        return 1, +1, (slice(fi_lo, fi_hi), fj_lo - 1, slice(fk_lo, fk_hi)), (slice(None), 0, slice(None))
    if face == "y_hi":
        if fj_hi >= config.shape_c[1]:
            raise ValueError("y_hi SAT needs an active coarse cell after the hole")
        return 1, -1, (slice(fi_lo, fi_hi), fj_hi, slice(fk_lo, fk_hi)), (slice(None), -1, slice(None))
    if face == "z_lo":
        if fk_lo <= 0:
            raise ValueError("z_lo SAT needs an active coarse cell before the hole")
        return 2, +1, (slice(fi_lo, fi_hi), slice(fj_lo, fj_hi), fk_lo - 1), (slice(None), slice(None), 0)
    if face == "z_hi":
        if fk_hi >= config.shape_c[2]:
            raise ValueError("z_hi SAT needs an active coarse cell after the hole")
        return 2, -1, (slice(fi_lo, fi_hi), slice(fj_lo, fj_hi), fk_hi), (slice(None), slice(None), -1)
    raise ValueError(f"unknown face {face!r}")


def _z_face_active_ghost_slices(
    config: DisjointSubgridConfig3D,
    face: str,
) -> tuple[tuple, tuple, tuple]:
    """Return ``(coarse_active, coarse_ghost, fine_face)`` for z-curl closure."""
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    c_xy = (slice(fi_lo, fi_hi), slice(fj_lo, fj_hi))
    if face == "z_lo":
        if fk_lo <= 0:
            raise ValueError("z_lo ghost closure needs an active coarse cell below")
        return (
            (*c_xy, fk_lo - 1),
            (*c_xy, fk_lo),
            (slice(None), slice(None), 0),
        )
    if face == "z_hi":
        if fk_hi >= config.shape_c[2]:
            raise ValueError("z_hi ghost closure needs an active coarse cell above")
        return (
            (*c_xy, fk_hi),
            (*c_xy, fk_hi - 1),
            (slice(None), slice(None), -1),
        )
    raise ValueError(f"unknown z face {face!r}")


def _available_z_faces(config: DisjointSubgridConfig3D) -> tuple[str, ...]:
    faces = []
    if config.fine_region[4] > 0:
        faces.append("z_lo")
    if config.fine_region[5] < config.shape_c[2]:
        faces.append("z_hi")
    return tuple(faces)


def _get_field(state: DisjointSubgridState3D, name: str, grid: str):
    return getattr(state, f"{name}_{grid}")


def _set_field(state: DisjointSubgridState3D, name: str, grid: str, value):
    return state._replace(**{f"{name}_{grid}": value})


def _add_field_delta(deltas: dict[str, jnp.ndarray], name: str, grid: str, sl, delta):
    key = f"{name}_{grid}"
    deltas[key] = deltas[key].at[sl].add(jnp.asarray(delta, dtype=deltas[key].dtype))


def _apply_faces_maxwell_sat_from_snapshot(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    faces: tuple[str, ...],
    *,
    strength: float | None = None,
    update_e: bool = True,
    update_h: bool = True,
) -> DisjointSubgridState3D:
    """Apply one or more face SATs from a frozen pre-SAT snapshot.

    The Stage-2 prototype is a split-step SAT approximation, but the E and H
    penalties for a given SAT application must still be evaluated from the
    same pre-SAT field state.  Mutating one face/component before computing
    another makes the result order-dependent at shared edges/corners and
    weakens the discrete-energy interpretation.  This helper accumulates all
    field deltas first and applies them once.
    """
    alpha_base = _sat_alpha(config)
    if strength is not None:
        alpha_base = float(strength * C0 * config.dt / config.dx_f)
    if alpha_base < 0:
        raise ValueError("SAT strength must be non-negative")

    field_names = [f"{name}_{grid}" for grid in ("c", "f") for name in (*_E_NAMES, *_H_NAMES)]
    deltas = {name: jnp.zeros_like(getattr(state, name)) for name in field_names}
    alpha = jnp.asarray(alpha_base, dtype=state.ex_c.dtype)
    z0 = jnp.asarray(Z0, dtype=state.ex_c.dtype)
    ratio = config.ratio

    for face in faces:
        axis, normal_sign, c_sl, f_sl = _face_slices(config, face)
        tangential_axes = [ax for ax in (0, 1, 2) if ax != axis]

        if update_e:
            for e_axis in tangential_axes:
                # (n × H)_e_axis = sign_e * H_h_axis
                h_axis = next(ax for ax in tangential_axes if ax != e_axis)
                sign_e = normal_sign * _levi_civita(e_axis, axis, h_axis)

                e_name = _E_NAMES[e_axis]
                h_name = _H_NAMES[h_axis]
                hc = _get_field(state, h_name, "c")
                hf = _get_field(state, h_name, "f")

                hc_face = hc[c_sl]
                hf_face = hf[f_sl]
                d_h_c = (
                    _restrict_face_for_config(
                        hf_face,
                        hc_face.shape,
                        ratio,
                        config.face_projection,
                    )
                    - hc_face
                )
                d_h_f = (
                    _prolong_face_for_config(
                        hc_face,
                        ratio,
                        config.face_projection,
                        hf_face.shape,
                    )
                    - hf_face
                )

                _add_field_delta(
                    deltas,
                    e_name,
                    "c",
                    c_sl,
                    alpha * z0 * sign_e * d_h_c,
                )
                _add_field_delta(
                    deltas,
                    e_name,
                    "f",
                    f_sl,
                    -alpha * z0 * sign_e * d_h_f,
                )

        if update_h:
            for h_axis in tangential_axes:
                # -(n × E)_h_axis = sign_h * E_e_axis
                e_axis = next(ax for ax in tangential_axes if ax != h_axis)
                sign_h = -normal_sign * _levi_civita(h_axis, axis, e_axis)

                h_name = _H_NAMES[h_axis]
                e_name = _E_NAMES[e_axis]
                ec = _get_field(state, e_name, "c")
                ef = _get_field(state, e_name, "f")

                ec_face = ec[c_sl]
                ef_face = ef[f_sl]
                d_e_c = (
                    _restrict_face_for_config(
                        ef_face,
                        ec_face.shape,
                        ratio,
                        config.face_projection,
                    )
                    - ec_face
                )
                d_e_f = (
                    _prolong_face_for_config(
                        ec_face,
                        ratio,
                        config.face_projection,
                        ef_face.shape,
                    )
                    - ef_face
                )

                _add_field_delta(
                    deltas,
                    h_name,
                    "c",
                    c_sl,
                    alpha / z0 * sign_h * d_e_c,
                )
                _add_field_delta(
                    deltas,
                    h_name,
                    "f",
                    f_sl,
                    -alpha / z0 * sign_h * d_e_f,
                )

    return state._replace(
        **{name: getattr(state, name) + delta for name, delta in deltas.items()}
    )


def apply_face_maxwell_sat(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    face: str,
    *,
    strength: float | None = None,
) -> DisjointSubgridState3D:
    """Apply impedance-balanced Maxwell SAT on one coarse/fine face."""
    return _apply_faces_maxwell_sat_from_snapshot(
        state, config, (face,), strength=strength,
    )


ALL_FACES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")


def apply_all_faces_maxwell_sat(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Apply six-face Maxwell cross-coupled SAT."""
    return _apply_faces_maxwell_sat_from_snapshot(state, config, ALL_FACES)


def apply_z_faces_maxwell_sat(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Apply Maxwell SAT on only the available z-normal coarse/fine faces.

    Public centered z-slab integration spans the physical x/y domain, so the
    fine block has no active coarse cells outside x/y faces.  This helper is
    the Stage-2 disjoint counterpart for that slab geometry: couple z-lo/z-hi
    artificial interfaces while x/y faces remain physical boundaries.
    """
    faces = []
    if config.fine_region[4] > 0:
        faces.append("z_lo")
    if config.fine_region[5] < config.shape_c[2]:
        faces.append("z_hi")
    if not faces:
        return state
    return _apply_faces_maxwell_sat_from_snapshot(state, config, tuple(faces))


def apply_z_faces_maxwell_e_sat(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Apply only E-field z-face SAT from tangential-H mismatch."""
    faces = []
    if config.fine_region[4] > 0:
        faces.append("z_lo")
    if config.fine_region[5] < config.shape_c[2]:
        faces.append("z_hi")
    if not faces:
        return state
    return _apply_faces_maxwell_sat_from_snapshot(
        state,
        config,
        tuple(faces),
        update_e=True,
        update_h=False,
    )


def apply_z_faces_maxwell_h_sat(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Apply only H-field z-face SAT from tangential-E mismatch."""
    faces = []
    if config.fine_region[4] > 0:
        faces.append("z_lo")
    if config.fine_region[5] < config.shape_c[2]:
        faces.append("z_hi")
    if not faces:
        return state
    return _apply_faces_maxwell_sat_from_snapshot(
        state,
        config,
        tuple(faces),
        update_e=False,
        update_h=True,
    )


def _apply_z_faces_upwind_pair_sat_from_snapshot(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    *,
    strength: float | None = None,
    update_e: bool = True,
    update_h: bool = True,
) -> DisjointSubgridState3D:
    """Apply material-SAT-style impedance-upwind z-face pair coupling.

    The earlier disjoint SAT is a same-time mismatch penalty: tangential-E
    receives tangential-H mismatch and tangential-H receives tangential-E
    mismatch.  A true characteristic/upwind interface state couples both jumps
    inside each tangential pair.  This research-only helper reuses the
    material-weighted pair algebra in vacuum:

    * pair A: ``U=E_x, V=H_y``
    * pair B: ``U=E_y, V=-H_x``

    It keeps the coarse/fine projection split explicit: coarse-side deltas use
    restricted fine traces and fine-side deltas use prolonged coarse traces.
    """
    scale = float(config.sat_strength if strength is None else strength)
    if scale < 0.0:
        raise ValueError("SAT strength must be non-negative")
    if scale == 0.0 or (not update_e and not update_h):
        return state

    base = state
    deltas = {
        name: jnp.zeros_like(getattr(base, name))
        for name in ("ex_c", "ey_c", "hx_c", "hy_c", "ex_f", "ey_f", "hx_f", "hy_f")
    }
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = config.fine_region
    c_xy = (slice(fi_lo, fi_hi), slice(fj_lo, fj_hi))
    eps0 = jnp.asarray(EPS_0, dtype=state.ex_c.dtype)
    mu0 = jnp.asarray(MU_0, dtype=state.ex_c.dtype)
    ratio = config.ratio

    def apply_pair(
        *,
        face: str,
        u_c,
        v_c,
        u_f,
        v_f,
        add_u_c,
        add_v_c,
        add_u_f,
        add_v_f,
    ):
        if face == "z_lo":
            c = (*c_xy, fk_lo - 1)
            f = (slice(None), slice(None), 0)
            coarse_shape = u_c[c].shape
            fine_shape = u_f[f].shape
            coarse = interface_pair_deltas(
                u_c[c],
                v_c[c],
                _restrict_face_for_config(
                    u_f[f],
                    coarse_shape,
                    ratio,
                    config.face_projection,
                ),
                _restrict_face_for_config(
                    v_f[f],
                    coarse_shape,
                    ratio,
                    config.face_projection,
                ),
                epsilon_lower=eps0,
                mu_lower=mu0,
                epsilon_upper=eps0,
                mu_upper=mu0,
                h_lower=config.dx_c,
                h_upper=config.dx_f,
                dt=config.dt,
            )
            fine = interface_pair_deltas(
                _prolong_face_for_config(
                    u_c[c],
                    ratio,
                    config.face_projection,
                    fine_shape,
                ),
                _prolong_face_for_config(
                    v_c[c],
                    ratio,
                    config.face_projection,
                    fine_shape,
                ),
                u_f[f],
                v_f[f],
                epsilon_lower=eps0,
                mu_lower=mu0,
                epsilon_upper=eps0,
                mu_upper=mu0,
                h_lower=config.dx_c,
                h_upper=config.dx_f,
                dt=config.dt,
            )
            if update_e:
                add_u_c(c, scale * coarse.du_lower)
                add_u_f(f, scale * fine.du_upper)
            if update_h:
                add_v_c(c, scale * coarse.dv_lower)
                add_v_f(f, scale * fine.dv_upper)
        elif face == "z_hi":
            c = (*c_xy, fk_hi)
            f = (slice(None), slice(None), -1)
            coarse_shape = u_c[c].shape
            fine_shape = u_f[f].shape
            coarse = interface_pair_deltas(
                _restrict_face_for_config(
                    u_f[f],
                    coarse_shape,
                    ratio,
                    config.face_projection,
                ),
                _restrict_face_for_config(
                    v_f[f],
                    coarse_shape,
                    ratio,
                    config.face_projection,
                ),
                u_c[c],
                v_c[c],
                epsilon_lower=eps0,
                mu_lower=mu0,
                epsilon_upper=eps0,
                mu_upper=mu0,
                h_lower=config.dx_f,
                h_upper=config.dx_c,
                dt=config.dt,
            )
            fine = interface_pair_deltas(
                u_f[f],
                v_f[f],
                _prolong_face_for_config(
                    u_c[c],
                    ratio,
                    config.face_projection,
                    fine_shape,
                ),
                _prolong_face_for_config(
                    v_c[c],
                    ratio,
                    config.face_projection,
                    fine_shape,
                ),
                epsilon_lower=eps0,
                mu_lower=mu0,
                epsilon_upper=eps0,
                mu_upper=mu0,
                h_lower=config.dx_f,
                h_upper=config.dx_c,
                dt=config.dt,
            )
            if update_e:
                add_u_c(c, scale * coarse.du_upper)
                add_u_f(f, scale * fine.du_lower)
            if update_h:
                add_v_c(c, scale * coarse.dv_upper)
                add_v_f(f, scale * fine.dv_lower)
        else:
            raise ValueError(f"unknown z face {face!r}")

    def add_ex_c(sl, delta):
        deltas["ex_c"] = deltas["ex_c"].at[sl].add(delta)

    def add_ey_c(sl, delta):
        deltas["ey_c"] = deltas["ey_c"].at[sl].add(delta)

    def add_hy_c(sl, delta):
        deltas["hy_c"] = deltas["hy_c"].at[sl].add(delta)

    def add_hx_c_from_pair_b(sl, delta_v):
        deltas["hx_c"] = deltas["hx_c"].at[sl].add(-delta_v)

    def add_ex_f(sl, delta):
        deltas["ex_f"] = deltas["ex_f"].at[sl].add(delta)

    def add_ey_f(sl, delta):
        deltas["ey_f"] = deltas["ey_f"].at[sl].add(delta)

    def add_hy_f(sl, delta):
        deltas["hy_f"] = deltas["hy_f"].at[sl].add(delta)

    def add_hx_f_from_pair_b(sl, delta_v):
        deltas["hx_f"] = deltas["hx_f"].at[sl].add(-delta_v)

    for face in ("z_lo", "z_hi"):
        if face == "z_lo" and fk_lo <= 0:
            continue
        if face == "z_hi" and fk_hi >= config.shape_c[2]:
            continue
        apply_pair(
            face=face,
            u_c=base.ex_c,
            v_c=base.hy_c,
            u_f=base.ex_f,
            v_f=base.hy_f,
            add_u_c=add_ex_c,
            add_v_c=add_hy_c,
            add_u_f=add_ex_f,
            add_v_f=add_hy_f,
        )
        apply_pair(
            face=face,
            u_c=base.ey_c,
            v_c=-base.hx_c,
            u_f=base.ey_f,
            v_f=-base.hx_f,
            add_u_c=add_ey_c,
            add_v_c=add_hx_c_from_pair_b,
            add_u_f=add_ey_f,
            add_v_f=add_hx_f_from_pair_b,
        )
    return base._replace(
        **{name: getattr(base, name) + delta for name, delta in deltas.items()}
    )


def apply_z_faces_upwind_pair_sat(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    *,
    strength: float | None = None,
) -> DisjointSubgridState3D:
    """Apply research-only impedance-upwind pair SAT on z faces."""
    return _apply_z_faces_upwind_pair_sat_from_snapshot(
        state,
        config,
        strength=strength,
    )


def apply_z_faces_upwind_pair_e_sat(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    *,
    strength: float | None = None,
) -> DisjointSubgridState3D:
    """Apply only E-field corrections from impedance-upwind z-face pair SAT."""
    return _apply_z_faces_upwind_pair_sat_from_snapshot(
        state,
        config,
        strength=strength,
        update_e=True,
        update_h=False,
    )


def apply_z_faces_upwind_pair_h_sat(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    *,
    strength: float | None = None,
) -> DisjointSubgridState3D:
    """Apply only H-field corrections from impedance-upwind z-face pair SAT."""
    return _apply_z_faces_upwind_pair_sat_from_snapshot(
        state,
        config,
        strength=strength,
        update_e=False,
        update_h=True,
    )


def _fill_coarse_z_ghost_from_fine(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    field_names: tuple[str, ...],
) -> DisjointSubgridState3D:
    """Fill one layer of coarse hole ghost cells from restricted fine faces."""
    updates = {}
    for name in field_names:
        coarse = getattr(state, f"{name}_c")
        fine = getattr(state, f"{name}_f")
        for face in _available_z_faces(config):
            _active_sl, ghost_sl, fine_sl = _z_face_active_ghost_slices(config, face)
            coarse = coarse.at[ghost_sl].set(
                _restrict_face_for_config(
                    fine[fine_sl],
                    coarse[ghost_sl].shape,
                    config.ratio,
                    config.face_projection,
                )
            )
        updates[f"{name}_c"] = coarse
    return state._replace(**updates)


def _prolong_coarse_z_face_to_fine(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    field_name: str,
    face: str,
) -> jnp.ndarray:
    """Prolong an active coarse z-face trace onto the matching fine face."""
    active_sl, _ghost_sl, fine_sl = _z_face_active_ghost_slices(config, face)
    coarse_face = getattr(state, f"{field_name}_c")[active_sl]
    fine_shape = getattr(state, f"{field_name}_f")[fine_sl].shape
    return _prolong_face_for_config(
        coarse_face,
        config.ratio,
        config.face_projection,
        fine_shape,
    )


def _restrict_fine_z_face_to_coarse(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    field_name: str,
    face: str,
) -> jnp.ndarray:
    """Restrict a fine z-face trace onto the matching active coarse face."""
    active_sl, _ghost_sl, fine_sl = _z_face_active_ghost_slices(config, face)
    coarse_shape = getattr(state, f"{field_name}_c")[active_sl].shape
    fine_face = getattr(state, f"{field_name}_f")[fine_sl]
    return _restrict_face_for_config(
        fine_face,
        coarse_shape,
        config.ratio,
        config.face_projection,
    )


def _fine_h_step_z_ghost(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fine H update with a coarse z-hi ghost trace in the forward z curl."""
    hx_f, hy_f, hz_f = _yee_h_step(
        state.ex_f, state.ey_f, state.ez_f,
        state.hx_f, state.hy_f, state.hz_f,
        config.dt, config.dx_f,
    )
    if config.fine_region[5] < config.shape_c[2]:
        ghost_ex = _prolong_coarse_z_face_to_fine(state, config, "ex", "z_hi")
        ghost_ey = _prolong_coarse_z_face_to_fine(state, config, "ey", "z_hi")
        coef = jnp.asarray(config.dt / (MU_0 * config.dx_f), dtype=hx_f.dtype)
        zhi = (slice(None), slice(None), -1)
        hx_f = hx_f.at[zhi].add(coef * ghost_ey)
        hy_f = hy_f.at[zhi].add(-coef * ghost_ex)
    return hx_f, hy_f, hz_f


def _fine_e_step_z_ghost(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fine E update with a coarse z-lo ghost trace in the backward z curl."""
    ex_f, ey_f, ez_f = _yee_e_step(
        state.ex_f, state.ey_f, state.ez_f,
        state.hx_f, state.hy_f, state.hz_f,
        config.dt, config.dx_f,
        apply_outer_pec=False,
    )
    if config.fine_region[4] > 0:
        ghost_hx = _prolong_coarse_z_face_to_fine(state, config, "hx", "z_lo")
        ghost_hy = _prolong_coarse_z_face_to_fine(state, config, "hy", "z_lo")
        coef = jnp.asarray(config.dt / (EPS_0 * config.dx_f), dtype=ex_f.dtype)
        zlo = (slice(None), slice(None), 0)
        ex_f = ex_f.at[zlo].add(coef * ghost_hy)
        ey_f = ey_f.at[zlo].add(-coef * ghost_hx)
    return ex_f, ey_f, ez_f


def step_disjoint_z_slab_ghost_curl_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Z-slab step with coarse/fine traces inserted into z-curl stencils.

    This research-only operator is closer to a Yee/SBP interface closure than
    post-step scalar trace projection: the active coarse cells see restricted
    fine traces through one layer of ghost cells, and the fine z-boundary curl
    terms see prolonged active coarse traces.  It remains fail-closed until the
    waveform, energy/residual, and crossval gates pass.
    """
    state = zero_coarse_hole(state, config)

    coarse_e_ghost = _fill_coarse_z_ghost_from_fine(state, config, _E_NAMES)
    hx_c, hy_c, hz_c = _yee_h_step(
        coarse_e_ghost.ex_c, coarse_e_ghost.ey_c, coarse_e_ghost.ez_c,
        state.hx_c, state.hy_c, state.hz_c,
        config.dt, config.dx_c,
    )
    hx_f, hy_f, hz_f = _fine_h_step_z_ghost(state, config)
    state = state._replace(
        hx_c=hx_c, hy_c=hy_c, hz_c=hz_c,
        hx_f=hx_f, hy_f=hy_f, hz_f=hz_f,
    )
    state = zero_coarse_hole(state, config)

    coarse_h_ghost = _fill_coarse_z_ghost_from_fine(state, config, _H_NAMES)
    ex_c, ey_c, ez_c = _yee_e_step(
        coarse_h_ghost.ex_c, coarse_h_ghost.ey_c, coarse_h_ghost.ez_c,
        coarse_h_ghost.hx_c, coarse_h_ghost.hy_c, coarse_h_ghost.hz_c,
        config.dt, config.dx_c,
        apply_outer_pec=True,
    )
    ex_f, ey_f, ez_f = _fine_e_step_z_ghost(state, config)
    state = state._replace(
        ex_c=ex_c, ey_c=ey_c, ez_c=ez_c,
        ex_f=ex_f, ey_f=ey_f, ez_f=ez_f,
        step=state.step + 1,
    )
    state = _apply_fine_pec_axes(state, "xy")
    return zero_coarse_hole(state, config)


def step_disjoint_z_slab_ghost_curl_sat_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Z-curl ghost step followed by the existing z-face Maxwell SAT."""
    state = step_disjoint_z_slab_ghost_curl_3d(state, config)
    state = apply_z_faces_maxwell_sat(state, config)
    state = _apply_fine_pec_axes(state, "xy")
    return zero_coarse_hole(state, config)


def _interface_dz(config: DisjointSubgridConfig3D) -> float:
    """Distance between staggered coarse/fine z traces used by metric closure."""
    return 0.5 * (config.dx_c + config.dx_f)


def _apply_h_z_metric_correction(
    hx,
    hy,
    *,
    ex_self,
    ey_self,
    ex_trace,
    ey_trace,
    sl,
    dt: float,
    dx_owner: float,
    dz_interface: float,
):
    """Replace zero-padded forward z-derivatives with interface metric ones."""
    inv_owner = jnp.asarray(1.0 / dx_owner, dtype=hx.dtype)
    inv_iface = jnp.asarray(1.0 / dz_interface, dtype=hx.dtype)
    coef = jnp.asarray(dt / MU_0, dtype=hx.dtype)
    corr_dey_dz = ey_trace * inv_iface + ey_self * (inv_owner - inv_iface)
    corr_dex_dz = ex_trace * inv_iface + ex_self * (inv_owner - inv_iface)
    hx = hx.at[sl].add(coef * corr_dey_dz)
    hy = hy.at[sl].add(-coef * corr_dex_dz)
    return hx, hy


def _apply_e_z_metric_correction(
    ex,
    ey,
    *,
    hx_self,
    hy_self,
    hx_trace,
    hy_trace,
    sl,
    dt: float,
    dx_owner: float,
    dz_interface: float,
):
    """Replace zero-padded backward z-derivatives with interface metric ones."""
    inv_owner = jnp.asarray(1.0 / dx_owner, dtype=ex.dtype)
    inv_iface = jnp.asarray(1.0 / dz_interface, dtype=ex.dtype)
    coef = jnp.asarray(dt / EPS_0, dtype=ex.dtype)
    corr_dhy_dz = -hy_trace * inv_iface + hy_self * (inv_iface - inv_owner)
    corr_dhx_dz = -hx_trace * inv_iface + hx_self * (inv_iface - inv_owner)
    ex = ex.at[sl].add(-coef * corr_dhy_dz)
    ey = ey.at[sl].add(coef * corr_dhx_dz)
    return ex, ey


def _coarse_h_step_z_metric(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Coarse H update with z-lo fine trace and effective interface distance."""
    hx_c, hy_c, hz_c = _yee_h_step(
        state.ex_c, state.ey_c, state.ez_c,
        state.hx_c, state.hy_c, state.hz_c,
        config.dt, config.dx_c,
    )
    if config.fine_region[4] > 0:
        active_sl, _ghost_sl, _fine_sl = _z_face_active_ghost_slices(config, "z_lo")
        trace_ex = _restrict_fine_z_face_to_coarse(state, config, "ex", "z_lo")
        trace_ey = _restrict_fine_z_face_to_coarse(state, config, "ey", "z_lo")
        hx_c, hy_c = _apply_h_z_metric_correction(
            hx_c,
            hy_c,
            ex_self=state.ex_c[active_sl],
            ey_self=state.ey_c[active_sl],
            ex_trace=trace_ex,
            ey_trace=trace_ey,
            sl=active_sl,
            dt=config.dt,
            dx_owner=config.dx_c,
            dz_interface=_interface_dz(config),
        )
    return hx_c, hy_c, hz_c


def _coarse_e_step_z_metric(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Coarse E update with z-hi fine trace and effective interface distance."""
    ex_c, ey_c, ez_c = _yee_e_step(
        state.ex_c, state.ey_c, state.ez_c,
        state.hx_c, state.hy_c, state.hz_c,
        config.dt, config.dx_c,
        apply_outer_pec=True,
    )
    if config.fine_region[5] < config.shape_c[2]:
        active_sl, _ghost_sl, _fine_sl = _z_face_active_ghost_slices(config, "z_hi")
        trace_hx = _restrict_fine_z_face_to_coarse(state, config, "hx", "z_hi")
        trace_hy = _restrict_fine_z_face_to_coarse(state, config, "hy", "z_hi")
        ex_c, ey_c = _apply_e_z_metric_correction(
            ex_c,
            ey_c,
            hx_self=state.hx_c[active_sl],
            hy_self=state.hy_c[active_sl],
            hx_trace=trace_hx,
            hy_trace=trace_hy,
            sl=active_sl,
            dt=config.dt,
            dx_owner=config.dx_c,
            dz_interface=_interface_dz(config),
        )
    return ex_c, ey_c, ez_c


def _fine_h_step_z_metric(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fine H update with z-hi coarse trace and effective interface distance."""
    hx_f, hy_f, hz_f = _yee_h_step(
        state.ex_f, state.ey_f, state.ez_f,
        state.hx_f, state.hy_f, state.hz_f,
        config.dt, config.dx_f,
    )
    if config.fine_region[5] < config.shape_c[2]:
        sl = (slice(None), slice(None), -1)
        trace_ex = _prolong_coarse_z_face_to_fine(state, config, "ex", "z_hi")
        trace_ey = _prolong_coarse_z_face_to_fine(state, config, "ey", "z_hi")
        hx_f, hy_f = _apply_h_z_metric_correction(
            hx_f,
            hy_f,
            ex_self=state.ex_f[sl],
            ey_self=state.ey_f[sl],
            ex_trace=trace_ex,
            ey_trace=trace_ey,
            sl=sl,
            dt=config.dt,
            dx_owner=config.dx_f,
            dz_interface=_interface_dz(config),
        )
    return hx_f, hy_f, hz_f


def _fine_e_step_z_metric(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fine E update with z-lo coarse trace and effective interface distance."""
    ex_f, ey_f, ez_f = _yee_e_step(
        state.ex_f, state.ey_f, state.ez_f,
        state.hx_f, state.hy_f, state.hz_f,
        config.dt, config.dx_f,
        apply_outer_pec=False,
    )
    if config.fine_region[4] > 0:
        sl = (slice(None), slice(None), 0)
        trace_hx = _prolong_coarse_z_face_to_fine(state, config, "hx", "z_lo")
        trace_hy = _prolong_coarse_z_face_to_fine(state, config, "hy", "z_lo")
        ex_f, ey_f = _apply_e_z_metric_correction(
            ex_f,
            ey_f,
            hx_self=state.hx_f[sl],
            hy_self=state.hy_f[sl],
            hx_trace=trace_hx,
            hy_trace=trace_hy,
            sl=sl,
            dt=config.dt,
            dx_owner=config.dx_f,
            dz_interface=_interface_dz(config),
        )
    return ex_f, ey_f, ez_f


def step_disjoint_z_slab_metric_curl_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Z-slab step with interface-distance-aware coarse/fine z-curl closure."""
    state = zero_coarse_hole(state, config)

    hx_c, hy_c, hz_c = _coarse_h_step_z_metric(state, config)
    hx_f, hy_f, hz_f = _fine_h_step_z_metric(state, config)
    state = state._replace(
        hx_c=hx_c, hy_c=hy_c, hz_c=hz_c,
        hx_f=hx_f, hy_f=hy_f, hz_f=hz_f,
    )
    state = zero_coarse_hole(state, config)

    ex_c, ey_c, ez_c = _coarse_e_step_z_metric(state, config)
    ex_f, ey_f, ez_f = _fine_e_step_z_metric(state, config)
    state = state._replace(
        ex_c=ex_c, ey_c=ey_c, ez_c=ez_c,
        ex_f=ex_f, ey_f=ey_f, ez_f=ez_f,
        step=state.step + 1,
    )
    state = _apply_fine_pec_axes(state, "xy")
    return zero_coarse_hole(state, config)


def step_disjoint_z_slab_metric_curl_sat_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Interface-distance-aware z-curl step followed by z-face Maxwell SAT."""
    state = step_disjoint_z_slab_metric_curl_3d(state, config)
    state = apply_z_faces_maxwell_sat(state, config)
    state = _apply_fine_pec_axes(state, "xy")
    return zero_coarse_hole(state, config)


def step_disjoint_z_slab_metric_curl_upwind_pair_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Metric z-curl step followed by impedance-upwind pair z-face SAT."""
    state = step_disjoint_z_slab_metric_curl_3d(state, config)
    state = apply_z_faces_upwind_pair_sat(state, config)
    state = _apply_fine_pec_axes(state, "xy")
    return zero_coarse_hole(state, config)


def step_disjoint_z_slab_metric_curl_split_upwind_pair_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Metric z-curl step with leapfrog-split impedance-upwind pair SAT."""
    state = zero_coarse_hole(state, config)

    hx_c, hy_c, hz_c = _coarse_h_step_z_metric(state, config)
    hx_f, hy_f, hz_f = _fine_h_step_z_metric(state, config)
    state = state._replace(
        hx_c=hx_c, hy_c=hy_c, hz_c=hz_c,
        hx_f=hx_f, hy_f=hy_f, hz_f=hz_f,
    )
    state = apply_z_faces_upwind_pair_h_sat(state, config)
    state = zero_coarse_hole(state, config)

    ex_c, ey_c, ez_c = _coarse_e_step_z_metric(state, config)
    ex_f, ey_f, ez_f = _fine_e_step_z_metric(state, config)
    state = state._replace(
        ex_c=ex_c, ey_c=ey_c, ez_c=ez_c,
        ex_f=ex_f, ey_f=ey_f, ez_f=ez_f,
        step=state.step + 1,
    )
    state = _apply_fine_pec_axes(state, "xy")
    state = apply_z_faces_upwind_pair_e_sat(state, config)
    state = _apply_fine_pec_axes(state, "xy")
    return zero_coarse_hole(state, config)


def apply_xlo_maxwell_sat(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
    *,
    strength: float | None = None,
) -> DisjointSubgridState3D:
    """Apply Maxwell cross-coupled SAT on the x-lo face.

    This is the first Stage-2 physics-coupling slice.  It couples the
    active coarse cell immediately left of the hole (`i = fi_lo - 1`) to
    the fine block's left boundary (`i = 0`).  The update is
    characteristic-scaled and cross-coupled:

    - tangential E receives tangential-H mismatch,
    - tangential H receives tangential-E mismatch.

    The signs follow the x-normal curl pairing (`Ey` with `Hz`, `Ez`
    with `Hy`).  This normalized prototype is intentionally conservative:
    it validates topology, field ownership, projection compatibility, and
    signal transfer before the later slice replaces the dimensionless
    strength with the full SBP-SAT coefficient derivation.
    """
    return apply_face_maxwell_sat(state, config, "x_lo", strength=strength)


def step_disjoint_xlo_sat_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Topology step plus prototype x-lo Maxwell SAT coupling."""
    state = step_disjoint_topology_3d(state, config)
    state = apply_xlo_maxwell_sat(state, config)
    return zero_coarse_hole(state, config)


def step_disjoint_sat_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Topology step plus six-face Maxwell SAT coupling."""
    state = step_disjoint_topology_3d(state, config)
    state = apply_all_faces_maxwell_sat(state, config)
    return zero_coarse_hole(state, config)


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


def step_disjoint_z_slab_sat_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Topology step plus z-face SAT for full-x/y centered slabs.

    The fine block spans physical x/y PEC faces and has artificial interfaces
    only along z.  This is a research-integration stepping primitive for the
    public centered z-slab disjoint lane; it is not yet a production gate.
    """
    state = step_disjoint_topology_3d(state, config)
    state = _apply_fine_pec_axes(state, "xy")
    state = apply_z_faces_maxwell_sat(state, config)
    state = _apply_fine_pec_axes(state, "xy")
    return zero_coarse_hole(state, config)


def step_disjoint_z_slab_leapfrog_sat_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Leapfrog-ordered z-face SAT for full-x/y centered slabs.

    The earlier research smoke step applies E and H SAT together after a full
    uncoupled Yee step.  This variant respects the Yee time staggering more
    closely: H receives the tangential-E mismatch after the H update and before
    the E update; E then receives the tangential-H mismatch after the E update.
    It remains a research prototype until waveform and crossval gates pass.
    """
    state = zero_coarse_hole(state, config)
    hx_c, hy_c, hz_c = _yee_h_step(
        state.ex_c, state.ey_c, state.ez_c,
        state.hx_c, state.hy_c, state.hz_c,
        config.dt, config.dx_c,
    )
    hx_f, hy_f, hz_f = _yee_h_step(
        state.ex_f, state.ey_f, state.ez_f,
        state.hx_f, state.hy_f, state.hz_f,
        config.dt, config.dx_f,
    )
    state = state._replace(
        hx_c=hx_c, hy_c=hy_c, hz_c=hz_c,
        hx_f=hx_f, hy_f=hy_f, hz_f=hz_f,
    )
    state = apply_z_faces_maxwell_h_sat(state, config)
    state = zero_coarse_hole(state, config)

    ex_c, ey_c, ez_c = _yee_e_step(
        state.ex_c, state.ey_c, state.ez_c,
        state.hx_c, state.hy_c, state.hz_c,
        config.dt, config.dx_c,
        apply_outer_pec=True,
    )
    ex_f, ey_f, ez_f = _yee_e_step(
        state.ex_f, state.ey_f, state.ez_f,
        state.hx_f, state.hy_f, state.hz_f,
        config.dt, config.dx_f,
        apply_outer_pec=False,
    )
    state = state._replace(
        ex_c=ex_c, ey_c=ey_c, ez_c=ez_c,
        ex_f=ex_f, ey_f=ey_f, ez_f=ez_f,
        step=state.step + 1,
    )
    state = _apply_fine_pec_axes(state, "xy")
    state = apply_z_faces_maxwell_e_sat(state, config)
    state = _apply_fine_pec_axes(state, "xy")
    return zero_coarse_hole(state, config)


def step_disjoint_z_slab_upwind_pair_sat_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> DisjointSubgridState3D:
    """Topology step plus impedance-upwind pair SAT for centered z slabs."""
    state = step_disjoint_topology_3d(state, config)
    state = _apply_fine_pec_axes(state, "xy")
    state = apply_z_faces_upwind_pair_sat(state, config)
    state = _apply_fine_pec_axes(state, "xy")
    return zero_coarse_hole(state, config)


def compute_disjoint_energy_3d(
    state: DisjointSubgridState3D,
    config: DisjointSubgridConfig3D,
) -> float:
    """Return disjoint coarse-exterior + fine-volume electromagnetic energy."""
    m = config.coarse_active_mask
    e_c = jnp.where(
        m,
        state.ex_c ** 2 + state.ey_c ** 2 + state.ez_c ** 2,
        0.0,
    )
    h_c = jnp.where(
        m,
        state.hx_c ** 2 + state.hy_c ** 2 + state.hz_c ** 2,
        0.0,
    )
    e_f = state.ex_f ** 2 + state.ey_f ** 2 + state.ez_f ** 2
    h_f = state.hx_f ** 2 + state.hy_f ** 2 + state.hz_f ** 2
    return float(
        EPS_0 * config.dx_c ** 3 * jnp.sum(e_c)
        + MU_0 * config.dx_c ** 3 * jnp.sum(h_c)
        + EPS_0 * config.dx_f ** 3 * jnp.sum(e_f)
        + MU_0 * config.dx_f ** 3 * jnp.sum(h_f)
    )
