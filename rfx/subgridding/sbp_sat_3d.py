"""Phase-1 3D SBP-SAT FDTD subgridding.

This lane is intentionally narrow:

- z-slab only
- one canonical stepper
- norm-compatible z-face operators

The coarse and fine grids use the same global timestep (limited by the
fine-grid CFL condition).  Coupling is applied only on the `z_lo` and
`z_hi` faces and only to tangential `(Ex, Ey)` and `(Hx, Hy)` traces.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0
from rfx.subgridding.face_ops import (
    ZFaceOps,
    build_zface_ops,
    prolong_zface,
    restrict_zface,
)

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)
PHASE1_3D_CFL = 0.99


class SubgridConfig3D(NamedTuple):
    """Configuration for the canonical Phase-1 z-slab lane."""

    nx_c: int
    ny_c: int
    nz_c: int
    dx_c: float
    fi_lo: int
    fi_hi: int
    fj_lo: int
    fj_hi: int
    fk_lo: int
    fk_hi: int
    nx_f: int
    ny_f: int
    nz_f: int
    dx_f: float
    dt: float
    ratio: int
    tau: float
    face_ops: ZFaceOps | None = None


class SubgridState3D(NamedTuple):
    """Field state for the canonical Phase-1 z-slab lane."""

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


def _region_shape(config: SubgridConfig3D) -> tuple[int, int, int]:
    return (
        config.fi_hi - config.fi_lo,
        config.fj_hi - config.fj_lo,
        config.fk_hi - config.fk_lo,
    )


def _get_face_ops(config: SubgridConfig3D) -> ZFaceOps:
    ni, nj, nk = _region_shape(config)
    if nk <= 0:
        raise ValueError(f"Invalid z slab thickness: fk_lo={config.fk_lo}, fk_hi={config.fk_hi}")
    if config.face_ops is not None:
        return config.face_ops
    return build_zface_ops((ni, nj), config.ratio, config.dx_c)


def validate_subgrid_config_3d(config: SubgridConfig3D) -> None:
    """Validate the canonical Phase-1 z-slab config invariants.

    This validator protects direct ``SubgridConfig3D`` / JIT entrypoints from
    silently widening Phase 1 back into partial-x/y or arbitrary-box support.
    """

    if config.ratio <= 1:
        raise ValueError(f"Phase-1 SBP-SAT ratio must be > 1, got {config.ratio}")
    if config.dx_c <= 0 or config.dx_f <= 0:
        raise ValueError("Phase-1 SBP-SAT spacings dx_c and dx_f must be positive")
    expected_dx_f = config.dx_c / config.ratio
    if not np.isclose(config.dx_f, expected_dx_f):
        raise ValueError(
            "Phase-1 SBP-SAT dx_f must equal dx_c / ratio; "
            f"got dx_c={config.dx_c}, ratio={config.ratio}, dx_f={config.dx_f}"
        )
    expected_dt = phase1_3d_dt(config.dx_f)
    if not np.isclose(config.dt, expected_dt):
        raise ValueError(
            "Phase-1 SBP-SAT dt must follow the canonical near-fine-CFL rule; "
            f"got dt={config.dt}, expected {expected_dt}"
        )
    if (config.fi_lo, config.fi_hi) != (0, config.nx_c):
        raise ValueError(
            "Phase-1 SBP-SAT supports full-span x only; "
            f"got fi_lo={config.fi_lo}, fi_hi={config.fi_hi}, nx_c={config.nx_c}"
        )
    if (config.fj_lo, config.fj_hi) != (0, config.ny_c):
        raise ValueError(
            "Phase-1 SBP-SAT supports full-span y only; "
            f"got fj_lo={config.fj_lo}, fj_hi={config.fj_hi}, ny_c={config.ny_c}"
        )
    if not (0 <= config.fk_lo < config.fk_hi <= config.nz_c):
        raise ValueError(
            "Phase-1 SBP-SAT requires a non-empty z slab inside the coarse grid; "
            f"got fk_lo={config.fk_lo}, fk_hi={config.fk_hi}, nz_c={config.nz_c}"
        )

    ni, nj, nk = _region_shape(config)
    expected_fine = (ni * config.ratio, nj * config.ratio, nk * config.ratio)
    actual_fine = (config.nx_f, config.ny_f, config.nz_f)
    if actual_fine != expected_fine:
        raise ValueError(
            "Phase-1 SBP-SAT fine shape must match coarse slab shape times ratio; "
            f"got {actual_fine}, expected {expected_fine}"
        )
    if config.face_ops is not None:
        if config.face_ops.coarse_shape != (ni, nj):
            raise ValueError(
                "Phase-1 SBP-SAT face_ops coarse shape mismatch; "
                f"got {config.face_ops.coarse_shape}, expected {(ni, nj)}"
            )
        if config.face_ops.fine_shape != (config.nx_f, config.ny_f):
            raise ValueError(
                "Phase-1 SBP-SAT face_ops fine shape mismatch; "
                f"got {config.face_ops.fine_shape}, expected {(config.nx_f, config.ny_f)}"
            )
        if config.face_ops.ratio != config.ratio:
            raise ValueError(
                "Phase-1 SBP-SAT face_ops ratio mismatch; "
                f"got {config.face_ops.ratio}, expected {config.ratio}"
            )


def phase1_3d_dt(dx_f: float) -> float:
    """Return the canonical Phase-1 3D timestep for fine spacing ``dx_f``."""

    return PHASE1_3D_CFL * dx_f / (C0 * np.sqrt(3.0))


def _default_fine_region(shape_c: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
    nx_c, ny_c, nz_c = shape_c
    fk_lo = max(1, nz_c // 3)
    fk_hi = min(nz_c - 1, max(fk_lo + 1, 2 * nz_c // 3))
    return (0, nx_c, 0, ny_c, fk_lo, fk_hi)


def _validate_phase1_fine_region(
    shape_c: tuple[int, int, int],
    fine_region: tuple[int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    nx_c, ny_c, nz_c = shape_c
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = fine_region
    if (fi_lo, fi_hi) != (0, nx_c) or (fj_lo, fj_hi) != (0, ny_c):
        raise ValueError(
            "Phase-1 init_subgrid_3d supports full-span x/y only; "
            f"got fine_region={fine_region}"
        )
    if not (0 <= fk_lo < fk_hi <= nz_c):
        raise ValueError(
            f"fine_region={fine_region} must satisfy 0 <= fk_lo < fk_hi <= {nz_c}"
        )
    return fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi


def init_subgrid_3d(
    shape_c: tuple[int, int, int] = (40, 40, 40),
    dx_c: float = 0.003,
    fine_region: tuple[int, int, int, int, int, int] | None = None,
    ratio: int = 3,
    courant: float | None = None,
    tau: float = 0.5,
) -> tuple[SubgridConfig3D, SubgridState3D]:
    """Initialize the canonical Phase-1 z-slab subgrid state.

    ``fine_region`` remains a legacy internal tuple, but Phase-1 only
    supports full-span ``x/y`` plus a refined ``z`` slab. ``courant`` is
    accepted as a compatibility alias for the canonical near-fine-CFL
    rule and rejects any non-canonical value.
    """

    nx_c, ny_c, nz_c = shape_c
    if ratio <= 1:
        raise ValueError(f"ratio must be > 1, got {ratio}")
    if fine_region is None:
        fine_region = _default_fine_region(shape_c)
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = _validate_phase1_fine_region(
        shape_c,
        fine_region,
    )
    dx_f = dx_c / ratio
    if courant is not None and not np.isclose(courant, PHASE1_3D_CFL):
        raise ValueError(
            "Phase-1 init_subgrid_3d uses the canonical near-fine-CFL rule "
            f"only ({PHASE1_3D_CFL}); got courant={courant}"
        )
    dt = phase1_3d_dt(dx_f)

    nx_f = (fi_hi - fi_lo) * ratio
    ny_f = (fj_hi - fj_lo) * ratio
    nz_f = (fk_hi - fk_lo) * ratio

    config = SubgridConfig3D(
        nx_c=nx_c,
        ny_c=ny_c,
        nz_c=nz_c,
        dx_c=dx_c,
        fi_lo=fi_lo,
        fi_hi=fi_hi,
        fj_lo=fj_lo,
        fj_hi=fj_hi,
        fk_lo=fk_lo,
        fk_hi=fk_hi,
        nx_f=nx_f,
        ny_f=ny_f,
        nz_f=nz_f,
        dx_f=dx_f,
        dt=float(dt),
        ratio=ratio,
        tau=tau,
        face_ops=build_zface_ops((fi_hi - fi_lo, fj_hi - fj_lo), ratio, dx_c),
    )
    validate_subgrid_config_3d(config)

    z = lambda s: jnp.zeros(s, dtype=jnp.float32)
    state = SubgridState3D(
        ex_c=z(shape_c),
        ey_c=z(shape_c),
        ez_c=z(shape_c),
        hx_c=z(shape_c),
        hy_c=z(shape_c),
        hz_c=z(shape_c),
        ex_f=z((nx_f, ny_f, nz_f)),
        ey_f=z((nx_f, ny_f, nz_f)),
        ez_f=z((nx_f, ny_f, nz_f)),
        hx_f=z((nx_f, ny_f, nz_f)),
        hy_f=z((nx_f, ny_f, nz_f)),
        hz_f=z((nx_f, ny_f, nz_f)),
        step=0,
    )
    return config, state


def _make_mats(shape):
    from rfx.core.yee import MaterialArrays

    return MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
    )


def _update_h_only(ex, ey, ez, hx, hy, hz, dt, dx, mats=None):
    from rfx.core.yee import FDTDState, update_h

    shape = ex.shape
    state = FDTDState(
        ex=ex,
        ey=ey,
        ez=ez,
        hx=hx,
        hy=hy,
        hz=hz,
        step=jnp.array(0, dtype=jnp.int32),
    )
    if mats is None:
        mats = _make_mats(shape)
    state = update_h(state, mats, dt, dx)
    return state.hx, state.hy, state.hz


def _update_e_only(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    dt,
    dx,
    mats=None,
    pec_mask=None,
    boundary_axes: str | None = "xyz",
):
    from rfx.boundaries.pec import apply_pec, apply_pec_mask
    from rfx.core.yee import FDTDState, update_e

    shape = ex.shape
    state = FDTDState(
        ex=ex,
        ey=ey,
        ez=ez,
        hx=hx,
        hy=hy,
        hz=hz,
        step=jnp.array(0, dtype=jnp.int32),
    )
    if mats is None:
        mats = _make_mats(shape)
    state = update_e(state, mats, dt, dx)
    if boundary_axes:
        state = apply_pec(state, axes=boundary_axes)
    if pec_mask is not None:
        state = apply_pec_mask(state, pec_mask)
    return state.ex, state.ey, state.ez


def sat_penalty_coefficients(ratio: int, tau: float) -> tuple[float, float]:
    """Return the canonical SAT penalty coefficients."""

    alpha_f = tau * ratio / (ratio + 1.0)
    alpha_c = tau * 1.0 / (ratio + 1.0)
    return float(alpha_c), float(alpha_f)


def _coarse_zface_slice(config: SubgridConfig3D, face: str) -> tuple[slice, slice, int]:
    k = config.fk_lo if face == "z_lo" else config.fk_hi - 1
    return (slice(config.fi_lo, config.fi_hi), slice(config.fj_lo, config.fj_hi), k)


def _fine_zface_slice(config: SubgridConfig3D, face: str) -> tuple[slice, slice, int]:
    k = 0 if face == "z_lo" else config.nz_f - 1
    return (slice(0, config.nx_f), slice(0, config.ny_f), k)


def extract_tangential_e_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract tangential E traces for one z face."""

    ex, ey, ez = fields
    del ez
    sl = _coarse_zface_slice(config, face) if grid == "coarse" else _fine_zface_slice(config, face)
    return ex[sl], ey[sl]


def extract_tangential_h_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract tangential H traces for one z face."""

    hx, hy, hz = fields
    del hz
    sl = _coarse_zface_slice(config, face) if grid == "coarse" else _fine_zface_slice(config, face)
    return hx[sl], hy[sl]


def scatter_tangential_e_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tangential: tuple[jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scatter tangential E traces back to one z face."""

    ex, ey, ez = fields
    ex_face, ey_face = tangential
    sl = _coarse_zface_slice(config, face) if grid == "coarse" else _fine_zface_slice(config, face)
    ex = ex.at[sl].set(ex_face)
    ey = ey.at[sl].set(ey_face)
    return ex, ey, ez


def scatter_tangential_h_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tangential: tuple[jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scatter tangential H traces back to one z face."""

    hx, hy, hz = fields
    hx_face, hy_face = tangential
    sl = _coarse_zface_slice(config, face) if grid == "coarse" else _fine_zface_slice(config, face)
    hx = hx.at[sl].set(hx_face)
    hy = hy.at[sl].set(hy_face)
    return hx, hy, hz


def _apply_sat_pair(
    coarse_face: jnp.ndarray,
    fine_face: jnp.ndarray,
    ops: ZFaceOps,
    alpha_c: float,
    alpha_f: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coarse_mismatch = restrict_zface(fine_face, ops) - coarse_face
    fine_mismatch = prolong_zface(coarse_face, ops) - fine_face
    return (
        coarse_face + alpha_c * coarse_mismatch,
        fine_face + alpha_f * fine_mismatch,
    )


def _zero_coarse_overlap_interior(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Suppress coarse-grid interior state inside the fine slab volume.

    The coarse field is only authoritative outside the refined slab and on
    the z-interface traces.  Interior coarse values inside the slab act as a
    redundant hidden dynamics path, so we zero the strict interior
    ``fk_lo+1:fk_hi-1`` region.
    """

    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni, nj, nk = _region_shape(config)
    if nk <= 2:
        return fields
    interior = (slice(fi, fi + ni), slice(fj, fj + nj), slice(fk + 1, fk + nk - 1))
    return tuple(field.at[interior].set(0.0) for field in fields)


def apply_sat_h_zfaces(
    coarse_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fine_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Apply SAT coupling to tangential H traces on both z faces."""

    ops = _get_face_ops(config)
    alpha_c, alpha_f = sat_penalty_coefficients(config.ratio, config.tau)
    coarse = coarse_fields
    fine = fine_fields
    for face in ("z_lo", "z_hi"):
        hx_c_face, hy_c_face = extract_tangential_h_face(coarse, config, face, grid="coarse")
        hx_f_face, hy_f_face = extract_tangential_h_face(fine, config, face, grid="fine")
        hx_c_face, hx_f_face = _apply_sat_pair(hx_c_face, hx_f_face, ops, alpha_c, alpha_f)
        hy_c_face, hy_f_face = _apply_sat_pair(hy_c_face, hy_f_face, ops, alpha_c, alpha_f)
        coarse = scatter_tangential_h_face(coarse, (hx_c_face, hy_c_face), config, face, grid="coarse")
        fine = scatter_tangential_h_face(fine, (hx_f_face, hy_f_face), config, face, grid="fine")
    return coarse, fine


def apply_sat_e_zfaces(
    coarse_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fine_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Apply SAT coupling to tangential E traces on both z faces."""

    ops = _get_face_ops(config)
    alpha_c, alpha_f = sat_penalty_coefficients(config.ratio, config.tau)
    coarse = coarse_fields
    fine = fine_fields
    for face in ("z_lo", "z_hi"):
        ex_c_face, ey_c_face = extract_tangential_e_face(coarse, config, face, grid="coarse")
        ex_f_face, ey_f_face = extract_tangential_e_face(fine, config, face, grid="fine")
        ex_c_face, ex_f_face = _apply_sat_pair(ex_c_face, ex_f_face, ops, alpha_c, alpha_f)
        ey_c_face, ey_f_face = _apply_sat_pair(ey_c_face, ey_f_face, ops, alpha_c, alpha_f)
        coarse = scatter_tangential_e_face(coarse, (ex_c_face, ey_c_face), config, face, grid="coarse")
        fine = scatter_tangential_e_face(fine, (ex_f_face, ey_f_face), config, face, grid="fine")
    return coarse, fine


def _shared_node_coupling_h_3d(state_c_fields, state_f_fields, config):
    """Compatibility wrapper for legacy imports.

    The canonical Phase-1 lane uses z-face-only H SAT coupling.
    """

    return apply_sat_h_zfaces(state_c_fields, state_f_fields, config)


def _shared_node_coupling_3d(state_c_fields, state_f_fields, config):
    """Compatibility wrapper for legacy imports.

    The canonical Phase-1 lane uses z-face-only E SAT coupling.
    """

    return apply_sat_e_zfaces(state_c_fields, state_f_fields, config)


def step_subgrid_3d(
    state: SubgridState3D,
    config: SubgridConfig3D,
    *,
    mats_c=None,
    mats_f=None,
    pec_mask_c=None,
    pec_mask_f=None,
) -> SubgridState3D:
    """Advance the canonical Phase-1 z-slab lane by one timestep."""

    hx_c, hy_c, hz_c = _update_h_only(
        state.ex_c,
        state.ey_c,
        state.ez_c,
        state.hx_c,
        state.hy_c,
        state.hz_c,
        config.dt,
        config.dx_c,
        mats=mats_c,
    )
    hx_f, hy_f, hz_f = _update_h_only(
        state.ex_f,
        state.ey_f,
        state.ez_f,
        state.hx_f,
        state.hy_f,
        state.hz_f,
        config.dt,
        config.dx_f,
        mats=mats_f,
    )
    (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f) = apply_sat_h_zfaces(
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        config,
    )
    hx_c, hy_c, hz_c = _zero_coarse_overlap_interior((hx_c, hy_c, hz_c), config)

    ex_c, ey_c, ez_c = _update_e_only(
        state.ex_c,
        state.ey_c,
        state.ez_c,
        hx_c,
        hy_c,
        hz_c,
        config.dt,
        config.dx_c,
        mats=mats_c,
        pec_mask=pec_mask_c,
        boundary_axes="xyz",
    )
    ex_f, ey_f, ez_f = _update_e_only(
        state.ex_f,
        state.ey_f,
        state.ez_f,
        hx_f,
        hy_f,
        hz_f,
        config.dt,
        config.dx_f,
        mats=mats_f,
        pec_mask=pec_mask_f,
        boundary_axes="xy",
    )
    (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f) = apply_sat_e_zfaces(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        config,
    )
    ex_c, ey_c, ez_c = _zero_coarse_overlap_interior((ex_c, ey_c, ez_c), config)

    return SubgridState3D(
        ex_c=ex_c,
        ey_c=ey_c,
        ez_c=ez_c,
        hx_c=hx_c,
        hy_c=hy_c,
        hz_c=hz_c,
        ex_f=ex_f,
        ey_f=ey_f,
        ez_f=ez_f,
        hx_f=hx_f,
        hy_f=hy_f,
        hz_f=hz_f,
        step=state.step + 1,
    )


def compute_energy_3d(state: SubgridState3D, config: SubgridConfig3D) -> float:
    """Total energy with coarse/fine overlap counted once."""

    dv_c = config.dx_c ** 3
    dv_f = config.dx_f ** 3
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni, nj, nk = _region_shape(config)

    mask = jnp.ones(state.ex_c.shape, dtype=jnp.bool_)
    mask = mask.at[fi : fi + ni, fj : fj + nj, fk : fk + nk].set(False)

    e_c = (
        float(jnp.sum(jnp.where(mask, state.ex_c**2 + state.ey_c**2 + state.ez_c**2, 0.0)))
        * EPS_0
        * dv_c
        + float(jnp.sum(jnp.where(mask, state.hx_c**2 + state.hy_c**2 + state.hz_c**2, 0.0)))
        * MU_0
        * dv_c
    )
    e_f = (
        float(jnp.sum(state.ex_f**2 + state.ey_f**2 + state.ez_f**2)) * EPS_0 * dv_f
        + float(jnp.sum(state.hx_f**2 + state.hy_f**2 + state.hz_f**2)) * MU_0 * dv_f
    )
    return e_c + e_f
