"""3D SBP-SAT FDTD subgridding.

The current implementation keeps the same global timestep on the coarse and
fine grids (limited by the fine-grid CFL condition) and uses explicit
restriction/prolongation operators on coarse/fine interface traces.

The original shipped lane was a full-span x/y z slab.  This module now
supports an arbitrary all-PEC refinement box in the low-level runtime while the
higher-level support surface remains governed by the support matrix and docs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0
from rfx.subgridding.face_ops import (
    EdgeOps,
    ZFaceOps,
    build_edge_ops,
    build_face_ops,
    build_zface_ops,
    prolong_edge,
    prolong_face,
    restrict_edge,
    restrict_face,
)
from rfx.subgridding.sbp_operators import (
    build_tensor_face_mortar,
    operator_projected_skew_eh_sat_face,
)

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)
PHASE1_3D_CFL = 0.99
_TIME_CENTERED_HELPER_RELAXATION = 0.02
_OBSERVABLE_PROXY_MODAL_RETRY_RELAXATION = 0.02
_PROPAGATION_AWARE_MODAL_RETRY_RELAXATION = 0.01


class SubgridConfig3D(NamedTuple):
    """Configuration for the current all-PEC subgrid lane."""

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


class FaceOrientation(NamedTuple):
    """Face-orientation metadata for box coupling."""

    face: str
    normal_axis: int
    normal_sign: int
    tangential_axes: tuple[int, int]
    tangential_e_components: tuple[str, str]
    tangential_h_components: tuple[str, str]


class EdgeOrientation(NamedTuple):
    """Edge-orientation metadata for box coupling."""

    edge: str
    varying_axis: int
    fixed_faces: tuple[str, str]
    e_component: str
    h_component: str


class CornerOrientation(NamedTuple):
    """Corner-orientation metadata for box coupling."""

    corner: str
    fixed_faces: tuple[str, str, str]


class _PrivateInterfaceOwnerState(NamedTuple):
    """Private per-interface owner state carried through solver internals."""

    face_phase_reference: jnp.ndarray
    face_magnitude_reference: jnp.ndarray
    face_update_count: jnp.ndarray
    face_proxy_reference_real: jnp.ndarray
    face_proxy_reference_imag: jnp.ndarray
    face_proxy_reference_prev_real: jnp.ndarray
    face_proxy_reference_prev_imag: jnp.ndarray
    face_proxy_weight: jnp.ndarray
    face_proxy_mask: jnp.ndarray
    face_packet_offsets: jnp.ndarray
    face_packet_lengths: jnp.ndarray
    face_normal_axis: jnp.ndarray
    face_normal_sign: jnp.ndarray
    face_tangential_axis_0: jnp.ndarray
    face_tangential_axis_1: jnp.ndarray
    source_owner_reference_real: jnp.ndarray
    source_owner_reference_imag: jnp.ndarray
    source_owner_reference_prev_real: jnp.ndarray
    source_owner_reference_prev_imag: jnp.ndarray
    source_owner_weight: jnp.ndarray
    source_owner_mask: jnp.ndarray
    source_incident_normalizer_real: jnp.ndarray
    source_incident_normalizer_imag: jnp.ndarray
    source_packet_offsets: jnp.ndarray
    source_packet_lengths: jnp.ndarray
    source_normal_axis: jnp.ndarray
    source_normal_sign: jnp.ndarray
    source_tangential_axis_0: jnp.ndarray
    source_tangential_axis_1: jnp.ndarray


class _PrivateInterfaceOwnerJointScore(NamedTuple):
    """Private owner-backed transverse phase/magnitude score."""

    transverse_magnitude_cv: jnp.ndarray
    transverse_phase_spread_deg: jnp.ndarray
    usable_face_count: jnp.ndarray


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
    private_interface_owner_state: _PrivateInterfaceOwnerState | None = None


FACE_ORIENTATIONS: dict[str, FaceOrientation] = {
    "x_lo": FaceOrientation("x_lo", 0, -1, (1, 2), ("ey", "ez"), ("hy", "hz")),
    "x_hi": FaceOrientation("x_hi", 0, +1, (1, 2), ("ey", "ez"), ("hy", "hz")),
    "y_lo": FaceOrientation("y_lo", 1, -1, (0, 2), ("ex", "ez"), ("hx", "hz")),
    "y_hi": FaceOrientation("y_hi", 1, +1, (0, 2), ("ex", "ez"), ("hx", "hz")),
    "z_lo": FaceOrientation("z_lo", 2, -1, (0, 1), ("ex", "ey"), ("hx", "hy")),
    "z_hi": FaceOrientation("z_hi", 2, +1, (0, 1), ("ex", "ey"), ("hx", "hy")),
}

EDGE_ORIENTATIONS: dict[str, EdgeOrientation] = {
    "x_lo_y_lo": EdgeOrientation("x_lo_y_lo", 2, ("x_lo", "y_lo"), "ez", "hz"),
    "x_lo_y_hi": EdgeOrientation("x_lo_y_hi", 2, ("x_lo", "y_hi"), "ez", "hz"),
    "x_hi_y_lo": EdgeOrientation("x_hi_y_lo", 2, ("x_hi", "y_lo"), "ez", "hz"),
    "x_hi_y_hi": EdgeOrientation("x_hi_y_hi", 2, ("x_hi", "y_hi"), "ez", "hz"),
    "x_lo_z_lo": EdgeOrientation("x_lo_z_lo", 1, ("x_lo", "z_lo"), "ey", "hy"),
    "x_lo_z_hi": EdgeOrientation("x_lo_z_hi", 1, ("x_lo", "z_hi"), "ey", "hy"),
    "x_hi_z_lo": EdgeOrientation("x_hi_z_lo", 1, ("x_hi", "z_lo"), "ey", "hy"),
    "x_hi_z_hi": EdgeOrientation("x_hi_z_hi", 1, ("x_hi", "z_hi"), "ey", "hy"),
    "y_lo_z_lo": EdgeOrientation("y_lo_z_lo", 0, ("y_lo", "z_lo"), "ex", "hx"),
    "y_lo_z_hi": EdgeOrientation("y_lo_z_hi", 0, ("y_lo", "z_hi"), "ex", "hx"),
    "y_hi_z_lo": EdgeOrientation("y_hi_z_lo", 0, ("y_hi", "z_lo"), "ex", "hx"),
    "y_hi_z_hi": EdgeOrientation("y_hi_z_hi", 0, ("y_hi", "z_hi"), "ex", "hx"),
}

CORNER_ORIENTATIONS: dict[str, CornerOrientation] = {
    "x_lo_y_lo_z_lo": CornerOrientation("x_lo_y_lo_z_lo", ("x_lo", "y_lo", "z_lo")),
    "x_lo_y_lo_z_hi": CornerOrientation("x_lo_y_lo_z_hi", ("x_lo", "y_lo", "z_hi")),
    "x_lo_y_hi_z_lo": CornerOrientation("x_lo_y_hi_z_lo", ("x_lo", "y_hi", "z_lo")),
    "x_lo_y_hi_z_hi": CornerOrientation("x_lo_y_hi_z_hi", ("x_lo", "y_hi", "z_hi")),
    "x_hi_y_lo_z_lo": CornerOrientation("x_hi_y_lo_z_lo", ("x_hi", "y_lo", "z_lo")),
    "x_hi_y_lo_z_hi": CornerOrientation("x_hi_y_lo_z_hi", ("x_hi", "y_lo", "z_hi")),
    "x_hi_y_hi_z_lo": CornerOrientation("x_hi_y_hi_z_lo", ("x_hi", "y_hi", "z_lo")),
    "x_hi_y_hi_z_hi": CornerOrientation("x_hi_y_hi_z_hi", ("x_hi", "y_hi", "z_hi")),
}


def _region_shape(config: SubgridConfig3D) -> tuple[int, int, int]:
    return (
        config.fi_hi - config.fi_lo,
        config.fj_hi - config.fj_lo,
        config.fk_hi - config.fk_lo,
    )


def _coarse_box_shape(config: SubgridConfig3D) -> tuple[int, int, int]:
    return _region_shape(config)


def _face_coarse_shape(config: SubgridConfig3D, face: str) -> tuple[int, int]:
    ni, nj, nk = _coarse_box_shape(config)
    axis = FACE_ORIENTATIONS[face].normal_axis
    if axis == 0:
        return (nj, nk)
    if axis == 1:
        return (ni, nk)
    return (ni, nj)


def _get_face_ops(config: SubgridConfig3D, face: str) -> ZFaceOps:
    coarse_shape = _face_coarse_shape(config, face)
    if coarse_shape[0] <= 0 or coarse_shape[1] <= 0:
        raise ValueError(f"Invalid face shape for {face}: coarse_shape={coarse_shape}")
    if face.startswith("z") and config.face_ops is not None:
        if (
            config.face_ops.coarse_shape == coarse_shape
            and config.face_ops.ratio == config.ratio
        ):
            return config.face_ops
    return build_face_ops(coarse_shape, config.ratio, config.dx_c)


def _get_edge_ops(config: SubgridConfig3D, edge: str) -> EdgeOps:
    ni, nj, nk = _coarse_box_shape(config)
    axis = EDGE_ORIENTATIONS[edge].varying_axis
    coarse_size = (ni, nj, nk)[axis]
    if coarse_size <= 0:
        raise ValueError(f"Invalid edge size for {edge}: {coarse_size}")
    return build_edge_ops(coarse_size, config.ratio, config.dx_c)


def validate_subgrid_config_3d(config: SubgridConfig3D) -> None:
    """Validate the current all-PEC box config invariants."""

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
    if not (0 <= config.fi_lo < config.fi_hi <= config.nx_c):
        raise ValueError(
            "SBP-SAT requires a non-empty x box inside the coarse grid; "
            f"got fi_lo={config.fi_lo}, fi_hi={config.fi_hi}, nx_c={config.nx_c}"
        )
    if not (0 <= config.fj_lo < config.fj_hi <= config.ny_c):
        raise ValueError(
            "SBP-SAT requires a non-empty y box inside the coarse grid; "
            f"got fj_lo={config.fj_lo}, fj_hi={config.fj_hi}, ny_c={config.ny_c}"
        )
    if not (0 <= config.fk_lo < config.fk_hi <= config.nz_c):
        raise ValueError(
            "SBP-SAT requires a non-empty z box inside the coarse grid; "
            f"got fk_lo={config.fk_lo}, fk_hi={config.fk_hi}, nz_c={config.nz_c}"
        )

    ni, nj, nk = _region_shape(config)
    expected_fine = (ni * config.ratio, nj * config.ratio, nk * config.ratio)
    actual_fine = (config.nx_f, config.ny_f, config.nz_f)
    if actual_fine != expected_fine:
        raise ValueError(
            "SBP-SAT fine shape must match coarse box shape times ratio; "
            f"got {actual_fine}, expected {expected_fine}"
        )
    if config.face_ops is not None:
        if config.face_ops.coarse_shape != (ni, nj):
            raise ValueError(
                "SBP-SAT z-face_ops coarse shape mismatch; "
                f"got {config.face_ops.coarse_shape}, expected {(ni, nj)}"
            )
        if config.face_ops.fine_shape != (config.nx_f, config.ny_f):
            raise ValueError(
                "SBP-SAT z-face_ops fine shape mismatch; "
                f"got {config.face_ops.fine_shape}, expected {(config.nx_f, config.ny_f)}"
            )
        if config.face_ops.ratio != config.ratio:
            raise ValueError(
                "SBP-SAT z-face_ops ratio mismatch; "
                f"got {config.face_ops.ratio}, expected {config.ratio}"
            )


def phase1_3d_dt(dx_f: float) -> float:
    """Return the canonical Phase-1 3D timestep for fine spacing ``dx_f``."""

    return PHASE1_3D_CFL * dx_f / (C0 * np.sqrt(3.0))


def _default_fine_region(
    shape_c: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
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
    if not (0 <= fi_lo < fi_hi <= nx_c):
        raise ValueError(f"fine_region={fine_region} has invalid x bounds")
    if not (0 <= fj_lo < fj_hi <= ny_c):
        raise ValueError(f"fine_region={fine_region} has invalid y bounds")
    if not (0 <= fk_lo < fk_hi <= nz_c):
        raise ValueError(f"fine_region={fine_region} has invalid z bounds")
    return fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi


def init_subgrid_3d(
    shape_c: tuple[int, int, int] = (40, 40, 40),
    dx_c: float = 0.003,
    fine_region: tuple[int, int, int, int, int, int] | None = None,
    ratio: int = 3,
    courant: float | None = None,
    tau: float = 0.5,
) -> tuple[SubgridConfig3D, SubgridState3D]:
    """Initialize the current all-PEC subgrid state.

    ``fine_region`` is given in coarse-grid index bounds
    ``(fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi)``. ``courant`` remains a
    compatibility alias for the canonical near-fine-CFL rule and rejects any
    non-canonical value.
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

    def z(s):
        return jnp.zeros(s, dtype=jnp.float32)

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
        private_interface_owner_state=_init_private_interface_owner_state(config),
    )
    return config, state


def _make_mats(shape):
    from rfx.core.yee import MaterialArrays

    return MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
    )


def _update_h_only(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    dt,
    dx,
    mats=None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    pmc_faces: frozenset[str] | None = None,
):
    from rfx.boundaries.pmc import apply_pmc_faces
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
    state = update_h(state, mats, dt, dx, periodic=periodic)
    if pmc_faces:
        state = apply_pmc_faces(state, pmc_faces)
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
    boundary_faces: frozenset[str] | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
):
    from rfx.boundaries.pec import apply_pec, apply_pec_faces, apply_pec_mask
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
    state = update_e(state, mats, dt, dx, periodic=periodic)
    if boundary_faces is not None:
        state = apply_pec_faces(state, boundary_faces)
    elif boundary_axes:
        state = apply_pec(state, axes=boundary_axes)
    if pec_mask is not None:
        state = apply_pec_mask(state, pec_mask)
    return state.ex, state.ey, state.ez


def sat_penalty_coefficients(ratio: int, tau: float) -> tuple[float, float]:
    """Return the canonical SAT penalty coefficients."""

    alpha_f = tau * ratio / (ratio + 1.0)
    alpha_c = tau * 1.0 / (ratio + 1.0)
    return float(alpha_c), float(alpha_f)


_COMPONENT_INDEX = {
    "ex": 0,
    "ey": 1,
    "ez": 2,
    "hx": 0,
    "hy": 1,
    "hz": 2,
}


def _face_is_internal(config: SubgridConfig3D, face: str) -> bool:
    if face == "x_lo":
        return config.fi_lo > 0
    if face == "x_hi":
        return config.fi_hi < config.nx_c
    if face == "y_lo":
        return config.fj_lo > 0
    if face == "y_hi":
        return config.fj_hi < config.ny_c
    if face == "z_lo":
        return config.fk_lo > 0
    if face == "z_hi":
        return config.fk_hi < config.nz_c
    raise ValueError(f"unknown face {face!r}")


def _active_faces(config: SubgridConfig3D) -> tuple[str, ...]:
    return tuple(face for face in FACE_ORIENTATIONS if _face_is_internal(config, face))


def _private_owner_face_packet_layout(
    config: SubgridConfig3D,
) -> tuple[np.ndarray, ...]:
    active_faces = _active_faces(config)
    lengths: list[int] = []
    masks: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    normal_axes: list[int] = []
    normal_signs: list[int] = []
    tangential_axis_0: list[int] = []
    tangential_axis_1: list[int] = []
    offset = 0
    offsets = []
    for face in active_faces:
        orientation = FACE_ORIENTATIONS[face]
        ops = _get_face_ops(config, face)
        coarse_mask, _ = _face_interior_masks(ops.coarse_shape, config.ratio)
        mask = np.asarray(coarse_mask, dtype=np.float32).reshape(-1)
        weight = (
            np.asarray(ops.coarse_norm, dtype=np.float32).reshape(-1) * mask
        )
        offsets.append(offset)
        lengths.append(int(mask.size))
        masks.append(mask)
        weights.append(weight)
        normal_axes.append(int(orientation.normal_axis))
        normal_signs.append(int(orientation.normal_sign))
        tangential_axis_0.append(int(orientation.tangential_axes[0]))
        tangential_axis_1.append(int(orientation.tangential_axes[1]))
        offset += int(mask.size)
    packet_size = offset
    if masks:
        packet_mask = np.concatenate(masks).astype(np.float32)
        packet_weight = np.concatenate(weights).astype(np.float32)
    else:
        packet_mask = np.zeros((0,), dtype=np.float32)
        packet_weight = np.zeros((0,), dtype=np.float32)
    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(lengths, dtype=np.int32),
        np.asarray(normal_axes, dtype=np.int32),
        np.asarray(normal_signs, dtype=np.int32),
        np.asarray(tangential_axis_0, dtype=np.int32),
        np.asarray(tangential_axis_1, dtype=np.int32),
        packet_weight,
        packet_mask,
        np.zeros((packet_size,), dtype=np.float32),
    )


def _init_private_interface_owner_state(
    config: SubgridConfig3D,
) -> _PrivateInterfaceOwnerState:
    face_count = len(_active_faces(config))
    (
        face_packet_offsets,
        face_packet_lengths,
        face_normal_axis,
        face_normal_sign,
        face_tangential_axis_0,
        face_tangential_axis_1,
        face_proxy_weight,
        face_proxy_mask,
        face_proxy_reference,
    ) = _private_owner_face_packet_layout(config)
    source_incident_normalizer_imag = np.zeros_like(face_proxy_reference)
    return _PrivateInterfaceOwnerState(
        face_phase_reference=jnp.zeros((face_count,), dtype=jnp.float32),
        face_magnitude_reference=jnp.zeros((face_count,), dtype=jnp.float32),
        face_update_count=jnp.zeros((face_count,), dtype=jnp.int32),
        face_proxy_reference_real=jnp.asarray(
            face_proxy_reference,
            dtype=jnp.float32,
        ),
        face_proxy_reference_imag=jnp.asarray(
            face_proxy_reference,
            dtype=jnp.float32,
        ),
        face_proxy_reference_prev_real=jnp.asarray(
            face_proxy_reference,
            dtype=jnp.float32,
        ),
        face_proxy_reference_prev_imag=jnp.asarray(
            face_proxy_reference,
            dtype=jnp.float32,
        ),
        face_proxy_weight=jnp.asarray(face_proxy_weight, dtype=jnp.float32),
        face_proxy_mask=jnp.asarray(face_proxy_mask, dtype=jnp.float32),
        face_packet_offsets=jnp.asarray(face_packet_offsets, dtype=jnp.int32),
        face_packet_lengths=jnp.asarray(face_packet_lengths, dtype=jnp.int32),
        face_normal_axis=jnp.asarray(face_normal_axis, dtype=jnp.int32),
        face_normal_sign=jnp.asarray(face_normal_sign, dtype=jnp.int32),
        face_tangential_axis_0=jnp.asarray(
            face_tangential_axis_0,
            dtype=jnp.int32,
        ),
        face_tangential_axis_1=jnp.asarray(
            face_tangential_axis_1,
            dtype=jnp.int32,
        ),
        source_owner_reference_real=jnp.asarray(
            face_proxy_reference,
            dtype=jnp.float32,
        ),
        source_owner_reference_imag=jnp.asarray(
            face_proxy_reference,
            dtype=jnp.float32,
        ),
        source_owner_reference_prev_real=jnp.asarray(
            face_proxy_reference,
            dtype=jnp.float32,
        ),
        source_owner_reference_prev_imag=jnp.asarray(
            face_proxy_reference,
            dtype=jnp.float32,
        ),
        source_owner_weight=jnp.asarray(face_proxy_weight, dtype=jnp.float32),
        source_owner_mask=jnp.asarray(face_proxy_mask, dtype=jnp.float32),
        source_incident_normalizer_real=jnp.asarray(
            face_proxy_mask,
            dtype=jnp.float32,
        ),
        source_incident_normalizer_imag=jnp.asarray(
            source_incident_normalizer_imag,
            dtype=jnp.float32,
        ),
        source_packet_offsets=jnp.asarray(face_packet_offsets, dtype=jnp.int32),
        source_packet_lengths=jnp.asarray(face_packet_lengths, dtype=jnp.int32),
        source_normal_axis=jnp.asarray(face_normal_axis, dtype=jnp.int32),
        source_normal_sign=jnp.asarray(face_normal_sign, dtype=jnp.int32),
        source_tangential_axis_0=jnp.asarray(
            face_tangential_axis_0,
            dtype=jnp.int32,
        ),
        source_tangential_axis_1=jnp.asarray(
            face_tangential_axis_1,
            dtype=jnp.int32,
        ),
    )


def _ensure_private_interface_owner_state(
    state: SubgridState3D,
    config: SubgridConfig3D,
) -> _PrivateInterfaceOwnerState:
    if state.private_interface_owner_state is None:
        return _init_private_interface_owner_state(config)
    return state.private_interface_owner_state


def _advance_private_interface_owner_state(
    owner_state: _PrivateInterfaceOwnerState,
) -> _PrivateInterfaceOwnerState:
    return owner_state._replace(
        face_update_count=owner_state.face_update_count + jnp.asarray(1, dtype=jnp.int32)
    )


def _stage_private_time_aligned_owner_packets(
    owner_state: _PrivateInterfaceOwnerState,
) -> _PrivateInterfaceOwnerState:
    """Stage the last completed source/interface packet pair for modal retry.

    The propagation-aware modal retry consumes source and interface packets from
    the same completed solver step.  The current source packet is overwritten
    before that retry in each step, so this private staging helper snapshots the
    previous current pair immediately before the overwrite.  It stays entirely
    inside ``_PrivateInterfaceOwnerState`` and does not create public observables,
    hooks, or benchmark monitor dependencies.
    """

    return owner_state._replace(
        face_proxy_reference_prev_real=owner_state.face_proxy_reference_real,
        face_proxy_reference_prev_imag=owner_state.face_proxy_reference_imag,
        source_owner_reference_prev_real=owner_state.source_owner_reference_real,
        source_owner_reference_prev_imag=owner_state.source_owner_reference_imag,
    )


def _masked_face_mean(value: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    denominator = jnp.maximum(
        jnp.sum(mask),
        jnp.asarray(1.0, dtype=mask.dtype),
    )
    return jnp.sum(value * mask) / denominator


def _private_owner_face_complex_distribution(
    *,
    coarse_e: tuple[jnp.ndarray, jnp.ndarray],
    fine_e: tuple[jnp.ndarray, jnp.ndarray],
    coarse_h: tuple[jnp.ndarray, jnp.ndarray],
    fine_h: tuple[jnp.ndarray, jnp.ndarray],
    ops: ZFaceOps,
    normal_sign: int,
) -> jnp.ndarray:
    eta0 = jnp.asarray(np.sqrt(MU_0 / EPS_0), dtype=coarse_e[0].dtype)
    sign = jnp.asarray(normal_sign, dtype=coarse_e[0].dtype)
    fine_e_restricted = tuple(restrict_face(trace, ops) for trace in fine_e)
    fine_h_restricted = tuple(restrict_face(trace, ops) for trace in fine_h)
    coarse_w1 = coarse_e[0] + sign * eta0 * (-coarse_h[1])
    coarse_w2 = coarse_e[1] + sign * eta0 * coarse_h[0]
    fine_w1 = fine_e_restricted[0] + sign * eta0 * (-fine_h_restricted[1])
    fine_w2 = fine_e_restricted[1] + sign * eta0 * fine_h_restricted[0]
    face_reference = 0.5 * (
        (coarse_w1 + 1j * coarse_w2) + (fine_w1 + 1j * fine_w2)
    )
    return face_reference


def _private_owner_face_complex_reference(
    *,
    coarse_e: tuple[jnp.ndarray, jnp.ndarray],
    fine_e: tuple[jnp.ndarray, jnp.ndarray],
    coarse_h: tuple[jnp.ndarray, jnp.ndarray],
    fine_h: tuple[jnp.ndarray, jnp.ndarray],
    ops: ZFaceOps,
    coarse_mask: jnp.ndarray,
    normal_sign: int,
) -> jnp.ndarray:
    face_reference = _private_owner_face_complex_distribution(
        coarse_e=coarse_e,
        fine_e=fine_e,
        coarse_h=coarse_h,
        fine_h=fine_h,
        ops=ops,
        normal_sign=normal_sign,
    )
    return _masked_face_mean(face_reference, coarse_mask)


def _update_private_interface_owner_state_from_scan(
    owner_state: _PrivateInterfaceOwnerState,
    *,
    e_post_sat_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    e_post_sat_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    h_post_sat_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    h_post_sat_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> _PrivateInterfaceOwnerState:
    """Capture private owner face references from the same-step E/H SAT scan."""

    phase_references = []
    magnitude_references = []
    packet_real = []
    packet_imag = []
    packet_weight = []
    packet_mask = []
    for face in _active_faces(config):
        orientation = FACE_ORIENTATIONS[face]
        ops = _get_face_ops(config, face)
        coarse_mask, _ = _face_interior_masks(ops.coarse_shape, config.ratio)
        coarse_e = extract_tangential_e_face(
            e_post_sat_coarse,
            config,
            face,
            grid="coarse",
        )
        fine_e = extract_tangential_e_face(
            e_post_sat_fine,
            config,
            face,
            grid="fine",
        )
        coarse_h = extract_tangential_h_face(
            h_post_sat_coarse,
            config,
            face,
            grid="coarse",
        )
        fine_h = extract_tangential_h_face(
            h_post_sat_fine,
            config,
            face,
            grid="fine",
        )
        face_distribution = _private_owner_face_complex_distribution(
            coarse_e=coarse_e,
            fine_e=fine_e,
            coarse_h=coarse_h,
            fine_h=fine_h,
            ops=ops,
            normal_sign=orientation.normal_sign,
        )
        face_reference = _masked_face_mean(face_distribution, coarse_mask)
        phase_references.append(jnp.angle(face_reference).astype(jnp.float32))
        magnitude_references.append(jnp.abs(face_reference).astype(jnp.float32))
        packet_real.append(
            jnp.ravel(jnp.real(face_distribution).astype(jnp.float32) * coarse_mask)
        )
        packet_imag.append(
            jnp.ravel(jnp.imag(face_distribution).astype(jnp.float32) * coarse_mask)
        )
        packet_weight.append(jnp.ravel(ops.coarse_norm * coarse_mask))
        packet_mask.append(jnp.ravel(coarse_mask))
    if not phase_references:
        return owner_state
    return owner_state._replace(
        face_phase_reference=jnp.stack(phase_references),
        face_magnitude_reference=jnp.stack(magnitude_references),
        face_proxy_reference_real=jnp.concatenate(packet_real),
        face_proxy_reference_imag=jnp.concatenate(packet_imag),
        face_proxy_weight=jnp.concatenate(packet_weight).astype(jnp.float32),
        face_proxy_mask=jnp.concatenate(packet_mask).astype(jnp.float32),
    )


def _update_private_source_owner_state_from_scan(
    owner_state: _PrivateInterfaceOwnerState,
    *,
    e_source_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    e_source_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    h_source_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    h_source_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> _PrivateInterfaceOwnerState:
    """Populate private source-owner packets before modal retry.

    This is a solver-local packetization step only: it samples the same private
    tangential E/H face distribution used by the owner-state diagnostics, writes
    it into the source-owner packet fields, and leaves the interface-owner packet
    untouched.  It does not consume benchmark monitors, public TFSF state, hooks,
    ports, S-parameters, or public observables.
    """

    packet_real = []
    packet_imag = []
    packet_weight = []
    packet_mask = []
    for face in _active_faces(config):
        orientation = FACE_ORIENTATIONS[face]
        ops = _get_face_ops(config, face)
        coarse_mask, _ = _face_interior_masks(ops.coarse_shape, config.ratio)
        coarse_e = extract_tangential_e_face(
            e_source_coarse,
            config,
            face,
            grid="coarse",
        )
        fine_e = extract_tangential_e_face(
            e_source_fine,
            config,
            face,
            grid="fine",
        )
        coarse_h = extract_tangential_h_face(
            h_source_coarse,
            config,
            face,
            grid="coarse",
        )
        fine_h = extract_tangential_h_face(
            h_source_fine,
            config,
            face,
            grid="fine",
        )
        source_distribution = _private_owner_face_complex_distribution(
            coarse_e=coarse_e,
            fine_e=fine_e,
            coarse_h=coarse_h,
            fine_h=fine_h,
            ops=ops,
            normal_sign=orientation.normal_sign,
        )
        packet_real.append(
            jnp.ravel(jnp.real(source_distribution).astype(jnp.float32) * coarse_mask)
        )
        packet_imag.append(
            jnp.ravel(jnp.imag(source_distribution).astype(jnp.float32) * coarse_mask)
        )
        packet_weight.append(jnp.ravel(ops.coarse_norm * coarse_mask))
        packet_mask.append(jnp.ravel(coarse_mask))
    if not packet_real:
        return owner_state
    source_mask = jnp.concatenate(packet_mask).astype(jnp.float32)
    return owner_state._replace(
        source_owner_reference_real=jnp.concatenate(packet_real),
        source_owner_reference_imag=jnp.concatenate(packet_imag),
        source_owner_weight=jnp.concatenate(packet_weight).astype(jnp.float32),
        source_owner_mask=source_mask,
        source_incident_normalizer_real=source_mask,
        source_incident_normalizer_imag=jnp.zeros_like(source_mask),
    )


def _apply_observable_proxy_modal_retry_face_helper(
    current_e_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_e_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_h_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_h_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    owner_state: _PrivateInterfaceOwnerState,
    config: SubgridConfig3D,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Retry a private observable-proxy modal E update from packed face state.

    The retry is deliberately solver-local and lagged: it reads only the
    private owner packet captured by previous same-step face scans, projects a
    small bounded correction onto the current tangential E modal components,
    and reuses the existing face scatter helpers.  It does not consume benchmark
    DFT/flux planes, hooks, or public observables.
    """

    e_coarse = current_e_coarse
    e_fine = current_e_fine
    h_coarse = current_h_coarse
    h_fine = current_h_fine
    packet_offset = 0
    for face_index, face in enumerate(_active_faces(config)):
        orientation = FACE_ORIENTATIONS[face]
        ops = _get_face_ops(config, face)
        packet_length = int(np.prod(ops.coarse_shape))
        packet_slice = slice(packet_offset, packet_offset + packet_length)
        packet_offset += packet_length
        packet_mask = owner_state.face_proxy_mask[packet_slice].reshape(
            ops.coarse_shape
        )
        target_real = owner_state.face_proxy_reference_real[packet_slice].reshape(
            ops.coarse_shape
        )
        target_imag = owner_state.face_proxy_reference_imag[packet_slice].reshape(
            ops.coarse_shape
        )
        active_scale = jnp.where(
            owner_state.face_update_count[face_index] > 0,
            jnp.asarray(1.0, dtype=target_real.dtype),
            jnp.asarray(0.0, dtype=target_real.dtype),
        )

        coarse_e = extract_tangential_e_face(e_coarse, config, face, grid="coarse")
        fine_e = extract_tangential_e_face(e_fine, config, face, grid="fine")
        coarse_h = extract_tangential_h_face(h_coarse, config, face, grid="coarse")
        fine_h = extract_tangential_h_face(h_fine, config, face, grid="fine")
        current_distribution = _private_owner_face_complex_distribution(
            coarse_e=coarse_e,
            fine_e=fine_e,
            coarse_h=coarse_h,
            fine_h=fine_h,
            ops=ops,
            normal_sign=orientation.normal_sign,
        )
        current_real = jnp.real(current_distribution).astype(jnp.float32)
        current_imag = jnp.imag(current_distribution).astype(jnp.float32)
        relaxation = jnp.asarray(
            _OBSERVABLE_PROXY_MODAL_RETRY_RELAXATION,
            dtype=current_real.dtype,
        )
        eps = jnp.asarray(1.0e-30, dtype=current_real.dtype)
        delta_real = relaxation * (target_real - current_real)
        delta_imag = relaxation * (target_imag - current_imag)
        bound_real = relaxation * (jnp.abs(target_real) + jnp.abs(current_real) + eps)
        bound_imag = relaxation * (jnp.abs(target_imag) + jnp.abs(current_imag) + eps)
        delta_real = (
            jnp.clip(delta_real, -bound_real, bound_real)
            * packet_mask
            * active_scale
        )
        delta_imag = (
            jnp.clip(delta_imag, -bound_imag, bound_imag)
            * packet_mask
            * active_scale
        )
        corrected_coarse_e = (
            coarse_e[0] + delta_real,
            coarse_e[1] + delta_imag,
        )
        corrected_fine_e = (
            fine_e[0] + prolong_face(delta_real, ops),
            fine_e[1] + prolong_face(delta_imag, ops),
        )
        e_coarse = scatter_tangential_e_face(
            e_coarse,
            corrected_coarse_e,
            config,
            face,
            grid="coarse",
        )
        e_fine = scatter_tangential_e_face(
            e_fine,
            corrected_fine_e,
            config,
            face,
            grid="fine",
        )
    return e_coarse, e_fine, h_coarse, h_fine


def _incident_normalized_source_packet(
    *,
    source_real: jnp.ndarray,
    source_imag: jnp.ndarray,
    normalizer_real: jnp.ndarray,
    normalizer_imag: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the private source-owner packet divided by its incident normalizer."""

    eps = jnp.asarray(1.0e-30, dtype=source_real.dtype)
    denominator = jnp.maximum(
        normalizer_real * normalizer_real + normalizer_imag * normalizer_imag,
        eps,
    )
    normalized_real = (
        source_real * normalizer_real + source_imag * normalizer_imag
    ) / denominator
    normalized_imag = (
        source_imag * normalizer_real - source_real * normalizer_imag
    ) / denominator
    return normalized_real, normalized_imag


def _private_transverse_modal_coupling_metric(
    *,
    gram: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a private fixed-shape modal-coupling metric and readiness gate.

    The metric is intentionally local to the private projection helper.  It
    strengthens only finite off-diagonal modal power couplings that already
    exist in the incident/reflected/transverse Gram packet, keeps the matrix
    shape fixed at 3x3, and returns a scalar gate so callers can fail closed
    without adding public state, hooks, observables, or runner inputs.
    """

    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    identity = jnp.eye(3, dtype=jnp.complex64)
    offdiag_mask = 1.0 - identity
    diag = jnp.maximum(jnp.real(jnp.diag(gram)), floor)
    diag_scale = jnp.sqrt(jnp.maximum(diag[:, None] * diag[None, :], floor))
    magnitude = jnp.abs(gram)
    normalized_coupling = magnitude / diag_scale
    bounded_coupling = jnp.clip(
        normalized_coupling,
        jnp.asarray(0.0, dtype=diag.dtype),
        jnp.asarray(0.35, dtype=diag.dtype),
    )
    phase = gram / jnp.maximum(magnitude, floor)
    coupling = (
        bounded_coupling.astype(jnp.complex64)
        * diag_scale.astype(jnp.complex64)
        * phase
        * offdiag_mask
        * active_outer
    )
    metric = gram + coupling
    metric_finite = jnp.all(jnp.isfinite(jnp.real(metric))) & jnp.all(
        jnp.isfinite(jnp.imag(metric))
    )
    metric_ready = metric_finite & (
        jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype)
    )
    safe_metric = jnp.where(metric_ready, metric, identity)
    return safe_metric, jnp.where(
        metric_ready,
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )


def _private_source_interface_transverse_modal_transfer_map(
    *,
    source_coeff: jnp.ndarray,
    interface_coeff: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a private fixed-shape source/interface modal transfer map.

    The map is a bounded minimum-norm correction from source-owner modal
    coefficients toward the interface residual target coefficients.  It is
    solver-local, fixed at 3x3, uses only existing private packet coefficients,
    and returns a scalar gate so callers can fail closed without public state,
    hooks, observables, runner inputs, or threshold changes.
    """

    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    source_active = source_coeff * active_complex
    interface_active = interface_coeff * active_complex
    delta_active = interface_active - source_active
    source_energy = jnp.sum(jnp.abs(source_active) ** 2)
    raw_transfer = (
        delta_active[:, None]
        * jnp.conj(source_active[None, :])
        / jnp.maximum(source_energy, floor).astype(jnp.complex64)
    )
    magnitude = jnp.abs(raw_transfer)
    bounded_magnitude = jnp.clip(
        magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    phase = raw_transfer / jnp.maximum(magnitude, floor)
    transfer = bounded_magnitude.astype(jnp.complex64) * phase * active_outer
    transfer_finite = jnp.all(jnp.isfinite(jnp.real(transfer))) & jnp.all(
        jnp.isfinite(jnp.imag(transfer))
    )
    transfer_ready = (
        transfer_finite
        & (source_energy > floor)
        & (jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype))
    )
    safe_transfer = jnp.where(
        transfer_ready,
        transfer,
        jnp.zeros((3, 3), dtype=jnp.complex64),
    )
    return safe_transfer, jnp.where(
        transfer_ready,
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )


def _private_target_basis_oriented_source_interface_transverse_modal_transfer_map(
    *,
    source_coeff: jnp.ndarray,
    interface_coeff: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a target-basis-oriented private source/interface transfer map.

    The earlier transfer helper was intentionally source-outer oriented: it used
    the source modal packet as the right basis for a bounded minimum-norm map.
    This private follow-up keeps the same fixed 3x3 shape and owner-packet
    inputs, but orients the right basis with the projected residual target
    itself.  The scalar source/target overlap denominator is fail-closed, so a
    near-orthogonal or non-finite target basis produces a zero map without
    public state, runner hooks, observables, or threshold changes.
    """

    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    source_active = source_coeff * active_complex
    target_active = (interface_coeff - source_coeff) * active_complex
    source_energy = jnp.sum(jnp.abs(source_active) ** 2)
    target_energy = jnp.sum(jnp.abs(target_active) ** 2)
    overlap = jnp.sum(jnp.conj(target_active) * source_active)
    overlap_magnitude = jnp.abs(overlap)
    raw_transfer = (
        target_active[:, None]
        * jnp.conj(target_active[None, :])
        / jnp.maximum(overlap_magnitude, floor).astype(jnp.complex64)
        * (jnp.conj(overlap) / jnp.maximum(overlap_magnitude, floor))
    )
    magnitude = jnp.abs(raw_transfer)
    bounded_magnitude = jnp.clip(
        magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    phase = raw_transfer / jnp.maximum(magnitude, floor)
    transfer = bounded_magnitude.astype(jnp.complex64) * phase * active_outer
    transfer_finite = jnp.all(jnp.isfinite(jnp.real(transfer))) & jnp.all(
        jnp.isfinite(jnp.imag(transfer))
    )
    orientation_ready = (
        transfer_finite
        & (source_energy > floor)
        & (target_energy > floor)
        & (overlap_magnitude > floor)
        & (jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype))
    )
    safe_transfer = jnp.where(
        orientation_ready,
        transfer,
        jnp.zeros((3, 3), dtype=jnp.complex64),
    )
    return safe_transfer, jnp.where(
        orientation_ready,
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )




def _private_target_basis_residual_phase_sign_source_interface_transverse_modal_transfer_map(
    *,
    source_coeff: jnp.ndarray,
    interface_coeff: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a private residual phase/sign-oriented transfer map.

    This helper is a bounded follow-up to the target-basis-oriented map.  It
    keeps the same private 3x3 modal packet shape, consumes no public state, and
    applies only a row-local residual phase/sign correction when the existing
    source, target, and target-oriented correction packets are finite and
    non-degenerate.  Non-finite or near-zero residuals fail closed to a zero map
    so callers can keep the solver-local projection path branch-free.
    """

    target_oriented_transfer, target_orientation_ready = (
        _private_target_basis_oriented_source_interface_transverse_modal_transfer_map(
            source_coeff=source_coeff,
            interface_coeff=interface_coeff,
            active=active,
            floor=floor,
        )
    )
    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    source_active = source_coeff * active_complex
    target_active = (interface_coeff - source_coeff) * active_complex
    target_oriented_correction = target_oriented_transfer @ source_active
    target_magnitude = jnp.abs(target_active)
    correction_magnitude = jnp.abs(target_oriented_correction)
    target_phase = target_active / jnp.maximum(target_magnitude, floor)
    correction_phase = target_oriented_correction / jnp.maximum(
        correction_magnitude,
        floor,
    )
    residual_phase_sign = target_phase * jnp.conj(correction_phase)
    phase_sign_map = residual_phase_sign[:, None] * target_oriented_transfer
    magnitude = jnp.abs(phase_sign_map)
    bounded_magnitude = jnp.clip(
        magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    phase = phase_sign_map / jnp.maximum(magnitude, floor)
    transfer = bounded_magnitude.astype(jnp.complex64) * phase * active_outer
    source_energy = jnp.sum(jnp.abs(source_active) ** 2)
    target_energy = jnp.sum(jnp.abs(target_active) ** 2)
    correction_energy = jnp.sum(jnp.abs(target_oriented_correction) ** 2)
    phase_sign_finite = jnp.all(jnp.isfinite(jnp.real(transfer))) & jnp.all(
        jnp.isfinite(jnp.imag(transfer))
    )
    phase_sign_ready = (
        phase_sign_finite
        & (target_orientation_ready > jnp.asarray(0.0, dtype=active.dtype))
        & (source_energy > floor)
        & (target_energy > floor)
        & (correction_energy > floor)
        & (jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype))
    )
    safe_transfer = jnp.where(
        phase_sign_ready,
        transfer,
        jnp.zeros((3, 3), dtype=jnp.complex64),
    )
    return safe_transfer, jnp.where(
        phase_sign_ready,
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )


def _private_target_basis_residual_phase_magnitude_balance_source_interface_transverse_modal_transfer_map(
    *,
    source_coeff: jnp.ndarray,
    interface_coeff: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a private residual phase/magnitude-balanced transfer map.

    This bounded follow-up keeps the same fixed 3x3 private modal packet
    contract as the residual phase/sign helper, but applies a row-local
    magnitude balance before the final 0.35 transfer clamp.  It consumes only
    existing source/interface modal coefficients and fails closed on non-finite
    or degenerate phase/magnitude evidence, so no public observables, runner
    state, hooks, exports, or threshold changes are needed.
    """

    phase_sign_transfer, phase_sign_ready = (
        _private_target_basis_residual_phase_sign_source_interface_transverse_modal_transfer_map(
            source_coeff=source_coeff,
            interface_coeff=interface_coeff,
            active=active,
            floor=floor,
        )
    )
    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    source_active = source_coeff * active_complex
    target_active = (interface_coeff - source_coeff) * active_complex
    phase_sign_correction = phase_sign_transfer @ source_active
    source_energy = jnp.sum(jnp.abs(source_active) ** 2)
    residual_correction = target_active - phase_sign_correction
    raw_balance = (
        residual_correction[:, None]
        * jnp.conj(source_active[None, :])
        / jnp.maximum(source_energy, floor).astype(jnp.complex64)
    )
    balance_magnitude = jnp.abs(raw_balance)
    bounded_balance_magnitude = jnp.clip(
        balance_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    balance_phase = raw_balance / jnp.maximum(balance_magnitude, floor)
    balance_map = bounded_balance_magnitude.astype(jnp.complex64) * balance_phase
    balanced_map = phase_sign_transfer + balance_map * active_outer
    magnitude = jnp.abs(balanced_map)
    bounded_magnitude = jnp.clip(
        magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    phase = balanced_map / jnp.maximum(magnitude, floor)
    transfer = bounded_magnitude.astype(jnp.complex64) * phase * active_outer
    target_energy = jnp.sum(jnp.abs(target_active) ** 2)
    correction_energy = jnp.sum(jnp.abs(phase_sign_correction) ** 2)
    balance_finite = (
        jnp.all(jnp.isfinite(jnp.real(raw_balance)))
        & jnp.all(jnp.isfinite(jnp.imag(raw_balance)))
        & jnp.all(jnp.isfinite(jnp.real(transfer)))
        & jnp.all(jnp.isfinite(jnp.imag(transfer)))
    )
    balance_ready = (
        balance_finite
        & (phase_sign_ready > jnp.asarray(0.0, dtype=active.dtype))
        & (source_energy > floor)
        & (target_energy > floor)
        & (correction_energy > floor)
        & (jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype))
    )
    safe_transfer = jnp.where(
        balance_ready,
        transfer,
        jnp.zeros((3, 3), dtype=jnp.complex64),
    )
    return safe_transfer, jnp.where(
        balance_ready,
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )


def _private_target_basis_residual_modal_coupling_source_interface_transverse_modal_transfer_map(
    *,
    source_coeff: jnp.ndarray,
    interface_coeff: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a private residual modal-coupling transfer map.

    This bounded follow-up consumes the retained residual phase/magnitude
    balance contract, but lets active target rows couple against active source
    columns directly inside the same fixed 3x3 private modal packet shape.  It
    uses no public observable, runner state, exports, hooks, or threshold
    changes and fails closed on non-finite or degenerate modal-coupling
    evidence.
    """

    balance_transfer, balance_ready = (
        _private_target_basis_residual_phase_magnitude_balance_source_interface_transverse_modal_transfer_map(
            source_coeff=source_coeff,
            interface_coeff=interface_coeff,
            active=active,
            floor=floor,
        )
    )
    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    source_active = source_coeff * active_complex
    target_active = (interface_coeff - source_coeff) * active_complex
    source_energy = jnp.sum(jnp.abs(source_active) ** 2)
    raw_coupling = (
        target_active[:, None]
        * jnp.conj(source_active[None, :])
        / jnp.maximum(source_energy, floor).astype(jnp.complex64)
    )
    coupling_magnitude = jnp.abs(raw_coupling)
    bounded_coupling_magnitude = jnp.clip(
        coupling_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    coupling_phase = raw_coupling / jnp.maximum(coupling_magnitude, floor)
    transfer = (
        bounded_coupling_magnitude.astype(jnp.complex64)
        * coupling_phase
        * active_outer
    )
    target_energy = jnp.sum(jnp.abs(target_active) ** 2)
    balance_correction = balance_transfer @ source_active
    coupling_correction = transfer @ source_active
    balance_energy = jnp.sum(jnp.abs(balance_correction) ** 2)
    coupling_energy = jnp.sum(jnp.abs(coupling_correction) ** 2)
    coupling_finite = (
        jnp.all(jnp.isfinite(jnp.real(raw_coupling)))
        & jnp.all(jnp.isfinite(jnp.imag(raw_coupling)))
        & jnp.all(jnp.isfinite(jnp.real(transfer)))
        & jnp.all(jnp.isfinite(jnp.imag(transfer)))
    )
    coupling_ready = (
        coupling_finite
        & (balance_ready > jnp.asarray(0.0, dtype=active.dtype))
        & (source_energy > floor)
        & (target_energy > floor)
        & (balance_energy > floor)
        & (coupling_energy > floor)
        & (jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype))
    )
    safe_transfer = jnp.where(
        coupling_ready,
        transfer,
        jnp.zeros((3, 3), dtype=jnp.complex64),
    )
    return safe_transfer, jnp.where(
        coupling_ready,
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )




def _private_target_basis_residual_modal_coupling_packet_basis_mismatch_source_interface_transverse_modal_transfer_map(
    *,
    source_coeff: jnp.ndarray,
    interface_coeff: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a private packet-basis mismatch transfer map.

    This bounded follow-up consumes the retained residual modal-coupling
    contract, then applies one additional fixed-shape source/interface packet
    basis mismatch correction from the residual left after the modal-coupling
    transfer.  It uses only existing private modal coefficients, keeps the same
    3x3 packet shape, clamps the total transfer magnitude to 0.35, and fails
    closed on non-finite or degenerate packet-basis evidence.
    """

    coupling_transfer, coupling_ready = (
        _private_target_basis_residual_modal_coupling_source_interface_transverse_modal_transfer_map(
            source_coeff=source_coeff,
            interface_coeff=interface_coeff,
            active=active,
            floor=floor,
        )
    )
    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    source_active = source_coeff * active_complex
    interface_active = interface_coeff * active_complex
    target_active = interface_active - source_active
    coupling_correction = coupling_transfer @ source_active
    residual_after_coupling = target_active - coupling_correction
    packet_basis_mismatch = (interface_active - source_active) * active_complex
    mismatch_energy = jnp.sum(jnp.abs(packet_basis_mismatch) ** 2)
    raw_mismatch = (
        residual_after_coupling[:, None]
        * jnp.conj(packet_basis_mismatch[None, :])
        / jnp.maximum(mismatch_energy, floor).astype(jnp.complex64)
    )
    mismatch_magnitude = jnp.abs(raw_mismatch)
    bounded_mismatch_magnitude = jnp.clip(
        mismatch_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    mismatch_phase = raw_mismatch / jnp.maximum(mismatch_magnitude, floor)
    mismatch_map = (
        bounded_mismatch_magnitude.astype(jnp.complex64)
        * mismatch_phase
        * active_outer
    )
    combined_map = coupling_transfer + mismatch_map
    combined_magnitude = jnp.abs(combined_map)
    bounded_combined_magnitude = jnp.clip(
        combined_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    combined_phase = combined_map / jnp.maximum(combined_magnitude, floor)
    transfer = bounded_combined_magnitude.astype(jnp.complex64) * combined_phase
    mismatch_correction = transfer @ source_active
    source_energy = jnp.sum(jnp.abs(source_active) ** 2)
    target_energy = jnp.sum(jnp.abs(target_active) ** 2)
    residual_energy = jnp.sum(jnp.abs(residual_after_coupling) ** 2)
    correction_energy = jnp.sum(jnp.abs(mismatch_correction) ** 2)
    mismatch_finite = (
        jnp.all(jnp.isfinite(jnp.real(raw_mismatch)))
        & jnp.all(jnp.isfinite(jnp.imag(raw_mismatch)))
        & jnp.all(jnp.isfinite(jnp.real(transfer)))
        & jnp.all(jnp.isfinite(jnp.imag(transfer)))
    )
    mismatch_ready = (
        mismatch_finite
        & (coupling_ready > jnp.asarray(0.0, dtype=active.dtype))
        & (source_energy > floor)
        & (target_energy > floor)
        & (mismatch_energy > floor)
        & (residual_energy > floor)
        & (correction_energy > floor)
        & (jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype))
    )
    safe_transfer = jnp.where(
        mismatch_ready,
        transfer,
        jnp.zeros((3, 3), dtype=jnp.complex64),
    )
    return safe_transfer, jnp.where(
        mismatch_ready,
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )



def _private_target_basis_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_source_interface_transverse_modal_transfer_map(
    *,
    source_coeff: jnp.ndarray,
    interface_coeff: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a private owner-packet weighting transfer map.

    This bounded follow-up consumes the retained packet-basis mismatch transfer
    map, then applies one source/interface owner-packet weighting correction
    from the residual left after the packet-basis mismatch hunk.  It keeps the
    same fixed 3x3 modal packet shape, reuses only existing private owner
    packets, clamps the total transfer magnitude to 0.35, and fails closed on
    non-finite or degenerate owner-weight evidence without public state,
    observables, runner hooks, exports, or threshold changes.
    """

    packet_basis_transfer, packet_basis_ready = (
        _private_target_basis_residual_modal_coupling_packet_basis_mismatch_source_interface_transverse_modal_transfer_map(
            source_coeff=source_coeff,
            interface_coeff=interface_coeff,
            active=active,
            floor=floor,
        )
    )
    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    source_active = source_coeff * active_complex
    interface_active = interface_coeff * active_complex
    target_active = interface_active - source_active
    packet_basis_correction = packet_basis_transfer @ source_active
    residual_after_packet_basis = target_active - packet_basis_correction
    source_mode_energy = jnp.abs(source_active) ** 2
    interface_mode_energy = jnp.abs(interface_active) ** 2
    owner_weight = jnp.sqrt(
        jnp.maximum(interface_mode_energy, floor) / jnp.maximum(source_mode_energy, floor)
    )
    owner_weight = jnp.clip(
        owner_weight,
        jnp.asarray(0.5, dtype=floor.dtype),
        jnp.asarray(2.0, dtype=floor.dtype),
    ) * active
    owner_weighted_energy = jnp.sum(source_mode_energy * owner_weight)
    raw_weighting = (
        residual_after_packet_basis[:, None]
        * jnp.conj(source_active[None, :])
        * owner_weight[None, :].astype(jnp.complex64)
        / jnp.maximum(owner_weighted_energy, floor).astype(jnp.complex64)
    )
    weighting_magnitude = jnp.abs(raw_weighting)
    bounded_weighting_magnitude = jnp.clip(
        weighting_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    weighting_phase = raw_weighting / jnp.maximum(weighting_magnitude, floor)
    weighting_map = (
        bounded_weighting_magnitude.astype(jnp.complex64)
        * weighting_phase
        * active_outer
    )
    combined_map = packet_basis_transfer + weighting_map
    combined_magnitude = jnp.abs(combined_map)
    bounded_combined_magnitude = jnp.clip(
        combined_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    combined_phase = combined_map / jnp.maximum(combined_magnitude, floor)
    transfer = bounded_combined_magnitude.astype(jnp.complex64) * combined_phase
    owner_weighting_correction = transfer @ source_active
    source_energy = jnp.sum(source_mode_energy)
    target_energy = jnp.sum(jnp.abs(target_active) ** 2)
    residual_energy = jnp.sum(jnp.abs(residual_after_packet_basis) ** 2)
    correction_energy = jnp.sum(jnp.abs(owner_weighting_correction) ** 2)
    weighting_finite = (
        jnp.all(jnp.isfinite(owner_weight))
        & jnp.all(jnp.isfinite(jnp.real(raw_weighting)))
        & jnp.all(jnp.isfinite(jnp.imag(raw_weighting)))
        & jnp.all(jnp.isfinite(jnp.real(transfer)))
        & jnp.all(jnp.isfinite(jnp.imag(transfer)))
    )
    weighting_ready = (
        weighting_finite
        & (packet_basis_ready > jnp.asarray(0.0, dtype=active.dtype))
        & (source_energy > floor)
        & (target_energy > floor)
        & (owner_weighted_energy > floor)
        & (residual_energy > floor)
        & (correction_energy > floor)
        & (jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype))
    )
    safe_transfer = jnp.where(
        weighting_ready,
        transfer,
        jnp.zeros((3, 3), dtype=jnp.complex64),
    )
    return safe_transfer, jnp.where(
        weighting_ready,
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )


def _private_target_basis_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_source_interface_transverse_modal_transfer_map(
    *,
    source_coeff: jnp.ndarray,
    interface_coeff: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a private modal energy/impedance weighted transfer map.

    This bounded follow-up consumes the retained owner-packet weighting helper
    and adds one fixed-shape modal energy/impedance correction from the residual
    left after that helper.  The row weights use residual-vs-target modal
    energy, the column weights use interface-vs-source modal impedance energy,
    and both are clipped to a private finite interval before the total 3x3 map
    is clamped to the existing 0.35 bound.  The helper remains solver-local and
    fails closed on non-finite or degenerate modal energy evidence without
    public observables, runner state, exports, hooks, or threshold changes.
    """

    owner_weighting_transfer, owner_weighting_ready = (
        _private_target_basis_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_source_interface_transverse_modal_transfer_map(
            source_coeff=source_coeff,
            interface_coeff=interface_coeff,
            active=active,
            floor=floor,
        )
    )
    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    source_active = source_coeff * active_complex
    interface_active = interface_coeff * active_complex
    target_active = interface_active - source_active
    owner_weighting_correction = owner_weighting_transfer @ source_active
    residual_after_owner_weighting = target_active - owner_weighting_correction
    source_mode_energy = jnp.abs(source_active) ** 2
    interface_mode_energy = jnp.abs(interface_active) ** 2
    target_mode_energy = jnp.abs(target_active) ** 2
    residual_mode_energy = jnp.abs(residual_after_owner_weighting) ** 2
    row_energy_weight = jnp.sqrt(
        jnp.maximum(residual_mode_energy, floor)
        / jnp.maximum(target_mode_energy, floor)
    )
    column_impedance_weight = jnp.sqrt(
        jnp.maximum(interface_mode_energy + residual_mode_energy, floor)
        / jnp.maximum(source_mode_energy + target_mode_energy, floor)
    )
    row_energy_weight = jnp.clip(
        row_energy_weight,
        jnp.asarray(0.5, dtype=floor.dtype),
        jnp.asarray(2.0, dtype=floor.dtype),
    ) * active
    column_impedance_weight = jnp.clip(
        column_impedance_weight,
        jnp.asarray(0.5, dtype=floor.dtype),
        jnp.asarray(2.0, dtype=floor.dtype),
    ) * active
    modal_energy_impedance_weight = jnp.sqrt(
        jnp.maximum(
            row_energy_weight[:, None] * column_impedance_weight[None, :],
            jnp.asarray(0.0, dtype=floor.dtype),
        )
    )
    modal_energy_impedance_energy = jnp.sum(
        source_mode_energy * column_impedance_weight
    )
    raw_modal_energy_impedance = (
        residual_after_owner_weighting[:, None]
        * jnp.conj(source_active[None, :])
        * modal_energy_impedance_weight.astype(jnp.complex64)
        / jnp.maximum(modal_energy_impedance_energy, floor).astype(jnp.complex64)
    )
    modal_energy_impedance_magnitude = jnp.abs(raw_modal_energy_impedance)
    bounded_modal_energy_impedance_magnitude = jnp.clip(
        modal_energy_impedance_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    modal_energy_impedance_phase = raw_modal_energy_impedance / jnp.maximum(
        modal_energy_impedance_magnitude,
        floor,
    )
    modal_energy_impedance_map = (
        bounded_modal_energy_impedance_magnitude.astype(jnp.complex64)
        * modal_energy_impedance_phase
        * active_outer
    )
    combined_map = owner_weighting_transfer + modal_energy_impedance_map
    combined_magnitude = jnp.abs(combined_map)
    bounded_combined_magnitude = jnp.clip(
        combined_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    combined_phase = combined_map / jnp.maximum(combined_magnitude, floor)
    transfer = bounded_combined_magnitude.astype(jnp.complex64) * combined_phase
    modal_energy_impedance_correction = transfer @ source_active
    source_energy = jnp.sum(source_mode_energy)
    target_energy = jnp.sum(target_mode_energy)
    residual_energy = jnp.sum(residual_mode_energy)
    correction_energy = jnp.sum(jnp.abs(modal_energy_impedance_correction) ** 2)
    modal_energy_impedance_finite = (
        jnp.all(jnp.isfinite(row_energy_weight))
        & jnp.all(jnp.isfinite(column_impedance_weight))
        & jnp.all(jnp.isfinite(modal_energy_impedance_weight))
        & jnp.all(jnp.isfinite(jnp.real(raw_modal_energy_impedance)))
        & jnp.all(jnp.isfinite(jnp.imag(raw_modal_energy_impedance)))
        & jnp.all(jnp.isfinite(jnp.real(transfer)))
        & jnp.all(jnp.isfinite(jnp.imag(transfer)))
    )
    modal_energy_impedance_ready = (
        modal_energy_impedance_finite
        & (owner_weighting_ready > jnp.asarray(0.0, dtype=active.dtype))
        & (source_energy > floor)
        & (target_energy > floor)
        & (residual_energy > floor)
        & (modal_energy_impedance_energy > floor)
        & (correction_energy > floor)
        & (jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype))
    )
    safe_transfer = jnp.where(
        modal_energy_impedance_ready,
        transfer,
        jnp.zeros((3, 3), dtype=jnp.complex64),
    )
    return safe_transfer, jnp.where(
        modal_energy_impedance_ready,
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )



def _private_target_basis_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_source_interface_transverse_modal_transfer_map(
    *,
    source_coeff: jnp.ndarray,
    interface_coeff: jnp.ndarray,
    active: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a private transverse energy redistribution transfer map.

    This bounded follow-up consumes the retained modal energy/impedance helper
    and applies one fixed-shape redistribution of the remaining transverse
    modal residual.  It keeps the existing 3x3 source/interface packet shape,
    clips all redistribution weights to a private finite interval, clamps the
    total transfer magnitude to 0.35, and fails closed on non-finite or
    degenerate transverse energy evidence without public observables, runner
    state, exports, hooks, or threshold changes.
    """

    modal_energy_impedance_transfer, modal_energy_impedance_ready = (
        _private_target_basis_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_source_interface_transverse_modal_transfer_map(
            source_coeff=source_coeff,
            interface_coeff=interface_coeff,
            active=active,
            floor=floor,
        )
    )
    active_complex = active.astype(jnp.complex64)
    active_outer = active_complex[:, None] * active_complex[None, :]
    source_active = source_coeff * active_complex
    interface_active = interface_coeff * active_complex
    target_active = interface_active - source_active
    modal_energy_impedance_correction = modal_energy_impedance_transfer @ source_active
    residual_after_modal_energy_impedance = (
        target_active - modal_energy_impedance_correction
    )
    source_mode_energy = jnp.abs(source_active) ** 2
    target_mode_energy = jnp.abs(target_active) ** 2
    residual_mode_energy = jnp.abs(residual_after_modal_energy_impedance) ** 2
    active_count = jnp.maximum(
        jnp.sum(active),
        jnp.asarray(1.0, dtype=active.dtype),
    )
    residual_mean_energy = jnp.sum(residual_mode_energy) / active_count
    transverse_row_weight = jnp.sqrt(
        jnp.maximum(residual_mode_energy, floor)
        / jnp.maximum(residual_mean_energy, floor)
    )
    transverse_row_weight = jnp.clip(
        transverse_row_weight,
        jnp.asarray(0.5, dtype=floor.dtype),
        jnp.asarray(2.0, dtype=floor.dtype),
    ) * active
    transverse_column_weight = jnp.sqrt(
        jnp.maximum(source_mode_energy + jnp.mean(source_mode_energy), floor)
        / jnp.maximum(target_mode_energy + residual_mode_energy, floor)
    )
    transverse_column_weight = jnp.clip(
        transverse_column_weight,
        jnp.asarray(0.5, dtype=floor.dtype),
        jnp.asarray(2.0, dtype=floor.dtype),
    ) * active
    transverse_energy = jnp.sum(source_mode_energy * transverse_column_weight)
    raw_transverse_redistribution = (
        residual_after_modal_energy_impedance[:, None]
        * jnp.conj(source_active[None, :])
        * transverse_row_weight[:, None].astype(jnp.complex64)
        * transverse_column_weight[None, :].astype(jnp.complex64)
        / jnp.maximum(transverse_energy, floor).astype(jnp.complex64)
    )
    transverse_redistribution_magnitude = jnp.abs(raw_transverse_redistribution)
    bounded_transverse_redistribution_magnitude = jnp.clip(
        transverse_redistribution_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    transverse_redistribution_phase = raw_transverse_redistribution / jnp.maximum(
        transverse_redistribution_magnitude,
        floor,
    )
    transverse_redistribution_map = (
        bounded_transverse_redistribution_magnitude.astype(jnp.complex64)
        * transverse_redistribution_phase
        * active_outer
    )
    combined_map = modal_energy_impedance_transfer + transverse_redistribution_map
    combined_magnitude = jnp.abs(combined_map)
    bounded_combined_magnitude = jnp.clip(
        combined_magnitude,
        jnp.asarray(0.0, dtype=floor.dtype),
        jnp.asarray(0.35, dtype=floor.dtype),
    )
    combined_phase = combined_map / jnp.maximum(combined_magnitude, floor)
    transfer = bounded_combined_magnitude.astype(jnp.complex64) * combined_phase
    transverse_correction = transfer @ source_active
    source_energy = jnp.sum(source_mode_energy)
    target_energy = jnp.sum(target_mode_energy)
    residual_energy = jnp.sum(residual_mode_energy)
    correction_energy = jnp.sum(jnp.abs(transverse_correction) ** 2)
    transverse_redistribution_finite = (
        jnp.all(jnp.isfinite(transverse_row_weight))
        & jnp.all(jnp.isfinite(transverse_column_weight))
        & jnp.all(jnp.isfinite(jnp.real(raw_transverse_redistribution)))
        & jnp.all(jnp.isfinite(jnp.imag(raw_transverse_redistribution)))
        & jnp.all(jnp.isfinite(jnp.real(transfer)))
        & jnp.all(jnp.isfinite(jnp.imag(transfer)))
    )
    transverse_redistribution_ready = (
        transverse_redistribution_finite
        & (modal_energy_impedance_ready > jnp.asarray(0.0, dtype=active.dtype))
        & (source_energy > floor)
        & (target_energy > floor)
        & (residual_energy > floor)
        & (residual_mean_energy > floor)
        & (transverse_energy > floor)
        & (correction_energy > floor)
        & (jnp.sum(active) > jnp.asarray(0.0, dtype=active.dtype))
    )
    safe_transfer = jnp.where(
        transverse_redistribution_ready,
        transfer,
        modal_energy_impedance_transfer,
    )
    return safe_transfer, jnp.where(
        modal_energy_impedance_ready > jnp.asarray(0.0, dtype=active.dtype),
        jnp.asarray(1.0, dtype=active.dtype),
        jnp.asarray(0.0, dtype=active.dtype),
    )


def _project_private_modal_basis_packets(
    *,
    source_real: jnp.ndarray,
    source_imag: jnp.ndarray,
    interface_real: jnp.ndarray,
    interface_imag: jnp.ndarray,
    normalizer_real: jnp.ndarray,
    normalizer_imag: jnp.ndarray,
    source_weight: jnp.ndarray,
    interface_weight: jnp.ndarray,
    source_mask: jnp.ndarray,
    interface_mask: jnp.ndarray,
    source_normal_sign: int | jnp.ndarray = 1,
    interface_normal_sign: int | jnp.ndarray = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project private source/interface packets onto shared target bases.

    This helper is deliberately solver-local and fixed-shape.  It consumes the
    existing private owner-state packets, calibrates their shared face-local
    SBP/mortar power metric from existing weights, masks, and normal signs,
    forms mask/weight-compatible incident, source-residual, and
    interface-residual modes, projects both private source and interface packets
    onto those modes, and subtracts only the shared projected packets.  Missing
    incident projection energy or incompatible metric shape returns a scalar
    zero gate so callers can fail closed without branching or exposing public
    observables.
    """

    dtype = source_real.dtype
    floor = jnp.asarray(1.0e-12, dtype=dtype)
    eps = jnp.asarray(1.0e-30, dtype=dtype)
    packet_mask = source_mask * interface_mask
    source_metric_weight = source_weight * packet_mask
    interface_metric_weight = interface_weight * packet_mask
    mortar_weight = jnp.sqrt(
        jnp.maximum(source_metric_weight * interface_metric_weight, 0.0)
    )
    source_energy = source_real * source_real + source_imag * source_imag
    interface_energy = interface_real * interface_real + interface_imag * interface_imag
    normalizer_energy = (
        normalizer_real * normalizer_real + normalizer_imag * normalizer_imag
    )
    characteristic_admittance = jnp.asarray(np.sqrt(EPS_0 / MU_0), dtype=dtype)
    active_weight = jnp.maximum(jnp.sum(mortar_weight), floor)
    source_metric_measure = jnp.maximum(jnp.sum(source_metric_weight), floor)
    interface_metric_measure = jnp.maximum(jnp.sum(interface_metric_weight), floor)
    source_power_density = (
        jnp.sum(source_metric_weight * source_energy) / source_metric_measure
    )
    interface_power_density = (
        jnp.sum(interface_metric_weight * interface_energy) / interface_metric_measure
    )
    normalizer_power_density = (
        jnp.sum(mortar_weight * normalizer_energy) / active_weight
    )
    coupled_power_density = (
        normalizer_power_density + 0.5 * (source_power_density + interface_power_density)
    )
    metric_reference_density = jnp.maximum(
        normalizer_power_density
        + jnp.sqrt(jnp.maximum(source_power_density * interface_power_density, 0.0)),
        floor,
    )
    face_metric_scale = jnp.clip(
        coupled_power_density / metric_reference_density,
        jnp.asarray(0.25, dtype=dtype),
        jnp.asarray(4.0, dtype=dtype),
    )
    source_sign = jnp.asarray(source_normal_sign, dtype=dtype)
    interface_sign = jnp.asarray(interface_normal_sign, dtype=dtype)
    metric_sign_contract = jnp.where(
        source_sign * interface_sign > jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )
    metric_shape_ready = jnp.where(
        (jnp.sum(mortar_weight) > floor) & (face_metric_scale > 0.0),
        metric_sign_contract,
        jnp.asarray(0.0, dtype=dtype),
    )
    packet_weight = (
        mortar_weight
        * characteristic_admittance
        * face_metric_scale
        * metric_shape_ready
    )
    source_power = jnp.sum(packet_weight * source_energy)

    def _complex_coeff(
        packet_real: jnp.ndarray,
        packet_imag: jnp.ndarray,
        mode_real: jnp.ndarray,
        mode_imag: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        coeff_real = jnp.sum(
            packet_weight * (packet_real * mode_real + packet_imag * mode_imag)
        )
        coeff_imag = jnp.sum(
            packet_weight * (packet_imag * mode_real - packet_real * mode_imag)
        )
        return coeff_real, coeff_imag

    def _apply_coeff(
        coeff_real: jnp.ndarray,
        coeff_imag: jnp.ndarray,
        mode_real: jnp.ndarray,
        mode_imag: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        projected_real = coeff_real * mode_real - coeff_imag * mode_imag
        projected_imag = coeff_real * mode_imag + coeff_imag * mode_real
        return projected_real, projected_imag

    def _mode_power(mode_real: jnp.ndarray, mode_imag: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(packet_weight * (mode_real * mode_real + mode_imag * mode_imag))

    def _normalize_mode(
        mode_real: jnp.ndarray,
        mode_imag: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        power = _mode_power(mode_real, mode_imag)
        mode_active = jnp.where(
            power > floor,
            jnp.asarray(1.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
        norm = jnp.sqrt(jnp.maximum(power, floor))
        unit_real = mode_active * mode_real / norm
        unit_imag = mode_active * mode_imag / norm
        return unit_real, unit_imag, mode_active

    def _remove_mode(
        packet_real: jnp.ndarray,
        packet_imag: jnp.ndarray,
        mode_real: jnp.ndarray,
        mode_imag: jnp.ndarray,
        mode_active: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        coeff_real, coeff_imag = _complex_coeff(
            packet_real,
            packet_imag,
            mode_real,
            mode_imag,
        )
        projected_real, projected_imag = _apply_coeff(
            coeff_real,
            coeff_imag,
            mode_real,
            mode_imag,
        )
        return (
            packet_real - mode_active * projected_real,
            packet_imag - mode_active * projected_imag,
        )

    def _modal_system() -> tuple[jnp.ndarray, jnp.ndarray]:
        modes_real = jnp.stack((incident_real, reflected_real, transverse_real))
        modes_imag = jnp.stack((incident_imag, reflected_imag, transverse_imag))
        mode_active = jnp.stack((incident_active, reflected_active, transverse_active))
        modes = (modes_real + 1j * modes_imag).astype(jnp.complex64)
        return modes, mode_active

    def _project_coeff(
        packet_real: jnp.ndarray,
        packet_imag: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        modes, mode_active = _modal_system()
        packet = (packet_real + 1j * packet_imag).astype(jnp.complex64)
        weight = packet_weight.astype(jnp.complex64)
        active = mode_active.astype(jnp.complex64)
        spatial_axes = tuple(range(2, modes.ndim + 1))
        gram = jnp.sum(
            weight[None, None, ...] * jnp.conj(modes[:, None, ...]) * modes[None, ...],
            axis=spatial_axes,
        )
        active_outer = active[:, None] * active[None, :]
        identity = jnp.eye(3, dtype=jnp.complex64)
        regularizer = floor.astype(jnp.complex64) * identity
        inactive_diagonal = jnp.diag(1.0 - active).astype(jnp.complex64)
        gram = gram * active_outer
        gram, coupling_ready = _private_transverse_modal_coupling_metric(
            gram=gram,
            active=mode_active,
            floor=floor,
        )
        gram = gram + inactive_diagonal + regularizer
        rhs = (
            jnp.sum(
                weight[None, ...] * jnp.conj(modes) * packet[None, ...],
                axis=tuple(range(1, modes.ndim)),
            )
            * active
            * coupling_ready.astype(jnp.complex64)
        )
        coeff = jnp.linalg.solve(gram, rhs)
        projected_coeff = coeff * active * coupling_ready.astype(jnp.complex64)
        return projected_coeff, coupling_ready

    def _packet_from_coeff(coeff: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        modes, mode_active = _modal_system()
        active = mode_active.astype(jnp.complex64)
        coeff_shape = (coeff.size,) + (1,) * (modes.ndim - 1)
        projected = jnp.sum((coeff * active).reshape(coeff_shape) * modes, axis=0)
        return jnp.real(projected).astype(dtype), jnp.imag(projected).astype(dtype)

    incident_real, incident_imag, incident_active = _normalize_mode(
        normalizer_real,
        normalizer_imag,
    )
    source_residual_real, source_residual_imag = _remove_mode(
        source_real,
        source_imag,
        incident_real,
        incident_imag,
        incident_active,
    )
    reflected_real, reflected_imag, reflected_active = _normalize_mode(
        source_residual_real,
        source_residual_imag,
    )
    interface_residual_real, interface_residual_imag = _remove_mode(
        interface_real,
        interface_imag,
        incident_real,
        incident_imag,
        incident_active,
    )
    interface_residual_real, interface_residual_imag = _remove_mode(
        interface_residual_real,
        interface_residual_imag,
        reflected_real,
        reflected_imag,
        reflected_active,
    )
    transverse_real, transverse_imag, transverse_active = _normalize_mode(
        interface_residual_real,
        interface_residual_imag,
    )
    source_coeff, source_coeff_ready = _project_coeff(
        source_real,
        source_imag,
    )
    interface_coeff, interface_coeff_ready = _project_coeff(
        interface_real,
        interface_imag,
    )
    mode_active = jnp.stack((incident_active, reflected_active, transverse_active))
    transfer_map, transfer_ready = (
        _private_target_basis_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_modal_energy_impedance_transverse_energy_redistribution_source_interface_transverse_modal_transfer_map(
            source_coeff=source_coeff,
            interface_coeff=interface_coeff,
            active=mode_active,
            floor=floor,
        )
    )
    transferred_source_coeff = source_coeff + transfer_ready.astype(
        jnp.complex64
    ) * (transfer_map @ source_coeff)
    projected_source_real, projected_source_imag = _packet_from_coeff(
        transferred_source_coeff
    )
    projected_interface_real, projected_interface_imag = _packet_from_coeff(
        interface_coeff
    )
    basis_ready = (incident_active + reflected_active + transverse_active) > 0.0

    projection_ready = (
        (incident_active > 0.0)
        & (source_power > eps)
        & basis_ready
        & (metric_shape_ready > 0.0)
        & (source_coeff_ready > 0.0)
        & (interface_coeff_ready > 0.0)
        & (transfer_ready > 0.0)
    )
    projected_target_real = jnp.where(
        projection_ready,
        projected_interface_real - projected_source_real,
        interface_real,
    )
    projected_target_imag = jnp.where(
        projection_ready,
        projected_interface_imag - projected_source_imag,
        interface_imag,
    )
    projection_gate = jnp.where(
        projection_ready,
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )
    return projected_target_real, projected_target_imag, projection_gate


def _private_modal_projection_normalizer_contract_gate(
    *,
    owner_state: _PrivateInterfaceOwnerState,
    face_index: int,
    packet_slice: slice,
    ops: ZFaceOps,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Gate private modal retry on the projection/normalizer contract.

    The gate is intentionally private and fail-closed.  It verifies that the
    source-owner and interface-owner packets use the same face packet layout,
    orientation metadata, masks, weights, and incident-normalizer availability
    before the modal retry consumes the packet pair.  It does not expose any
    public observable or configuration surface.
    """

    expected_length = jnp.asarray(int(np.prod(ops.coarse_shape)), dtype=jnp.int32)
    packet_eps = jnp.asarray(1.0e-6, dtype=dtype)
    source_mask = owner_state.source_owner_mask[packet_slice].reshape(
        ops.coarse_shape
    )
    interface_mask = owner_state.face_proxy_mask[packet_slice].reshape(
        ops.coarse_shape
    )
    source_weight = owner_state.source_owner_weight[packet_slice].reshape(
        ops.coarse_shape
    )
    interface_weight = owner_state.face_proxy_weight[packet_slice].reshape(
        ops.coarse_shape
    )
    normalizer_real = owner_state.source_incident_normalizer_real[
        packet_slice
    ].reshape(ops.coarse_shape)
    normalizer_imag = owner_state.source_incident_normalizer_imag[
        packet_slice
    ].reshape(ops.coarse_shape)

    face_offset = owner_state.face_packet_offsets[face_index]
    source_offset = owner_state.source_packet_offsets[face_index]
    face_length = owner_state.face_packet_lengths[face_index]
    source_length = owner_state.source_packet_lengths[face_index]
    layout_contract = (
        (face_offset == source_offset)
        & (face_length == source_length)
        & (face_length == expected_length)
    )

    face_normal_axis = owner_state.face_normal_axis[face_index]
    source_normal_axis = owner_state.source_normal_axis[face_index]
    face_normal_sign = owner_state.face_normal_sign[face_index]
    source_normal_sign = owner_state.source_normal_sign[face_index]
    orientation_contract = (
        (face_normal_axis == source_normal_axis)
        & (face_normal_sign == source_normal_sign)
        & (
            owner_state.face_tangential_axis_0[face_index]
            == owner_state.source_tangential_axis_0[face_index]
        )
        & (
            owner_state.face_tangential_axis_1[face_index]
            == owner_state.source_tangential_axis_1[face_index]
        )
    )
    mask_contract = jnp.max(jnp.abs(source_mask - interface_mask)) <= packet_eps
    active_measure = jnp.maximum(
        jnp.sum(interface_mask),
        jnp.asarray(1.0, dtype=dtype),
    )
    weight_contract = (
        jnp.sum(jnp.abs(source_weight - interface_weight) * interface_mask)
        <= packet_eps * active_measure
    )
    normalizer_energy = (
        normalizer_real * normalizer_real + normalizer_imag * normalizer_imag
    )
    normalizer_contract = jnp.sum(normalizer_energy * source_mask) > packet_eps
    contract_ready = (
        layout_contract
        & orientation_contract
        & mask_contract
        & weight_contract
        & normalizer_contract
    )
    return jnp.where(
        contract_ready,
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )


def _apply_propagation_aware_modal_retry_face_helper(
    current_e_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_e_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_h_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_h_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    owner_state: _PrivateInterfaceOwnerState,
    config: SubgridConfig3D,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Apply a private source/interface-aware modal E retry.

    This hunk is intentionally bounded and solver-local.  It consumes only the
    private source-owner packet, incident normalizer packet, and interface-owner
    packet already carried by ``_PrivateInterfaceOwnerState``.  It does not read
    benchmark DFT/flux/TFSF/port/S-parameter state, hooks, or public observables.

    The source packet is treated as an incident modal basis and subtracted from
    the lagged interface packet before a small correction is scattered back onto
    tangential E faces.  If no source-owner packet has been populated yet, the
    helper is a no-op; the existing observable-proxy helper remains the only
    active lagged correction.
    """

    e_coarse = current_e_coarse
    e_fine = current_e_fine
    h_coarse = current_h_coarse
    h_fine = current_h_fine
    packet_offset = 0
    for face_index, face in enumerate(_active_faces(config)):
        orientation = FACE_ORIENTATIONS[face]
        ops = _get_face_ops(config, face)
        packet_length = int(np.prod(ops.coarse_shape))
        packet_slice = slice(packet_offset, packet_offset + packet_length)
        packet_offset += packet_length

        interface_mask = owner_state.face_proxy_mask[packet_slice].reshape(
            ops.coarse_shape
        )
        source_mask = owner_state.source_owner_mask[packet_slice].reshape(
            ops.coarse_shape
        )
        packet_mask = interface_mask * source_mask
        interface_real = owner_state.face_proxy_reference_prev_real[
            packet_slice
        ].reshape(
            ops.coarse_shape
        )
        interface_imag = owner_state.face_proxy_reference_prev_imag[
            packet_slice
        ].reshape(
            ops.coarse_shape
        )
        source_real = owner_state.source_owner_reference_prev_real[
            packet_slice
        ].reshape(ops.coarse_shape)
        source_imag = owner_state.source_owner_reference_prev_imag[
            packet_slice
        ].reshape(ops.coarse_shape)
        normalizer_real = owner_state.source_incident_normalizer_real[
            packet_slice
        ].reshape(ops.coarse_shape)
        normalizer_imag = owner_state.source_incident_normalizer_imag[
            packet_slice
        ].reshape(ops.coarse_shape)
        source_weight = owner_state.source_owner_weight[packet_slice].reshape(
            ops.coarse_shape
        )
        interface_weight = owner_state.face_proxy_weight[packet_slice].reshape(
            ops.coarse_shape
        )
        target_real, target_imag, projection_gate = (
            _project_private_modal_basis_packets(
                source_real=source_real,
                source_imag=source_imag,
                interface_real=interface_real,
                interface_imag=interface_imag,
                normalizer_real=normalizer_real,
                normalizer_imag=normalizer_imag,
                source_weight=source_weight,
                interface_weight=interface_weight,
                source_mask=source_mask,
                interface_mask=interface_mask,
                source_normal_sign=owner_state.source_normal_sign[face_index],
                interface_normal_sign=owner_state.face_normal_sign[face_index],
            )
        )
        contract_gate = _private_modal_projection_normalizer_contract_gate(
            owner_state=owner_state,
            face_index=face_index,
            packet_slice=packet_slice,
            ops=ops,
            dtype=source_real.dtype,
        )
        source_energy = (source_real * source_real + source_imag * source_imag) * (
            source_mask
        )
        eps = jnp.asarray(1.0e-30, dtype=source_real.dtype)
        source_active_scale = jnp.where(
            jnp.sum(source_energy) > eps,
            jnp.asarray(1.0, dtype=source_real.dtype),
            jnp.asarray(0.0, dtype=source_real.dtype),
        )
        interface_active_scale = jnp.where(
            owner_state.face_update_count[face_index] > 0,
            jnp.asarray(1.0, dtype=source_real.dtype),
            jnp.asarray(0.0, dtype=source_real.dtype),
        )
        active_scale = (
            source_active_scale
            * interface_active_scale
            * contract_gate
            * projection_gate
        )
        coarse_e = extract_tangential_e_face(e_coarse, config, face, grid="coarse")
        fine_e = extract_tangential_e_face(e_fine, config, face, grid="fine")
        coarse_h = extract_tangential_h_face(h_coarse, config, face, grid="coarse")
        fine_h = extract_tangential_h_face(h_fine, config, face, grid="fine")
        current_distribution = _private_owner_face_complex_distribution(
            coarse_e=coarse_e,
            fine_e=fine_e,
            coarse_h=coarse_h,
            fine_h=fine_h,
            ops=ops,
            normal_sign=orientation.normal_sign,
        )
        current_real = jnp.real(current_distribution).astype(jnp.float32)
        current_imag = jnp.imag(current_distribution).astype(jnp.float32)
        relaxation = jnp.asarray(
            _PROPAGATION_AWARE_MODAL_RETRY_RELAXATION,
            dtype=current_real.dtype,
        )
        delta_real = relaxation * (target_real - current_real)
        delta_imag = relaxation * (target_imag - current_imag)
        bound_real = relaxation * (jnp.abs(target_real) + jnp.abs(current_real) + eps)
        bound_imag = relaxation * (jnp.abs(target_imag) + jnp.abs(current_imag) + eps)
        delta_real = (
            jnp.clip(delta_real, -bound_real, bound_real)
            * packet_mask
            * active_scale
        )
        delta_imag = (
            jnp.clip(delta_imag, -bound_imag, bound_imag)
            * packet_mask
            * active_scale
        )
        corrected_coarse_e = (
            coarse_e[0] + delta_real,
            coarse_e[1] + delta_imag,
        )
        corrected_fine_e = (
            fine_e[0] + prolong_face(delta_real, ops),
            fine_e[1] + prolong_face(delta_imag, ops),
        )
        e_coarse = scatter_tangential_e_face(
            e_coarse,
            corrected_coarse_e,
            config,
            face,
            grid="coarse",
        )
        e_fine = scatter_tangential_e_face(
            e_fine,
            corrected_fine_e,
            config,
            face,
            grid="fine",
        )
    return e_coarse, e_fine, h_coarse, h_fine


def _private_interface_owner_joint_score(
    owner_state: _PrivateInterfaceOwnerState,
) -> _PrivateInterfaceOwnerJointScore:
    """Score the private owner face references without exposing an observable."""

    magnitudes = owner_state.face_magnitude_reference
    phases = owner_state.face_phase_reference
    if magnitudes.size == 0:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return _PrivateInterfaceOwnerJointScore(
            transverse_magnitude_cv=zero,
            transverse_phase_spread_deg=zero,
            usable_face_count=jnp.asarray(0, dtype=jnp.int32),
        )
    eps = jnp.asarray(1.0e-30, dtype=magnitudes.dtype)
    magnitude_mean = jnp.maximum(jnp.mean(jnp.abs(magnitudes)), eps)
    magnitude_cv = jnp.std(magnitudes) / magnitude_mean
    phase_spread = (jnp.max(phases) - jnp.min(phases)) * (
        jnp.asarray(180.0, dtype=phases.dtype) / jnp.asarray(np.pi, dtype=phases.dtype)
    )
    usable_face_count = jnp.asarray(magnitudes.size, dtype=jnp.int32)
    return _PrivateInterfaceOwnerJointScore(
        transverse_magnitude_cv=magnitude_cv.astype(jnp.float32),
        transverse_phase_spread_deg=phase_spread.astype(jnp.float32),
        usable_face_count=usable_face_count,
    )


def _touching_outer_faces(config: SubgridConfig3D) -> frozenset[str]:
    return frozenset(
        face for face in FACE_ORIENTATIONS if not _face_is_internal(config, face)
    )


def _fixed_face_index(config: SubgridConfig3D, face: str, *, grid: str) -> int:
    if grid == "coarse":
        return {
            "x_lo": config.fi_lo,
            "x_hi": config.fi_hi - 1,
            "y_lo": config.fj_lo,
            "y_hi": config.fj_hi - 1,
            "z_lo": config.fk_lo,
            "z_hi": config.fk_hi - 1,
        }[face]
    return {
        "x_lo": 0,
        "x_hi": config.nx_f - 1,
        "y_lo": 0,
        "y_hi": config.ny_f - 1,
        "z_lo": 0,
        "z_hi": config.nz_f - 1,
    }[face]


def _full_axis_slice(config: SubgridConfig3D, axis: int, *, grid: str):
    if grid == "coarse":
        if axis == 0:
            return slice(config.fi_lo, config.fi_hi)
        if axis == 1:
            return slice(config.fj_lo, config.fj_hi)
        return slice(config.fk_lo, config.fk_hi)
    if axis == 0:
        return slice(0, config.nx_f)
    if axis == 1:
        return slice(0, config.ny_f)
    return slice(0, config.nz_f)


def _face_slice(config: SubgridConfig3D, face: str, *, grid: str) -> tuple:
    axis = FACE_ORIENTATIONS[face].normal_axis
    parts = []
    for ax in range(3):
        if ax == axis:
            parts.append(_fixed_face_index(config, face, grid=grid))
        else:
            parts.append(_full_axis_slice(config, ax, grid=grid))
    return tuple(parts)


def _get_component(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], name: str
) -> jnp.ndarray:
    return fields[_COMPONENT_INDEX[name]]


def _set_component(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    name: str,
    value: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    out = list(fields)
    out[_COMPONENT_INDEX[name]] = value
    return tuple(out)


def extract_tangential_e_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract tangential E traces for one oriented face."""

    sl = _face_slice(config, face, grid=grid)
    comps = FACE_ORIENTATIONS[face].tangential_e_components
    return tuple(_get_component(fields, comp)[sl] for comp in comps)


def extract_tangential_h_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract tangential H traces for one oriented face."""

    sl = _face_slice(config, face, grid=grid)
    comps = FACE_ORIENTATIONS[face].tangential_h_components
    return tuple(_get_component(fields, comp)[sl] for comp in comps)


def scatter_tangential_e_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tangential: tuple[jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scatter tangential E traces back to one oriented face."""

    sl = _face_slice(config, face, grid=grid)
    comps = FACE_ORIENTATIONS[face].tangential_e_components
    out = fields
    for comp, arr in zip(comps, tangential):
        field_arr = _get_component(out, comp)
        out = _set_component(out, comp, field_arr.at[sl].set(arr))
    return out


def scatter_tangential_h_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tangential: tuple[jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scatter tangential H traces back to one oriented face."""

    sl = _face_slice(config, face, grid=grid)
    comps = FACE_ORIENTATIONS[face].tangential_h_components
    out = fields
    for comp, arr in zip(comps, tangential):
        field_arr = _get_component(out, comp)
        out = _set_component(out, comp, field_arr.at[sl].set(arr))
    return out


def _face_interior_masks(
    coarse_shape: tuple[int, int], ratio: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coarse_mask = np.zeros(coarse_shape, dtype=np.float32)
    if coarse_shape[0] > 2 and coarse_shape[1] > 2:
        coarse_mask[1:-1, 1:-1] = 1.0

    fine_shape = (coarse_shape[0] * ratio, coarse_shape[1] * ratio)
    fine_mask = np.zeros(fine_shape, dtype=np.float32)
    if coarse_shape[0] > 2 and coarse_shape[1] > 2:
        fine_mask[ratio:-ratio, ratio:-ratio] = 1.0
    return jnp.asarray(coarse_mask), jnp.asarray(fine_mask)


def _apply_sat_pair_face(
    coarse_face: jnp.ndarray,
    fine_face: jnp.ndarray,
    ops: ZFaceOps,
    alpha_c: float,
    alpha_f: float,
    coarse_mask: jnp.ndarray,
    fine_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coarse_mismatch = restrict_face(fine_face, ops) - coarse_face
    fine_mismatch = prolong_face(coarse_face, ops) - fine_face
    return (
        coarse_face + alpha_c * coarse_mismatch * coarse_mask,
        fine_face + alpha_f * fine_mismatch * fine_mask,
    )


def _apply_operator_projected_skew_eh_face_helper(
    current_e_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_e_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_h_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_h_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Apply the private same-call operator-projected skew E/H face helper.

    The helper is deliberately internal to the experimental SBP-SAT path.  It
    consumes all eight tangential E/H face traces in one same-call local
    context, applies the face-normal orientation through ``normal_sign`` exactly
    once, then scatters only face-interior updates through the existing
    tangential face scatter helpers.  Edge/corner SAT ownership remains in the
    existing edge/corner paths.
    """

    alpha_c, alpha_f = sat_penalty_coefficients(config.ratio, config.tau)
    e_coarse = current_e_coarse
    e_fine = current_e_fine
    h_coarse = current_h_coarse
    h_fine = current_h_fine
    for face in _active_faces(config):
        orientation = FACE_ORIENTATIONS[face]
        ops = _get_face_ops(config, face)
        mortar = build_tensor_face_mortar(
            ops.coarse_shape,
            ratio=config.ratio,
            dx_c=np.sqrt(float(ops.coarse_area)),
        )
        coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, config.ratio)
        e_c_face = extract_tangential_e_face(e_coarse, config, face, grid="coarse")
        e_f_face = extract_tangential_e_face(e_fine, config, face, grid="fine")
        h_c_face = extract_tangential_h_face(h_coarse, config, face, grid="coarse")
        h_f_face = extract_tangential_h_face(h_fine, config, face, grid="fine")
        (
            e_c_t1,
            e_c_t2,
            h_c_t1,
            h_c_t2,
            e_f_t1,
            e_f_t2,
            h_f_t1,
            h_f_t2,
        ) = operator_projected_skew_eh_sat_face(
            ex_c=e_c_face[0],
            ey_c=e_c_face[1],
            hx_c=h_c_face[0],
            hy_c=h_c_face[1],
            ex_f=e_f_face[0],
            ey_f=e_f_face[1],
            hx_f=h_f_face[0],
            hy_f=h_f_face[1],
            mortar=mortar,
            alpha_c=alpha_c,
            alpha_f=alpha_f,
            coarse_mask=coarse_mask,
            fine_mask=fine_mask,
            normal_sign=orientation.normal_sign,
            include_scalar_projection=False,
        )
        e_coarse = scatter_tangential_e_face(
            e_coarse,
            (e_c_t1, e_c_t2),
            config,
            face,
            grid="coarse",
        )
        e_fine = scatter_tangential_e_face(
            e_fine,
            (e_f_t1, e_f_t2),
            config,
            face,
            grid="fine",
        )
        h_coarse = scatter_tangential_h_face(
            h_coarse,
            (h_c_t1, h_c_t2),
            config,
            face,
            grid="coarse",
        )
        h_fine = scatter_tangential_h_face(
            h_fine,
            (h_f_t1, h_f_t2),
            config,
            face,
            grid="fine",
        )
    return e_coarse, e_fine, h_coarse, h_fine


def _time_centered_face_trace_energy(
    coarse_e: tuple[jnp.ndarray, jnp.ndarray],
    fine_e: tuple[jnp.ndarray, jnp.ndarray],
    coarse_h: tuple[jnp.ndarray, jnp.ndarray],
    fine_h: tuple[jnp.ndarray, jnp.ndarray],
    ops: ZFaceOps,
    coarse_mask: jnp.ndarray,
    fine_mask: jnp.ndarray,
) -> jnp.ndarray:
    coarse_e_energy = jnp.sum(
        (coarse_e[0] ** 2 + coarse_e[1] ** 2) * ops.coarse_norm * coarse_mask
    )
    fine_e_energy = jnp.sum(
        (fine_e[0] ** 2 + fine_e[1] ** 2) * ops.fine_norm * fine_mask
    )
    coarse_h_energy = jnp.sum(
        (coarse_h[0] ** 2 + coarse_h[1] ** 2) * ops.coarse_norm * coarse_mask
    )
    fine_h_energy = jnp.sum(
        (fine_h[0] ** 2 + fine_h[1] ** 2) * ops.fine_norm * fine_mask
    )
    return 0.5 * EPS_0 * (coarse_e_energy + fine_e_energy) + 0.5 * MU_0 * (
        coarse_h_energy + fine_h_energy
    )


def _time_centered_face_interface_work(
    coarse_e: tuple[jnp.ndarray, jnp.ndarray],
    fine_e: tuple[jnp.ndarray, jnp.ndarray],
    coarse_h: tuple[jnp.ndarray, jnp.ndarray],
    fine_h: tuple[jnp.ndarray, jnp.ndarray],
    ops: ZFaceOps,
    coarse_mask: jnp.ndarray,
    *,
    dt: float,
    normal_sign: int,
) -> jnp.ndarray:
    coarse_sn = normal_sign * (coarse_e[0] * coarse_h[1] - coarse_e[1] * coarse_h[0])
    fine_sn = normal_sign * (fine_e[0] * fine_h[1] - fine_e[1] * fine_h[0])
    return dt * jnp.sum(
        (coarse_sn - restrict_face(fine_sn, ops)) * ops.coarse_norm * coarse_mask
    )


def _minimum_norm_quadratic_root(
    quadratic_a: jnp.ndarray,
    quadratic_b: jnp.ndarray,
    quadratic_c: jnp.ndarray,
) -> jnp.ndarray:
    eps = jnp.asarray(1.0e-36, dtype=quadratic_c.dtype)
    zero = jnp.asarray(0.0, dtype=quadratic_c.dtype)
    one = jnp.asarray(1.0, dtype=quadratic_c.dtype)

    b_safe = jnp.where(jnp.abs(quadratic_b) > eps, quadratic_b, one)
    linear_root = jnp.where(
        jnp.abs(quadratic_b) > eps,
        -quadratic_c / b_safe,
        zero,
    )

    discriminant = jnp.maximum(
        quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c,
        zero,
    )
    root = jnp.sqrt(discriminant)
    denominator = 2.0 * jnp.where(jnp.abs(quadratic_a) > eps, quadratic_a, one)
    quadratic_root_pos = (-quadratic_b + root) / denominator
    quadratic_root_neg = (-quadratic_b - root) / denominator
    quadratic_root = jnp.where(
        jnp.abs(quadratic_root_pos) <= jnp.abs(quadratic_root_neg),
        quadratic_root_pos,
        quadratic_root_neg,
    )

    return jnp.where(
        jnp.abs(quadratic_c) <= eps,
        zero,
        jnp.where(jnp.abs(quadratic_a) <= eps, linear_root, quadratic_root),
    )


def _time_centered_paired_face_amplitude(
    *,
    e_pre_coarse: tuple[jnp.ndarray, jnp.ndarray],
    e_pre_fine: tuple[jnp.ndarray, jnp.ndarray],
    e_post_coarse: tuple[jnp.ndarray, jnp.ndarray],
    e_post_fine: tuple[jnp.ndarray, jnp.ndarray],
    h_pre_coarse: tuple[jnp.ndarray, jnp.ndarray],
    h_pre_fine: tuple[jnp.ndarray, jnp.ndarray],
    h_post_coarse: tuple[jnp.ndarray, jnp.ndarray],
    h_post_fine: tuple[jnp.ndarray, jnp.ndarray],
    coarse_h2_direction: jnp.ndarray,
    fine_h2_direction: jnp.ndarray,
    ops: ZFaceOps,
    coarse_mask: jnp.ndarray,
    fine_mask: jnp.ndarray,
    dt: float,
    normal_sign: int,
) -> jnp.ndarray:
    before_energy = _time_centered_face_trace_energy(
        e_pre_coarse,
        e_pre_fine,
        h_pre_coarse,
        h_pre_fine,
        ops,
        coarse_mask,
        fine_mask,
    )
    after_energy_zero = _time_centered_face_trace_energy(
        e_post_coarse,
        e_post_fine,
        h_post_coarse,
        h_post_fine,
        ops,
        coarse_mask,
        fine_mask,
    )
    h_bar_coarse_zero = tuple(
        0.5 * (h_pre + h_post) for h_pre, h_post in zip(h_pre_coarse, h_post_coarse)
    )
    h_bar_fine_zero = tuple(
        0.5 * (h_pre + h_post) for h_pre, h_post in zip(h_pre_fine, h_post_fine)
    )
    interface_work_zero = _time_centered_face_interface_work(
        e_pre_coarse,
        e_pre_fine,
        h_bar_coarse_zero,
        h_bar_fine_zero,
        ops,
        coarse_mask,
        dt=dt,
        normal_sign=normal_sign,
    )

    quadratic_a = (
        0.5
        * MU_0
        * (
            jnp.sum(coarse_h2_direction**2 * ops.coarse_norm * coarse_mask)
            + jnp.sum(fine_h2_direction**2 * ops.fine_norm * fine_mask)
        )
    )
    energy_linear = MU_0 * (
        jnp.sum(h_post_coarse[1] * coarse_h2_direction * ops.coarse_norm * coarse_mask)
        + jnp.sum(h_post_fine[1] * fine_h2_direction * ops.fine_norm * fine_mask)
    )
    work_linear = (
        0.5
        * dt
        * jnp.sum(
            normal_sign
            * (
                e_pre_coarse[0] * coarse_h2_direction
                - restrict_face(e_pre_fine[0] * fine_h2_direction, ops)
            )
            * ops.coarse_norm
            * coarse_mask
        )
    )
    quadratic_b = energy_linear + work_linear
    quadratic_c = after_energy_zero - before_energy + interface_work_zero
    return _minimum_norm_quadratic_root(quadratic_a, quadratic_b, quadratic_c)


def _apply_time_centered_paired_face_helper(
    current_h_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    current_h_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    *,
    h_pre_sat_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    h_pre_sat_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    h_post_sat_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    h_post_sat_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    e_pre_sat_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    e_pre_sat_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    e_post_sat_coarse: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    e_post_sat_fine: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Apply the private same-call centered-H paired-face correction.

    The helper consumes only production-local face traces already present in the
    SBP-SAT step functions: H immediately before/after H SAT, and E immediately
    before/after E SAT.  It intentionally has no public switch or hook route.
    """

    coarse = current_h_coarse
    fine = current_h_fine
    for face in _active_faces(config):
        orientation = FACE_ORIENTATIONS[face]
        ops = _get_face_ops(config, face)
        coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, config.ratio)
        coarse_h_mode = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * coarse_mask
        fine_h_mode = prolong_face(coarse_h_mode, ops)

        h_pre_coarse = extract_tangential_h_face(
            h_pre_sat_coarse, config, face, grid="coarse"
        )
        h_pre_fine = extract_tangential_h_face(
            h_pre_sat_fine, config, face, grid="fine"
        )
        h_post_coarse = extract_tangential_h_face(
            h_post_sat_coarse, config, face, grid="coarse"
        )
        h_post_fine = extract_tangential_h_face(
            h_post_sat_fine, config, face, grid="fine"
        )
        e_pre_coarse = extract_tangential_e_face(
            e_pre_sat_coarse, config, face, grid="coarse"
        )
        e_pre_fine = extract_tangential_e_face(
            e_pre_sat_fine, config, face, grid="fine"
        )
        e_post_coarse = extract_tangential_e_face(
            e_post_sat_coarse, config, face, grid="coarse"
        )
        e_post_fine = extract_tangential_e_face(
            e_post_sat_fine, config, face, grid="fine"
        )

        amplitude = _time_centered_paired_face_amplitude(
            e_pre_coarse=e_pre_coarse,
            e_pre_fine=e_pre_fine,
            e_post_coarse=e_post_coarse,
            e_post_fine=e_post_fine,
            h_pre_coarse=h_pre_coarse,
            h_pre_fine=h_pre_fine,
            h_post_coarse=h_post_coarse,
            h_post_fine=h_post_fine,
            coarse_h2_direction=coarse_h_mode,
            fine_h2_direction=fine_h_mode,
            ops=ops,
            coarse_mask=coarse_mask,
            fine_mask=fine_mask,
            dt=config.dt,
            normal_sign=orientation.normal_sign,
        )
        amplitude = amplitude * _TIME_CENTERED_HELPER_RELAXATION
        current_h_coarse_face = extract_tangential_h_face(
            coarse, config, face, grid="coarse"
        )
        current_h_fine_face = extract_tangential_h_face(fine, config, face, grid="fine")
        corrected_h_coarse_face = (
            current_h_coarse_face[0],
            current_h_coarse_face[1] + amplitude * coarse_h_mode,
        )
        corrected_h_fine_face = (
            current_h_fine_face[0],
            current_h_fine_face[1] + amplitude * fine_h_mode,
        )
        coarse = scatter_tangential_h_face(
            coarse, corrected_h_coarse_face, config, face, grid="coarse"
        )
        fine = scatter_tangential_h_face(
            fine, corrected_h_fine_face, config, face, grid="fine"
        )
    return coarse, fine


def _edge_slice(config: SubgridConfig3D, edge: str, *, grid: str) -> tuple:
    meta = EDGE_ORIENTATIONS[edge]
    axis = meta.varying_axis
    parts = [None, None, None]
    for ax in range(3):
        if ax == axis:
            parts[ax] = _full_axis_slice(config, ax, grid=grid)
        else:
            face = (
                meta.fixed_faces[0]
                if FACE_ORIENTATIONS[meta.fixed_faces[0]].normal_axis == ax
                else meta.fixed_faces[1]
            )
            parts[ax] = _fixed_face_index(config, face, grid=grid)
    return tuple(parts)


def _extract_edge_component(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    edge: str,
    *,
    grid: str,
    field_type: str,
) -> jnp.ndarray:
    meta = EDGE_ORIENTATIONS[edge]
    comp = meta.e_component if field_type == "e" else meta.h_component
    return _get_component(fields, comp)[_edge_slice(config, edge, grid=grid)]


def _scatter_edge_component(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    values: jnp.ndarray,
    config: SubgridConfig3D,
    edge: str,
    *,
    grid: str,
    field_type: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    meta = EDGE_ORIENTATIONS[edge]
    comp = meta.e_component if field_type == "e" else meta.h_component
    arr = _get_component(fields, comp)
    return _set_component(
        fields, comp, arr.at[_edge_slice(config, edge, grid=grid)].set(values)
    )


def _edge_interior_masks(
    coarse_size: int, ratio: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coarse_mask = np.zeros((coarse_size,), dtype=np.float32)
    if coarse_size > 2:
        coarse_mask[1:-1] = 1.0
    fine_mask = np.zeros((coarse_size * ratio,), dtype=np.float32)
    if coarse_size > 2:
        fine_mask[ratio:-ratio] = 1.0
    return jnp.asarray(coarse_mask), jnp.asarray(fine_mask)


def _apply_sat_pair_edge(
    coarse_line: jnp.ndarray,
    fine_line: jnp.ndarray,
    ops: EdgeOps,
    alpha_c: float,
    alpha_f: float,
    coarse_mask: jnp.ndarray,
    fine_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coarse_mismatch = restrict_edge(fine_line, ops) - coarse_line
    fine_mismatch = prolong_edge(coarse_line, ops) - fine_line
    return (
        coarse_line + alpha_c * coarse_mismatch * coarse_mask,
        fine_line + alpha_f * fine_mismatch * fine_mask,
    )


def _corner_index(
    config: SubgridConfig3D, corner: str, *, grid: str
) -> tuple[int, int, int]:
    meta = CORNER_ORIENTATIONS[corner]
    return tuple(
        _fixed_face_index(config, face, grid=grid) for face in meta.fixed_faces
    )


def _apply_sat_pair_scalar(
    coarse_value: jnp.ndarray,
    fine_value: jnp.ndarray,
    alpha_c: float,
    alpha_f: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (
        coarse_value + alpha_c * (fine_value - coarse_value),
        fine_value + alpha_f * (coarse_value - fine_value),
    )


def _zero_coarse_overlap_interior(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Suppress strict coarse-grid interior state inside the fine box."""
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni, nj, nk = _region_shape(config)
    if ni <= 2 or nj <= 2 or nk <= 2:
        return fields
    interior = (
        slice(fi + 1, fi + ni - 1),
        slice(fj + 1, fj + nj - 1),
        slice(fk + 1, fk + nk - 1),
    )
    return tuple(field.at[interior].set(0.0) for field in fields)


def _active_edges(config: SubgridConfig3D) -> tuple[str, ...]:
    faces = set(_active_faces(config))
    return tuple(
        edge
        for edge, meta in EDGE_ORIENTATIONS.items()
        if meta.fixed_faces[0] in faces and meta.fixed_faces[1] in faces
    )


def _active_corners(config: SubgridConfig3D) -> tuple[str, ...]:
    faces = set(_active_faces(config))
    return tuple(
        corner
        for corner, meta in CORNER_ORIENTATIONS.items()
        if all(face in faces for face in meta.fixed_faces)
    )


def apply_sat_h_interfaces(
    coarse_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fine_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Apply SAT coupling to tangential H traces on active faces/edges/corners."""

    alpha_c, alpha_f = sat_penalty_coefficients(config.ratio, config.tau)
    coarse = coarse_fields
    fine = fine_fields
    for face in _active_faces(config):
        ops = _get_face_ops(config, face)
        coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, config.ratio)
        hx_c_face, hy_c_face = extract_tangential_h_face(
            coarse, config, face, grid="coarse"
        )
        hx_f_face, hy_f_face = extract_tangential_h_face(
            fine, config, face, grid="fine"
        )
        hx_c_face, hx_f_face = _apply_sat_pair_face(
            hx_c_face, hx_f_face, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        hy_c_face, hy_f_face = _apply_sat_pair_face(
            hy_c_face, hy_f_face, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        coarse = scatter_tangential_h_face(
            coarse, (hx_c_face, hy_c_face), config, face, grid="coarse"
        )
        fine = scatter_tangential_h_face(
            fine, (hx_f_face, hy_f_face), config, face, grid="fine"
        )

    for edge in _active_edges(config):
        ops = _get_edge_ops(config, edge)
        coarse_mask, fine_mask = _edge_interior_masks(ops.coarse_size, config.ratio)
        hc = _extract_edge_component(
            coarse, config, edge, grid="coarse", field_type="h"
        )
        hf = _extract_edge_component(fine, config, edge, grid="fine", field_type="h")
        hc, hf = _apply_sat_pair_edge(
            hc, hf, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        coarse = _scatter_edge_component(
            coarse, hc, config, edge, grid="coarse", field_type="h"
        )
        fine = _scatter_edge_component(
            fine, hf, config, edge, grid="fine", field_type="h"
        )

    for corner in _active_corners(config):
        c_idx = _corner_index(config, corner, grid="coarse")
        f_idx = _corner_index(config, corner, grid="fine")
        for comp in ("hx", "hy", "hz"):
            c_arr = _get_component(coarse, comp)
            f_arr = _get_component(fine, comp)
            c_val, f_val = _apply_sat_pair_scalar(
                c_arr[c_idx], f_arr[f_idx], alpha_c, alpha_f
            )
            coarse = _set_component(coarse, comp, c_arr.at[c_idx].set(c_val))
            fine = _set_component(fine, comp, f_arr.at[f_idx].set(f_val))
    return coarse, fine


def apply_sat_e_interfaces(
    coarse_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fine_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Apply SAT coupling to tangential E traces on active faces/edges/corners."""

    alpha_c, alpha_f = sat_penalty_coefficients(config.ratio, config.tau)
    coarse = coarse_fields
    fine = fine_fields
    for face in _active_faces(config):
        ops = _get_face_ops(config, face)
        coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, config.ratio)
        ex_c_face, ey_c_face = extract_tangential_e_face(
            coarse, config, face, grid="coarse"
        )
        ex_f_face, ey_f_face = extract_tangential_e_face(
            fine, config, face, grid="fine"
        )
        ex_c_face, ex_f_face = _apply_sat_pair_face(
            ex_c_face, ex_f_face, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        ey_c_face, ey_f_face = _apply_sat_pair_face(
            ey_c_face, ey_f_face, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        coarse = scatter_tangential_e_face(
            coarse, (ex_c_face, ey_c_face), config, face, grid="coarse"
        )
        fine = scatter_tangential_e_face(
            fine, (ex_f_face, ey_f_face), config, face, grid="fine"
        )

    for edge in _active_edges(config):
        ops = _get_edge_ops(config, edge)
        coarse_mask, fine_mask = _edge_interior_masks(ops.coarse_size, config.ratio)
        ec = _extract_edge_component(
            coarse, config, edge, grid="coarse", field_type="e"
        )
        ef = _extract_edge_component(fine, config, edge, grid="fine", field_type="e")
        ec, ef = _apply_sat_pair_edge(
            ec, ef, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        coarse = _scatter_edge_component(
            coarse, ec, config, edge, grid="coarse", field_type="e"
        )
        fine = _scatter_edge_component(
            fine, ef, config, edge, grid="fine", field_type="e"
        )

    for corner in _active_corners(config):
        c_idx = _corner_index(config, corner, grid="coarse")
        f_idx = _corner_index(config, corner, grid="fine")
        for comp in ("ex", "ey", "ez"):
            c_arr = _get_component(coarse, comp)
            f_arr = _get_component(fine, comp)
            c_val, f_val = _apply_sat_pair_scalar(
                c_arr[c_idx], f_arr[f_idx], alpha_c, alpha_f
            )
            coarse = _set_component(coarse, comp, c_arr.at[c_idx].set(c_val))
            fine = _set_component(fine, comp, f_arr.at[f_idx].set(f_val))
    return coarse, fine


def _shared_node_coupling_h_3d(state_c_fields, state_f_fields, config):
    """Compatibility wrapper for legacy imports.

    The current low-level lane uses oriented-face H SAT coupling.
    """

    return apply_sat_h_interfaces(state_c_fields, state_f_fields, config)


def _shared_node_coupling_3d(state_c_fields, state_f_fields, config):
    """Compatibility wrapper for legacy imports.

    The current low-level lane uses oriented-face E SAT coupling.
    """

    return apply_sat_e_interfaces(state_c_fields, state_f_fields, config)


def step_subgrid_3d_with_cpml(
    state: SubgridState3D,
    config: SubgridConfig3D,
    *,
    cpml_params,
    cpml_state,
    grid_c,
    cpml_axes: str,
    mats_c=None,
    mats_f=None,
    pec_mask_c=None,
    pec_mask_f=None,
    outer_pec_faces: frozenset[str] = frozenset(),
    outer_pmc_faces: frozenset[str] = frozenset(),
    periodic: tuple[bool, bool, bool] = (False, False, False),
    fine_periodic: tuple[bool, bool, bool] = (False, False, False),
    private_post_h_hook=None,
    private_post_e_hook=None,
):
    """Advance one subgrid step with CPML on the coarse outer boundary.

    This is the first bounded absorbing-coexistence path: CPML is applied only
    to the coarse domain boundary, while the fine box is required by API
    preflight to remain inside the physical domain and outside the active CPML
    pad plus one coarse-cell guard.  The fine grid therefore still uses only
    SBP-SAT interface coupling and any physical PEC faces it actually touches.
    """

    from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h
    from rfx.boundaries.pmc import apply_pmc_faces
    from rfx.boundaries.pec import apply_pec_faces, apply_pec_mask
    from rfx.core.yee import FDTDState, update_e, update_h

    if outer_pec_faces or outer_pmc_faces:
        raise ValueError(
            "SBP-SAT subgridding does not yet support mixed reflector + CPML "
            "absorbing faces"
        )
    if any(periodic) or any(fine_periodic):
        raise ValueError(
            "SBP-SAT subgridding does not yet support mixed periodic + CPML "
            "absorbing faces"
        )

    if mats_c is None:
        mats_c = _make_mats(state.ex_c.shape)
    if mats_f is None:
        mats_f = _make_mats(state.ex_f.shape)
    private_interface_owner_state = _ensure_private_interface_owner_state(state, config)

    coarse_h = FDTDState(
        ex=state.ex_c,
        ey=state.ey_c,
        ez=state.ez_c,
        hx=state.hx_c,
        hy=state.hy_c,
        hz=state.hz_c,
        step=jnp.array(0, dtype=jnp.int32),
    )
    coarse_h = update_h(coarse_h, mats_c, config.dt, config.dx_c, periodic=periodic)
    coarse_h, cpml_new = apply_cpml_h(
        coarse_h, cpml_params, cpml_state, grid_c, cpml_axes, materials=mats_c
    )
    if outer_pmc_faces:
        coarse_h = apply_pmc_faces(coarse_h, outer_pmc_faces)
    hx_c, hy_c, hz_c = coarse_h.hx, coarse_h.hy, coarse_h.hz

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
        periodic=fine_periodic,
        pmc_faces=frozenset(
            face for face in _touching_outer_faces(config) if face in outer_pmc_faces
        ),
    )
    if private_post_h_hook is not None:
        hook_state = private_post_h_hook(
            SubgridState3D(
                ex_c=state.ex_c,
                ey_c=state.ey_c,
                ez_c=state.ez_c,
                hx_c=hx_c,
                hy_c=hy_c,
                hz_c=hz_c,
                ex_f=state.ex_f,
                ey_f=state.ey_f,
                ez_f=state.ez_f,
                hx_f=hx_f,
                hy_f=hy_f,
                hz_f=hz_f,
                step=state.step,
                private_interface_owner_state=private_interface_owner_state,
            )
        )
        hx_f, hy_f, hz_f = hook_state.hx_f, hook_state.hy_f, hook_state.hz_f
        private_interface_owner_state = _ensure_private_interface_owner_state(
            hook_state,
            config,
        )
    h_pre_sat_coarse = (hx_c, hy_c, hz_c)
    h_pre_sat_fine = (hx_f, hy_f, hz_f)
    (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f) = apply_sat_h_interfaces(
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        config,
    )
    h_post_sat_coarse = (hx_c, hy_c, hz_c)
    h_post_sat_fine = (hx_f, hy_f, hz_f)
    hx_c, hy_c, hz_c = _zero_coarse_overlap_interior((hx_c, hy_c, hz_c), config)

    coarse_e = FDTDState(
        ex=state.ex_c,
        ey=state.ey_c,
        ez=state.ez_c,
        hx=hx_c,
        hy=hy_c,
        hz=hz_c,
        step=jnp.array(0, dtype=jnp.int32),
    )
    coarse_e = update_e(coarse_e, mats_c, config.dt, config.dx_c, periodic=periodic)
    coarse_e, cpml_new = apply_cpml_e(
        coarse_e, cpml_params, cpml_new, grid_c, cpml_axes, materials=mats_c
    )
    if outer_pec_faces:
        coarse_e = apply_pec_faces(coarse_e, outer_pec_faces)
    if pec_mask_c is not None:
        coarse_e = apply_pec_mask(coarse_e, pec_mask_c)
    ex_c, ey_c, ez_c = coarse_e.ex, coarse_e.ey, coarse_e.ez

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
        boundary_axes=None,
        boundary_faces=frozenset(
            face for face in _touching_outer_faces(config) if face in outer_pec_faces
        ),
        periodic=fine_periodic,
    )
    if private_post_e_hook is not None:
        hook_state = private_post_e_hook(
            SubgridState3D(
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
                step=state.step,
                private_interface_owner_state=private_interface_owner_state,
            )
        )
        ex_f, ey_f, ez_f = hook_state.ex_f, hook_state.ey_f, hook_state.ez_f
        private_interface_owner_state = _ensure_private_interface_owner_state(
            hook_state,
            config,
        )
    e_pre_sat_coarse = (ex_c, ey_c, ez_c)
    e_pre_sat_fine = (ex_f, ey_f, ez_f)
    (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f) = apply_sat_e_interfaces(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        config,
    )
    e_post_sat_coarse = (ex_c, ey_c, ez_c)
    e_post_sat_fine = (ex_f, ey_f, ez_f)
    (
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
    ) = _apply_operator_projected_skew_eh_face_helper(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        config,
    )
    e_post_sat_coarse = (ex_c, ey_c, ez_c)
    e_post_sat_fine = (ex_f, ey_f, ez_f)
    h_post_sat_coarse = (hx_c, hy_c, hz_c)
    h_post_sat_fine = (hx_f, hy_f, hz_f)
    (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f) = _apply_time_centered_paired_face_helper(
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        h_pre_sat_coarse=h_pre_sat_coarse,
        h_pre_sat_fine=h_pre_sat_fine,
        h_post_sat_coarse=h_post_sat_coarse,
        h_post_sat_fine=h_post_sat_fine,
        e_pre_sat_coarse=e_pre_sat_coarse,
        e_pre_sat_fine=e_pre_sat_fine,
        e_post_sat_coarse=e_post_sat_coarse,
        e_post_sat_fine=e_post_sat_fine,
        config=config,
    )
    (
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
    ) = _apply_observable_proxy_modal_retry_face_helper(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        private_interface_owner_state,
        config,
    )
    private_interface_owner_state = _stage_private_time_aligned_owner_packets(
        private_interface_owner_state
    )
    private_interface_owner_state = _update_private_source_owner_state_from_scan(
        private_interface_owner_state,
        e_source_coarse=(ex_c, ey_c, ez_c),
        e_source_fine=(ex_f, ey_f, ez_f),
        h_source_coarse=(hx_c, hy_c, hz_c),
        h_source_fine=(hx_f, hy_f, hz_f),
        config=config,
    )
    (
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
    ) = _apply_propagation_aware_modal_retry_face_helper(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        private_interface_owner_state,
        config,
    )
    private_interface_owner_state = _update_private_interface_owner_state_from_scan(
        private_interface_owner_state,
        e_post_sat_coarse=(ex_c, ey_c, ez_c),
        e_post_sat_fine=(ex_f, ey_f, ez_f),
        h_post_sat_coarse=(hx_c, hy_c, hz_c),
        h_post_sat_fine=(hx_f, hy_f, hz_f),
        config=config,
    )
    ex_c, ey_c, ez_c = _zero_coarse_overlap_interior((ex_c, ey_c, ez_c), config)

    return (
        SubgridState3D(
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
            private_interface_owner_state=_advance_private_interface_owner_state(
                private_interface_owner_state
            ),
        ),
        cpml_new,
    )


def step_subgrid_3d(
    state: SubgridState3D,
    config: SubgridConfig3D,
    *,
    mats_c=None,
    mats_f=None,
    pec_mask_c=None,
    pec_mask_f=None,
    outer_pec_faces: frozenset[str] = frozenset(
        {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}
    ),
    outer_pmc_faces: frozenset[str] = frozenset(),
    periodic: tuple[bool, bool, bool] = (False, False, False),
    fine_periodic: tuple[bool, bool, bool] = (False, False, False),
    private_post_h_hook=None,
    private_post_e_hook=None,
) -> SubgridState3D:
    """Advance the current all-PEC subgrid lane by one timestep."""

    private_interface_owner_state = _ensure_private_interface_owner_state(state, config)

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
        periodic=periodic,
        pmc_faces=outer_pmc_faces,
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
        periodic=fine_periodic,
        pmc_faces=frozenset(
            face for face in _touching_outer_faces(config) if face in outer_pmc_faces
        ),
    )
    if private_post_h_hook is not None:
        hook_state = private_post_h_hook(
            SubgridState3D(
                ex_c=state.ex_c,
                ey_c=state.ey_c,
                ez_c=state.ez_c,
                hx_c=hx_c,
                hy_c=hy_c,
                hz_c=hz_c,
                ex_f=state.ex_f,
                ey_f=state.ey_f,
                ez_f=state.ez_f,
                hx_f=hx_f,
                hy_f=hy_f,
                hz_f=hz_f,
                step=state.step,
                private_interface_owner_state=private_interface_owner_state,
            )
        )
        hx_f, hy_f, hz_f = hook_state.hx_f, hook_state.hy_f, hook_state.hz_f
        private_interface_owner_state = _ensure_private_interface_owner_state(
            hook_state,
            config,
        )
    h_pre_sat_coarse = (hx_c, hy_c, hz_c)
    h_pre_sat_fine = (hx_f, hy_f, hz_f)
    (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f) = apply_sat_h_interfaces(
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        config,
    )
    h_post_sat_coarse = (hx_c, hy_c, hz_c)
    h_post_sat_fine = (hx_f, hy_f, hz_f)
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
        boundary_axes=None,
        boundary_faces=outer_pec_faces,
        periodic=periodic,
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
        boundary_axes=None,
        boundary_faces=frozenset(
            face for face in _touching_outer_faces(config) if face in outer_pec_faces
        ),
        periodic=fine_periodic,
    )
    if private_post_e_hook is not None:
        hook_state = private_post_e_hook(
            SubgridState3D(
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
                step=state.step,
                private_interface_owner_state=private_interface_owner_state,
            )
        )
        ex_f, ey_f, ez_f = hook_state.ex_f, hook_state.ey_f, hook_state.ez_f
        private_interface_owner_state = _ensure_private_interface_owner_state(
            hook_state,
            config,
        )
    e_pre_sat_coarse = (ex_c, ey_c, ez_c)
    e_pre_sat_fine = (ex_f, ey_f, ez_f)
    (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f) = apply_sat_e_interfaces(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        config,
    )
    e_post_sat_coarse = (ex_c, ey_c, ez_c)
    e_post_sat_fine = (ex_f, ey_f, ez_f)
    (
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
    ) = _apply_operator_projected_skew_eh_face_helper(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        config,
    )
    e_post_sat_coarse = (ex_c, ey_c, ez_c)
    e_post_sat_fine = (ex_f, ey_f, ez_f)
    h_post_sat_coarse = (hx_c, hy_c, hz_c)
    h_post_sat_fine = (hx_f, hy_f, hz_f)
    (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f) = _apply_time_centered_paired_face_helper(
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        h_pre_sat_coarse=h_pre_sat_coarse,
        h_pre_sat_fine=h_pre_sat_fine,
        h_post_sat_coarse=h_post_sat_coarse,
        h_post_sat_fine=h_post_sat_fine,
        e_pre_sat_coarse=e_pre_sat_coarse,
        e_pre_sat_fine=e_pre_sat_fine,
        e_post_sat_coarse=e_post_sat_coarse,
        e_post_sat_fine=e_post_sat_fine,
        config=config,
    )
    (
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
    ) = _apply_observable_proxy_modal_retry_face_helper(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        private_interface_owner_state,
        config,
    )
    private_interface_owner_state = _stage_private_time_aligned_owner_packets(
        private_interface_owner_state
    )
    private_interface_owner_state = _update_private_source_owner_state_from_scan(
        private_interface_owner_state,
        e_source_coarse=(ex_c, ey_c, ez_c),
        e_source_fine=(ex_f, ey_f, ez_f),
        h_source_coarse=(hx_c, hy_c, hz_c),
        h_source_fine=(hx_f, hy_f, hz_f),
        config=config,
    )
    (
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
    ) = _apply_propagation_aware_modal_retry_face_helper(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        private_interface_owner_state,
        config,
    )
    private_interface_owner_state = _update_private_interface_owner_state_from_scan(
        private_interface_owner_state,
        e_post_sat_coarse=(ex_c, ey_c, ez_c),
        e_post_sat_fine=(ex_f, ey_f, ez_f),
        h_post_sat_coarse=(hx_c, hy_c, hz_c),
        h_post_sat_fine=(hx_f, hy_f, hz_f),
        config=config,
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
        private_interface_owner_state=_advance_private_interface_owner_state(
            private_interface_owner_state
        ),
    )


def compute_energy_3d(state: SubgridState3D, config: SubgridConfig3D) -> float:
    """Total energy with coarse/fine overlap counted once."""

    dv_c = config.dx_c**3
    dv_f = config.dx_f**3
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni, nj, nk = _region_shape(config)

    mask = jnp.ones(state.ex_c.shape, dtype=jnp.bool_)
    mask = mask.at[fi : fi + ni, fj : fj + nj, fk : fk + nk].set(False)

    e_c = (
        float(
            jnp.sum(jnp.where(mask, state.ex_c**2 + state.ey_c**2 + state.ez_c**2, 0.0))
        )
        * EPS_0
        * dv_c
        + float(
            jnp.sum(jnp.where(mask, state.hx_c**2 + state.hy_c**2 + state.hz_c**2, 0.0))
        )
        * MU_0
        * dv_c
    )
    e_f = (
        float(jnp.sum(state.ex_f**2 + state.ey_f**2 + state.ez_f**2)) * EPS_0 * dv_f
        + float(jnp.sum(state.hx_f**2 + state.hy_f**2 + state.hz_f**2)) * MU_0 * dv_f
    )
    return e_c + e_f
