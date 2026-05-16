"""JIT-compiled subgridded FDTD runner via jax.lax.scan.

Replaces the Python-loop runner (runner.py) with a fully JIT-compiled
version that achieves 50-100x speedup. Both coarse and fine grids
are updated per step with SBP-SAT interface coupling.  The public
``Simulation.add_refinement`` path is a full-x/y z-slab refinement, so only the
two z faces are artificial SAT interfaces; x/y faces remain physical
boundaries.

Handles both CPML (absorbing) and PEC (reflecting) coarse-grid
boundaries. When cpml_layers == 0 the CPML subsystem is skipped
entirely and PEC is applied on all domain faces.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state,
    update_h, update_e,
)
from rfx.boundaries.pec import apply_pec, apply_pec_faces, apply_pec_mask
from rfx.grid import Grid
from rfx.subgridding.sbp_sat_3d import (
    SubgridConfig3D,
    _shared_node_coupling_3d,
    _shared_node_coupling_h_3d,
)
from rfx.subgridding.material_sat import interface_pair_deltas
from rfx.core.yee import EPS_0, MU_0


class SubgridResult(NamedTuple):
    """Result from JIT-compiled subgridded simulation."""
    state_c: FDTDState
    state_f: FDTDState
    time_series: jnp.ndarray
    config: SubgridConfig3D
    dt: float
    time_series_c: jnp.ndarray | None = None
    lumped_sparam_v_dft_f: jnp.ndarray | None = None
    lumped_sparam_i_dft_f: jnp.ndarray | None = None
    lumped_sparam_freqs_f: jnp.ndarray | None = None
    lumped_sparam_impedances_f: jnp.ndarray | None = None
    lumped_sparam_cell_counts_f: jnp.ndarray | None = None
    ntff_data_f: object | None = None
    ntff_box_f: object | None = None


class ZSlabCouplingPlan(NamedTuple):
    """Static coarse-plane ownership plan for z-slab coupling."""

    k_lo_c: int
    k_hi_c: int
    couple_zlo: bool
    couple_zhi: bool


def _uses_endpoint_fine_shape(config) -> bool:
    """Return whether fine dimensions follow endpoint-node shape math."""
    ni = config.fi_hi - config.fi_lo
    nj = config.fj_hi - config.fj_lo
    nk = config.fk_hi - config.fk_lo
    return (
        config.nx_f == (ni - 1) * config.ratio + 1
        and config.ny_f == (nj - 1) * config.ratio + 1
        and config.nz_f == (nk - 1) * config.ratio + 1
    )


def _trapezoid_weights(n: int, dtype):
    """Return 1-D endpoint-node trapezoid quadrature weights."""
    weights = jnp.ones((n,), dtype=dtype)
    if n > 1:
        weights = weights.at[0].set(0.5)
        weights = weights.at[-1].set(0.5)
    return weights


def _restrict_axis_linear_adjoint(arr, n_coarse: int, ratio: int, axis: int):
    """Restrict one node-aligned axis with the adjoint of linear prolongation.

    The inner product uses endpoint-node trapezoid weights and the physical
    spacing ratio ``h_f / h_c = 1 / ratio``.  This makes restriction compatible
    with ``_prolong_axis_linear`` instead of merely sampling colocated nodes.
    """
    arr_m = jnp.moveaxis(arr, axis, 0)
    n_fine = arr_m.shape[0]
    dtype = arr_m.dtype
    if n_coarse < 2:
        return jnp.moveaxis(jnp.mean(arr_m, axis=0, keepdims=True), 0, axis)

    idx = jnp.arange(n_fine)
    base = jnp.minimum(idx // ratio, n_coarse - 2)
    frac = (idx - base * ratio).astype(dtype) / float(ratio)
    fine_w = _trapezoid_weights(n_fine, dtype)
    coarse_w = _trapezoid_weights(n_coarse, dtype)
    lo_coeff = (1.0 - frac) * fine_w / (float(ratio) * coarse_w[base])
    hi_coeff = frac * fine_w / (float(ratio) * coarse_w[base + 1])

    coeff_shape = (n_fine,) + (1,) * (arr_m.ndim - 1)
    out = jnp.zeros((n_coarse,) + arr_m.shape[1:], dtype=dtype)
    out = out.at[base].add(lo_coeff.reshape(coeff_shape) * arr_m)
    out = out.at[base + 1].add(hi_coeff.reshape(coeff_shape) * arr_m)
    return jnp.moveaxis(out, 0, axis)


def _restrict_node_aligned_2d(fine_face, n_coarse_i: int, n_coarse_j: int, ratio: int):
    """Restrict a node-aligned fine face with the prolongation adjoint.

    The public full-x/y z-slab runner uses endpoint-node faces.  For a coarse
    face with ``N`` endpoint nodes, the matching fine face has
    ``(N - 1) * ratio + 1`` nodes.  Restriction must therefore be the adjoint of
    the bilinear endpoint-node prolongation under the face quadrature; sampling
    only colocated nodes preserves coordinates but is not an energy-compatible
    SAT projection.
    """
    tmp = _restrict_axis_linear_adjoint(fine_face, n_coarse_i, ratio, axis=0)
    return _restrict_axis_linear_adjoint(tmp, n_coarse_j, ratio, axis=1)


def _restrict_node_aligned_2d_sample(fine_face, n_coarse_i: int, n_coarse_j: int, ratio: int):
    """Sample colocated fine nodes on a z-face for projection diagnostics."""
    return fine_face[::ratio, ::ratio][:n_coarse_i, :n_coarse_j]


def _prolong_axis_linear(arr, n_fine: int, ratio: int, axis: int):
    """Linearly interpolate a node-aligned coarse array along one axis."""
    n_coarse = arr.shape[axis]
    if n_coarse < 2:
        return jnp.repeat(arr, ratio, axis=axis)[:n_fine]
    idx = jnp.arange(n_fine)
    base = jnp.minimum(idx // ratio, n_coarse - 2)
    frac = (idx - base * ratio).astype(arr.dtype) / float(ratio)
    lo = jnp.take(arr, base, axis=axis)
    hi = jnp.take(arr, base + 1, axis=axis)
    shape = [1] * arr.ndim
    shape[axis] = n_fine
    frac = frac.reshape(shape)
    return (1.0 - frac) * lo + frac * hi


def _prolong_node_aligned_2d(coarse_face, n_fine_i: int, n_fine_j: int, ratio: int):
    """Prolong a coarse face to a node-aligned fine face by bilinear interpolation."""
    tmp = _prolong_axis_linear(coarse_face, n_fine_i, ratio, axis=0)
    return _prolong_axis_linear(tmp, n_fine_j, ratio, axis=1)


MODAL_TRACE_DELTA_FILTERS = frozenset(
    {
        "modal_linear_delta",
        "modal_xy_linear_delta",
        "modal_mixed_quadratic_delta",
        "modal_xy_plus_mixed_delta",
        "modal_low2_delta",
    }
)


def _modal_axis_basis(n: int, kind: str, dtype):
    """Return a normalized low-order modal basis vector for one face axis."""
    constant = jnp.ones((n,), dtype=dtype) / jnp.sqrt(jnp.asarray(n, dtype=dtype))
    if kind == "constant":
        return constant
    coord = jnp.linspace(-1.0, 1.0, n, dtype=dtype)
    linear = coord - jnp.sum(coord * constant) * constant
    linear = linear / jnp.maximum(
        jnp.sqrt(jnp.sum(linear * linear)),
        jnp.asarray(1e-30, dtype=dtype),
    )
    if kind == "linear":
        return linear
    if kind == "quadratic":
        quadratic = coord * coord
        quadratic = quadratic - jnp.sum(quadratic * constant) * constant
        quadratic = quadratic - jnp.sum(quadratic * linear) * linear
        return quadratic / jnp.maximum(
            jnp.sqrt(jnp.sum(quadratic * quadratic)),
            jnp.asarray(1e-30, dtype=dtype),
        )
    raise ValueError(f"unknown modal axis basis kind={kind!r}")


def _modal_face_component(face_delta, y_kind: str, x_kind: str):
    """Project a 2-D face delta onto one separable low-order modal component."""
    y_basis = _modal_axis_basis(face_delta.shape[0], y_kind, face_delta.dtype)
    x_basis = _modal_axis_basis(face_delta.shape[1], x_kind, face_delta.dtype)
    basis = y_basis[:, None] * x_basis[None, :]
    return jnp.sum(face_delta * basis) * basis


def _modal_filter_face_delta(face_delta, mode: str):
    """Return a low-order modal subset of a z-face trace delta.

    This diagnostic filter is based only on the current simulated pre/post
    trace delta.  It does not use reference fields or probe-error artifacts.
    """
    specs_by_mode = {
        "modal_linear_delta": (
            ("constant", "linear"),
            ("linear", "constant"),
            ("linear", "linear"),
        ),
        "modal_xy_linear_delta": (("linear", "linear"),),
        "modal_mixed_quadratic_delta": (
            ("linear", "quadratic"),
            ("quadratic", "linear"),
        ),
        "modal_xy_plus_mixed_delta": (
            ("linear", "linear"),
            ("linear", "quadratic"),
            ("quadratic", "linear"),
        ),
        "modal_low2_delta": (
            ("constant", "linear"),
            ("linear", "constant"),
            ("linear", "linear"),
            ("constant", "quadratic"),
            ("quadratic", "constant"),
            ("linear", "quadratic"),
            ("quadratic", "linear"),
            ("quadratic", "quadratic"),
        ),
    }
    if mode not in specs_by_mode:
        raise ValueError(f"unknown modal trace delta filter={mode!r}")
    filtered = jnp.zeros_like(face_delta)
    for y_kind, x_kind in specs_by_mode[mode]:
        filtered = filtered + _modal_face_component(face_delta, y_kind, x_kind)
    return filtered


def _z_slab_face_ks(config, *, use_exterior_z_interfaces: bool = False):
    """Return coarse z-face indices used by z-slab SAT coupling.

    The production z-slab runner is still the historical overlap topology:
    coarse and fine both hold the refined z slab, so the coupled coarse planes
    are the slab boundary planes.  The ``use_exterior_z_interfaces`` diagnostic
    instead couples fine faces to the immediately exterior coarse planes and
    masks the whole coarse shadow volume.  That approximates the Stage-2
    disjoint ownership model for falsification experiments without promoting a
    new support surface.
    """
    if use_exterior_z_interfaces:
        return config.fk_lo - 1, config.fk_hi
    return config.fk_lo, config.fk_hi - 1


def _z_slab_coupling_plan(
    config,
    *,
    use_exterior_z_interfaces: bool = False,
    use_boundary_terminated_exterior_z_interfaces: bool = False,
) -> ZSlabCouplingPlan:
    """Return coarse z planes and which faces are artificial interfaces.

    ``use_boundary_terminated_exterior_z_interfaces`` is a diagnostic-only
    extension of the exterior/disjoint-like experiment.  If the fine slab
    touches a physical PEC z boundary, that side is not an artificial
    coarse/fine interface and SAT coupling is skipped there; the opposite side
    still couples to the immediately exterior coarse plane when it exists.
    """
    if use_exterior_z_interfaces and use_boundary_terminated_exterior_z_interfaces:
        raise ValueError(
            "use_exterior_z_interfaces and "
            "use_boundary_terminated_exterior_z_interfaces are mutually exclusive"
        )
    if use_boundary_terminated_exterior_z_interfaces:
        couple_zlo = config.fk_lo > 0
        couple_zhi = config.fk_hi < config.nz_c
        return ZSlabCouplingPlan(
            k_lo_c=config.fk_lo - 1 if couple_zlo else config.fk_lo,
            k_hi_c=config.fk_hi if couple_zhi else config.fk_hi - 1,
            couple_zlo=bool(couple_zlo),
            couple_zhi=bool(couple_zhi),
        )
    k_lo_c, k_hi_c = _z_slab_face_ks(
        config,
        use_exterior_z_interfaces=use_exterior_z_interfaces,
    )
    return ZSlabCouplingPlan(
        k_lo_c=k_lo_c,
        k_hi_c=k_hi_c,
        couple_zlo=True,
        couple_zhi=True,
    )


def _z_slab_coupling_e_3d(
    state_c_fields,
    state_f_fields,
    config,
    *,
    use_exterior_z_interfaces: bool = False,
    use_boundary_terminated_exterior_z_interfaces: bool = False,
):
    """SAT coupling for the two artificial z interfaces of a full-x/y slab.

    The public ``Simulation.add_refinement`` runner refines a z interval across
    the full physical x/y cross-section.  Therefore only z-lo/z-hi are
    artificial coarse/fine interfaces; x/y faces are physical boundaries and
    must not be SAT-coupled as if they were embedded interfaces.
    """
    ex_c, ey_c, ez_c = state_c_fields[:3]
    ex_f, ey_f, ez_f = state_f_fields[:3]
    ratio = config.ratio
    fi, fj = config.fi_lo, config.fj_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    plan = _z_slab_coupling_plan(
        config,
        use_exterior_z_interfaces=use_exterior_z_interfaces,
        use_boundary_terminated_exterior_z_interfaces=(
            use_boundary_terminated_exterior_z_interfaces
        ),
    )
    alpha_f = config.tau * ratio / (ratio + 1.0)
    alpha_c = config.tau / (ratio + 1.0)

    def couple(ec_arr, ef_arr, c_slice, f_slice):
        ec_face = ec_arr[c_slice]
        ef_face = ef_arr[f_slice]
        ef_ds = _restrict_node_aligned_2d(ef_face, ni, nj, ratio)
        ec_us = _prolong_node_aligned_2d(ec_face, config.nx_f, config.ny_f, ratio)
        ec_arr = ec_arr.at[c_slice].add(alpha_c * (ef_ds - ec_face))
        ef_arr = ef_arr.at[f_slice].add(alpha_f * (ec_us - ef_face))
        return ec_arr, ef_arr

    # z-lo and z-hi: tangential E = Ex, Ey.  Ez is normal to the interface.
    if plan.couple_zlo:
        c_lo = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_lo_c)
        f_lo = (slice(None), slice(None), 0)
        ex_c, ex_f = couple(ex_c, ex_f, c_lo, f_lo)
        ey_c, ey_f = couple(ey_c, ey_f, c_lo, f_lo)

    if plan.couple_zhi:
        c_hi = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_hi_c)
        f_hi = (slice(None), slice(None), -1)
        ex_c, ex_f = couple(ex_c, ex_f, c_hi, f_hi)
        ey_c, ey_f = couple(ey_c, ey_f, c_hi, f_hi)
    return (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f)


def _z_slab_coupling_h_3d(
    state_c_fields,
    state_f_fields,
    config,
    *,
    use_exterior_z_interfaces: bool = False,
    use_boundary_terminated_exterior_z_interfaces: bool = False,
):
    """SAT coupling for tangential H on z-lo/z-hi slab interfaces."""
    hx_c, hy_c, hz_c = state_c_fields
    hx_f, hy_f, hz_f = state_f_fields
    ratio = config.ratio
    fi, fj = config.fi_lo, config.fj_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    plan = _z_slab_coupling_plan(
        config,
        use_exterior_z_interfaces=use_exterior_z_interfaces,
        use_boundary_terminated_exterior_z_interfaces=(
            use_boundary_terminated_exterior_z_interfaces
        ),
    )
    alpha_f = config.tau * ratio / (ratio + 1.0)
    alpha_c = config.tau / (ratio + 1.0)

    def couple(hc_arr, hf_arr, c_slice, f_slice):
        hc_face = hc_arr[c_slice]
        hf_face = hf_arr[f_slice]
        hf_ds = _restrict_node_aligned_2d(hf_face, ni, nj, ratio)
        hc_us = _prolong_node_aligned_2d(hc_face, config.nx_f, config.ny_f, ratio)
        hc_arr = hc_arr.at[c_slice].add(alpha_c * (hf_ds - hc_face))
        hf_arr = hf_arr.at[f_slice].add(alpha_f * (hc_us - hf_face))
        return hc_arr, hf_arr

    # z-lo and z-hi: tangential H = Hx, Hy.  Hz is normal to the interface.
    if plan.couple_zlo:
        c_lo = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_lo_c)
        f_lo = (slice(None), slice(None), 0)
        hx_c, hx_f = couple(hx_c, hx_f, c_lo, f_lo)
        hy_c, hy_f = couple(hy_c, hy_f, c_lo, f_lo)

    if plan.couple_zhi:
        c_hi = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_hi_c)
        f_hi = (slice(None), slice(None), -1)
        hx_c, hx_f = couple(hx_c, hx_f, c_hi, f_hi)
        hy_c, hy_f = couple(hy_c, hy_f, c_hi, f_hi)
    return (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f)


def _box_node_aligned_coupling_e_3d(
    state_c_fields,
    state_f_fields,
    config,
    *,
    exterior_interfaces: bool = False,
    couple_xlo: bool = True,
    couple_xhi: bool = True,
    couple_ylo: bool = True,
    couple_yhi: bool = True,
    couple_zlo: bool = True,
    couple_zhi: bool = True,
):
    """Endpoint-node SAT coupling for tangential E on all 6 box faces.

    The legacy generic 6-face helper in :mod:`sbp_sat_3d` assumes block-shaped
    fine faces.  The public JIT runner now uses endpoint-node fine dimensions
    ``(N_coarse - 1) * ratio + 1`` so local x/y windows need the same
    node-aligned restriction/prolongation pair used by the z-slab path.
    """
    ex_c, ey_c, ez_c = state_c_fields[:3]
    ex_f, ey_f, ez_f = state_f_fields[:3]
    ratio = config.ratio
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    nk = config.fk_hi - fk
    c_xlo = fi - 1 if exterior_interfaces else fi
    c_xhi = config.fi_hi if exterior_interfaces else config.fi_hi - 1
    c_ylo = fj - 1 if exterior_interfaces else fj
    c_yhi = config.fj_hi if exterior_interfaces else config.fj_hi - 1
    c_zlo = fk - 1 if exterior_interfaces else fk
    c_zhi = config.fk_hi if exterior_interfaces else config.fk_hi - 1
    alpha_f = config.tau * ratio / (ratio + 1.0)
    alpha_c = config.tau / (ratio + 1.0)

    def couple(ec_arr, ef_arr, c_slice, f_slice, n_a, n_b, nf_a, nf_b):
        ec_face = ec_arr[c_slice]
        ef_face = ef_arr[f_slice]
        ef_ds = _restrict_node_aligned_2d(ef_face, n_a, n_b, ratio)
        ec_us = _prolong_node_aligned_2d(ec_face, nf_a, nf_b, ratio)
        ec_arr = ec_arr.at[c_slice].add(alpha_c * (ef_ds - ec_face))
        ef_arr = ef_arr.at[f_slice].add(alpha_f * (ec_us - ef_face))
        return ec_arr, ef_arr

    # x faces: tangential E = Ey, Ez; face coordinates are y,z.
    if nj > 0 and nk > 0 and config.ny_f > 0 and config.nz_f > 0:
        if couple_xlo:
            c_lo = (c_xlo, slice(fj, fj + nj), slice(fk, fk + nk))
            f_lo = (0, slice(None), slice(None))
            ey_c, ey_f = couple(ey_c, ey_f, c_lo, f_lo, nj, nk, config.ny_f, config.nz_f)
            ez_c, ez_f = couple(ez_c, ez_f, c_lo, f_lo, nj, nk, config.ny_f, config.nz_f)
        if couple_xhi:
            c_hi = (c_xhi, slice(fj, fj + nj), slice(fk, fk + nk))
            f_hi = (-1, slice(None), slice(None))
            ey_c, ey_f = couple(ey_c, ey_f, c_hi, f_hi, nj, nk, config.ny_f, config.nz_f)
            ez_c, ez_f = couple(ez_c, ez_f, c_hi, f_hi, nj, nk, config.ny_f, config.nz_f)

    # y faces: tangential E = Ex, Ez; face coordinates are x,z.
    if ni > 0 and nk > 0 and config.nx_f > 0 and config.nz_f > 0:
        if couple_ylo:
            c_lo = (slice(fi, fi + ni), c_ylo, slice(fk, fk + nk))
            f_lo = (slice(None), 0, slice(None))
            ex_c, ex_f = couple(ex_c, ex_f, c_lo, f_lo, ni, nk, config.nx_f, config.nz_f)
            ez_c, ez_f = couple(ez_c, ez_f, c_lo, f_lo, ni, nk, config.nx_f, config.nz_f)
        if couple_yhi:
            c_hi = (slice(fi, fi + ni), c_yhi, slice(fk, fk + nk))
            f_hi = (slice(None), -1, slice(None))
            ex_c, ex_f = couple(ex_c, ex_f, c_hi, f_hi, ni, nk, config.nx_f, config.nz_f)
            ez_c, ez_f = couple(ez_c, ez_f, c_hi, f_hi, ni, nk, config.nx_f, config.nz_f)

    # z faces: tangential E = Ex, Ey; face coordinates are x,y.
    if ni > 0 and nj > 0 and config.nx_f > 0 and config.ny_f > 0:
        if couple_zlo:
            c_lo = (slice(fi, fi + ni), slice(fj, fj + nj), c_zlo)
            f_lo = (slice(None), slice(None), 0)
            ex_c, ex_f = couple(ex_c, ex_f, c_lo, f_lo, ni, nj, config.nx_f, config.ny_f)
            ey_c, ey_f = couple(ey_c, ey_f, c_lo, f_lo, ni, nj, config.nx_f, config.ny_f)
        if couple_zhi:
            c_hi = (slice(fi, fi + ni), slice(fj, fj + nj), c_zhi)
            f_hi = (slice(None), slice(None), -1)
            ex_c, ex_f = couple(ex_c, ex_f, c_hi, f_hi, ni, nj, config.nx_f, config.ny_f)
            ey_c, ey_f = couple(ey_c, ey_f, c_hi, f_hi, ni, nj, config.nx_f, config.ny_f)

    return (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f)


def _box_node_aligned_coupling_h_3d(
    state_c_fields,
    state_f_fields,
    config,
    *,
    exterior_interfaces: bool = False,
    couple_xlo: bool = True,
    couple_xhi: bool = True,
    couple_ylo: bool = True,
    couple_yhi: bool = True,
    couple_zlo: bool = True,
    couple_zhi: bool = True,
):
    """Endpoint-node SAT coupling for tangential H on all 6 box faces."""
    hx_c, hy_c, hz_c = state_c_fields
    hx_f, hy_f, hz_f = state_f_fields
    ratio = config.ratio
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    nk = config.fk_hi - fk
    c_xlo = fi - 1 if exterior_interfaces else fi
    c_xhi = config.fi_hi if exterior_interfaces else config.fi_hi - 1
    c_ylo = fj - 1 if exterior_interfaces else fj
    c_yhi = config.fj_hi if exterior_interfaces else config.fj_hi - 1
    c_zlo = fk - 1 if exterior_interfaces else fk
    c_zhi = config.fk_hi if exterior_interfaces else config.fk_hi - 1
    alpha_f = config.tau * ratio / (ratio + 1.0)
    alpha_c = config.tau / (ratio + 1.0)

    def couple(hc_arr, hf_arr, c_slice, f_slice, n_a, n_b, nf_a, nf_b):
        hc_face = hc_arr[c_slice]
        hf_face = hf_arr[f_slice]
        hf_ds = _restrict_node_aligned_2d(hf_face, n_a, n_b, ratio)
        hc_us = _prolong_node_aligned_2d(hc_face, nf_a, nf_b, ratio)
        hc_arr = hc_arr.at[c_slice].add(alpha_c * (hf_ds - hc_face))
        hf_arr = hf_arr.at[f_slice].add(alpha_f * (hc_us - hf_face))
        return hc_arr, hf_arr

    # x faces: tangential H = Hy, Hz; face coordinates are y,z.
    if nj > 0 and nk > 0 and config.ny_f > 0 and config.nz_f > 0:
        if couple_xlo:
            c_lo = (c_xlo, slice(fj, fj + nj), slice(fk, fk + nk))
            f_lo = (0, slice(None), slice(None))
            hy_c, hy_f = couple(hy_c, hy_f, c_lo, f_lo, nj, nk, config.ny_f, config.nz_f)
            hz_c, hz_f = couple(hz_c, hz_f, c_lo, f_lo, nj, nk, config.ny_f, config.nz_f)
        if couple_xhi:
            c_hi = (c_xhi, slice(fj, fj + nj), slice(fk, fk + nk))
            f_hi = (-1, slice(None), slice(None))
            hy_c, hy_f = couple(hy_c, hy_f, c_hi, f_hi, nj, nk, config.ny_f, config.nz_f)
            hz_c, hz_f = couple(hz_c, hz_f, c_hi, f_hi, nj, nk, config.ny_f, config.nz_f)

    # y faces: tangential H = Hx, Hz; face coordinates are x,z.
    if ni > 0 and nk > 0 and config.nx_f > 0 and config.nz_f > 0:
        if couple_ylo:
            c_lo = (slice(fi, fi + ni), c_ylo, slice(fk, fk + nk))
            f_lo = (slice(None), 0, slice(None))
            hx_c, hx_f = couple(hx_c, hx_f, c_lo, f_lo, ni, nk, config.nx_f, config.nz_f)
            hz_c, hz_f = couple(hz_c, hz_f, c_lo, f_lo, ni, nk, config.nx_f, config.nz_f)
        if couple_yhi:
            c_hi = (slice(fi, fi + ni), c_yhi, slice(fk, fk + nk))
            f_hi = (slice(None), -1, slice(None))
            hx_c, hx_f = couple(hx_c, hx_f, c_hi, f_hi, ni, nk, config.nx_f, config.nz_f)
            hz_c, hz_f = couple(hz_c, hz_f, c_hi, f_hi, ni, nk, config.nx_f, config.nz_f)

    # z faces: tangential H = Hx, Hy; face coordinates are x,y.
    if ni > 0 and nj > 0 and config.nx_f > 0 and config.ny_f > 0:
        if couple_zlo:
            c_lo = (slice(fi, fi + ni), slice(fj, fj + nj), c_zlo)
            f_lo = (slice(None), slice(None), 0)
            hx_c, hx_f = couple(hx_c, hx_f, c_lo, f_lo, ni, nj, config.nx_f, config.ny_f)
            hy_c, hy_f = couple(hy_c, hy_f, c_lo, f_lo, ni, nj, config.nx_f, config.ny_f)
        if couple_zhi:
            c_hi = (slice(fi, fi + ni), slice(fj, fj + nj), c_zhi)
            f_hi = (slice(None), slice(None), -1)
            hx_c, hx_f = couple(hx_c, hx_f, c_hi, f_hi, ni, nj, config.nx_f, config.ny_f)
            hy_c, hy_f = couple(hy_c, hy_f, c_hi, f_hi, ni, nj, config.nx_f, config.ny_f)

    return (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f)


def _sync_z_slab_coarse_tangential_from_fine(
    tan_a_c,
    tan_b_c,
    tan_a_f,
    tan_b_f,
    config,
    *,
    use_exterior_z_interfaces: bool = False,
    use_boundary_terminated_exterior_z_interfaces: bool = False,
):
    """Overwrite coarse z-face tangential traces from restricted fine traces.

    This is a diagnostic transfer-contract experiment, not the default SAT.  It
    lets research scripts test whether the current broad-waveform blocker is
    dominated by coarse shadow/interface-face drift after coupling.  The synced
    coarse planes must match the active coupling plan; in exterior/disjoint-like
    diagnostics the artificial interface is the immediately exterior coarse
    plane, not the historical overlapping coarse shadow face.
    """
    ratio = config.ratio
    fi, fj = config.fi_lo, config.fj_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    plan = _z_slab_coupling_plan(
        config,
        use_exterior_z_interfaces=use_exterior_z_interfaces,
        use_boundary_terminated_exterior_z_interfaces=(
            use_boundary_terminated_exterior_z_interfaces
        ),
    )
    if plan.couple_zlo:
        c_lo = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_lo_c)
        f_lo = (slice(None), slice(None), 0)
        tan_a_c = tan_a_c.at[c_lo].set(
            _restrict_node_aligned_2d(tan_a_f[f_lo], ni, nj, ratio)
        )
        tan_b_c = tan_b_c.at[c_lo].set(
            _restrict_node_aligned_2d(tan_b_f[f_lo], ni, nj, ratio)
        )
    if plan.couple_zhi:
        c_hi = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_hi_c)
        f_hi = (slice(None), slice(None), -1)
        tan_a_c = tan_a_c.at[c_hi].set(
            _restrict_node_aligned_2d(tan_a_f[f_hi], ni, nj, ratio)
        )
        tan_b_c = tan_b_c.at[c_hi].set(
            _restrict_node_aligned_2d(tan_b_f[f_hi], ni, nj, ratio)
        )
    return tan_a_c, tan_b_c


def _restrict_node_aligned_3d_sample(fine_block, n_coarse_i: int, n_coarse_j: int, n_coarse_k: int, ratio: int):
    """Sample colocated fine nodes for diagnostic coarse-shadow overwrite."""
    return fine_block[::ratio, ::ratio, ::ratio][:n_coarse_i, :n_coarse_j, :n_coarse_k]


def _blend_to_sample(coarse_values, fine_sample, sync_scale: float):
    return coarse_values + sync_scale * (fine_sample - coarse_values)


def _sync_z_slab_coarse_shadow_from_fine(fields_c, fields_f, config, *, sync_scale: float = 1.0):
    """Overwrite the coarse shadow slab from colocated fine-grid fields.

    This diagnostic-only operation tests whether coarse fields inside the
    refined overlap pollute interface updates.  It intentionally uses colocated
    samples rather than an adjoint face projection because the target is the
    coarse shadow *volume* state, not a SAT face quadrature.
    """
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    nk = config.fk_hi - fk
    c_sl = (slice(fi, fi + ni), slice(fj, fj + nj), slice(fk, fk + nk))
    synced = []
    for coarse_arr, fine_arr in zip(fields_c, fields_f):
        synced.append(
            coarse_arr.at[c_sl].set(
                _blend_to_sample(
                    coarse_arr[c_sl],
                    _restrict_node_aligned_3d_sample(fine_arr, ni, nj, nk, config.ratio),
                    sync_scale,
                )
            )
        )
    return tuple(synced)


def _mask_z_slab_coarse_shadow_interior(fields_c, config):
    """Zero the coarse shadow slab interior while preserving interface faces.

    Diagnostic-only: this tests whether the overlapping coarse volume update
    inside the refined slab is contaminating the interface transfer.  The z
    interface planes themselves are preserved so SAT/diagnostic face traces
    still have a coarse-side state.
    """
    if config.fk_hi - config.fk_lo <= 2:
        return fields_c
    fi, fj = config.fi_lo, config.fj_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    c_sl = (
        slice(fi, fi + ni),
        slice(fj, fj + nj),
        slice(config.fk_lo + 1, config.fk_hi - 1),
    )
    return tuple(arr.at[c_sl].set(0.0) for arr in fields_c)


def _mask_z_slab_coarse_shadow_all(fields_c, config):
    """Zero the full overlapping coarse shadow slab.

    Diagnostic-only: this approximates disjoint/exterior ownership by ensuring
    the coarse grid does not evolve any state inside the fine-owned z slab.
    Exterior-interface SAT variants couple to the adjacent coarse planes
    outside this zeroed slab.
    """
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    c_sl = (slice(fi, fi + ni), slice(fj, fj + nj), slice(fk, config.fk_hi))
    return tuple(arr.at[c_sl].set(0.0) for arr in fields_c)


def _mask_box_coarse_shadow_all(fields_c, config):
    """Zero the overlapping coarse volume for a local fine box.

    Diagnostic-only: local x/y boxes currently use an overlapping coarse/fine
    ownership model.  This helper approximates a disjoint local-box ownership
    experiment when paired with exterior box-face SAT coupling.
    """
    c_sl = (
        slice(config.fi_lo, config.fi_hi),
        slice(config.fj_lo, config.fj_hi),
        slice(config.fk_lo, config.fk_hi),
    )
    return tuple(arr.at[c_sl].set(0.0) for arr in fields_c)


def _z_slab_material_coupling_h_3d(
    state_c_fields,
    state_f_fields,
    mats_c,
    mats_f,
    config,
    *,
    use_exterior_z_interfaces: bool = False,
    use_boundary_terminated_exterior_z_interfaces: bool = False,
    material_sat_scale: float = 1.0,
    material_sat_coarse_scale: float = 1.0,
    material_sat_fine_scale: float = 1.0,
    material_sat_h_coarse_scale: float = 1.0,
    material_sat_h_fine_scale: float = 1.0,
    material_sat_zlo_scale: float = 1.0,
    material_sat_zhi_scale: float = 1.0,
    material_sat_h_zlo_scale: float = 1.0,
    material_sat_h_zhi_scale: float = 1.0,
    material_sat_pair_a_zlo_scale: float = 1.0,
    material_sat_pair_b_zlo_scale: float = 1.0,
    material_sat_zlo_common_trace_projection: str = "dual",
    material_sat_zhi_common_trace_projection: str = "dual",
    material_sat_zhi_coarse_eps_blend: float = 0.0,
    material_sat_face_projection: str = "node_adjoint",
):
    """Material-weighted H correction for z-lo/z-hi artificial interfaces."""
    ex_c, ey_c, _ez_c, hx_c, hy_c, hz_c = state_c_fields
    ex_f, ey_f, _ez_f, hx_f, hy_f, hz_f = state_f_fields
    ratio = config.ratio
    fi, fj = config.fi_lo, config.fj_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    plan = _z_slab_coupling_plan(
        config,
        use_exterior_z_interfaces=use_exterior_z_interfaces,
        use_boundary_terminated_exterior_z_interfaces=(
            use_boundary_terminated_exterior_z_interfaces
        ),
    )
    coarse_scale = (
        material_sat_scale
        * material_sat_coarse_scale
        * material_sat_h_coarse_scale
    )
    fine_scale = (
        material_sat_scale
        * material_sat_fine_scale
        * material_sat_h_fine_scale
    )

    def down(face):
        if material_sat_face_projection == "node_adjoint":
            return _restrict_node_aligned_2d(face, ni, nj, ratio)
        if material_sat_face_projection == "sample":
            return _restrict_node_aligned_2d_sample(face, ni, nj, ratio)
        raise ValueError(
            f"Unknown material_sat_face_projection={material_sat_face_projection!r}"
        )

    def up(face):
        return _prolong_node_aligned_2d(face, config.nx_f, config.ny_f, ratio)

    def zlo_common_trace_projection():
        mode = material_sat_zlo_common_trace_projection
        if mode in ("dual", "coarse", "fine", "average"):
            return mode
        raise ValueError(
            "Unknown material_sat_zlo_common_trace_projection="
            f"{mode!r}"
        )

    def zhi_common_trace_projection():
        mode = material_sat_zhi_common_trace_projection
        if mode in ("dual", "coarse", "fine", "average"):
            return mode
        raise ValueError(
            "Unknown material_sat_zhi_common_trace_projection="
            f"{mode!r}"
        )

    def face_coeffs_c(k):
        sl = (slice(fi, fi + ni), slice(fj, fj + nj), k)
        return mats_c.eps_r[sl] * EPS_0, mats_c.mu_r[sl] * MU_0

    def face_coeffs_f(k):
        sl = (slice(None), slice(None), k)
        return mats_f.eps_r[sl] * EPS_0, mats_f.mu_r[sl] * MU_0

    def apply_zlo(hx_c_arr, hy_c_arr, hx_f_arr, hy_f_arr):
        c = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_lo_c)
        f = (slice(None), slice(None), 0)
        eps_c, mu_c = face_coeffs_c(plan.k_lo_c)
        eps_f, mu_f = face_coeffs_f(0)

        # Coarse side: lower=coarse, upper=restricted fine.
        pair_a_c = interface_pair_deltas(
            ex_c[c],
            hy_c_arr[c],
            down(ex_f[f]),
            down(hy_f_arr[f]),
            epsilon_lower=eps_c,
            mu_lower=mu_c,
            epsilon_upper=down(eps_f),
            mu_upper=down(mu_f),
            h_lower=config.dx_c,
            h_upper=config.dx_f,
            dt=config.dt,
        )
        pair_b_c = interface_pair_deltas(
            ey_c[c],
            -hx_c_arr[c],
            down(ey_f[f]),
            -down(hx_f_arr[f]),
            epsilon_lower=eps_c,
            mu_lower=mu_c,
            epsilon_upper=down(eps_f),
            mu_upper=down(mu_f),
            h_lower=config.dx_c,
            h_upper=config.dx_f,
            dt=config.dt,
        )
        zlo_coarse_scale = (
            coarse_scale * material_sat_zlo_scale * material_sat_h_zlo_scale
        )
        zlo_fine_scale = (
            fine_scale * material_sat_zlo_scale * material_sat_h_zlo_scale
        )
        zlo_pair_a_coarse_scale = zlo_coarse_scale * material_sat_pair_a_zlo_scale
        zlo_pair_a_fine_scale = zlo_fine_scale * material_sat_pair_a_zlo_scale
        zlo_pair_b_coarse_scale = zlo_coarse_scale * material_sat_pair_b_zlo_scale
        zlo_pair_b_fine_scale = zlo_fine_scale * material_sat_pair_b_zlo_scale
        trace_projection = zlo_common_trace_projection()
        if trace_projection in ("fine", "average"):
            pair_a_f = interface_pair_deltas(
                up(ex_c[c]),
                up(hy_c[c]),
                ex_f[f],
                hy_f_arr[f],
                epsilon_lower=up(eps_c),
                mu_lower=up(mu_c),
                epsilon_upper=eps_f,
                mu_upper=mu_f,
                h_lower=config.dx_c,
                h_upper=config.dx_f,
                dt=config.dt,
            )
            pair_b_f = interface_pair_deltas(
                up(ey_c[c]),
                -up(hx_c[c]),
                ey_f[f],
                -hx_f_arr[f],
                epsilon_lower=up(eps_c),
                mu_lower=up(mu_c),
                epsilon_upper=eps_f,
                mu_upper=mu_f,
                h_lower=config.dx_c,
                h_upper=config.dx_f,
                dt=config.dt,
            )
            if trace_projection == "fine":
                pair_a_dv_lower = -(config.dt / (mu_c * config.dx_c)) * (
                    down(pair_a_f.trace.u_star) - ex_c[c]
                )
                pair_b_dv_lower = -(config.dt / (mu_c * config.dx_c)) * (
                    down(pair_b_f.trace.u_star) - ey_c[c]
                )
                pair_a_dv_upper = pair_a_f.dv_upper
                pair_b_dv_upper = pair_b_f.dv_upper
            else:
                pair_a_u_star_c = 0.5 * (
                    pair_a_c.trace.u_star + down(pair_a_f.trace.u_star)
                )
                pair_b_u_star_c = 0.5 * (
                    pair_b_c.trace.u_star + down(pair_b_f.trace.u_star)
                )
                pair_a_u_star_f = 0.5 * (
                    up(pair_a_c.trace.u_star) + pair_a_f.trace.u_star
                )
                pair_b_u_star_f = 0.5 * (
                    up(pair_b_c.trace.u_star) + pair_b_f.trace.u_star
                )
                pair_a_dv_lower = -(config.dt / (mu_c * config.dx_c)) * (
                    pair_a_u_star_c - ex_c[c]
                )
                pair_b_dv_lower = -(config.dt / (mu_c * config.dx_c)) * (
                    pair_b_u_star_c - ey_c[c]
                )
                pair_a_dv_upper = +(config.dt / (mu_f * config.dx_f)) * (
                    pair_a_u_star_f - ex_f[f]
                )
                pair_b_dv_upper = +(config.dt / (mu_f * config.dx_f)) * (
                    pair_b_u_star_f - ey_f[f]
                )
            hy_c_arr = hy_c_arr.at[c].add(zlo_pair_a_coarse_scale * pair_a_dv_lower)
            hx_c_arr = hx_c_arr.at[c].add(-zlo_pair_b_coarse_scale * pair_b_dv_lower)
            hy_f_arr = hy_f_arr.at[f].add(zlo_pair_a_fine_scale * pair_a_dv_upper)
            hx_f_arr = hx_f_arr.at[f].add(-zlo_pair_b_fine_scale * pair_b_dv_upper)
        else:
            hy_c_arr = hy_c_arr.at[c].add(zlo_pair_a_coarse_scale * pair_a_c.dv_lower)
            hx_c_arr = hx_c_arr.at[c].add(-zlo_pair_b_coarse_scale * pair_b_c.dv_lower)
            if trace_projection == "coarse":
                pair_a_dv_upper = +(config.dt / (mu_f * config.dx_f)) * (
                    up(pair_a_c.trace.u_star) - ex_f[f]
                )
                pair_b_dv_upper = +(config.dt / (mu_f * config.dx_f)) * (
                    up(pair_b_c.trace.u_star) - ey_f[f]
                )
                hy_f_arr = hy_f_arr.at[f].add(
                    zlo_pair_a_fine_scale * pair_a_dv_upper
                )
                hx_f_arr = hx_f_arr.at[f].add(
                    -zlo_pair_b_fine_scale * pair_b_dv_upper
                )
            else:
                # Fine side: lower=prolonged coarse, upper=fine.
                pair_a_f = interface_pair_deltas(
                    up(ex_c[c]),
                    up(hy_c[c]),
                    ex_f[f],
                    hy_f_arr[f],
                    epsilon_lower=up(eps_c),
                    mu_lower=up(mu_c),
                    epsilon_upper=eps_f,
                    mu_upper=mu_f,
                    h_lower=config.dx_c,
                    h_upper=config.dx_f,
                    dt=config.dt,
                )
                pair_b_f = interface_pair_deltas(
                    up(ey_c[c]),
                    -up(hx_c[c]),
                    ey_f[f],
                    -hx_f_arr[f],
                    epsilon_lower=up(eps_c),
                    mu_lower=up(mu_c),
                    epsilon_upper=eps_f,
                    mu_upper=mu_f,
                    h_lower=config.dx_c,
                    h_upper=config.dx_f,
                    dt=config.dt,
                )
                hy_f_arr = hy_f_arr.at[f].add(zlo_pair_a_fine_scale * pair_a_f.dv_upper)
                hx_f_arr = hx_f_arr.at[f].add(-zlo_pair_b_fine_scale * pair_b_f.dv_upper)
        return hx_c_arr, hy_c_arr, hx_f_arr, hy_f_arr

    def apply_zhi(hx_c_arr, hy_c_arr, hx_f_arr, hy_f_arr):
        c = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_hi_c)
        f = (slice(None), slice(None), -1)
        eps_c, mu_c = face_coeffs_c(plan.k_hi_c)
        eps_f, mu_f = face_coeffs_f(-1)
        if material_sat_zhi_coarse_eps_blend != 0.0:
            eps_c = (
                (1.0 - material_sat_zhi_coarse_eps_blend) * eps_c
                + material_sat_zhi_coarse_eps_blend * down(eps_f)
            )

        # Coarse side: lower=restricted fine, upper=coarse.
        pair_a_c = interface_pair_deltas(
            down(ex_f[f]),
            down(hy_f_arr[f]),
            ex_c[c],
            hy_c_arr[c],
            epsilon_lower=down(eps_f),
            mu_lower=down(mu_f),
            epsilon_upper=eps_c,
            mu_upper=mu_c,
            h_lower=config.dx_f,
            h_upper=config.dx_c,
            dt=config.dt,
        )
        pair_b_c = interface_pair_deltas(
            down(ey_f[f]),
            -down(hx_f_arr[f]),
            ey_c[c],
            -hx_c_arr[c],
            epsilon_lower=down(eps_f),
            mu_lower=down(mu_f),
            epsilon_upper=eps_c,
            mu_upper=mu_c,
            h_lower=config.dx_f,
            h_upper=config.dx_c,
            dt=config.dt,
        )
        zhi_coarse_scale = (
            coarse_scale * material_sat_zhi_scale * material_sat_h_zhi_scale
        )
        zhi_fine_scale = (
            fine_scale * material_sat_zhi_scale * material_sat_h_zhi_scale
        )
        trace_projection = zhi_common_trace_projection()
        if trace_projection in ("fine", "average"):
            pair_a_f = interface_pair_deltas(
                ex_f[f],
                hy_f_arr[f],
                up(ex_c[c]),
                up(hy_c[c]),
                epsilon_lower=eps_f,
                mu_lower=mu_f,
                epsilon_upper=up(eps_c),
                mu_upper=up(mu_c),
                h_lower=config.dx_f,
                h_upper=config.dx_c,
                dt=config.dt,
            )
            pair_b_f = interface_pair_deltas(
                ey_f[f],
                -hx_f_arr[f],
                up(ey_c[c]),
                -up(hx_c[c]),
                epsilon_lower=eps_f,
                mu_lower=mu_f,
                epsilon_upper=up(eps_c),
                mu_upper=up(mu_c),
                h_lower=config.dx_f,
                h_upper=config.dx_c,
                dt=config.dt,
            )
            if trace_projection == "fine":
                pair_a_dv_upper = +(config.dt / (mu_c * config.dx_c)) * (
                    down(pair_a_f.trace.u_star) - ex_c[c]
                )
                pair_b_dv_upper = +(config.dt / (mu_c * config.dx_c)) * (
                    down(pair_b_f.trace.u_star) - ey_c[c]
                )
                pair_a_dv_lower = pair_a_f.dv_lower
                pair_b_dv_lower = pair_b_f.dv_lower
            else:
                pair_a_u_star_c = 0.5 * (
                    pair_a_c.trace.u_star + down(pair_a_f.trace.u_star)
                )
                pair_b_u_star_c = 0.5 * (
                    pair_b_c.trace.u_star + down(pair_b_f.trace.u_star)
                )
                pair_a_u_star_f = 0.5 * (
                    up(pair_a_c.trace.u_star) + pair_a_f.trace.u_star
                )
                pair_b_u_star_f = 0.5 * (
                    up(pair_b_c.trace.u_star) + pair_b_f.trace.u_star
                )
                pair_a_dv_upper = +(config.dt / (mu_c * config.dx_c)) * (
                    pair_a_u_star_c - ex_c[c]
                )
                pair_b_dv_upper = +(config.dt / (mu_c * config.dx_c)) * (
                    pair_b_u_star_c - ey_c[c]
                )
                pair_a_dv_lower = -(config.dt / (mu_f * config.dx_f)) * (
                    pair_a_u_star_f - ex_f[f]
                )
                pair_b_dv_lower = -(config.dt / (mu_f * config.dx_f)) * (
                    pair_b_u_star_f - ey_f[f]
                )
            hy_c_arr = hy_c_arr.at[c].add(zhi_coarse_scale * pair_a_dv_upper)
            hx_c_arr = hx_c_arr.at[c].add(-zhi_coarse_scale * pair_b_dv_upper)
            hy_f_arr = hy_f_arr.at[f].add(zhi_fine_scale * pair_a_dv_lower)
            hx_f_arr = hx_f_arr.at[f].add(-zhi_fine_scale * pair_b_dv_lower)
        else:
            hy_c_arr = hy_c_arr.at[c].add(zhi_coarse_scale * pair_a_c.dv_upper)
            hx_c_arr = hx_c_arr.at[c].add(-zhi_coarse_scale * pair_b_c.dv_upper)
            if trace_projection == "coarse":
                pair_a_dv_lower = -(config.dt / (mu_f * config.dx_f)) * (
                    up(pair_a_c.trace.u_star) - ex_f[f]
                )
                pair_b_dv_lower = -(config.dt / (mu_f * config.dx_f)) * (
                    up(pair_b_c.trace.u_star) - ey_f[f]
                )
                hy_f_arr = hy_f_arr.at[f].add(zhi_fine_scale * pair_a_dv_lower)
                hx_f_arr = hx_f_arr.at[f].add(-zhi_fine_scale * pair_b_dv_lower)
            else:
                # Fine side: lower=fine, upper=prolonged coarse.
                pair_a_f = interface_pair_deltas(
                    ex_f[f],
                    hy_f_arr[f],
                    up(ex_c[c]),
                    up(hy_c[c]),
                    epsilon_lower=eps_f,
                    mu_lower=mu_f,
                    epsilon_upper=up(eps_c),
                    mu_upper=up(mu_c),
                    h_lower=config.dx_f,
                    h_upper=config.dx_c,
                    dt=config.dt,
                )
                pair_b_f = interface_pair_deltas(
                    ey_f[f],
                    -hx_f_arr[f],
                    up(ey_c[c]),
                    -up(hx_c[c]),
                    epsilon_lower=eps_f,
                    mu_lower=mu_f,
                    epsilon_upper=up(eps_c),
                    mu_upper=up(mu_c),
                    h_lower=config.dx_f,
                    h_upper=config.dx_c,
                    dt=config.dt,
                )
                hy_f_arr = hy_f_arr.at[f].add(zhi_fine_scale * pair_a_f.dv_lower)
                hx_f_arr = hx_f_arr.at[f].add(-zhi_fine_scale * pair_b_f.dv_lower)
        return hx_c_arr, hy_c_arr, hx_f_arr, hy_f_arr

    if plan.couple_zlo:
        hx_c, hy_c, hx_f, hy_f = apply_zlo(hx_c, hy_c, hx_f, hy_f)
    if plan.couple_zhi:
        hx_c, hy_c, hx_f, hy_f = apply_zhi(hx_c, hy_c, hx_f, hy_f)
    return (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f)


def _z_slab_material_coupling_e_3d(
    state_c_fields,
    state_f_fields,
    mats_c,
    mats_f,
    config,
    *,
    use_exterior_z_interfaces: bool = False,
    use_boundary_terminated_exterior_z_interfaces: bool = False,
    material_sat_scale: float = 1.0,
    material_sat_coarse_scale: float = 1.0,
    material_sat_fine_scale: float = 1.0,
    material_sat_e_coarse_scale: float = 1.0,
    material_sat_e_fine_scale: float = 1.0,
    material_sat_zlo_scale: float = 1.0,
    material_sat_zhi_scale: float = 1.0,
    material_sat_e_zlo_scale: float = 1.0,
    material_sat_e_zhi_scale: float = 1.0,
    material_sat_pair_a_zlo_scale: float = 1.0,
    material_sat_pair_b_zlo_scale: float = 1.0,
    material_sat_zlo_common_trace_projection: str = "dual",
    material_sat_zhi_common_trace_projection: str = "dual",
    material_sat_normal_e_scale: float = 0.0,
    material_sat_zhi_coarse_eps_blend: float = 0.0,
    material_sat_face_projection: str = "node_adjoint",
):
    """Material-weighted E correction for z-lo/z-hi artificial interfaces."""
    ex_c, ey_c, ez_c, hx_c, hy_c, _hz_c = state_c_fields
    ex_f, ey_f, ez_f, hx_f, hy_f, _hz_f = state_f_fields
    ratio = config.ratio
    fi, fj = config.fi_lo, config.fj_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    plan = _z_slab_coupling_plan(
        config,
        use_exterior_z_interfaces=use_exterior_z_interfaces,
        use_boundary_terminated_exterior_z_interfaces=(
            use_boundary_terminated_exterior_z_interfaces
        ),
    )
    coarse_scale = (
        material_sat_scale
        * material_sat_coarse_scale
        * material_sat_e_coarse_scale
    )
    fine_scale = (
        material_sat_scale
        * material_sat_fine_scale
        * material_sat_e_fine_scale
    )

    def down(face):
        if material_sat_face_projection == "node_adjoint":
            return _restrict_node_aligned_2d(face, ni, nj, ratio)
        if material_sat_face_projection == "sample":
            return _restrict_node_aligned_2d_sample(face, ni, nj, ratio)
        raise ValueError(
            f"Unknown material_sat_face_projection={material_sat_face_projection!r}"
        )

    def up(face):
        return _prolong_node_aligned_2d(face, config.nx_f, config.ny_f, ratio)

    def zlo_common_trace_projection():
        mode = material_sat_zlo_common_trace_projection
        if mode in ("dual", "coarse", "fine", "average"):
            return mode
        raise ValueError(
            "Unknown material_sat_zlo_common_trace_projection="
            f"{mode!r}"
        )

    def zhi_common_trace_projection():
        mode = material_sat_zhi_common_trace_projection
        if mode in ("dual", "coarse", "fine", "average"):
            return mode
        raise ValueError(
            "Unknown material_sat_zhi_common_trace_projection="
            f"{mode!r}"
        )

    def face_coeffs_c(k):
        sl = (slice(fi, fi + ni), slice(fj, fj + nj), k)
        return mats_c.eps_r[sl] * EPS_0, mats_c.mu_r[sl] * MU_0

    def face_coeffs_f(k):
        sl = (slice(None), slice(None), k)
        return mats_f.eps_r[sl] * EPS_0, mats_f.mu_r[sl] * MU_0

    alpha_f = config.tau * ratio / (ratio + 1.0)
    alpha_c = config.tau / (ratio + 1.0)

    def couple_normal_ez(ez_c_arr, ez_f_arr, c, f, eps_c, eps_f):
        if material_sat_normal_e_scale == 0.0:
            return ez_c_arr, ez_f_arr
        d_c = eps_c * ez_c_arr[c]
        d_f = eps_f * ez_f_arr[f]
        d_f_down = down(d_f)
        d_c_up = up(d_c)
        ez_c_star = 0.5 * (d_c + d_f_down) / eps_c
        ez_f_star = 0.5 * (d_f + d_c_up) / eps_f
        scale = material_sat_scale * material_sat_normal_e_scale
        ez_c_arr = ez_c_arr.at[c].add(scale * alpha_c * (ez_c_star - ez_c_arr[c]))
        ez_f_arr = ez_f_arr.at[f].add(scale * alpha_f * (ez_f_star - ez_f_arr[f]))
        return ez_c_arr, ez_f_arr

    def apply_zlo(ex_c_arr, ey_c_arr, ez_c_arr, ex_f_arr, ey_f_arr, ez_f_arr):
        c = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_lo_c)
        f = (slice(None), slice(None), 0)
        eps_c, mu_c = face_coeffs_c(plan.k_lo_c)
        eps_f, mu_f = face_coeffs_f(0)

        pair_a_c = interface_pair_deltas(
            ex_c_arr[c],
            hy_c[c],
            down(ex_f_arr[f]),
            down(hy_f[f]),
            epsilon_lower=eps_c,
            mu_lower=mu_c,
            epsilon_upper=down(eps_f),
            mu_upper=down(mu_f),
            h_lower=config.dx_c,
            h_upper=config.dx_f,
            dt=config.dt,
        )
        pair_b_c = interface_pair_deltas(
            ey_c_arr[c],
            -hx_c[c],
            down(ey_f_arr[f]),
            -down(hx_f[f]),
            epsilon_lower=eps_c,
            mu_lower=mu_c,
            epsilon_upper=down(eps_f),
            mu_upper=down(mu_f),
            h_lower=config.dx_c,
            h_upper=config.dx_f,
            dt=config.dt,
        )
        zlo_coarse_scale = (
            coarse_scale * material_sat_zlo_scale * material_sat_e_zlo_scale
        )
        zlo_fine_scale = (
            fine_scale * material_sat_zlo_scale * material_sat_e_zlo_scale
        )
        zlo_pair_a_coarse_scale = zlo_coarse_scale * material_sat_pair_a_zlo_scale
        zlo_pair_a_fine_scale = zlo_fine_scale * material_sat_pair_a_zlo_scale
        zlo_pair_b_coarse_scale = zlo_coarse_scale * material_sat_pair_b_zlo_scale
        zlo_pair_b_fine_scale = zlo_fine_scale * material_sat_pair_b_zlo_scale
        trace_projection = zlo_common_trace_projection()
        if trace_projection in ("fine", "average"):
            pair_a_f = interface_pair_deltas(
                up(ex_c[c]),
                up(hy_c[c]),
                ex_f_arr[f],
                hy_f[f],
                epsilon_lower=up(eps_c),
                mu_lower=up(mu_c),
                epsilon_upper=eps_f,
                mu_upper=mu_f,
                h_lower=config.dx_c,
                h_upper=config.dx_f,
                dt=config.dt,
            )
            pair_b_f = interface_pair_deltas(
                up(ey_c[c]),
                -up(hx_c[c]),
                ey_f_arr[f],
                -hx_f[f],
                epsilon_lower=up(eps_c),
                mu_lower=up(mu_c),
                epsilon_upper=eps_f,
                mu_upper=mu_f,
                h_lower=config.dx_c,
                h_upper=config.dx_f,
                dt=config.dt,
            )
            if trace_projection == "fine":
                pair_a_du_lower = -(config.dt / (eps_c * config.dx_c)) * (
                    down(pair_a_f.trace.v_star) - hy_c[c]
                )
                pair_b_du_lower = -(config.dt / (eps_c * config.dx_c)) * (
                    down(pair_b_f.trace.v_star) + hx_c[c]
                )
                pair_a_du_upper = pair_a_f.du_upper
                pair_b_du_upper = pair_b_f.du_upper
            else:
                pair_a_v_star_c = 0.5 * (
                    pair_a_c.trace.v_star + down(pair_a_f.trace.v_star)
                )
                pair_b_v_star_c = 0.5 * (
                    pair_b_c.trace.v_star + down(pair_b_f.trace.v_star)
                )
                pair_a_v_star_f = 0.5 * (
                    up(pair_a_c.trace.v_star) + pair_a_f.trace.v_star
                )
                pair_b_v_star_f = 0.5 * (
                    up(pair_b_c.trace.v_star) + pair_b_f.trace.v_star
                )
                pair_a_du_lower = -(config.dt / (eps_c * config.dx_c)) * (
                    pair_a_v_star_c - hy_c[c]
                )
                pair_b_du_lower = -(config.dt / (eps_c * config.dx_c)) * (
                    pair_b_v_star_c + hx_c[c]
                )
                pair_a_du_upper = +(config.dt / (eps_f * config.dx_f)) * (
                    pair_a_v_star_f - hy_f[f]
                )
                pair_b_du_upper = +(config.dt / (eps_f * config.dx_f)) * (
                    pair_b_v_star_f + hx_f[f]
                )
            ex_c_arr = ex_c_arr.at[c].add(zlo_pair_a_coarse_scale * pair_a_du_lower)
            ey_c_arr = ey_c_arr.at[c].add(zlo_pair_b_coarse_scale * pair_b_du_lower)
            ex_f_arr = ex_f_arr.at[f].add(zlo_pair_a_fine_scale * pair_a_du_upper)
            ey_f_arr = ey_f_arr.at[f].add(zlo_pair_b_fine_scale * pair_b_du_upper)
        else:
            ex_c_arr = ex_c_arr.at[c].add(zlo_pair_a_coarse_scale * pair_a_c.du_lower)
            ey_c_arr = ey_c_arr.at[c].add(zlo_pair_b_coarse_scale * pair_b_c.du_lower)
            if trace_projection == "coarse":
                pair_a_du_upper = +(config.dt / (eps_f * config.dx_f)) * (
                    up(pair_a_c.trace.v_star) - hy_f[f]
                )
                pair_b_du_upper = +(config.dt / (eps_f * config.dx_f)) * (
                    up(pair_b_c.trace.v_star) + hx_f[f]
                )
                ex_f_arr = ex_f_arr.at[f].add(
                    zlo_pair_a_fine_scale * pair_a_du_upper
                )
                ey_f_arr = ey_f_arr.at[f].add(
                    zlo_pair_b_fine_scale * pair_b_du_upper
                )
            else:
                pair_a_f = interface_pair_deltas(
                    up(ex_c[c]),
                    up(hy_c[c]),
                    ex_f_arr[f],
                    hy_f[f],
                    epsilon_lower=up(eps_c),
                    mu_lower=up(mu_c),
                    epsilon_upper=eps_f,
                    mu_upper=mu_f,
                    h_lower=config.dx_c,
                    h_upper=config.dx_f,
                    dt=config.dt,
                )
                pair_b_f = interface_pair_deltas(
                    up(ey_c[c]),
                    -up(hx_c[c]),
                    ey_f_arr[f],
                    -hx_f[f],
                    epsilon_lower=up(eps_c),
                    mu_lower=up(mu_c),
                    epsilon_upper=eps_f,
                    mu_upper=mu_f,
                    h_lower=config.dx_c,
                    h_upper=config.dx_f,
                    dt=config.dt,
                )
                ex_f_arr = ex_f_arr.at[f].add(zlo_pair_a_fine_scale * pair_a_f.du_upper)
                ey_f_arr = ey_f_arr.at[f].add(zlo_pair_b_fine_scale * pair_b_f.du_upper)
        ez_c_arr, ez_f_arr = couple_normal_ez(ez_c_arr, ez_f_arr, c, f, eps_c, eps_f)
        return ex_c_arr, ey_c_arr, ez_c_arr, ex_f_arr, ey_f_arr, ez_f_arr

    def apply_zhi(ex_c_arr, ey_c_arr, ez_c_arr, ex_f_arr, ey_f_arr, ez_f_arr):
        c = (slice(fi, fi + ni), slice(fj, fj + nj), plan.k_hi_c)
        f = (slice(None), slice(None), -1)
        eps_c, mu_c = face_coeffs_c(plan.k_hi_c)
        eps_f, mu_f = face_coeffs_f(-1)
        if material_sat_zhi_coarse_eps_blend != 0.0:
            eps_c = (
                (1.0 - material_sat_zhi_coarse_eps_blend) * eps_c
                + material_sat_zhi_coarse_eps_blend * down(eps_f)
            )

        pair_a_c = interface_pair_deltas(
            down(ex_f_arr[f]),
            down(hy_f[f]),
            ex_c_arr[c],
            hy_c[c],
            epsilon_lower=down(eps_f),
            mu_lower=down(mu_f),
            epsilon_upper=eps_c,
            mu_upper=mu_c,
            h_lower=config.dx_f,
            h_upper=config.dx_c,
            dt=config.dt,
        )
        pair_b_c = interface_pair_deltas(
            down(ey_f_arr[f]),
            -down(hx_f[f]),
            ey_c_arr[c],
            -hx_c[c],
            epsilon_lower=down(eps_f),
            mu_lower=down(mu_f),
            epsilon_upper=eps_c,
            mu_upper=mu_c,
            h_lower=config.dx_f,
            h_upper=config.dx_c,
            dt=config.dt,
        )
        zhi_coarse_scale = (
            coarse_scale * material_sat_zhi_scale * material_sat_e_zhi_scale
        )
        zhi_fine_scale = (
            fine_scale * material_sat_zhi_scale * material_sat_e_zhi_scale
        )
        trace_projection = zhi_common_trace_projection()
        if trace_projection in ("fine", "average"):
            pair_a_f = interface_pair_deltas(
                ex_f_arr[f],
                hy_f[f],
                up(ex_c[c]),
                up(hy_c[c]),
                epsilon_lower=eps_f,
                mu_lower=mu_f,
                epsilon_upper=up(eps_c),
                mu_upper=up(mu_c),
                h_lower=config.dx_f,
                h_upper=config.dx_c,
                dt=config.dt,
            )
            pair_b_f = interface_pair_deltas(
                ey_f_arr[f],
                -hx_f[f],
                up(ey_c[c]),
                -up(hx_c[c]),
                epsilon_lower=eps_f,
                mu_lower=mu_f,
                epsilon_upper=up(eps_c),
                mu_upper=up(mu_c),
                h_lower=config.dx_f,
                h_upper=config.dx_c,
                dt=config.dt,
            )
            if trace_projection == "fine":
                pair_a_du_upper = +(config.dt / (eps_c * config.dx_c)) * (
                    down(pair_a_f.trace.v_star) - hy_c[c]
                )
                pair_b_du_upper = +(config.dt / (eps_c * config.dx_c)) * (
                    down(pair_b_f.trace.v_star) + hx_c[c]
                )
                pair_a_du_lower = pair_a_f.du_lower
                pair_b_du_lower = pair_b_f.du_lower
            else:
                pair_a_v_star_c = 0.5 * (
                    pair_a_c.trace.v_star + down(pair_a_f.trace.v_star)
                )
                pair_b_v_star_c = 0.5 * (
                    pair_b_c.trace.v_star + down(pair_b_f.trace.v_star)
                )
                pair_a_v_star_f = 0.5 * (
                    up(pair_a_c.trace.v_star) + pair_a_f.trace.v_star
                )
                pair_b_v_star_f = 0.5 * (
                    up(pair_b_c.trace.v_star) + pair_b_f.trace.v_star
                )
                pair_a_du_upper = +(config.dt / (eps_c * config.dx_c)) * (
                    pair_a_v_star_c - hy_c[c]
                )
                pair_b_du_upper = +(config.dt / (eps_c * config.dx_c)) * (
                    pair_b_v_star_c + hx_c[c]
                )
                pair_a_du_lower = -(config.dt / (eps_f * config.dx_f)) * (
                    pair_a_v_star_f - hy_f[f]
                )
                pair_b_du_lower = -(config.dt / (eps_f * config.dx_f)) * (
                    pair_b_v_star_f + hx_f[f]
                )
            ex_c_arr = ex_c_arr.at[c].add(zhi_coarse_scale * pair_a_du_upper)
            ey_c_arr = ey_c_arr.at[c].add(zhi_coarse_scale * pair_b_du_upper)
            ex_f_arr = ex_f_arr.at[f].add(zhi_fine_scale * pair_a_du_lower)
            ey_f_arr = ey_f_arr.at[f].add(zhi_fine_scale * pair_b_du_lower)
        else:
            ex_c_arr = ex_c_arr.at[c].add(zhi_coarse_scale * pair_a_c.du_upper)
            ey_c_arr = ey_c_arr.at[c].add(zhi_coarse_scale * pair_b_c.du_upper)
            if trace_projection == "coarse":
                pair_a_du_lower = -(config.dt / (eps_f * config.dx_f)) * (
                    up(pair_a_c.trace.v_star) - hy_f[f]
                )
                pair_b_du_lower = -(config.dt / (eps_f * config.dx_f)) * (
                    up(pair_b_c.trace.v_star) + hx_f[f]
                )
                ex_f_arr = ex_f_arr.at[f].add(zhi_fine_scale * pair_a_du_lower)
                ey_f_arr = ey_f_arr.at[f].add(zhi_fine_scale * pair_b_du_lower)
            else:
                pair_a_f = interface_pair_deltas(
                    ex_f_arr[f],
                    hy_f[f],
                    up(ex_c[c]),
                    up(hy_c[c]),
                    epsilon_lower=eps_f,
                    mu_lower=mu_f,
                    epsilon_upper=up(eps_c),
                    mu_upper=up(mu_c),
                    h_lower=config.dx_f,
                    h_upper=config.dx_c,
                    dt=config.dt,
                )
                pair_b_f = interface_pair_deltas(
                    ey_f_arr[f],
                    -hx_f[f],
                    up(ey_c[c]),
                    -up(hx_c[c]),
                    epsilon_lower=eps_f,
                    mu_lower=mu_f,
                    epsilon_upper=up(eps_c),
                    mu_upper=up(mu_c),
                    h_lower=config.dx_f,
                    h_upper=config.dx_c,
                    dt=config.dt,
                )
                ex_f_arr = ex_f_arr.at[f].add(zhi_fine_scale * pair_a_f.du_lower)
                ey_f_arr = ey_f_arr.at[f].add(zhi_fine_scale * pair_b_f.du_lower)
        ez_c_arr, ez_f_arr = couple_normal_ez(ez_c_arr, ez_f_arr, c, f, eps_c, eps_f)
        return ex_c_arr, ey_c_arr, ez_c_arr, ex_f_arr, ey_f_arr, ez_f_arr

    if plan.couple_zlo:
        ex_c, ey_c, ez_c, ex_f, ey_f, ez_f = apply_zlo(
            ex_c, ey_c, ez_c, ex_f, ey_f, ez_f
        )
    if plan.couple_zhi:
        ex_c, ey_c, ez_c, ex_f, ey_f, ez_f = apply_zhi(
            ex_c, ey_c, ez_c, ex_f, ey_f, ez_f
        )
    return (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f)


def run_subgridded_jit(
    grid_c: Grid,
    mats_c: MaterialArrays,
    mats_f: MaterialArrays,
    config: SubgridConfig3D,
    n_steps: int,
    *,
    pec_mask_c=None,
    pec_mask_f=None,
    sources_f: list | None = None,
    sources_c: list | None = None,
    probe_indices_f: list | None = None,
    probe_components: list | None = None,
    probe_indices_c: list | None = None,
    probe_components_c: list | None = None,
    lumped_sparam_indices_f: list | None = None,
    lumped_sparam_components_f: list | None = None,
    lumped_sparam_impedances_f: list | None = None,
    lumped_sparam_cell_counts_f: list | None = None,
    lumped_sparam_freqs_f=None,
    ntff_box_f=None,
    ntff_data_f=None,
    use_material_sat: bool | None = None,
    sync_coarse_interface_from_fine: bool = False,
    sync_coarse_shadow_from_fine: bool = False,
    sync_box_coarse_shadow_from_fine: bool = False,
    mask_coarse_shadow_interior: bool = False,
    use_exterior_z_interfaces: bool = False,
    use_boundary_terminated_exterior_z_interfaces: bool = False,
    ghost_exterior_coarse_shadow_from_fine: bool = False,
    material_sat_scale: float = 1.0,
    material_sat_coarse_scale: float = 1.0,
    material_sat_fine_scale: float = 1.0,
    material_sat_e_coarse_scale: float = 1.0,
    material_sat_e_fine_scale: float = 1.0,
    material_sat_h_coarse_scale: float = 1.0,
    material_sat_h_fine_scale: float = 1.0,
    material_sat_zlo_scale: float = 1.0,
    material_sat_zhi_scale: float = 1.0,
    material_sat_e_zlo_scale: float = 1.0,
    material_sat_e_zhi_scale: float = 1.0,
    material_sat_h_zlo_scale: float = 1.0,
    material_sat_h_zhi_scale: float = 1.0,
    material_sat_pair_a_zlo_scale: float = 1.0,
    material_sat_pair_b_zlo_scale: float = 1.0,
    material_sat_zlo_common_trace_projection: str = "dual",
    material_sat_zhi_common_trace_projection: str = "dual",
    material_sat_normal_e_scale: float = 0.0,
    material_sat_zhi_coarse_eps_blend: float = 0.0,
    defer_material_h_sat_until_after_e: bool = False,
    material_sat_face_projection: str = "node_adjoint",
    inject_sources_before_e_coupling: bool = False,
    use_exterior_box_interfaces: bool = False,
    inject_sources_on_coarse_shadow: bool = False,
) -> SubgridResult:
    """Run subgridded FDTD via jax.lax.scan.

    Parameters
    ----------
    grid_c : Grid -- coarse grid (full domain, with or without CPML)
    mats_c : MaterialArrays -- coarse materials
    mats_f : MaterialArrays -- fine materials
    config : SubgridConfig3D
    n_steps : int
    pec_mask_c, pec_mask_f : boolean arrays or None
    sources_f : list of (i, j, k, component, waveform_array)
    sources_c : list of (i, j, k, component, waveform_array)
        Diagnostic-only coarse-shadow soft sources paired with ``sources_f``.
    probe_indices_f : list of (i, j, k)
    probe_components : list of component names
    probe_indices_c : list of (i, j, k), optional
        Diagnostic-only coarse-grid point probes. The high-level Simulation API
        records fine probes only; this stream helps interface-transfer
        diagnostics inspect the coarse side without changing public results.
    probe_components_c : list of component names, optional
    use_material_sat : bool or None
        Static selector for material-weighted artificial-interface coupling.
        ``None`` preserves the historical auto-detection from interface
        material arrays.  Pass an explicit bool when differentiating/jitting
        through material arrays so the branch decision is not made from a JAX
        tracer.  For contained-material research gates whose artificial
        interfaces remain vacuum, this should be ``False``.
    sync_coarse_interface_from_fine : bool
        Diagnostic-only experiment: after z-slab H/E coupling, overwrite the
        coarse tangential z-interface traces from the restricted fine traces.
        This is not a public support mode.
    sync_coarse_shadow_from_fine : bool
        Diagnostic-only experiment: keep the overlapping coarse shadow slab
        synchronized from colocated fine fields before/after updates. This is
        not a public support mode.
    sync_box_coarse_shadow_from_fine : bool
        Diagnostic-only local x/y experiment: keep a non-full-x/y overlapping
        coarse shadow box synchronized from colocated fine fields before/after
        updates. This tests local-box transfer ownership and is not a public
        support mode.  The sync overwrites the full coarse-shadow volume from
        the colocated fine samples on every E/H component before and after the
        H and E updates.
    mask_coarse_shadow_interior : bool
        Diagnostic-only experiment: zero the overlapping coarse shadow interior
        while preserving the z-interface faces. This is not a public support
        mode.
    use_exterior_z_interfaces : bool
        Diagnostic-only experiment: couple fine z faces to the immediately
        exterior coarse planes and zero the full overlapping coarse shadow slab.
        This approximates disjoint coarse/fine ownership for falsification
        experiments and is not a public support mode.
    use_boundary_terminated_exterior_z_interfaces : bool
        Diagnostic-only experiment: exterior/disjoint-like ownership for slabs
        that touch a physical PEC z boundary.  SAT coupling is skipped on
        physical z-boundary faces and retained only on sides with an active
        exterior coarse plane.  This tests whether the 83.3% exterior-coverage
        ceiling is the remaining broad-waveform blocker; it is not a public
        support mode.
    ghost_exterior_coarse_shadow_from_fine : bool
        Diagnostic-only experiment for exterior/disjoint-like ownership: before
        each coarse H/E update, populate the otherwise fine-owned coarse shadow
        slab from colocated fine fields so coarse exterior curl stencils see
        fine-grid ghost values instead of zeros.  The shadow slab is masked
        again after the coarse update; it remains non-owned and is not a public
        support mode.
    material_sat_scale : float
        Diagnostic-only multiplier on the impedance-upwind/material SAT
        corrections.  The production value is 1.0; any other value is a
        claims-blocking research knob.
    material_sat_coarse_scale, material_sat_fine_scale : float
        Diagnostic-only side-specific multipliers on coarse/fine material SAT
        corrections.  These probe whether the exterior/two-interface residual
        is caused by an incorrect side norm in the current closure.  Production
        value is 1.0 for both.
    material_sat_e_coarse_scale, material_sat_e_fine_scale,
    material_sat_h_coarse_scale, material_sat_h_fine_scale : float
        Diagnostic-only E/H-specific side multipliers layered on top of the
        side scales.  These probe whether the centered residual is an E/H phase
        split rather than one scalar side norm.  Production value is 1.0 for all.
    material_sat_zlo_scale, material_sat_zhi_scale : float
        Diagnostic-only z-interface-specific multipliers layered on top of the
        side and E/H scales.  These test lower-vs-upper two-interface energy
        balance.  Production value is 1.0 for both.
    material_sat_e_zlo_scale, material_sat_e_zhi_scale,
    material_sat_h_zlo_scale, material_sat_h_zhi_scale : float
        Diagnostic-only E/H-specific z-interface multipliers layered on top of
        the z-interface scale.  These test lower-interface phase/field balance.
        Production value is 1.0 for all.
    material_sat_pair_a_zlo_scale, material_sat_pair_b_zlo_scale : float
        Diagnostic-only lower-interface tangential-pair multipliers.  Pair A is
        ``(E_x, H_y)`` and pair B is ``(E_y, -H_x)``.  These test whether the
        centered residual is a tangential transfer imbalance rather than an
        E/H-wide or interface-wide scalar.  Production value is 1.0 for both.
    material_sat_zlo_common_trace_projection : {"dual", "coarse", "fine", "average"}
        Diagnostic-only selector for the lower z-interface common trace used by
        material SAT.  ``"dual"`` is the current behavior: coarse and fine
        sides compute their own projected common states.  ``"coarse"`` computes
        the common trace on the coarse face and prolongs it to the fine side;
        ``"fine"`` computes the common trace on the fine face and restricts it
        to the coarse side.  ``"average"`` averages the projected coarse-face
        and fine-face common states before applying both sides.  This probes
        whether centered lower-interface residuals come from non-single-valued
        coarse/fine trace construction.  Production value is ``"dual"``.
    material_sat_zhi_common_trace_projection : {"dual", "coarse", "fine", "average"}
        Diagnostic-only selector for the upper z-interface common trace used by
        material SAT.  This mirrors ``material_sat_zlo_common_trace_projection``
        and probes whether material-jump phase drift comes from non-single-valued
        coarse/fine trace construction at the artificial dielectric/vacuum
        interface.  Production value is ``"dual"``.
    material_sat_normal_e_scale : float
        Diagnostic-only normal electric displacement continuity penalty on
        material z interfaces.  The current production material SAT couples
        tangential characteristic pairs only; nonzero values test whether
        dielectric/vacuum artificial-interface drift is dominated by missing
        normal-D transfer.  Production value is 0.0.
    material_sat_zhi_coarse_eps_blend : float
        Diagnostic-only blend for the coarse-side epsilon used by material SAT
        at the z-hi artificial interface.  ``0`` uses the coarse cell-center
        material, while ``1`` uses the restricted fine-side epsilon for the SAT
        impedance only.  This tests whether material-jump phase drift comes
        from coarse-side interface material aliasing.  Production value is 0.0.
    defer_material_h_sat_until_after_e : bool
        Diagnostic-only H-SAT ordering variant for material z slabs.  When
        true, material H-SAT is computed after the H update but withheld from
        the H state used by the subsequent E curl/E-SAT; the corrected H state
        is committed only after E-SAT.  This probes whether centered support
        needs a different time-centered two-interface operator.  Production
        value is False.
    material_sat_face_projection : {"node_adjoint", "sample"}
        Diagnostic-only material-SAT z-face restriction variant.  The
        production value is the current adjoint node-aligned projection;
        ``"sample"`` tests colocated fine-node restriction as a spatial
        transfer falsification.
    inject_sources_before_e_coupling : bool
        Diagnostic-only experiment: inject fine-grid electric sources before
        z-slab E SAT coupling instead of after. This tests source/SAT ordering
        as a blocker hypothesis and is not a public support mode.
    use_exterior_box_interfaces : bool
        Diagnostic-only local x/y experiment: for non-full-x/y boxes, couple
        fine faces to immediately exterior coarse planes and mask the coarse
        shadow volume, approximating disjoint local-box ownership.
    inject_sources_on_coarse_shadow : bool
        Diagnostic-only experiment: also inject soft source waveforms on the
        overlapping coarse grid when sources lie inside the refined region.
    """
    sources_f = sources_f or []
    sources_c = sources_c or []
    probe_indices_f = probe_indices_f or []
    probe_components = probe_components or []
    probe_indices_c = probe_indices_c or []
    probe_components_c = probe_components_c or []
    lumped_sparam_indices_f = lumped_sparam_indices_f or []
    lumped_sparam_components_f = lumped_sparam_components_f or []
    lumped_sparam_impedances_f = lumped_sparam_impedances_f or []
    lumped_sparam_cell_counts_f = lumped_sparam_cell_counts_f or []
    use_ntff = ntff_box_f is not None and ntff_data_f is not None

    dt = config.dt
    dx_c = config.dx_c
    dx_f = config.dx_f

    # Override coarse grid dt to match subgrid global timestep
    grid_c.dt = dt

    # ---- Subsystem flags (resolved at trace time, not inside scan) ----
    use_cpml = grid_c.cpml_layers > 0
    pad_x_lo = int(getattr(grid_c, "pad_x_lo", grid_c.cpml_layers))
    pad_x_hi = int(getattr(grid_c, "pad_x_hi", grid_c.cpml_layers))
    pad_y_lo = int(getattr(grid_c, "pad_y_lo", grid_c.cpml_layers))
    pad_y_hi = int(getattr(grid_c, "pad_y_hi", grid_c.cpml_layers))
    pad_z_lo = int(getattr(grid_c, "pad_z_lo", grid_c.cpml_layers))
    pad_z_hi = int(getattr(grid_c, "pad_z_hi", grid_c.cpml_layers))
    is_full_xy_z_slab = (
        config.fi_lo == pad_x_lo
        and config.fj_lo == pad_y_lo
        and config.fi_hi == grid_c.nx - pad_x_hi
        and config.fj_hi == grid_c.ny - pad_y_hi
    )
    grid_nx = int(getattr(grid_c, "nx", config.nx_c))
    grid_ny = int(getattr(grid_c, "ny", config.ny_c))
    grid_nz = int(getattr(grid_c, "nz", config.nz_c))
    fine_physical_pec_faces = set()
    if not use_cpml:
        if config.fi_lo <= pad_x_lo:
            fine_physical_pec_faces.add("x_lo")
        if config.fi_hi >= grid_nx - pad_x_hi:
            fine_physical_pec_faces.add("x_hi")
        if config.fj_lo <= pad_y_lo:
            fine_physical_pec_faces.add("y_lo")
        if config.fj_hi >= grid_ny - pad_y_hi:
            fine_physical_pec_faces.add("y_hi")
        if config.fk_lo <= pad_z_lo:
            fine_physical_pec_faces.add("z_lo")
        if config.fk_hi >= grid_nz - pad_z_hi:
            fine_physical_pec_faces.add("z_hi")
    else:
        pec_faces = getattr(grid_c, "pec_faces", set())
        if "x_lo" in pec_faces and config.fi_lo <= pad_x_lo:
            fine_physical_pec_faces.add("x_lo")
        if "x_hi" in pec_faces and config.fi_hi >= grid_nx - pad_x_hi:
            fine_physical_pec_faces.add("x_hi")
        if "y_lo" in pec_faces and config.fj_lo <= pad_y_lo:
            fine_physical_pec_faces.add("y_lo")
        if "y_hi" in pec_faces and config.fj_hi >= grid_ny - pad_y_hi:
            fine_physical_pec_faces.add("y_hi")
        if "z_lo" in pec_faces and config.fk_lo <= pad_z_lo:
            fine_physical_pec_faces.add("z_lo")
        if "z_hi" in pec_faces and config.fk_hi >= grid_nz - pad_z_hi:
            fine_physical_pec_faces.add("z_hi")
    use_exterior_owned_z_slab = (
        use_exterior_z_interfaces
        or use_boundary_terminated_exterior_z_interfaces
    )
    use_exterior_owned_box = bool(use_exterior_box_interfaces) and not is_full_xy_z_slab
    sync_box_shadow_from_fine = (
        bool(sync_box_coarse_shadow_from_fine) and not is_full_xy_z_slab
    )

    def _sync_box_shadow(fields_c, fields_f, component_names):
        return _sync_z_slab_coarse_shadow_from_fine(
            fields_c,
            fields_f,
            config,
        )
    use_exterior_shadow_ghost = (
        ghost_exterior_coarse_shadow_from_fine
        and use_exterior_owned_z_slab
        and is_full_xy_z_slab
    )
    if use_exterior_z_interfaces and use_boundary_terminated_exterior_z_interfaces:
        raise ValueError(
            "use_exterior_z_interfaces and "
            "use_boundary_terminated_exterior_z_interfaces are mutually exclusive"
        )
    if use_exterior_z_interfaces:
        if not is_full_xy_z_slab:
            raise ValueError(
                "use_exterior_z_interfaces is only implemented for full-x/y z slabs"
            )
        if config.fk_lo <= 0 or config.fk_hi >= config.nz_c:
            raise ValueError(
                "use_exterior_z_interfaces needs active coarse cells immediately "
                "outside both z-slab faces"
            )
    if use_boundary_terminated_exterior_z_interfaces:
        if not is_full_xy_z_slab:
            raise ValueError(
                "use_boundary_terminated_exterior_z_interfaces is only "
                "implemented for full-x/y z slabs"
            )
        if use_cpml and not (
            ("z_lo" in fine_physical_pec_faces)
            ^ ("z_hi" in fine_physical_pec_faces)
        ):
            raise ValueError(
                "use_boundary_terminated_exterior_z_interfaces is only "
                "implemented when exactly one refined z face touches a PEC "
                "physical boundary; fine-grid CPML faces are not implemented"
            )
        if config.fk_lo > 0 and config.fk_hi < config.nz_c:
            raise ValueError(
                "use_boundary_terminated_exterior_z_interfaces requires the "
                "slab to touch at least one physical z boundary"
            )
        if config.fk_lo <= 0:
            fine_physical_pec_faces.add("z_lo")
        if config.fk_hi >= config.nz_c:
            fine_physical_pec_faces.add("z_hi")
    box_artificial_face_kwargs = {
        "exterior_interfaces": use_exterior_owned_box,
        "couple_xlo": (
            "x_lo" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fi_lo > 0)
        ),
        "couple_xhi": (
            "x_hi" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fi_hi < config.nx_c)
        ),
        "couple_ylo": (
            "y_lo" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fj_lo > 0)
        ),
        "couple_yhi": (
            "y_hi" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fj_hi < config.ny_c)
        ),
        "couple_zlo": (
            "z_lo" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fk_lo > 0)
        ),
        "couple_zhi": (
            "z_hi" not in fine_physical_pec_faces
            and (not use_exterior_owned_box or config.fk_hi < config.nz_c)
        ),
    }
    c_xy = (slice(config.fi_lo, config.fi_hi), slice(config.fj_lo, config.fj_hi))
    f_xy = (slice(None), slice(None))

    def _face_nonvacuum(mats, face_slice):
        return (
            jnp.any(jnp.abs(mats.eps_r[face_slice] - 1.0) > 1e-12)
            or jnp.any(jnp.abs(mats.mu_r[face_slice] - 1.0) > 1e-12)
        )

    # Material weighting is needed only when the artificial coarse/fine
    # interface itself sees non-vacuum impedance.  Material discontinuities
    # wholly inside the fine slab should use ordinary Yee material updates
    # internally and the established vacuum z-interface SAT at the slab faces.
    if use_material_sat is None:
        use_material_sat = bool(
            _face_nonvacuum(mats_c, (*c_xy, config.fk_lo))
            or _face_nonvacuum(mats_c, (*c_xy, config.fk_hi - 1))
            or _face_nonvacuum(mats_f, (*f_xy, 0))
            or _face_nonvacuum(mats_f, (*f_xy, -1))
        )

    cpml_params = None
    cpml_state = None
    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
        cpml_params, cpml_state = init_cpml(grid_c)

    # Initialize field states
    shape_c = (config.nx_c, config.ny_c, config.nz_c)
    shape_f = (config.nx_f, config.ny_f, config.nz_f)

    state_c = init_state(shape_c)
    state_f = init_state(shape_f)

    # Precompute source waveforms matrix (n_steps, n_sources)
    if sources_f:
        src_waveforms = jnp.stack([jnp.array(s[4]) for s in sources_f], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)
    if sources_c and inject_sources_on_coarse_shadow:
        src_waveforms_c = jnp.stack([jnp.array(s[4]) for s in sources_c], axis=-1)
    else:
        src_waveforms_c = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    src_meta = [(s[0], s[1], s[2], s[3]) for s in sources_f]
    src_meta_c = (
        [(s[0], s[1], s[2], s[3]) for s in sources_c]
        if inject_sources_on_coarse_shadow
        else []
    )
    prb_meta = [(p[0], p[1], p[2], c) for p, c in
                zip(probe_indices_f, probe_components)]
    n_probes = len(prb_meta)
    prb_meta_c = [(p[0], p[1], p[2], c) for p, c in
                  zip(probe_indices_c, probe_components_c)]
    n_probes_c = len(prb_meta_c)
    lumped_sparam_meta = [
        (p[0], p[1], p[2], c, float(z0))
        for p, c, z0 in zip(
            lumped_sparam_indices_f,
            lumped_sparam_components_f,
            lumped_sparam_impedances_f,
        )
    ]
    n_lumped_sparams = len(lumped_sparam_meta)
    if lumped_sparam_freqs_f is None:
        lumped_sparam_freqs_arr = jnp.zeros((0,), dtype=jnp.float32)
    else:
        lumped_sparam_freqs_arr = jnp.asarray(lumped_sparam_freqs_f, dtype=jnp.float32)
    lumped_sparam_impedances_arr = (
        jnp.asarray(lumped_sparam_impedances_f, dtype=jnp.float32)
        if n_lumped_sparams
        else jnp.zeros((0,), dtype=jnp.float32)
    )
    lumped_sparam_cell_counts_arr = (
        jnp.asarray(lumped_sparam_cell_counts_f, dtype=jnp.float32)
        if n_lumped_sparams
        else jnp.zeros((0,), dtype=jnp.float32)
    )

    use_pec_mask_c = pec_mask_c is not None
    use_pec_mask_f = pec_mask_f is not None

    # Pack carry — include CPML state only when CPML is active
    carry_init = {
        "c": (state_c.ex, state_c.ey, state_c.ez,
              state_c.hx, state_c.hy, state_c.hz),
        "f": (state_f.ex, state_f.ey, state_f.ez,
              state_f.hx, state_f.hy, state_f.hz),
    }
    if use_cpml:
        carry_init["cpml"] = cpml_state
    if n_lumped_sparams > 0:
        dft_shape = (n_lumped_sparams, lumped_sparam_freqs_arr.shape[0])
        carry_init["lumped_sparam_v"] = jnp.zeros(dft_shape, dtype=jnp.complex64)
        carry_init["lumped_sparam_i"] = jnp.zeros(dft_shape, dtype=jnp.complex64)
    if use_ntff:
        carry_init["ntff"] = ntff_data_f

    cpml_axes = "xyz"

    def step_fn(carry, xs):
        step_idx, src_vals, src_vals_c = xs
        ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = carry["c"]
        ex_f, ey_f, ez_f, hx_f, hy_f, hz_f = carry["f"]

        def _inject_fine_sources(ex_arr, ey_arr, ez_arr):
            for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
                if sc == "ez":
                    ez_arr = ez_arr.at[si, sj, sk].add(src_vals[idx_s])
                elif sc == "ex":
                    ex_arr = ex_arr.at[si, sj, sk].add(src_vals[idx_s])
                elif sc == "ey":
                    ey_arr = ey_arr.at[si, sj, sk].add(src_vals[idx_s])
            return ex_arr, ey_arr, ez_arr

        def _inject_coarse_sources(ex_arr, ey_arr, ez_arr):
            for idx_s, (si, sj, sk, sc) in enumerate(src_meta_c):
                if sc == "ez":
                    ez_arr = ez_arr.at[si, sj, sk].add(src_vals_c[idx_s])
                elif sc == "ex":
                    ex_arr = ex_arr.at[si, sj, sk].add(src_vals_c[idx_s])
                elif sc == "ey":
                    ey_arr = ey_arr.at[si, sj, sk].add(src_vals_c[idx_s])
            return ex_arr, ey_arr, ez_arr

        if sync_coarse_shadow_from_fine and is_full_xy_z_slab:
            ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _sync_z_slab_coarse_shadow_from_fine(
                (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                (ex_f, ey_f, ez_f, hx_f, hy_f, hz_f),
                config,
            )
        if sync_box_shadow_from_fine:
            ex_c, ey_c, ez_c = _sync_box_shadow(
                (ex_c, ey_c, ez_c),
                (ex_f, ey_f, ez_f),
                ("ex", "ey", "ez"),
            )
            hx_c, hy_c, hz_c = _sync_box_shadow(
                (hx_c, hy_c, hz_c),
                (hx_f, hy_f, hz_f),
                ("hx", "hy", "hz"),
            )
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            if use_exterior_shadow_ghost:
                ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _sync_z_slab_coarse_shadow_from_fine(
                    (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                    (ex_f, ey_f, ez_f, hx_f, hy_f, hz_f),
                    config,
                )
            else:
                ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _mask_z_slab_coarse_shadow_all(
                    (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                    config,
                )
        elif mask_coarse_shadow_interior and is_full_xy_z_slab:
            ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _mask_z_slab_coarse_shadow_interior(
                (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                config,
            )
        elif use_exterior_owned_box:
            ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _mask_box_coarse_shadow_all(
                (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
                config,
            )

        # === Coarse H update ===
        st_c = FDTDState(ex=ex_c, ey=ey_c, ez=ez_c,
                         hx=hx_c, hy=hy_c, hz=hz_c,
                         step=step_idx)
        st_c = update_h(st_c, mats_c, dt, dx_c)
        if use_cpml:
            st_c, cpml_new = apply_cpml_h(st_c, cpml_params, carry["cpml"],
                                           grid_c, cpml_axes)
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            hx_c, hy_c, hz_c = _mask_z_slab_coarse_shadow_all(
                (st_c.hx, st_c.hy, st_c.hz),
                config,
            )
            st_c = st_c._replace(hx=hx_c, hy=hy_c, hz=hz_c)
        elif use_exterior_owned_box:
            hx_c, hy_c, hz_c = _mask_box_coarse_shadow_all(
                (st_c.hx, st_c.hy, st_c.hz),
                config,
            )
            st_c = st_c._replace(hx=hx_c, hy=hy_c, hz=hz_c)

        # === Fine H update ===
        st_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                         hx=hx_f, hy=hy_f, hz=hz_f,
                         step=step_idx)
        st_f = update_h(st_f, mats_f, dt, dx_f)

        # === SAT H-coupling (tangential H on all 6 faces) ===
        if is_full_xy_z_slab:
            if use_material_sat:
                (hx_c_new, hy_c_new, hz_c_new), (hx_f_new, hy_f_new, hz_f_new) = \
                    _z_slab_material_coupling_h_3d(
                        (st_c.ex, st_c.ey, st_c.ez, st_c.hx, st_c.hy, st_c.hz),
                        (st_f.ex, st_f.ey, st_f.ez, st_f.hx, st_f.hy, st_f.hz),
                        mats_c,
                        mats_f,
                        config,
                        use_exterior_z_interfaces=use_exterior_z_interfaces,
                        use_boundary_terminated_exterior_z_interfaces=(
                            use_boundary_terminated_exterior_z_interfaces
                        ),
                        material_sat_scale=material_sat_scale,
                        material_sat_coarse_scale=material_sat_coarse_scale,
                        material_sat_fine_scale=material_sat_fine_scale,
                        material_sat_h_coarse_scale=material_sat_h_coarse_scale,
                        material_sat_h_fine_scale=material_sat_h_fine_scale,
                        material_sat_zlo_scale=material_sat_zlo_scale,
                        material_sat_zhi_scale=material_sat_zhi_scale,
                        material_sat_h_zlo_scale=material_sat_h_zlo_scale,
                        material_sat_h_zhi_scale=material_sat_h_zhi_scale,
                        material_sat_pair_a_zlo_scale=(
                            material_sat_pair_a_zlo_scale
                        ),
                        material_sat_pair_b_zlo_scale=(
                            material_sat_pair_b_zlo_scale
                        ),
                        material_sat_zlo_common_trace_projection=(
                            material_sat_zlo_common_trace_projection
                        ),
                        material_sat_zhi_common_trace_projection=(
                            material_sat_zhi_common_trace_projection
                        ),
                        material_sat_zhi_coarse_eps_blend=material_sat_zhi_coarse_eps_blend,
                        material_sat_face_projection=material_sat_face_projection,
                    )
            else:
                (hx_c_new, hy_c_new, hz_c_new), (hx_f_new, hy_f_new, hz_f_new) = \
                    _z_slab_coupling_h_3d(
                        (st_c.hx, st_c.hy, st_c.hz),
                        (st_f.hx, st_f.hy, st_f.hz),
                        config,
                        use_exterior_z_interfaces=use_exterior_z_interfaces,
                        use_boundary_terminated_exterior_z_interfaces=(
                            use_boundary_terminated_exterior_z_interfaces
                        ),
                    )
        else:
            if _uses_endpoint_fine_shape(config):
                (hx_c_new, hy_c_new, hz_c_new), (hx_f_new, hy_f_new, hz_f_new) = \
                    _box_node_aligned_coupling_h_3d(
                        (st_c.hx, st_c.hy, st_c.hz),
                        (st_f.hx, st_f.hy, st_f.hz),
                        config,
                        **box_artificial_face_kwargs,
                    )
            else:
                (hx_c_new, hy_c_new, hz_c_new), (hx_f_new, hy_f_new, hz_f_new) = \
                    _shared_node_coupling_h_3d(
                        (st_c.hx, st_c.hy, st_c.hz),
                        (st_f.hx, st_f.hy, st_f.hz),
                        config,
                    )
        if sync_coarse_interface_from_fine and is_full_xy_z_slab:
            hx_c_new, hy_c_new = _sync_z_slab_coarse_tangential_from_fine(
                hx_c_new,
                hy_c_new,
                hx_f_new,
                hy_f_new,
                config,
                use_exterior_z_interfaces=use_exterior_z_interfaces,
                use_boundary_terminated_exterior_z_interfaces=(
                    use_boundary_terminated_exterior_z_interfaces
                ),
            )
        if sync_coarse_shadow_from_fine and is_full_xy_z_slab:
            hx_c_new, hy_c_new, hz_c_new = _sync_z_slab_coarse_shadow_from_fine(
                (hx_c_new, hy_c_new, hz_c_new),
                (hx_f_new, hy_f_new, hz_f_new),
                config,
            )
        if sync_box_shadow_from_fine:
            hx_c_new, hy_c_new, hz_c_new = _sync_box_shadow(
                (hx_c_new, hy_c_new, hz_c_new),
                (hx_f_new, hy_f_new, hz_f_new),
                ("hx", "hy", "hz"),
            )
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            hx_c_new, hy_c_new, hz_c_new = _mask_z_slab_coarse_shadow_all(
                (hx_c_new, hy_c_new, hz_c_new),
                config,
            )
        elif mask_coarse_shadow_interior and is_full_xy_z_slab:
            hx_c_new, hy_c_new, hz_c_new = _mask_z_slab_coarse_shadow_interior(
                (hx_c_new, hy_c_new, hz_c_new),
                config,
            )
        elif use_exterior_owned_box:
            hx_c_new, hy_c_new, hz_c_new = _mask_box_coarse_shadow_all(
                (hx_c_new, hy_c_new, hz_c_new),
                config,
            )
        defer_h_sat_for_e = (
            defer_material_h_sat_until_after_e
            and is_full_xy_z_slab
            and use_material_sat
        )
        if not defer_h_sat_for_e:
            st_c = st_c._replace(hx=hx_c_new, hy=hy_c_new, hz=hz_c_new)
            st_f = st_f._replace(hx=hx_f_new, hy=hy_f_new, hz=hz_f_new)
        if use_exterior_shadow_ghost and not defer_h_sat_for_e:
            ex_c_ghost, ey_c_ghost, ez_c_ghost, hx_c_ghost, hy_c_ghost, hz_c_ghost = (
                _sync_z_slab_coarse_shadow_from_fine(
                    (st_c.ex, st_c.ey, st_c.ez, st_c.hx, st_c.hy, st_c.hz),
                    (st_f.ex, st_f.ey, st_f.ez, st_f.hx, st_f.hy, st_f.hz),
                    config,
                )
            )
            st_c = st_c._replace(
                ex=ex_c_ghost,
                ey=ey_c_ghost,
                ez=ez_c_ghost,
                hx=hx_c_ghost,
                hy=hy_c_ghost,
                hz=hz_c_ghost,
            )

        # === Coarse E update + boundary ===
        st_c = update_e(st_c, mats_c, dt, dx_c)
        if use_cpml:
            st_c, cpml_new = apply_cpml_e(st_c, cpml_params, cpml_new,
                                           grid_c, cpml_axes)
        st_c = apply_pec(st_c)
        if use_pec_mask_c:
            st_c = apply_pec_mask(st_c, pec_mask_c)
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            ex_c, ey_c, ez_c = _mask_z_slab_coarse_shadow_all(
                (st_c.ex, st_c.ey, st_c.ez),
                config,
            )
            st_c = st_c._replace(ex=ex_c, ey=ey_c, ez=ez_c)
        elif use_exterior_owned_box:
            ex_c, ey_c, ez_c = _mask_box_coarse_shadow_all(
                (st_c.ex, st_c.ey, st_c.ez),
                config,
            )
            st_c = st_c._replace(ex=ex_c, ey=ey_c, ez=ez_c)

        # === Fine E update + PEC mask ===
        st_f = update_e(st_f, mats_f, dt, dx_f)
        if use_pec_mask_f:
            st_f = apply_pec_mask(st_f, pec_mask_f)
        if fine_physical_pec_faces:
            st_f = apply_pec_faces(st_f, fine_physical_pec_faces)
        if inject_sources_before_e_coupling:
            ex_src, ey_src, ez_src = _inject_fine_sources(st_f.ex, st_f.ey, st_f.ez)
            st_f = st_f._replace(ex=ex_src, ey=ey_src, ez=ez_src)

        # === SBP-SAT coupling ===
        if is_full_xy_z_slab:
            if use_material_sat:
                (ex_c_new, ey_c_new, ez_c_new), (ex_f_new, ey_f_new, ez_f_new) = \
                    _z_slab_material_coupling_e_3d(
                        (
                            st_c.ex,
                            st_c.ey,
                            st_c.ez,
                            st_c.hx,
                            st_c.hy,
                            st_c.hz,
                        ),
                        (
                            st_f.ex,
                            st_f.ey,
                            st_f.ez,
                            st_f.hx,
                            st_f.hy,
                            st_f.hz,
                        ),
                        mats_c,
                        mats_f,
                        config,
                        use_exterior_z_interfaces=use_exterior_z_interfaces,
                        use_boundary_terminated_exterior_z_interfaces=(
                            use_boundary_terminated_exterior_z_interfaces
                        ),
                        material_sat_scale=material_sat_scale,
                        material_sat_coarse_scale=material_sat_coarse_scale,
                        material_sat_fine_scale=material_sat_fine_scale,
                        material_sat_e_coarse_scale=material_sat_e_coarse_scale,
                        material_sat_e_fine_scale=material_sat_e_fine_scale,
                        material_sat_zlo_scale=material_sat_zlo_scale,
                        material_sat_zhi_scale=material_sat_zhi_scale,
                        material_sat_e_zlo_scale=material_sat_e_zlo_scale,
                        material_sat_e_zhi_scale=material_sat_e_zhi_scale,
                        material_sat_pair_a_zlo_scale=(
                            material_sat_pair_a_zlo_scale
                        ),
                        material_sat_pair_b_zlo_scale=(
                            material_sat_pair_b_zlo_scale
                        ),
                        material_sat_zlo_common_trace_projection=(
                            material_sat_zlo_common_trace_projection
                        ),
                        material_sat_zhi_common_trace_projection=(
                            material_sat_zhi_common_trace_projection
                        ),
                        material_sat_normal_e_scale=material_sat_normal_e_scale,
                        material_sat_zhi_coarse_eps_blend=material_sat_zhi_coarse_eps_blend,
                        material_sat_face_projection=material_sat_face_projection,
                    )
            else:
                (ex_c_new, ey_c_new, ez_c_new), (ex_f_new, ey_f_new, ez_f_new) = \
                    _z_slab_coupling_e_3d(
                        (st_c.ex, st_c.ey, st_c.ez),
                        (st_f.ex, st_f.ey, st_f.ez),
                        config,
                        use_exterior_z_interfaces=use_exterior_z_interfaces,
                        use_boundary_terminated_exterior_z_interfaces=(
                            use_boundary_terminated_exterior_z_interfaces
                        ),
                    )
        else:
            if _uses_endpoint_fine_shape(config):
                (ex_c_new, ey_c_new, ez_c_new), (ex_f_new, ey_f_new, ez_f_new) = \
                    _box_node_aligned_coupling_e_3d(
                        (st_c.ex, st_c.ey, st_c.ez),
                        (st_f.ex, st_f.ey, st_f.ez),
                        config,
                        **box_artificial_face_kwargs,
                    )
            else:
                (ex_c_new, ey_c_new, ez_c_new), (ex_f_new, ey_f_new, ez_f_new) = \
                    _shared_node_coupling_3d(
                        (st_c.ex, st_c.ey, st_c.ez),
                        (st_f.ex, st_f.ey, st_f.ez),
                        config,
                    )
        if sync_coarse_interface_from_fine and is_full_xy_z_slab:
            ex_c_new, ey_c_new = _sync_z_slab_coarse_tangential_from_fine(
                ex_c_new,
                ey_c_new,
                ex_f_new,
                ey_f_new,
                config,
                use_exterior_z_interfaces=use_exterior_z_interfaces,
                use_boundary_terminated_exterior_z_interfaces=(
                    use_boundary_terminated_exterior_z_interfaces
                ),
            )

        if n_lumped_sparams > 0:
            phase = jnp.exp(
                -1j
                * 2.0
                * jnp.pi
                * lumped_sparam_freqs_arr
                * (step_idx.astype(jnp.float32) * dt)
            )
            v_dft = carry["lumped_sparam_v"]
            i_dft = carry["lumped_sparam_i"]

            def _sample_lumped_vi(i, j, k, comp):
                if comp == "ez":
                    voltage = -ez_f_new[i, j, k] * dx_f
                    current = (
                        st_f.hy[i, j, k]
                        - st_f.hy[i - 1, j, k]
                        - st_f.hx[i, j, k]
                        + st_f.hx[i, j - 1, k]
                    ) * dx_f
                elif comp == "ex":
                    voltage = -ex_f_new[i, j, k] * dx_f
                    current = (
                        st_f.hz[i, j, k]
                        - st_f.hz[i, j - 1, k]
                        - st_f.hy[i, j, k]
                        + st_f.hy[i, j, k - 1]
                    ) * dx_f
                elif comp == "ey":
                    voltage = -ey_f_new[i, j, k] * dx_f
                    current = (
                        st_f.hx[i, j, k]
                        - st_f.hx[i, j, k - 1]
                        - st_f.hz[i, j, k]
                        + st_f.hz[i - 1, j, k]
                    ) * dx_f
                else:
                    raise ValueError(f"Unknown lumped S-parameter component: {comp!r}")
                return voltage, current

            for idx_lp, (li, lj, lk, lc, _z0) in enumerate(lumped_sparam_meta):
                voltage, current = _sample_lumped_vi(li, lj, lk, lc)
                v_dft = v_dft.at[idx_lp].add(voltage * phase * dt)
                i_dft = i_dft.at[idx_lp].add(current * phase * dt)

        # === Source injection on fine grid ===
        if not inject_sources_before_e_coupling:
            ex_f_new, ey_f_new, ez_f_new = _inject_fine_sources(
                ex_f_new,
                ey_f_new,
                ez_f_new,
            )

        if not use_cpml:
            st_c_after = apply_pec(
                FDTDState(
                    ex=ex_c_new, ey=ey_c_new, ez=ez_c_new,
                    hx=st_c.hx, hy=st_c.hy, hz=st_c.hz,
                    step=step_idx,
                )
            )
            ex_c_new, ey_c_new, ez_c_new = st_c_after.ex, st_c_after.ey, st_c_after.ez
        if fine_physical_pec_faces:
            st_f_after = apply_pec_faces(
                FDTDState(
                    ex=ex_f_new, ey=ey_f_new, ez=ez_f_new,
                    hx=st_f.hx, hy=st_f.hy, hz=st_f.hz,
                    step=step_idx,
                ),
                fine_physical_pec_faces,
            )
            ex_f_new, ey_f_new, ez_f_new = st_f_after.ex, st_f_after.ey, st_f_after.ez
        if sync_coarse_shadow_from_fine and is_full_xy_z_slab:
            ex_c_new, ey_c_new, ez_c_new = _sync_z_slab_coarse_shadow_from_fine(
                (ex_c_new, ey_c_new, ez_c_new),
                (ex_f_new, ey_f_new, ez_f_new),
                config,
            )
        if sync_box_shadow_from_fine:
            ex_c_new, ey_c_new, ez_c_new = _sync_box_shadow(
                (ex_c_new, ey_c_new, ez_c_new),
                (ex_f_new, ey_f_new, ez_f_new),
                ("ex", "ey", "ez"),
            )
        if use_exterior_owned_z_slab and is_full_xy_z_slab:
            ex_c_new, ey_c_new, ez_c_new = _mask_z_slab_coarse_shadow_all(
                (ex_c_new, ey_c_new, ez_c_new),
                config,
            )
        elif mask_coarse_shadow_interior and is_full_xy_z_slab:
            ex_c_new, ey_c_new, ez_c_new = _mask_z_slab_coarse_shadow_interior(
                (ex_c_new, ey_c_new, ez_c_new),
                config,
            )
        elif use_exterior_owned_box:
            ex_c_new, ey_c_new, ez_c_new = _mask_box_coarse_shadow_all(
                (ex_c_new, ey_c_new, ez_c_new),
                config,
            )
        if inject_sources_on_coarse_shadow and src_meta_c:
            ex_c_new, ey_c_new, ez_c_new = _inject_coarse_sources(
                ex_c_new,
                ey_c_new,
                ez_c_new,
            )
        if defer_h_sat_for_e:
            st_c = st_c._replace(hx=hx_c_new, hy=hy_c_new, hz=hz_c_new)
            st_f = st_f._replace(hx=hx_f_new, hy=hy_f_new, hz=hz_f_new)

        # === Probe samples ===
        if n_probes > 0:
            def _get_field(comp, i, j, k):
                if comp == "ez":
                    return ez_f_new[i, j, k]
                if comp == "ex":
                    return ex_f_new[i, j, k]
                if comp == "ey":
                    return ey_f_new[i, j, k]
                if comp == "hx":
                    return st_f.hx[i, j, k]
                if comp == "hy":
                    return st_f.hy[i, j, k]
                return st_f.hz[i, j, k]

            samples = [_get_field(pc, pi, pj, pk)
                       for pi, pj, pk, pc in prb_meta]
            probe_out = jnp.stack(samples)
        else:
            probe_out = jnp.zeros(0, dtype=jnp.float32)

        if n_probes_c > 0:
            def _get_coarse_field(comp, i, j, k):
                if comp == "ez":
                    return ez_c_new[i, j, k]
                if comp == "ex":
                    return ex_c_new[i, j, k]
                if comp == "ey":
                    return ey_c_new[i, j, k]
                if comp == "hx":
                    return st_c.hx[i, j, k]
                if comp == "hy":
                    return st_c.hy[i, j, k]
                return st_c.hz[i, j, k]

            coarse_samples = [_get_coarse_field(pc, pi, pj, pk)
                              for pi, pj, pk, pc in prb_meta_c]
            probe_out_c = jnp.stack(coarse_samples)
        else:
            probe_out_c = jnp.zeros(0, dtype=jnp.float32)

        if use_ntff:
            from rfx.farfield import accumulate_ntff

            ntff_state_f = FDTDState(
                ex=ex_f_new,
                ey=ey_f_new,
                ez=ez_f_new,
                hx=st_f.hx,
                hy=st_f.hy,
                hz=st_f.hz,
                step=step_idx,
            )
            ntff_new = accumulate_ntff(
                carry["ntff"],
                ntff_state_f,
                ntff_box_f,
                dt,
                step_idx,
            )

        new_carry = {
            "c": (ex_c_new, ey_c_new, ez_c_new,
                  st_c.hx, st_c.hy, st_c.hz),
            "f": (ex_f_new, ey_f_new, ez_f_new,
                  st_f.hx, st_f.hy, st_f.hz),
        }
        if use_cpml:
            new_carry["cpml"] = cpml_new
        if n_lumped_sparams > 0:
            new_carry["lumped_sparam_v"] = v_dft
            new_carry["lumped_sparam_i"] = i_dft
        if use_ntff:
            new_carry["ntff"] = ntff_new

        return new_carry, (probe_out, probe_out_c)

    # Run scan
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms, src_waveforms_c)
    final_carry, probe_series = jax.lax.scan(step_fn, carry_init, xs)
    time_series, time_series_c = probe_series

    # Unpack final state
    ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = final_carry["c"]
    ex_f, ey_f, ez_f, hx_f, hy_f, hz_f = final_carry["f"]

    final_c = FDTDState(ex=ex_c, ey=ey_c, ez=ez_c,
                        hx=hx_c, hy=hy_c, hz=hz_c,
                        step=jnp.array(n_steps, dtype=jnp.int32))
    final_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                        hx=hx_f, hy=hy_f, hz=hz_f,
                        step=jnp.array(n_steps, dtype=jnp.int32))

    return SubgridResult(
        state_c=final_c,
        state_f=final_f,
        time_series=time_series,
        config=config,
        dt=dt,
        time_series_c=time_series_c,
        lumped_sparam_v_dft_f=(
            final_carry["lumped_sparam_v"] if n_lumped_sparams > 0 else None
        ),
        lumped_sparam_i_dft_f=(
            final_carry["lumped_sparam_i"] if n_lumped_sparams > 0 else None
        ),
        lumped_sparam_freqs_f=(
            lumped_sparam_freqs_arr if n_lumped_sparams > 0 else None
        ),
        lumped_sparam_impedances_f=(
            lumped_sparam_impedances_arr if n_lumped_sparams > 0 else None
        ),
        lumped_sparam_cell_counts_f=(
            lumped_sparam_cell_counts_arr if n_lumped_sparams > 0 else None
        ),
        ntff_data_f=final_carry["ntff"] if use_ntff else None,
        ntff_box_f=ntff_box_f if use_ntff else None,
    )
