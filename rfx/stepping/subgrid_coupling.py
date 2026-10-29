from __future__ import annotations

from typing import NamedTuple
from rfx import _realized
import jax.numpy as jnp
from rfx.subgridding.material_sat import interface_pair_deltas
from rfx.core.yee import EPS_0, MU_0
from rfx.model.materials import sat_face_mu
from rfx.model.subgrid_options import SubgridRunOptions


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
    opts: SubgridRunOptions = SubgridRunOptions(),
):
    """Material-weighted H correction for z-lo/z-hi artificial interfaces."""
    from rfx.sources.wire_radius import require_radius_update
    for mats in (mats_c, mats_f):
        require_radius_update(mats, lane="subgridded", unsupported=True)
    # Unpack the relevant option fields into same-named locals so the body
    # below is byte-unchanged from the historical keyword-argument signature.
    use_exterior_z_interfaces = opts.use_exterior_z_interfaces
    use_boundary_terminated_exterior_z_interfaces = (
        opts.use_boundary_terminated_exterior_z_interfaces
    )
    material_sat_scale = opts.material_sat_scale
    material_sat_coarse_scale = opts.material_sat_coarse_scale
    material_sat_fine_scale = opts.material_sat_fine_scale
    material_sat_h_coarse_scale = opts.material_sat_h_coarse_scale
    material_sat_h_fine_scale = opts.material_sat_h_fine_scale
    material_sat_zlo_scale = opts.material_sat_zlo_scale
    material_sat_zhi_scale = opts.material_sat_zhi_scale
    material_sat_h_zlo_scale = opts.material_sat_h_zlo_scale
    material_sat_h_zhi_scale = opts.material_sat_h_zhi_scale
    material_sat_pair_a_zlo_scale = opts.material_sat_pair_a_zlo_scale
    material_sat_pair_b_zlo_scale = opts.material_sat_pair_b_zlo_scale
    material_sat_zlo_common_trace_projection = (
        opts.material_sat_zlo_common_trace_projection
    )
    material_sat_zhi_common_trace_projection = (
        opts.material_sat_zhi_common_trace_projection
    )
    material_sat_zhi_coarse_eps_blend = opts.material_sat_zhi_coarse_eps_blend
    material_sat_face_projection = opts.material_sat_face_projection
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

    def face_coeffs(materials, sl, site):
        eps = materials.eps_r[sl]
        mu = sat_face_mu(materials, sl)
        if _realized.ACTIVE is not None:
            # Each SAT pair consumes its own tangential H face material.
            eps, mu = _realized.face(materials, sl, eps, mu, site)
        return eps * EPS_0, tuple(component * MU_0 for component in mu)

    def face_coeffs_c(k):
        return face_coeffs(mats_c, (slice(fi, fi + ni), slice(fj, fj + nj), k), "sat.c")

    def face_coeffs_f(k):
        return face_coeffs(mats_f, (slice(None), slice(None), k), "sat.f")

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
            mu_lower=mu_c[1],
            epsilon_upper=down(eps_f),
            mu_upper=down(mu_f[1]),
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
            mu_lower=mu_c[0],
            epsilon_upper=down(eps_f),
            mu_upper=down(mu_f[0]),
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
                mu_lower=up(mu_c[1]),
                epsilon_upper=eps_f,
                mu_upper=mu_f[1],
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
                mu_lower=up(mu_c[0]),
                epsilon_upper=eps_f,
                mu_upper=mu_f[0],
                h_lower=config.dx_c,
                h_upper=config.dx_f,
                dt=config.dt,
            )
            if trace_projection == "fine":
                pair_a_dv_lower = -(config.dt / (mu_c[1] * config.dx_c)) * (
                    down(pair_a_f.trace.u_star) - ex_c[c]
                )
                pair_b_dv_lower = -(config.dt / (mu_c[0] * config.dx_c)) * (
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
                pair_a_dv_lower = -(config.dt / (mu_c[1] * config.dx_c)) * (
                    pair_a_u_star_c - ex_c[c]
                )
                pair_b_dv_lower = -(config.dt / (mu_c[0] * config.dx_c)) * (
                    pair_b_u_star_c - ey_c[c]
                )
                pair_a_dv_upper = +(config.dt / (mu_f[1] * config.dx_f)) * (
                    pair_a_u_star_f - ex_f[f]
                )
                pair_b_dv_upper = +(config.dt / (mu_f[0] * config.dx_f)) * (
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
                pair_a_dv_upper = +(config.dt / (mu_f[1] * config.dx_f)) * (
                    up(pair_a_c.trace.u_star) - ex_f[f]
                )
                pair_b_dv_upper = +(config.dt / (mu_f[0] * config.dx_f)) * (
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
                    mu_lower=up(mu_c[1]),
                    epsilon_upper=eps_f,
                    mu_upper=mu_f[1],
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
                    mu_lower=up(mu_c[0]),
                    epsilon_upper=eps_f,
                    mu_upper=mu_f[0],
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
            mu_lower=down(mu_f[1]),
            epsilon_upper=eps_c,
            mu_upper=mu_c[1],
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
            mu_lower=down(mu_f[0]),
            epsilon_upper=eps_c,
            mu_upper=mu_c[0],
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
                mu_lower=mu_f[1],
                epsilon_upper=up(eps_c),
                mu_upper=up(mu_c[1]),
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
                mu_lower=mu_f[0],
                epsilon_upper=up(eps_c),
                mu_upper=up(mu_c[0]),
                h_lower=config.dx_f,
                h_upper=config.dx_c,
                dt=config.dt,
            )
            if trace_projection == "fine":
                pair_a_dv_upper = +(config.dt / (mu_c[1] * config.dx_c)) * (
                    down(pair_a_f.trace.u_star) - ex_c[c]
                )
                pair_b_dv_upper = +(config.dt / (mu_c[0] * config.dx_c)) * (
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
                pair_a_dv_upper = +(config.dt / (mu_c[1] * config.dx_c)) * (
                    pair_a_u_star_c - ex_c[c]
                )
                pair_b_dv_upper = +(config.dt / (mu_c[0] * config.dx_c)) * (
                    pair_b_u_star_c - ey_c[c]
                )
                pair_a_dv_lower = -(config.dt / (mu_f[1] * config.dx_f)) * (
                    pair_a_u_star_f - ex_f[f]
                )
                pair_b_dv_lower = -(config.dt / (mu_f[0] * config.dx_f)) * (
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
                pair_a_dv_lower = -(config.dt / (mu_f[1] * config.dx_f)) * (
                    up(pair_a_c.trace.u_star) - ex_f[f]
                )
                pair_b_dv_lower = -(config.dt / (mu_f[0] * config.dx_f)) * (
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
                    mu_lower=mu_f[1],
                    epsilon_upper=up(eps_c),
                    mu_upper=up(mu_c[1]),
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
                    mu_lower=mu_f[0],
                    epsilon_upper=up(eps_c),
                    mu_upper=up(mu_c[0]),
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
    opts: SubgridRunOptions = SubgridRunOptions(),
):
    """Material-weighted E correction for z-lo/z-hi artificial interfaces."""
    from rfx.sources.wire_radius import require_radius_update
    for mats in (mats_c, mats_f):
        require_radius_update(mats, lane="subgridded", unsupported=True)
    # Unpack the relevant option fields into same-named locals so the body
    # below is byte-unchanged from the historical keyword-argument signature.
    use_exterior_z_interfaces = opts.use_exterior_z_interfaces
    use_boundary_terminated_exterior_z_interfaces = (
        opts.use_boundary_terminated_exterior_z_interfaces
    )
    material_sat_scale = opts.material_sat_scale
    material_sat_coarse_scale = opts.material_sat_coarse_scale
    material_sat_fine_scale = opts.material_sat_fine_scale
    material_sat_e_coarse_scale = opts.material_sat_e_coarse_scale
    material_sat_e_fine_scale = opts.material_sat_e_fine_scale
    material_sat_zlo_scale = opts.material_sat_zlo_scale
    material_sat_zhi_scale = opts.material_sat_zhi_scale
    material_sat_e_zlo_scale = opts.material_sat_e_zlo_scale
    material_sat_e_zhi_scale = opts.material_sat_e_zhi_scale
    material_sat_pair_a_zlo_scale = opts.material_sat_pair_a_zlo_scale
    material_sat_pair_b_zlo_scale = opts.material_sat_pair_b_zlo_scale
    material_sat_zlo_common_trace_projection = (
        opts.material_sat_zlo_common_trace_projection
    )
    material_sat_zhi_common_trace_projection = (
        opts.material_sat_zhi_common_trace_projection
    )
    material_sat_normal_e_scale = opts.material_sat_normal_e_scale
    material_sat_zhi_coarse_eps_blend = opts.material_sat_zhi_coarse_eps_blend
    material_sat_face_projection = opts.material_sat_face_projection
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

    def face_coeffs(materials, sl, site):
        eps = materials.eps_r[sl]
        mu = sat_face_mu(materials, sl)
        if _realized.ACTIVE is not None:
            # Each SAT pair consumes its own tangential H face material.
            eps, mu = _realized.face(materials, sl, eps, mu, site)
        return eps * EPS_0, tuple(component * MU_0 for component in mu)

    def face_coeffs_c(k):
        return face_coeffs(mats_c, (slice(fi, fi + ni), slice(fj, fj + nj), k), "sat.c")

    def face_coeffs_f(k):
        return face_coeffs(mats_f, (slice(None), slice(None), k), "sat.f")

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
            mu_lower=mu_c[1],
            epsilon_upper=down(eps_f),
            mu_upper=down(mu_f[1]),
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
            mu_lower=mu_c[0],
            epsilon_upper=down(eps_f),
            mu_upper=down(mu_f[0]),
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
                mu_lower=up(mu_c[1]),
                epsilon_upper=eps_f,
                mu_upper=mu_f[1],
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
                mu_lower=up(mu_c[0]),
                epsilon_upper=eps_f,
                mu_upper=mu_f[0],
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
                    mu_lower=up(mu_c[1]),
                    epsilon_upper=eps_f,
                    mu_upper=mu_f[1],
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
                    mu_lower=up(mu_c[0]),
                    epsilon_upper=eps_f,
                    mu_upper=mu_f[0],
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
            mu_lower=down(mu_f[1]),
            epsilon_upper=eps_c,
            mu_upper=mu_c[1],
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
            mu_lower=down(mu_f[0]),
            epsilon_upper=eps_c,
            mu_upper=mu_c[0],
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
                mu_lower=mu_f[1],
                epsilon_upper=up(eps_c),
                mu_upper=up(mu_c[1]),
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
                mu_lower=mu_f[0],
                epsilon_upper=up(eps_c),
                mu_upper=up(mu_c[0]),
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
                    mu_lower=mu_f[1],
                    epsilon_upper=up(eps_c),
                    mu_upper=up(mu_c[1]),
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
                    mu_lower=mu_f[0],
                    epsilon_upper=up(eps_c),
                    mu_upper=up(mu_c[0]),
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


