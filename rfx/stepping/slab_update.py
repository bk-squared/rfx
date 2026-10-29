"""Distributed slab update."""
from __future__ import annotations

from functools import partial
from jax import lax
from jax.sharding import PartitionSpec as P
from rfx import _realized
from rfx.core.drives import drive_layout
from rfx.core.drives import inject_drives
from rfx.core.yee import FDTDState
from rfx.core.yee import MU_0
from rfx.core.yee import MaterialArrays
from rfx.core.yee import _shift_bwd
from rfx.core.yee import _shift_fwd
from rfx.core.yee import cell_owned_component_materials
from rfx.core.yee import component_e_materials
from rfx.core.yee import component_h_materials
from rfx.core.yee import e_update_coeffs
from rfx.core.yee import map_lumped
from rfx.materials.debye import DebyeCoeffs
from rfx.materials.debye import DebyeState
from rfx.materials.debye import debye_e_coeffs
from rfx.materials.debye import debye_e_component
from rfx.materials.debye import per_component
from rfx.materials.lorentz import LorentzState
from rfx.materials.lorentz import lorentz_e_coeffs
from rfx.materials.lorentz import lorentz_e_component
from rfx.materials.lorentz import lorentz_p_component
from rfx.materials.lorentz import mixed_e_component_coeffs
from rfx.model.electric_metrics import _slab_model_rows
from rfx.model.electric_metrics import _slab_x_lo_view
from rfx.model.electric_metrics import material_drive_scales as _material_drive_scales
from rfx.model.electric_metrics import slab_cell_sizes
from rfx.model.electric_metrics import slab_e_component_materials
from rfx.model.electric_metrics import slab_metric_kwargs
from rfx.stepping.rank import rank_shard_map
from rfx.stepping.slab import fill_pole_rows
import jax
import jax.numpy as jnp
import numpy as np


material_drive_scales = partial(_material_drive_scales, reduce_devices=False)


def _update_h_local(state, materials, dt, dx, magnetic_factors=(None, None, None), *, high_wall: tuple[bool, bool, bool]):
    """H update on a local slab (including ghost cells).

    Identical to yee.update_h but without jit decorator and always
    non-periodic (ghost cells handle inter-device coupling).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    mu = component_h_materials(materials, high_wall=high_wall)
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "yee.H")
    mu_x, mu_y, mu_z = (m * MU_0 for m in mu)
    ch_x, ch_y, ch_z = dt / mu_x, dt / mu_y, dt / mu_z

    curl_x = (
        (_shift_fwd(ez, 1) - ez) / dx
        - (_shift_fwd(ey, 2) - ey) / dx
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) / dx
        - (_shift_fwd(ez, 0) - ez) / dx
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) / dx
        - (_shift_fwd(ex, 1) - ex) / dx
    )

    hx = state.hx - ch_x * curl_x
    hy = state.hy - ch_y * curl_y
    hz = state.hz - ch_z * curl_z

    from rfx.boundaries.pmc import magnetic_update_fields
    hx, hy, hz = magnetic_update_fields((hx, hy, hz), magnetic_factors)

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local(state, e_coeffs, dx, *, boundary=None, frame=None, owned=None):
    """E update on a local slab (including ghost cells).

    Identical to yee.update_e but without jit decorator and always
    non-periodic. ``e_coeffs`` is the slab's per-component
    ``((ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z))``, as :func:`slab_e_coeffs`
    builds them: from the mean eps and sigma over the four cells around each
    edge, with a lumped element on its own edge only -- the single-device
    rule (#1210, #1236, #1303).
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)
    (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z) = e_coeffs

    curl_x = (
        (hz - backward(hz, 1)) / dx
        - (hy - backward(hy, 2)) / dx
    )
    curl_y = (
        (hx - backward(hx, 2)) / dx
        - (hz - backward(hz, 0)) / dx
    )
    curl_z = (
        (hy - backward(hy, 0)) / dx
        - (hx - backward(hx, 1)) / dx
    )

    ex = ca_x * state.ex + cb_x * curl_x
    ey = ca_y * state.ey + cb_y * curl_y
    ez = ca_z * state.ez + cb_z * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


def _update_e_debye_local(state, debye_coeffs, debye_state, dt, dx, *, boundary=None, frame=None, owned=None):
    """E update with Debye ADE on a local slab (always non-periodic).

    Per E component (#1260), the single-device body's arithmetic
    (``rfx.materials.debye.debye_e_component``) without its dtype narrowing.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)

    curl_x = ((hz - backward(hz, 1)) - (hy - backward(hy, 2))) / dx
    curl_y = ((hx - backward(hx, 2)) - (hz - backward(hz, 0))) / dx
    curl_z = ((hy - backward(hy, 0)) - (hx - backward(hx, 1))) / dx

    ex_new, px_new = debye_e_component(debye_coeffs, 0, state.ex, curl_x,
                                       debye_state.px)
    ey_new, py_new = debye_e_component(debye_coeffs, 1, state.ey, curl_y,
                                       debye_state.py)
    ez_new, pz_new = debye_e_component(debye_coeffs, 2, state.ez, curl_z,
                                       debye_state.pz)

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_debye = DebyeState(px=px_new, py=py_new, pz=pz_new)
    return new_fdtd, new_debye


def _update_e_lorentz_local(state, lorentz_coeffs, lor_state, dt, dx, *, boundary=None, frame=None, owned=None):
    """E update with Lorentz/Drude ADE on a local slab (always non-periodic).

    Per E component (#1260), as ``rfx.stepping.poles.update_e_lorentz``
    without its dtype narrowing.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)

    curl_x = ((hz - backward(hz, 1)) - (hy - backward(hy, 2))) / dx
    curl_y = ((hx - backward(hx, 2)) - (hz - backward(hz, 0))) / dx
    curl_z = ((hy - backward(hy, 0)) - (hx - backward(hx, 1))) / dx

    px_new = lorentz_p_component(lorentz_coeffs, 0, state.ex, lor_state.px,
                                 lor_state.px_prev)
    py_new = lorentz_p_component(lorentz_coeffs, 1, state.ey, lor_state.py,
                                 lor_state.py_prev)
    pz_new = lorentz_p_component(lorentz_coeffs, 2, state.ez, lor_state.pz,
                                 lor_state.pz_prev)

    dpx = jnp.sum(px_new - lor_state.px, axis=0)
    dpy = jnp.sum(py_new - lor_state.py, axis=0)
    dpz = jnp.sum(pz_new - lor_state.pz, axis=0)

    ex_new = lorentz_e_component(lorentz_coeffs, 0, state.ex, curl_x, dpx)
    ey_new = lorentz_e_component(lorentz_coeffs, 1, state.ey, curl_y, dpy)
    ez_new = lorentz_e_component(lorentz_coeffs, 2, state.ez, curl_z, dpz)

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_lor = LorentzState(
        px=px_new, py=py_new, pz=pz_new,
        px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
    )
    return new_fdtd, new_lor


def _update_e_local_with_dispersion(state, materials, dt, dx,
                                     debye=None, lorentz=None, *, high_wall, slab=None, e_materials=None, rank=None, boundary=None, frame=None, owned=None):
    """E update on a local slab with optional Debye/Lorentz dispersion.

    Every model uses the per-component edge means of
    :func:`slab_e_component_materials` (``slab = (nx_per, nx)``). Runners
    with dispersion pass the means formed before the loop and finish the
    material-dependent ADE coefficients here. Prebuilt coefficients remain
    accepted by grid-level callers.

    Returns (new_state, new_debye_state_or_None, new_lorentz_state_or_None).
    """
    if debye is None and lorentz is None:
        return _update_e_local(
            state, slab_e_coeffs(materials, *slab, dt, rank=rank, high_wall=high_wall), dx,
            boundary=boundary, frame=frame, owned=owned), None, None

    if e_materials is not None:
        if debye is not None:
            debye = (slab_dispersion_coeffs(debye[0], e_materials, dt, *slab, rank=rank), debye[1])
        if lorentz is not None:
            lorentz = (slab_dispersion_coeffs(lorentz[0], e_materials, dt, *slab, rank=rank), lorentz[1])

    if debye is not None and lorentz is None:
        debye_coeffs, debye_state = debye
        new_state, new_debye = _update_e_debye_local(
            state, debye_coeffs, debye_state, dt, dx, boundary=boundary, frame=frame, owned=owned)
        return new_state, new_debye, None

    if lorentz is not None and debye is None:
        lorentz_coeffs, lorentz_state = lorentz
        new_state, new_lorentz = _update_e_lorentz_local(
            state, lorentz_coeffs, lorentz_state, dt, dx, boundary=boundary, frame=frame, owned=owned)
        return new_state, None, new_lorentz

    # Mixed Debye + Lorentz: the single-device combined coefficients
    # (``mixed_e_component_coeffs``), per component (#1260). This body used to
    # take ca/cb/cc from the Debye build and the Lorentz dP coefficient as
    # 1/gamma_Lorentz; the combined update needs 1/gamma_total = 1/(gamma_Lorentz
    # + sum beta) there, which differs on every edge that carries both a Debye
    # and a Lorentz pole -- with the edge mean, every edge on a Debye|Lorentz
    # interface.
    debye_coeffs, debye_state = debye
    lorentz_coeffs, lorentz_state = lorentz
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)

    curl_x = ((hz - backward(hz, 1)) - (hy - backward(hy, 2))) / dx
    curl_y = ((hx - backward(hx, 2)) - (hz - backward(hz, 0))) / dx
    curl_z = ((hy - backward(hy, 0)) - (hx - backward(hx, 1))) / dx
    e_old = (state.ex, state.ey, state.ez)
    curls = (curl_x, curl_y, curl_z)
    p_l = (lorentz_state.px, lorentz_state.py, lorentz_state.pz)
    p_l_prev = (lorentz_state.px_prev, lorentz_state.py_prev,
                lorentz_state.pz_prev)
    p_d = (debye_state.px, debye_state.py, debye_state.pz)

    p_l_new = tuple(lorentz_p_component(lorentz_coeffs, c, e_old[c], p_l[c],
                                        p_l_prev[c]) for c in range(3))
    e_new, p_d_new = [], []
    for c in range(3):
        dp_l = jnp.sum(p_l_new[c] - p_l[c], axis=0)
        ca, cb, cc_d, cc_l = mixed_e_component_coeffs(
            debye_coeffs, lorentz_coeffs, c, dt)
        e_c = (ca * e_old[c] + cb * curls[c]
               + jnp.sum(cc_d * p_d[c], axis=0) - cc_l * dp_l)
        e_new.append(e_c)
        beta_c = per_component(debye_coeffs.beta, "beta")[c]
        p_d_new.append(debye_coeffs.alpha * p_d[c]
                       + beta_c * (e_c[None] + e_old[c][None]))

    new_fdtd = state._replace(ex=e_new[0], ey=e_new[1], ez=e_new[2],
                              step=state.step + 1)
    new_debye_st = DebyeState(px=p_d_new[0], py=p_d_new[1], pz=p_d_new[2])
    new_lor_st = LorentzState(
        px=p_l_new[0], py=p_l_new[1], pz=p_l_new[2],
        px_prev=lorentz_state.px, py_prev=lorentz_state.py, pz_prev=lorentz_state.pz,
    )
    return new_fdtd, new_debye_st, new_lor_st


def _update_h_local_nu(state, materials, dt,
                      inv_dx_slab, inv_dy_full, inv_dz_full,
                      inv_dx_h_slab, inv_dy_h_full, inv_dz_h_full, *, high_wall, cell_sizes=None):
    """H update on a local slab using NU inverse spacings.

    Mirrors ``rfx/core/yee.py::update_h_nu`` but accepts pre-sliced
    per-device ``inv_dx`` / ``inv_dx_h`` (length nx_local), while
    y/z spacings are replicated (full-axis).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    if cell_sizes is None:
        cell_sizes = tuple(1 / jnp.where(v > 0, v, v[-2])
                           for v in (inv_dx_h_slab, inv_dy_h_full, inv_dz_h_full))
    mu = component_h_materials(materials, high_wall=high_wall, cell_sizes=cell_sizes)
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "yee.H", cell_sizes=cell_sizes)
    mu_x, mu_y, mu_z = (m * MU_0 for m in mu)
    ch_x, ch_y, ch_z = dt / mu_x, dt / mu_y, dt / mu_z

    curl_x = (
        (_shift_fwd(ez, 1) - ez) * inv_dy_h_full[None, :, None]
        - (_shift_fwd(ey, 2) - ey) * inv_dz_h_full[None, None, :]
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) * inv_dz_h_full[None, None, :]
        - (_shift_fwd(ez, 0) - ez) * inv_dx_h_slab[:, None, None]
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) * inv_dx_h_slab[:, None, None]
        - (_shift_fwd(ex, 1) - ex) * inv_dy_h_full[None, :, None]
    )

    hx = state.hx - ch_x * curl_x
    hy = state.hy - ch_y * curl_y
    hz = state.hz - ch_z * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local_nu(state, e_coeffs,
                      inv_dx_slab, inv_dy_full, inv_dz_full, *, boundary=None, frame=None, owned=None):
    """E update on a local slab using NU inverse (cell-local) spacings.

    Mirrors ``rfx/core/yee.py::update_e_nu``. ``e_coeffs`` is the slab's
    per-component ``((ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z))`` (#1303,
    :func:`slab_e_coeffs` / :func:`component_e_coeffs`).
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    def backward(h, axis):
        return _h_backward(h, axis, boundary=boundary, frame=frame, owned=owned)
    (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z) = e_coeffs

    curl_x = (
        (hz - backward(hz, 1)) * inv_dy_full[None, :, None]
        - (hy - backward(hy, 2)) * inv_dz_full[None, None, :]
    )
    curl_y = (
        (hx - backward(hx, 2)) * inv_dz_full[None, None, :]
        - (hz - backward(hz, 0)) * inv_dx_slab[:, None, None]
    )
    curl_z = (
        (hy - backward(hy, 0)) * inv_dx_slab[:, None, None]
        - (hx - backward(hx, 1)) * inv_dy_full[None, :, None]
    )

    ex = ca_x * state.ex + cb_x * curl_x
    ey = ca_y * state.ey + cb_y * curl_y
    ez = ca_z * state.ez + cb_z * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


def update_h_nu_shmap(st, mat, mesh, dt,
                      inv_dx_sharded, inv_dy_rep, inv_dz_rep,
                      inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep, *, ranks, nx_per, nx, high_wall, cell_sizes=None):
    """H update on the NU distributed path, via ``shard_map``.

    Shared by ``distributed_nu.run_nonuniform_distributed_pec`` and by the
    ``is_nu`` branch of ``distributed_v2.run_distributed``.

    Parameters
    ----------
    st, mat
        The sharded :class:`FDTDState` and :class:`MaterialArrays`.
    mesh
        The 1-D x-axis :class:`jax.sharding.Mesh`.
    dt
        Timestep, closed over as a Python float by both original copies.
    inv_dx_sharded, inv_dx_h_sharded
        Per-device slabs of the inverse x spacings (cell-local and
        mean-spacing), sharded over ``"x"``.
    inv_dy_rep, inv_dz_rep, inv_dy_h_rep, inv_dz_h_rep
        Full-axis inverse y/z spacings, replicated on every device.

    Returns the state with ``hx``/``hy``/``hz``/``step`` replaced. The E-field
    components are passed in because the H curl reads them, and returned
    untouched by not being in ``out_specs``.
    """
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(mat, lane="distributed H shard", unsupported=True)
    if cell_sizes is None:
        cell_sizes = tuple(1 / jnp.where(v > 0, v, v[-2]) for v in
                           (inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep))
    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(
            P("x"), P("x"), P("x"),  # ex, ey, ez
            P("x"), P("x"), P("x"),  # hx, hy, hz
            P(),                     # step
            P("x"), P("x"), P("x"),  # eps_r, sigma, mu_r
            P("x"), P(None), P(None),  # inv_dx, inv_dy, inv_dz
            P("x"), P(None), P(None),  # inv_dx_h, inv_dy_h, inv_dz_h
            (P("x"), P(None), P(None)),  # original primal cell lengths
        ),
        out_specs=(P("x"), P("x"), P("x"), P()),
        check_rep=False,
    )
    def _h(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
           invdx, invdy, invdz, invdxh, invdyh, invdzh, widths, *, rank):
        _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
        _mat = MaterialArrays(eps_r=eps_r, sigma=sigma,
                             mu_r=magnetic_low_ghost(mu_r, rank, nx_per=nx_per, nx=nx, high_wall=high_wall))
        from rfx.model.electric_metrics import slab_wall_view
        widths = (slab_wall_view(widths[0], nx_per, nx, rank, high_wall=high_wall), *widths[1:])
        new_st = _update_h_local_nu(
            _st, _mat, dt, invdx, invdy, invdz, invdxh, invdyh, invdzh,
            cell_sizes=widths, high_wall=(False, *high_wall[1:]))
        return new_st.hx, new_st.hy, new_st.hz, new_st.step

    hx, hy, hz, step = _h(
        ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
        mat.eps_r, mat.sigma, mat.mu_r,
        inv_dx_sharded, inv_dy_rep, inv_dz_rep,
        inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep,
        cell_sizes,
    )
    return st._replace(hx=hx, hy=hy, hz=hz, step=step)


def update_e_nu_shmap(st, mat, mesh, dt,
                      inv_dx_sharded, inv_dy_rep, inv_dz_rep, nx_per, nx,
                      e_materials=None, *, ranks, high_wall, boundary=None, ghost=1, pad_x=0):
    """E update on the NU distributed path, via ``shard_map``.

    The E sibling of :func:`update_h_nu_shmap`, shared by the same two
    runners and merged for the same reason -- inventory §2.4 measured the two
    inner kernels at 0.947 similarity, the entire residue being the ``def``
    name. It needs three inverse-spacing arrays rather than six because the E
    curl reads only the cell-local spacings, not the mean-spacing ones.

    Each component's eps and sigma are the mean over the four cells around
    its edge, the single-device rule (#1303): ``e_materials``, the x-sharded
    means :func:`slab_e_materials_shmap` built before the loop, or, when it
    is ``None``, :func:`slab_e_component_materials` of ``mat`` on each slab,
    from ``nx_per`` (real rows per slab) and ``nx`` (the unpadded global x
    count). The coefficients are formed here, in the loop, either way.

    Returns the state with ``ex``/``ey``/``ez``/``step`` replaced. Dispersion
    is NOT handled here: ``distributed_v2.run_distributed`` refuses Debye and
    Lorentz on the NU path upstream, and its caller passes the polarisation
    state straight back out.
    """
    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(
            P("x"), P("x"), P("x"),
            P("x"), P("x"), P("x"),
            P(),
            P("x"), P("x"), P("x"),
            P("x"), P("x"), P("x"),  # lumped and film component records
            P("x"), P(None), P(None),
            P("x"),                  # e_materials: None or ((eps...), (sig...))
        ),
        out_specs=(P("x"), P("x"), P("x"), P()),
        check_rep=False,
    )
    def _e(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
           sigma_lumped, eps_r_lumped, sigma_film, invdx, invdy, invdz, means, *, rank):
        _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
        if means is None:
            _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r,
                                  sigma_film=sigma_film,
                                  sigma_lumped=sigma_lumped,
                                  eps_r_lumped=eps_r_lumped)
            coeffs = slab_e_coeffs(_mat, nx_per, nx, dt, rank=rank, high_wall=high_wall)
        else:
            coeffs = component_e_coeffs(means, dt)
        new_st = _update_e_local_nu(_st, coeffs, invdx, invdy, invdz,
            **_slab_curl_kwargs(boundary, hx.shape, ghost, pad_x, rank, mesh.size))
        return new_st.ex, new_st.ey, new_st.ez, new_st.step

    ex, ey, ez, step = _e(
        ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
        mat.eps_r, mat.sigma, mat.mu_r,
        getattr(mat, "sigma_lumped", None), getattr(mat, "eps_r_lumped", None),
        mat.sigma_film, inv_dx_sharded, inv_dy_rep, inv_dz_rep, e_materials,
    )
    return st._replace(ex=ex, ey=ey, ez=ez, step=step)


def inject_sources_shmap(st, src_vals_step, mesh, n_src,
                         src_local_specs, src_device_ids, *, ranks, drives=None):
    """Inject sources on their owning device using ``shard_map``.

    ``shard_map`` gives each device its own slab; device identity inside
    the kernel comes from the explicit sharded rank input, and a source whose owner
    is some other device contributes ``jnp.where(... , 0.0)``.  So every
    device runs the same grouped add and only the owner's lands.

    ``in_specs`` are ``ex``, ``ey``, ``ez`` sharded on ``P("x")`` and
    ``src_vals_step`` slab-local (``P("x")``), shape (n_devices, n_sources).
    Each device receives its stored row, with all non-owned entries zero.

    ``src_local_specs[i]`` is ``(li, lj, lk, lc)``: the LOCAL index triple
    on the owning device plus the component name; ``src_device_ids[i]`` is
    that owner.  Both are Python data read at trace time, so the traced
    graph depends on their VALUES, not just their shapes.

    Shared by the uniform and graded runners. Their trace-time layout uses
    these local indices; unchanged waveform samples arrive as scan inputs.
    """
    if n_src == 0:
        return st

    if drives is None:
        drives = drive_layout(src_local_specs, src_vals_step.dtype,
                              electric_only_path="distributed")
    device_ids = np.asarray(src_device_ids, dtype=np.int32)

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"), P("x")),
        out_specs=(P("x"), P("x"), P("x")),
        check_rep=False,
    )
    def _inject(ex, ey, ez, sv, *, rank):
        from rfx.core.yee import FDTDState
        local = FDTDState(ex, ey, ez, None, None, None, None)
        local = inject_drives(local, drives, jnp.where(rank == device_ids, sv[0], 0.0))
        return local.ex, local.ey, local.ez

    ex, ey, ez = _inject(ranks, st.ex, st.ey, st.ez, src_vals_step)
    return st._replace(ex=ex, ey=ey, ez=ez)


def _h_backward(h, axis, *, boundary=None, frame=None, owned=None):
    """Keep nonmagnetic slab programs unchanged; share the one magnetic image."""
    if boundary is None or not boundary.pmc_faces:
        return _shift_bwd(h, axis)
    from rfx.core.yee import h_neighbor
    return h_neighbor(h, axis, boundary=boundary,
                      frame=None if frame is None else frame.get("xyz"[axis]), owned=owned)


def slab_dispersion_coeffs(coeffs, e_materials, dt, nx_per, nx, rank=None):
    """Finish ADE E coefficients inside the loop from the shared edge means."""
    if isinstance(coeffs, DebyeCoeffs):
        full = debye_e_coeffs(e_materials, dt, coeffs.alpha, coeffs.beta)
    else:
        full = lorentz_e_coeffs(e_materials, dt, coeffs.a, coeffs.b, coeffs.c)
    return _pad_slab_dispersion_coeffs(full, dt, nx_per, nx, rank)


def _pad_slab_dispersion_coeffs(coeffs, dt, nx_per, nx, rank=None):
    return fill_pole_rows(coeffs, dt, nx_per, nx, rank=rank)


def slab_e_coeffs(materials, nx_per, nx, dt, rank=None, *, high_wall: tuple[bool, bool, bool]):
    """The E coefficients of one x slab (#1303): :func:`component_e_coeffs`
    of :func:`rfx.model.electric_metrics.slab_e_component_materials`. With :func:`slab_e_materials_shmap`
    the one path the multi-device E updates take to their coefficients."""
    return component_e_coeffs(
        slab_e_component_materials(materials, nx_per, nx, rank, high_wall=high_wall), dt)


def component_e_coeffs(e_materials, dt):
    """``((ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z))`` from per-component
    ``((eps_x, eps_y, eps_z), (sig_x, sig_y, sig_z))``:
    :func:`rfx.core.yee.e_update_coeffs` of each component."""
    pairs = [e_update_coeffs(e, s, dt) for e, s in zip(*e_materials)]
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def magnetic_low_ghost(mu_r, rank, ghost=1, *, nx_per, nx, high_wall: tuple[bool, bool, bool]):
    """Replicate the physical low-x cell; preserve inter-rank material ghosts.

    Legacy staging pads outside with vacuum. That pad is not a neighbour
    cell of the new face mean. Only rank zero's exterior ghost is replaced.
    """
    from rfx.model.electric_metrics import slab_wall_view
    return slab_wall_view(mu_r, nx_per, nx, rank, high_wall=high_wall, ghost=ghost)


def _slab_curl_kwargs(boundary, shape, ghost, pad_x, rank, n_devices):
    """Only magnetic slabs trace ownership; transverse axes use the full extent."""
    if boundary is None or not boundary.pmc_faces:
        return {}
    return dict(boundary=boundary, frame=cpml_slab_frame(shape, ghost, pad_x),
                owned={"x_lo": rank == 0, "x_hi": rank == n_devices - 1})


def cpml_slab_frame(shape, ghost, pad_x):
    """The last rank's physical x end is static on every rank."""
    return {"x": (ghost, shape[0] - ghost - pad_x)}


def cpml_absorbing_faces(grid):
    """Select the realized absorbing faces of the global model."""
    from rfx.boundaries.depths import grid_face_depths
    return tuple(face.name for face in grid_face_depths(grid) if face.realized > 0)


def cpml_h_materials(mu_r, rank, ghost, *, nx_per, nx, high_wall: tuple[bool, bool, bool]):
    """Reuse the uniform slab's component means, including its low ghost."""
    from types import SimpleNamespace
    if mu_r is None:
        means = component_h_materials(MaterialArrays(None, None, 1.0), high_wall=high_wall)
    else:
        mu_r = magnetic_low_ghost(mu_r, rank, ghost, nx_per=nx_per, nx=nx, high_wall=high_wall)
        means = component_h_materials(MaterialArrays(None, None, mu_r), high_wall=(False, *high_wall[1:]))
    return SimpleNamespace(mu_r=mu_r, components=SimpleNamespace(mu_update=means))


def slab_e_materials_shmap(mat, mesh, nx_per, nx, *, ranks, high_wall: tuple[bool, bool, bool], grid=None):
    """:func:`slab_e_component_materials` of every slab, as x-sharded
    arrays, for a runner that averages once, before its time loop, and
    forms the coefficients (:func:`component_e_coeffs`) inside it.

    Why a runner would (#1303): inside the loop the mean is loop-invariant
    and XLA hoists it out of the loop anyway, so the forward program holds
    the same six coefficient slabs either way. Under ``jax.grad`` it does
    not: taken inside the loop (in the E update and again in the CPML), the
    four-cell mean is transposed at every step -- measured on CPU, 2.2x the
    cell-owned lane's time per step and 2.4x its backward scratch; taken
    here, once, about the cell-owned lane's. The coefficients stay inside
    the loop: built here instead, the permittivity gradient of a 1.5e16
    objective came back non-finite at four cells next to the current sources
    (``test_a_large_objective_gives_a_finite_gradient_on_both_lanes``; the
    mechanism was not established -- #1317 recorded a float32 overflow of
    ``Cb``'s reverse pass in the same test). ``jax.checkpoint`` recomputes
    this mean in the backward pass instead of keeping it on the tape.
    """
    @partial(rank_shard_map, mesh=mesh, in_specs=(P("x"),), out_specs=P("x"),
             check_rep=False)
    def _mean(local, *, rank):
        return slab_e_component_materials(local, nx_per, nx, rank=rank,
            high_wall=high_wall, cell_sizes=slab_cell_sizes(grid, rank))

    return jax.checkpoint(_mean)(ranks, MaterialArrays(
        eps_r=mat.eps_r, sigma=mat.sigma, mu_r=mat.mu_r,
        sigma_film=mat.sigma_film,
        sigma_lumped=getattr(mat, "sigma_lumped", None),
        eps_r_lumped=getattr(mat, "eps_r_lumped", None)))


def sample_probes_shmap(st, mesh, n_prb, prb_local_specs, prb_device_ids,
                        *, reduce_devices=True, ranks):
    """Sample probes on their owning devices, then sum across devices.

    Mirror of :func:`inject_sources_shmap` on the read side: every device
    reads at the same local index, masks with ``jnp.where`` on device
    identity, and ``lax.psum`` over ``"x"`` leaves exactly the owner's
    value.  ``out_specs=P()`` because the psum result is replicated.

    With ``reduce_devices=False``, return masked samples of global shape
    ``(n_devices, n_prb)`` on ``P("x")`` without a collective. A scan can
    stack these and sum its device axis once after the time loop. The
    default retains the per-step replicated result for the NU runner.

    Extracted verbatim from
    ``distributed_nu.py::run_nonuniform_distributed_pec._sample_probes_shmap``
    and ``distributed_v2.py::run_distributed._sample_probes_shmap``.
    """
    if n_prb == 0:
        shape = (0,) if reduce_devices else (mesh.size, 0)
        return jnp.zeros(shape, dtype=jnp.float32)

    @partial(
        rank_shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"),
                  P("x"), P("x"), P("x")),
        out_specs=P() if reduce_devices else P("x"),
        check_rep=False,
    )
    def _sample(ex, ey, ez, hx, hy, hz, *, rank):
        device_idx = rank
        samples = []
        for idx_p in range(n_prb):
            li, lj, lk, lc = prb_local_specs[idx_p]
            dev_id = prb_device_ids[idx_p]
            if lc == "ex":
                raw = ex[li, lj, lk]
            elif lc == "ey":
                raw = ey[li, lj, lk]
            elif lc == "ez":
                raw = ez[li, lj, lk]
            elif lc == "hx":
                raw = hx[li, lj, lk]
            elif lc == "hy":
                raw = hy[li, lj, lk]
            else:
                raw = hz[li, lj, lk]
            val = jnp.where(device_idx == dev_id, raw, 0.0)
            samples.append(val)
        samples = jnp.stack(samples)
        return lax.psum(samples, "x") if reduce_devices else samples[None, :]

    return _sample(ranks, st.ex, st.ey, st.ez, st.hx, st.hy, st.hz)


def _concrete_on_mesh(arr, sharding):
    """Place concrete data on addressable devices only (also for replication)."""
    if isinstance(arr, jax.core.Tracer):
        return jax.device_put(arr, sharding)
    if isinstance(arr, jax.Array) and arr.sharding == sharding:
        return arr
    host = np.asarray(arr)
    return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])


def _update_e_dispersive_local_nu(
    state: FDTDState,
    materials: MaterialArrays,
    dt: float,
    inv_dx: jnp.ndarray,
    inv_dy: jnp.ndarray,
    inv_dz: jnp.ndarray,
    *,
    debye=None,
    lorentz=None,
    e_old: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None, boundary=None, frame=None, owned=None,
):
    """Per-rank slab-aware NU dispersive E update.

    Thin wrapper around :func:`rfx.nonuniform._update_e_nu_dispersive`
    so the shard_map call site stays uncluttered and so the ADE
    Ordering Contract has a single chokepoint.

    Phase 2D ADE Ordering Contract: ``e_old`` MUST be the snapshot of
    ``state.ex/ey/ez`` taken BEFORE the H ghost exchange.  See the
    docstring of :func:`run_nonuniform_distributed_pec` for the full
    sequence; passing ``e_old=None`` here would let the helper fall
    back to ``state.ex/ey/ez``, but on the distributed path that is
    NEVER what the caller wants — pass the snapshot explicitly.
    """
    from rfx.stepping.graded_setup import _update_e_nu_dispersive

    return _update_e_nu_dispersive(
        state, materials, dt, inv_dx, inv_dy, inv_dz,
        debye=debye, lorentz=lorentz, e_old=e_old, boundary=boundary, frame=frame, owned=owned,
    )
