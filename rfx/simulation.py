"""Compiled FDTD simulation runner.

Composes Yee updates, boundaries, sources, and probes into a single
JIT-compiled time loop via jax.lax.scan.  All subsystem selection
(CPML, Debye) is resolved at Python trace-time so the compiled
function contains only the needed code paths.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state,
    update_e, update_h, EPS_0, _shift_bwd,
)
from rfx.boundaries.pec import apply_pec


# ---------------------------------------------------------------------------
# Source / probe specifications
# ---------------------------------------------------------------------------

class SourceSpec(NamedTuple):
    """Precomputed point source for the compiled runner.

    waveform : (n_steps,) float array — precomputed values added to the
        field component at (i, j, k) each timestep.
    """
    i: int
    j: int
    k: int
    component: str
    waveform: jnp.ndarray


class ProbeSpec(NamedTuple):
    """Point probe that records a field component each timestep."""
    i: int
    j: int
    k: int
    component: str


class SimResult(NamedTuple):
    """Compiled simulation output.

    time_series : (n_steps, n_probes) float array, or (n_steps, 0) if
        no probes were specified.
    ntff_data : NTFFData or None
        Accumulated near-to-far-field DFT data (if NTFF box was used).
    """
    state: FDTDState
    time_series: jnp.ndarray
    ntff_data: object = None


# ---------------------------------------------------------------------------
# Helpers to build source / probe specs
# ---------------------------------------------------------------------------

def make_source(grid: Grid, position, component, waveform_fn, n_steps):
    """Create a SourceSpec by precomputing a waveform function.

    Parameters
    ----------
    grid : Grid
    position : (x, y, z) in metres
    component : "ex", "ey", or "ez"
    waveform_fn : callable(t) -> value
    n_steps : int
    """
    idx = grid.position_to_index(position)
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = jax.vmap(waveform_fn)(times)
    return SourceSpec(i=idx[0], j=idx[1], k=idx[2],
                      component=component, waveform=waveform)


def make_port_source(grid: Grid, port, materials: MaterialArrays, n_steps):
    """Create a SourceSpec for a lumped port (Cb-corrected waveform).

    The port impedance must already be folded into *materials* via
    ``setup_lumped_port()``.
    """
    idx = grid.position_to_index(port.position)
    i, j, k = idx

    eps = materials.eps_r[i, j, k] * EPS_0
    sigma = materials.sigma[i, j, k]
    loss = sigma * grid.dt / (2.0 * eps)
    cb = (grid.dt / eps) / (1.0 + loss)

    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = (cb / grid.dx) * jax.vmap(port.excitation)(times)
    return SourceSpec(i=i, j=j, k=k,
                      component=port.component, waveform=waveform)


def make_probe(grid: Grid, position, component):
    """Create a ProbeSpec from a physical position."""
    idx = grid.position_to_index(position)
    return ProbeSpec(i=idx[0], j=idx[1], k=idx[2], component=component)


def _update_e_with_optional_dispersion(
    state: FDTDState,
    materials: MaterialArrays,
    dt: float,
    dx: float,
    *,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    periodic: tuple = (False, False, False),
) -> tuple[FDTDState, object | None, object | None]:
    """Update E with standard, Debye, Lorentz, or mixed dispersion."""
    if debye is None and lorentz is None:
        return update_e(state, materials, dt, dx, periodic=periodic), None, None

    if debye is not None and lorentz is None:
        from rfx.materials.debye import update_e_debye

        debye_coeffs, debye_state = debye
        new_state, new_debye = update_e_debye(
            state, debye_coeffs, debye_state, dt, dx, periodic=periodic)
        return new_state, new_debye, None

    if lorentz is not None and debye is None:
        from rfx.materials.lorentz import update_e_lorentz

        lorentz_coeffs, lorentz_state = lorentz
        new_state, new_lorentz = update_e_lorentz(
            state, lorentz_coeffs, lorentz_state, dt, dx, periodic=periodic)
        return new_state, None, new_lorentz

    # Mixed Debye + Lorentz update.
    from rfx.materials.debye import DebyeState
    from rfx.materials.lorentz import LorentzState

    debye_coeffs, debye_state = debye
    lorentz_coeffs, lorentz_state = lorentz

    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    hx, hy, hz = state.hx, state.hy, state.hz

    curl_x = ((hz - bwd(hz, 1)) - (hy - bwd(hy, 2))) / dx
    curl_y = ((hx - bwd(hx, 2)) - (hz - bwd(hz, 0))) / dx
    curl_z = ((hy - bwd(hy, 0)) - (hx - bwd(hx, 1))) / dx

    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    # Explicit Lorentz polarization update first.
    px_l_new = (
        lorentz_coeffs.a * lorentz_state.px
        + lorentz_coeffs.b * lorentz_state.px_prev
        + lorentz_coeffs.c * ex_old[None]
    )
    py_l_new = (
        lorentz_coeffs.a * lorentz_state.py
        + lorentz_coeffs.b * lorentz_state.py_prev
        + lorentz_coeffs.c * ey_old[None]
    )
    pz_l_new = (
        lorentz_coeffs.a * lorentz_state.pz
        + lorentz_coeffs.b * lorentz_state.pz_prev
        + lorentz_coeffs.c * ez_old[None]
    )

    dpx_l = jnp.sum(px_l_new - lorentz_state.px, axis=0)
    dpy_l = jnp.sum(py_l_new - lorentz_state.py, axis=0)
    dpz_l = jnp.sum(pz_l_new - lorentz_state.pz, axis=0)

    beta_sum = jnp.sum(debye_coeffs.beta, axis=0)
    gamma_base = 1.0 / lorentz_coeffs.cc
    gamma_total = jnp.maximum(gamma_base + beta_sum, EPS_0 * 1e-10)
    numer_base = lorentz_coeffs.ca * gamma_base

    ca = (numer_base - beta_sum) / gamma_total
    cb = dt / gamma_total
    cc_debye = (1.0 - debye_coeffs.alpha) / gamma_total
    cc_lorentz = 1.0 / gamma_total

    ex_new = (
        ca * ex_old
        + cb * curl_x
        + jnp.sum(cc_debye * debye_state.px, axis=0)
        - cc_lorentz * dpx_l
    )
    ey_new = (
        ca * ey_old
        + cb * curl_y
        + jnp.sum(cc_debye * debye_state.py, axis=0)
        - cc_lorentz * dpy_l
    )
    ez_new = (
        ca * ez_old
        + cb * curl_z
        + jnp.sum(cc_debye * debye_state.pz, axis=0)
        - cc_lorentz * dpz_l
    )

    new_fdtd = state._replace(
        ex=ex_new,
        ey=ey_new,
        ez=ez_new,
        step=state.step + 1,
    )
    new_debye = DebyeState(
        px=debye_coeffs.alpha * debye_state.px + debye_coeffs.beta * (ex_new[None] + ex_old[None]),
        py=debye_coeffs.alpha * debye_state.py + debye_coeffs.beta * (ey_new[None] + ey_old[None]),
        pz=debye_coeffs.alpha * debye_state.pz + debye_coeffs.beta * (ez_new[None] + ez_old[None]),
    )
    new_lorentz = LorentzState(
        px=px_l_new,
        py=py_l_new,
        pz=pz_l_new,
        px_prev=lorentz_state.px,
        py_prev=lorentz_state.py,
        pz_prev=lorentz_state.pz,
    )

    return new_fdtd, new_debye, new_lorentz


# ---------------------------------------------------------------------------
# Compiled runner
# ---------------------------------------------------------------------------

def run(
    grid: Grid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    boundary: str = "pec",
    cpml_axes: str = "xyz",
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    sources: list[SourceSpec] | None = None,
    probes: list[ProbeSpec] | None = None,
    ntff: object | None = None,
    checkpoint: bool = False,
) -> SimResult:
    """Run a compiled FDTD simulation via ``jax.lax.scan``.

    Parameters
    ----------
    grid : Grid
    materials : MaterialArrays
    n_steps : int
    boundary : "pec" or "cpml"
    cpml_axes : axes string for CPML (default "xyz")
    debye : (DebyeCoeffs, DebyeState) tuple, or None
    lorentz : (LorentzCoeffs, LorentzState) tuple, or None
    sources : list of SourceSpec (precomputed waveforms)
    probes : list of ProbeSpec (point time-series recorders)
    ntff : NTFFBox or None
        If provided, accumulate near-to-far-field DFT on a Huygens box.
    checkpoint : bool
        If True, wrap the scan body with ``jax.checkpoint`` to
        trade compute for memory during reverse-mode AD.  Reduces
        backward-pass memory from O(n_steps) to O(1) per step.

    Returns
    -------
    SimResult with final state, time series, and optional NTFF data.
    """
    sources = sources or []
    probes = probes or []

    dt = grid.dt
    dx = grid.dx

    # ---- subsystem flags (resolved at trace time) ----
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_ntff = ntff is not None

    # ---- initialise states ----
    fdtd = init_state(grid.shape)

    carry_init: dict = {"fdtd": fdtd}

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state = init_cpml(grid)
        carry_init["cpml"] = cpml_state

    if use_debye:
        debye_coeffs, debye_state = debye
        carry_init["debye"] = debye_state

    if use_lorentz:
        lorentz_coeffs, lorentz_state = lorentz
        carry_init["lorentz"] = lorentz_state

    if use_ntff:
        from rfx.farfield import init_ntff_data, accumulate_ntff
        carry_init["ntff"] = init_ntff_data(ntff)

    # ---- precompute source waveform matrix (n_steps, n_sources) ----
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # Static source/probe metadata (captured by closure)
    src_meta = [(s.i, s.j, s.k, s.component) for s in sources]
    prb_meta = [(p.i, p.j, p.k, p.component) for p in probes]

    # ---- scan body ----
    def step_fn(carry, xs):
        _step_idx, src_vals = xs
        st = carry["fdtd"]

        # H update
        st = update_h(st, materials, dt, dx)
        if use_cpml:
            st, cpml_new = apply_cpml_h(
                st, cpml_params, carry["cpml"], grid, cpml_axes)

        st, debye_new, lorentz_new = _update_e_with_optional_dispersion(
            st,
            materials,
            dt,
            dx,
            debye=(debye_coeffs, carry["debye"]) if use_debye else None,
            lorentz=(lorentz_coeffs, carry["lorentz"]) if use_lorentz else None,
        )

        if use_cpml:
            st, cpml_new = apply_cpml_e(
                st, cpml_params, cpml_new, grid, cpml_axes)

        # PEC
        st = apply_pec(st)

        # Soft sources
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            field = getattr(st, sc)
            field = field.at[si, sj, sk].add(src_vals[idx_s])
            st = st._replace(**{sc: field})

        # Probe samples
        samples = [getattr(st, pc)[pi, pj, pk]
                   for pi, pj, pk, pc in prb_meta]
        output = jnp.stack(samples) if samples else jnp.zeros(0)

        # NTFF accumulation
        if use_ntff:
            ntff_new = accumulate_ntff(
                carry["ntff"], st, ntff, dt, _step_idx)

        # Rebuild carry
        new_carry: dict = {"fdtd": st}
        if use_cpml:
            new_carry["cpml"] = cpml_new
        if use_debye:
            new_carry["debye"] = debye_new
        if use_lorentz:
            new_carry["lorentz"] = lorentz_new
        if use_ntff:
            new_carry["ntff"] = ntff_new

        return new_carry, output

    # ---- run ----
    body = jax.checkpoint(step_fn) if checkpoint else step_fn
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)
    final_carry, time_series = jax.lax.scan(body, carry_init, xs)

    return SimResult(
        state=final_carry["fdtd"],
        time_series=time_series,
        ntff_data=final_carry.get("ntff"),
    )
