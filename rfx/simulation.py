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
    update_e, update_h, EPS_0,
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
    """
    state: FDTDState
    time_series: jnp.ndarray


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
    waveform = jnp.array(
        [float(waveform_fn(step * grid.dt)) for step in range(n_steps)]
    )
    return SourceSpec(i=idx[0], j=idx[1], k=idx[2],
                      component=component, waveform=waveform)


def make_port_source(grid: Grid, port, materials: MaterialArrays, n_steps):
    """Create a SourceSpec for a lumped port (Cb-corrected waveform).

    The port impedance must already be folded into *materials* via
    ``setup_lumped_port()``.
    """
    idx = grid.position_to_index(port.position)
    i, j, k = idx

    eps = float(materials.eps_r[i, j, k]) * EPS_0
    sigma = float(materials.sigma[i, j, k])
    loss = sigma * grid.dt / (2.0 * eps)
    cb = (grid.dt / eps) / (1.0 + loss)

    waveform = jnp.array([
        float(port.excitation(step * grid.dt)) * cb / grid.dx
        for step in range(n_steps)
    ])
    return SourceSpec(i=i, j=j, k=k,
                      component=port.component, waveform=waveform)


def make_probe(grid: Grid, position, component):
    """Create a ProbeSpec from a physical position."""
    idx = grid.position_to_index(position)
    return ProbeSpec(i=idx[0], j=idx[1], k=idx[2], component=component)


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
    sources: list[SourceSpec] | None = None,
    probes: list[ProbeSpec] | None = None,
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
    sources : list of SourceSpec (precomputed waveforms)
    probes : list of ProbeSpec (point time-series recorders)

    Returns
    -------
    SimResult with final state and (n_steps, n_probes) time series.
    """
    sources = sources or []
    probes = probes or []

    dt = grid.dt
    dx = grid.dx

    # ---- subsystem flags (resolved at trace time) ----
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0
    use_debye = debye is not None

    # ---- initialise states ----
    fdtd = init_state(grid.shape)

    carry_init: dict = {"fdtd": fdtd}

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state = init_cpml(grid)
        carry_init["cpml"] = cpml_state

    if use_debye:
        from rfx.materials.debye import update_e_debye
        debye_coeffs, debye_state = debye
        carry_init["debye"] = debye_state

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

        # E update (Debye or standard)
        if use_debye:
            st, debye_new = update_e_debye(
                st, debye_coeffs, carry["debye"], dt, dx)
        else:
            st = update_e(st, materials, dt, dx)

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

        # Rebuild carry
        new_carry: dict = {"fdtd": st}
        if use_cpml:
            new_carry["cpml"] = cpml_new
        if use_debye:
            new_carry["debye"] = debye_new

        return new_carry, output

    # ---- run ----
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)
    final_carry, time_series = jax.lax.scan(step_fn, carry_init, xs)

    return SimResult(
        state=final_carry["fdtd"],
        time_series=time_series,
    )
