"""Measure an absorber's plane-wave return and a record's depth sensitivity."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from rfx.boundaries import cpml
from rfx.boundaries.pec import apply_pec_faces
from rfx.core.yee import init_materials, init_state, update_e, update_h
from rfx.grid import C0, Grid


@lru_cache(maxsize=16)
def _line(n_layers, kappa_max, interior, dx, transverse_phase=None,
          nonuniform=False, face_layers=None, axis="x"):
    """Cache compilation only; each call starts with new fields and profiles."""
    normal = "xyz".index(axis)
    electric = "xyz"[(normal + 2) % 3]
    magnetic = "xyz"[(normal + 1) % 3]
    cells = [2, 2, 2]
    cells[normal] = interior
    domain = tuple(n * dx for n in cells)
    depths = None if face_layers is None else dict(zip((f"{axis}_lo", f"{axis}_hi"), face_layers))
    if nonuniform:
        from rfx.nonuniform import make_nonuniform_grid
        from rfx.core.yee import update_e_nu, update_h_nu
        from rfx.boundaries.pmc import apply_pmc_faces
        grid = make_nonuniform_grid(
            domain[:2], np.full(cells[2], dx), dx,
            cpml_layers=n_layers, cpml_axes=axis, face_layers=depths,
        )
    else:
        grid = Grid(freq_max=30e9, domain=domain,
                    dx=dx, cpml_layers=n_layers, cpml_axes=axis, kappa_max=kappa_max,
                    face_layers=depths)
    dt = float(grid.dt)

    @jax.jit
    def propagate(params, auxiliary, source_index, waveform):
        state = init_state(grid.shape)
        materials = init_materials(grid.shape)
        bloch = None
        if transverse_phase is not None:
            bloch = (1.0, transverse_phase, 1.0)
            state = jax.tree.map(lambda x: x.astype(jnp.complex64), state)
            auxiliary = jax.tree.map(lambda x: x.astype(jnp.complex64), auxiliary)

        def step(carry, value):
            state, auxiliary = carry
            if nonuniform:
                state = update_h_nu(state, materials, dt, grid.inv_dx_h,
                                    grid.inv_dy_h, grid.inv_dz_h)
                state = apply_pmc_faces(state, {f"{magnetic}_lo", f"{magnetic}_hi"})
            else:
                state = update_h(state, materials, dt, dx, periodic=tuple(a != axis for a in "xyz"), bloch=bloch)
            state, auxiliary = cpml.apply_cpml_h(state, params, auxiliary, grid, axes=axis)
            if nonuniform:
                state = update_e_nu(state, materials, dt, grid.inv_dx,
                                    grid.inv_dy, grid.inv_dz)
                state = apply_pec_faces(state, {f"{electric}_lo", f"{electric}_hi"})
            else:
                state = update_e(state, materials, dt, dx, periodic=tuple(a != axis for a in "xyz"), bloch=bloch)
            state, auxiliary = cpml.apply_cpml_e(state, params, auxiliary, grid, axes=axis)
            state = apply_pec_faces(state, {f"{axis}_lo", f"{axis}_hi"})
            sheet = [slice(None)] * 3
            probe = [1] * 3
            sheet[normal] = probe[normal] = source_index
            field = getattr(state, f"e{electric}").at[tuple(sheet)].add(value)
            state = state._replace(**{f"e{electric}": field})
            return (state, auxiliary), field[tuple(probe)]

        return jax.lax.scan(step, (state, auxiliary), waveform)[1]

    return grid, propagate


def _reflection_db(short, reference, dt, freqs):
    """Echo/incident amplitude spectra in an echo-isolating rectangular window."""
    short, reference = np.asarray(short, dtype=float), np.asarray(reference, dtype=float)
    if not np.all(np.isfinite(short)) or not np.all(np.isfinite(reference)):
        raise ValueError("nonfinite absorber record")
    nfft = max(1 << 16, 1 << (len(reference) - 1).bit_length())
    bins = np.fft.rfftfreq(nfft, dt)
    incident = np.abs(np.fft.rfft(reference, nfft))
    if np.any(np.interp(freqs, bins, incident) <= 1e-8 * incident.max()):
        raise ValueError("requested frequency has no resolved incident spectrum")
    ratio = np.abs(np.fft.rfft(short - reference, nfft)) / np.maximum(incident, 1e-30)
    return 20 * np.log10(np.maximum(np.interp(freqs, bins, ratio), 1e-30))


def plane_wave_reflection(n_layers, kappa_max, face, freqs, *,
                          near=30, far=400, reference_distance=900, dx=1e-3,
                          nonuniform=False, face_layers=None, axis="x"):
    """Return 20 log10 |R(f)| for one vacuum x face, frequencies in Hz.

    An Ez sheet on a periodic 3 by 3 transverse grid launches a TEM pulse.
    Subtract a long-domain record at the source plane to remove the incident
    field. The record ends at 90% of the far face's vacuum round trip, so only
    the near face can return; both reference faces are farther still. The
    denominator is the incident plane wave, with no geometrical spreading.
    The outer x walls are PEC, as in the main solver's CPML termination.

    Defaults reproduce the normal-incidence rig used in the #1012 review:
    1 mm cells, a 12 ps differentiated Gaussian centred at 60 ps, float32.
    This measures this discrete absorber, not a continuum or mesh error.

    The NU operators have no transverse periodic option. ``nonuniform=True``
    uses y magnetic walls and z electric walls around the same uniform TEM
    sheet, with the production NU curls and CPML. This invariant polarization
    has Ez/Hy constant across the cross section and zero transverse curls.
    ``face_layers=(lo, hi)`` allows unequal depths within ``n_layers``' budget.
    """
    freqs = np.asarray(freqs, dtype=float)
    if face not in ("lo", "hi"):
        raise ValueError("face must be 'lo' or 'hi'")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y' or 'z'")
    if n_layers < 2 or not 0 < near < far < reference_distance or dx <= 0:
        raise ValueError("require layers >= 2 and 0 < near < far < reference_distance")
    grid, propagate = _line(n_layers, kappa_max, near + far, dx,
                            nonuniform=nonuniform, face_layers=face_layers, axis=axis)
    dt = float(grid.dt)
    if freqs.ndim != 1 or not len(freqs) or np.any(~np.isfinite(freqs)) or np.any(
        (freqs <= 0) | (freqs >= 0.5 / dt)
    ):
        raise ValueError("frequencies must be finite, positive and below Nyquist")
    n_steps = int(0.9 * (2 * far * dx / C0) / dt)
    if n_steps * dt <= 120e-12 + 2 * (near + n_layers) * dx / C0:
        raise ValueError("increase far to include the near wall's round trip and the pulse")
    t = np.arange(n_steps) * dt
    arg = (t - 60e-12) / 12e-12
    waveform = jnp.asarray(-arg * np.exp(-arg**2), dtype=jnp.float32)
    source_index = getattr(grid, f"pad_{axis}_lo") + (near if face == "lo" else far)
    params, auxiliary = cpml.init_cpml(grid, kappa_max=kappa_max)
    short = propagate(params, auxiliary, source_index, waveform)
    ref_grid, ref_propagate = _line(n_layers, kappa_max, 2 * reference_distance, dx,
                                    nonuniform=nonuniform, face_layers=face_layers, axis=axis)
    params, auxiliary = cpml.init_cpml(ref_grid, kappa_max=kappa_max)
    reference = ref_propagate(params, auxiliary, getattr(ref_grid, f"pad_{axis}_lo") + reference_distance, waveform)
    return _reflection_db(short, reference, dt, freqs)


def oblique_plane_wave_reflection(n_layers=8, frequency=10e9, *, duration=5.1e-9):
    """Return |R| in dB at 45 degrees, TMz, for the low x face at kappa=1.

    The same Yee/CPML operators as the normal and 2-D rigs propagate a
    Bloch-periodic transverse plane wave. Set kx=ky using the discrete Yee
    dispersion relation, so the carrier's numerical angle is 45 degrees.
    A 300 ps analytic Gaussian pulse has fixed ky; only its 10 GHz carrier
    is judged. The 5.1 ns window includes the near echo but precedes the
    far face's return (600 mm away); the reference faces are 1500 mm away.
    """
    dx, tau = 1e-3, 300e-12
    dt = float(_line(n_layers, 1, 660, dx)[0].dt)
    ky = 2 / dx * np.arcsin(np.sin(np.pi * frequency * dt) * dx / (C0 * dt * np.sqrt(2)))
    phase = complex(np.exp(-1j * ky * dx))
    t = np.arange(int(duration / dt)) * dt
    waveform = jnp.asarray(np.exp(-((t - 5 * tau) / tau)**2)
                           * np.exp(-2j * np.pi * frequency * (t - 5 * tau)), dtype=jnp.complex64)
    records = []
    for interior, distance in ((660, 60), (3000, 1500)):
        grid, propagate = _line(n_layers, 1, interior, dx, phase)
        params, auxiliary = cpml.init_cpml(grid)
        records.append(np.asarray(propagate(params, auxiliary, n_layers + distance, waveform)))
    # The analytic source uses exp(-i omega t), hence the positive DFT sign.
    weight = np.exp(2j * np.pi * frequency * t)
    incident = np.dot(records[1], weight)
    echo = np.dot(records[0] - records[1], weight)
    if not np.isfinite(incident) or abs(incident) < 1e-8 or not np.isfinite(echo):
        raise ValueError("unresolved oblique incident/echo spectrum")
    return float(20 * np.log10(max(abs(echo / incident), 1e-30)))


def absorber_sensitivity(measure, layers):
    """Return elementwise |measure(2L)-measure(L)| / |measure(2L)|.

    ``measure(depth)`` must return the same scalar/array observable in linear
    units at each depth (complex S parameters are supported; do not pass dB).
    All inputs other than absorber depth must stay fixed. Zero at both depths
    gives zero; a nonzero change relative to zero gives infinity. No tolerance
    is imposed: the caller decides how much motion its record can tolerate.
    """
    if not isinstance(layers, (int, np.integer)) or layers < 1:
        raise ValueError("layers must be a positive integer")
    shallow, deep = np.asarray(measure(layers)), np.asarray(measure(2 * layers))
    if shallow.shape != deep.shape or not np.all(np.isfinite(shallow)) or not np.all(np.isfinite(deep)):
        raise ValueError("measurements must have matching shapes and finite values")
    delta, scale = np.abs(deep - shallow), np.abs(deep)
    return np.divide(delta, scale, out=np.where(delta == 0, 0.0, np.inf), where=scale != 0)


def tfsf_auxiliary_reflection(n_layers, face, freqs, *, far=400, reference_distance=900):
    """Return the 1-D TF/SF auxiliary absorber's |R| in dB at 1 mm spacing.

    Use its shipped initializer, coefficients, end termination and split
    updates. Move only the soft source to isolate the near face's echo, as
    in ``plane_wave_reflection``; keep the shipped R_asymptotic=1e-6.
    This tests incident-wave generation before any injection into a 3-D box.
    """
    from rfx.sources.tfsf import init_tfsf, update_tfsf_1d_e, update_tfsf_1d_h

    if face not in ("lo", "hi"):
        raise ValueError("face must be 'lo' or 'hi'")
    dx = 1e-3
    dt = float(_line(8, 1, 430, dx)[0].dt)
    n_steps = int(0.9 * (2 * far * dx / C0) / dt)
    if not 30 < far < reference_distance or n_steps * dt <= 120e-12 + 2 * (30 + n_layers) * dx / C0:
        raise ValueError("increase far/reference_distance to isolate the complete near-face echo")

    def trace(interior, distance):
        # init_tfsf adds 20 margin cells and nx-4 mapped cells.
        cfg, state = init_tfsf(interior - 15, dx, dt, aux_n_cpml=n_layers)
        source_index = n_layers + distance
        cfg = cfg._replace(src_idx=source_index, src_t0=60e-12,
                           src_tau=12e-12, src_amp=0.5)

        def step(state, t):
            state = update_tfsf_1d_h(cfg, state, dx, dt)
            state = update_tfsf_1d_e(cfg, state, dx, dt, t)
            return state, state.e1d[source_index]

        return jax.lax.scan(step, state, jnp.arange(n_steps) * dt)[1]

    short = trace(30 + far, 30 if face == "lo" else far)
    reference = trace(2 * reference_distance, reference_distance)
    return _reflection_db(short, reference, dt, freqs)
