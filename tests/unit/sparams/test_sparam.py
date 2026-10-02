"""S-parameter extraction validation (lumped-port limiting cases).

Tests:
1. Lumped port in a PEC cavity: S11 near 0 dB in-band (full reflection).
   A lossless PEC cavity dissipates nothing, so |S11| ≈ 1; pinned as a
   loose limiting-case bound (mid-band mean |S11| > -3 dB).
2. Lumped port injects energy: E and H fields become non-zero after
   driving the port (excitation smoke test).

There is NO matched-load absorption gate (S11 < -20 dB) here: a lumped port
terminated in its own impedance in an open (CPML) domain is not exercised in
this file, so that absorption physics is not asserted.
"""

import functools

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.grid import Grid
from rfx.ringdown import RingdownSpec
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse, LumpedPort, setup_lumped_port, apply_lumped_port
from rfx.probes.probes import (
    init_sparam_probe, update_lumped_drive_ref_probe,
    update_sparam_probe, extract_s11,
)


_C0 = 299792458.0


@functools.lru_cache(maxsize=1)
def _pec_cavity_port_s11():
    """One 60-period run of a 50 ohm one-cell lumped port at the centre of a
    closed PEC box, shared by the two cavity tests below. Returns
    ``(freqs_hz, s11, f_tm110_hz)`` with TM110 from the REALIZED box."""
    a, b, d = 0.05, 0.05, 0.025
    grid = Grid(freq_max=5e9, domain=(a, b, d), cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    # Lumped port at center
    port_pos = (a / 2, b / 2, d / 2)
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0)
    port = LumpedPort(
        position=port_pos,
        component="ez",
        impedance=50.0,
        excitation=pulse,
    )

    # Fold port impedance into materials
    materials = setup_lumped_port(grid, port, materials)

    # Frequency points for S-parameter extraction
    freqs = jnp.linspace(1e9, 5e9, 50)
    dt, dx = grid.dt, grid.dx
    num_steps = grid.num_timesteps(num_periods=60)
    sprobe = init_sparam_probe(grid, port, freqs, dft_total_steps=num_steps)

    for n in range(num_steps):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        # Both slots, the way the JIT scan uses them: the PRE-injection drive
        # sample into v_ref, the physical V/I AFTER injection
        # (scripts/diagnostics/lumped_port_known_load_line.py).  This loop
        # sampled only before injection while extract_s11 became the DRIVEN
        # terminal reflection, and the mismatched pair read |S11| up to 1.4468
        # on this lossless cavity — above unity, which a passive structure
        # cannot do.
        sprobe = update_lumped_drive_ref_probe(sprobe, state, grid, port, dt)
        state = apply_lumped_port(state, grid, port, t, materials)
        sprobe = update_sparam_probe(sprobe, state, grid, port, dt)

    s11 = np.asarray(extract_s11(sprobe, z0=50.0))
    # Realized, not declared: apply_pec zeroes the tangential E on node planes
    # 0 and n-1, so the box is (n-1)*dx on a side -- 50.96 x 50.96 mm here,
    # not the declared 50 x 50 mm. Ez-polarised TM110 (no z variation):
    # f = c/2 * sqrt(1/a^2 + 1/b^2) = 4.1595 GHz (4.2397 GHz declared).
    a_r = (grid.shape[0] - 1) * dx
    b_r = (grid.shape[1] - 1) * dx
    f_tm110 = _C0 / 2.0 * np.sqrt(1.0 / a_r ** 2 + 1.0 / b_r ** 2)
    return np.asarray(freqs), s11, float(f_tm110)


#: Below this, the box has no mode: |S11| = 1 exactly and Zin is reactive.
_BELOW_MODE = 0.9


def test_lumped_port_pec_cavity_s11():
    """A lumped port in a lossless PEC box, below the box's first mode.

    A closed PEC box dissipates nothing, so a port that drives it sees a pure
    reactance: |S11| = 1 at every frequency. Below the first mode (TM110,
    4.1595 GHz on the realized 50.96 mm box; this test reads f <= 0.9 x TM110
    = 3.74 GHz, 34 bins) that is a clean statement and it is live here: the
    same float32 bar as before, |S11| <= 1 + 1e-3.

    Measured (60-period record, the test's own loop): max |S11| 1.00059 at
    3.04 GHz, Re Zin -8.0 ... +15.2 ohm around a mean of 0.01 ohm on a |Zin|
    of ~800-3000 ohm. With the record at 240 periods the same bins reach
    1.00133 (3.69 GHz): the leakage of the truncated TM110 ringing (Q ~ 2000,
    e-fold 153 ns; #1255), smaller here than at the mode. Before #1236 the
    port was also a 50 ohm resistor on the Ex and Ey edges at its node, and
    the box read Re Zin = +15.0 ohm on average below the mode (8.5 ... 26.6):
    a loss the lossless box does not have, which kept |S11| below 1 (max
    0.99961) and also damped the ringing. Split from the band around the
    mode by PI decision 2026-09-24 (the other half is the wire-port
    completion test below).
    """
    freqs, s11, f_tm110 = _pec_cavity_port_s11()
    s11_mag = np.abs(s11)
    s11_db = 20 * np.log10(np.maximum(s11_mag, 1e-10))

    # The original lower bound: a PEC cavity reflects most of what it is
    # given (mid-band mean above -3 dB).
    mid_band = (freqs > 1.5e9) & (freqs < 4.5e9)
    assert np.mean(s11_db[mid_band]) > -3.0, \
        f"Mean S11 {np.mean(s11_db[mid_band]):.1f} dB too low for PEC cavity"

    below = freqs <= _BELOW_MODE * f_tm110
    assert int(below.sum()) == 34, "the below-mode band moved; re-derive it"
    assert np.max(s11_mag[below]) <= 1.0 + 1e-3, (
        f"|S11| = {np.max(s11_mag[below]):.5f} below the box's first mode "
        f"({_BELOW_MODE} x TM110 = {_BELOW_MODE * f_tm110 / 1e9:.3f} GHz) "
        "exceeds unity on a lossless PEC cavity, which a passive structure "
        "cannot do")


def _pec_cavity_simulation(*, wire):
    """The low-level cavity fixture's geometry and drive through the public API."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.025), cpml_layers=0)
    sim = Simulation(
        freq_max=5e9, domain=(0.05, 0.05, 0.025), boundary="pec",
        cpml_layers=0, dx=grid.dx,
    )
    sim.add_port(
        position=(0.025, 0.025, 0.0125), component="ez", impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0),
        **({"extent": grid.dx} if wire else {}),
    )
    return sim, grid


def test_wire_port_pec_cavity_s11_around_the_first_mode():
    """A lossless box reflects fully; the port is its only loss.

    TM110 has Q ≈ 2000 and rings with a 153 ns amplitude e-fold time. A plain
    12 ns record reads |S11| = 0.975–1.009 around the mode; wire-port ringdown
    completion restores unity within 1e-6. Check the same band around and
    above TM110 (> 0.9 times the realized-box mode) as the former lumped-port
    xfail, with a two-sided 1e-3 bound and the completion's own witnesses.
    """
    sim, grid = _pec_cavity_simulation(wire=True)
    freqs = jnp.linspace(1e9, 5e9, 50)
    n_steps = grid.num_timesteps(num_periods=60)
    result = sim.run(
        n_steps=n_steps, s_param_n_steps=n_steps, s_param_freqs=freqs,
        compute_s_params=True, ringdown=RingdownSpec(),
    )
    a_r = (grid.shape[0] - 1) * grid.dx
    b_r = (grid.shape[1] - 1) * grid.dx
    f_tm110 = _C0 / 2.0 * np.sqrt(1.0 / a_r ** 2 + 1.0 / b_r ** 2)
    around = np.asarray(freqs) > _BELOW_MODE * f_tm110
    assert int(around.sum()) == 16, "the around-mode band moved; re-derive it"
    s11 = np.asarray(result.ringdown.s_params)[0, 0]
    deviation = np.max(np.abs(np.abs(s11[around]) - 1.0))
    assert deviation <= 1e-3, f"max ||S11| - 1| = {deviation:.6g} in a lossless box"
    report = result.ringdown.report
    assert report.completed and report.ok, report.failure
    assert report.witnesses
    assert all(w.ok for w in report.witnesses), report.witnesses


def test_lumped_port_pec_cavity_ringdown_refused():
    """Completion currently refuses the same box's one-cell lumped port."""
    sim, grid = _pec_cavity_simulation(wire=False)
    n_steps = grid.num_timesteps(num_periods=60)
    with pytest.raises(NotImplementedError, match="one-cell lumped port"):
        sim.run(
            n_steps=n_steps, s_param_n_steps=n_steps,
            s_param_freqs=jnp.linspace(1e9, 5e9, 50),
            compute_s_params=True, ringdown=RingdownSpec(),
        )


def test_lumped_port_injects_energy():
    """Lumped port should inject energy into the simulation."""
    grid = Grid(freq_max=3e9, domain=(0.05, 0.05, 0.025), cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    port_pos = (0.025, 0.025, 0.0125)
    pulse = GaussianPulse(f0=2e9, bandwidth=0.5, amplitude=1.0)
    port = LumpedPort(
        position=port_pos,
        component="ez",
        impedance=50.0,
        excitation=pulse,
    )

    # Fold port impedance into materials
    materials = setup_lumped_port(grid, port, materials)

    dt, dx = grid.dt, grid.dx

    # Run 100 steps with port excitation
    for n in range(100):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        state = apply_lumped_port(state, grid, port, t, materials)

    # Check that fields are non-zero
    total_e = float((state.ex**2 + state.ey**2 + state.ez**2).sum())
    total_h = float((state.hx**2 + state.hy**2 + state.hz**2).sum())

    print("\nAfter 100 steps with lumped port:")
    print(f"  Total E²: {total_e:.4e}")
    print(f"  Total H²: {total_h:.4e}")

    assert total_e > 0, "Lumped port did not inject any E-field energy"
    assert total_h > 0, "Lumped port did not inject any H-field energy"
