"""Tests for Debye dispersive materials via ADE.

Validates the Debye ADE implementation against analytical predictions:
1. Coefficients match hand-computed values
2. Debye medium slows propagation (higher effective ε at low freq)
3. Energy is bounded (no ADE instability)
4. Frequency-dependent permittivity matches Debye formula
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import C0
from rfx.core.yee import (
    init_state, init_materials, update_h, EPS_0, MU_0,
)
from rfx.boundaries.pec import apply_pec
from rfx.materials.debye import (
    DebyePole, init_debye, update_e_debye,
)


def _total_energy(state, dx):
    """Total EM energy in the grid."""
    e_sq = jnp.sum(state.ex**2 + state.ey**2 + state.ez**2)
    h_sq = jnp.sum(state.hx**2 + state.hy**2 + state.hz**2)
    return float(0.5 * EPS_0 * e_sq * dx**3 + 0.5 * MU_0 * h_sq * dx**3)


def test_debye_coefficients():
    """Verify ADE coefficients match hand-computed values for a single pole."""
    shape = (5, 5, 5)
    dt = 1e-12  # 1 ps
    tau = 10e-12  # 10 ps
    delta_eps = 3.0
    eps_inf = 2.0

    materials = init_materials(shape)
    materials = materials._replace(
        eps_r=jnp.full(shape, eps_inf, dtype=jnp.float32)
    )

    poles = [DebyePole(delta_eps=delta_eps, tau=tau)]
    coeffs, dstate = init_debye(poles, materials, dt)

    # Hand-computed coefficients
    alpha_expected = (2 * tau - dt) / (2 * tau + dt)
    beta_expected = EPS_0 * delta_eps * dt / (2 * tau + dt)
    gamma_expected = EPS_0 * eps_inf + beta_expected  # sigma=0
    ca_expected = (EPS_0 * eps_inf - beta_expected) / gamma_expected
    cb_expected = dt / gamma_expected
    cc_expected = (1 - alpha_expected) / gamma_expected

    assert abs(float(coeffs.alpha[0, 2, 2, 2]) - alpha_expected) < 1e-6, \
        f"alpha: {float(coeffs.alpha[0,2,2,2]):.8f} vs {alpha_expected:.8f}"
    assert abs(float(coeffs.beta[0, 2, 2, 2]) - beta_expected) / beta_expected < 1e-4, \
        "beta mismatch"
    assert abs(float(coeffs.ca[2, 2, 2]) - ca_expected) / abs(ca_expected) < 1e-4, \
        f"ca: {float(coeffs.ca[2,2,2]):.8f} vs {ca_expected:.8f}"
    assert abs(float(coeffs.cb[2, 2, 2]) - cb_expected) / cb_expected < 1e-4, \
        "cb mismatch"
    assert abs(float(coeffs.cc[0, 2, 2, 2]) - cc_expected) / cc_expected < 1e-4, \
        "cc mismatch"

    # Polarization state should be zeros
    assert float(jnp.max(jnp.abs(dstate.px))) == 0.0
    assert float(jnp.max(jnp.abs(dstate.py))) == 0.0
    assert float(jnp.max(jnp.abs(dstate.pz))) == 0.0

    print(f"\nDebye coefficients (τ={tau*1e12:.0f} ps, Δε={delta_eps}, ε_∞={eps_inf}):")
    print(f"  α={alpha_expected:.6f}, β={beta_expected:.4e}")
    print(f"  Ca={ca_expected:.6f}, Cb={cb_expected:.4e}, Cc={cc_expected:.4e}")


def test_debye_medium_slower_propagation():
    """Pulse in Debye medium arrives later at a downstream monitor.

    Arrival-time approach: record |Ez| time series at a fixed monitor,
    compare the step of peak arrival.  Near-lossless Debye (ε_∞=4,
    tiny Δε) behaves like a simple dielectric — pulse should arrive
    ~2× later than in vacuum (v ≈ c/√ε_∞).
    """
    nx = 200
    shape = (nx, 6, 6)
    dx = 0.0005  # 0.5 mm
    dt = 0.99 * dx / (C0 * np.sqrt(3))

    f0 = 10e9
    tau_pulse = 1.0 / (f0 * 0.5 * np.pi)
    t0_pulse = 3.0 * tau_pulse
    src_x = 20
    mon_x = 80   # 30 mm downstream
    cy, cz = 3, 3
    n_steps = 600

    peak_steps = {}
    for label, debye_poles, eps_inf in [
        ("vacuum", [], 1.0),
        # Near-lossless dielectric via Debye path (ε_∞=4, tiny Δε)
        ("debye", [DebyePole(delta_eps=0.01, tau=1e-9)], 4.0),
    ]:
        materials = init_materials(shape)
        if eps_inf != 1.0:
            materials = materials._replace(
                eps_r=jnp.full(shape, eps_inf, dtype=jnp.float32)
            )

        if debye_poles:
            coeffs, dstate = init_debye(debye_poles, materials, dt)
        else:
            coeffs, dstate = None, None

        state = init_state(shape)
        mon_series = []

        for step in range(n_steps):
            t = step * dt
            state = update_h(state, materials, dt, dx)

            if coeffs is not None:
                state, dstate = update_e_debye(state, coeffs, dstate, dt, dx)
            else:
                from rfx.core.yee import update_e
                state = update_e(state, materials, dt, dx)

            state = apply_pec(state)

            if t < 6 * t0_pulse:
                arg = (t - t0_pulse) / tau_pulse
                src_val = (-2.0 * arg) * np.exp(-(arg**2))
                state = state._replace(
                    ez=state.ez.at[src_x, :, :].add(src_val)
                )

            mon_series.append(float(state.ez[mon_x, cy, cz]))

        peak_step = int(np.argmax(np.abs(np.array(mon_series))))
        peak_steps[label] = peak_step

    vac_step = peak_steps["vacuum"]
    deb_step = peak_steps["debye"]

    print(f"\nArrival at monitor x={mon_x} ({(mon_x-src_x)*dx*1e3:.0f} mm):")
    print(f"  Vacuum peak step: {vac_step}")
    print(f"  Debye  peak step: {deb_step}")

    # Debye medium (ε_∞=4) should arrive later
    assert deb_step > vac_step, \
        f"Debye peak ({deb_step}) should arrive after vacuum ({vac_step})"
    # Expected ratio ~ √4 = 2
    if vac_step > 0:
        ratio = deb_step / vac_step
        print(f"  Time ratio: {ratio:.2f}x (expected ~{np.sqrt(4.0):.1f}x)")
        assert ratio > 1.3, f"Expected >1.3x time ratio, got {ratio:.2f}"


def test_debye_energy_bounded():
    """Debye ADE should not produce energy growth (no instability).

    Initialize cavity mode in Debye medium, run for many steps.
    Energy should decay (Debye is dissipative) or stay bounded.
    """
    shape = (20, 20, 20)
    dx = 0.002
    dt = 0.99 * dx / (C0 * np.sqrt(3))

    # Debye medium: water-like (ε_∞=4, Δε=76, τ=8ps)
    # Using reduced values to keep field dynamics reasonable
    eps_inf = 4.0
    materials = init_materials(shape)
    materials = materials._replace(
        eps_r=jnp.full(shape, eps_inf, dtype=jnp.float32)
    )

    poles = [DebyePole(delta_eps=10.0, tau=50e-12)]
    coeffs, dstate = init_debye(poles, materials, dt)

    # Initialize with cavity mode
    Lx = (shape[0] - 1) * dx
    Ly = (shape[1] - 1) * dx
    x = np.arange(shape[0]) * dx
    y = np.arange(shape[1]) * dx
    ez_init = np.sin(np.pi * x[:, None, None] / Lx) * \
              np.sin(np.pi * y[None, :, None] / Ly) * \
              np.ones((1, 1, shape[2]))

    state = init_state(shape)
    state = state._replace(ez=jnp.array(ez_init, dtype=jnp.float32))
    initial_energy = _total_energy(state, dx)

    max_energy = initial_energy
    for step in range(1000):
        state = update_h(state, materials, dt, dx)
        state, dstate = update_e_debye(state, coeffs, dstate, dt, dx)
        state = apply_pec(state)

        if step % 100 == 0:
            e = _total_energy(state, dx)
            max_energy = max(max_energy, e)

    final_energy = _total_energy(state, dx)

    print("\nDebye energy stability:")
    print(f"  Initial: {initial_energy:.4e}, Max: {max_energy:.4e}, Final: {final_energy:.4e}")

    # Energy should never exceed initial by more than a small margin
    # (Debye is dissipative, energy should decrease)
    assert max_energy < initial_energy * 1.05, \
        f"Energy grew: max={max_energy:.4e} > 1.05 * initial={initial_energy:.4e}"

    # Debye medium is lossy at high frequencies → energy should decay
    assert final_energy < initial_energy * 0.9, \
        f"Expected Debye loss: final/initial = {final_energy/initial_energy:.4f}"


def test_debye_mask_selective():
    """Debye dispersion should only apply where mask is True."""
    shape = (20, 10, 10)
    dx = 0.002
    dt = 0.99 * dx / (C0 * np.sqrt(3))

    materials = init_materials(shape)

    # Apply Debye only in the right half (x >= 10)
    mask = jnp.zeros(shape, dtype=bool)
    mask = mask.at[10:, :, :].set(True)

    poles = [DebyePole(delta_eps=5.0, tau=20e-12)]
    coeffs, dstate = init_debye(poles, materials, dt, mask=mask)

    # In vacuum region (x < 10): ca should be standard vacuum coefficient
    # ca_vac = (eps0 - 0 - 0) / (eps0 + 0 + 0) = 1.0
    ca_vac = float(coeffs.ca[5, 5, 5])
    # In Debye region (x >= 10): ca should be modified
    ca_deb = float(coeffs.ca[15, 5, 5])

    print("\nDebye mask test:")
    print(f"  Ca(vacuum) = {ca_vac:.6f}")
    print(f"  Ca(Debye)  = {ca_deb:.6f}")

    # Vacuum: beta=0, sigma=0, so ca = (eps0 - 0) / (eps0 + 0) = 1.0
    assert abs(ca_vac - 1.0) < 1e-4, f"Vacuum ca={ca_vac}, expected 1.0"
    # Debye: ca < 1 (beta > 0 shifts it)
    assert ca_deb < 1.0, f"Debye ca={ca_deb}, expected < 1.0"

    # Alpha should be 0 in vacuum region, nonzero in Debye region
    alpha_vac = float(coeffs.alpha[0, 5, 5, 5])
    alpha_deb = float(coeffs.alpha[0, 15, 5, 5])
    assert alpha_vac == 0.0, f"Vacuum alpha={alpha_vac}, expected 0"
    assert alpha_deb > 0.0, f"Debye alpha={alpha_deb}, expected > 0"


def test_debye_two_poles():
    """Two Debye poles should both contribute to dispersion."""
    shape = (10, 10, 10)
    dt = 1e-12

    materials = init_materials(shape)

    poles = [
        DebyePole(delta_eps=3.0, tau=10e-12),
        DebyePole(delta_eps=5.0, tau=100e-12),
    ]
    coeffs, dstate = init_debye(poles, materials, dt)

    # Should have 2 poles in alpha/beta arrays
    assert coeffs.alpha.shape[0] == 2
    assert coeffs.beta.shape[0] == 2
    assert coeffs.cc.shape[0] == 2

    # P state should have 2 poles
    assert dstate.px.shape[0] == 2

    # Each pole should have different alpha (different tau)
    a0 = float(coeffs.alpha[0, 5, 5, 5])
    a1 = float(coeffs.alpha[1, 5, 5, 5])
    assert a0 != a1, f"Two poles should have different alpha: {a0} vs {a1}"

    # Expected values
    a0_exp = (2 * 10e-12 - dt) / (2 * 10e-12 + dt)
    a1_exp = (2 * 100e-12 - dt) / (2 * 100e-12 + dt)
    assert abs(a0 - a0_exp) < 1e-6
    assert abs(a1 - a1_exp) < 1e-6

    print(f"\nTwo-pole Debye: α₁={a0:.6f}, α₂={a1:.6f}")
