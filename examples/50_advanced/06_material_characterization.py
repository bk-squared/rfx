"""Example: S-parameter Material Characterization

Demonstrates the differentiable material fitting workflow:
  1. Generate synthetic S-parameters from a known Debye material (water)
  2. Use differentiable_material_fit to recover the Debye poles
  3. Compare recovered vs original poles and permittivity spectra

This showcases rfx's unique capability of differentiating through the
full FDTD simulation to fit dispersive material models directly to
S-parameter data.

Saves: examples/50_advanced/06_material_characterization.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import (
    Simulation, Box, GaussianPulse, DebyePole,
)
from rfx.differentiable_material_fit import differentiable_material_fit

OUT_DIR = "examples/50_advanced"


def debye_permittivity(freqs, eps_inf, poles):
    """Compute complex permittivity from Debye model.

    eps(f) = eps_inf + sum_k delta_eps_k / (1 + j*2*pi*f*tau_k)
    """
    omega = 2 * np.pi * freqs
    eps = np.full_like(freqs, eps_inf, dtype=complex)
    for pole in poles:
        eps += pole.delta_eps / (1.0 + 1j * omega * pole.tau)
    return eps


def sim_factory(eps_inf, debye_poles, lorentz_poles):
    """Build a fixture simulation for material characterization.

    Simple transmission line geometry: a dielectric slab (DUT) between
    two ports.
    """
    f_max = 10e9
    dx = 1.5e-3
    dom_x = 0.03  # 30 mm
    dom_y = 0.01
    dom_z = 0.01

    sim = Simulation(
        freq_max=f_max,
        domain=(dom_x, dom_y, dom_z),
        boundary="pec",
        dx=dx,
    )

    # Register DUT material
    sim.add_material("dut", eps_r=float(eps_inf), debye_poles=debye_poles)

    # DUT slab in the center
    sim.add(Box((0.01, 0, 0), (0.02, dom_y, dom_z)), material="dut")

    # Input port
    sim.add_port(
        (0.004, dom_y / 2, dom_z / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f_max / 2, bandwidth=0.8),
    )
    sim.add_probe((0.004, dom_y / 2, dom_z / 2), component="ez")

    # Output port/probe
    sim.add_port(
        (0.026, dom_y / 2, dom_z / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f_max / 2, bandwidth=0.8, amplitude=0.0),
    )
    sim.add_probe((0.026, dom_y / 2, dom_z / 2), component="ez")

    return sim


def main():
    # ---- Ground truth: water at 20C (single Debye pole) ----
    eps_inf_true = 4.9
    true_poles = [DebyePole(delta_eps=74.1, tau=8.3e-12)]

    print("Ground truth material: Water at 20C")
    print(f"  eps_inf = {eps_inf_true}")
    print(f"  Debye pole: delta_eps = {true_poles[0].delta_eps}, tau = {true_poles[0].tau*1e12:.1f} ps")

    # ---- Generate synthetic S-parameters ----
    print("\nGenerating synthetic S-parameters from ground truth ...")
    true_sim = sim_factory(eps_inf_true, true_poles, [])
    true_result = true_sim.run(n_steps=800, compute_s_params=True)

    if true_result.s_params is None:
        print("Warning: S-params not computed, using synthetic proxy")
        freqs = np.linspace(1e9, 10e9, 50)
        s_measured = np.zeros((2, 2, len(freqs)), dtype=complex)
        eps_true_f = debye_permittivity(freqs, eps_inf_true, true_poles)
        gamma = (np.sqrt(eps_true_f) - 1) / (np.sqrt(eps_true_f) + 1)
        s_measured[0, 0, :] = gamma
        s_measured[1, 0, :] = 1 - np.abs(gamma)**2
    else:
        s_measured = np.asarray(true_result.s_params)
        freqs = np.asarray(true_result.freqs)

    print(f"S-param shape: {s_measured.shape}, freq range: {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz")

    # ---- Fit material using differentiable FDTD ----
    n_iter = 20
    print(f"\nFitting material model ({n_iter} iterations) ...")
    fit_result = differentiable_material_fit(
        sim_factory,
        s_measured,
        freqs,
        n_debye_poles=1,
        n_lorentz_poles=0,
        n_iterations=n_iter,
        learning_rate=0.05,
        verbose=True,
    )

    # ---- Compare results ----
    eps_inf_fit = fit_result.eps_inf
    fit_poles = fit_result.debye_poles

    print(f"\n{'='*50}")
    print(f"Material Characterization Results")
    print(f"{'='*50}")
    print(f"{'Parameter':<20s}  {'True':>12s}  {'Recovered':>12s}  {'Error':>10s}")
    print("-" * 60)
    print(f"{'eps_inf':<20s}  {eps_inf_true:12.2f}  {eps_inf_fit:12.2f}  {abs(eps_inf_fit-eps_inf_true)/eps_inf_true*100:9.1f}%")

    if fit_poles:
        de_true = true_poles[0].delta_eps
        de_fit = fit_poles[0].delta_eps
        tau_true = true_poles[0].tau
        tau_fit = fit_poles[0].tau
        print(f"{'delta_eps':<20s}  {de_true:12.2f}  {de_fit:12.2f}  {abs(de_fit-de_true)/de_true*100:9.1f}%")
        print(f"{'tau (ps)':<20s}  {tau_true*1e12:12.2f}  {tau_fit*1e12:12.2f}  {abs(tau_fit-tau_true)/tau_true*100:9.1f}%")

    print(f"\nConverged: {fit_result.converged}")
    print(f"Final loss: {fit_result.loss_history[-1]:.6e}")

    # ---- Compute permittivity spectra ----
    freq_plot = np.linspace(0.1e9, 10e9, 200)
    eps_true = debye_permittivity(freq_plot, eps_inf_true, true_poles)
    eps_fit = debye_permittivity(freq_plot, eps_inf_fit, fit_poles) if fit_poles else np.full_like(freq_plot, eps_inf_fit, dtype=complex)

    # ---- 4-panel figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("S-Parameter Material Characterization (Differentiable FDTD)", fontsize=14, fontweight="bold")

    # Panel 1: Real part of permittivity
    ax = axes[0, 0]
    ax.plot(freq_plot / 1e9, eps_true.real, "b-", lw=2, label="True (water)")
    ax.plot(freq_plot / 1e9, eps_fit.real, "r--", lw=2, label="Recovered")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("eps' (real part)")
    ax.set_title("Permittivity: Real Part")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Imaginary part (loss)
    ax = axes[0, 1]
    ax.plot(freq_plot / 1e9, -eps_true.imag, "b-", lw=2, label="True")
    ax.plot(freq_plot / 1e9, -eps_fit.imag, "r--", lw=2, label="Recovered")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("-eps'' (loss factor)")
    ax.set_title("Permittivity: Loss Factor")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Loss convergence
    ax = axes[1, 0]
    ax.semilogy(fit_result.loss_history, "b.-", lw=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("S-parameter loss")
    ax.set_title("Fitting Convergence")
    ax.grid(True, alpha=0.3)

    # Panel 4: S11 comparison
    ax = axes[1, 1]
    if s_measured.shape[0] >= 1:
        s11_meas_db = 20 * np.log10(np.maximum(np.abs(s_measured[0, 0, :]), 1e-30))
        ax.plot(freqs / 1e9, s11_meas_db, "b-", lw=1.5, label="Measured (synthetic)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S11| (dB)")
    ax.set_title("S11 — Measured Reference")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/06_material_characterization.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
