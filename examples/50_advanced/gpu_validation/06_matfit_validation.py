"""GPU Accuracy Validation: Material Characterization (Debye Fitting)

Validates the material characterization workflow against known water
Debye parameters at 20 degrees C.

Validation criteria:
  - Recover eps_inf within 10% of true value (4.9)
  - Recover delta_eps within 20% of true value (74.1)
  - Recover tau within 20% of true value (8.3 ps)
  - FDTD S11 with fitted material matches true material within 1 dB RMS

Reference:
  Kaatze, "Complex permittivity of water as a function of frequency
  and temperature", J. Chem. Eng. Data, 1989.
  Water at 20C: eps_inf=4.9, delta_eps=74.1, tau=8.3 ps

Exit 0 on PASS, 1 on FAIL.
"""

import sys
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box, GaussianPulse, DebyePole
from rfx.material_fit import fit_debye, eval_debye

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

# Thresholds
EPS_INF_THRESH = 10.0   # % error
DELTA_EPS_THRESH = 20.0  # % error
TAU_THRESH = 20.0         # % error


def debye_permittivity(freqs, eps_inf, poles):
    """Compute complex permittivity from Debye model."""
    omega = 2 * np.pi * freqs
    eps = np.full_like(freqs, eps_inf, dtype=complex)
    for pole in poles:
        eps += pole.delta_eps / (1.0 + 1j * omega * pole.tau)
    return eps


def build_slab_sim(eps_inf, debye_poles, dx=0.8e-3):
    """Build a transmission-line fixture with a dielectric slab."""
    f_max = 10e9
    dom_x = 0.04
    dom_y = 0.012
    dom_z = 0.012

    sim = Simulation(
        freq_max=f_max,
        domain=(dom_x, dom_y, dom_z),
        boundary="pec",
        dx=dx,
    )

    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
    sim.add(Box((0.012, 0, 0), (0.028, dom_y, dom_z)), material="dut")

    sim.add_port(
        (0.005, dom_y / 2, dom_z / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f_max / 2, bandwidth=0.8),
    )
    sim.add_probe((0.005, dom_y / 2, dom_z / 2), component="ez")
    sim.add_probe((0.035, dom_y / 2, dom_z / 2), component="ez")

    return sim


def main():
    t_start = time.time()

    # --- Ground truth: water at 20C ---
    eps_inf_true = 4.9
    true_poles = [DebyePole(delta_eps=74.1, tau=8.3e-12)]
    dx = 0.8e-3

    print("=" * 60)
    print("GPU VALIDATION: Material Characterization (Debye Fit)")
    print("=" * 60)
    print(f"Ground truth: Water at 20C (Kaatze 1989)")
    print(f"  eps_inf   = {eps_inf_true}")
    print(f"  delta_eps = {true_poles[0].delta_eps}")
    print(f"  tau       = {true_poles[0].tau * 1e12:.1f} ps")
    print(f"Resolution  : dx = {dx*1e3:.1f} mm")
    print()

    # --- Generate synthetic permittivity data ---
    freqs = np.linspace(0.1e9, 10e9, 200)
    eps_true = debye_permittivity(freqs, eps_inf_true, true_poles)

    # Realistic measurement noise (2%)
    rng = np.random.default_rng(42)
    noise_level = 0.02
    eps_noisy = eps_true * (1 + noise_level * (rng.standard_normal(len(freqs)) +
                                                1j * rng.standard_normal(len(freqs))))

    print(f"Synthetic data: {len(freqs)} points, {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz")
    print(f"Noise level: {noise_level*100:.0f}%")

    # --- 1-pole Debye fit ---
    print("\nFitting 1-pole Debye model ...")
    fit_1 = fit_debye(freqs, eps_noisy, n_poles=1)

    eps_inf_fit = fit_1.eps_inf
    delta_eps_fit = fit_1.poles[0].delta_eps
    tau_fit = fit_1.poles[0].tau

    eps_inf_err = abs(eps_inf_fit - eps_inf_true) / eps_inf_true * 100
    delta_eps_err = abs(delta_eps_fit - true_poles[0].delta_eps) / true_poles[0].delta_eps * 100
    tau_err = abs(tau_fit - true_poles[0].tau) / true_poles[0].tau * 100

    print(f"  eps_inf   = {eps_inf_fit:.3f}  (true: {eps_inf_true}, err: {eps_inf_err:.2f}%)")
    print(f"  delta_eps = {delta_eps_fit:.3f}  (true: {true_poles[0].delta_eps}, err: {delta_eps_err:.2f}%)")
    print(f"  tau       = {tau_fit * 1e12:.3f} ps  (true: {true_poles[0].tau*1e12:.1f} ps, err: {tau_err:.2f}%)")
    print(f"  Fit error = {fit_1.fit_error * 100:.3f}%")

    # --- 2-pole fit for comparison ---
    print("\nFitting 2-pole Debye model ...")
    fit_2 = fit_debye(freqs, eps_noisy, n_poles=2)
    print(f"  Fit error = {fit_2.fit_error * 100:.3f}%")
    for i, p in enumerate(fit_2.poles):
        print(f"  Pole {i+1}: delta_eps={p.delta_eps:.3f}, tau={p.tau*1e12:.3f} ps")

    # --- FDTD validation: compare S11 from true vs fitted material ---
    print("\nRunning FDTD with true material ...")
    sim_true = build_slab_sim(eps_inf_true, true_poles, dx=dx)
    result_true = sim_true.run(n_steps=2000, compute_s_params=True)

    print("Running FDTD with fitted material ...")
    sim_fit = build_slab_sim(fit_1.eps_inf, list(fit_1.poles), dx=dx)
    result_fit = sim_fit.run(n_steps=2000, compute_s_params=True)

    # --- S11 comparison ---
    s11_rms_err = None
    if (result_true.s_params is not None and result_fit.s_params is not None and
            result_true.freqs is not None):
        s11_true_db = 20 * np.log10(np.maximum(
            np.abs(np.asarray(result_true.s_params)[0, 0, :]), 1e-30))
        s11_fit_db = 20 * np.log10(np.maximum(
            np.abs(np.asarray(result_fit.s_params)[0, 0, :]), 1e-30))
        # RMS difference in dB
        s11_rms_err = float(np.sqrt(np.mean((s11_true_db - s11_fit_db) ** 2)))
        print(f"S11 RMS difference: {s11_rms_err:.3f} dB")

    # --- Evaluate fitted model for plotting ---
    freq_dense = np.linspace(0.1e9, 10e9, 500)
    eps_true_dense = debye_permittivity(freq_dense, eps_inf_true, true_poles)
    eps_fit1_dense = eval_debye(freq_dense, fit_1.eps_inf, fit_1.poles)

    # --- Validation ---
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Parameter':<15s}  {'True':>12s}  {'Recovered':>12s}  {'Error':>8s}  {'Thresh':>8s}")
    print("-" * 60)
    print(f"{'eps_inf':<15s}  {eps_inf_true:12.3f}  {eps_inf_fit:12.3f}  {eps_inf_err:7.2f}%  {EPS_INF_THRESH:7.1f}%")
    print(f"{'delta_eps':<15s}  {true_poles[0].delta_eps:12.3f}  {delta_eps_fit:12.3f}  {delta_eps_err:7.2f}%  {DELTA_EPS_THRESH:7.1f}%")
    print(f"{'tau (ps)':<15s}  {true_poles[0].tau*1e12:12.3f}  {tau_fit*1e12:12.3f}  {tau_err:7.2f}%  {TAU_THRESH:7.1f}%")
    if s11_rms_err is not None:
        print(f"{'S11 RMS (dB)':<15s}  {'':>12s}  {s11_rms_err:12.3f}  {'':>8s}  {'< 1.0':>8s}")
    print(f"{'Fit error':<15s}  {'':>12s}  {fit_1.fit_error*100:11.3f}%")

    criterion_eps_inf = eps_inf_err < EPS_INF_THRESH
    criterion_delta_eps = delta_eps_err < DELTA_EPS_THRESH
    criterion_tau = tau_err < TAU_THRESH
    criterion_s11 = s11_rms_err is not None and s11_rms_err < 1.0

    print(f"\nCriteria:")
    print(f"  eps_inf error < {EPS_INF_THRESH}%    : {'PASS' if criterion_eps_inf else 'FAIL'} ({eps_inf_err:.2f}%)")
    print(f"  delta_eps error < {DELTA_EPS_THRESH}% : {'PASS' if criterion_delta_eps else 'FAIL'} ({delta_eps_err:.2f}%)")
    print(f"  tau error < {TAU_THRESH}%       : {'PASS' if criterion_tau else 'FAIL'} ({tau_err:.2f}%)")
    print(f"  S11 RMS < 1 dB         : {'PASS' if criterion_s11 else 'FAIL'} ({s11_rms_err:.3f} dB)" if s11_rms_err is not None else "  S11 RMS < 1 dB         : N/A")
    print(f"Elapsed time             : {elapsed:.1f}s")

    # --- Figures ---
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("GPU Validation: Material Characterization (Kaatze 1989 Water)",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # Panel 1: Real permittivity
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(freq_dense / 1e9, eps_true_dense.real, "b-", lw=2, label="True")
    ax.plot(freq_dense / 1e9, eps_fit1_dense.real, "r--", lw=2, label="1-pole fit")
    ax.plot(freqs / 1e9, eps_noisy.real, "k.", ms=1.5, alpha=0.3, label="Noisy data")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("eps' (real)")
    ax.set_title("Permittivity: Real Part")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Imaginary permittivity
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(freq_dense / 1e9, -eps_true_dense.imag, "b-", lw=2, label="True")
    ax.plot(freq_dense / 1e9, -eps_fit1_dense.imag, "r--", lw=2, label="1-pole fit")
    ax.plot(freqs / 1e9, -eps_noisy.imag, "k.", ms=1.5, alpha=0.3, label="Noisy data")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("-eps'' (loss)")
    ax.set_title("Permittivity: Loss Factor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Fit residuals
    ax = fig.add_subplot(gs[0, 2])
    eps_eval1 = eval_debye(freqs, fit_1.eps_inf, fit_1.poles)
    resid_1 = np.abs(eps_eval1 - eps_true) / np.abs(eps_true) * 100
    ax.semilogy(freqs / 1e9, resid_1, "r-", lw=1.5, label=f"1-pole ({fit_1.fit_error*100:.2f}%)")
    if fit_2.fit_error < fit_1.fit_error:
        eps_eval2 = eval_debye(freqs, fit_2.eps_inf, fit_2.poles)
        resid_2 = np.abs(eps_eval2 - eps_true) / np.abs(eps_true) * 100
        ax.semilogy(freqs / 1e9, resid_2, "g-", lw=1.5, label=f"2-pole ({fit_2.fit_error*100:.2f}%)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Fit Residuals vs True")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: FDTD S11 comparison
    ax = fig.add_subplot(gs[1, 0])
    if result_true.s_params is not None and result_true.freqs is not None:
        f_ghz = np.asarray(result_true.freqs) / 1e9
        ax.plot(f_ghz, s11_true_db, "b-", lw=1.5, label="True material")
    if result_fit.s_params is not None and result_fit.freqs is not None:
        f_ghz2 = np.asarray(result_fit.freqs) / 1e9
        ax.plot(f_ghz2, s11_fit_db, "r--", lw=1.5, label="Fitted material")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S11| (dB)")
    ax.set_title("FDTD S11: True vs Fitted")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: Time-domain comparison
    ax = fig.add_subplot(gs[1, 1])
    ts_true = np.asarray(result_true.time_series)
    ts_fit = np.asarray(result_fit.time_series)
    dt = result_true.dt
    t_ns = np.arange(ts_true.shape[0]) * dt * 1e9
    ax.plot(t_ns, ts_true[:, 0], "b-", lw=0.8, alpha=0.8, label="True")
    ax.plot(t_ns, ts_fit[:, 0], "r--", lw=0.8, alpha=0.8, label="Fitted")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ez at port")
    ax.set_title("Time-Domain Port Signal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    all_pass = criterion_eps_inf and criterion_delta_eps and criterion_tau and criterion_s11
    verdict = "PASS" if all_pass else "FAIL"
    lines = [
        "Material Characterization Validation",
        "-" * 40,
        f"Material: Water at 20C (Kaatze 1989)",
        f"dx = {dx*1e3:.1f} mm, noise = {noise_level*100:.0f}%",
        "",
        f"{'Param':<12s} {'True':>10s} {'Fit':>10s} {'Err':>7s}",
        "-" * 40,
        f"{'eps_inf':<12s} {eps_inf_true:10.2f} {eps_inf_fit:10.2f} {eps_inf_err:6.1f}%",
        f"{'delta_eps':<12s} {true_poles[0].delta_eps:10.2f} {delta_eps_fit:10.2f} {delta_eps_err:6.1f}%",
        f"{'tau (ps)':<12s} {true_poles[0].tau*1e12:10.2f} {tau_fit*1e12:10.2f} {tau_err:6.1f}%",
        "",
        f"S11 RMS diff: {s11_rms_err:.3f} dB" if s11_rms_err else "S11 RMS diff: N/A",
        f"Fit error: {fit_1.fit_error*100:.3f}%",
        "",
        f"Verdict: {verdict}",
        f"Time: {elapsed:.1f}s",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    out_path = os.path.join(OUT_DIR, "06_matfit_validation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # --- Pass/Fail ---
    passed = all_pass
    if passed:
        print(f"\nPASS: All Debye parameters recovered within thresholds")
        sys.exit(0)
    else:
        print(f"\nFAIL: One or more parameters outside threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
