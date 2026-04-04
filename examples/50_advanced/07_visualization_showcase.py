"""Example: Comprehensive Visualization Showcase

Demonstrates ALL rfx visualization capabilities in a single script:
  1. Field slice at multiple timesteps
  2. S-parameter plot (dB magnitude)
  3. S-parameter phase plot
  4. Smith chart
  5. Radiation pattern (polar)
  6. Convergence study plot
  7. Parametric sweep plot
  8. Field animation (GIF)

Uses a dipole antenna as the test structure since it produces
meaningful results across all visualization types.

Saves multiple PNGs + one GIF to examples/50_advanced/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import (
    Simulation, Box, GaussianPulse, SnapshotSpec,
    plot_field_slice, plot_s_params, plot_smith,
    plot_radiation_pattern, plot_time_series,
    save_field_animation,
    convergence_study, parametric_sweep, plot_sweep,
)
from rfx.farfield import compute_far_field, radiation_pattern, directivity

OUT_DIR = "examples/50_advanced"


def build_dipole_sim(dx=None):
    """Build a compact dipole antenna simulation."""
    f0 = 5e9
    C0 = 3e8
    lam = C0 / f0
    cpml = 6  # fewer CPML layers for compact domain
    if dx is None:
        dx = lam / 15  # ~4 mm — coarser for CPU speed
    dom = lam * 1.2  # ~72 mm — enough room for CPML + NTFF

    sim = Simulation(
        freq_max=f0 * 1.5,
        domain=(dom, dom, dom),
        boundary="cpml",
        cpml_layers=cpml,
        dx=dx,
    )

    center = (dom / 2, dom / 2, dom / 2)

    # Dipole port
    sim.add_port(
        center,
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f0, bandwidth=0.6),
    )
    sim.add_probe(center, component="ez")

    # Off-axis probe
    sim.add_probe(
        (dom / 2 + dom * 0.1, dom / 2, dom / 2),
        component="ez",
    )

    # NTFF box — ensure enough clearance from CPML
    ntff_margin = (cpml + 2) * dx
    sim.add_ntff_box(
        corner_lo=(ntff_margin, ntff_margin, ntff_margin),
        corner_hi=(dom - ntff_margin, dom - ntff_margin, dom - ntff_margin),
        freqs=jnp.array([f0]),
    )

    return sim, f0


def main():
    print("=" * 60)
    print("rfx Visualization Showcase")
    print("=" * 60)

    f0 = 5e9
    C0 = 3e8
    lam = C0 / f0

    # ==================================================================
    # 1. Main simulation with snapshots
    # ==================================================================
    print("\n[1/8] Running main simulation with field snapshots ...")
    sim, f0 = build_dipole_sim()

    snapshot_spec = SnapshotSpec(
        interval=20,
        components=("ez",),
        slice_axis=2,  # z-axis
        slice_index=None,  # center
    )

    result = sim.run(
        n_steps=500,
        compute_s_params=True,
        snapshot=snapshot_spec,
    )
    grid = sim._build_grid()
    print(f"  Steps: 500, dt={result.dt*1e12:.2f} ps")

    # ==================================================================
    # 2. Field slices at multiple timesteps
    # ==================================================================
    print("[2/8] Plotting field slices ...")
    if result.snapshots is not None and "ez" in result.snapshots:
        snap = np.asarray(result.snapshots["ez"])
        n_frames = snap.shape[0]
        n_show = min(4, n_frames)
        frame_indices = np.linspace(0, n_frames - 1, n_show, dtype=int)

        fig_field, axes_f = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
        if n_show == 1:
            axes_f = [axes_f]
        for i, fi in enumerate(frame_indices):
            frame = snap[fi]
            if frame.ndim == 3:
                frame = frame[:, :, frame.shape[2] // 2]
            vmax = float(np.max(np.abs(snap))) or 1.0
            axes_f[i].imshow(frame.T, origin="lower", cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax, aspect="equal")
            axes_f[i].set_title(f"Step {fi * snapshot_spec.interval}")
            axes_f[i].set_xlabel("x (cells)")
            if i == 0:
                axes_f[i].set_ylabel("y (cells)")
        fig_field.suptitle("Ez Field Evolution (z-midplane)", fontsize=12)
        plt.tight_layout()
        fig_field.savefig(f"{OUT_DIR}/07_field_slices.png", dpi=150)
        plt.close(fig_field)
        print(f"  Saved: {OUT_DIR}/07_field_slices.png")
    else:
        # Fallback: use final state
        fig_field = plot_field_slice(result.state, grid, component="ez", axis="z")
        fig_field.savefig(f"{OUT_DIR}/07_field_slices.png", dpi=150)
        plt.close(fig_field)
        print(f"  Saved: {OUT_DIR}/07_field_slices.png (final state only)")

    # ==================================================================
    # 3. S-parameter plot (dB)
    # ==================================================================
    print("[3/8] Plotting S-parameters (dB) ...")
    if result.s_params is not None and result.freqs is not None:
        fig_sp = plot_s_params(
            np.asarray(result.s_params),
            np.asarray(result.freqs),
            title="S-Parameters (Dipole Antenna)",
        )
        fig_sp.savefig(f"{OUT_DIR}/07_s_params_db.png", dpi=150)
        plt.close(fig_sp)
        print(f"  Saved: {OUT_DIR}/07_s_params_db.png")

        # ==============================================================
        # 4. S-parameter phase
        # ==============================================================
        print("[4/8] Plotting S-parameter phase ...")
        s_params = np.asarray(result.s_params)
        freqs_ghz = np.asarray(result.freqs) / 1e9

        fig_phase, ax_ph = plt.subplots(figsize=(8, 5))
        n_ports = s_params.shape[0]
        for i in range(n_ports):
            for j in range(n_ports):
                phase_deg = np.degrees(np.angle(s_params[i, j, :]))
                ax_ph.plot(freqs_ghz, phase_deg, label=f"S{i+1}{j+1}")
        ax_ph.set_xlabel("Frequency (GHz)")
        ax_ph.set_ylabel("Phase (degrees)")
        ax_ph.set_title("S-Parameter Phase")
        ax_ph.legend()
        ax_ph.grid(True, alpha=0.3)
        fig_phase.tight_layout()
        fig_phase.savefig(f"{OUT_DIR}/07_s_params_phase.png", dpi=150)
        plt.close(fig_phase)
        print(f"  Saved: {OUT_DIR}/07_s_params_phase.png")

        # ==============================================================
        # 5. Smith chart
        # ==============================================================
        print("[5/8] Plotting Smith chart ...")
        s11 = s_params[0, 0, :]
        freqs_arr = np.asarray(result.freqs)
        fig_smith, ax_smith = plt.subplots(figsize=(7, 7))
        plot_smith(s11, freqs_arr, ax=ax_smith, markers=[f0],
                   title="Smith Chart (Dipole S11)")
        fig_smith.savefig(f"{OUT_DIR}/07_smith_chart.png", dpi=150)
        plt.close(fig_smith)
        print(f"  Saved: {OUT_DIR}/07_smith_chart.png")
    else:
        print("  [3-5] Skipped: no S-parameters available")

    # ==================================================================
    # 6. Radiation pattern
    # ==================================================================
    print("[6/8] Plotting radiation pattern ...")
    if result.ntff_data is not None and result.ntff_box is not None:
        theta = np.linspace(0, np.pi, 181)
        phi = np.array([0.0, np.pi / 2])
        ff = compute_far_field(result.ntff_data, result.ntff_box, grid, theta, phi)
        D = directivity(ff)
        pat = radiation_pattern(ff)

        fig_rad, (ax_e, ax_h) = plt.subplots(1, 2, figsize=(12, 5),
                                               subplot_kw={"projection": "polar"})
        # E-plane
        e_plane = np.maximum(pat[0, :, 0], -40) + 40
        ax_e.plot(theta, e_plane, "b-", lw=2)
        ax_e.plot(-theta + 2 * np.pi, e_plane, "b-", lw=2, alpha=0.5)
        ax_e.set_theta_zero_location("N")
        ax_e.set_theta_direction(-1)
        ax_e.set_title(f"E-plane, D={D[0]:.1f} dBi", pad=15)

        # H-plane
        h_plane = np.maximum(pat[0, :, 1], -40) + 40
        ax_h.plot(theta, h_plane, "r-", lw=2)
        ax_h.plot(-theta + 2 * np.pi, h_plane, "r-", lw=2, alpha=0.5)
        ax_h.set_theta_zero_location("N")
        ax_h.set_theta_direction(-1)
        ax_h.set_title(f"H-plane, D={D[0]:.1f} dBi", pad=15)

        fig_rad.suptitle(f"Radiation Pattern at {f0/1e9:.1f} GHz", fontsize=12)
        plt.tight_layout()
        fig_rad.savefig(f"{OUT_DIR}/07_radiation_pattern.png", dpi=150)
        plt.close(fig_rad)
        print(f"  Saved: {OUT_DIR}/07_radiation_pattern.png")
    else:
        print("  Skipped: no NTFF data available")

    # ==================================================================
    # 7. Convergence study
    # ==================================================================
    print("[7/8] Running convergence study ...")

    def cavity_factory(dx_val):
        """PEC cavity for convergence."""
        sim_c = Simulation(
            freq_max=10e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            dx=dx_val,
        )
        sim_c.add_source((0.01, 0.01, 0.01), "ez",
                         waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
        sim_c.add_probe((0.01, 0.01, 0.01), "ez")
        return sim_c

    def peak_metric(r):
        return float(np.max(np.abs(np.asarray(r.time_series))))

    dx_list = [2e-3, 1.5e-3, 1e-3]
    conv = convergence_study(
        cavity_factory, dx_list, peak_metric,
        n_steps=300,
    )
    print(conv.summary())
    fig_conv = conv.plot(title="Mesh Convergence (Peak Ez)")
    fig_conv.savefig(f"{OUT_DIR}/07_convergence.png", dpi=150)
    plt.close(fig_conv)
    print(f"  Saved: {OUT_DIR}/07_convergence.png")

    # ==================================================================
    # 8. Parametric sweep
    # ==================================================================
    print("[8/8] Running parametric sweep ...")

    def sweep_factory(eps_val):
        """Vary substrate eps_r."""
        sim_s = Simulation(
            freq_max=10e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            dx=1.5e-3,
        )
        sim_s.add_material("sub", eps_r=float(eps_val))
        sim_s.add(Box((0, 0, 0), (0.02, 0.02, 0.005)), material="sub")
        sim_s.add_port((0.01, 0.01, 0.008), "ez", impedance=50.0,
                       waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
        sim_s.add_probe((0.01, 0.01, 0.008), "ez")
        return sim_s

    eps_values = [1.0, 2.2, 4.4, 6.15, 9.8]
    sweep = parametric_sweep(sweep_factory, "eps_r", eps_values, n_steps=400)
    fig_sweep = plot_sweep(sweep, metric="peak_field",
                           title="Parametric Sweep: Peak Field vs Substrate eps_r")
    fig_sweep.savefig(f"{OUT_DIR}/07_parametric_sweep.png", dpi=150)
    plt.close(fig_sweep)
    print(f"  Saved: {OUT_DIR}/07_parametric_sweep.png")

    # ==================================================================
    # 9. Field animation (GIF) — bonus
    # ==================================================================
    print("\n[Bonus] Saving field animation ...")
    if result.snapshots is not None and "ez" in result.snapshots:
        try:
            save_field_animation(
                result,
                f"{OUT_DIR}/07_field_animation.gif",
                component="ez",
                slice_axis="z",
                fps=10,
                interval=1,
            )
            print(f"  Saved: {OUT_DIR}/07_field_animation.gif")
        except Exception as e:
            print(f"  Animation skipped: {e}")
    else:
        print("  Animation skipped: no snapshots")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"Visualization Showcase Complete")
    print(f"{'='*60}")
    print(f"Files saved in {OUT_DIR}/:")
    print(f"  07_field_slices.png       - Field evolution at multiple timesteps")
    print(f"  07_s_params_db.png        - S-parameter magnitude (dB)")
    print(f"  07_s_params_phase.png     - S-parameter phase")
    print(f"  07_smith_chart.png        - Smith chart")
    print(f"  07_radiation_pattern.png  - Far-field polar radiation pattern")
    print(f"  07_convergence.png        - Mesh convergence study")
    print(f"  07_parametric_sweep.png   - Parametric sweep")
    print(f"  07_field_animation.gif    - Field animation")


if __name__ == "__main__":
    main()
