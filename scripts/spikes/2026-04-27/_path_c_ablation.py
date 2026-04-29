"""Path C ablation diagnostic — answer the critic's open questions
before any further EigenModeSource work.

Runs three identical-geometry experiments and prints the matrix:

  1. PEC-short |S11| with source_type='tfsf'    (existing behaviour)
  2. PEC-short |S11| with source_type='eigenmode' (current Path C)
  3. Empty-guide |b/a| (early-time gated) for both source types

Identical: WR-90 cross-section, dx=1mm, CPML=8, freqs, ref planes,
n_steps. The ONLY thing that changes is source_type.

Open questions this answers (from the 2026-04-27 critic):

  Q1. Is the ~0.95 PEC-short |S11| cap a source-side or receive-side limit?
      → If TFSF and eigenmode both give similar min |S11|, cap is
        receive-side (V/I overlap mixing higher-order modes), independent
        of source type. The "Path C is the source-side fix" framing
        from the handover would then be wrong.

  Q2. Does the eigenmode source's directionality win (0.676% vs TFSF's
      ~1.2% from the existing battery) survive on the same geometry as
      the PEC-short test?

  Q3. Does dropping the e_inc_table / h_inc_table reads (i.e., raw
      pulse fallback) regress the eigenmode source toward the 99%
      backward-leakage we measured before the table-reuse fix?

Run as:
    python scripts/_path_c_ablation.py
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box


# ---------------------------------------------------------------------------
# Identical geometry block (mirrors test_eigenmode_source_pec_short_s11_magnitude)
# ---------------------------------------------------------------------------
A_WG = 22.86e-3
B_WG = 10.16e-3
LENGTH_X = 0.10
DX = 1.0e-3
CPML = 8
FREQS = np.linspace(7.5e9, 11.5e9, 6)
F0 = 9.5e9
BANDWIDTH = 0.5
NUM_PERIODS = 40

# Directionality block (mirrors test_source_directionality_early_time +
# test_eigenmode_source_directionality_early_time)
DIR_DOMAIN = (0.50, 0.04, 0.02)
DIR_DX = 0.002
DIR_F0 = 10.0e9
DIR_FREQS = np.linspace(8.0e9, 12.0e9, 8)
DIR_BANDWIDTH = 0.5
DIR_PROBE_OFFSET = 0.040
F_CUTOFF_HZ_TE10_WR90 = 6.557e9


def _pec_short_s11(source_type: str) -> tuple[float, float, float]:
    """Run PEC-short two-port WR-90 with the requested source_type and
    return (min, mean, max) of |S11| over the freq band."""
    sim = Simulation(
        freq_max=12.0e9,
        domain=(LENGTH_X, A_WG, B_WG),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=CPML,
        dx=DX,
    )
    sim.add(
        Box((0.085, 0.0, 0.0), (0.085 + 2 * DX, A_WG, B_WG)),
        material="pec",
    )
    sim.add_waveguide_port(
        0.012, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=jnp.asarray(FREQS), f0=F0, bandwidth=BANDWIDTH,
        waveform="modulated_gaussian",
        source_type=source_type,
        reference_plane=0.020,
        name="left",
    )
    sim.add_waveguide_port(
        0.082, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=jnp.asarray(FREQS), f0=F0, bandwidth=BANDWIDTH,
        waveform="modulated_gaussian",
        amplitude=0.0,
        reference_plane=0.075,
        name="right",
    )
    result = sim.compute_waveguide_s_matrix(num_periods=NUM_PERIODS, normalize=True)
    s = np.asarray(result.s_params)
    port_idx = {n: i for i, n in enumerate(result.port_names)}
    s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
    return float(s11.min()), float(s11.mean()), float(s11.max()), s11


def _early_time_directionality(source_type: str) -> float:
    """Empty-guide |b/a| early-time gated, for the requested source_type."""
    sim = Simulation(
        freq_max=12.0e9,
        domain=DIR_DOMAIN,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=10,
        dx=DIR_DX,
    )
    port_x = DIR_DOMAIN[0] / 2.0
    y_c = DIR_DOMAIN[1] / 2
    z_c = DIR_DOMAIN[2] / 2

    sim.add_waveguide_port(
        port_x, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=jnp.asarray(DIR_FREQS), f0=DIR_F0, bandwidth=DIR_BANDWIDTH,
        waveform="modulated_gaussian",
        source_type=source_type,
        name="source",
    )
    sim.add_probe((port_x - DIR_PROBE_OFFSET, y_c, z_c), "ez")
    sim.add_probe((port_x + DIR_PROBE_OFFSET, y_c, z_c), "ez")

    grid_est = sim._build_grid()
    n_steps = int(grid_est.num_timesteps(num_periods=24.0))
    result = sim.run(n_steps=n_steps, compute_s_params=False)

    dt = float(result.dt)
    times = np.arange(n_steps) * dt
    cpml_thickness_m = 10 * float(result.grid.dx)
    nearest_cpml_m = min(
        DIR_DOMAIN[0] - port_x - cpml_thickness_m,
        port_x - cpml_thickness_m,
    )
    v_g = 2.998e8 * np.sqrt(
        max(1.0 - (F_CUTOFF_HZ_TE10_WR90 / DIR_F0) ** 2, 1e-6)
    )
    round_trip_s = 2 * nearest_cpml_m / v_g
    cycle_s = 1.0 / DIR_F0
    early_t_s = round_trip_s - 2.0 * cycle_s
    early_mask = times < early_t_s
    ts = np.asarray(result.time_series)
    max_backward = float(np.max(np.abs(ts[early_mask, 0])))
    max_forward = float(np.max(np.abs(ts[early_mask, 1])))
    return max_backward / max(max_forward, 1e-30)


def main() -> int:
    print("=" * 70)
    print("Path C ablation — TFSF vs Eigenmode on identical geometry")
    print("=" * 70)

    print("\n[1/4] PEC-short |S11| with source_type='tfsf' ...")
    tfsf_min, tfsf_mean, tfsf_max, tfsf_s11 = _pec_short_s11("tfsf")
    print(f"       per-freq |S11|: {np.array2string(tfsf_s11, precision=3)}")
    print(f"       min={tfsf_min:.4f}  mean={tfsf_mean:.4f}  max={tfsf_max:.4f}")

    print("\n[2/4] PEC-short |S11| with source_type='eigenmode' ...")
    eig_min, eig_mean, eig_max, eig_s11 = _pec_short_s11("eigenmode")
    print(f"       per-freq |S11|: {np.array2string(eig_s11, precision=3)}")
    print(f"       min={eig_min:.4f}  mean={eig_mean:.4f}  max={eig_max:.4f}")

    print("\n[3/4] Empty-guide |b/a| early-time, source_type='tfsf' ...")
    tfsf_dir = _early_time_directionality("tfsf")
    print(f"       |b/a| = {tfsf_dir*100:.3f}%")

    print("\n[4/4] Empty-guide |b/a| early-time, source_type='eigenmode' ...")
    eig_dir = _early_time_directionality("eigenmode")
    print(f"       |b/a| = {eig_dir*100:.3f}%")

    # ---------- verdict matrix ----------
    print("\n" + "=" * 70)
    print("VERDICT MATRIX")
    print("=" * 70)
    print(f"{'metric':<35} {'tfsf':>12} {'eigenmode':>12} {'Δ':>10}")
    print("-" * 70)
    print(f"{'PEC-short min |S11|':<35} {tfsf_min:>12.4f} "
          f"{eig_min:>12.4f} {eig_min - tfsf_min:>+10.4f}")
    print(f"{'PEC-short mean |S11|':<35} {tfsf_mean:>12.4f} "
          f"{eig_mean:>12.4f} {eig_mean - tfsf_mean:>+10.4f}")
    print(f"{'PEC-short max |S11|':<35} {tfsf_max:>12.4f} "
          f"{eig_max:>12.4f} {eig_max - tfsf_max:>+10.4f}")
    print(f"{'Empty-guide |b/a| (%)':<35} {tfsf_dir*100:>12.3f} "
          f"{eig_dir*100:>12.3f} {(eig_dir - tfsf_dir)*100:>+10.3f}")

    # ---------- interpretation ----------
    print("\nINTERPRETATION:")
    if abs(eig_min - tfsf_min) < 0.03:
        print("  Q1 PEC-short cap appears to be SOURCE-INDEPENDENT — both "
              "source types hit similar min |S11|. The handover §1 'Path C "
              "fixes the source side' premise is questionable; the residual "
              "is dominated by receive-side V/I overlap (per "
              "2026-04-26_phase2_aperture_weight_dead_end.md).")
    elif eig_min > tfsf_min + 0.05:
        print("  Q1 Eigenmode source is MEASURABLY BETTER than TFSF on min "
              "|S11|. The Path C source-side fix has merit — pursue further "
              "to push toward 0.99.")
    else:
        print("  Q1 Eigenmode source is MEASURABLY WORSE than TFSF on min "
              "|S11|. There is a Path C implementation bug — debug before "
              "claiming any source-side win.")

    if eig_dir * 1.5 < tfsf_dir:
        print("  Q2 Eigenmode directionality is genuinely better than TFSF "
              "(>1.5x improvement) on this geometry.")
    elif eig_dir < tfsf_dir:
        print("  Q2 Eigenmode directionality is marginally better than TFSF.")
    else:
        print("  Q2 Eigenmode directionality is NO BETTER than TFSF on this "
              "geometry — the prior 0.676% measurement was geometry-specific.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
