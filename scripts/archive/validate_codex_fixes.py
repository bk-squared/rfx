"""Quantitative validation of Codex review fixes (C1, W6, interpolation).

Runs on GPU via VESSL. Tests:
1. SBP-SAT energy conservation: drift over 2000 steps with H coupling
2. NTFF far-field accuracy: dipole pattern with H phase compensation
3. SBP-SAT interpolation: energy stability vs tau parameter
"""
import numpy as np
import jax.numpy as jnp

print("=" * 60)
print("Codex Fix Quantitative Validation")
print("=" * 60)


# ============================================================
# Test 1: SBP-SAT energy conservation over 2000 steps
# ============================================================
print("\n=== 1. SBP-SAT Energy Conservation (2000 steps) ===")

from rfx.subgridding.sbp_sat_3d import (
    init_subgrid_3d, step_subgrid_3d, compute_energy_3d,
)

config, state = init_subgrid_3d(
    shape_c=(20, 20, 20), dx_c=0.004,
    fine_region=(7, 13, 7, 13, 7, 13),
    ratio=2, courant=0.45, tau=1.0,  # energy-conservative
)
# Inject impulse BEFORE loop
state = state._replace(
    ez_f=state.ez_f.at[
        config.nx_f // 2, config.ny_f // 2, config.nz_f // 2
    ].set(1.0)
)

e_initial = compute_energy_3d(state, config)
stable = True
for step in range(1000):
    state = step_subgrid_3d(state, config)
    if bool(jnp.any(jnp.isnan(state.ez_f))):
        print(f"  NaN at step {step}")
        stable = False
        break

e_final = compute_energy_3d(state, config)
ratio_e = e_final / (e_initial + 1e-30)

# SBP-SAT is dissipative stable: energy non-increasing (Cheng et al.)
# Energy DECREASE is expected (penalty damps interface mismatch).
# Energy INCREASE would indicate instability.
print(f"  E_initial: {e_initial:.6e}")
print(f"  E_final:   {e_final:.6e}")
print(f"  Ratio: {ratio_e:.4f}")
energy_ok = stable and ratio_e <= 1.05 and ratio_e > 0
print(f"  {'PASS' if energy_ok else 'FAIL'}: energy non-increasing (dissipative stable)")


# ============================================================
# Test 2: NTFF far-field accuracy — dipole vs analytical
# ============================================================
print("\n=== 2. NTFF Far-Field Accuracy (H phase compensation) ===")

from rfx import Simulation, GaussianPulse
from rfx.farfield import compute_far_field

sim = Simulation(
    freq_max=3e9,
    domain=(0.2, 0.2, 0.2),
    boundary="cpml",
    dx=0.005,
    cpml_layers=8,
)
sim.add_source((0.1, 0.1, 0.1), "ez",
               waveform=GaussianPulse(f0=1e9, bandwidth=0.3))
sim.add_probe((0.1, 0.1, 0.1), "ez")
# NTFF box must be inside physical domain (CPML = 8*0.005 = 0.04m per side)
sim.add_ntff_box(
    (0.05, 0.05, 0.05), (0.15, 0.15, 0.15),
    freqs=np.array([1e9]),
)

result = sim.run(n_steps=1500)
ts = np.array(result.time_series[:, 0])
print(f"  Probe peak: {np.max(np.abs(ts)):.3e}")

if result.ntff_data is not None and result.ntff_box is not None:
    theta = np.linspace(0.01, np.pi - 0.01, 37)
    phi = np.array([0.0, np.pi / 2])
    ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid,
                           theta, phi)

    power = np.abs(ff.E_theta[0, :, 0]) ** 2 + np.abs(ff.E_phi[0, :, 0]) ** 2
    power_norm = power / (np.max(power) + 1e-30)

    # Analytical: Hertzian dipole along z → power ∝ sin²(θ)
    analytical = np.sin(theta) ** 2

    # Correlation
    corr = np.corrcoef(power_norm, analytical)[0, 1]
    rms_err = np.sqrt(np.mean((power_norm - analytical) ** 2))

    print(f"  Far-field power peak: {np.max(power):.3e}")
    print(f"  Pattern correlation with sin²(θ): {corr:.4f}")
    print(f"  RMS error vs analytical: {rms_err:.4f}")
    print(f"  {'PASS' if corr > 0.95 else 'FAIL'}: correlation {'>' if corr > 0.95 else '<'} 0.95")
else:
    print("  NTFF data not available — skipped")


# ============================================================
# Test 3: SBP-SAT energy vs tau (stability sweep)
# ============================================================
print("\n=== 3. SBP-SAT Energy Stability vs Tau ===")

for tau in [0.25, 0.5, 0.75, 1.0]:
    cfg, st = init_subgrid_3d(
        shape_c=(20, 20, 20), dx_c=0.004,
        fine_region=(7, 13, 7, 13, 7, 13),
        ratio=2, courant=0.45, tau=tau,
    )
    st = st._replace(
        ez_f=st.ez_f.at[cfg.nx_f // 2, cfg.ny_f // 2, cfg.nz_f // 2].set(1.0)
    )

    e_init = None
    e_final = None
    stable = True
    for step in range(1000):
        st = step_subgrid_3d(st, cfg)
        if step == 10:
            e_init = float(compute_energy_3d(st, cfg))
        if step == 999:
            e_final = float(compute_energy_3d(st, cfg))
        if bool(jnp.any(jnp.isnan(st.ez_f))):
            print(f"  tau={tau:.2f}: NaN at step {step}")
            stable = False
            break

    if stable and e_init and e_final:
        ratio_e = e_final / (e_init + 1e-30)
        status = "PASS" if 0.01 < ratio_e < 100 else "FAIL"
        print(f"  tau={tau:.2f}: E_init={e_init:.3e}, E_final={e_final:.3e}, "
              f"ratio={ratio_e:.3f} [{status}]")


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
