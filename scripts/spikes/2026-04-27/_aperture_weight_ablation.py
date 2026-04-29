"""Aperture +face PEC ghost-cell weight ablation.

NOTE: this spike depends on env-var-driven weight overrides that were
added to ``rfx/sources/waveguide_port.py`` during the 2026-04-27 session
and reverted at the end. To rerun, re-apply the env-var hooks (see
``docs/research_notes/2026-04-27_mesh_conv_xfail_root_cause.md`` for the
full reproduction recipe).

Background. The 2026-04-27 DROP-weight fix (0.5 → 0.0) closed
PEC-short |S11| ≥ 0.99 (Meep class) but regressed mesh-conv |S21|
on test_mesh_convergence_s21_scaled_cpml in the validation battery.

Hypothesis. The TE10 mode integrand sin²(π·u/a) structurally vanishes
at u=a but is uniform in v. Dropping the v_hi cell loses 1/N_v of
normalisation, with N_v varying with mesh. Dropping u_hi only loses
~ (π·dx/(2a))² ≈ 0.18%/0.06%/0.03% — negligible.

So: drop u_hi (correct, mode vanishes), keep v_hi (mode is uniform
there, dropping wastes 1/N_v of normalisation).

Test. 2x2 ablation over (DROP_U, DROP_V) with both gates:
- PEC-short min |S11| (target ≥ 0.99)
- Mesh-conv |S21| (coarse_delta vs fine_delta monotone)

Run from repo root:
    python scripts/spikes/2026-04-27/_aperture_weight_ablation.py
"""
from __future__ import annotations

import os
import sys
import subprocess

CONFIGS = [
    # (drop_u, drop_v, label)
    (1, 1, "DROP both (current)"),
    (1, 0, "DROP u, KEEP v (mode-aware: TE10)"),
    (0, 1, "KEEP u, DROP v"),
    (0, 0, "KEEP both"),
]


WORKER_BLOCK = '''
import os, sys, numpy as np, jax.numpy as jnp
sys.path.insert(0, "tests")
import test_waveguide_port_validation_battery as mod

# --- PEC-short ---
freqs = np.linspace(5.0e9, 7.0e9, 6)
sim = mod._build_sim(freqs, pec_short_x=0.085, waveform="modulated_gaussian")
s, _, port_idx = mod._s_matrix(sim, num_periods=40, normalize=False)
s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
print(f"PEC_S11_PER_FREQ {' '.join(f'{x:.4f}' for x in s11)}")
print(f"PEC_S11_MIN {float(s11.min()):.4f}")
print(f"PEC_S11_MAX {float(s11.max()):.4f}")

# --- Mesh-conv ---
freq = 6.0e9
obstacles = [((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)]
TARGET_CPML_M = 0.030
resolutions = [0.003, 0.002, 0.0015]
s21_values = []
for dx in resolutions:
    layers = max(8, int(round(TARGET_CPML_M / dx)))
    sim = mod._build_sim([freq], dx=dx, cpml_layers=layers,
                         obstacles=obstacles, waveform="modulated_gaussian")
    s, _, port_idx = mod._s_matrix(sim, num_periods=40, normalize=True)
    s21 = float(np.abs(s[port_idx["right"], port_idx["left"], 0]))
    s21_values.append(s21)
print(f"MESH_S21 {' '.join(f'{x:.4f}' for x in s21_values)}")
coarse_delta = abs(s21_values[0] - s21_values[1])
fine_delta = abs(s21_values[1] - s21_values[2])
print(f"MESH_DELTAS coarse={coarse_delta:.4f} fine={fine_delta:.4f}")
print(f"MESH_PASS {1 if fine_delta <= coarse_delta + 0.005 else 0}")
'''


def main():
    rows = []
    for drop_u, drop_v, label in CONFIGS:
        env = os.environ.copy()
        env["RFX_DROP_U_FACE"] = str(drop_u)
        env["RFX_DROP_V_FACE"] = str(drop_v)
        env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")
        print(f"\n=== {label}  (DROP_U={drop_u}, DROP_V={drop_v}) ===", flush=True)
        proc = subprocess.run(
            [sys.executable, "-c", WORKER_BLOCK],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        out = proc.stdout
        err = proc.stderr
        print(out)
        if proc.returncode != 0:
            print("STDERR:", err[-2000:])
            continue
        record = {"label": label, "drop_u": drop_u, "drop_v": drop_v}
        for line in out.splitlines():
            if line.startswith("PEC_S11_MIN "):
                record["pec_min"] = float(line.split()[1])
            elif line.startswith("PEC_S11_MAX "):
                record["pec_max"] = float(line.split()[1])
            elif line.startswith("MESH_DELTAS "):
                parts = line.split()
                record["coarse"] = float(parts[1].split("=")[1])
                record["fine"] = float(parts[2].split("=")[1])
            elif line.startswith("MESH_PASS "):
                record["mesh_pass"] = int(line.split()[1])
        rows.append(record)

    print("\n\n=========== SUMMARY ===========")
    print(f"{'label':<35} {'min|S11|':<10} {'max|S11|':<10} {'coarse_d':<10} {'fine_d':<10} mesh_pass")
    for r in rows:
        print(f"{r['label']:<35} {r.get('pec_min', 0):<10.4f} {r.get('pec_max', 0):<10.4f} "
              f"{r.get('coarse', 0):<10.4f} {r.get('fine', 0):<10.4f} {r.get('mesh_pass', 0)}")


if __name__ == "__main__":
    main()
