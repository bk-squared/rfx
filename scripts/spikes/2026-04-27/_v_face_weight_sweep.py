"""Sweep v_hi PEC ghost-cell weight from 0.0..1.0.

NOTE: depends on env-var-driven weight overrides reverted at end of
session. See docs/research_notes/2026-04-27_mesh_conv_xfail_root_cause.md.

Result of the 2x2 ablation: DROP_U doesn't matter, only DROP_V matters.
Now sweep v_hi weight on a finer grid to see if there's a sweet spot
that satisfies both PEC-short |S11| ≥ 0.99 and mesh-conv pass.
"""
from __future__ import annotations

import os
import sys
import subprocess

WEIGHTS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


WORKER_BLOCK = '''
import os, sys, numpy as np
sys.path.insert(0, "tests")
import test_waveguide_port_validation_battery as mod

# --- PEC-short ---
freqs = np.linspace(5.0e9, 7.0e9, 6)
sim = mod._build_sim(freqs, pec_short_x=0.085, waveform="modulated_gaussian")
s, _, port_idx = mod._s_matrix(sim, num_periods=40, normalize=False)
s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
print(f"PEC_S11_PER_FREQ {' '.join(f'{x:.4f}' for x in s11)}")
print(f"PEC_S11_MIN {float(s11.min()):.4f}")

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
    for w in WEIGHTS:
        env = os.environ.copy()
        env["RFX_V_FACE_WEIGHT"] = str(w)
        env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")
        print(f"\n=== v_hi weight = {w:.2f} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, "-c", WORKER_BLOCK],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-1000:])
            continue
        record = {"w": w}
        for line in proc.stdout.splitlines():
            if line.startswith("PEC_S11_MIN "):
                record["pec_min"] = float(line.split()[1])
            elif line.startswith("MESH_DELTAS "):
                parts = line.split()
                record["coarse"] = float(parts[1].split("=")[1])
                record["fine"] = float(parts[2].split("=")[1])
            elif line.startswith("MESH_PASS "):
                record["mesh_pass"] = int(line.split()[1])
            elif line.startswith("MESH_S21 "):
                record["s21"] = " ".join(line.split()[1:])
        rows.append(record)

    print("\n\n=========== SUMMARY ===========")
    print(f"{'w':<6} {'min|S11|':<10} {'coarse_d':<10} {'fine_d':<10} {'fine-coarse':<12} mesh_pass S21_per_dx")
    for r in rows:
        d = r.get("fine", 0) - r.get("coarse", 0)
        print(f"{r['w']:<6.2f} {r.get('pec_min', 0):<10.4f} {r.get('coarse', 0):<10.4f} "
              f"{r.get('fine', 0):<10.4f} {d:<+12.4f} {r.get('mesh_pass', 0):<9} {r.get('s21', '')}")


if __name__ == "__main__":
    main()
