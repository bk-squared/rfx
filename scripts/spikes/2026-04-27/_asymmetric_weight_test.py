"""Test asymmetric template-norm vs sim-extract v_hi weights.

NOTE: depends on env-var-driven weight overrides AND aperture_dA_norm_np
template-norm split, both reverted at end of session. See
docs/research_notes/2026-04-27_mesh_conv_xfail_root_cause.md for the
reproduction recipe.

Hypothesis: drop v_hi cell on EXTRACT (removes standing-wave
contamination → PEC-short closes) but keep it with weight=1.0 on
TEMPLATE NORMALISATION (proper 1/N_v normalisation → mesh-conv
recovers).

Tests:
  Config A: NORM_W=1.0, EXTRACT_W=0.0   (asymmetric — predicted to win)
  Config B: NORM_W=0.0, EXTRACT_W=1.0   (inverse — sanity check)
  Config C: NORM_W=0.0, EXTRACT_W=0.0   (current default)
  Config D: NORM_W=1.0, EXTRACT_W=1.0   (KEEP both)
"""
from __future__ import annotations

import os
import sys
import subprocess

CONFIGS = [
    # (norm_w, extract_w, label)
    (1.0, 0.0, "ASYM: norm=1.0, extract=0.0 (KEEP template, DROP sim)"),
    (0.0, 1.0, "INVERSE: norm=0.0, extract=1.0"),
    (0.0, 0.0, "DROP both (current canonical)"),
    (1.0, 1.0, "KEEP both"),
    (0.5, 0.0, "norm=0.5, extract=0.0"),
    (1.0, 0.5, "norm=1.0, extract=0.5"),
]


WORKER = '''
import numpy as np, sys
sys.path.insert(0, "tests")
import test_waveguide_port_validation_battery as mod

# PEC-short
freqs = np.linspace(5.0e9, 7.0e9, 6)
sim = mod._build_sim(freqs, pec_short_x=0.085, waveform="modulated_gaussian")
s, _, port_idx = mod._s_matrix(sim, num_periods=40, normalize=False)
s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
print(f"PEC_S11_MIN {float(s11.min()):.4f}")
print(f"PEC_S11_MAX {float(s11.max()):.4f}")
print(f"PEC_PER {' '.join(f'{x:.4f}' for x in s11)}")

# Mesh-conv
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
    for norm_w, extract_w, label in CONFIGS:
        env = os.environ.copy()
        env["RFX_V_FACE_WEIGHT"] = str(extract_w)
        env["RFX_V_FACE_NORM_WEIGHT"] = str(norm_w)
        # u-axis stays at default DROP (mode vanishes there structurally).
        env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")
        print(f"\n=== {label} (norm={norm_w}, extract={extract_w}) ===", flush=True)
        proc = subprocess.run(
            [sys.executable, "-c", WORKER],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-1500:])
            continue
        record = {"label": label, "norm_w": norm_w, "extract_w": extract_w}
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
    print(f"{'norm':<5} {'extr':<5} {'min|S11|':<10} {'coarse_d':<10} {'fine_d':<10} mesh_pass S21_per_dx label")
    for r in rows:
        print(f"{r['norm_w']:<5.2f} {r['extract_w']:<5.2f} "
              f"{r.get('pec_min', 0):<10.4f} "
              f"{r.get('coarse', 0):<10.4f} "
              f"{r.get('fine', 0):<10.4f} "
              f"{r.get('mesh_pass', 0):<9} {r.get('s21', ''):<25} {r['label']}")


if __name__ == "__main__":
    main()
