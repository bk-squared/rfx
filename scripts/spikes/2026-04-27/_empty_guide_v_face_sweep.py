"""Probe: is the v_hi anomaly geometry-specific (PEC-short only) or always present?

NOTE: depends on env-var-driven weight overrides reverted at end of
session. See docs/research_notes/2026-04-27_mesh_conv_xfail_root_cause.md.

Run empty-guide |S11| under v_hi weight = {0.0, 1.0}. If the anomaly is
always present in the SIM field, both should be affected. If it's
specific to strong-reflector geometry, only PEC-short cares.
"""
from __future__ import annotations
import os
import sys
import subprocess


WORKER = '''
import numpy as np, sys
sys.path.insert(0, "tests")
import test_waveguide_port_validation_battery as mod

# Empty guide, normalize=True
freqs = np.linspace(4.5e9, 8.0e9, 10)
sim = mod._build_sim(freqs, waveform="modulated_gaussian")
s, _, port_idx = mod._s_matrix(sim, num_periods=40, normalize=True)
s11_norm = np.abs(s[port_idx["left"], port_idx["left"], :])
print(f"EMPTY_NORM_TRUE_MAX {float(s11_norm.max()):.4f}")
print(f"EMPTY_NORM_TRUE_PER_FREQ {' '.join(f'{x:.4f}' for x in s11_norm)}")

# Empty guide, normalize=False
sim2 = mod._build_sim(freqs, waveform="modulated_gaussian")
s2, _, port_idx = mod._s_matrix(sim2, num_periods=40, normalize=False)
s11_raw = np.abs(s2[port_idx["left"], port_idx["left"], :])
print(f"EMPTY_NORM_FALSE_MAX {float(s11_raw.max()):.4f}")
print(f"EMPTY_NORM_FALSE_PER_FREQ {' '.join(f'{x:.4f}' for x in s11_raw)}")
'''


def main():
    for w in [0.0, 1.0]:
        env = os.environ.copy()
        env["RFX_V_FACE_WEIGHT"] = str(w)
        env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")
        print(f"\n=== v_hi weight = {w:.2f} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, "-c", WORKER],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-1000:])


if __name__ == "__main__":
    main()
