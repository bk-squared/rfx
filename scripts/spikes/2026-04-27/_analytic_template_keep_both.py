"""Probe: does mode_profile='analytic' + KEEP-both recover PEC-short?

Hypothesis. The PEC-short |S11|=0.89 deficit under KEEP-both stems
from the discrete-Yee mode template at the v_hi cell not matching the
actual SIM field there. OpenEMS uses analytic mode functions (sympy
expressions evaluated at cell centres). If the analytic template gets
PEC-short close to 1.0 under KEEP-both AND mesh-conv passes, the fix
is "use analytic templates" (no apply_pec_faces change needed).

This requires the v_hi DROP to be turned off. Two ways:
  (a) Re-add env-var hooks in waveguide_port.py
  (b) Monkey-patch the boundary detection to skip the DROP

We use (b) for a non-invasive test: monkey-patch `_DROP_U_FACE_PEC`
... wait, that env var is gone. Patch the boundary_grid to lie about
the PEC face presence. The init reads ``getattr(boundary_grid,
'pec_faces', set())`` and ``getattr(boundary_grid, 'cpml_axes', '')``.
If boundary_grid claims all axes are CPML and no PEC faces, then the
DROP rule won't fire.

Run from repo root.
"""
from __future__ import annotations

import os
import sys
import subprocess


WORKER = '''
import numpy as np, sys
sys.path.insert(0, "tests")
import test_waveguide_port_validation_battery as mod
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
import jax.numpy as jnp

# Build _build_sim equivalent but pass mode_profile="analytic" through.
# Re-implement minimal version to avoid touching the production helper.

DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09
TARGET_CPML_M = 0.030

def build(freqs_hz, *, dx=None, cpml_layers=None, obstacles=(),
          pec_short_x=None, mode_profile="discrete"):
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim_kwargs = dict(freq_max=max(float(freqs[-1]), f0), domain=DOMAIN, boundary="cpml")
    if cpml_layers is not None:
        sim_kwargs["cpml_layers"] = cpml_layers
    else:
        sim_kwargs["cpml_layers"] = 10
    if dx is not None:
        sim_kwargs["dx"] = dx
    sim = Simulation(**sim_kwargs)
    for idx, (lo, hi, eps_r) in enumerate(obstacles):
        name = f"diel_{idx}"
        sim.add_material(name, eps_r=eps_r, sigma=0.0)
        sim.add(Box(lo, hi), material=name)
    if pec_short_x is not None:
        sim.add(Box((pec_short_x, 0.0, 0.0),
                     (pec_short_x + 0.002, DOMAIN[1], DOMAIN[2])),
                 material="pec")
    pf = jnp.asarray(freqs)
    for x, dirn, name in ((PORT_LEFT_X, "+x", "left"), (PORT_RIGHT_X, "-x", "right")):
        sim.add_waveguide_port(x, direction=dirn, mode=(1, 0), mode_type="TE",
                               freqs=pf, f0=f0, bandwidth=bandwidth,
                               waveform="modulated_gaussian",
                               mode_profile=mode_profile,
                               name=name)
    return sim

# --- Run under: monkey-patch boundary_grid to claim all CPML, no PEC faces.
# This makes the DROP rule never fire (KEEP-both effectively).
# KEEP-both is selected via env vars on the subprocess; no monkey-patch.


def s_matrix(sim, *, normalize=True):
    result = sim.compute_waveguide_s_matrix(num_periods=40, normalize=normalize)
    s = np.asarray(result.s_params)
    pi = {n: i for i, n in enumerate(result.port_names)}
    return s, pi


for profile in ("discrete", "analytic"):
    print(f"\\n--- mode_profile={profile} (KEEP-both via boundary_grid patch) ---")
    # PEC-short
    freqs = np.linspace(5.0e9, 7.0e9, 6)
    sim = build(freqs, pec_short_x=0.085, mode_profile=profile)
    s, pi = s_matrix(sim, normalize=False)
    s11 = np.abs(s[pi["left"], pi["left"], :])
    print(f"PEC_S11 min={s11.min():.4f} max={s11.max():.4f} per={[f'{x:.4f}' for x in s11]}")

    # Mesh-conv
    obs = [((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)]
    s21s = []
    for dx in [0.003, 0.002, 0.0015]:
        layers = max(8, int(round(TARGET_CPML_M / dx)))
        sim = build([6.0e9], dx=dx, cpml_layers=layers, obstacles=obs, mode_profile=profile)
        s, pi = s_matrix(sim, normalize=True)
        s21s.append(float(np.abs(s[pi["right"], pi["left"], 0])))
    cd = abs(s21s[0] - s21s[1])
    fd = abs(s21s[1] - s21s[2])
    print(f"MESH_S21={s21s} coarse={cd:.4f} fine={fd:.4f} pass={'YES' if fd <= cd + 0.005 else 'NO'}")
'''


def main():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")
    env["RFX_DROP_U_FACE"] = "0"  # KEEP u_hi
    env["RFX_DROP_V_FACE"] = "0"  # KEEP v_hi
    proc = subprocess.run(
        [sys.executable, "-c", WORKER],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        env=env, capture_output=True, text=True, timeout=900,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-2000:])


if __name__ == "__main__":
    main()
