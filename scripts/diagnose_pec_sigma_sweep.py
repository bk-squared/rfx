"""Sweep σ in PEC material from 1e7..1e14 to test if 8% loss scales with σ.

Threshold for pec_mask is σ ≥ 1e6 (in rasterize.py). All values tested are
above threshold so mask is populated. As σ → ∞, intermediate E (= cb·curl_H
with cb ≈ 2/σ) shrinks → dissipation drops. As σ ↓ to 1e7, intermediate E
grows but mask still zeros it.

Predicted (if loss is from σ-dissipation):
  - σ=1e7:  high loss → low reflection
  - σ=1e10: medium → 0.914
  - σ=1e14: very low loss → reflection → 1.000
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Run as subprocesses so each sigma gets a fresh JIT cache and fresh import.
RUNNER = """
from __future__ import annotations
import sys, os
from pathlib import Path
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

C0 = 299_792_458.0
MU_0 = 4.0 * np.pi * 1e-7

import rfx.api as _api
sigma_val = float(os.environ['PEC_SIGMA'])
_api.MATERIAL_LIBRARY['pec'] = {'eps_r': 1.0, 'sigma': sigma_val}

from rfx.sources import waveguide_port as wg
_CAPS = []
_orig = wg.extract_waveguide_port_waves
def _capture(cfg, *, ref_shift=0.0):
    a, b = _orig(cfg, ref_shift=ref_shift)
    _CAPS.append({
        'v_probe_t': np.asarray(cfg.v_probe_t).copy(),
        'i_probe_t': np.asarray(cfg.i_probe_t).copy(),
        'n': int(cfg.n_steps_recorded), 'dt': float(cfg.dt),
        'freqs': np.asarray(cfg.freqs).copy(),
        'f_cutoff': float(cfg.f_cutoff),
    })
    return a, b
wg.extract_waveguide_port_waves = _capture
_api.extract_waveguide_port_waves = _capture

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())

sim = Simulation(
    freq_max=float(FREQS_HZ[-1]) * 1.1,
    domain=(0.200, A_WG, B_WG),
    boundary=BoundarySpec(
        x=Boundary(lo='cpml', hi='cpml'),
        y=Boundary(lo='pec', hi='pec'),
        z=Boundary(lo='pec', hi='pec'),
    ),
    cpml_layers=20, dx=DX_M,
)
sim.add(Box((0.155, 0.0, 0.0), (0.155 + 2 * DX_M, A_WG, B_WG)),
        material='pec')
sim.add_waveguide_port(
    0.040, direction='+x', mode=(1, 0), mode_type='TE',
    freqs=jnp.asarray(FREQS_HZ), f0=F0, bandwidth=0.6,
    waveform='modulated_gaussian', reference_plane=0.050, name='left',
    probe_offset=114,
)

result = sim.run(num_periods=200, compute_s_params=False)
wp = result.waveguide_ports
cfg = wp['left'] if isinstance(wp, dict) else wp[0]
_ = wg.extract_waveguide_port_waves(cfg, ref_shift=0.0)
cap = _CAPS[-1]

n = cap['n']; dt = cap['dt']; fc = cap['f_cutoff']
freqs = cap['freqs']
omega = 2 * np.pi * freqs.astype(np.float64)
beta_arr = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
Z = omega * MU_0 / np.maximum(beta_arr, 1e-30)
n_idx = np.arange(n)
phase = np.exp(-1j * omega[None, :] * n_idx[:, None] * dt)
Vd = 2.0 * dt * (cap['v_probe_t'][:n].astype(np.float64) @ phase)
Id_raw = 2.0 * dt * (cap['i_probe_t'][:n].astype(np.float64) @ phase)
Id = Id_raw * np.exp(+1j * omega * 0.5 * dt)
fwd = 0.5 * (Vd + Z * Id)
bwd = 0.5 * (Vd - Z * Id)
refl = np.abs(bwd) / np.maximum(np.abs(fwd), 1e-30)
print(f'SIGMA_RESULT sigma={sigma_val:.0e} mean_r={refl.mean():.4f} '
      f'min={refl.min():.4f} max={refl.max():.4f}')
"""

# Write runner to temporary file in scripts dir
runner_path = _ROOT / "scripts" / "_pec_sigma_runner_inner.py"
runner_path.write_text(RUNNER)

import os

results = []
for sigma in [1e7, 1e8, 1e9, 1e10, 1e12, 1e14, 1e16]:
    print(f"--- sigma = {sigma:.0e} ---", flush=True)
    env = os.environ.copy()
    env["PEC_SIGMA"] = str(sigma)
    proc = subprocess.run(
        ["python", str(runner_path)],
        env=env, capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if "SIGMA_RESULT" in line:
            print(f"  {line}")
            parts = dict(p.split("=") for p in line.split() if "=" in p)
            results.append((sigma, float(parts["mean_r"])))
    if proc.returncode != 0:
        print(f"  STDERR (last 5 lines):")
        for ln in proc.stderr.splitlines()[-5:]:
            print(f"    {ln}")

print("\n=== SUMMARY ===")
print(f"{'sigma':>10s} {'<|r|>':>8s}")
for s, r in results:
    print(f"{s:10.0e} {r:8.4f}")
print()
print("Predicted IF loss is from σ-dissipation at PEC cells:")
print("  - small σ (1e7) → high intermediate E → MORE loss → LOWER reflection")
print("  - large σ (1e16) → tiny intermediate E → ~ZERO loss → HIGHER reflection")
