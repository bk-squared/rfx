"""Replace apply_pec_mask with face-style explicit-set zeroing at a plane.

Mimics apply_pec_faces({x_hi}) semantics inside the mask path:
  Ey[plane_idx, :, :] = 0
  Ez[plane_idx, :, :] = 0

For a 1-cell wall at i=175 (Setup A's grid, Box at 0.155-0.156, mask cell 175):
this zeros Ey[175, ALL j, ALL k] and Ez[175, ALL j, ALL k] — including corner
cells at j=Ny-1, k=Nz-1 that the mask formula EXCLUDES (because they have
no PEC neighbor in y or z).

If r → 1.000 → the bug is the mask formula skipping corner cells.
If r → 0.914 → some other issue.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

C0 = 299_792_458.0
MU_0 = 4.0 * np.pi * 1e-7

import rfx.boundaries.pec as pec_mod  # noqa: E402

PLANE_IDX = 175  # = Box at phys 155 mm with CPML 20 layers


def _pec_mask_plane_face_style(state, pec_mask):
    """Zero Ey, Ez at PLANE_IDX (all j, k). Ignores pec_mask shape."""
    return state._replace(
        ey=state.ey.at[PLANE_IDX, :, :].set(0.0),
        ez=state.ez.at[PLANE_IDX, :, :].set(0.0),
    )


pec_mod.apply_pec_mask = _pec_mask_plane_face_style

from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS = []
_orig = wg.extract_waveguide_port_waves


def _capture(cfg, *, ref_shift=0.0):
    a, b = _orig(cfg, ref_shift=ref_shift)
    _CAPS.append({
        "v_probe_t": np.asarray(cfg.v_probe_t).copy(),
        "i_probe_t": np.asarray(cfg.i_probe_t).copy(),
        "n": int(cfg.n_steps_recorded), "dt": float(cfg.dt),
        "freqs": np.asarray(cfg.freqs).copy(),
        "f_cutoff": float(cfg.f_cutoff),
    })
    return a, b


wg.extract_waveguide_port_waves = _capture
import rfx.api as _api  # noqa: E402
_api.extract_waveguide_port_waves = _capture

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())

sim = Simulation(
    freq_max=float(FREQS_HZ[-1]) * 1.1,
    domain=(0.200, A_WG, B_WG),
    boundary=BoundarySpec(
        x=Boundary(lo="cpml", hi="cpml"),
        y=Boundary(lo="pec", hi="pec"),
        z=Boundary(lo="pec", hi="pec"),
    ),
    cpml_layers=20, dx=DX_M,
)
# Box at 0.155 to 0.156 — 1-cell-wide PEC at cell 175. Just to populate
# pec_mask so apply_pec_mask is invoked.
sim.add(Box((0.155, 0.0, 0.0), (0.156, A_WG, B_WG)),
        material="pec")
sim.add_waveguide_port(
    0.040, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=jnp.asarray(FREQS_HZ), f0=F0, bandwidth=0.6,
    waveform="modulated_gaussian", reference_plane=0.050, name="left",
    probe_offset=114,
)

result = sim.run(num_periods=200, compute_s_params=False)
wp = result.waveguide_ports
cfg = wp["left"] if isinstance(wp, dict) else wp[0]
_ = wg.extract_waveguide_port_waves(cfg, ref_shift=0.0)
cap = _CAPS[-1]

n = cap["n"]; dt = cap["dt"]; fc = cap["f_cutoff"]
freqs = cap["freqs"]
omega = 2 * np.pi * freqs.astype(np.float64)
beta = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
Z = omega * MU_0 / np.maximum(beta, 1e-30)
n_idx = np.arange(n)
phase = np.exp(-1j * omega[None, :] * n_idx[:, None] * dt)
Vd = 2.0 * dt * (cap["v_probe_t"][:n].astype(np.float64) @ phase)
Id_raw = 2.0 * dt * (cap["i_probe_t"][:n].astype(np.float64) @ phase)
Id = Id_raw * np.exp(+1j * omega * 0.5 * dt)
fwd = 0.5 * (Vd + Z * Id)
bwd = 0.5 * (Vd - Z * Id)
refl = np.abs(bwd) / np.maximum(np.abs(fwd), 1e-30)
arg = np.degrees(np.angle(Vd / (Z * Id)))

print(f"\nPatch: replace apply_pec_mask with face-style 'ey/ez[{PLANE_IDX}, :, :] = 0'")
print(f"|r| mean = {refl.mean():.4f}  range [{refl.min():.4f}, {refl.max():.4f}]")
print(f"arg(V/ZI) at f=10.3 GHz = {arg[10]:.3f}°")
print(f"\nReference:")
print(f"  Setup A (apply_pec_mask formula):   r=0.9141, arg=75.7°")
print(f"  Setup B.2 (apply_pec_faces x_hi):   r=1.0000, arg=90.0°")
print(f"  Setup E (apply_pec_faces, no Box):  r=1.0000, arg=90.0°")
