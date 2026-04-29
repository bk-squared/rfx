"""One-shot runner for a single apply_pec_mask variant. Invoked as a subprocess
to ensure a fresh JIT cache. Reads variant name from sys.argv[1].
"""
from __future__ import annotations
import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

C0 = 299_792_458.0
MU_0 = 4.0 * np.pi * 1e-7

import rfx.boundaries.pec as pec_mod  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "V0"


def _pec_mask_no_ex(state, pec_mask):
    mask_ey = pec_mask & (jnp.roll(pec_mask, 1, axis=1) | jnp.roll(pec_mask, -1, axis=1))
    mask_ez = pec_mask & (jnp.roll(pec_mask, 1, axis=2) | jnp.roll(pec_mask, -1, axis=2))
    return state._replace(
        ey=state.ey * (1.0 - mask_ey.astype(state.ey.dtype)),
        ez=state.ez * (1.0 - mask_ez.astype(state.ez.dtype)),
    )


def _pec_mask_left_face_only(state, pec_mask):
    left_neighbor_not_pec = jnp.roll(~pec_mask, +1, axis=0)
    left_face_mask = pec_mask & left_neighbor_not_pec
    return state._replace(
        ey=state.ey * (1.0 - left_face_mask.astype(state.ey.dtype)),
        ez=state.ez * (1.0 - left_face_mask.astype(state.ez.dtype)),
    )


def _pec_mask_left_face_and_inside(state, pec_mask):
    """Zero Ey/Ez at LEFT FACE and zero Ex at INTERIOR cells (no left-face Ex effect)."""
    left_neighbor_not_pec = jnp.roll(~pec_mask, +1, axis=0)
    left_face_mask = pec_mask & left_neighbor_not_pec
    # Standard ex-zero everywhere PEC has x-neighbor PEC (interior of wall)
    mask_ex_inside = pec_mask & (jnp.roll(pec_mask, 1, axis=0) | jnp.roll(pec_mask, -1, axis=0))
    return state._replace(
        ex=state.ex * (1.0 - mask_ex_inside.astype(state.ex.dtype)),
        ey=state.ey * (1.0 - left_face_mask.astype(state.ey.dtype)),
        ez=state.ez * (1.0 - left_face_mask.astype(state.ez.dtype)),
    )


_orig = pec_mod.apply_pec_mask


def _pec_mask_NOOP(state, pec_mask):
    """Identity — no PEC enforcement at all. If reflection still = 0.914,
    apply_pec_mask isn't even being called."""
    return state


patches = {
    "V0": _orig,
    "V1": _pec_mask_no_ex,
    "V2": _pec_mask_left_face_only,
    "V3": _pec_mask_left_face_and_inside,
    "V_NOOP": _pec_mask_NOOP,
}
pec_mod.apply_pec_mask = patches[VARIANT]
print(f"[runner] using variant {VARIANT}: "
      f"{patches[VARIANT].__name__}", flush=True)

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402
from rfx.sources import waveguide_port as wg  # noqa: E402

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
sim.add(Box((0.155, 0.0, 0.0), (0.155 + 2*DX_M, A_WG, B_WG)),
        material="pec")
port_freqs = jnp.asarray(FREQS_HZ)
sim.add_waveguide_port(
    0.040, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=port_freqs, f0=F0, bandwidth=0.6,
    waveform="modulated_gaussian", reference_plane=0.050, name="left",
    probe_offset=114,
)

result = sim.run(num_periods=200)
wp = result.waveguide_ports
cfg = wp["left"] if isinstance(wp, dict) else wp[0]

# Decompose at probe_x (wall-front) and ref_x
n = int(cfg.n_steps_recorded)
dt = float(cfg.dt); fc = float(cfg.f_cutoff)
freqs = np.asarray(cfg.freqs)
omega = 2 * np.pi * freqs.astype(np.float64)
beta_arr = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
Z = omega * MU_0 / np.maximum(beta_arr, 1e-30)
n_idx = np.arange(n)
phase = np.exp(-1j * omega[None, :] * n_idx[:, None] * dt)


def _decomp(V_t, I_t):
    Vd = 2.0 * dt * (np.asarray(V_t)[:n].astype(np.float64) @ phase)
    Id_raw = 2.0 * dt * (np.asarray(I_t)[:n].astype(np.float64) @ phase)
    Id = Id_raw * np.exp(+1j * omega * 0.5 * dt)
    return 0.5*(Vd + Z*Id), 0.5*(Vd - Z*Id)


fwd_p, bwd_p = _decomp(cfg.v_probe_t, cfg.i_probe_t)
fwd_r, bwd_r = _decomp(cfg.v_ref_t, cfg.i_ref_t)
wall = np.abs(bwd_p) / np.maximum(np.abs(fwd_p), 1e-30)
ref = np.abs(bwd_r) / np.maximum(np.abs(fwd_r), 1e-30)
print(f"[{VARIANT}] |bwd|/|fwd| @ wall (154mm): mean {wall.mean():.4f}  "
      f"range [{wall.min():.4f}, {wall.max():.4f}]")
print(f"[{VARIANT}] |bwd|/|fwd| @ ref  (43mm):  mean {ref.mean():.4f}  "
      f"range [{ref.min():.4f}, {ref.max():.4f}]")
