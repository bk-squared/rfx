"""Surgical patches to apply_pec_mask — find which step causes the 8.6% loss.

Variants:
  V0: standard apply_pec_mask (baseline, expect ~0.914)
  V1: don't zero Ex (only Ey, Ez)
  V2: zero only at LEFT FACE of wall (mimics 1-cell wall enforcement)
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
from rfx.sources import waveguide_port as wg  # noqa: E402

_orig_pec_mask = pec_mod.apply_pec_mask


def _pec_mask_no_ex(state, pec_mask):
    mask_ey = pec_mask & (jnp.roll(pec_mask, 1, axis=1) | jnp.roll(pec_mask, -1, axis=1))
    mask_ez = pec_mask & (jnp.roll(pec_mask, 1, axis=2) | jnp.roll(pec_mask, -1, axis=2))
    return state._replace(
        ex=state.ex,
        ey=state.ey * (1.0 - mask_ey.astype(state.ey.dtype)),
        ez=state.ez * (1.0 - mask_ez.astype(state.ez.dtype)),
    )


def _pec_mask_left_face_only(state, pec_mask):
    """Zero Ey/Ez ONLY at cells whose left x-neighbor is NOT PEC.
    For a 2-cell wall at [155, 156]: cell 155 has left-neighbor 154 (vacuum)
    → zeroed. Cell 156 has left-neighbor 155 (PEC) → NOT zeroed."""
    left_neighbor_not_pec = jnp.roll(~pec_mask, +1, axis=0)
    left_face_mask = pec_mask & left_neighbor_not_pec
    return state._replace(
        ex=state.ex,
        ey=state.ey * (1.0 - left_face_mask.astype(state.ey.dtype)),
        ez=state.ez * (1.0 - left_face_mask.astype(state.ez.dtype)),
    )


_CAPS: list[dict] = []
_orig_extract = wg.extract_waveguide_port_waves


def _capture(cfg, *, ref_shift=0.0):
    a, b = _orig_extract(cfg, ref_shift=ref_shift)
    _CAPS.append({
        "v_ref_t": np.asarray(cfg.v_ref_t).copy(),
        "i_ref_t": np.asarray(cfg.i_ref_t).copy(),
        "v_probe_t": np.asarray(cfg.v_probe_t).copy(),
        "i_probe_t": np.asarray(cfg.i_probe_t).copy(),
        "n_steps_recorded": int(cfg.n_steps_recorded),
        "dt": float(cfg.dt), "freqs": np.asarray(cfg.freqs).copy(),
        "f_cutoff": float(cfg.f_cutoff),
        "ref_x_m": float(cfg.reference_x_m),
        "probe_x_m": float(cfg.probe_x_m),
    })
    return a, b


wg.extract_waveguide_port_waves = _capture
import rfx.api as _api_mod  # noqa: E402
_api_mod.extract_waveguide_port_waves = _capture

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())


def _build():
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
    return sim


def _decompose(cap):
    n = cap["n_steps_recorded"]
    dt = cap["dt"]; fc = cap["f_cutoff"]
    freqs = cap["freqs"]
    omega = 2 * np.pi * freqs.astype(np.float64)
    beta_arr = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
    Z = omega * MU_0 / np.maximum(beta_arr, 1e-30)
    n_idx = np.arange(n)
    phase = np.exp(-1j * omega[None, :] * n_idx[:, None] * dt)

    def _dft(x):
        return 2.0 * dt * (x.astype(np.float64) @ phase)

    def _decomp(V_t, I_t):
        Vd = _dft(V_t)
        Id = _dft(I_t) * np.exp(+1j * omega * 0.5 * dt)
        return 0.5*(Vd + Z*Id), 0.5*(Vd - Z*Id)

    fwd_p, bwd_p = _decomp(cap["v_probe_t"][:n], cap["i_probe_t"][:n])
    fwd_r, bwd_r = _decomp(cap["v_ref_t"][:n], cap["i_ref_t"][:n])
    return freqs, fwd_p, bwd_p, fwd_r, bwd_r


for label, patch in [
    ("V0: standard mask",     _orig_pec_mask),
    ("V1: skip Ex",           _pec_mask_no_ex),
    ("V2: only LEFT FACE",    _pec_mask_left_face_only),
]:
    print(f"\n=== {label} ===", flush=True)
    pec_mod.apply_pec_mask = patch
    _CAPS.clear()
    sim = _build()
    result = sim.run(num_periods=200)
    wp = result.waveguide_ports
    cfg = wp["left"] if isinstance(wp, dict) else wp[0]
    _ = wg.extract_waveguide_port_waves(cfg, ref_shift=0.0)
    cap = _CAPS[-1]
    freqs, fwd_p, bwd_p, fwd_r, bwd_r = _decompose(cap)
    wall = np.abs(bwd_p) / np.maximum(np.abs(fwd_p), 1e-30)
    ref = np.abs(bwd_r) / np.maximum(np.abs(fwd_r), 1e-30)
    print(f"  |bwd|/|fwd| @ wall (154mm): mean {wall.mean():.4f}")
    print(f"  |bwd|/|fwd| @ ref  (43mm):  mean {ref.mean():.4f}")

pec_mod.apply_pec_mask = _orig_pec_mask
