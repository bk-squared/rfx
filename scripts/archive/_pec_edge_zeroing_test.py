"""Test whether the EDGE-cell zeroing is what fixes the 8% loss.

The mask formula misses Ez[175, *, Nz-1] and Ey[175, Ny-1, *] — cells just
OUTSIDE the wall in y or z direction. Test variants:

  M_BASE: apply_pec_mask formula (original — produces r=0.9141)
  M+EZ_EDGE: also zero Ez[175, :, Nz-1] (top z-edge of wall)
  M+EY_EDGE: also zero Ey[175, Ny-1, :] (right y-edge of wall)
  M+BOTH: both edges
  FACE: at[175, :, :].set(0.0) for both (gives r=1.0000)

Read variant from sys.argv[1].
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

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "M_BASE"
PLANE_IDX = 175

_orig_mask = pec_mod.apply_pec_mask


def _m_base(state, pec_mask):
    return _orig_mask(state, pec_mask)


def _m_plus_ez_edge(state, pec_mask):
    state = _orig_mask(state, pec_mask)
    return state._replace(
        ez=state.ez.at[PLANE_IDX, :, -1].set(0.0)  # top z edge
    )


def _m_plus_ey_edge(state, pec_mask):
    state = _orig_mask(state, pec_mask)
    return state._replace(
        ey=state.ey.at[PLANE_IDX, -1, :].set(0.0)  # right y edge
    )


def _m_plus_both(state, pec_mask):
    state = _orig_mask(state, pec_mask)
    return state._replace(
        ez=state.ez.at[PLANE_IDX, :, -1].set(0.0),
        ey=state.ey.at[PLANE_IDX, -1, :].set(0.0),
    )


def _face(state, pec_mask):
    return state._replace(
        ey=state.ey.at[PLANE_IDX, :, :].set(0.0),
        ez=state.ez.at[PLANE_IDX, :, :].set(0.0),
    )


variants = {
    "M_BASE": _m_base,
    "M+EZ_EDGE": _m_plus_ez_edge,
    "M+EY_EDGE": _m_plus_ey_edge,
    "M+BOTH": _m_plus_both,
    "FACE": _face,
}
pec_mod.apply_pec_mask = variants[VARIANT]
print(f"[runner] variant {VARIANT}", flush=True)

from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS = []
_orig_extract = wg.extract_waveguide_port_waves


def _capture(cfg, *, ref_shift=0.0):
    a, b = _orig_extract(cfg, ref_shift=ref_shift)
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
print(f"VARIANT_RESULT {VARIANT}: |r|={refl.mean():.4f} arg={arg[10]:.2f}deg")
