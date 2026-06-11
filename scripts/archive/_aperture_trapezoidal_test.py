"""Test trapezoidal aperture weighting in modal V/I integration.

Critic's hypothesis: the 8% deficit is a measurement bug — the modal V/I
integral uses uniform weights (1.0 per cell) but should use trapezoidal
weights (0.5 at each PEC-bounded edge) to correctly handle the boundary
cells of an aperture enclosed by PEC walls.

Patch: replace _aperture_dA so that:
  weights[0]    = 0.5 (first u cell — at -y face, PEC)
  weights[Nu-1] = 0.5 (last u cell — at +y face, PEC)
  weights[1..Nu-2] = 1.0
similarly for v.

If r → 1.000 with this patch (without ANY apply_pec_mask change), the
critic's hypothesis is confirmed.

Read variant from sys.argv[1]:
  UNIFORM:    no change (baseline)
  TRAP_BOTH:  trapezoidal both u and v
  TRAP_U:     trapezoidal u only
  TRAP_V:     trapezoidal v only
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

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "UNIFORM"

import rfx.sources.waveguide_port as wg_mod  # noqa: E402

_orig_aperture_dA = wg_mod._aperture_dA


def _aperture_dA_trap_both(cfg):
    u_w = cfg.u_widths.copy() if hasattr(cfg.u_widths, 'copy') else cfg.u_widths
    v_w = cfg.v_widths.copy() if hasattr(cfg.v_widths, 'copy') else cfg.v_widths
    u_w = jnp.asarray(u_w)
    v_w = jnp.asarray(v_w)
    nu = u_w.shape[0]
    nv = v_w.shape[0]
    u_weight = jnp.ones(nu).at[0].set(0.5).at[-1].set(0.5)
    v_weight = jnp.ones(nv).at[0].set(0.5).at[-1].set(0.5)
    u_eff = u_w * u_weight
    v_eff = v_w * v_weight
    return u_eff[:, None] * v_eff[None, :]


def _aperture_dA_trap_u(cfg):
    u_w = jnp.asarray(cfg.u_widths)
    v_w = jnp.asarray(cfg.v_widths)
    nu = u_w.shape[0]
    u_weight = jnp.ones(nu).at[0].set(0.5).at[-1].set(0.5)
    return (u_w * u_weight)[:, None] * v_w[None, :]


def _aperture_dA_trap_v(cfg):
    u_w = jnp.asarray(cfg.u_widths)
    v_w = jnp.asarray(cfg.v_widths)
    nv = v_w.shape[0]
    v_weight = jnp.ones(nv).at[0].set(0.5).at[-1].set(0.5)
    return u_w[:, None] * (v_w * v_weight)[None, :]


variants = {
    "UNIFORM": _orig_aperture_dA,
    "TRAP_BOTH": _aperture_dA_trap_both,
    "TRAP_U": _aperture_dA_trap_u,
    "TRAP_V": _aperture_dA_trap_v,
}
wg_mod._aperture_dA = variants[VARIANT]
print(f"[runner] variant {VARIANT}", flush=True)

# Now also need to capture the cfg time series — same as before.
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
sim.add(Box((0.155, 0.0, 0.0), (0.155 + 2 * DX_M, A_WG, B_WG)),
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
print(f"VARIANT_RESULT {VARIANT}: |r|={refl.mean():.4f} arg={arg[10]:.2f}deg "
      f"range=[{refl.min():.4f},{refl.max():.4f}]")
