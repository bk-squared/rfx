"""Test aperture trim + trapezoidal weighting.

The ghost cell (k=Nz-1) is OUTSIDE WR-90's z extent. Cells inside WR-90:
k=0..Nz-2. Boundary cells inside WR-90 are k=0 and k=Nz-2 (not Nz-1).

Variants:
  V0_BASELINE:  unchanged uniform weights, all Nu × Nv cells
  V1_TRIM_GHOST: skip ghost cell (k=Nz-1), uniform on k=0..Nz-2
  V2_TRIM_TRAP: trim ghost AND half-weight at k=0 / k=Nz-2 (true boundary)
  V3_GHOST_HALF: ghost cell weighted 0.5, others 1.0 (compromise)
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

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "V0_BASELINE"

import rfx.sources.waveguide_port as wg_mod  # noqa: E402

_orig = wg_mod._aperture_dA


def _v0_baseline(cfg):
    return _orig(cfg)


def _v1_trim_ghost(cfg):
    """Zero the ghost cell weight (k=Nz-1, j=Ny-1)."""
    u_w = jnp.asarray(cfg.u_widths)
    v_w = jnp.asarray(cfg.v_widths)
    nu = u_w.shape[0]; nv = v_w.shape[0]
    u_mask = jnp.ones(nu).at[-1].set(0.0)  # drop last u cell
    v_mask = jnp.ones(nv).at[-1].set(0.0)  # drop last v cell
    return (u_w * u_mask)[:, None] * (v_w * v_mask)[None, :]


def _v2_trim_trap(cfg):
    """Drop ghost AND trapezoidal at true boundary (k=0 and k=Nz-2)."""
    u_w = jnp.asarray(cfg.u_widths)
    v_w = jnp.asarray(cfg.v_widths)
    nu = u_w.shape[0]; nv = v_w.shape[0]
    u_weight = jnp.ones(nu).at[0].set(0.5).at[-2].set(0.5).at[-1].set(0.0)
    v_weight = jnp.ones(nv).at[0].set(0.5).at[-2].set(0.5).at[-1].set(0.0)
    return (u_w * u_weight)[:, None] * (v_w * v_weight)[None, :]


def _v3_ghost_half(cfg):
    """Ghost cell weighted 0.5; rest uniform."""
    u_w = jnp.asarray(cfg.u_widths)
    v_w = jnp.asarray(cfg.v_widths)
    nu = u_w.shape[0]; nv = v_w.shape[0]
    u_weight = jnp.ones(nu).at[-1].set(0.5)
    v_weight = jnp.ones(nv).at[-1].set(0.5)
    return (u_w * u_weight)[:, None] * (v_w * v_weight)[None, :]


def _v4_trim_v_only(cfg):
    """Drop ghost in v only, keep u uniform."""
    u_w = jnp.asarray(cfg.u_widths)
    v_w = jnp.asarray(cfg.v_widths)
    nv = v_w.shape[0]
    v_mask = jnp.ones(nv).at[-1].set(0.0)
    return u_w[:, None] * (v_w * v_mask)[None, :]


variants = {
    "V0_BASELINE": _v0_baseline,
    "V1_TRIM_GHOST": _v1_trim_ghost,
    "V2_TRIM_TRAP": _v2_trim_trap,
    "V3_GHOST_HALF": _v3_ghost_half,
    "V4_TRIM_V_ONLY": _v4_trim_v_only,
}
wg_mod._aperture_dA = variants[VARIANT]
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
print(f"VARIANT_RESULT {VARIANT}: |r|={refl.mean():.4f} arg={arg[10]:.2f}deg")
