"""Test hypothesis: 8% deficit in Setup A is from σ=1e10 lossy update at PEC cells.

Even though apply_pec_mask zeros E at PEC cells, update_e first computes
E_new = ca·E_old + cb·curl_H with ca≈-1, cb≈2/σ — a tiny intermediate E.
The lossy contribution σ·E²·vol·dt is nonzero, accumulating over 7691 steps.

Patch: monkey-patch the rfx PEC material specification from σ=1e10 to σ=0
so that update_e at PEC cells uses ca=1, cb=dt/eps (vacuum). The mask still
zeros E, so reflection should not change — but power dissipation goes to
literally zero.

Expected: |bwd|/|fwd| → 1.000 (matching boundary path) if hypothesis correct.
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

# Patch PEC material BEFORE importing rfx.api
import rfx.api as _api  # noqa: E402

# Show current PEC material spec
print(f"Original PEC material entry: {_api.MATERIAL_LIBRARY['pec']}")
_api.MATERIAL_LIBRARY["pec"] = {"eps_r": 1.0, "sigma": 0.0}
print(f"Patched  PEC material entry: {_api.MATERIAL_LIBRARY['pec']}")

# Now run the existing diagnose_VI_at_wall geometry
from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS: list[dict] = []
_orig = wg.extract_waveguide_port_waves


def _capture(cfg, *, ref_shift=0.0):
    a, b = _orig(cfg, ref_shift=ref_shift)
    _CAPS.append({
        "v_probe_t": np.asarray(cfg.v_probe_t).copy(),
        "i_probe_t": np.asarray(cfg.i_probe_t).copy(),
        "n": int(cfg.n_steps_recorded), "dt": float(cfg.dt),
        "freqs": np.asarray(cfg.freqs).copy(),
        "f_cutoff": float(cfg.f_cutoff),
        "probe_x_m": float(cfg.probe_x_m),
    })
    return a, b


wg.extract_waveguide_port_waves = _capture
_api.extract_waveguide_port_waves = _capture

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
    sim.add(Box((0.155, 0.0, 0.0), (0.155 + 2 * DX_M, A_WG, B_WG)),
            material="pec")
    sim.add_waveguide_port(
        0.040, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=jnp.asarray(FREQS_HZ), f0=F0, bandwidth=0.6,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
        probe_offset=114,
    )
    return sim


def _decomp(cap):
    n = cap["n"]; dt = cap["dt"]; fc = cap["f_cutoff"]
    freqs = cap["freqs"]
    omega = 2 * np.pi * freqs.astype(np.float64)
    beta_arr = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
    Z = omega * MU_0 / np.maximum(beta_arr, 1e-30)
    n_idx = np.arange(n)
    phase = np.exp(-1j * omega[None, :] * n_idx[:, None] * dt)
    Vd = 2.0 * dt * (cap["v_probe_t"][:n].astype(np.float64) @ phase)
    Id_raw = 2.0 * dt * (cap["i_probe_t"][:n].astype(np.float64) @ phase)
    Id = Id_raw * np.exp(+1j * omega * 0.5 * dt)
    fwd = 0.5 * (Vd + Z * Id)
    bwd = 0.5 * (Vd - Z * Id)
    return freqs, fwd, bwd


sim = _build()
result = sim.run(num_periods=200, compute_s_params=False)
wp = result.waveguide_ports
cfg = wp["left"] if isinstance(wp, dict) else wp[0]
_ = wg.extract_waveguide_port_waves(cfg, ref_shift=0.0)
cap = _CAPS[-1]
freqs, fwd, bwd = _decomp(cap)
refl = np.abs(bwd) / np.maximum(np.abs(fwd), 1e-30)

print(f"\n=== With σ=0 in PEC material (mask still zeros E) ===")
print(f"  probe_x = {cap['probe_x_m']*1000:.1f} mm")
print(f"  |bwd|/|fwd| = mean {refl.mean():.4f}  range [{refl.min():.4f}, {refl.max():.4f}]")
for k in [0, 5, 10, 15, 20]:
    print(f"    f={freqs[k]/1e9:5.2f} GHz: {refl[k]:.4f}")

print()
print("Compare:")
print(f"  Setup A (σ=1e10, default):           |bwd|/|fwd| = 0.9141")
print(f"  Setup B.2 (boundary, no PEC mat):    |bwd|/|fwd| = 1.0000")
print(f"  Setup A (σ=0, mask only):            |bwd|/|fwd| = {refl.mean():.4f}")
