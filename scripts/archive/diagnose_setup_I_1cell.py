"""Setup I: DOMAIN_X=0.200 + 1-cell-wide Box-PEC at cell 175 only.

For 1-cell wall, apply_pec_mask zeros Ey[175], Ez[175] but NOT Ex[175]
(no x-neighbor PEC). This exactly matches apply_pec_faces({x_hi}) semantics
on a single plane.

If r → 1.000: the 2-cell wall thickness is the bug (apply_pec_mask zeroing
the second cell creates the loss).
If r → 0.914: the bug isn't the wall thickness; something else.
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
    refl = np.abs(bwd) / np.maximum(np.abs(fwd), 1e-30)
    arg = np.degrees(np.angle(Vd / (Z * Id)))
    return refl, arg


def _build(box_lo, box_hi, label):
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
    sim.add(Box((box_lo, 0.0, 0.0), (box_hi, A_WG, B_WG)),
            material="pec")
    sim.add_waveguide_port(
        0.040, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=jnp.asarray(FREQS_HZ), f0=F0, bandwidth=0.6,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
        probe_offset=114,
    )
    return sim


for label, lo, hi in [
    ("I-1cell  Box(0.155, 0.156)", 0.155, 0.156),
    ("J-2cell  Box(0.155, 0.157) [original A]", 0.155, 0.157),
    ("K-3cell  Box(0.155, 0.158)", 0.155, 0.158),
    ("L-5cell  Box(0.155, 0.160)", 0.155, 0.160),
    ("M-10cell Box(0.155, 0.165)", 0.155, 0.165),
]:
    print(f"\n=== Setup {label} ===", flush=True)
    _CAPS.clear()
    sim = _build(lo, hi, label)
    result = sim.run(num_periods=200, compute_s_params=False)
    wp = result.waveguide_ports
    cfg = wp["left"] if isinstance(wp, dict) else wp[0]
    _ = wg.extract_waveguide_port_waves(cfg, ref_shift=0.0)
    refl, arg = _decomp(_CAPS[-1])
    print(f"   |r| mean = {refl.mean():.4f}  range [{refl.min():.4f}, {refl.max():.4f}]")
    print(f"   arg(V/ZI) at f=10.3 GHz: {arg[10]:.3f}°")
