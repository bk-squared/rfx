"""Disentangle: is the bug in apply_pec_mask, in the +x CPML, or in their combo?

Setup labels:
  A     : DOMAIN=0.200, +x CPML,    Box-PEC mask at 155-157  → r=0.914
  B.2   : DOMAIN=0.155, +x PEC face boundary                  → r=1.000
  E     : DOMAIN=0.200, +x PEC face boundary, no Box          → r=1.000
  F     : DOMAIN=0.200, +x CPML, no Box (empty)               → r=0     [forward only]
  G     : DOMAIN=0.200, +x PEC face boundary, Box-PEC at 155-157 (mask)
            → tests apply_pec_mask WITHOUT the +x CPML
  H     : DOMAIN=0.200, +x CPML, +x_hi pec_face, Box at 155-157 (mask)
            → would have x_hi conflict — skip
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


def _build(label):
    if label == "A":
        bnd = BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        )
        domain_x = 0.200
        add_box = True
    elif label == "G":  # NO +x CPML, but keep Box (mask) at internal wall
        bnd = BoundarySpec(
            x=Boundary(lo="cpml", hi="pec"),  # PEC at +x face (passive),
                                              # but Box still creates internal wall
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        )
        domain_x = 0.200
        add_box = True
    elif label == "F":  # +x CPML, no Box (empty long guide)
        bnd = BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        )
        domain_x = 0.200
        add_box = False
    else:
        raise ValueError(label)

    sim = Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(domain_x, A_WG, B_WG),
        boundary=bnd, cpml_layers=20, dx=DX_M,
    )
    if add_box:
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
    refl = np.abs(bwd) / np.maximum(np.abs(fwd), 1e-30)
    arg_VZI = np.degrees(np.angle(Vd / (Z * Id)))
    return refl, arg_VZI


for label in ["A", "G", "F"]:
    print(f"\n=== Setup {label} ===", flush=True)
    _CAPS.clear()
    sim = _build(label)
    result = sim.run(num_periods=200, compute_s_params=False)
    wp = result.waveguide_ports
    cfg = wp["left"] if isinstance(wp, dict) else wp[0]
    _ = wg.extract_waveguide_port_waves(cfg, ref_shift=0.0)
    refl, arg = _decomp(_CAPS[-1])
    print(f"   |r| mean = {refl.mean():.4f}  range [{refl.min():.4f}, {refl.max():.4f}]")
    print(f"   arg(V/ZI) at f=10.3 GHz: {arg[10]:.3f}°")
