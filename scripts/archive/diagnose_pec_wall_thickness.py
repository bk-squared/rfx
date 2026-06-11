"""Test whether the 2-cell PEC wall is the source of the 8% reflection deficit.

The PEC short in crossval/11 is a 2-cell-thick (= 2 mm) Box at x=155 mm.
A perfect reflector should give |bwd|/|fwd|=1 inside the same run, but rfx
yields ~0.91 frequency-flat. If the PEC wall is leaking, thickening it to
20 cells (20 mm) should push the ratio toward 1.0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp  # noqa: E402
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Patch _extract_global_waves to capture V/I/Z
from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS: list[dict] = []
_orig = wg._extract_global_waves


def _capture(cfg, V, I):
    _CAPS.append({
        "V": np.asarray(V).copy(),
        "I": np.asarray(wg._co_located_current_spectrum(cfg, I)).copy(),
        "Z": np.asarray(wg._compute_mode_impedance(cfg.freqs, cfg.f_cutoff,
                        cfg.mode_type, dt=cfg.dt, dx=cfg.dx)).copy(),
        "f": np.asarray(cfg.freqs).copy(),
        "ref_x_m": float(cfg.reference_x_m),
        "dir": cfg.direction,
    })
    return _orig(cfg, V, I)


wg._extract_global_waves = _capture

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

# crossval/11 constants
A_WG, B_WG = 22.86e-3, 10.16e-3
F_CUTOFF_TE10 = 6.557e9
DX_M = 0.001
DOMAIN_X, DOMAIN_Y, DOMAIN_Z = 0.200, A_WG, B_WG
PORT_LEFT_X, PORT_RIGHT_X = 0.040, 0.160
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())
BW = 0.6
NUM_PERIODS = 200
CPML_LAYERS = 20

PEC_X = PORT_RIGHT_X - 0.005   # 155 mm


def _build(pec_thickness_cells: int) -> Simulation:
    sim = Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(DOMAIN_X, DOMAIN_Y, DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=CPML_LAYERS,
        dx=DX_M,
    )
    pec_lo = (PEC_X, 0.0, 0.0)
    pec_hi = (PEC_X + pec_thickness_cells * DX_M, DOMAIN_Y, DOMAIN_Z)
    sim.add(Box(pec_lo, pec_hi), material="pec")
    port_freqs = jnp.asarray(FREQS_HZ)
    sim.add_waveguide_port(
        PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=F0, bandwidth=BW,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=F0, bandwidth=BW,
        waveform="modulated_gaussian", reference_plane=0.150, name="right",
    )
    return sim


def _run(thickness):
    _CAPS.clear()
    sim = _build(thickness)
    sim.compute_waveguide_s_matrix(num_periods=NUM_PERIODS, normalize=True)
    # capture[3] is the device run @ LEFT (drive=L, recv=L)
    cap = _CAPS[3]
    V, I, Z = cap["V"], cap["I"], cap["Z"]
    fwd = 0.5 * (V + Z * I)
    bwd = 0.5 * (V - Z * I)
    return cap["f"], fwd, bwd


for thickness in (2, 5, 20):
    print(f"\n=== PEC wall thickness = {thickness} cell(s) "
          f"({thickness * DX_M * 1000:.0f} mm) ===", flush=True)
    f, fwd, bwd = _run(thickness)
    ratio = np.abs(bwd) / np.abs(fwd)
    print(f"  |bwd|/|fwd|: mean={ratio.mean():.4f} min={ratio.min():.4f} "
          f"max={ratio.max():.4f}")
    for k in (0, 5, 10, 15, 20):
        print(f"    f={f[k]/1e9:5.2f} GHz   |bwd|/|fwd| = {ratio[k]:.4f}")
