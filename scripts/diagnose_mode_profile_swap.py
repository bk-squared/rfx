"""Test mode_profile="analytic" vs "discrete" on PEC-short |bwd|/|fwd|.

Old dead-list says the swap gave 0.002% change pre-architecture-refactor.
Re-test on the current post-scan rect-DFT architecture.
"""
from __future__ import annotations
import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS: list[dict] = []
_orig = wg._extract_global_waves


def _capture(cfg, V, I):
    Vc = np.asarray(V).copy()
    Ic = np.asarray(wg._co_located_current_spectrum(cfg, I)).copy()
    Zc = np.asarray(wg._compute_mode_impedance(cfg.freqs, cfg.f_cutoff,
                    cfg.mode_type, dt=cfg.dt, dx=cfg.dx)).copy()
    _CAPS.append({"V": Vc, "I": Ic, "Z": Zc, "f": np.asarray(cfg.freqs).copy(),
                  "ref_x_m": float(cfg.reference_x_m),
                  "dir": cfg.direction})
    return _orig(cfg, V, I)


wg._extract_global_waves = _capture

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
DOMAIN_X, DOMAIN_Y, DOMAIN_Z = 0.200, A_WG, B_WG
PORT_LEFT_X, PORT_RIGHT_X = 0.040, 0.160
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())
BW = 0.6
NUM_PERIODS = 200
CPML_LAYERS = 20
PEC_X = PORT_RIGHT_X - 0.005


def _build(mode_profile):
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
    sim.add(Box((PEC_X, 0.0, 0.0), (PEC_X + 2*DX_M, DOMAIN_Y, DOMAIN_Z)),
            material="pec")
    port_freqs = jnp.asarray(FREQS_HZ)
    sim.add_waveguide_port(
        PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=F0, bandwidth=BW,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
        mode_profile=mode_profile,
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=F0, bandwidth=BW,
        waveform="modulated_gaussian", reference_plane=0.150, name="right",
        mode_profile=mode_profile,
    )
    return sim


def _run(mode_profile):
    _CAPS.clear()
    sim = _build(mode_profile)
    sim.compute_waveguide_s_matrix(num_periods=NUM_PERIODS, normalize=True)
    cap = _CAPS[3]   # device run @ LEFT
    V, I, Z = cap["V"], cap["I"], cap["Z"]
    fwd = 0.5 * (V + Z * I)
    bwd = 0.5 * (V - Z * I)
    return cap["f"], fwd, bwd


for profile in ("discrete", "analytic"):
    print(f"\n=== mode_profile = {profile!r} ===", flush=True)
    f, fwd, bwd = _run(profile)
    ratio = np.abs(bwd) / np.abs(fwd)
    print(f"  |bwd|/|fwd|: mean={ratio.mean():.4f} "
          f"min={ratio.min():.4f} max={ratio.max():.4f}")
    for k in (0, 5, 10, 15, 20):
        print(f"    f={f[k]/1e9:5.2f} GHz   |bwd|/|fwd| = {ratio[k]:.4f}")
