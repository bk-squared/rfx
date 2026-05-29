"""Multi-mode over-moded waveguide broad-E5 sweep (normalize='flux').

Over-moded narrow-b rectangular guide (a=22.86 mm, b=4 mm) so the TE10+TE20
two-mode window (13.1-19.67 GHz) is wide and TE20 sits comfortably above
cutoff in the 15.5-18.5 GHz band. A centered full-cross-section dielectric
slab preserves modal symmetry: TE10->TE10 and TE20->TE20 each follow the
single-mode analytic Airy reflection/transmission with their own modal
beta; TE10<->TE20 coupling is exactly 0 by parity. This is the analytic
truth for a multi-mode S-matrix.

Sweeps dx in {200,250} um x eps_r in {2,4}. Records the full multi-mode
S-matrix (4 channels: L_TE10, L_TE20, R_TE10, R_TE20) per case.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

C0 = 299_792_458.0
A = 22.86e-3
B = 4.0e-3
BAND = (15.5e9, 18.5e9)
N_FREQS = 7
BW = 0.3
CPML = 24
NUM_PERIODS = 100
DOMAIN_X = 200e-3
PL = 40e-3
PR = 160e-3
RL = 50e-3
RR = 150e-3
SLAB_L = 2.0e-3
DX_VALUES = (200e-6, 250e-6)
EPS_R_VALUES = (2.0, 4.0)
OUT_DIR = REPO / ".omx/physics-gate/2026-05-27-waveguide-multimode-broad-e5-flux/rfx-sweep"


def build_sim(dx, eps_r, freqs):
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1, domain=(DOMAIN_X, A, B),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec")),
        cpml_layers=CPML, dx=dx)
    c = 0.5 * (PL + PR)
    sim.add_material("slab", eps_r=eps_r, sigma=0.0)
    sim.add(Box((c - 0.5*SLAB_L, 0.0, 0.0), (c + 0.5*SLAB_L, A, B)), material="slab")
    pf = jnp.asarray(freqs)
    f0 = float(np.mean(freqs))
    sim.add_waveguide_port(PL, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=BW, waveform="modulated_gaussian",
        reference_plane=RL, name="left", n_modes=2)
    sim.add_waveguide_port(PR, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=BW, waveform="modulated_gaussian",
        reference_plane=RR, name="right", n_modes=2)
    return sim


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freqs = np.linspace(BAND[0], BAND[1], N_FREQS)
    FC10 = C0/(2*A); FC20 = C0/A
    manifest = {
        "schema": "rfx.waveguide_multimode_broad_e5_flux_sweep_manifest",
        "schema_version": 1,
        "waveguide": "over-moded a=22.86mm b=4mm (TE10+TE20)",
        "a_m": A, "b_m": B,
        "band_hz": list(BAND),
        "fc_te10_hz": FC10, "fc_te20_hz": FC20,
        "te20_cutoff_ratio_range": [BAND[0]/FC20, BAND[1]/FC20],
        "cpml_layers": CPML, "normalize": "flux", "num_periods": NUM_PERIODS,
        "bandwidth_frac": BW,
        "domain_m": [DOMAIN_X, A, B],
        "ports_x_m": [PL, PR], "reference_planes_x_m": [RL, RR],
        "slab_length_m": SLAB_L,
        "freqs_hz": [float(f) for f in freqs],
        "cases": [],
    }
    print(f"multimode over-moded a={A*1e3} b={B*1e3}mm "
          f"TE10 fc={FC10/1e9:.2f} TE20 fc={FC20/1e9:.2f}")
    print(f"band {BAND[0]/1e9}-{BAND[1]/1e9} GHz (TE20 ratio "
          f"{BAND[0]/FC20:.2f}-{BAND[1]/FC20:.2f}), L={SLAB_L*1e3}mm")
    for dx in DX_VALUES:
        for eps_r in EPS_R_VALUES:
            tag = f"dx{int(dx*1e6)}_slab_er{int(eps_r)}"
            t0 = time.time()
            print(f"  [{tag}] starting", flush=True)
            sim = build_sim(dx, eps_r, freqs)
            r = sim.compute_waveguide_s_matrix(num_periods=NUM_PERIODS, normalize="flux")
            s = np.asarray(r.s_params)
            names = list(r.port_names)
            dt = time.time() - t0
            out_path = OUT_DIR / f"{tag}.npz"
            np.savez(out_path, freqs_hz=np.asarray(r.freqs), s_params=s,
                     port_names=np.array(names), dx_m=dx, eps_r=eps_r,
                     slab_length_m=SLAB_L, cpml_layers=CPML, num_periods=NUM_PERIODS)
            manifest["cases"].append({
                "tag": tag, "dx_m": dx, "eps_r": eps_r,
                "geometry": f"slab_er{int(eps_r)}",
                "port_names": names,
                "rfx_npz": str(out_path.relative_to(REPO)),
                "cells_per_lambda_max_hz": float(C0/BAND[1]/dx),
                "wallclock_s": dt,
            })
            print(f"  [{tag}] done in {dt:.1f}s -> {out_path.name}", flush=True)
    manifest["jax_default_backend"] = jax.default_backend()
    manifest["jax_version"] = jax.__version__
    manifest["numpy_version"] = np.__version__
    mp = OUT_DIR / "rfx_multimode_flux_sweep_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {mp}")


if __name__ == "__main__":
    main()
