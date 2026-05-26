"""Band-parameterized WR sweep using normalize='flux' (power-flux extraction).

Runs a 4-case sweep (2 dx × 2 eps_r slab) for any rectangular WR
standard using the flux-extraction recipe:
- CPML_LAYERS = 24
- normalize = "flux"  (extract_waveguide_s_matrix_flux, sig 1+2 silver bullet)
- cells/lambda_min >= 60
- dx divides slab L into integer cells
- slab L chosen so Airy |S11| > 0.1 across band (no FP null inside)

Per-band specs selected via --band {WR28, WR62, WR15, WR340, WR10}.
Outputs per-case NPZ + manifest to .omx/physics-gate/<dated>/rfx-sweep/.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass
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


@dataclass
class BandSpec:
    name: str
    tag: str
    a_m: float
    b_m: float
    band_hz: tuple[float, float]
    slab_L_m: float
    dx_values_m: tuple[float, ...]
    domain_x_m: float
    port_left_x_m: float
    port_right_x_m: float
    ref_left_x_m: float
    ref_right_x_m: float
    cpml_layers: int = 24
    num_periods: int = 60
    bandwidth_frac: float = 0.4
    n_freqs: int = 11
    eps_r_values: tuple[float, ...] = (2.0, 4.0)

    @property
    def fc_te10_hz(self) -> float:
        return C0 / (2 * self.a_m)


# Each band's specs chosen so:
# - slab L keeps Airy |S11| > 0.1 across band (no Fabry-Perot null)
# - dx values divide L into integer cells
# - cells/lambda_min >= 60 at top frequency
# - port-to-slab distance >= ~1 lambda_g
BANDS = {
    "WR28": BandSpec(
        name="WR-28", tag="wr28_kaband",
        a_m=7.112e-3, b_m=3.556e-3,
        band_hz=(26.5e9, 40.0e9),
        slab_L_m=1.0e-3,        # short L: |S11| [0.37,0.39] eps_r=2, [0.68,0.75] eps_r=4, no FP null (L=4mm had eps_r=2 null in band)
        dx_values_m=(50e-6, 100e-6),  # 1000/50=20, 1000/100=10 cells
        domain_x_m=60e-3,
        port_left_x_m=15e-3, port_right_x_m=45e-3,
        ref_left_x_m=20e-3, ref_right_x_m=40e-3,
    ),
    "WR62": BandSpec(
        name="WR-62", tag="wr62_kuband",
        a_m=15.799e-3, b_m=7.899e-3,
        band_hz=(12.4e9, 18.0e9),
        slab_L_m=3.0e-3,        # short L: |S11| stays 0.41-0.46 (no FP null)
        dx_values_m=(200e-6, 250e-6),
        domain_x_m=150e-3,      # ~3 lambda_g @ band low edge
        port_left_x_m=30e-3, port_right_x_m=120e-3,
        ref_left_x_m=50e-3, ref_right_x_m=100e-3,
    ),
    "WR15": BandSpec(
        name="WR-15", tag="wr15_vband",
        a_m=3.7592e-3, b_m=1.8796e-3,
        band_hz=(50e9, 75e9),
        slab_L_m=0.8e-3,
        dx_values_m=(25e-6, 32e-6),  # cells/lambda_d @ 75 GHz eps_r=4: 80, 62
        domain_x_m=20e-3,
        port_left_x_m=5e-3, port_right_x_m=15e-3,
        ref_left_x_m=6.5e-3, ref_right_x_m=13.5e-3,
    ),
    "WR340": BandSpec(
        name="WR-340", tag="wr340_sband",
        a_m=86.36e-3, b_m=43.18e-3,
        band_hz=(2.2e9, 3.3e9),
        slab_L_m=12.0e-3,
        dx_values_m=(1000e-6, 1500e-6),
        domain_x_m=400e-3,
        port_left_x_m=100e-3, port_right_x_m=300e-3,
        ref_left_x_m=120e-3, ref_right_x_m=280e-3,
    ),
    "WR10": BandSpec(
        name="WR-10", tag="wr10_wband",
        a_m=2.54e-3, b_m=1.27e-3,
        band_hz=(75e9, 110e9),
        slab_L_m=0.6e-3,
        dx_values_m=(25e-6, 40e-6),
        domain_x_m=15e-3,
        port_left_x_m=4e-3, port_right_x_m=11e-3,
        ref_left_x_m=5e-3, ref_right_x_m=10e-3,
    ),
}


def build_sim(spec: BandSpec, dx: float, eps_r: float, freqs: np.ndarray):
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(spec.domain_x_m, spec.a_m, spec.b_m),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=spec.cpml_layers, dx=dx,
    )
    c = 0.5 * (spec.port_left_x_m + spec.port_right_x_m)
    sim.add_material("slab", eps_r=eps_r, sigma=0.0)
    sim.add(Box((c - 0.5*spec.slab_L_m, 0.0, 0.0),
                (c + 0.5*spec.slab_L_m, spec.a_m, spec.b_m)),
            material="slab")
    pf = jnp.asarray(freqs)
    f0 = float(np.mean(freqs))
    sim.add_waveguide_port(
        spec.port_left_x_m, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=spec.bandwidth_frac,
        waveform="modulated_gaussian",
        reference_plane=spec.ref_left_x_m, name="left",
    )
    sim.add_waveguide_port(
        spec.port_right_x_m, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=spec.bandwidth_frac,
        waveform="modulated_gaussian",
        reference_plane=spec.ref_right_x_m, name="right",
    )
    return sim


def run_band(spec: BandSpec, date_tag: str):
    out_dir = REPO / f".omx/physics-gate/{date_tag}-waveguide-broad-e5-{spec.tag}-flux/rfx-sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.linspace(spec.band_hz[0], spec.band_hz[1], spec.n_freqs)
    manifest = {
        "schema": f"rfx.waveguide_{spec.tag}_flux_sweep_manifest",
        "schema_version": 1,
        "waveguide": spec.name,
        "band_hz": list(spec.band_hz),
        "fc_te10_hz": spec.fc_te10_hz,
        "cutoff_ratio_range": [float(spec.band_hz[0]) / spec.fc_te10_hz,
                                float(spec.band_hz[1]) / spec.fc_te10_hz],
        "cpml_layers": spec.cpml_layers,
        "normalize": "flux",
        "num_periods": spec.num_periods,
        "bandwidth_frac": spec.bandwidth_frac,
        "domain_m": [spec.domain_x_m, spec.a_m, spec.b_m],
        "ports_x_m": [spec.port_left_x_m, spec.port_right_x_m],
        "reference_planes_x_m": [spec.ref_left_x_m, spec.ref_right_x_m],
        "slab_length_m": spec.slab_L_m,
        "freqs_hz": [float(f) for f in freqs],
        "cases": [],
    }
    print(f"=== {spec.name} ({spec.tag}) flux ===")
    print(f"  a={spec.a_m*1e3} mm, b={spec.b_m*1e3} mm, fc={spec.fc_te10_hz/1e9:.3f} GHz")
    print(f"  band {spec.band_hz[0]/1e9:.1f}-{spec.band_hz[1]/1e9:.1f} GHz "
          f"(cutoff ratio {manifest['cutoff_ratio_range'][0]:.2f}-{manifest['cutoff_ratio_range'][1]:.2f})")
    print(f"  slab L={spec.slab_L_m*1e3} mm, eps_r ∈ {list(spec.eps_r_values)}")
    print(f"  dx ∈ {[d*1e6 for d in spec.dx_values_m]} µm "
          f"(cells/λ@top={[int(C0/spec.band_hz[1]/d) for d in spec.dx_values_m]})")

    for dx in spec.dx_values_m:
        for eps_r in spec.eps_r_values:
            tag = f"dx{int(dx*1e6)}_slab_er{int(eps_r)}"
            out_path = out_dir / f"{tag}.npz"
            t0 = time.time()
            print(f"  [{tag}] starting", flush=True)
            sim = build_sim(spec, dx, eps_r, freqs)
            result = sim.compute_waveguide_s_matrix(
                num_periods=spec.num_periods, normalize="flux")
            s = np.asarray(result.s_params)
            pi = {n: i for i, n in enumerate(result.port_names)}
            fr = np.asarray(result.freqs)
            s11 = s[pi["left"], pi["left"], :]
            s21 = s[pi["right"], pi["left"], :]
            dt = time.time() - t0
            cells = (spec.domain_x_m/dx) * (spec.a_m/dx) * (spec.b_m/dx)
            np.savez(out_path,
                     freqs_hz=fr, s11=s11, s21=s21,
                     dx_m=dx, geometry=f"slab_er{int(eps_r)}",
                     cpml_layers=spec.cpml_layers,
                     eps_r=eps_r, slab_length_m=spec.slab_L_m,
                     num_periods=spec.num_periods, normalize="flux")
            manifest["cases"].append({
                "tag": tag, "dx_m": dx, "geometry": f"slab_er{int(eps_r)}",
                "eps_r": eps_r,
                "rfx_npz": str(out_path.relative_to(REPO)),
                "n_cells_total": int(cells),
                "cells_per_lambda_max_hz": float(C0/spec.band_hz[1]/dx),
                "wallclock_s": dt,
            })
            print(f"  [{tag}] done in {dt:.1f}s -> {out_path.name}", flush=True)

    manifest["jax_default_backend"] = jax.default_backend()
    manifest["jax_enable_x64"] = bool(jax.config.read("jax_enable_x64"))
    manifest["jax_version"] = jax.__version__
    manifest["numpy_version"] = np.__version__
    mp = out_dir / f"rfx_{spec.tag}_flux_sweep_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {mp}")
    return mp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--band", required=True, choices=sorted(BANDS.keys()))
    p.add_argument("--date-tag", default="2026-05-26")
    args = p.parse_args()
    spec = BANDS[args.band]
    run_band(spec, args.date_tag)


if __name__ == "__main__":
    main()
