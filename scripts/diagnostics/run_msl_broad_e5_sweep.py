"""MSL broad-E5 envelope sweep driver (microstrip_line_port family).

Runs ``compute_msl_s_matrix`` on a parameterised case grid that stays inside
the validated lane (uniform Yee, quasi-TEM, N-probe LSQ extractor with
``strict_extractor=True``). Each case dumps an NPZ + per-case manifest so
the envelope builder can aggregate them into one E5 artifact.

Case grid (16 cases total):
  substrate    in {RO4003C, Teflon}                          (2)
  band         in {low: 2-10 GHz, high: 10-20 GHz}           (2)
  geometry     in {thru, half_wave_open_stub}                (2)
  dx_resolution in {lambda_g_min/30, lambda_g_min/40}        (2)

Each case targets a 50 Ω trace width (Hammerstad-Jensen). The stub geometry
is an open-circuited λ/4 branch line attached at mid-line.

Scope notes (docs/research_notes/20260528_msl_broad_e5_scope.md):
  - Architectural blockers (NU MSL, eigenmode source) are intentionally
    out of scope.
  - External openEMS reference comparison is layered on top by the
    envelope builder once the rfx sweep dumps land.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

C0 = 299_792_458.0


@dataclass(frozen=True)
class SubstrateSpec:
    name: str
    eps_r: float
    h_sub_m: float
    z0_target_ohm: float = 50.0


@dataclass(frozen=True)
class BandSpec:
    name: str
    freq_lo_hz: float
    freq_hi_hz: float
    n_freqs: int = 21

    @property
    def freq_max_hz(self) -> float:
        return self.freq_hi_hz

    @property
    def f0_hz(self) -> float:
        return 0.5 * (self.freq_lo_hz + self.freq_hi_hz)


SUBSTRATES = {
    "ro4003c": SubstrateSpec("RO4003C", eps_r=3.38, h_sub_m=0.787e-3),
    "teflon": SubstrateSpec("Teflon", eps_r=2.20, h_sub_m=0.508e-3),
}

BANDS = {
    "low": BandSpec("low", freq_lo_hz=2e9, freq_hi_hz=10e9, n_freqs=21),
    "high": BandSpec("high", freq_lo_hz=10e9, freq_hi_hz=20e9, n_freqs=21),
}

GEOMETRIES = ("thru", "open_stub")
# Mesh resolution as cells per substrate height (Yee staircase bias control).
# 4 is the rfx-warned minimum; 6 gives a finer cross-mesh consistency check.
# Both values still meet the wavelength criterion in every case below.
DX_RESOLUTIONS = ("sub4", "sub6")


def _hammerstad_jensen_width(z0_target: float, eps_r: float, h: float) -> float:
    """Hammerstad-Jensen closed form for trace width given target Z0.

    Inverts the standard MSL Z0(W/h) curve. Validity: 0.1 <= W/h <= 20.
    """
    eta0 = 376.73
    A = (z0_target / 60.0) * np.sqrt((eps_r + 1) / 2.0) + (eps_r - 1) / (eps_r + 1) * (
        0.23 + 0.11 / eps_r
    )
    B = eta0 * np.pi / (2.0 * z0_target * np.sqrt(eps_r))
    # Two regimes
    w_over_h_low = 8 * np.exp(A) / (np.exp(2 * A) - 2)
    w_over_h_high = (2 / np.pi) * (
        B - 1 - np.log(2 * B - 1)
        + (eps_r - 1) / (2 * eps_r) * (np.log(B - 1) + 0.39 - 0.61 / eps_r)
    )
    w_over_h = w_over_h_low if w_over_h_low < 2 else w_over_h_high
    return float(w_over_h * h)


def _eps_eff_quasitem(w: float, h: float, eps_r: float) -> float:
    """Hammerstad-Jensen quasi-TEM effective permittivity."""
    u = w / h
    return float(
        (eps_r + 1) / 2.0
        + (eps_r - 1) / 2.0 * (1.0 + 12.0 / u) ** -0.5
    )


def _lambda_g_min_m(eps_eff: float, freq_max_hz: float) -> float:
    return C0 / (freq_max_hz * np.sqrt(eps_eff))


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    substrate_key: str
    band_key: str
    geometry: str
    dx_resolution: str

    @property
    def cells_per_h_sub(self) -> int:
        return {"sub4": 4, "sub6": 6}[self.dx_resolution]


def enumerate_cases() -> list[CaseSpec]:
    """16-case grid: 2 substrates × 2 bands × 2 geom × 2 dx."""
    cases: list[CaseSpec] = []
    for sk in ("ro4003c", "teflon"):
        for bk in ("low", "high"):
            for geom in GEOMETRIES:
                for dxr in DX_RESOLUTIONS:
                    cid = f"{sk}-{bk}-{geom}-{dxr}"
                    cases.append(CaseSpec(cid, sk, bk, geom, dxr))
    return cases


def case_geometry_params(case: CaseSpec) -> dict:
    """Resolve all derived geometry/mesh params for a case.

    dx is set to ``h_sub / cells_per_h_sub`` so the substrate-air
    interface lands cleanly on a Yee face (avoids ``compute_msl_s_matrix``'s
    mixed-cell danger zone in [0.10, 0.40] fractional part of h_sub/dx).
    A sanity check then verifies dx also satisfies the wavelength
    criterion ``dx <= lambda_g_min / 20`` over the band.
    """
    sub = SUBSTRATES[case.substrate_key]
    band = BANDS[case.band_key]
    w_m = _hammerstad_jensen_width(sub.z0_target_ohm, sub.eps_r, sub.h_sub_m)
    eps_eff = _eps_eff_quasitem(w_m, sub.h_sub_m, sub.eps_r)
    lg_min = _lambda_g_min_m(eps_eff, band.freq_hi_hz)

    dx = sub.h_sub_m / float(case.cells_per_h_sub)
    if dx > lg_min / 20.0:
        raise RuntimeError(
            f"{case.case_id}: h_sub/{case.cells_per_h_sub} dx={dx*1e6:.0f}um "
            f"violates dx <= lambda_g_min/20 = {lg_min/20*1e6:.0f}um. "
            "Reduce band or increase cells_per_h_sub."
        )

    # Trace length: thru-line = 4 λ_g_min; open-stub: feed + λ/4 branch + extra.
    if case.geometry == "thru":
        trace_len_m = 4.0 * lg_min
    else:
        trace_len_m = 4.0 * lg_min  # main line same length
    # Air margin top + cpml pad
    air_top_m = 6.0 * sub.h_sub_m
    cpml_layers = 8

    # Port margin: must clear CPML pad AND give ≥ 2·h_sub buffer between
    # port and CPML inner face (rfx warning, strict ``<``). Add a 1.1×
    # safety factor on the h_sub buffer so we don't sit exactly on the
    # warning threshold.
    port_margin = max(
        (cpml_layers + 4) * dx,
        cpml_layers * dx + 2.2 * sub.h_sub_m,
    )

    domain_x_m = trace_len_m + 2.0 * port_margin
    domain_y_m = 12.0 * w_m + 2.0 * (cpml_layers + 4) * dx
    domain_z_m = sub.h_sub_m + air_top_m + 2.0 * (cpml_layers + 4) * dx
    return dict(
        substrate=sub,
        band=band,
        w_m=w_m,
        eps_eff=eps_eff,
        lg_min=lg_min,
        dx=dx,
        trace_len_m=trace_len_m,
        domain_x_m=domain_x_m,
        domain_y_m=domain_y_m,
        domain_z_m=domain_z_m,
        cpml_layers=cpml_layers,
        port_margin=port_margin,
    )


def build_simulation(case: CaseSpec, geo: dict) -> Simulation:
    """Build a Simulation for the case. PEC ground at z=0, substrate above,
    trace strip at z=h_sub, MSL port at each end."""
    sub: SubstrateSpec = geo["substrate"]
    band: BandSpec = geo["band"]

    sim = Simulation(
        freq_max=band.freq_hi_hz,
        domain=(geo["domain_x_m"], geo["domain_y_m"], geo["domain_z_m"]),
        dx=geo["dx"],
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="cpml", hi="cpml"),
            z=Boundary(lo="pec", hi="cpml"),
        ),
        cpml_layers=geo["cpml_layers"],
    )

    sim.add_material(f"sub_{sub.name.lower()}", eps_r=sub.eps_r, sigma=0.0)
    # Substrate slab
    z_ground = 0.0  # PEC face at z=0
    z_sub_top = z_ground + sub.h_sub_m
    sim.add(
        Box((0.0, 0.0, z_ground), (geo["domain_x_m"], geo["domain_y_m"], z_sub_top)),
        material=f"sub_{sub.name.lower()}",
    )

    # Trace strip at z_sub_top..z_sub_top + dx, centred in y, full x.
    y_centre = geo["domain_y_m"] / 2.0
    w = geo["w_m"]
    sim.add(
        Box(
            (0.0, y_centre - w / 2, z_sub_top),
            (geo["domain_x_m"], y_centre + w / 2, z_sub_top + geo["dx"]),
        ),
        material="pec",
    )

    # Optional open-stub branch (λ/4 at band centre).
    if case.geometry == "open_stub":
        stub_x_centre = geo["domain_x_m"] / 2.0
        lg_centre = C0 / (band.f0_hz * np.sqrt(geo["eps_eff"]))
        stub_len_m = lg_centre / 4.0
        stub_w = w  # same trace width for simplicity
        sim.add(
            Box(
                (stub_x_centre - stub_w / 2, y_centre + w / 2, z_sub_top),
                (
                    stub_x_centre + stub_w / 2,
                    y_centre + w / 2 + stub_len_m,
                    z_sub_top + geo["dx"],
                ),
            ),
            material="pec",
        )

    # MSL ports at each end of the main trace (margin precomputed in
    # case_geometry_params so domain sizing matches).
    port_margin = geo["port_margin"]
    sim.add_msl_port(
        position=(port_margin, y_centre, z_ground),
        width=w,
        height=sub.h_sub_m,
        direction="+x",
        impedance=sub.z0_target_ohm,
    )
    sim.add_msl_port(
        position=(geo["domain_x_m"] - port_margin, y_centre, z_ground),
        width=w,
        height=sub.h_sub_m,
        direction="-x",
        impedance=sub.z0_target_ohm,
    )
    return sim


def run_case(case: CaseSpec, *, num_periods: float = 40.0,
             out_dir: Path) -> dict:
    geo = case_geometry_params(case)
    sub: SubstrateSpec = geo["substrate"]
    band: BandSpec = geo["band"]

    t0 = time.time()
    sim = build_simulation(case, geo)
    freqs = jnp.linspace(band.freq_lo_hz, band.freq_hi_hz, band.n_freqs)

    res = sim.compute_msl_s_matrix(
        n_freqs=int(band.n_freqs),
        freqs=freqs,
        num_periods=num_periods,
        strict_extractor=False,  # honesty warns only; envelope gate still checks
    )
    t_run = time.time() - t0

    s = np.array(res.S)
    z0 = np.array(res.Z0)
    beta = np.array(res.beta)
    freqs_out = np.array(res.freqs)

    s11 = np.abs(s[0, 0, :])
    s21 = np.abs(s[1, 0, :])
    z0_extracted = np.real(z0[0])
    z0_hj_truth = sub.z0_target_ohm
    z0_err = np.abs(z0_extracted - z0_hj_truth) / z0_hj_truth

    case_npz = out_dir / f"{case.case_id}.npz"
    np.savez(
        case_npz,
        freqs_hz=freqs_out,
        s_matrix=s,
        z0_extracted=z0,
        beta_extracted=beta,
        w_m=geo["w_m"],
        dx=geo["dx"],
        domain=(geo["domain_x_m"], geo["domain_y_m"], geo["domain_z_m"]),
        eps_eff=geo["eps_eff"],
    )

    summary = dict(
        case_id=case.case_id,
        substrate=sub.name,
        eps_r=sub.eps_r,
        h_sub_m=sub.h_sub_m,
        band=band.name,
        freq_lo_hz=band.freq_lo_hz,
        freq_hi_hz=band.freq_hi_hz,
        geometry=case.geometry,
        dx_resolution=case.dx_resolution,
        w_m=geo["w_m"],
        dx=geo["dx"],
        eps_eff=geo["eps_eff"],
        wall_time_s=t_run,
        max_s11=float(s11.max()),
        mean_s21=float(s21.mean()),
        z0_extracted_median=float(np.median(z0_extracted)),
        z0_target=float(z0_hj_truth),
        z0_relative_error_median=float(np.median(z0_err)),
        npz_path=str(case_npz.relative_to(REPO)),
    )
    print(
        f"[{case.case_id}] {t_run:.1f}s  "
        f"max|S11|={summary['max_s11']:.3f}  "
        f"mean|S21|={summary['mean_s21']:.3f}  "
        f"Z0_med={summary['z0_extracted_median']:.1f}Ω "
        f"(err {100*summary['z0_relative_error_median']:.1f}%)"
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=".omx/physics-gate/2026-05-28-msl-broad-e5-sweep")
    p.add_argument("--num-periods", type=float, default=40.0)
    p.add_argument("--case", action="append", default=None,
                   help="run only these case_ids (repeatable). Default: all 16.")
    args = p.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = enumerate_cases()
    if args.case:
        cases = [c for c in cases if c.case_id in set(args.case)]
        if not cases:
            sys.exit("--case filter matched zero cases")

    print(f"running {len(cases)} cases to {out_dir}")
    summaries = []
    for c in cases:
        summaries.append(run_case(c, num_periods=args.num_periods, out_dir=out_dir))

    manifest = {
        "sweep_id": "msl_broad_e5_2026_05_28",
        "n_cases": len(summaries),
        "num_periods": args.num_periods,
        "cases": summaries,
    }
    manifest_path = out_dir / "msl_broad_e5_sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
