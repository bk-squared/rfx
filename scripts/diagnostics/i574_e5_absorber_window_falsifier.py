#!/usr/bin/env python3
"""#574 Step 0 — is the E5 absorber revision the binding lever, or the window?

#576 corrected the E4-lane causality: absorber depth and record length are
CO-CONDITIONS, and on the E4 geometry (dx = 1 mm) either lever alone removes
~99.7% of a lossless PEC short's over-unity |S11|:

    CPML  num_periods   max|S11|
      20       60        1.019948
      20      120        1.000063     <- window alone
      46       60        1.000091     <- absorber alone
      46      120        1.000061

The E5 sweep sits at dx = 0.25 mm, and its committed absorber was 24 cells =
6.0 mm = 0.099 lambda_g — in PHYSICAL terms 3.3x thinner than the E4 lane's
already-insufficient 20 mm, while `num_periods = 60` is the same physical
duration. So the E4 ratio must NOT be extrapolated across the resolution
change: the levers are expected to price differently here, and which one binds
decides the regeneration's configuration (absorber = 1.38x cells at 1x steps;
window = 2x steps at 1x memory). That decision is what this script measures,
BEFORE the ~10 h sweep is spent.

Two geometries, because they answer different questions:

  * pec_short — a lossless TOTAL reflector, so |S11| must be exactly 1 and any
    excess is unambiguous. This is the sharp discriminator and is directly
    comparable to the E4 table above.
  * slab (eps_r = 4, L = 4 mm, graded ratio 2.0) — the observable the sweep
    actually reports, and the strongest reflector in its matrix. Lossless, so
    column power |S11|^2 + |S21|^2 must be 1.

Settling witness: the OBSERVABLE-INVARIANCE form (case 19's idiom, `ring_down_witness`
in validation/crossval/19_wr90_iris_filter_aghanim.py), not an energy-dB
reading — rfx exposes no total-energy monitor, and on a lossless structure
truncation announces itself first as non-passive column power (case 19: colpow
1.58 at np = 100 on a Q~87 cavity) and then as a moved observable. Both are
reported per cell here, so "did the window bind" is read off the np 60 -> 120
column rather than asserted.

    python scripts/diagnostics/i574_e5_absorber_window_falsifier.py --cell 24x60
    python scripts/diagnostics/i574_e5_absorber_window_falsifier.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

C0 = 299_792_458.0
A, B = 22.86e-3, 10.16e-3
FC = C0 / (2 * A)
BAND = (8.2e9, 12.4e9)
N_FREQS = 11
BW = 0.4
DX = 0.25e-3                      # the E5 sweep's base resolution
DOMAIN_X = 200e-3
PL, PR = 40e-3, 160e-3
RL, RR = 50e-3, 150e-3
PEC_SHORT_X = 145e-3
PEC_SHORT_T = 2e-3                # PHYSICAL thickness, matching the E4 lane's
                                  # 2*dx at dx = 1 mm (8 cells here, not 2)
GRADING_RATIO = 2.0

_LAM_G_LOW = (C0 / BAND[0]) / np.sqrt(1.0 - (FC / BAND[0]) ** 2)

# The committed value and the derived one, as CELL COUNTS at this dx.
CPML_COMMITTED = 24               # 6.0 mm  = 0.099 lambda_g
CPML_DERIVED = int(np.ceil(0.75 * _LAM_G_LOW / DX))   # 183 -> 0.76 lambda_g
NP_COMMITTED = 60.0
NP_DOUBLED = 120.0

OUT = REPO / ".omx/i574-step0-absorber-window"


def _graded_dy(total: float, base_dx: float, ratio: float) -> np.ndarray:
    n = int(round(total / base_dx))
    x = np.linspace(-1.0, 1.0, n)
    w = 1.0 + (ratio - 1.0) * np.abs(x)
    return w / w.sum() * total


def run_cell(geometry: str, cpml: int, num_periods: float) -> dict:
    freqs = np.linspace(BAND[0], BAND[1], N_FREQS)
    dy = _graded_dy(A, DX, GRADING_RATIO)
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(DOMAIN_X, A, B),
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=cpml, dx=DX, dy_profile=dy,
    )
    if geometry == "pec_short":
        sim.add(Box((PEC_SHORT_X, 0, 0), (PEC_SHORT_X + PEC_SHORT_T, A, B)),
                material="pec")
    elif geometry == "slab":
        c = 0.5 * (PL + PR)
        sim.add_material("slab", eps_r=4.0, sigma=0.0)
        sim.add(Box((c - 2e-3, 0, 0), (c + 2e-3, A, B)), material="slab")
    else:
        raise SystemExit(f"unknown geometry {geometry!r}")

    pf = jnp.asarray(freqs)
    f0 = float(np.mean(freqs))
    for x, d, name, rp in ((PL, "+x", "left", RL), (PR, "-x", "right", RR)):
        sim.add_waveguide_port(x, direction=d, mode=(1, 0), mode_type="TE",
                               freqs=pf, f0=f0, bandwidth=BW,
                               waveform="modulated_gaussian",
                               reference_plane=rp, name=name)

    t0 = time.time()
    r = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize="flux")
    wall = time.time() - t0

    s = np.asarray(r.s_params)
    pi = {n: i for i, n in enumerate(r.port_names)}
    s11 = s[pi["left"], pi["left"], :]
    s21 = s[pi["right"], pi["left"], :]
    colpow = np.abs(s11) ** 2 + np.abs(s21) ** 2
    fr = np.asarray(r.freqs)
    return {
        "geometry": geometry, "cpml_layers": int(cpml),
        "cpml_fraction_lambda_g_low": round(cpml * DX / _LAM_G_LOW, 4),
        "num_periods": float(num_periods), "wall_s": round(wall, 1),
        "freqs_ghz": [round(float(f) / 1e9, 6) for f in fr],
        "s11_mag": [round(float(v), 9) for v in np.abs(s11)],
        "s21_mag": [round(float(v), 9) for v in np.abs(s21)],
        "col_power": [round(float(v), 9) for v in colpow],
        "max_s11": float(np.abs(s11).max()),
        "argmax_s11_ghz": float(fr[int(np.argmax(np.abs(s11)))] / 1e9),
        "max_col_power": float(colpow.max()),
        # Lane provenance: rfx's core is f32/complex64 and the port DFT
        # accumulators are complex128-declared, so CPU and GPU can differ in the
        # last digits. Recording the backend makes a cross-lane mismatch
        # attributable instead of mysterious.
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "argmax_colpow_ghz": float(fr[int(np.argmax(colpow))] / 1e9),
    }


_CELLS = {
    f"{CPML_COMMITTED}x{int(NP_COMMITTED)}": (CPML_COMMITTED, NP_COMMITTED),
    f"{CPML_COMMITTED}x{int(NP_DOUBLED)}": (CPML_COMMITTED, NP_DOUBLED),
    f"{CPML_DERIVED}x{int(NP_COMMITTED)}": (CPML_DERIVED, NP_COMMITTED),
    f"{CPML_DERIVED}x{int(NP_DOUBLED)}": (CPML_DERIVED, NP_DOUBLED),
}


def summarize(out_root: Path) -> int:
    """Print the 2x2 (or whatever subset exists) from committed cell artifacts.

    Lives here rather than in the job definition because a summary that is
    re-derived from the artifacts is checkable, while one printed inline by a
    batch runner is a claim with nothing behind it. Reports how many of the 8
    cells are present so a partial matrix cannot read as a complete one.
    """
    paths = sorted(out_root.glob("*.json"))
    if not paths:
        print(f"no cell artifacts in {out_root}")
        return 1
    rows = [json.loads(p.read_text()) for p in paths]
    rows.sort(key=lambda r: (r["geometry"], r["cpml_layers"], r["num_periods"]))
    print(f"{'geometry':10} {'CPML':>5} {'lam_g':>6} {'np':>4} {'max|S11|':>10} "
          f"{'@GHz':>6} {'colpow':>9} {'wall_s':>7} {'lane':>8}")
    for r in rows:
        print(f"{r['geometry']:10} {r['cpml_layers']:>5} "
              f"{r['cpml_fraction_lambda_g_low']:>6.3f} {r['num_periods']:>4.0f} "
              f"{r['max_s11']:>10.6f} {r['argmax_s11_ghz']:>6.2f} "
              f"{r['max_col_power']:>9.6f} {r['wall_s']:>7.0f} "
              f"{r.get('backend', '?'):>8}")
    print(f"cells present: {len(rows)}/8")
    for geom in sorted({r["geometry"] for r in rows}):
        g = {(r["cpml_layers"], r["num_periods"]): r for r in rows
             if r["geometry"] == geom}
        base = g.get((CPML_COMMITTED, NP_COMMITTED))
        if base is None:
            continue
        # METRIC BY GEOMETRY. |S11|-1 is only meaningful for a TOTAL reflector:
        # a dielectric slab reflects ~0.79, so |S11|-1 = -0.205 there, which is
        # not an excess at all and made an earlier revision of this summary
        # print "-0.3% of the excess removed" as if it meant something. For the
        # slab the passivity observable is COLUMN POWER (|S11|^2+|S21|^2 = 1 for
        # a lossless two-port); for the PEC short both work and column power is
        # the stricter of the two.
        key_metric = "max_s11" if geom == "pec_short" else "max_col_power"
        label_metric = "max|S11|" if geom == "pec_short" else "max column power"
        print(f"\n  {geom}: {label_metric} excess over unity, "
              f"and what each lever removes")
        e0 = base[key_metric] - 1.0
        print(f"    committed (cpml {CPML_COMMITTED}, np {NP_COMMITTED:.0f}): "
              f"{e0:+.6f}")
        for key, label in (((CPML_COMMITTED, NP_DOUBLED), "window alone"),
                           ((CPML_DERIVED, NP_COMMITTED), "absorber alone"),
                           ((CPML_DERIVED, NP_DOUBLED), "both")):
            cell = g.get(key)
            if cell is None:
                print(f"    {label:<15}: MISSING")
                continue
            e = cell[key_metric] - 1.0
            frac = (1.0 - e / e0) * 100.0 if e0 != 0 else float("nan")
            print(f"    {label:<15}: {e:+.6f}  ({frac:.1f}% of the excess removed)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="pec_short",
                    choices=("pec_short", "slab"))
    ap.add_argument("--cell", choices=sorted(_CELLS), action="append",
                    help="CPMLxNUM_PERIODS; repeatable. default = all four")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="artifact directory; default .omx/i574-step0-absorber-window. "
                         "Separate directories keep a GPU lane's results from "
                         "overwriting a concurrent CPU lane's — the two are a "
                         "cross-check, so they must not share filenames.")
    ap.add_argument("--summarize", action="store_true",
                    help="print the 2x2 from existing artifacts and exit")
    args = ap.parse_args(argv)
    cells = sorted(_CELLS) if (args.all or not args.cell) else args.cell

    out_root = Path(args.out_dir) if args.out_dir else OUT
    if not out_root.is_absolute():
        out_root = REPO / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        return summarize(out_root)
    print(f"#574 Step 0: {args.geometry}, dx={DX * 1e6:.0f}um, "
          f"lambda_g(8.2GHz)={_LAM_G_LOW * 1e3:.1f}mm")
    print(f"  committed absorber {CPML_COMMITTED} cells = "
          f"{CPML_COMMITTED * DX * 1e3:.1f}mm = "
          f"{CPML_COMMITTED * DX / _LAM_G_LOW:.3f} lambda_g")
    print(f"  derived   absorber {CPML_DERIVED} cells = "
          f"{CPML_DERIVED * DX * 1e3:.1f}mm = "
          f"{CPML_DERIVED * DX / _LAM_G_LOW:.3f} lambda_g")
    for key in cells:
        cpml, npd = _CELLS[key]
        print(f"\n[{args.geometry} {key}] cpml={cpml} np={npd:.0f} starting",
              flush=True)
        res = run_cell(args.geometry, cpml, npd)
        path = out_root / f"{args.geometry}_{key}.json"
        path.write_text(json.dumps(res, indent=2) + "\n")
        print(f"[{args.geometry} {key}] {res['wall_s']:.0f}s  "
              f"max|S11|={res['max_s11']:.6f} @ {res['argmax_s11_ghz']:.2f}GHz  "
              f"max colpow={res['max_col_power']:.6f} @ "
              f"{res['argmax_colpow_ghz']:.2f}GHz -> {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
