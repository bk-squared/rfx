#!/usr/bin/env python3
"""#496 instance 1 — do `test_waveguide_twoport_contract_v1`'s bindings survive
a discipline-compliant absorber, and what does it cost?

THE QUESTION, in the issue's own words: "I did not determine whether the
contract tests' bindings survive a corrected absorber. That is the actual
question here and it needs someone to run it, not to be settled inside a
hygiene PR." Both outcomes are legitimate — correct the configuration, or record
that the file runs outside the discipline deliberately with the passivity
numbers annotated as absorber-limited rather than as extractor characterization.
This script produces the numbers that decide it.

HOW THE COMPARISON IS ISOLATED, and a correction to an earlier revision of this
file. rfx adds CPML as padding OUTSIDE the user domain (rfx/grid.py; the
preflight message at rfx/api/__init__.py:680 says so explicitly), so raising
cpml_layers at a FIXED domain leaves the interior — ports, DUT, reference
planes, port separation, port-to-absorber standoff — bit-identical. Verified
directly: at domain=(0.12, 0.04, 0.02) the grid goes (86,43,32) -> (122,79,68)
-> (150,107,96) for cpml 10 -> 28 -> 42, i.e. shape - 2*cpml is a constant 66
on every axis.

An earlier revision of this probe grew the domain by (N-10)*dx and shifted the
ports and DUT with it, on the false premise that 28 cells "would swallow the
ports". That padding moved the port-to-absorber standoff (10.0 -> 43.7 -> 70.0
mm) in LOCKSTEP with the absorber depth (18.7 -> 52.5 -> 78.7 mm). The two are
collinear over such a sweep, so it could not attribute anything to the absorber
— and rfx documents port-to-CPML proximity as an |S11|-inflating mechanism in
its own right (rfx/api/_preflight.py). The default here is now the clean sweep;
--pad-domain reproduces the flawed one for the record.

WHAT IS REPORTED, per configuration: every statistic the committed tests
actually assert (so "do the bindings survive" is read off the table rather than
argued), the passivity maxima the issue quotes as possibly absorber-limited
(1.211 dielectric / 1.534 PEC short), and wall-clock — because this is a
fast-suite file and the issue asks for the cost before committing to a change.

BASELINE FIDELITY IS THE FIRST CHECK. The N=10 row must reproduce what the
committed tests measure; if it does not, this driver is not measuring the thing
under discussion and no other row means anything.

Run:
  PYTHONPATH=/root/workspace/bk-workspace/rfx-residual \
    python scripts/diagnostics/i496_contract_v1_absorber_probe.py --cpml 10
  ... --cpml 28        # 0.5 lambda_g, the discipline floor
  ... --cpml 42        # 0.75 lambda_g, what #576's E4 correction targeted
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
C0 = 299_792_458.0

# Verbatim from tests/unit/sparams/test_waveguide_twoport_contract_v1.py — imported by value
# rather than restated loosely, because a probe that drifts from the file it is
# probing answers about a configuration nobody ships.
FREQS = np.linspace(4.5e9, 8.0e9, 20)
F0 = float(FREQS.mean())
BW = max(0.2, min(0.8, (FREQS[-1] - FREQS[0]) / max(F0, 1.0)))
BASE_DOMAIN = (0.12, 0.04, 0.02)
BASE_CPML = 10
PORT_L_X, PORT_R_X = 0.01, 0.09
PEC_BOX = ((0.05, 0.0, 0.0), (0.055, 0.04, 0.02))
DIEL_BOX = ((0.05, 0.0, 0.0), (0.07, 0.04, 0.02))
NUM_PERIODS = 40


def build(kind, cpml, dx, *, left_ref=None, right_ref=None, pad_domain=False):
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.geometry.csg import Box

    # pad=0 is the isolating comparison: CPML is added outside the domain, so
    # the interior is untouched and only the absorber changes.
    pad = ((cpml - BASE_CPML) * dx) if pad_domain else 0.0
    dom_x = BASE_DOMAIN[0] + 2.0 * pad
    sim = Simulation(freq_max=max(float(FREQS[-1]), F0),
                     domain=(dom_x, BASE_DOMAIN[1], BASE_DOMAIN[2]),
                     boundary="cpml", cpml_layers=cpml)
    if kind == "pec_short":
        sim.add_material("pec_like", eps_r=1.0, sigma=1e10)
        lo, hi = PEC_BOX
        sim.add(Box((lo[0] + pad, lo[1], lo[2]), (hi[0] + pad, hi[1], hi[2])),
                material="pec_like")
    elif kind == "dielectric":
        sim.add_material("diel", eps_r=4.0, sigma=0.0)
        lo, hi = DIEL_BOX
        sim.add(Box((lo[0] + pad, lo[1], lo[2]), (hi[0] + pad, hi[1], hi[2])),
                material="diel")
    elif kind != "empty":
        raise ValueError(kind)

    for x, d, name, ref in ((PORT_L_X + pad, "+x", "left", left_ref),
                            (PORT_R_X + pad, "-x", "right", right_ref)):
        sim.add_waveguide_port(x, direction=d, mode=(1, 0), mode_type="TE",
                               freqs=jnp.asarray(FREQS), f0=F0, bandwidth=BW,
                               ref_offset=3, probe_offset=15, name=name,
                               reference_plane=ref)
    return sim, dom_x, pad


def stats(s):
    s = np.asarray(s)
    colpow = np.sum(np.abs(s) ** 2, axis=0)
    recip = np.abs(np.abs(s[1, 0, :]) - np.abs(s[0, 1, :])) / np.maximum(
        np.maximum(np.abs(s[1, 0, :]), np.abs(s[0, 1, :])), 1e-12)
    return dict(
        mean_s11=float(np.mean(np.abs(s[0, 0, :]))),
        max_s11=float(np.max(np.abs(s[0, 0, :]))),
        mean_s21=float(np.mean(np.abs(s[1, 0, :]))),
        max_abs_s21_minus_1=float(np.max(np.abs(np.abs(s[1, 0, :]) - 1.0))),
        mean_colpow=float(np.mean(colpow)),
        max_colpow=float(np.max(colpow)),
        mean_recip=float(np.mean(recip)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpml", type=int, required=True)
    # A LONGER domain takes longer to settle. If num_periods stays fixed while
    # the domain grows, a null result cannot be told apart from truncation --
    # the co-condition #576 established for absorber vs record length. Sweeping
    # this is how the corrected rows earn the right to be read as "the absorber
    # was the term", rather than "the window hid it".
    ap.add_argument("--num-periods", dest="num_periods", type=float,
                    default=NUM_PERIODS)
    ap.add_argument("--pad-domain", dest="pad_domain", action="store_true",
                    help="reproduce the earlier, CONFOUNDED sweep that grew the "
                         "domain with the absorber (see module docstring)")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "0")
    import jax
    print(f"jax {jax.__version__} backend={jax.default_backend()}", flush=True)

    # dx is chosen by rfx from freq_max when not given explicitly — the same
    # expression rfx/api/__init__.py uses, not the 1.87 mm the advisory rounded
    # to, so the fractions below are this run's and not a transcribed constant.
    freq_max = max(float(FREQS[-1]), F0)
    dx = C0 / freq_max / 20.0
    # TWO cutoffs, and the discipline uses the NUMERICAL one. The analytic
    # c/2a = 3.747 GHz gives lambda_g = 120.3 mm; rfx's far-port advisory uses
    # the port's numerical TE10 cutoff (3.476 GHz, lambda_g = 104.9 mm), which
    # is why 28 cells reads 0.500 lambda_g to the advisory and 0.436 here. The
    # advisory's own verdict is captured per case in `warnings`, so the reported
    # fraction never has to be trusted on its own.
    fc_analytic = C0 / (2.0 * BASE_DOMAIN[1])
    f_lo = float(FREQS[0])
    lam_analytic = (C0 / f_lo) / np.sqrt(1.0 - (fc_analytic / f_lo) ** 2)
    FC_NUMERICAL = 3.476e9          # as reported by the advisory for this port
    lam_num = (C0 / f_lo) / np.sqrt(1.0 - (FC_NUMERICAL / f_lo) ** 2)
    frac = args.cpml * dx / lam_num
    print(f"dx={dx*1e3:.4f} mm  cpml={args.cpml} = {args.cpml*dx*1e3:.1f} mm  "
          f"-> {frac:.3f} lambda_g (numerical fc {FC_NUMERICAL/1e9:.3f} GHz, "
          f"lam_g {lam_num*1e3:.1f} mm; analytic fc would say "
          f"{args.cpml*dx/lam_analytic:.3f})  pad_domain={args.pad_domain}",
          flush=True)
    lam_g = lam_num

    out = {"driver": "i496_contract_v1_absorber_probe", "cpml_layers": args.cpml,
           "dx_m": dx, "lambda_g_low_m": float(lam_g),
           "cpml_fraction_lambda_g_low": float(frac),
           "num_periods": args.num_periods, "cases": {}}

    # Reference planes are ABSOLUTE x, so they shift with the padded geometry —
    # otherwise the ref-plane test would compare two different de-embed offsets
    # and attribute the difference to the absorber.
    pad_now = ((args.cpml - BASE_CPML) * dx) if args.pad_domain else 0.0
    runs = [("empty", "empty", {}),
            ("dielectric", "dielectric", {}),
            ("pec_short", "pec_short", {}),
            ("dielectric_refshift", "dielectric",
             {"left_ref": 0.02 + pad_now, "right_ref": 0.08 + pad_now})]
    for kind, base_kind, kw in runs:
        t0 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim, dom_x, _pad = build(base_kind, args.cpml, dx,
                                     pad_domain=args.pad_domain, **kw)
            r = sim.compute_waveguide_s_matrix(num_periods=args.num_periods,
                                               normalize=True)
            msgs = [str(w.message) for w in caught]
        dt = time.time() - t0
        st = stats(r.s_params)
        st["wallclock_s"] = dt
        st["domain_x_m"] = dom_x
        st["n_cells_x"] = int(round(dom_x / dx))
        st["warnings"] = [m for m in msgs
                          if any(k in m.lower()
                                 for k in ("absorber", "passiv", "ring-down",
                                           "lambda", "column"))]
        out["cases"][kind] = st
        print(f"  {kind:22} mean|S11|={st['mean_s11']:.4f} "
              f"mean|S21|={st['mean_s21']:.4f} meanCP={st['mean_colpow']:.4f} "
              f"maxCP={st['max_colpow']:.4f} recip={st['mean_recip']:.2e} "
              f"({dt:.1f}s, {st['n_cells_x']} x-cells)", flush=True)
        for m in st["warnings"]:
            print(f"      warn: {m[:150]}", flush=True)

    outdir = Path(args.outdir or (REPO / ".omx" / "i496-contract-v1-absorber"))
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"cpml{args.cpml}" + ("" if args.num_periods == NUM_PERIODS
                                else f"_np{int(args.num_periods)}")
    p = outdir / f"contract_v1_{tag}.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
