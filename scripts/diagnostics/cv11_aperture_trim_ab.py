"""cv11 A/B: is the pec-short |S11| change the aperture trim, or the #931 core?

WHY THIS EXISTS. Deleting cv11's in-script port-aperture trim (see
``validation/crossval/11_waveguide_port_wr90.py``, "#931 LATTICE OWNERSHIP"
(a)) was pre-declared to TIGHTEN the pec-short per-bin ``|S11|`` envelope.
The post-#931 run measured the opposite:

    run 369367259004 (origin/main d990e18c, trim present):
        [pec-short |S11|]  max_diff 0.0146  mean 0.0114
        [pec-short round-trip phase] max 9.99 deg / mean 6.16 deg
        [pec-short vs conj(MEEP)] |S| 0.2125 / 0.0563, angle 27.12 / 22.26 deg

    run 369367259194 (feat/931-crossval-b, trim deleted):
        [pec-short |S11|]  max_diff 0.0560  mean 0.0275
        [pec-short round-trip phase] max 17.20 deg / mean 9.71 deg
        [pec-short vs conj(MEEP)] |S| 0.2545 / 0.0594, angle 10.98 / 6.81 deg

Those two runs differ by TWO things at once — the trim, and the whole #931
core (stage A-F, including the short's far wall at 147 mm, which the volume
rule adds). Attributing the magnitude change to the trim from that pair
alone would be inference, not measurement (workspace rule R5).

WHAT THIS DOES. Both arms on ONE checkout, one variable: the pec-short leg
of cv11, built through the case's own ``_build_sim`` / ``_s_params``, with
and without the ``y_range``/``z_range`` aperture trim. Everything else —
mesh, source, reference planes, run length, the realized PEC short — is the
same object. It also prints each arm's ``cfg.f_cutoff``, which is the
quantity the trim was written to move.

    python scripts/diagnostics/cv11_aperture_trim_ab.py [--out-json PATH]

Two solves, ~2 min each on CPU. The verdict at the bottom splits the
observed baseline -> post step into the trim's own contribution (measured
here, one variable) and the remainder, which is the core's.

MEASURED, run 369367259198 (this branch, both arms):

    trim      cfg.f_cutoff 6.807677 GHz   |S11| [0.9289, 0.9811]  max dev 0.0711
    no_trim   cfg.f_cutoff 6.512162 GHz   |S11| [0.9440, 0.9888]  max dev 0.0560

So REMOVING the trim improves this leg by 0.0152, and the 0.0146 -> 0.0560
degradation from the pre-change baseline is NOT the trim's: about +0.057 of
it is the #931 core's, on the waveguide S-matrix lane. That lane is where
stage C replaced a sigma = 1e10 cell fill with the realized PEC edges
(commit 0184d64c) — the fold the inventory critic flagged as belonging to no
group. Filed for the core, not compensated for here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

# The pec-short |S11| deviation the pre-change baseline measured (run
# 369367259004, origin/main d990e18c, trim present). Quoted so the arms
# below can be compared against the era they are meant to explain.
BASELINE_MAX_DEV = 0.0146

REPO_ROOT = Path(__file__).resolve().parents[2]
CV11 = REPO_ROOT / "validation/crossval/11_waveguide_port_wr90.py"


def _load_cv11():
    spec = importlib.util.spec_from_file_location("_cv11_aperture_ab", CV11)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build(cv11, *, trim: bool):
    """cv11's pec-short simulation, with or without the aperture trim.

    Built from the case's own ``_build_sim`` so the two arms cannot drift
    from the shipped geometry; the trim is then re-applied by rebuilding the
    two port entries, which is the ONLY difference between the arms.
    """
    import jax.numpy as jnp

    sim = cv11._build_sim(cv11.FREQS_HZ, pec_short_x=cv11.PEC_SHORT_X)
    if not trim:
        return sim
    kw = dict(y_range=(0.0, cv11.A_WG_REALIZED - cv11.DX_M),
              z_range=(0.0, cv11.B_WG_REALIZED - cv11.DX_M))
    sim._waveguide_ports = []
    port_freqs = jnp.asarray(cv11.FREQS_HZ)
    sim.add_waveguide_port(
        cv11.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=cv11.F0_HZ, bandwidth=cv11.BANDWIDTH_REL,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
        **kw)
    sim.add_waveguide_port(
        cv11.PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=cv11.F0_HZ, bandwidth=cv11.BANDWIDTH_REL,
        waveform="modulated_gaussian", reference_plane=0.150, name="right",
        **kw)
    return sim


def _cutoff(cv11, sim) -> float:
    import jax.numpy as jnp
    grid = sim._build_grid()
    cfg = sim._build_waveguide_port_config(
        sim._waveguide_ports[0], grid, jnp.asarray(cv11.FREQS_HZ), 100)
    return float(cfg.f_cutoff)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    cv11 = _load_cv11()
    short = cv11.assert_realized_short(
        cv11._build_sim(cv11.FREQS_HZ, pec_short_x=cv11.PEC_SHORT_X))
    print(f"realized PEC-short walls: "
          f"{[round(x * 1e3, 3) for x in short['planes_m']]} mm "
          f"({short['n_cells']} cells) — identical in both arms")

    out = {"short": short, "arms": {}}
    for label, trim in (("trim", True), ("no_trim", False)):
        sim = _build(cv11, trim=trim)
        f_cut = _cutoff(cv11, sim)
        t0 = time.time()
        f_hz, s11, _ = cv11._s_params(sim, normalize=False)
        dt = time.time() - t0
        mag = np.abs(np.asarray(s11))
        dev = np.abs(mag - 1.0)
        arm = dict(
            f_cutoff_hz=f_cut,
            f_cutoff_vs_quote_realized_pct=(f_cut - cv11.F_CUTOFF_TE10)
            / cv11.F_CUTOFF_TE10 * 100.0,
            max_dev=float(dev.max()), mean_dev=float(dev.mean()),
            min_mag=float(mag.min()), max_mag=float(mag.max()),
            solve_s=dt, mag=mag.tolist(),
            freqs_hz=np.asarray(f_hz).tolist(),
        )
        out["arms"][label] = arm
        print(f"\n[{label}] cfg.f_cutoff = {f_cut / 1e9:.6f} GHz "
              f"({arm['f_cutoff_vs_quote_realized_pct']:+.3f}% vs the "
              f"quote-realized {cv11.F_CUTOFF_TE10 / 1e9:.6f} GHz)")
        print(f"[{label}] |S11| envelope [{arm['min_mag']:.4f}, "
              f"{arm['max_mag']:.4f}], max||S11|-1| = {arm['max_dev']:.4f}, "
              f"mean = {arm['mean_dev']:.4f}   ({dt:.1f} s)")

    a, b = out["arms"]["trim"], out["arms"]["no_trim"]
    trim_own = b["max_dev"] - a["max_dev"]      # removing the trim, this arm
    observed = b["max_dev"] - BASELINE_MAX_DEV  # baseline -> post, two changes
    core_own = observed - trim_own              # what the trim does not explain
    out["verdict"] = dict(
        baseline_max_dev=BASELINE_MAX_DEV,
        trim_removal_step=trim_own,
        baseline_to_post_step=observed,
        core_step=core_own,
        reading=("removing the trim IMPROVES this leg; the observed "
                 "degradation is the #931 core's")
        if trim_own < 0 else
        ("removing the trim degrades this leg"))
    print("\n" + "=" * 68)
    print(f"max||S11|-1| on THIS checkout:  trim {a['max_dev']:.4f}  ->  "
          f"no_trim {b['max_dev']:.4f}   (removing the trim: {trim_own:+.4f})")
    print(f"baseline 369367259004 (main d990e18c, trim): "
          f"{BASELINE_MAX_DEV:.4f}  ->  post {b['max_dev']:.4f}  "
          f"({observed:+.4f}, TWO variables)")
    print(f"so the #931 core accounts for {core_own:+.4f} and the trim's own "
          f"contribution is {trim_own:+.4f}")
    print(f"reading: {out['verdict']['reading']}")
    print("=" * 68)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
