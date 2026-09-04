"""Case-list generator for the waveguide `normalize=False` envelope sweep.

PRE-MEASUREMENT. This emits the case lists that
``docs/design_notes/waveguide_vi_envelope_sweep_predeclaration.md`` §4 (the R/F
table), §5 (rungs and the C leg) and §1.3 (Stage 0) bind. It runs no FDTD and
decides no verdict.

    python scripts/waveguide_vi_envelope_cases.py <out_dir>

writes one JSON per stage, in the execution order of §10:

    00_preflight_note.json   (a marker; the zero-FDTD artifacts are a script)
    01_stage0.json           S0-a, S0-b        (§1.3)
    01_stage0c.json          S0-b's escalation, run ONLY if S0-b fails
    02_anchor_R5.json        R5, four rungs    (§3.5 falsifier, sets the M2 bar)
    03_interior.json         R3, R4, R6, R7
    04_low_bracket.json      R2, R1 (two absorber lanes), R0
    05_ceiling.json          C-0, then C-A and C-S, plus the C-leg twins
    06_falsifiers.json       R5-X, F1, the b/a = 1/3 control
    07_falsifier_f64.json    F2 — its own file because x64 is process-global
    99_smoke_local.json      the two cases the local smoke test runs

Declared choices this file makes that the pre-declaration leaves open, so they
are reviewable rather than buried:

* **Drive bandwidth.** Fractional bandwidth = 2.9 x (r_hi - r_lo) / r_centre,
  the rule the low-r scouting used. For the four bands that scouting actually
  ran, its own literal values are kept instead of the formula's (they differ by
  under 2 %), so those cases stay directly comparable with the prior evidence
  the pre-declaration §0/§3.4 quotes. R5 and everything derived from it use the
  committed fixture drive verbatim (f0 = 10.0 GHz, bandwidth = 0.5, the 17
  committed bins) — §3.5 compares R5 against CHECK 1's ladder, which was
  measured on that drive, so R5 cannot use a different one.
* **Blade dimensions.** §5.2 pins the blade to the N=18 lattice but not its
  size. See ``waveguide_vi_envelope_sweep.py``: 2 cells thick (2.54 mm, one
  coarse cell), 6 cells wide (7.62 mm = a/3), full height, centred in x.
* **R7's band.** §4 writes "[2.05, 2.40] rel. discrete TE20". Read as f/f_c
  with the TE20 lock, matching R6 — the alternative reading (2.05 x the TE20
  cutoff) lands at 4.1 f_c, far outside the sweep.
* **Absorber f_low.** §1.1 says "the band's OWN low edge". Taken as the band's
  NOMINAL low edge so one band has one physical absorber across its ladder;
  each record also carries the exactly-scaled 1x/2x/4x layer count.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.environ.get("RFX_WT") or str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tests._waveguide_chain_battery_fixture as F        # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from waveguide_vi_envelope_sweep import (                  # noqa: E402
    DX_BY_N, FC_CONTINUOUS_HZ, bandwidth_for,
)

N_TRAV = 4.0
N_TRAV_TWIN = 10.0
K_MAIN = 3.0
K_THIN = 1.5
R5_R_LO = float(F.FREQS[0]) / FC_CONTINUOUS_HZ            # 1.28107
R5_R_HI = float(F.FREQS[-1]) / FC_CONTINUOUS_HZ           # 1.76908

# The literal drive bandwidths the low-r scouting ran, kept so the sweep's
# cases at those bands stay comparable with the evidence §3.4 quotes.
BANDWIDTH_LITERAL = {
    (1.010, 1.030): 0.056,
    (1.017, 1.045): 0.0788,
    (1.023, 1.060): 0.1030,
    (1.030, 1.080): 0.14,
}

# §5.2: bins at these ratios of EACH RUNG'S OWN discrete TE20 cutoff.
C_TE20_RATIOS = [0.94, 0.96, 0.98, 0.99, 0.995, 0.999,
                 1.001, 1.005, 1.01, 1.02, 1.04, 1.06]


def _bandwidth(r_lo: float, r_hi: float) -> float:
    return BANDWIDTH_LITERAL.get((round(r_lo, 6), round(r_hi, 6)),
                                 bandwidth_for(r_lo, r_hi))


def _lane(N: int, r_lo: float) -> str:
    """§8: anything with N >= 36 or r_lo < 1.03 goes to VESSL."""
    return "vessl" if (N >= 36 or r_lo < 1.03) else "local"


def _ktag(K: float) -> str:
    return ("K%.1f" % K).replace(".", "p")


def case(case_id: str, band_id: str, N: int, *, r_lo: float, r_hi: float,
         n_bins: int | None = None, lock: str = "none", K: float = K_MAIN,
         n_trav: float = N_TRAV, dut: str = "thru", precision: str = "float32",
         port_variant: str = "shipped", b_over_a: float | None = None,
         freqs_hz=None, f0_hz=None, bandwidth=None, te20_ratios=None,
         role: str = "") -> dict:
    if N not in DX_BY_N:
        raise ValueError(f"N={N} is not a ladder rung {sorted(DX_BY_N)}")
    dx = DX_BY_N[N]
    b_m = F.A_M * (b_over_a if b_over_a is not None else 4.0 / 9.0)
    b_cells = b_m / dx
    # Pre-declaration §5.1 / the fixture's own rule: no case is emitted whose
    # narrow wall falls between nodes.
    if abs(b_cells - round(b_cells)) > 1e-9:
        raise AssertionError(
            f"{case_id}: b/dx = {b_cells} is not integral at N={N} "
            f"(b={b_m}, dx={dx}) — the guide would rasterize between nodes")
    c = dict(case_id=case_id, band_id=band_id, role=role, N=N, dx_m=dx,
             r_lo=r_lo, r_hi=r_hi, lock=lock, K=K, n_trav=n_trav, dut=dut,
             precision=precision, port_variant=port_variant,
             bandwidth=float(bandwidth if bandwidth is not None
                             else _bandwidth(r_lo, r_hi)),
             lane=_lane(N, r_lo), b_cells=int(round(b_cells)))
    if n_bins is not None:
        c["n_bins"] = int(n_bins)
    if b_over_a is not None:
        c["b_over_a"] = float(b_over_a)
    if freqs_hz is not None:
        c["freqs_hz"] = [float(v) for v in freqs_hz]
    if f0_hz is not None:
        c["f0_hz"] = float(f0_hz)
    if te20_ratios is not None:
        c["te20_ratios"] = [float(v) for v in te20_ratios]
    return c


def r5_case(case_id: str, N: int, *, K: float = K_MAIN, n_trav: float = N_TRAV,
            precision: str = "float32", b_over_a: float | None = None,
            role: str = "") -> dict:
    """R5 and everything derived from it: the COMMITTED band, verbatim."""
    return case(case_id, "R5", N, r_lo=R5_R_LO, r_hi=R5_R_HI, lock="none", K=K,
                n_trav=n_trav, precision=precision, b_over_a=b_over_a,
                freqs_hz=F.FREQS, f0_hz=F.F0_HZ, bandwidth=F.BANDWIDTH,
                role=role)


# --------------------------------------------------------------------------
# §4 — the R table
# --------------------------------------------------------------------------
R_TABLE = {
    "R0": dict(r_lo=1.010, r_hi=1.030, n_bins=9, lock="te10", rungs=(9, 18, 36),
               Ks=(K_MAIN,), role="out-of-scope, measured rather than asserted"),
    "R1": dict(r_lo=1.017, r_hi=1.045, n_bins=9, lock="te10",
               rungs=(9, 18, 36, 72), Ks=(K_MAIN, K_THIN),
               role="the declared-failure bracket (§3.4)"),
    "R2": dict(r_lo=1.023, r_hi=1.060, n_bins=9, lock="te10",
               rungs=(9, 18, 36, 72), Ks=(K_MAIN, K_THIN),
               role="the expected lowest claiming point"),
    "R3": dict(r_lo=1.030, r_hi=1.080, n_bins=9, lock="te10",
               rungs=(9, 18, 36, 72), Ks=(K_MAIN,), continuous_twin=True,
               role="near-cutoff anchor"),
    "R4": dict(r_lo=1.080, r_hi=1.160, n_bins=9, lock="none",
               rungs=(9, 18, 36, 72), Ks=(K_MAIN,),
               role="bridges near-cutoff to the committed band"),
    "R6": dict(r_lo=1.80, r_hi=1.95, n_bins=9, lock="te20",
               rungs=(9, 18, 36, 72), Ks=(K_MAIN,), role="upper interior"),
    "R7": dict(r_lo=2.05, r_hi=2.40, n_bins=9, lock="te20",
               rungs=(18, 36, 72), Ks=(K_MAIN,),
               role="empty guide above the TE20 cutoff"),
}


def r_band_cases(band: str) -> list[dict]:
    spec = R_TABLE[band]
    out = []
    finest = max(spec["rungs"])
    for K in spec["Ks"]:
        for N in spec["rungs"]:
            out.append(case(f"{band}_N{N}_{_ktag(K)}", band, N,
                            r_lo=spec["r_lo"], r_hi=spec["r_hi"],
                            n_bins=spec["n_bins"], lock=spec["lock"], K=K,
                            role=spec["role"]))
        # §4: the n_trav = 10 truncation twin sits at the band's FINEST rung,
        # never at a coarse one — the twin's effect grows with rung.
        out.append(case(f"{band}_N{finest}_{_ktag(K)}_t10", band, finest,
                        r_lo=spec["r_lo"], r_hi=spec["r_hi"],
                        n_bins=spec["n_bins"], lock=spec["lock"], K=K,
                        n_trav=N_TRAV_TWIN, role="run-length twin (§4)"))
    if spec.get("continuous_twin"):
        for N in spec["rungs"]:
            out.append(case(f"{band}_N{N}_{_ktag(K_MAIN)}_cont", band, N,
                            r_lo=spec["r_lo"], r_hi=spec["r_hi"],
                            n_bins=spec["n_bins"], lock="none", K=K_MAIN,
                            role="continuous-axis twin (§4)"))
    return out


def r5_cases() -> list[dict]:
    out = [r5_case(f"R5_N{N}_{_ktag(K_MAIN)}", N, role="the anchor (§3.5)")
           for N in (9, 18, 36, 72)]
    out.append(r5_case(f"R5_N72_{_ktag(K_MAIN)}_t10", 72, n_trav=N_TRAV_TWIN,
                       role="run-length twin (§4)"))
    return out


def stage0_cases() -> list[dict]:
    """§1.3 — must pass before any headline case runs."""
    out = [r5_case("S0a_R5_N72_K3p0", 72, K=3.0, role="S0-a K=3.0"),
           r5_case("S0a_R5_N72_K4p5", 72, K=4.5, role="S0-a K=4.5 control")]
    spec = R_TABLE["R2"]
    for K in (3.0, 4.5):
        out.append(case(f"S0b_R2_N36_{_ktag(K)}", "R2", 36, r_lo=spec["r_lo"],
                        r_hi=spec["r_hi"], n_bins=spec["n_bins"],
                        lock=spec["lock"], K=K, role=f"S0-b K={K}"))
    return out


def stage0c_cases() -> list[dict]:
    """§1.3 escalation — run ONLY if S0-b fails. If this fails too, K becomes
    band-dependent and is re-derived before any headline case runs."""
    spec = R_TABLE["R2"]
    return [case(f"S0c_R2_N72_{_ktag(K)}", "R2", 72, r_lo=spec["r_lo"],
                 r_hi=spec["r_hi"], n_bins=spec["n_bins"], lock=spec["lock"],
                 K=K, role=f"S0-c K={K}") for K in (3.0, 4.5)]


def ceiling_cases() -> list[dict]:
    """§5.2 — C-0 first, then C-A and C-S together. Never C-A alone."""
    out = []
    r_lo, r_hi = 0.94 * 2.0, 1.06 * 2.0        # nominal, for layout and drive
    for tag, dut in (("C0", "thru"), ("CA", "blade_offset"), ("CS", "blade_centred")):
        for N in (18, 36, 72):
            out.append(case(f"{tag}_N{N}_{_ktag(K_MAIN)}", tag, N, r_lo=r_lo,
                            r_hi=r_hi, lock="te20_ratio", te20_ratios=C_TE20_RATIOS,
                            K=K_MAIN, dut=dut,
                            role={"C0": "per-bin baseline for P_j",
                                  "CA": "brackets the ceiling, per bin",
                                  "CS": "attribution falsifier"}[tag]))
    # §5.2 constraint 3: the C leg gets its OWN thickness twin and its OWN
    # run-length twin. At 0.999-1.02 of the TE20 cutoff that mode is evanescent
    # or has v_g -> 0, so an absorber sized on lam_g(TE10) is effectively no
    # absorber for the mode that carries the finding.
    out.append(case(f"CA_N36_{_ktag(4.5)}", "CA", 36, r_lo=r_lo, r_hi=r_hi,
                    lock="te20_ratio", te20_ratios=C_TE20_RATIOS, K=4.5,
                    dut="blade_offset", role="C-leg thickness twin (§5.2.3)"))
    out.append(case(f"CA_N36_{_ktag(K_MAIN)}_t10", "CA", 36, r_lo=r_lo, r_hi=r_hi,
                    lock="te20_ratio", te20_ratios=C_TE20_RATIOS, K=K_MAIN,
                    n_trav=N_TRAV_TWIN, dut="blade_offset",
                    role="C-leg run-length twin (§5.2.3)"))
    out.append(case(f"C0_N36_{_ktag(4.5)}", "C0", 36, r_lo=r_lo, r_hi=r_hi,
                    lock="te20_ratio", te20_ratios=C_TE20_RATIOS, K=4.5,
                    dut="thru", role="baseline for the thickness twin"))
    return out


def falsifier_cases() -> list[dict]:
    """§4 — R5-X, F1, and the b/a = 1/3 control. Run last."""
    out = []
    # R5-X: two absorbers at both rungs. At a fixed K the leak's share grows at
    # the fine rung, so an under-absorbed K=3.5 at N=144 would produce exactly
    # the publishable-looking failure R5-X exists to test; K=4.5 is the control.
    for K in (3.5, 4.5):
        for N in (72, 144):
            out.append(r5_case(f"R5X_N{N}_{_ktag(K)}", N, K=K,
                               role="does second order continue past a/72"))
    # F1: the mechanism falsifier — shipped vs the one-cell E-plane instrument.
    spec = R_TABLE["R2"]
    for variant, tag in (("shipped", "shipped"), ("instrumented_e_plane", "instr")):
        out.append(case(f"F1_R2_N36_{tag}", "F1", 36, r_lo=spec["r_lo"],
                        r_hi=spec["r_hi"], n_bins=spec["n_bins"],
                        lock=spec["lock"], K=K_MAIN, port_variant=variant,
                        role="mechanism falsifier (§4 F1)"))
    # b/a control (§9.1): 1/3 puts fc_TE01 at 19.67 GHz, clear of everything, so
    # b/a moves alone. b/a = 1/2 is deliberately NOT used — there
    # fc_TE01 = c/a = fc_TE20 exactly and the two ceilings coincide.
    for N in (18, 36):
        out.append(r5_case(f"BA13_R5_N{N}", N, b_over_a=1.0 / 3.0,
                           role="b/a = 1/3 one-point control (§9.1)"))
    return out


def f64_cases() -> list[dict]:
    """§4 F2 — is the fine-rung `asy` floor precision or physics. Its own file:
    jax_enable_x64 is process-global."""
    return [r5_case("F2_R5_N72_f64", 72, precision="float64",
                    role="precision floor vs absorber floor (§4 F2)")]


def smoke_cases() -> list[dict]:
    """The local smoke test: the smallest real case, and one near-cutoff case."""
    out = [r5_case("SMOKE_R5_N9_K3p0", 9, role="smallest real case")]
    spec = R_TABLE["R1"]
    out.append(case("SMOKE_R1_N9_K3p0", "R1", 9, r_lo=spec["r_lo"],
                    r_hi=spec["r_hi"], n_bins=spec["n_bins"], lock=spec["lock"],
                    K=K_MAIN, role="near-cutoff case at the coarse rung"))
    return out


STAGES = {
    "01_stage0": stage0_cases,
    "01_stage0c": stage0c_cases,
    "02_anchor_R5": r5_cases,
    "03_interior": lambda: sum((r_band_cases(b) for b in ("R3", "R4", "R6", "R7")), []),
    "04_low_bracket": lambda: sum((r_band_cases(b) for b in ("R2", "R1", "R0")), []),
    "05_ceiling": ceiling_cases,
    "06_falsifiers": falsifier_cases,
    "07_falsifier_f64": f64_cases,
    "99_smoke_local": smoke_cases,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0
    for stage, fn in STAGES.items():
        cases = fn()
        for c in cases:
            if c["case_id"] in seen and stage not in ("99_smoke_local",):
                raise AssertionError(f"duplicate case id {c['case_id']}")
            seen.add(c["case_id"])
        (out / f"{stage}.json").write_text(json.dumps(cases, indent=1))
        n_vessl = sum(1 for c in cases if c["lane"] == "vessl")
        print(f"{stage:20s} {len(cases):3d} cases  ({n_vessl} vessl, "
              f"{len(cases) - n_vessl} local)")
        total += len(cases)
    print(f"TOTAL {total} cases -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
