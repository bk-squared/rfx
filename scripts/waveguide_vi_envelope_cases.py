"""Case-list generator for the waveguide `normalize=False` envelope sweep.

PRE-MEASUREMENT. This emits the case lists that
``docs/design_notes/waveguide_vi_envelope_sweep_predeclaration.md`` **revision 2**
binds — §1.3 (Stage 0), §4 (the R/F table) and §5 (rungs and the ceiling leg).
It runs no FDTD and decides no verdict.

    python scripts/waveguide_vi_envelope_cases.py <out_dir>

writes one JSON per stage, in the execution order of §10:

    01_stage0.json           S0-a, S0-b, S0-c — all three unconditional (§1.3)
    02_anchor_R5.json        R5 at 9/18/36; its N=72 pair IS S0-a (§4)
    03_interior.json         R3, R4, R6, R7
    04_low_bracket.json      R2, R1 in their declared thickness lanes, then R0
    05_ceiling.json          C-0, C-A, C-S, then the two C twins (§5.2)
    06_falsifiers.json       F1 (three variants), the run-length twins, b/a = 1/3
    07_falsifier_f64.json    F2 — its own file because x64 is process-global
    99_smoke_local.json      the two cases the local smoke test runs

A case appears in exactly ONE stage. Where §4 says a pair is shared — R5's N=72
pair is S0-a, R2's N=72 pair is S0-c — Stage 0 owns it and the R stage does not
re-emit it. ``main`` cross-checks the emitted set against ``DECLARED_CASES``,
built independently from the §4/§1.3/§5.2 tables, so a case cannot go missing in
that bookkeeping.

Declared choices this file makes that the pre-declaration leaves open, so they
are reviewable rather than buried:

* **Drive bandwidth.** Fractional bandwidth = 2.9 x (r_hi - r_lo) / r_centre,
  the rule the low-frequency archive used. For the four bands that archive
  actually ran, its own literal values are kept instead of the formula's (they
  differ by under 2 %), so those cases stay directly comparable with the prior
  evidence §3.4 quotes. R5 and everything derived from it use the committed
  fixture drive verbatim (f0 = 10.0 GHz, bandwidth = 0.5, the 17 committed
  bins) — §3.5 compares R5 against an archive measured on that drive.
* **Blade dimensions.** §5.2 pins the blade to the N=18 lattice but not its
  size. See ``waveguide_vi_envelope_sweep.py``: 2 cells thick (2.54 mm, one
  coarse cell), 6 cells wide (7.62 mm = a/3), full height, centred in x.
* **R6's bins.** §4 writes the band as [1.80 f_c, 0.999 x the rung's discrete
  TE20 cutoff] — a fixed bottom and a rung-dependent top. Implemented literally
  (``lock="te20_top"``), so at N=9 the band is [1.80, 1.9576] f_c and at N=72
  [1.80, 1.9974] f_c.
* **R7's bins.** §4's table column says "yes (TE20)" but §4.1 re-declares the
  band as [2.05, 2.18] and computes its TE01 margins (2.5 / 3.0 / 3.2 % at
  N = 18/36/72) from those FIXED numbers. Under a TE20 lock the N=18 top would
  land at 2.213 f_c, 0.9 % from that rung's discrete TE01 — breaking the margin
  the same paragraph declares. The arithmetic wins: R7's bins are fixed in f_c.
* **Which rung carries the C twins.** §5.2 constraint 3 fixes the BIN (1.001 x
  the discrete TE20 cutoff) but not the rung. N=36, following §4.1's own
  reasoning that a thickness axis is tested where it is cheapest.
* **The C domain-length twin's factor.** 2.0x the port-to-blade distance.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tests._waveguide_chain_battery_fixture as F        # noqa: E402
from waveguide_vi_envelope_sweep import (                  # noqa: E402
    DX_BY_N, FC_CONTINUOUS_HZ, bandwidth_for, sinc_te20,
)

N_TRAV = 4.0
N_TRAV_TWIN = 10.0
K_MAIN = 3.0
R5_R_LO = float(F.FREQS[0]) / FC_CONTINUOUS_HZ            # 1.28107
R5_R_HI = float(F.FREQS[-1]) / FC_CONTINUOUS_HZ           # 1.76908

# The literal drive bandwidths the low-frequency archive ran, kept so the
# sweep's cases at those bands stay comparable with the evidence §3.4 quotes.
BANDWIDTH_LITERAL = {
    (1.010, 1.030): 0.056,
    (1.017, 1.045): 0.0788,
    (1.023, 1.060): 0.1030,
    (1.030, 1.080): 0.14,
}

# §5.2: bins at these ratios of EACH RUNG'S OWN discrete TE20 cutoff.
C_TE20_RATIOS = [0.94, 0.96, 0.98, 0.99, 0.995, 0.999,
                 1.001, 1.005, 1.01, 1.02, 1.04, 1.06]
C_TWIN_RATIO = 1.001          # §5.2 constraint 3: the twins run at this bin
C_TWIN_N = 36
C_TWIN_K = 4.5
C_TWIN_DOMAIN_MULT = 2.0
C_R_LO, C_R_HI = 0.94 * 2.0, 1.06 * 2.0   # nominal, for layout and drive


def _bandwidth(r_lo: float, r_hi: float) -> float:
    return BANDWIDTH_LITERAL.get((round(r_lo, 6), round(r_hi, 6)),
                                 bandwidth_for(r_lo, r_hi))


def _lane(N: int, r_lo: float) -> str:
    """§8: anything with N >= 36 or r_lo < 1.03 runs on VESSL."""
    return "vessl" if (N >= 36 or r_lo < 1.03) else "local"


def _ktag(K: float) -> str:
    return ("K%.1f" % K).replace(".", "p")


def case(case_id: str, band_id: str, N: int, *, r_lo: float, r_hi: float,
         n_bins: int | None = None, lock: str = "none", K: float = K_MAIN,
         n_trav: float = N_TRAV, dut: str = "thru", precision: str = "float32",
         port_variant: str = "prod", b_over_a: float | None = None,
         domain_mult: float = 1.0, freqs_hz=None, f0_hz=None, bandwidth=None,
         te20_ratios=None, te20_ratio_lo=None, te20_ratio_hi=None,
         role: str = "") -> dict:
    if N not in DX_BY_N:
        raise ValueError(f"N={N} is not a ladder rung {sorted(DX_BY_N)}")
    dx = DX_BY_N[N]
    b_m = F.A_M * (b_over_a if b_over_a is not None else 4.0 / 9.0)
    b_cells = b_m / dx
    # §5.1 and the fixture's own rule: no case is emitted whose narrow wall
    # falls between nodes.
    if abs(b_cells - round(b_cells)) > 1e-9:
        raise AssertionError(
            f"{case_id}: b/dx = {b_cells} is not integral at N={N} "
            f"(b={b_m}, dx={dx}) — the guide would rasterize between nodes")
    c = dict(case_id=case_id, band_id=band_id, role=role, N=N, dx_m=dx,
             r_lo=r_lo, r_hi=r_hi, lock=lock, K=K, n_trav=n_trav, dut=dut,
             precision=precision, port_variant=port_variant,
             domain_mult=float(domain_mult),
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
    if te20_ratio_lo is not None:
        c["te20_ratio_lo"] = float(te20_ratio_lo)
    if te20_ratio_hi is not None:
        c["te20_ratio_hi"] = float(te20_ratio_hi)
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
# §4 — the R table, revision 2
# --------------------------------------------------------------------------
# ``lanes`` maps a rung to the thickness lanes it runs. R1 and R2 run three
# thicknesses at N=36 — the §0.5 discriminator, one guide wavelength apart —
# and two at N=72, where +1 lam_g costs ~175 s instead of ~7 s.
R_TABLE = {
    "R0": dict(r_lo=1.010, r_hi=1.030, n_bins=9, lock="te10",
               lanes={9: (3.0,), 18: (3.0,), 36: (3.0,)}, continuous_twin=True,
               role="out-of-scope, measured not asserted"),
    "R1": dict(r_lo=1.017, r_hi=1.045, n_bins=9, lock="te10",
               lanes={9: (3.0,), 18: (3.0,), 36: (3.0, 4.0, 5.0), 72: (3.0, 4.5)},
               runlen_twin_rungs=(9, 18),
               role="the declared-failure bracket"),
    "R2": dict(r_lo=1.023, r_hi=1.060, n_bins=9, lock="te10",
               lanes={9: (3.0,), 18: (3.0,), 36: (3.0, 4.0, 5.0), 72: ()},
               runlen_twin_rungs=(9, 18),
               role="expected lowest claiming point; its N=72 pair IS S0-c"),
    "R3": dict(r_lo=1.030, r_hi=1.080, n_bins=9, lock="te10",
               lanes={9: (3.0,), 18: (3.0,), 36: (3.0,), 72: (3.0,)},
               continuous_twin=True, role="near-cutoff anchor"),
    "R4": dict(r_lo=1.080, r_hi=1.160, n_bins=9, lock="none",
               lanes={9: (3.0,), 18: (3.0,), 36: (3.0,), 72: (3.0,)},
               role="bridges near-cutoff to the committed band"),
    # R6's top edge is the rung's own discrete TE20 cutoff x 0.999; its bottom
    # is fixed at 1.80 f_c. R7's bins are fixed in f_c (see the module
    # docstring: §4.1's TE01 margins are computed from the fixed numbers).
    "R6": dict(r_lo=1.80, r_hi=None, n_bins=9, lock="te20_top",
               te20_ratio_hi=0.999,
               lanes={9: (3.0,), 18: (3.0,), 36: (3.0,), 72: (3.0,)},
               role="upper interior, to the ceiling"),
    "R7": dict(r_lo=2.05, r_hi=2.18, n_bins=9, lock="none",
               lanes={18: (3.0,), 36: (3.0,), 72: (3.0,)},
               role="empty guide above the TE20 cutoff"),
}
# Owned by Stage 0, so the R stage must not re-emit them (§4). S0-a IS R5's
# N=72 rung, S0-c IS R2's N=72 pair, and S0-b's K=3.0 half IS R2's N=36 K=3.0
# lane — running any of them twice would be the same configuration under two
# ids, which reads as two independent measurements in a results directory.
STAGE0_OWNED = {("R5", 72, 3.0), ("R5", 72, 4.5),
                ("R2", 72, 3.0), ("R2", 72, 4.5), ("R2", 36, 3.0)}


def _r_case(band: str, N: int, K: float, *, lock=None, n_trav=N_TRAV,
            suffix="", role=None) -> dict:
    spec = R_TABLE[band]
    r_hi = spec["r_hi"]
    kw = {}
    if spec["lock"] == "te20_top":
        kw["te20_ratio_hi"] = spec["te20_ratio_hi"]
        # r_hi is rung-dependent; the nominal value only sizes the drive.
        r_hi = 2.0 * sinc_te20(9) * spec["te20_ratio_hi"]
    return case(f"{band}_N{N}_{_ktag(K)}{suffix}", band, N, r_lo=spec["r_lo"],
                r_hi=r_hi, n_bins=spec["n_bins"],
                lock=spec["lock"] if lock is None else lock, K=K,
                n_trav=n_trav, role=role or spec["role"], **kw)


def r_band_cases(band: str) -> list[dict]:
    spec = R_TABLE[band]
    out = []
    for N in sorted(spec["lanes"]):
        for K in spec["lanes"][N]:
            if (band, N, K) in STAGE0_OWNED:
                continue
            out.append(_r_case(band, N, K))
    if spec.get("continuous_twin"):
        for N in sorted(spec["lanes"]):
            if spec["lanes"][N]:
                out.append(_r_case(band, N, K_MAIN, lock="none", suffix="_cont",
                                   role="continuous-axis twin (§4)"))
    return out


def r5_cases() -> list[dict]:
    """R5 at 9/18/36. Its N=72 pair is S0-a and lives in Stage 0."""
    return [r5_case(f"R5_N{N}_{_ktag(K_MAIN)}", N, role="the anchor (§3.5)")
            for N in (9, 18, 36)]


def stage0_cases() -> list[dict]:
    """§1.3 — all three cases run, unconditionally, before any headline case.

    S0-a and S0-c double as the N=72 rungs of R5 and R2 (§4), so they are
    emitted once, here, and the R stages do not repeat them.
    """
    out = []
    for K in (3.0, 4.5):
        out.append(r5_case(f"S0a_R5_N72_{_ktag(K)}", 72, K=K,
                           role="S0-a, and R5's own N=72 rung"))
    spec = R_TABLE["R2"]
    for K in (3.0, 4.5):
        out.append(case(f"S0b_R2_N36_{_ktag(K)}", "R2", 36, r_lo=spec["r_lo"],
                        r_hi=spec["r_hi"], n_bins=spec["n_bins"],
                        lock=spec["lock"], K=K, role="S0-b"))
    for K in (3.0, 4.5):
        out.append(case(f"S0c_R2_N72_{_ktag(K)}", "R2", 72, r_lo=spec["r_lo"],
                        r_hi=spec["r_hi"], n_bins=spec["n_bins"],
                        lock=spec["lock"], K=K,
                        role="S0-c, and R2's own N=72 rung"))
    return out


def ceiling_cases() -> list[dict]:
    """§5.2 — C-0 first, then C-A and C-S together, then the two twins."""
    out = []
    for tag, dut in (("C0", "thru"), ("CA", "blade_offset"), ("CS", "blade_centred")):
        for N in (18, 36, 72):
            out.append(case(f"{tag}_N{N}_{_ktag(K_MAIN)}", tag, N, r_lo=C_R_LO,
                            r_hi=C_R_HI, lock="te20_ratio",
                            te20_ratios=C_TE20_RATIOS, K=K_MAIN, dut=dut,
                            role={"C0": "per-bin baseline for P_j",
                                  "CA": "brackets the ceiling, per bin",
                                  "CS": "attribution falsifier"}[tag]))
    # Constraint 3: both twins run at the SAME single bin, 1.001 x the rung's
    # own discrete TE20 cutoff, and each carries its own C-0 baseline — an
    # uncorrected twin compares against the wrong unity.
    #   thickness twin — reported in lam_g,TE20 at that bin, not lam_g,TE10
    #   domain-length twin — the independent axis a steady-state leak must move
    #                        and a true unaccounted-power ceiling must not
    for tag, dut in (("C0", "thru"), ("CA", "blade_offset")):
        out.append(case(f"{tag}_N{C_TWIN_N}_b1001_{_ktag(K_MAIN)}", tag, C_TWIN_N,
                        r_lo=C_R_LO, r_hi=C_R_HI, lock="te20_ratio",
                        te20_ratios=[C_TWIN_RATIO], K=K_MAIN, dut=dut,
                        role="C twin reference bin (§5.2.3)"))
        out.append(case(f"{tag}_N{C_TWIN_N}_b1001_{_ktag(C_TWIN_K)}", tag, C_TWIN_N,
                        r_lo=C_R_LO, r_hi=C_R_HI, lock="te20_ratio",
                        te20_ratios=[C_TWIN_RATIO], K=C_TWIN_K, dut=dut,
                        role="C thickness twin (§5.2.3)"))
        out.append(case(f"{tag}_N{C_TWIN_N}_b1001_{_ktag(K_MAIN)}_L2", tag, C_TWIN_N,
                        r_lo=C_R_LO, r_hi=C_R_HI, lock="te20_ratio",
                        te20_ratios=[C_TWIN_RATIO], K=K_MAIN, dut=dut,
                        domain_mult=C_TWIN_DOMAIN_MULT,
                        role="C domain-length twin (§5.2.3)"))
    return out


def falsifier_cases() -> list[dict]:
    """§4.2 F1, §4.1's run-length twins, and the §9 b/a control. Run last."""
    out = []
    spec = R_TABLE["R2"]
    # F1: three variants. A single variant that lowers `asy` establishes
    # nothing; `anti` is the reversed case the discriminator needs.
    for variant in ("prod", "plane", "anti"):
        out.append(case(f"F1_R2_N36_{variant}", "F1", 36, r_lo=spec["r_lo"],
                        r_hi=spec["r_hi"], n_bins=spec["n_bins"],
                        lock=spec["lock"], K=K_MAIN, port_variant=variant,
                        role=f"mechanism discriminator, variant {variant} (§4.2)"))
    # §4.1: the run-length twins sit at N = 9 and 18 of R1 and R2, where the
    # truncation effect on `asy` is largest and the case is nearly free.
    for band in ("R1", "R2"):
        for N in R_TABLE[band]["runlen_twin_rungs"]:
            out.append(_r_case(band, N, K_MAIN, n_trav=N_TRAV_TWIN, suffix="_t10",
                               role="run-length twin, n_trav=10 (§4.1)"))
    # §9.1: b/a = 1/3 puts fc_TE01 at 19.67 GHz, clear of everything, so b/a
    # moves alone. b/a = 1/2 is deliberately NOT used — there fc_TE01 = c/a =
    # fc_TE20 exactly and the two ceilings coincide.
    for N in (18, 36):
        out.append(r5_case(f"BA13_R5_N{N}", N, b_over_a=1.0 / 3.0,
                           role="b/a = 1/3 one-point control (§9.1)"))
    return out


def f64_cases() -> list[dict]:
    """§4.2 F2 — is the fine-rung floor precision. Its own file and its own
    process: an x64 flip is process-global and has contaminated shards before."""
    return [r5_case("F2_R5_N72_f64", 72, precision="float64",
                    role="precision floor vs port floor (§4.2 F2)")]


def smoke_cases() -> list[dict]:
    """The local smoke test: the smallest real case, and one near-cutoff case."""
    spec = R_TABLE["R1"]
    return [r5_case("SMOKE_R5_N9_K3p0", 9, role="smallest real case"),
            case("SMOKE_R1_N9_K3p0", "R1", 9, r_lo=spec["r_lo"], r_hi=spec["r_hi"],
                 n_bins=spec["n_bins"], lock=spec["lock"], K=K_MAIN,
                 role="near-cutoff case at the coarse rung")]


STAGES = {
    "01_stage0": stage0_cases,
    "02_anchor_R5": r5_cases,
    "03_interior": lambda: sum((r_band_cases(b) for b in ("R3", "R4", "R6", "R7")), []),
    "04_low_bracket": lambda: sum((r_band_cases(b) for b in ("R2", "R1", "R0")), []),
    "05_ceiling": ceiling_cases,
    "06_falsifiers": falsifier_cases,
    "07_falsifier_f64": f64_cases,
}


def declared_ladder_cases() -> set[tuple]:
    """(band, N, K) the §4 table and §1.3 declare, built independently of the
    emitters above so a case cannot go missing in the Stage-0 bookkeeping."""
    want = {("R5", N, 3.0) for N in (9, 18, 36, 72)} | {("R5", 72, 4.5)}
    for band, spec in R_TABLE.items():
        for N, Ks in spec["lanes"].items():
            for K in Ks:
                want.add((band, N, K))
    want |= {("R2", 72, 3.0), ("R2", 72, 4.5)}          # S0-c
    want |= {("R2", 36, 4.5)}                           # S0-b's control lane
    return want


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    emitted_ladder: set[tuple] = set()
    total = 0
    for stage, fn in STAGES.items():
        cases = fn()
        for c in cases:
            if c["case_id"] in seen:
                raise AssertionError(f"duplicate case id {c['case_id']}")
            seen.add(c["case_id"])
            # Only plain ladder cases count toward the completeness check;
            # twins, variants and controls are extra by construction.
            if (c["band_id"] not in ("C0", "CA", "CS", "F1")
                    and c["n_trav"] == N_TRAV and c["port_variant"] == "prod"
                    and c["precision"] == "float32" and c["dut"] == "thru"
                    and c["domain_mult"] == 1.0 and "b_over_a" not in c
                    and not c["case_id"].endswith("_cont")):
                band = "R5" if c["band_id"] == "R5" else c["band_id"]
                emitted_ladder.add((band, c["N"], c["K"]))
        (out / f"{stage}.json").write_text(json.dumps(cases, indent=1))
        n_vessl = sum(1 for c in cases if c["lane"] == "vessl")
        print(f"{stage:20s} {len(cases):3d} cases  ({n_vessl} vessl, "
              f"{len(cases) - n_vessl} local)")
        total += len(cases)

    (out / "99_smoke_local.json").write_text(json.dumps(smoke_cases(), indent=1))
    print(f"{'99_smoke_local':20s} {len(smoke_cases()):3d} cases  (smoke, not part of the sweep)")

    want = declared_ladder_cases()
    missing = want - emitted_ladder
    extra = emitted_ladder - want
    if missing or extra:
        raise AssertionError(
            f"case list does not match the §4/§1.3 tables — missing={sorted(missing)} "
            f"extra={sorted(extra)}")
    print(f"completeness: {len(want)} declared (band, N, K) ladder cases, all emitted once")
    print(f"TOTAL {total} sweep cases -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
