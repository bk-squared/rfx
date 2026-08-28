"""Pre-registered scoring metric for the Phase-2 dual-band notch benchmark.

FROZEN 2026-08-27, BEFORE any optimization run. Anything computed with a
modified threshold, band edge, or aggregation rule is POST HOC and must be
labelled as such in every place it is reported.

Why this file exists
--------------------
Phase-1's headline was retracted (NOTE_xval1_verdict.md) for two reasons, and
the second one is a metric failure: the objective was ``|S21(f0)|^2`` at a
SINGLE frequency, which cannot distinguish a notch from a brick and cannot see
a bandwidth at all. This module fixes the metric before the experiments run.

The spec
--------
Through-line on 30 mm of microstrip (dx=127 um, h=254 um, w=600 um,
eps_eff=2.87, F_MAX=9 GHz), all design metal inside a 12 x 9 mm box:

  * reject 5.150-5.350 GHz (WLAN lower, 200 MHz) by >= 20 dB ACROSS the band
  * reject 5.725-5.825 GHz (WLAN upper, 100 MHz) by >= 20 dB ACROSS the band
  * keep the two rejections SEPARATE (transmission recovers between them)
  * preserve the passband elsewhere

Everything below is expressed on INSERTION LOSS relative to the empty line,

    IL(f) = 20*log10|S21_empty(f)| - 20*log10|S21_dut(f)|      [dB, positive
                                                                = attenuation]

extracted by the imperative ``Simulation.compute_msl_s_matrix`` on hard-PEC
``Box`` geometry (the tier-1 independent path), never from the differentiable
operator.

Run the self-test:  python research/metal_to/score_dualband.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

# ---------------------------------------------------------------------------
# 1. Frequency plan  (integer MHz throughout -- float band edges silently drop
#    boundary samples, which was observed while calibrating this metric)
# ---------------------------------------------------------------------------
# Verification window: 90 periods at F_MAX = 9 GHz -> T = 10.0 ns,
# DFT resolution 1/T = 100 MHz.  Base sampling is 50 MHz = 2 samples per
# resolution cell (the Nyquist rule for a record-limited spectrum: no feature
# the record can resolve can hide between two samples).  The three critical
# segments are sampled at 25 MHz (4 samples/cell) -- pure oversampling, no new
# information, but it makes the worst-case (min/max) extraction safe.
#
# HARD LIMIT ON INTERPRETATION: nothing narrower than 100 MHz may be claimed,
# and no notch centre may be quoted to better than +-50 MHz, whatever the
# sample spacing suggests.

F_BASE_STEP_MHZ = 50
F_FINE_STEP_MHZ = 25
F_LO_MHZ, F_HI_MHZ = 3100, 8600

BAND_L_MHZ = (5150, 5350)      # WLAN lower, 200 MHz
BAND_U_MHZ = (5725, 5825)      # WLAN upper, 100 MHz
GUARD_MHZ = 100                # = one verification resolution cell

# derived, but written out so the file is readable without running it
GAP_MHZ = (5450, 5625)         # inter-band, guards removed  (175 MHz)
PASS_LO_MHZ = (3100, 5050)     # 1.95 GHz
PASS_HI_MHZ = (5925, 8600)     # 2.675 GHz
# unscored transition zones: 5050-5150, 5350-5450, 5625-5725, 5825-5925


def scoring_grid_mhz() -> np.ndarray:
    """The 123-point verification grid. THIS list is the pre-registered one."""
    pts = set(range(F_LO_MHZ, F_HI_MHZ + 1, F_BASE_STEP_MHZ))
    for lo, hi in (BAND_L_MHZ, GAP_MHZ, BAND_U_MHZ):
        pts |= set(range(lo, hi + 1, F_FINE_STEP_MHZ))
    pts |= {PASS_LO_MHZ[1], GAP_MHZ[0], GAP_MHZ[1], PASS_HI_MHZ[0]}
    return np.array(sorted(pts), dtype=int)


def descent_grid_mhz() -> np.ndarray:
    """Reduced 68-point grid for the 45-period descent window (res 200 MHz).

    Base step 100 MHz = 2 samples/cell at that window. NEVER used to report a
    number: it exists only to make each optimizer iteration cheap.
    """
    pts = set(range(F_LO_MHZ, F_HI_MHZ + 1, 100))
    pts |= set(range(*BAND_L_MHZ, 50)) | {BAND_L_MHZ[1]}
    pts |= set(range(*GAP_MHZ, 50)) | {GAP_MHZ[1]}
    pts |= set(range(BAND_U_MHZ[0], BAND_U_MHZ[1] + 1, 25))
    pts |= {PASS_LO_MHZ[1], PASS_HI_MHZ[0]}
    return np.array(sorted(pts), dtype=int)


# ---------------------------------------------------------------------------
# 2. Thresholds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    r_req_db: float = 20.0      # required rejection ACROSS each stopband
    r_cap_db: float = 25.0      # IL is clipped here before ANY aggregation
    il_gap_db: float = 10.0     # max loss allowed in the inter-band gap
                                # (10 dB = the UWB literature's own notch-band
                                #  boundary; below it the gap is not notched)
    il_pass_db: float = 1.0     # passband insertion-loss allowance
    term_cap_db: float = 20.0   # each of the four terms saturates here
    rl_pass_db: float = -10.0   # passband return-loss gate (not part of M)


SCORE = Thresholds()                       # M  -- the pre-registered scalar
RELAXED = Thresholds(r_req_db=15.0, r_cap_db=20.0, il_gap_db=13.0,
                     il_pass_db=2.0, rl_pass_db=-8.0)   # M_relaxed -- arm-D
                                                        # benchmark-kill gate


# ---------------------------------------------------------------------------
# 3. Validity gates -- a number that fails these is NOT QUOTABLE
# ---------------------------------------------------------------------------
SETTLING_MAX_DB = -40.0       # rfx ring-down witness (project rule)
PASSIVITY_MAX = 0.05          # max singular-value clip applied
EMPTY_CAL_MAX_DB = 0.10       # |IL_empty| tolerance on the scoring grid
                              # (measured floor on this fixture: 0.011 dB)

# Anti-degeneracy: a broadband blocker is not a filter and is not ranked at
# all, whatever M says. The bound is deliberately far above anything a
# plausible filter reaches (a merged single 25 dB notch measures ~9.6 dB here;
# a solid brick measures 25 dB, the clip).
DEGENERATE_IL_PASS_MEAN_DB = 12.0


@dataclass
class Validity:
    settled: bool
    settling_worst_db: float
    passivity_worst: float
    unreliable_unsaturated: list = field(default_factory=list)
    empty_cal_max_db: float | None = None
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.settled
                and self.passivity_worst <= PASSIVITY_MAX
                and not self.unreliable_unsaturated
                and (self.empty_cal_max_db is None
                     or self.empty_cal_max_db <= EMPTY_CAL_MAX_DB))


def check_validity(settling_db, passivity_correction, reliable, freqs_mhz,
                   il_clipped, thr: Thresholds = SCORE,
                   empty_cal_max_db: float | None = None) -> Validity:
    """Apply the three run-level gates.

    ``reliable`` is rfx's per-(port, bin) wave-split mask. It goes False at
    deep nulls BY DESIGN -- at a -40 dB notch the passive port's V/I split is
    genuinely low-signal and the extractor refuses to certify the depth. That
    is exactly why IL is clipped at ``r_cap_db`` first: a flagged bin whose
    clipped IL already sits at the cap contributes nothing further to the
    score, so its uncertainty cannot move a ranking. A flagged bin that is
    NOT saturated is a real problem (a collapsed plane somewhere the score
    depends on) and invalidates the run.
    """
    settling = np.asarray(settling_db, dtype=float).ravel()
    worst = float(np.max(settling)) if settling.size else float("nan")
    pcorr = np.asarray(passivity_correction, dtype=float).ravel()
    pworst = float(np.max(pcorr)) if pcorr.size else 0.0
    bad = []
    if reliable is not None:
        bin_ok = np.all(np.asarray(reliable, dtype=bool), axis=0).ravel()
        unsat = il_clipped < thr.r_cap_db - 1e-9
        bad = [int(f) for f, ok, u in zip(freqs_mhz, bin_ok, unsat)
               if u and not ok]
    return Validity(settled=bool(settling.size and worst <= SETTLING_MAX_DB),
                    settling_worst_db=worst, passivity_worst=pworst,
                    unreliable_unsaturated=bad,
                    empty_cal_max_db=empty_cal_max_db)


# ---------------------------------------------------------------------------
# 4. The metric
# ---------------------------------------------------------------------------
def _mask(f_mhz, lo, hi):
    return (f_mhz >= lo) & (f_mhz <= hi)


def _seg_mean_excess(f_mhz, y_db, segments, allowance):
    """Bandwidth-weighted mean of max(0, IL - allowance) over segments.

    Trapezoidal in frequency and divided by total bandwidth, so the value does
    NOT depend on how densely each segment happens to be sampled.
    """
    num = den = 0.0
    for lo, hi in segments:
        m = _mask(f_mhz, lo, hi)
        if m.sum() < 2:
            continue
        x = f_mhz[m].astype(float)
        e = np.maximum(0.0, y_db[m] - allowance)
        # numpy>=2.0 renamed trapz -> trapezoid; the deployment container
        # (nvcr jax:24.10) still ships numpy<2. Same function, same result.
        _trap = getattr(np, "trapezoid", None) or np.trapz
        num += float(_trap(e, x))
        den += float(x[-1] - x[0])
    return num / den if den > 0 else 0.0


def _bw20(f_mhz, il_db, band, level_db):
    """Contiguous bandwidth around the band's worst point at >= level_db.

    Reported only; quantised to the 100 MHz record resolution when quoted.
    """
    m = _mask(f_mhz, *band)
    if not m.any():
        return 0.0
    i0 = int(np.argmin(np.where(m, il_db, np.inf)))
    ok = il_db >= level_db
    if not ok[i0]:
        return 0.0
    i, j = i0, i0
    while i > 0 and ok[i - 1]:
        i -= 1
    while j < len(f_mhz) - 1 and ok[j + 1]:
        j += 1
    return float(f_mhz[j] - f_mhz[i])


@dataclass
class Result:
    M: float
    S_L: float
    S_U: float
    S_G: float
    S_P: float
    Omega: float
    R_L: float
    R_U: float
    R_L_raw: float
    R_U_raw: float
    BW20_L_MHz: float
    BW20_U_MHz: float
    IL_gap_max: float
    IL_pass_max: float
    IL_pass_mean: float
    frac_pass_within_allowance: float
    f_notch_L_MHz: float
    f_notch_U_MHz: float
    RL_pass_worst_db: float | None
    A_rad_pass_max: float | None
    A_rad_L: float | None
    A_rad_U: float | None
    spec_pass: bool
    degenerate: bool
    thresholds: dict
    validity: dict | None = None

    def as_dict(self):
        return asdict(self)


def score(freqs_mhz, il_db, s11_db=None, s21_db_abs=None,
          thr: Thresholds = SCORE,
          band_l=BAND_L_MHZ, band_u=BAND_U_MHZ, guard_mhz=GUARD_MHZ,
          f_lo=F_LO_MHZ, f_hi=F_HI_MHZ, validity: Validity | None = None):
    """Compute M and the reported sub-metrics.

    Parameters
    ----------
    freqs_mhz : integer MHz, ascending -- must be the pre-registered grid
    il_db     : insertion loss vs the empty line, positive = attenuation
    s11_db    : optional, for the passband return-loss gate
    s21_db_abs: optional absolute |S21| in dB, for the radiation witness
                A = 1 - |S11|^2 - |S21|^2
    band_l/band_u : (lo, hi) MHz -- parameterised so arm E can reuse this
                    unchanged at the 2.4/5.8 control ratio

    Returns
    -------
    Result. ``M`` is the ranking scalar, LOWER IS BETTER, units dB,
    0 = meets the whole mask.
    """
    f = np.asarray(freqs_mhz, dtype=int)
    il = np.minimum(np.asarray(il_db, dtype=float), thr.r_cap_db)

    gap = (band_l[1] + guard_mhz, band_u[0] - guard_mhz)
    p_lo = (f_lo, band_l[0] - guard_mhz)
    p_hi = (band_u[1] + guard_mhz, f_hi)
    # If the two bands are far apart (arm E: 2.4 / 5.8 GHz), merging is not a
    # physical risk. Fold the gap into the ordinary passband and drop S_G.
    gap_wide = (gap[1] - gap[0]) > 1000
    pass_segs = [p_lo, p_hi] + ([gap] if gap_wide else [])

    mL, mU = _mask(f, *band_l), _mask(f, *band_u)
    if mL.sum() < 2 or mU.sum() < 2:
        raise ValueError("scoring grid does not sample both stopbands")

    R_L, R_U = float(il[mL].min()), float(il[mU].min())
    R_L_raw = float(np.asarray(il_db)[mL].min())
    R_U_raw = float(np.asarray(il_db)[mU].min())

    S_L = max(0.0, thr.r_req_db - R_L)
    S_U = max(0.0, thr.r_req_db - R_U)

    if gap_wide:
        S_G, gap_max = 0.0, float("nan")
    else:
        mG = _mask(f, *gap)
        gap_max = float(il[mG].max())
        S_G = min(max(0.0, gap_max - thr.il_gap_db), thr.term_cap_db)

    S_P = min(_seg_mean_excess(f, il, pass_segs, thr.il_pass_db),
              thr.term_cap_db)

    M = S_L + S_U + S_G + S_P

    mP = np.zeros_like(f, dtype=bool)
    for lo, hi in pass_segs:
        mP |= _mask(f, lo, hi)
    il_p = il[mP]
    IL_pass_max = float(il_p.max())
    IL_pass_mean = _seg_mean_excess(f, il, pass_segs, 0.0)
    frac_ok = float((il_p <= thr.il_pass_db).mean())
    q95 = float(np.percentile(il_p, 95))

    margins = [R_L - thr.r_req_db, R_U - thr.r_req_db, thr.il_pass_db - q95]
    if not gap_wide:
        margins.append(thr.il_gap_db - gap_max)
    Omega = float(min(margins))

    RL_worst = float(np.asarray(s11_db)[mP].max()) if s11_db is not None else None

    A_p = A_L = A_U = None
    if s11_db is not None and s21_db_abs is not None:
        a = 1.0 - 10 ** (np.asarray(s11_db) / 10.0) - 10 ** (np.asarray(s21_db_abs) / 10.0)
        A_p, A_L, A_U = float(a[mP].max()), float(a[mL].max()), float(a[mU].max())

    spec = (M <= 1e-9) and (RL_worst is None or RL_worst <= thr.rl_pass_db)
    degenerate = bool(IL_pass_mean > DEGENERATE_IL_PASS_MEAN_DB)

    return Result(
        M=M, S_L=S_L, S_U=S_U, S_G=S_G, S_P=S_P, Omega=Omega,
        R_L=R_L, R_U=R_U, R_L_raw=R_L_raw, R_U_raw=R_U_raw,
        BW20_L_MHz=_bw20(f, il, band_l, thr.r_req_db),
        BW20_U_MHz=_bw20(f, il, band_u, thr.r_req_db),
        IL_gap_max=gap_max, IL_pass_max=IL_pass_max,
        IL_pass_mean=IL_pass_mean, frac_pass_within_allowance=frac_ok,
        # notch centres from the UNCLIPPED trace (clipping ties every deep
        # bin at the cap); quote to +-50 MHz, never finer
        f_notch_L_MHz=float(f[mL][int(np.argmax(np.asarray(il_db)[mL]))]),
        f_notch_U_MHz=float(f[mU][int(np.argmax(np.asarray(il_db)[mU]))]),
        RL_pass_worst_db=RL_worst, A_rad_pass_max=A_p, A_rad_L=A_L, A_rad_U=A_U,
        spec_pass=bool(spec), degenerate=degenerate, thresholds=asdict(thr),
        validity=(asdict(validity) if validity is not None else None),
    )


def rank_key(r: Result):
    """Lexicographic ranking: degenerate last, then M ascending, then Omega
    descending. Omega only ever separates designs that already have M = 0."""
    return (r.degenerate, r.M, -r.Omega)


# ---------------------------------------------------------------------------
# 5. Differentiable descent surrogate
# ---------------------------------------------------------------------------
# The score uses min/max, which is what the SPEC means but is not what one
# wants inside Adam. The surrogate below is a hinge loss in LINEAR power whose
# hinge points are the score's own thresholds, so it saturates in exactly the
# same places the score does (no reward for depth past r_req, none for
# passband loss below the allowance). Write it with jnp for the optimizer.
#
#   t_r = 10**(-r_req/10)     t_g = 10**(-il_gap/10)    t_p = 10**(-il_pass/10)
#
#   J =        mean_{B_L} relu(|S21|^2 - t_r)
#       +      mean_{B_U} relu(|S21|^2 - t_r)
#       + 2.0 *mean_{G}   relu(t_g - |S21|^2)
#       + 1.25*mean_{P}   relu(t_p - |S21|^2)
#
# The 2.0 / 1.25 weights equalise each term's value at full violation
# (0.99 : 0.501 : 0.794 -> ~1.0 each), matching M's equal-weight-per-
# requirement structure. J is NOT a score: log both J and M(45-period) every
# iteration so surrogate/score divergence is visible from inside the run --
# that divergence is precisely what the Phase-1 same-operator evaluation hid.
SURROGATE_WEIGHTS = dict(band=1.0, gap=2.0, passband=1.25)


# ---------------------------------------------------------------------------
# 6. Self-test on the Stage-0 measurements
# ---------------------------------------------------------------------------
def _selftest():
    import json
    import glob
    from pathlib import Path

    here = Path(__file__).resolve().parent
    g = scoring_grid_mhz()
    print(f"scoring grid: {len(g)} points, {g[0]}-{g[-1]} MHz")
    print(f"descent grid: {len(descent_grid_mhz())} points")

    # ---- synthetic battery: does the metric order designs the way the
    # pre-registration says it should? Single-transmission-zero model,
    # parameterised by each notch's 20-dB bandwidth.
    gf = g.astype(float)

    def _zero(f, f0, bw20, depth_max):
        d = (f - f0) / f0
        k = np.sqrt(99.0) * (bw20 / 2.0) / f0
        e = k / np.sqrt(10 ** (depth_max / 10.0) - 1.0)
        return 10 * np.log10(1.0 + (k / np.sqrt(d ** 2 + e ** 2)) ** 2)

    def _combo(specs, floor=0.2):
        lin = np.ones_like(gf)
        for s in specs:
            lin *= 10 ** (-_zero(gf, *s) / 10.0)
        return -10 * np.log10(lin) + floor

    ideal = np.full(len(g), 0.2)
    ideal[(g >= BAND_L_MHZ[0]) & (g <= BAND_L_MHZ[1])] = 21.0
    ideal[(g >= BAND_U_MHZ[0]) & (g <= BAND_U_MHZ[1])] = 21.0

    battery = [
        ("ANCHOR  empty line (IL = 0)", np.zeros(len(g))),
        ("ANCHOR  solid brick (IL = 60)", np.full(len(g), 60.0)),
        ("IDEAL   flat 21 dB across both bands", ideal),
        ("R5      DEEP but OFF-CENTRE  45 dB, BW20 120 MHz @5.10/5.70",
         _combo([(5100, 120, 45), (5700, 120, 45)])),
        ("R5      SHALLOW but CENTRED  22 dB, BW20 220/120 MHz",
         _combo([(5250, 220, 22), (5775, 120, 22)])),
        ("R4      SYMMETRIC BW 220/220 (upper notch too wide)",
         _combo([(5250, 220, 25), (5775, 220, 25)])),
        ("R4      ASYMMETRIC BW 220/120 (matches the spec)",
         _combo([(5250, 220, 25), (5775, 120, 25)])),
        ("R2      MERGED single notch 25 dB, BW20 800 MHz",
         _combo([(5490, 800, 25)])),
        ("        lower band only", _combo([(5250, 220, 25)])),
    ]
    print("\nSynthetic battery (R4 = bandwidth asymmetry, R5 = deep-vs-centred):")
    for name, il in battery:
        r = score(g, il)
        flag = "PASS" if r.spec_pass else ("DEGEN" if r.degenerate else "")
        print(f"  {name:60s} M={r.M:6.2f}  "
              f"[{r.S_L:5.2f} {r.S_U:5.2f} {r.S_G:5.2f} {r.S_P:5.2f}]  "
              f"Om={r.Omega:7.2f}  ILp_mean={r.IL_pass_mean:5.2f}  {flag}")
    assert battery[2][0].startswith("IDEAL") and score(g, ideal).spec_pass
    m_deep = score(g, battery[3][1]).M
    m_shal = score(g, battery[4][1]).M
    assert m_shal < m_deep, "R5: shallow-but-centred must beat deep-but-off-centre"
    m_sym = score(g, battery[5][1]).M
    m_asym = score(g, battery[6][1]).M
    assert m_asym < m_sym, "R4: correct bandwidth asymmetry must beat symmetric"
    assert score(g, np.full(len(g), 60.0)).degenerate, "brick must be degenerate"
    assert not score(g, battery[7][1]).degenerate, "merged notch is bad, not degenerate"

    print("\nStage-0 classical two-stub (8 mm apart, lambda/4 at each centre):")
    for p in sorted(glob.glob(str(here / "out_vessl" / "stage0" / "window_*.json")),
                    key=lambda s: int(s.split("_")[-1][:-5])):
        d = json.load(open(p))
        if len(d["freqs_GHz"]) < 50:
            continue
        f = np.round(np.array(d["freqs_GHz"]) * 1000).astype(int)
        il = -np.array(d["s21_db"])          # empty line measured at 0.00 dB
        r = score(f, il)
        rr = score(f, il, thr=RELAXED)
        print(f"  periods={d['periods']:5.0f} res={d['dft_res_GHz']:.3f} GHz  "
              f"M={r.M:6.2f}  M_relaxed={rr.M:6.2f}  "
              f"[S_L={r.S_L:5.2f} S_U={r.S_U:5.2f} S_G={r.S_G:5.2f} S_P={r.S_P:5.2f}]  "
              f"R_L={r.R_L_raw:6.2f} R_U={r.R_U_raw:6.2f} gap_max={r.IL_gap_max:5.2f} "
              f"Omega={r.Omega:6.2f} BW20=({r.BW20_L_MHz:.0f},{r.BW20_U_MHz:.0f})MHz "
              f"f_notch=({r.f_notch_L_MHz:.0f},{r.f_notch_U_MHz:.0f})MHz "
              f"ILpass_max={r.IL_pass_max:.2f} frac1dB={r.frac_pass_within_allowance:.2f}")
    print("\nNote: Stage-0 JSONs cover 4.0-8.0 GHz only, so S_P above is a "
          "PARTIAL passband (3.1-4.0 and 8.0-8.6 GHz missing).")


if __name__ == "__main__":
    _selftest()
