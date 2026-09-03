"""WR-90 4th-order iris filter vs mode-matching — frozen-fixture gates (item 3 S3).

Locks the committed record of
``validation/crossval/19_wr90_iris_filter_aghanim.py --write-fixture``
(``tests/fixtures/wr90_iris_filter/fixture.json``) with an in-test
re-implementation of the TEn0 mode-matching cascade — 5 irises, arbitrary
aperture position, re-typed from the physics and sharing only numpy with the
producer.

WHAT THAT RE-IMPLEMENTATION IS AND IS NOT. It agrees with the producer to
0.0e+00 — bit-identically — at every frequency checked. That is the signature of
a REGRESSION LOCK, not of independent confirmation: the same closed-form overlap
integrals and the same Redheffer algebra evaluated in the same order will agree
exactly whether or not both are right. It is kept because it catches drift in
the committed rows, and it is described honestly rather than as a second
opinion. Four SECONDARY CHECKS are re-run here rather than trusted from
generation time. Read the limits below before relying on any of them: three of
the four are weaker than they look, and none is a formulation-independent check
of the five-iris cascade.
  * the N=1 centred limit must reduce to the ODD-mode single-iris formulation,
    which is the object PR #480 confirmed against a formulation-independent
    FDFD solver at 5.8e-4;
  * the L -> 0 collapse must turn two thin irises into one thick one;
  * lossless unitarity must hold at every evaluation, and mirror symmetry must
    hold for the reversed cascade.
LIMITS. An earlier revision of this docstring called those four "independence
axes" and claimed an overlap error would break the single-iris reduction. Both
were wrong: three distinct injected errors in
`_ovl` left ALL FOUR axes above silent, because `_iris_s11_oddmode` calls the same
`_ovl`. Unitarity also constrains only the propagating sub-block, so it does not
validate the evanescent columns, and mirror symmetry holds by construction for a
symmetric geometry. What caught those injected errors was comparison against the
committed data. The formulation-independent check of the five-iris cascade now
EXISTS: `validation/crossval/comparators/fdfd_hplane.py`, a 2-D H-plane FDFD sharing only
numpy/scipy with the cascade, run at fixture-generation time and committed in the
fixture's `fdfd_formulation_independent` block: THREE levels (r=2,3,4), BOTH
Richardson estimates (two-estimate consistency 0.37/0.36 MHz per the porting
handoff's protocol), FDFD(3,4) vs cascade -1.09 MHz f0 / +0.98 MHz BW, and three
reflection zeros at every level -- an earlier biased mask realized the apertures
2h wide and produced a spurious fourth zero, found by an independent port review
and fixed before the record was generated. This file recomputes all of it from
the committed level curves and re-runs the solver's own gates; the full sweep is
too heavy for CI and lives in the regeneration.

Posture, carrying every lesson from #475/#476/#480 plus this stage's own:
  * GATED: centre frequency f0 of the -10 dB |S11| span, rfx vs the oracle
    evaluated on the AS-REALIZED geometry, plus the structural reflection-zero
    COUNT (an integer, depth-independent). Band edges and bandwidth are NOT
    gated: the iris-thickness leg of the node-plane convention is unsettled at
    the half-cell level and moves them ~22-40 MHz per cell against f0's
    ~2.4 MHz, so a gate on them would pin the convention, not the solver.
  * GATED (setup, not physics): the ring-down, feed-clearance and
    absorber-depth witnesses must each hold to one frequency bin. A resonant
    band read off an unsettled or absorber-limited run is not a measurement.
  * REPORTED, never gated: worst in-band return loss (the reference's own two
    solvers disagree by 0.7 dB on it and on which ripple peak is worst),
    individual ripple levels, every zero DEPTH, the coarse a/60 rung, phase.
  * THE COMPARATOR'S INPUTS ARE GATED TOO, and this is the lesson of this
    stage: the oracle must be fed the geometry that was BUILT, not the geometry
    that was drawn. Feeding it the drawn cell counts biases f0 by +107.5 MHz at
    the shipped geometry -- FIVE times the reference's own 21.9 MHz CST-vs-HFSS
    spread -- and the envelope-times-1.5 gate rule does not catch that, because
    that rule bounds SCATTER and this is BIAS. It launders the bias into a
    ~162 MHz "measured" gate, 46% of the passband, which pins nothing. So the
    electrical cell counts are re-derived HERE from the committed node indices
    and checked against both the rule and the compensation identity, and the
    bias itself is re-measured rather than quoted from prose.

No FDTD runs here; regeneration is the crossval script's job. Gates must not be
re-tuned to look tighter than the recorded physics.
"""
from __future__ import annotations

import ast
import cmath
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests/fixtures/wr90_iris_filter/fixture.json"
_ARTIFACT = _REPO_ROOT / "validation/crossval/_19_iris_filter_results/rfx.json"
_SCRIPT = _REPO_ROOT / "validation/crossval/19_wr90_iris_filter_aghanim.py"

C0 = 299792458.0
MU0 = 4e-7 * np.pi
A = 22.86e-3
# The gate is `envelope x multiplier`. The envelope is anchored to data; the
# multiplier used to be a second, unanchored degree of freedom -- an
# independent battery found that a find-replace of 1.5 -> 3.0 plus three
# constants doubles the gate with every guard still passing. That finding
# became issue #528, and #539 gave the multiplier ONE repo-wide definition,
# tests/_gate_policy.py, which this case consumes like every other gated
# case. The falsifiers in tests/contracts/test_gate_policy_is_shared.py re-derive this
# case's gate from the shared constant (discovered via the fixture glob), so
# a local widening is caught from OUTSIDE this file.
from tests._gate_policy import gate_from_envelope  # noqa: E402


# --------------------------------------------------------------------------- #
# Re-typed N-iris TEn0 cascade, arbitrary aperture offset. A regression lock on
# the committed rows, NOT a second opinion — see the module docstring.
# --------------------------------------------------------------------------- #
def _gam(n, w, k):
    return np.sqrt(complex((n * np.pi / w) ** 2 - k * k))


def _ovl(a, d, x0, n, m):
    """<guide mode n | aperture mode m> for an aperture [x0, x0+d]."""
    al, be = n * np.pi / a, m * np.pi / d

    def iss(p, q, L):
        if abs(p - q) < 1e-30:
            return L / 2 - np.sin(2 * p * L) / (4 * p)
        return (np.sin((p - q) * L) / (p - q) - np.sin((p + q) * L) / (p + q)) / 2

    def ics(p, q, L):
        if abs(p - q) < 1e-30:
            return (1 - np.cos(2 * q * L)) / (4 * q) if q > 0 else 0.0
        return ((1 - np.cos((q + p) * L)) / (q + p)
                + (1 - np.cos((q - p) * L)) / (q - p)) / 2

    return (np.sqrt(2 / a) * np.sqrt(2 / d)
            * (np.cos(al * x0) * iss(al, be, d) + np.sin(al * x0) * ics(al, be, d)))


def _star(sa, sb):
    A11, A12, A21, A22 = sa
    B11, B12, B21, B22 = sb
    i1 = np.linalg.inv(np.eye(A22.shape[0]) - A22 @ B11)
    i2 = np.linalg.inv(np.eye(B11.shape[0]) - B11 @ A22)
    return (A11 + A12 @ B11 @ i1 @ A21, A12 @ i2 @ B12,
            B21 @ i1 @ A21, B22 + B21 @ A22 @ i2 @ B12)


def _step(a, d, x0, k, n_a, n_b):
    """S-matrix of the guide->aperture step, in power-normalised mode bases."""
    Na, Nb = np.arange(1, n_a + 1), np.arange(1, n_b + 1)
    gA = np.array([_gam(n, a, k) for n in Na])
    gB = np.array([_gam(m, d, k) for m in Nb])
    w = k * C0
    YA, YB = gA / (1j * w * MU0), gB / (1j * w * MU0)
    Cm = np.array([[_ovl(a, d, x0, n, m) for m in Nb] for n in Na])
    YAd = np.diag(YA)
    Minv = np.linalg.inv(np.diag(YB) + Cm.T @ YAd @ Cm)
    T_ba = 2 * Minv @ Cm.T @ YAd
    R_aa = Cm @ T_ba - np.eye(n_a)
    R_bb = Minv @ (np.diag(YB) - Cm.T @ YAd @ Cm)
    T_ab = Cm @ (np.eye(n_b) + R_bb)
    sYA, sYB = np.sqrt(YA), np.sqrt(YB)
    S = ((sYA[:, None] * R_aa) / sYA[None, :],
         (sYA[:, None] * T_ab) / sYB[None, :],
         (sYB[:, None] * T_ba) / sYA[None, :],
         (sYB[:, None] * R_bb) / sYB[None, :])
    return S, gA, gB


def _line(g, L):
    P = np.diag(np.exp(-g * L))
    z = np.zeros_like(P)
    return (z, P, P, z)


def _filter_s11(a, aps, offs, ths, cavs, f, n_a=90):
    """|S11| of an N-iris cascade; evanescent modes are carried across cavities."""
    k = 2 * np.pi * f / C0
    total = None
    gA_ref = None
    for i, (d, x0, t) in enumerate(zip(aps, offs, ths)):
        n_b = max(4, int(round(n_a * d / a)))
        S, gA, gB = _step(a, d, x0, k, n_a, n_b)
        gA_ref = gA
        rev = (S[3], S[2], S[1], S[0])
        iris = _star(_star(S, _line(gB, t)), rev)
        total = iris if total is None else _star(total, iris)
        if i < len(cavs):
            total = _star(total, _line(gA, cavs[i]))
    s11, s21 = total[0][0, 0], total[2][0, 0]
    assert gA_ref is not None
    # lossless unitarity rides with every evaluation
    assert abs(abs(s11) ** 2 + abs(s21) ** 2 - 1) < 1e-6, (f, abs(s11), abs(s21))
    return abs(s11)


def _band(curve, freqs, threshold_db=10.0):
    """The -threshold_db band by INTERPOLATED crossing, mirroring the script.

    A sampled edge is quantised to one 10 MHz bin, which would put grid spacing
    rather than physics into the envelope the gate is derived from.
    """
    s11 = np.asarray(curve, dtype=float)
    f = np.asarray(freqs, dtype=float)
    rl = -20 * np.log10(np.clip(s11, 1e-12, None))
    inb = rl >= threshold_db
    assert inb.sum() >= 2 and not inb[0] and not inb[-1], "band touches window edge"
    i_lo = int(np.argmax(inb))
    i_hi = int(len(inb) - 1 - np.argmax(inb[::-1]))

    def cross(i_in, i_out):
        y_in, y_out = rl[i_in] - threshold_db, rl[i_out] - threshold_db
        if y_in == y_out:
            return float(f[i_in])
        return float(f[i_in] + (y_in / (y_in - y_out)) * (f[i_out] - f[i_in]))

    lo, hi = cross(i_lo, i_lo - 1), cross(i_hi, i_hi + 1)
    zeros = [float(f[i]) for i in range(1, len(f) - 1)
             if lo <= f[i] <= hi and s11[i] < s11[i - 1] and s11[i] < s11[i + 1]]
    # Mirrors the producer: worst RL over the WHOLE span (a threshold-masked
    # statistic cannot report a threshold violation), plus the contiguity
    # structure, because the outermost crossings do not define a passband.
    span = slice(i_lo, i_hi + 1)
    holes = int((~inb[span]).sum())
    idx = np.where(inb[span])[0]
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    longest = max(runs, key=len)
    return dict(lo=lo, hi=hi, f0=0.5 * (lo + hi), bw=hi - lo, zeros=zeros,
                worst_rl_db=float(rl[span].min()),
                span_holes=holes, n_span_bins=int(inb[span].size),
                longest_contiguous_hz=float(f[i_lo + longest[-1]]
                                            - f[i_lo + longest[0]]),
                contiguous=bool(holes == 0))


@pytest.fixture(scope="module")
def fixture() -> dict:
    with open(_FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def script_src() -> str:
    return _SCRIPT.read_text()


def row_s11(fixture):
    return fixture["gated_rfx"]["s11"]


def _rows(fixture):
    rows = [fixture["gated_rfx"]]
    if fixture.get("coarse_diagnostic"):
        rows.append(fixture["coarse_diagnostic"])
    return rows


# --------------------------------------------------------------------------- #
# Record integrity
# --------------------------------------------------------------------------- #
def test_fixture_and_artifact_are_the_same_record(fixture):
    with open(_ARTIFACT) as f:
        artifact = json.load(f)
    assert artifact == fixture, "fixture and committed artifact diverged"


# --------------------------------------------------------------------------- #
# INTERNAL CONSISTENCY. Every scalar is recomputed from the data it summarises.
#
# This block exists because an independent reviewer ran 42 mutations against an
# earlier revision of this file and 27 survived, 12 of them from one omission:
# nothing recomputed a summary from its own trace. The sharpest survivor scaled
# |S11| by 1.09 and re-pinned the digest honestly while leaving max_colpow at
# 1.0065 — true column power 1.1958 across 97 of 131 bins, i.e. a grossly
# non-passive record, and the whole suite passed. A digest cannot substitute for
# this: re-pinning a hash is a one-line edit in the same commit as the data,
# whereas a recomputation has to be satisfied by the numbers themselves.
# --------------------------------------------------------------------------- #
def _freqs(fixture):
    return np.asarray(fixture["config"]["freqs_hz"], dtype=float)


def test_every_committed_band_is_recomputed_from_its_trace(fixture):
    freqs = _freqs(fixture)
    for row in _rows(fixture):
        for trace_key, band_key in (("s11", "band"), ("oracle_s11", "oracle_band")):
            mine = _band(row[trace_key], freqs)
            got = row[band_key]
            for k, v in mine.items():
                if isinstance(v, float):
                    assert got[k] == pytest.approx(v, abs=1e-6, rel=1e-9), (
                        row["cells_per_a"], band_key, k, got[k], v)
                else:
                    assert got[k] == v, (row["cells_per_a"], band_key, k)


def test_committed_passivity_is_recomputed_from_the_traces(fixture):
    """max_colpow AND the violating-bin footprint, from s11/s21 themselves."""
    for row in _rows(fixture):
        s11 = np.asarray(row["s11"], dtype=float)
        s21 = np.asarray(row["s21"], dtype=float)
        colpow = s11 ** 2 + s21 ** 2
        assert row["max_colpow"] == pytest.approx(float(colpow.max()), abs=5e-5)
        over = [int(i) for i in np.where(colpow > 1.02)[0]]
        assert row["colpow_over_102_bins"] == over, (
            row["cells_per_a"], len(over), len(row["colpow_over_102_bins"]))


def test_reported_deltas_are_recomputed_from_the_bands(fixture):
    freqs = _freqs(fixture)
    for row in _rows(fixture):
        mine, ora = _band(row["s11"], freqs), _band(row["oracle_s11"], freqs)
        for key, want in (("d_f0_mhz", mine["f0"] - ora["f0"]),
                          ("d_lo_mhz", mine["lo"] - ora["lo"]),
                          ("d_hi_mhz", mine["hi"] - ora["hi"]),
                          ("d_bw_mhz", mine["bw"] - ora["bw"])):
            if key in row:
                assert row[key] == pytest.approx(want / 1e6, abs=5e-3), (
                    row["cells_per_a"], key)
        # the algebraic identity the earlier revision missed: the "asymmetric
        # edge residual" and the "bandwidth deficit" are one fact
        if "d_bw_mhz" in row:
            assert row["d_bw_mhz"] == pytest.approx(
                row["d_hi_mhz"] - row["d_lo_mhz"], abs=1e-2)


def _zeros_interpolated(curve, freqs, lo, hi):
    """Reflection-zero frequencies with the grid quantisation removed.

    The committed `zeros` are raw grid samples, which is correct for the COUNT --
    an integer is quantisation-insensitive -- but not for comparing frequencies:
    on a 10 MHz grid two traces whose zeros are physically 12 MHz apart can land
    one or two bins apart, reading 10 or 20 MHz. Measured here: sampled diffs
    20.0/20.0/10.0 MHz against interpolated 15.9/16.8/11.2, so quantisation was
    contributing about 3 MHz of the discrepancy and the interpolated values sit
    where the f0 residual (+12.08 MHz) says they should.

    Parabolic vertex through the three samples bracketing each interior minimum.
    This is the same correction already applied to the band edges, and it removes
    a metric artefact rather than widening a tolerance.
    """
    # Vertex on |S11|^2, not |S11|: a reflection zero is locally LINEAR in
    # |S11| (a V, not a parabola), so a parabolic vertex on the magnitude is
    # biased — measured by an independent battery at 0.65 MHz mean / 0.83 worst
    # on synthetic nulls at this grid, systematically undershooting. On the
    # squared magnitude the null is locally quadratic and the vertex is exact.
    y = np.asarray(curve, dtype=float) ** 2
    f = np.asarray(freqs, dtype=float)
    df = f[1] - f[0]
    out = []
    for i in range(1, len(y) - 1):
        if lo <= f[i] <= hi and y[i] < y[i - 1] and y[i] < y[i + 1]:
            denom = y[i - 1] - 2 * y[i] + y[i + 1]
            off = 0.5 * (y[i - 1] - y[i + 1]) / denom if denom != 0 else 0.0
            out.append(float(f[i] + off * df))
    return out


def _all_traces(fixture):
    """Every committed trace, keyed by where it lives.

    Recomputation is a RELATIVE integrity check: it cannot detect an edit to the
    trace it recomputes from, because it then agrees by construction. Its
    absoluteness comes entirely from an anchor outside the row. An earlier
    revision anchored only `gated_rfx`, which left the coarse rung and every
    witness leg circular. That matters here because the f0 envelope is
    residual-dominated -- 0.06 MHz spread against a 12.08 MHz residual, measured
    on this case's population -- so a one-bin edit to any unanchored leg
    transfers 1:1 into the envelope and 1.5:1 into the gate. (The spread and
    residual are this case's numbers; the transfer argument is the independent
    reviewer's.) So the anchor covers every trace, and both components of each,
    since the membership criterion reads column power.
    """
    out = {}
    for tag in ("gated_rfx", "coarse_diagnostic"):
        out[tag] = {k: fixture[tag][k] for k in ("s11", "s21", "oracle_s11")}
    for r in fixture["ring_down_witness"]:
        out[f"ring_{int(r['num_periods'])}"] = {k: r[k] for k in ("s11", "s21")}
    for r in fixture["b_invariance_witness"]:
        out[f"b_{r['b_cells']}"] = {k: r[k] for k in ("s11", "s21")}
    for key, legs in (("feed_clearance_witness", ("mid", "generous")),
                      ("absorber_depth_witness", ("mid", "deep"))):
        for leg in legs:
            out[f"{key}_{leg}"] = {k: fixture[key][leg][k] for k in ("s11", "s21")}
    for r, lv in fixture["fdfd_formulation_independent"]["levels"].items():
        out[f"fdfd_r{r}"] = {"s11": lv["s11"]}
    return out


def _legs_with_colpow(fixture):
    """Every leg whose column power decides envelope membership."""
    out = []
    for r in fixture["ring_down_witness"]:
        out.append((f"ring np{int(r['num_periods'])}", r))
    for r in fixture["b_invariance_witness"]:
        out.append((f"b={r['b_cells']} cells", r))
    for key, legs in (("feed_clearance_witness", ("mid", "generous")),
                      ("absorber_depth_witness", ("mid", "deep"))):
        for leg in legs:
            out.append((f"{key}:{leg}", fixture[key][leg]))
    return out


def test_the_membership_criterions_input_is_recomputed_on_every_leg(fixture):
    """Anchor the criterion AND its input.

    Membership is decided by column power, which is s11^2 + s21^2. An earlier
    revision anchored `s11` on the witness legs and committed `max_colpow` as a
    free scalar, so the quantity deciding membership sat outside the anchor. That
    permitted a two-step: admit the under-settled leg to widen the envelope, and
    push another leg above the threshold to keep the exclusion count non-zero, so
    the "criterion is exercised" guard still passed. Both moves are colpow edits.
    """
    for tag, leg in _legs_with_colpow(fixture):
        s11 = np.asarray(leg["s11"], dtype=float)
        s21 = np.asarray(leg["s21"], dtype=float)
        want = float((s11 ** 2 + s21 ** 2).max())
        assert leg["max_colpow"] == pytest.approx(want, abs=5e-5), (
            "committed column power does not follow from the leg's own traces",
            tag, leg["max_colpow"], want)


def test_committed_population_is_the_criterion_based_selection(fixture):
    """The envelope's membership must follow a CRITERION, not a value list.

    A hardcoded exclusion tuple lets a future failing row be dropped by adding
    its `num_periods` to the tuple. Membership is therefore checked here against
    the rule: every candidate leg enters unless it fails the settling criterion
    (column power > 1.02), and every excluded leg must fail it.
    """
    g = fixture["gates"]
    committed = {e["config"] for e in g["f0_envelope_population"]}
    gated_np = fixture["gated_rfx"]["num_periods"]

    excluded_with_reason = []
    for r in fixture["ring_down_witness"]:
        if r["num_periods"] == gated_np:
            continue
        tag = f"ring np{int(r['num_periods'])}"
        if r["max_colpow"] > 1.02:
            excluded_with_reason.append((tag, r["max_colpow"]))
            assert tag not in committed, (
                "a leg failing the settling criterion is in the envelope", tag)
        else:
            assert tag in committed, ("a settled leg is missing from the "
                                      "envelope", tag, sorted(committed))
    assert excluded_with_reason, (
        "no leg is excluded by the criterion, so the criterion is untested — "
        "the short run that demonstrates it can fire has gone missing")

    # The EXCLUSION RECORD itself is anchored, not free text plus a free number:
    # an independent battery rewrote f0_population_excluded[0] to a passing
    # colpow and an unrelated reason and the suite stayed green, so an auditor
    # asking "why was this leg dropped?" would have read an unverified answer.
    # Each excluded entry must name a real leg, carry that leg's colpow as
    # recomputed from its own committed traces, and exceed the criterion.
    book = {e["config"]: e for e in fixture["gates"]["f0_population_excluded"]}
    assert set(book) == {t for t, _ in excluded_with_reason}, (
        "the committed exclusion record does not match the criterion selection",
        sorted(book), sorted(t for t, _ in excluded_with_reason))
    for r in fixture["ring_down_witness"]:
        tag = f"ring np{int(r['num_periods'])}"
        if tag in book:
            s11 = np.asarray(r["s11"], dtype=float)
            s21 = np.asarray(r["s21"], dtype=float)
            true_cp = float((s11 ** 2 + s21 ** 2).max())
            assert book[tag]["max_colpow"] == pytest.approx(true_cp, abs=5e-5), (
                "an exclusion entry's colpow does not follow from the leg's own "
                "traces", tag)
            assert true_cp > 1.02, (
                "an exclusion entry cites a colpow that does not fail the "
                "criterion", tag, true_cp)


def test_f0_envelope_is_recomputed_from_its_population(fixture):
    """The gate must terminate in physics, not in a literal.

    An earlier revision let the gate, the envelope, the script constant and all
    four test pins be widened coherently in one commit and stay green, because
    no assert tied the envelope to the rows it claims to summarise.
    """
    g = fixture["gates"]
    freqs = _freqs(fixture)
    ora_f0 = _band(fixture["gated_rfx"]["oracle_s11"], freqs)["f0"]

    seen = {}
    seen["gated"] = _band(fixture["gated_rfx"]["s11"], freqs)["f0"]
    for r in fixture["ring_down_witness"]:
        if r["num_periods"] not in (400.0, 200.0):
            seen[f"ring{int(r['num_periods'])}"] = r["f0"]
    for r in fixture["b_invariance_witness"]:
        if r["b_cells"] != fixture["config"]["b_cells"]:
            seen[f"b{r['b_cells']}"] = r["f0"]
    for key, legs in (("feed_clearance_witness", ("mid", "generous")),
                      ("absorber_depth_witness", ("mid", "deep"))):
        for leg in legs:
            d = fixture[key][leg]
            seen[f"{key}:{leg}"] = 0.5 * (d["lo"] + d["hi"])

    residuals = {k: (v - ora_f0) / 1e6 for k, v in seen.items()}
    assert len(residuals) == len(g["f0_envelope_population"]), (
        "population size changed", sorted(residuals), g["f0_envelope_population"])
    env = max(abs(d) for d in residuals.values())
    assert g["f0_measured_envelope_mhz"] == pytest.approx(env, abs=5e-3)
    assert g["f0_gate_mhz"] == gate_from_envelope(max(env, 1e-9), quantum=1)
    for entry in g["f0_envelope_population"]:
        assert any(abs(entry["d_f0_mhz"] - d) < 5e-3 for d in residuals.values()), (
            "committed population entry not reproducible", entry)

    # the under-settled run must NOT be buying slack
    short = next(r for r in fixture["ring_down_witness"]
                 if r["num_periods"] == 200.0)
    assert short["max_colpow"] > 1.02, "np=200 is the settling counterexample"
    assert not any("200" in e["config"] for e in g["f0_envelope_population"]), (
        "a run that fails the settling criterion is inflating the envelope")


def test_witness_legs_are_tied_to_the_gated_row(fixture):
    """Every witness must reference the configuration actually gated.

    All six witness mutations survived the earlier revision: their f0/bw could
    be moved hundreds of MHz off the gated row and nothing noticed, because no
    assert connected them.
    """
    freqs = _freqs(fixture)
    gated = _band(fixture["gated_rfx"]["s11"], freqs)
    gated_np = fixture["gated_rfx"]["num_periods"]

    # BOOKKEEPING, not corroboration. These legs ARE the gated configuration --
    # the producer reuses the gated row rather than paying for a bit-identical
    # repeat -- so equality is definitional and is asserted EXACTLY. A loose
    # tolerance would imply a comparison between independent runs that is not
    # happening, and would excuse a drift it should forbid.
    ring_ref = next(r for r in fixture["ring_down_witness"]
                    if r["num_periods"] == gated_np)
    assert ring_ref["f0"] == gated["f0"]
    assert ring_ref["bw"] == gated["bw"]
    assert ring_ref["s11"] == fixture["gated_rfx"]["s11"]
    b_ref = next(r for r in fixture["b_invariance_witness"]
                 if r["b_cells"] == fixture["config"]["b_cells"])
    assert b_ref["f0"] == gated["f0"]
    assert b_ref["s11"] == fixture["gated_rfx"]["s11"]
    for key in ("feed_clearance_witness", "absorber_depth_witness"):
        leg = fixture[key]["gated"]
        assert leg["lo"] == gated["lo"], key
        assert leg["hi"] == gated["hi"], key


def test_witness_deltas_are_recomputed_from_their_legs(fixture):
    """The committed deltas are the WORST over the interior and outer legs.

    An earlier revision recomputed from the outer leg only, which matched by
    coincidence (the outer leg happened to dominate). If the interior sample
    ever dominates a side -- the non-monotonic case this record exists to catch
    -- an outer-only recomputation would verify the wrong quantity.
    """
    for key, legs in (("feed_clearance_witness", ("mid", "generous")),
                      ("absorber_depth_witness", ("mid", "deep"))):
        w = fixture[key]
        for side in ("lo", "hi"):
            want = max(abs(w[leg][side] - w["gated"][side]) for leg in legs) / 1e6
            assert w[f"d_{side}_mhz"] == pytest.approx(want, abs=5e-3), (key, side)


def test_aperture_nodes_match_the_reference_dimensions_and_the_builder(fixture):
    """Pin aperture POSITION and WIDTH against an independent source.

    The earlier form asserted `lo - 1 == (cells - d_c) // 2` with
    `d_c = hi - lo + 2`, which reduces to "the pair is mirror-symmetric about the
    guide centre" — an identity in (lo, hi) that a uniformly inflated or deflated
    set still satisfies. A reviewer reduced it and was right. Here `d_c` comes
    from the PAPER's apertures through the builder's own rules (round to cells,
    then bump parity so symmetric fins are realizable), so neither the width nor
    the position is free.
    """
    cells = fixture["config"]["gated_cells_per_a"]
    dx = A / cells
    nodes = fixture["gated_rfx"]["aperture_nodes"]
    aps_mm = fixture["reference"]["apertures_mm"]
    assert len(nodes) == len(aps_mm) == 5
    # This zips POSITIONALLY, and should stay that way: Aghanim's aperture set is
    # symmetric ([10.27, 6.65, 6.18, 6.65, 10.27]), so a permutation preserves
    # the multiset of widths and leaves every aperture individually centred, and
    # a sorted-collection comparison would accept a different filter.
    # It is not the only guard, though, and an independent battery measured which
    # one actually fires: a permuted `aperture_nodes` is caught by FOUR tests,
    # the strongest being the committed-oracle comparison, because `oracle_s11`
    # is both committed and recomputed from the geometry -- so ANY aperture
    # change breaks it, not just a reordering. Keep the positional form as the
    # local guard; do not rely on it as the only one.
    for (lo, hi), d_mm in zip(nodes, aps_mm):
        d_c = round(d_mm * 1e-3 / dx)
        d_c += (cells - d_c) % 2          # the builder's parity bump
        fin_c = (cells - d_c) // 2
        assert lo == fin_c + 1, ("aperture start is not where the builder puts it",
                                 lo, fin_c + 1, d_mm)
        assert hi == lo + d_c - 2, ("aperture width does not match the reference",
                                    hi, lo + d_c - 2, d_mm)


def test_gated_traces_are_bit_pinned(fixture):
    """A digest of the committed traces, pinned HERE — a third location.

    The mutation battery for this case found one edit that no tolerance-based
    gate can catch by construction: cyclically shifting the rfx trace by three
    bins moves the measured disagreement from +17.1 MHz to -12.9 MHz, i.e. it
    makes the record look BETTER, and |rfx - oracle| <= gate passes either way.
    Editing both data files together also defeats the fixture==artifact check.

    So the traces are additionally pinned by digest in the TEST SOURCE, which a
    coordinated edit of the two JSON files does not reach. This is a different
    guarantee from the gates: not "the physics is within tolerance" but "the
    committed record has not been altered". Regenerating the fixture therefore
    requires re-pinning this digest deliberately, which is the intent.
    """
    payload = json.dumps(_all_traces(fixture), sort_keys=True,
                         separators=(",", ":"))
    assert hashlib.sha256(payload.encode()).hexdigest() == _PIN_TRACE_SHA256, (
        "committed traces changed; if this was a deliberate regeneration, "
        "update _PIN_TRACE_SHA256 in the same commit as the new fixture")


def test_gate_is_hard_pinned_and_equals_the_derived_relation(fixture):
    """Hard pin AND the derived relation. Either alone is self-ratifying.

    The literal pin catches a coherent gate+envelope edit; the derived relation
    catches a hand-edited gate. Neither is sufficient, and neither anchors the
    envelope to data — that is
    `test_f0_envelope_is_recomputed_from_its_population`, which is the assert a
    reviewer defeated this pair without.
    """
    g = fixture["gates"]
    assert g["f0_gate_mhz"] == _PIN_F0_GATE_MHZ
    assert g["f0_measured_envelope_mhz"] == pytest.approx(_PIN_F0_ENV_MHZ, abs=1e-4)
    assert g["f0_gate_mhz"] == gate_from_envelope(
        max(g["f0_measured_envelope_mhz"], 1e-9), quantum=1)
    # band edges and bandwidth are carried as REPORTED residuals, so a gate key
    # for them must not reappear without a fresh sensitivity argument
    assert "edge_gate_mhz" not in g and "bw_gate_mhz" not in g, (
        "edges/BW were re-gated; their comparator-input uncertainty (~20 MHz "
        "from the unsettled iris-thickness convention) exceeds any gate on them")


def test_script_live_gate_constant_matches_fixture(fixture, script_src):
    """The live script constant, and that the self-check is REACHABLE.

    A substring grep for the self-check is defeated by
    `if False:  # abs(gate - required) > 1e-9:`, so the condition is located in
    the AST instead and its test must not be a constant.
    """
    g = fixture["gates"]
    m = re.search(r"^GATE_F0_MHZ = ([0-9.]+)", script_src, re.M)
    assert m, "GATE_F0_MHZ not found as a live module constant"
    assert float(m.group(1)) == float(g["f0_gate_mhz"])

    tree = ast.parse(script_src)
    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and "required" in ast.dump(n.test)
              and "GATE_F0_MHZ" in ast.dump(n.test)]
    assert guards, "the gate == ceil(env*multiplier) self-check is gone"
    for g_node in guards:
        assert not isinstance(g_node.test, ast.Constant), (
            "the self-check was disabled with a constant condition")

    # Asserting the guard EXISTS does not check WHICH multiplier it uses: a
    # local literal inside the guard can be edited with the guard still present
    # and passing. The script must derive `required` through the shared
    # repo-wide policy (#528/#539), not a fresh local literal.
    assert "from tests._gate_policy import gate_from_envelope" in script_src, (
        "the script no longer imports the shared gate policy helper")
    assert re.search(r"required\s*=\s*gate_from_envelope\(", script_src), (
        "the script's self-check no longer derives the gate through "
        "tests._gate_policy.gate_from_envelope")
    assert re.search(r"required\s*=\s*np\.ceil", script_src) is None, (
        "a local ceil derivation reappeared in the script's self-check "
        "alongside (or instead of) the shared helper")


# --------------------------------------------------------------------------- #
# The comparator's INPUTS — the failure this stage actually hit.
# --------------------------------------------------------------------------- #
def test_case_is_discovered_and_bound_by_the_shared_gate_policy(fixture):
    """The multiplier is a repo-wide convention with ONE definition (#528/#539).

    This test's predecessor scanned sibling gate tests for their local `1.5`
    literals; #539 removed every one of those by design (they all consume
    tests/_gate_policy.py now), which turned the scan into a guaranteed
    failure on any merge with main. The binding this file needs is different:
    that THIS case is inside the shared policy's blast radius. The falsifiers
    in tests/contracts/test_gate_policy_is_shared.py re-derive every discovered case's
    gate from the shared constant and prove a widened multiplier moves them
    all together -- being discovered there is what makes a local widening
    here visible from outside this file.
    """
    from tests.contracts.test_gate_policy_is_shared import _REAL_CASES
    me = ("tests/fixtures/wr90_iris_filter/fixture.json",
          ("gates", "f0_measured_envelope_mhz"),
          ("gates", "f0_gate_mhz"),
          1)
    assert me in _REAL_CASES, (
        "case 19 is no longer discovered by the shared gate-policy "
        "falsifiers -- its fixture keys or the discovery pattern drifted, so "
        "the shared-multiplier guarantee no longer covers this case",
        _REAL_CASES)
    g = fixture["gates"]
    assert g["f0_gate_mhz"] == gate_from_envelope(
        g["f0_measured_envelope_mhz"], quantum=1)


def test_contiguity_lock_is_gated_and_fires_on_the_split_shape(fixture, script_src):
    """Joint-review N1: f0 is computed from the OUTERMOST -10 dB crossings, so
    the f0 gate alone cannot see a split passband -- a future regeneration
    whose band collapsed into separated resonances could ship green with its
    bridged midpoint inside the 19 MHz gate. The lock is the committed
    envelope (exactly one interior hole bin), recomputed here from the trace,
    and the committed coarse rung is the demonstration that the instrument
    fires on the very shape it exists to catch.
    """
    freqs = np.asarray(fixture["config"]["freqs_hz"], dtype=float)
    gated = _band(fixture["gated_rfx"]["s11"], freqs)
    assert gated["span_holes"] <= 1
    assert gated["span_holes"] == fixture["gated_rfx"]["band"]["span_holes"]
    # the split shape this lock exists for: the a/60 rung's "band" is two
    # separated resonances -- it must violate the lock, or the lock is inert
    coarse = _band(fixture["coarse_diagnostic"]["s11"], freqs)
    assert coarse["span_holes"] > 1, coarse["span_holes"]
    # and the script carries the gate as a live, reachable check
    assert re.search(r"^MAX_SPAN_HOLES_GATED = 1\b", script_src, re.M), (
        "the contiguity lock constant is gone or was widened")
    assert 'contig_ok = meas["span_holes"] <= MAX_SPAN_HOLES_GATED' in script_src
    assert "ok &= f0_ok and zeros_ok and contig_ok" in script_src, (
        "the contiguity lock no longer participates in the exit code")


def test_zero_count_gate_is_robust_across_the_thickness_ambiguity_band(fixture):
    """Joint-review N3: the iris-thickness electrical leg is the one unsettled
    convention input (measured ~(t_c - 0.68)*dx against the built
    (t_c - 1)*dx), and the zero count is a gated integer -- so the count must
    be invariant across that acknowledged ambiguity band, or the gate is an
    artifact of picking a convention. The committed sweep says it is; this
    test asserts the committed rows and re-solves ONE interior point with
    this file's own re-typed cascade (an implementation the producer does not
    share).
    """
    rows = fixture["iris_thickness_zero_count_sweep"]["rows"]
    gated_count = len(fixture["oracle_rasterized_band"]["zeros"])
    assert rows[0]["t_elec_cells"] == 8.0 and rows[-1]["t_elec_cells"] == 8.5
    assert len(rows) == 11
    assert all(r["zeros"] == gated_count == 3 for r in rows), rows
    # the sweep must actually exercise the input: bandwidth moves ~20 MHz
    # across the band while the gated integer stays fixed
    assert rows[0]["bw_hz"] - rows[-1]["bw_hz"] > 10e6
    eg = fixture["electrical_geometry"]
    cfg = fixture["config"]
    freqs = np.asarray(cfg["freqs_hz"], dtype=float)
    dx = A / cfg["gated_cells_per_a"]
    aps, offs = _aps_offs(fixture, dx)
    mid = rows[5]
    assert mid["t_elec_cells"] == 8.25
    mine = _band([_filter_s11(A, aps, offs, [8.25 * dx] * 5,
                              [c * dx for c in eg["cavity_cells"]], f)
                  for f in freqs], freqs)
    assert len(mine["zeros"]) == mid["zeros"]
    assert mine["f0"] == pytest.approx(mid["f0_hz"], abs=2e6)
    assert mine["bw"] == pytest.approx(mid["bw_hz"], abs=3e6)


def test_electrical_geometry_is_rederived_from_committed_node_indices(fixture):
    """Re-derive the oracle's lengths from the rasterised metal, independently.

    The electrical length of a region is the distance between its bounding
    zeroed node planes, so a cavity drawn with L_c cells of clear space is
    (L_c + 1)*dx and an iris drawn t_c cells thick is (t_c - 1)*dx. Total
    length is a face-continuity check across region types (NOT a uniqueness
    argument -- it holds for every interface offset sigma); what it catches is
    mixing conventions between region types.
    """
    eg = fixture["electrical_geometry"]
    row = fixture["gated_rfx"]
    x_runs = [tuple(r) for r in row["iris_x_nodes"]]
    assert len(x_runs) == 5

    th_cells = [hi - lo for lo, hi in x_runs]
    cav_cells = [x_runs[i + 1][0] - x_runs[i][1] for i in range(4)]
    assert th_cells == [eg["iris_thickness_cells"]] * 5, (th_cells, eg)
    assert cav_cells == list(eg["cavity_cells"]), (cav_cells, eg)

    drawn_t = eg["drawn_iris_thickness_cells"]
    drawn_L = list(eg["drawn_cavity_cells"])
    assert th_cells == [drawn_t - 1] * 5
    assert cav_cells == [v + 1 for v in drawn_L]

    span = drawn_t * 5 + sum(drawn_L)
    assert sum(th_cells) + sum(cav_cells) == span - 1, "total length not conserved"


def test_drawn_counts_are_the_electrical_space_compensation(fixture):
    """t_c = round(t/dx) + 1 and L_c = round(L/dx) - 1, from the reference dims."""
    eg = fixture["electrical_geometry"]
    ref = fixture["reference"]
    cfg = fixture["config"]
    dx = A / cfg["gated_cells_per_a"]
    assert eg["drawn_iris_thickness_cells"] == round(
        ref["iris_thickness_mm"] * 1e-3 / dx) + 1
    assert list(eg["drawn_cavity_cells"]) == [
        round(v * 1e-3 / dx) - 1 for v in ref["cavities_mm"]]
    # and the aperture needs no correction: d_c*dx already IS the electrical width
    for (lo, hi), d_mm in zip(fixture["gated_rfx"]["aperture_nodes"],
                              ref["apertures_mm"]):
        n_open = hi - lo + 1
        realized_mm = (n_open + 1) * dx * 1e3
        # the producer rounds, so the bound is dx/2. A dx bound admitted an
        # outer-aperture +1-node mutation that would have improved d_lo from
        # 17.08 to 6.76 MHz.
        assert abs(realized_mm - d_mm) <= 0.5 * dx * 1e3 + 1e-9, (realized_mm, d_mm)


def test_using_drawn_counts_would_bias_f0_and_is_recorded_as_such(fixture):
    """The comparator-input bias is re-measured here, not asserted in prose.

    This is the test that would have caught the original defect. It runs the
    INDEPENDENT oracle twice -- once on the realised geometry, once on the drawn
    cell counts -- and confirms the recorded cost of confusing them.
    """
    eg = fixture["electrical_geometry"]
    cfg = fixture["config"]
    freqs = np.asarray(cfg["freqs_hz"], dtype=float)
    dx = A / cfg["gated_cells_per_a"]
    aps, offs = _aps_offs(fixture, dx)

    real = _band([_filter_s11(A, aps, offs,
                              [eg["iris_thickness_cells"] * dx] * 5,
                              [c * dx for c in eg["cavity_cells"]], f)
                  for f in freqs], freqs)
    drawn = _band([_filter_s11(A, aps, offs,
                               [eg["drawn_iris_thickness_cells"] * dx] * 5,
                               [c * dx for c in eg["drawn_cavity_cells"]], f)
                   for f in freqs], freqs)
    bias_mhz = (drawn["f0"] - real["f0"]) / 1e6
    assert bias_mhz == pytest.approx(eg["cost_of_using_intended_counts_mhz"], abs=1.0)
    spread_mhz = fixture["reference"]["digitized_scalars"][
        "solver_spread_f0_hz"] / 1e6
    assert abs(bias_mhz) > 2 * spread_mhz, (
        "the recorded bias no longer exceeds the reference's own solver spread, "
        "so the prose justifying this test is stale")


def _aps_offs(fixture, dx):
    """Apertures and their left offsets, from the committed node indices."""
    aps, offs = [], []
    for lo, hi in fixture["gated_rfx"]["aperture_nodes"]:
        aps.append((hi - lo + 2) * dx)
        offs.append((lo - 1) * dx)
    return aps, offs


# --------------------------------------------------------------------------- #
# The gated physics
# --------------------------------------------------------------------------- #
def _iris_s11_oddmode(a, d, t, f, n_a=40):
    """Single centred iris, ODD modes only — the S1 formulation.

    A different basis (odd modes on the half-symmetric problem, 40 of them
    instead of 90 general ones) and the object PR #480 confirmed against a
    formulation-independent FDFD solver at 5.8e-4. If the general cascade's
    overlap algebra were wrong, this reduction would not close.
    """
    k = 2 * np.pi * f / C0
    n_b = max(4, int(round(n_a * d / a)))
    Na, Nb = np.arange(1, 2 * n_a, 2), np.arange(1, 2 * n_b, 2)
    gA = np.array([_gam(n, a, k) for n in Na])
    gB = np.array([_gam(m, d, k) for m in Nb])
    w = k * C0
    YA, YB = gA / (1j * w * MU0), gB / (1j * w * MU0)
    x0 = (a - d) / 2
    Cm = np.array([[_ovl(a, d, x0, n, m) for m in Nb] for n in Na])
    YAd = np.diag(YA)
    Minv = np.linalg.inv(np.diag(YB) + Cm.T @ YAd @ Cm)
    T_ba = 2 * Minv @ Cm.T @ YAd
    R_aa = Cm @ T_ba - np.eye(n_a)
    R_bb = Minv @ (np.diag(YB) - Cm.T @ YAd @ Cm)
    T_ab = Cm @ (np.eye(n_b) + R_bb)
    sYA, sYB = np.sqrt(YA), np.sqrt(YB)
    S = ((sYA[:, None] * R_aa) / sYA[None, :],
         (sYA[:, None] * T_ab) / sYB[None, :],
         (sYB[:, None] * T_ba) / sYA[None, :],
         (sYB[:, None] * R_bb) / sYB[None, :])
    tot = _star(_star(S, _line(gB, t)), (S[3], S[2], S[1], S[0]))
    return abs(tot[0][0, 0])


def test_cascade_reduces_to_the_fdfd_confirmed_single_iris():
    """N=1 centred: the general cascade must reproduce the S1 odd-mode result."""
    a, d, t = A, 10.16e-3, 2.032e-3
    x0 = (a - d) / 2
    worst = 0.0
    for f in (9.0e9, 10.0e9, 11.0e9, 12.0e9):
        mine = _filter_s11(a, [d], [x0], [t], [], f)
        s1 = _iris_s11_oddmode(a, d, t, f)
        worst = max(worst, abs(mine - s1))
    assert worst < 2e-3, f"N=1 reduction to the S1 formulation fails: {worst:.2e}"


def test_overlap_closed_form_matches_numerical_quadrature():
    """The ONE genuinely independent check of the oracle's inputs.

    An independent reviewer showed that three distinct injected errors in `_ovl`
    left every advertised "independence axis" silent, because the odd-mode
    reduction calls the same `_ovl`. Quadrature is a different route: it never
    touches the `iss`/`ics` closed forms or the cos/sin offset decomposition.

    Gauss-Legendre from numpy rather than scipy, so CI gains no dependency.
    Includes the exact-degeneracy case at d = a/3.75 = 6.096 mm (the as-realized
    centre aperture), where n*pi/a equals m*pi/d exactly for several mode pairs
    and both closed forms hit a vanishing denominator.
    """
    nodes, weights = np.polynomial.legendre.leggauss(400)
    worst = 0.0
    checked = 0
    for d in (10.16e-3, 6.604e-3, 6.096e-3):
        for x0 in ((A - d) / 2, (A - d) / 2 - 0.19e-3):
            u = 0.5 * d * (nodes + 1.0)
            for n in (1, 3, 15, 30, 45, 89):
                for m in (1, 2, 4, 8, 12, 24):
                    quad = float(0.5 * d * np.sum(
                        weights
                        * np.sin(n * np.pi * (u + x0) / A)
                        * np.sin(m * np.pi * u / d))
                        * np.sqrt(2 / A) * np.sqrt(2 / d))
                    worst = max(worst, abs(_ovl(A, d, x0, n, m) - quad))
                    checked += 1
    assert checked >= 200, checked
    assert worst < 1e-12, (
        "the overlap closed form disagrees with quadrature; the iss/ics forms or "
        f"the offset decomposition are wrong: {worst:.3e}")


def test_inheritance_from_the_fdfd_confirmed_case18_oracle_is_executed(fixture):
    """Execute the inheritance instead of asserting it in prose.

    `claim_scope` and the manifest both lean on case 18's oracle having been
    confirmed against a formulation-independent 2-D H-plane FDFD (PR #480). A
    reviewer pointed out that this file never actually compared against THAT
    object, so the inheritance was a prose claim. It is cheap to run: import the
    merged sibling's oracle and reduce this cascade to N=1 against it.
    """
    sibling = _REPO_ROOT / "tests/crossval/test_wr90_iris_modematch_gates.py"
    spec = importlib.util.spec_from_file_location("_case18_gates", sibling)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    case18_iris = mod._iris_s11

    worst = 0.0
    for d in (18.288e-3, 12.192e-3, 7.62e-3):
        for f in (8.5e9, 10.3e9, 12.0e9):
            mine = _filter_s11(A, [d], [(A - d) / 2], [1.524e-3], [], f)
            worst = max(worst, abs(mine - case18_iris(A, d, 1.524e-3, f)))
    assert worst < 2e-3, (
        "the N=1 reduction no longer matches the merged case-18 oracle, which is "
        f"the object carrying the FDFD confirmation: {worst:.3e}")


def test_cascade_closes_the_collapse_limit():
    """Two thin irises with a vanishing gap must equal one thick iris."""
    a, d, gap = A, 6.604e-3, 0.2e-3
    for f in (10.8e9, 11.1e9):
        pair = _filter_s11(a, [d, d], [(a - d) / 2] * 2, [2.0e-3] * 2, [gap], f)
        one = _filter_s11(a, [d], [(a - d) / 2], [2 * 2.0e-3 + gap], [], f)
        assert abs(pair - one) < 3e-3, (f, pair, one)


def test_cascade_is_mirror_symmetric(fixture):
    """Reversing a symmetric cascade must leave |S11| unchanged."""
    eg = fixture["electrical_geometry"]
    cfg = fixture["config"]
    dx = A / cfg["gated_cells_per_a"]
    aps, offs = _aps_offs(fixture, dx)
    ths = [eg["iris_thickness_cells"] * dx] * 5
    cavs = [c * dx for c in eg["cavity_cells"]]
    for f in (10.8e9, 11.0e9):
        fwd = _filter_s11(A, aps, offs, ths, cavs, f)
        rev = _filter_s11(A, aps[::-1], offs[::-1], ths[::-1], cavs[::-1], f)
        assert abs(fwd - rev) < 1e-9, (f, fwd, rev)


def test_gated_band_within_gate_against_the_locked_oracle(fixture):
    eg = fixture["electrical_geometry"]
    cfg = fixture["config"]
    g = fixture["gates"]
    freqs = np.asarray(cfg["freqs_hz"], dtype=float)
    dx = A / cfg["gated_cells_per_a"]
    aps, offs = _aps_offs(fixture, dx)

    oracle = _band([_filter_s11(A, aps, offs,
                                [eg["iris_thickness_cells"] * dx] * 5,
                                [c * dx for c in eg["cavity_cells"]], f)
                    for f in freqs], freqs)
    rfx = _band(fixture["gated_rfx"]["s11"], freqs)

    assert abs(rfx["f0"] - oracle["f0"]) / 1e6 <= g["f0_gate_mhz"], (
        "gated centre frequency outside the gate",
        (rfx["f0"] - oracle["f0"]) / 1e6)
    assert len(rfx["zeros"]) == len(oracle["zeros"]), (
        "structural reflection-zero count differs from the oracle")

    # Zero FREQUENCIES, not just the count. The case declares zero frequencies
    # meaningful and zero depths not values, so the frequencies are gated at the
    # edge tolerance. This is also the only pin on the SHAPE of the committed
    # rfx trace: a mutation battery found that a 3-bin cyclic shift of the trace
    # survived the two band-edge scalars alone, because the shift happened to
    # move the existing +17.1 MHz offset to -12.9 MHz, still inside the gate.
    assert len(row_s11(fixture)) == len(freqs), "trace length != frequency grid"
    # Zero FREQUENCIES are held to the f0 gate, with the grid quantisation removed
    # first (see _zeros_interpolated). This check earned its place by catching an
    # aperture mutation the band comparison alone would have missed.
    zr = _zeros_interpolated(fixture["gated_rfx"]["s11"], freqs,
                             rfx["lo"], rfx["hi"])
    zo = _zeros_interpolated(fixture["gated_rfx"]["oracle_s11"], freqs,
                             oracle["lo"], oracle["hi"])
    assert len(zr) == len(zo) == len(rfx["zeros"]), (
        "interpolated zero count disagrees with the sampled count",
        len(zr), len(zo), len(rfx["zeros"]))
    for got, want in zip(zr, zo):
        assert abs(got - want) / 1e6 <= g["f0_gate_mhz"], (
            "reflection-zero frequency moved beyond the f0 gate",
            got / 1e9, want / 1e9)


def test_committed_oracle_curve_matches_the_retyped_one(fixture):
    """The producer's own oracle row, re-typed here (regression lock, see module doc)."""
    eg = fixture["electrical_geometry"]
    cfg = fixture["config"]
    freqs = np.asarray(cfg["freqs_hz"], dtype=float)
    dx = A / cfg["gated_cells_per_a"]
    aps, offs = _aps_offs(fixture, dx)
    mine = np.array([_filter_s11(A, aps, offs,
                                 [eg["iris_thickness_cells"] * dx] * 5,
                                 [c * dx for c in eg["cavity_cells"]], f)
                     for f in freqs])
    theirs = np.asarray(fixture["gated_rfx"]["oracle_s11"], dtype=float)
    assert np.max(np.abs(mine - theirs)) < 5e-3, np.max(np.abs(mine - theirs))


def test_passband_is_inside_the_frequency_window_on_every_row(fixture):
    """A band edge pinned to a scan boundary is not a measurement."""
    freqs = np.asarray(fixture["config"]["freqs_hz"], dtype=float)
    for row in _rows(fixture):
        b = _band(row["s11"], freqs)
        assert freqs[0] < b["lo"] < b["hi"] < freqs[-1]


# --------------------------------------------------------------------------- #
# Setup witnesses — gated, because a resonant number needs them
# --------------------------------------------------------------------------- #
def test_ring_down_is_settled_at_the_gated_num_periods(fixture):
    ring = {r["num_periods"]: r for r in fixture["ring_down_witness"]}
    # the interior sample is required, not optional: an endpoint-only scan
    # cannot detect non-monotonic sensitivity (the PR #475 failure mode)
    assert {200.0, 400.0, 600.0, 800.0} <= set(ring), (
        "the ring-down scan lost its interior sample", sorted(ring))
    bin_hz = float(np.diff(fixture["config"]["freqs_hz"])[0])
    assert abs(ring[400.0]["f0"] - ring[800.0]["f0"]) <= bin_hz
    assert abs(ring[400.0]["bw"] - ring[800.0]["bw"]) <= bin_hz
    assert ring[400.0]["max_colpow"] <= 1.02
    # truncation shows up as non-passivity FIRST: the short run must be the
    # offender, otherwise this witness is not probing what it claims to
    assert ring[200.0]["max_colpow"] > ring[400.0]["max_colpow"]


def test_feed_clearance_and_absorber_depth_hold_to_one_bin(fixture):
    bin_mhz = float(np.diff(fixture["config"]["freqs_hz"])[0]) / 1e6
    for key in ("feed_clearance_witness", "absorber_depth_witness"):
        w = fixture[key]
        assert w["passed"] is True, key
        assert w["d_lo_mhz"] <= bin_mhz and w["d_hi_mhz"] <= bin_mhz, (key, w)
    clear = fixture["feed_clearance_witness"]
    assert (clear["generous"]["standoff_mm"]
            > 4 * clear["gated"]["standoff_mm"]), "clearance scan is too timid"
    absb = fixture["absorber_depth_witness"]
    # a bare `>` passes 111 vs 110; the recorded scan is 183 vs 110, and a timid
    # deepening would not probe absorber-limiting at all
    assert absb["deep"]["cpml_cells"] >= 1.5 * absb["gated"]["cpml_cells"], (
        "absorber-depth scan is too timid to test absorber limiting",
        absb["gated"]["cpml_cells"], absb["deep"]["cpml_cells"])
    assert absb["deep"]["cpml_fraction"] > absb["gated"]["cpml_fraction"]


def test_b_invariance_witness_is_measured_not_assumed(fixture):
    binv = {r["b_cells"]: r for r in fixture["b_invariance_witness"]}
    assert set(binv) == {4, 6, 8}, (
        "the b-invariance scan lost its interior sample", sorted(binv))
    bin_hz = float(np.diff(fixture["config"]["freqs_hz"])[0])
    assert abs(binv[8]["f0"] - binv[4]["f0"]) <= bin_hz
    assert abs(binv[8]["bw"] - binv[4]["bw"]) <= bin_hz


# --------------------------------------------------------------------------- #
# Prose is recomputed from the committed rows
# --------------------------------------------------------------------------- #
def test_paper_anchor_numbers_are_recomputed_from_committed_bands(fixture):
    nom = fixture["oracle_nominal_band"]
    paper = fixture["reference"]["digitized_scalars"]
    for tag in ("hfss", "cst"):
        ref = paper[tag]
        assert abs(nom["f0"] - ref["f0"]) < 40e6, tag
    assert abs(nom["f0"] - paper["cst"]["f0"]) < abs(
        nom["f0"] - paper["hfss"]["f0"]), (
        "the anchor no longer sits closer to CST, which the prose claims")
    assert len(nom["zeros"]) == 4, "the nominal design must show four zeros"



def test_script_and_fixture_claim_scope_are_the_same_text(fixture, script_src):
    """Two prose copies exist; bind them so they cannot drift.

    The merged sibling has this binding and this file lacked it, so the script's
    literal and the committed fixture's copy could diverge silently.
    """
    tree = ast.parse(script_src)
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "claim_scope":
                    found = ast.literal_eval(v)
    assert found is not None, "no claim_scope literal in the script"
    assert found == fixture["claim_scope"], (
        "the script's claim_scope and the committed one have diverged",
        len(found), len(fixture["claim_scope"]))


def test_every_population_axis_is_actually_varied(fixture):
    """Distinct axis VALUES, and interior samples strictly between the endpoints.

    Membership is keyed by label, and the labels are derived from values, so a leg
    whose configuration silently duplicates an endpoint would collapse two members
    into one — and because the producer builds the population from the same rows,
    the count check on both sides would agree and pass. This asserts the axes are
    varied in the dimension that matters rather than only in their names.
    """
    ring = sorted(r["num_periods"] for r in fixture["ring_down_witness"])
    assert len(ring) == len(set(ring)) >= 4, ring
    b = sorted(r["b_cells"] for r in fixture["b_invariance_witness"])
    assert len(b) == len(set(b)) >= 3, b

    for key, legs, field in (
            ("feed_clearance_witness", ("gated", "mid", "generous"), "port_cells"),
            ("absorber_depth_witness", ("gated", "mid", "deep"), "cpml_fraction")):
        vals = [fixture[key][leg][field] for leg in legs]
        assert len(vals) == len(set(vals)), (
            "a setup leg duplicates another in the dimension it claims to vary",
            key, field, vals)
        lo, mid_v, hi = vals
        assert lo < mid_v < hi, (
            "the interior sample is not strictly between the endpoints, so the "
            "axis is sampled at its ends only", key, field, vals)

    # and the same for the two scans, whose interior samples must interpolate
    assert ring[0] < 400.0 < 600.0 < ring[-1] or 600.0 in ring, ring
    assert b[0] < 6 < b[-1], b


def test_fdfd_witness_is_recomputed_from_its_committed_levels(fixture):
    """The formulation-independent block, re-derived rather than trusted.

    Every scalar in `fdfd_formulation_independent` is recomputed from the
    committed level curves: per-level bands via `_band`, both Richardson
    estimates via the first-order formula (h ∝ 1/r), the two-estimate
    consistency the porting handoff mandates, and the headline agreement
    numbers from the finer pair. Every level must show THREE reflection zeros:
    an earlier mask realized the apertures 2h wide and produced a spurious
    fourth zero at r=2,3, so the count is pinned per level.
    """
    fd = fixture["fdfd_formulation_independent"]
    freqs = _freqs(fixture)
    bands = {}
    for r in ("2", "3", "4"):
        lv = fd["levels"][r]
        mine = _band(lv["s11"], freqs)
        for k in ("lo", "hi", "f0", "bw"):
            assert lv["band"][k] == pytest.approx(mine[k], abs=1e-6, rel=1e-9), (r, k)
        assert len(mine["zeros"]) == 3, (
            "an FDFD level does not show three reflection zeros; either the "
            "spurious-fourth-zero mask regressed or the physics changed",
            r, len(mine["zeros"]))
        assert lv["worst_unitarity"] < 1e-6, (r, lv["worst_unitarity"])
        bands[int(r)] = mine

    rich = {}
    for tag, (ra, rb) in (("richardson_23", (2, 3)), ("richardson_34", (3, 4))):
        for k in ("lo", "hi", "f0", "bw"):
            want = (rb * bands[rb][k] - ra * bands[ra][k]) / (rb - ra)
            assert fd[tag][k] == pytest.approx(want, abs=1e-3), (tag, k)
        rich[tag] = fd[tag]

    # the handoff's two-estimate protocol: the extrapolations must agree before
    # either is trusted, and the recorded consistency must be the recomputed one
    for k in ("lo", "hi", "f0", "bw"):
        want = abs(rich["richardson_34"][k] - rich["richardson_23"][k]) / 1e6
        assert fd["richardson_consistency_mhz"][k] == pytest.approx(want, abs=5e-3), k
    assert fd["richardson_consistency_mhz"]["f0"] < 3.0, (
        "the two Richardson estimates disagree in f0 beyond trust",
        fd["richardson_consistency_mhz"])
    assert fd["richardson_consistency_mhz"]["bw"] < 5.0, (
        "the two Richardson estimates disagree in BW beyond trust",
        fd["richardson_consistency_mhz"])

    cascade = _band(fixture["gated_rfx"]["oracle_s11"], freqs)
    rfx = _band(fixture["gated_rfx"]["s11"], freqs)
    assert fd["d_f0_vs_cascade_mhz"] == pytest.approx(
        (fd["richardson_34"]["f0"] - cascade["f0"]) / 1e6, abs=5e-3)
    assert fd["d_bw_vs_cascade_mhz"] == pytest.approx(
        (fd["richardson_34"]["bw"] - cascade["bw"]) / 1e6, abs=5e-3)
    assert fd["d_f0_rfx_vs_fdfd_mhz"] == pytest.approx(
        (rfx["f0"] - fd["richardson_34"]["f0"]) / 1e6, abs=5e-3)

    # the substantive claims:
    # (a) two formulations sharing only numpy/scipy agree to a few MHz
    assert abs(fd["d_f0_vs_cascade_mhz"]) < 3.0
    assert abs(fd["d_bw_vs_cascade_mhz"]) < 3.0
    # (b) rfx differs from BOTH routes by the same amount -> residual is rfx-side
    assert abs(fd["d_f0_rfx_vs_fdfd_mhz"]
               - fixture["gated_rfx"]["d_f0_mhz"]) < 3.0
    # (c) the solver's own gates were clean at generation time
    assert fd["self_test"]["empty_s11"] < 1e-12
    assert fd["self_test"]["unitarity"] < 1e-6


def _fdfd_module():
    spec = importlib.util.spec_from_file_location(
        "fdfd_hplane", _REPO_ROOT / "validation/crossval/comparators/fdfd_hplane.py")
    fd_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd_mod)
    return fd_mod


# The gated configuration, in one place: the falsifiers below must perturb the
# SAME call the gate makes, or they falsify a different solve than the one CI
# runs.
_GATED_SELF_TEST = (A, 11.0e9, 90, 1,
                    [40, 26, 24, 26, 40], [56, 62, 62, 56], 8, 45)
_U_REFINED_BOUND = 1e-11          # U1, derived below
_U_RATIO_FLOOR = 100.0            # U2
_U_POLISH_FLOOR = 1000.0          # U3


def test_fdfd_solver_gates_run_live_and_committed_curves_are_reproducible():
    """Condition 4 of the solver's contract, executed in CI, plus one live anchor.

    The empty-guide transparency test is the gate that caught the solver's one
    real historical bug (a missing /h in the discrete propagation constant), so
    it runs here on every CI pass, not only at generation time. One live r=2
    solve at one frequency then ties the committed curves to the live solver —
    a regeneration-stable anchor that catches curve edits without any digest.

    THE UNITARITY WITNESS IS NOT A RECORDED NUMBER (#884).
    ------------------------------------------------------
    `self_test`'s `unitarity` is |S11|^2 + |S21|^2 - 1 evaluated on `spsolve`'s
    answer. cond_1(A) ~ 9.85e11 with a backward error of ~3 eps, so that number
    is five decades of LU roundoff sitting on top of the physics. It has no
    build-independent value: the four `permc_spec` orderings — which solve the
    same system and are mathematically identical — give 2.5759e-09 (COLAMD, the
    default and the committed path), 5.9404e-09, 1.7857e-08 and 4.5878e-08 on
    one machine in one process, a 1.25-decade band. Earlier out-of-tree runs
    through `spsolve(permc_spec=NATURAL)` reached 1.9689e-07, widening it to
    1.883 decades. Either way it is wider than the 1.0328-decade Python
    3.10 -> 3.11 gap that #884 was filed for, and CI-3.11's 1.5806898456816043e-08
    sits inside the band. A recorded 17-digit sample of that distribution,
    gated to one decade against a re-run, is a category error; it was faithful
    when it was written and it was never reproducible.

    Three checks replace it, none of which records a roundoff realization.

    U1 — the method's own unitarity, against a DERIVED bound.
        `refined_unitarity` refines on the same LU factor with an exactly
        accumulated residual (`math.fsum` per row); what survives is the
        discretization's unitarity, ~6e-14. The bound is derived, not fitted:
        S11 and S21 are inner products of nx - 1 = 89 O(1) terms, so their
        evaluation floor is sqrt(89) * eps ~ 2.09e-15, and u = ||S11|^2 +
        |S21|^2 - 1| inherits 2(|S11| + |S21|) = 2.408 times that, ~5.0e-15.
        The gate is 1e-11: ~2000x that analytic floor, and 159x the worst
        value measured over 4 `permc_spec` orderings x 2 refinement steps x
        2 venvs (worst by the min rule 6.2950e-14, NATURAL; worst single step
        3.3595e-13, 30x). All eight measurements are bit-identical between
        jax 0.10.2 / numpy 2.4.6 / scipy 1.17.1 and jax 0.6.2 / numpy 2.2.6 /
        scipy 1.15.3, which is the point: this quantity does not move with the
        build, and the one it replaces moves 1.25 decades (1.88 through
        `spsolve`) without anything changing at all. It is also 3.6 decades TIGHTER than the 1e-6 gate on
        the raw residual — this is not a widening.

    U2 — the diagnosis itself, as a same-run ratio.
        `u_raw / u_refined > 100` asserts that the sweep-visible residual IS
        conditioning noise. If the discretization ever stops being unitary,
        u_refined rises with u_raw, the ratio collapses, and this reading is
        flagged as expired rather than silently carried. Measured 4.15e+04
        (COLAMD) to 9.14e+05 (MMD_ATA), i.e. 415x margin at worst; taking the
        ensemble's worst numerator against its worst denominator still leaves
        min(u_raw)/max(u_refined) = 4.09e+04. It is the weakest of the three —
        no hard lower bound on u_raw is derivable — so it is set two decades
        under the worst observation and used as a corroborator.

    U3 — anti-polishing, one-sided.
        The two-sided decade test existed because an independently designed
        mutation set empty_s11 to 1e-16 and unitarity to 1e-12 and the whole
        suite stayed green. That job survives without a two-sided comparison:
        a committed LU realization of a cond ~ 1e12 solve cannot be a
        machine-eps number, so `committed > 1000 * u_refined`. Measured floor
        6.2061e-11 on the default ordering (8.1046e-12 to 6.2950e-11 across
        the four); the committed 1.4655321400880439e-09 clears the WORST of
        those by 23.3x, and the 1e-12 polishing attack fails even the lowest
        of them by 8.1x. The upper side stays the solver's own acceptance
        tolerance.

    DETECTION POWER, AND THE HONEST NEGATIVE. See
    `test_the_unitarity_witness_fires_on_loss_and_on_the_historical_defect`:
    U1 catches a lossy fill from Im(eps_r) ~ 7e-14 where the gate it replaces
    is blind until Im(eps_r) ~ 1e-6, four decades less sensitive, and the
    historical missing-/h defect fires U2 at ratio 29. But NO unitarity
    witness — this one or any other — catches a lossless geometry off-by-one:
    the first aperture at 42 cells instead of 40 is a different but still
    perfectly lossless structure, and u_refined stays at 1.93e-14. What catches
    that is the live r=2 anchor at the bottom of this test, at |delta| =
    9.553e-03 against its 1e-5 gate, 955x over. The three witnesses are not
    interchangeable: the anchor gates the geometry, unitarity gates
    losslessness, empty_s11 gates port transparency.
    """
    fd_mod = _fdfd_module()

    w = fd_mod.self_test(*_GATED_SELF_TEST)
    assert w["empty_s11"] < 1e-10, w
    assert abs(w["empty_s21"] - 1.0) < 1e-10, w
    assert w["unitarity"] < 1e-6, w

    # U1/U2: the method's unitarity, and the assertion that the raw residual is
    # conditioning noise. One extra splu + 2 triangular solves on the already
    # assembled matrix.
    u = fd_mod.refined_unitarity(*_GATED_SELF_TEST)
    assert u["unitarity_refined"] < _U_REFINED_BOUND, (
        "the discretization is no longer unitary at the arithmetic floor -- "
        "this is a physics regression, not roundoff", u)
    assert u["unitarity_raw"] / u["unitarity_refined"] > _U_RATIO_FLOOR, (
        "the raw and refined unitarity residuals have converged: the "
        "sweep-visible residual is no longer dominated by LU roundoff, so "
        "#884's reading of this witness has expired and needs re-deriving", u)

    # The COMMITTED witness scalars must not be POLISHED -- an independently
    # designed mutation set empty_s11 to 1e-16 and unitarity to 1e-12 and the
    # whole suite stayed green, because nothing compared the committed values
    # to anything. Polishing is a different attack from moving numbers, and it
    # survived every other guard.
    with open(_FIXTURE) as f:
        st = json.load(f)["fdfd_formulation_independent"]["self_test"]

    # U3: one-sided. `unitarity` is a roundoff realization (see the docstring),
    # so it is gated as a realization -- above the floor a refined solve sets,
    # below the solver's own acceptance tolerance -- and NOT compared decade-
    # wise against a re-run, which is what #884's red actually was.
    committed_u = st["unitarity"]
    assert committed_u > _U_POLISH_FLOOR * u["unitarity_refined"], (
        "the committed unitarity witness is too close to machine epsilon to be "
        "a realization of a cond ~ 1e12 factorization -- polished evidence",
        committed_u, u["unitarity_refined"])
    assert committed_u < 1e-6, ("the committed unitarity witness is outside the "
                                "solver's own acceptance tolerance", committed_u)

    # `empty_s11` KEEPS the two-sided decade test. #884's derivation covers the
    # unitarity witness only: it supplies a refined quantity, a derived bound
    # and two falsifiers for that one, and none of the three for this one. So
    # this comparison is left exactly as it was -- but it is on notice. The
    # empty-guide residual is the same class of quantity, and it spreads 0.8585
    # decades across the same four orderings (4.6699e-14 COLAMD, 7.1550e-14,
    # 1.2905e-13, 3.3711e-13 NATURAL) against this 1.0-decade gate. Committed
    # vs live here is 0.0295 decades, so the margin left is 0.03 decades on a
    # gate whose quantity legitimately moves 0.86. It has not fired only because
    # cond_1(A_empty) ~ 4.01e4 is four decades better conditioned than the
    # loaded problem. If it goes red on a new wheel, that is this latent defect
    # and not a physics change; the fix is a derivation of its own, not a wider
    # window.
    for key, live in (("empty_s11", w["empty_s11"]),):
        committed = st[key]
        assert committed > 0 and live > 0, (key, committed, live)
        decades = abs(math.log10(committed) - math.log10(live))
        assert decades < 1.0, (
            "a committed self-test witness is an order of magnitude away from "
            "the live re-run -- polished or stale evidence", key, committed, live)

    with open(_FIXTURE) as f:
        fixture = json.load(f)
    fd = fixture["fdfd_formulation_independent"]
    freqs = _freqs(fixture)
    i = 65                                    # 11.05 GHz on the 131-point grid
    s11, _, _ = fd_mod.solve(A, float(freqs[i]), 90, 2,
                             [40, 26, 24, 26, 40], [56, 62, 62, 56], 8, 45)
    assert fd["levels"]["2"]["s11"][i] == pytest.approx(abs(s11), abs=1e-5), (
        "the committed r=2 curve does not reproduce from the live solver at a "
        "spot frequency", freqs[i], fd["levels"]["2"]["s11"][i], abs(s11))


def test_the_unitarity_witness_fires_on_loss_and_on_the_historical_defect():
    """The #884 witness's detection power, measured rather than asserted.

    A gate that has never been shown to fire is a decoration. Each falsifier
    below perturbs the SAME call the gate makes and costs one extra solve of
    the gated configuration (~0.25 s each), so it runs in CI rather than in a
    notebook nobody re-runs.

    (a) LOSS. A lossy fill eps_r = 1 + i*Im is injected by scaling the
        frequency by sqrt(eps_r), which is exactly what enters k, the interior
        operator and the port DtN. u_refined tracks the absorbed power
        linearly, 1.4223e+02 * Im, so U1's 1e-11 bound puts the detection
        threshold at Im(eps_r) ~ 7.0e-14. At Im = 1e-13 u_refined = 1.4737e-11
        and U1 fires with 1.47x margin, while u_raw = 2.0864e-08 is nowhere
        near the 1e-6 gate this replaces -- that gate does not fire until
        Im(eps_r) ~ 1e-6 (u_raw = 1.4222e-04); at 1e-9 it is still passing at
        1.5338e-07. Four decades of sensitivity, against the one defect
        unitarity exists to catch.

    (b) THE HISTORICAL DEFECT. The missing /h in `discrete_gamma` -- the solver's
        one real bug, per condition 4 of its contract. It is caught twice: the
        empty guide reflects |S11| = 1.0 instead of 5e-14, and U2 fires at
        ratio 29.0 (u_raw 6.4393e-15, u_refined 2.2204e-16). Note that U1 does
        NOT fire on it: the broken structure is still lossless, which is why
        U2 is carried alongside U1 rather than dropped as redundant.

    (c) THE HONEST NEGATIVE, asserted so it cannot rot into a claim. A lossless
        geometry off-by-one -- the first aperture at 42 cells instead of 40 --
        moves u_raw by a decade but leaves u_refined at 1.9318e-14, and BOTH
        U1 and U2 pass. They should: the perturbed filter is a different but
        perfectly lossless two-port, so |S11|^2 + |S21|^2 = 1 still holds. No
        unitarity witness can catch this, and anyone reading the committed
        1.4655e-09 as evidence that the geometry is right has misread it. The
        live r=2 anchor in the test above is what catches it, 955x over its
        gate; that assertion is this one's complement.

    All numbers here are bit-identical on jax 0.10.2 / numpy 2.4.6 /
    scipy 1.17.1 and on jax 0.6.2 / numpy 2.2.6 / scipy 1.15.3.
    """
    fd_mod = _fdfd_module()
    a, freq, base, r, aps, cav, t, marg = _GATED_SELF_TEST

    def fires(u):
        return (u["unitarity_refined"] >= _U_REFINED_BOUND
                or u["unitarity_raw"] / u["unitarity_refined"] <= _U_RATIO_FLOOR)

    # (a) loss at Im(eps_r) = 1e-13 -- seven decades under where the replaced
    # gate would begin to move, and 1.4x over where this one does
    lossy = fd_mod.refined_unitarity(a, freq * cmath.sqrt(complex(1.0, 1e-13)),
                                     base, r, aps, cav, t, marg)
    assert fires(lossy), (
        "a lossy fill at Im(eps_r) = 1e-13 does not fire the unitarity "
        "witness; the losslessness gate has lost its detection power", lossy)
    assert lossy["unitarity_refined"] >= _U_REFINED_BOUND, lossy   # it is U1 that fires
    assert lossy["unitarity_raw"] < 1e-6, (
        "the gate this replaced is supposed to be blind here -- if it now "
        "fires, the four-decade sensitivity claim in the docstring is stale",
        lossy)

    # (b) the historical missing-/h defect, on the module's own entry points
    original = fd_mod.discrete_gamma
    try:
        fd_mod.discrete_gamma = lambda lam, k, h: original(lam, k, h) * h
        e11, _, _ = fd_mod.solve(a, freq, base, r, aps, cav, t, marg, empty=True)
        broken = fd_mod.refined_unitarity(a, freq, base, r, aps, cav, t, marg)
    finally:
        fd_mod.discrete_gamma = original
    assert abs(e11) == pytest.approx(1.0, abs=1e-6), (
        "the missing-/h defect no longer reflects a full wave off the empty "
        "guide; the falsifier has stopped exercising the historical bug", e11)
    assert fires(broken), (broken,)
    assert broken["unitarity_raw"] / broken["unitarity_refined"] <= _U_RATIO_FLOOR, (
        "it is U2 that catches the missing /h", broken)

    # (c) the negative: lossless geometry error, invisible to unitarity by
    # construction and caught by the r=2 anchor instead
    off_by_one = fd_mod.refined_unitarity(a, freq, base, r, [42, 26, 24, 26, 40],
                                          cav, t, marg)
    assert not fires(off_by_one), (
        "a LOSSLESS geometry perturbation now fires the unitarity witness. "
        "That is not an improvement -- it means the witness is responding to "
        "something other than loss, and the docstring's claim about what each "
        "of the three gates covers needs re-deriving", off_by_one)


def test_residual_is_reported_as_mesh_normalised_but_not_gated(fixture):
    """REPORTED: the residual expressed in cells, and why it is not a gate.

    The f0 residual is +12.08 MHz at a/90 and +19.85 MHz at a/60. Converted to a
    cavity-length offset by asking the oracle what offset nulls each one -- a
    measurement, not an application of a sensitivity coefficient -- those are
    -0.1169 and -0.1241 cell: the same fraction of a cell at two meshes, where
    Yee dispersion would have given 0.083 at the finer one.

    That is recorded rather than gated, deliberately. Two mesh points fit a
    constant within some tolerance no matter what, so a mesh-invariance gate here
    would add a second criterion with no discriminating power, not a better one --
    and tightening a number whose origin is unknown buys nothing. Making the
    mesh-invariance a claim needs a third rung (a/120). The formulation-
    independent FDFD check exists now and settles the ORACLE side of the
    residual; it says nothing about the mesh-invariance of the rfx side.

    What this test does assert is the part that cannot be misread: the residual
    grows with cell size, so it does not behave like a frequency-independent
    solver error. Bounds are loose on purpose; this is a shape check.
    """
    freqs = _freqs(fixture)
    fine, coarse = fixture["gated_rfx"], fixture.get("coarse_diagnostic")
    if not coarse:
        pytest.skip("no coarse rung committed")
    res = {}
    for row in (fine, coarse):
        r = _band(row["s11"], freqs)
        o = _band(row["oracle_s11"], freqs)
        res[row["cells_per_a"]] = (r["f0"] - o["f0"]) / 1e6
    fine_r = res[fine["cells_per_a"]]
    coarse_r = res[coarse["cells_per_a"]]
    assert 0 < fine_r < coarse_r, (
        "the residual no longer grows with cell size, so its character has "
        "changed and the reported cell-normalised figures are stale", res)
    # and it must not shrink as fast as dx^2, which would make it dispersion
    ratio = coarse_r / fine_r
    dx_ratio = fine["cells_per_a"] / coarse["cells_per_a"]
    assert ratio < dx_ratio ** 2, (
        "the residual scales like dx^2 or faster, i.e. like dispersion rather "
        "than like a fixed geometric offset", ratio, dx_ratio ** 2)


def test_paper_anchor_is_recomputed_from_the_reference_dimensions(fixture):
    """Run the oracle on the PAPER's dimensions rather than trusting the record.

    Previously this compared two committed records to each other, so fabricated
    nominal-band scalars survived.
    """
    ref = fixture["reference"]
    freqs = _freqs(fixture)
    aps = [v * 1e-3 for v in ref["apertures_mm"]]
    offs = [(A - d) / 2 for d in aps]
    ths = [ref["iris_thickness_mm"] * 1e-3] * 5
    cav = [v * 1e-3 for v in ref["cavities_mm"]]
    mine = _band([_filter_s11(A, aps, offs, ths, cav, f) for f in freqs], freqs)
    got = fixture["oracle_nominal_band"]
    for key in ("lo", "hi", "f0", "bw", "worst_rl_db"):
        assert got[key] == pytest.approx(mine[key], abs=1e-6, rel=1e-9), key
    assert len(got["zeros"]) == len(mine["zeros"]) == 4, (
        "the nominal published design must show four reflection zeros")

def test_snap_decomposition_is_recomputed(fixture):
    nom = fixture["oracle_nominal_band"]
    ras = fixture["oracle_rasterized_band"]
    snap_mhz = (ras["f0"] - nom["f0"]) / 1e6
    spread_mhz = fixture["reference"]["digitized_scalars"][
        "solver_spread_f0_hz"] / 1e6
    assert abs(snap_mhz) < spread_mhz, (
        "the gated mesh's snap no longer sits inside the reference's own "
        "solver spread; the mesh choice or the compensation regressed")
    assert len(ras["zeros"]) == 3, (
        "the rasterised design is recorded as losing exactly one of the four "
        "structural zeros; that count changed")


def test_claim_scope_prose_matches_the_committed_numbers(fixture):
    scope = fixture["claim_scope"]
    assert "PLACEHOLDER" not in scope
    g = fixture["gates"]
    assert f"{g['f0_gate_mhz']:g} MHz" in scope
    assert f"{g['f0_measured_envelope_mhz']:.4f}" in scope
    # Framing rules this stage must not quietly drop. Matched case-insensitively
    # because capitalisation is emphasis, not content.
    low = scope.lower()
    for phrase in ("topology first", "not exonerated", "snapped",
                   "as-snapped", "experimental", "regression lock",
                   "bounding zeroed node planes"):
        assert phrase in low, phrase


def test_non_gated_quantities_are_declared_non_gated(fixture, script_src):
    posture = fixture["gates"]["posture"]
    for phrase in ("worst-case RL", "ripple levels", "zero depths", "phase",
                   "band edges and bandwidth", "contiguity"):
        assert phrase in posture, phrase
    assert "GATED: centre frequency" in posture
    tree = ast.parse(script_src)
    doc = ast.get_docstring(tree) or ""
    assert "REPORTED" in doc and "GATED" in doc
    assert "EXPERIMENTAL" in doc, "the support-matrix fence left the docstring"


def test_setup_conventions_are_content_pinned(script_src):
    """The facts a future edit must not quietly drop."""
    for phrase in (
            "bounding zeroed node planes",
            "round(t/dx) + 1",
            "round(L/dx) - 1",
            "NOT a monotone",
            "EXTERIOR to the requested domain",
            "interpolated in dB",
    ):
        assert phrase in script_src, phrase


def test_operating_point_is_grid_exact_on_every_row(fixture):
    for row in _rows(fixture):
        cells = row["cells_per_a"]
        dx_mm = round(A / cells * 1e3, 4)
        assert row["dx_mm"] == dx_mm, (row["cells_per_a"], row["dx_mm"])
        for lo, hi in row["aperture_nodes"]:
            assert hi > lo, "empty aperture"


def test_passivity_is_gated_on_the_gated_row_and_bounded_elsewhere(fixture):
    """The gated row carries the tight bound; a diagnostic row only a sanity one.

    Holding a REPORTED rung to the gated threshold would let a diagnostic
    configuration fail the suite, which inverts the posture — the coarse rung is
    committed as evidence for the mesh choice, not as a claim.
    """
    # TWO SEPARATE QUESTIONS, two separate instruments (the reviewer's
    # correction to an earlier revision that had tightened the bound to 1.05):
    #
    # 1) "Is this row grossly non-passive?" -- the PHYSICAL bound, kept
    #    generous (1.02 gated / 1.10 coarse) because the coarse over-unity is
    #    a documented Yee/near-cutoff discretisation artefact that is not
    #    physically pinned: a legitimate reconfiguration could move it, and a
    #    tightened bound would then red a healthy regeneration and invite the
    #    forbidden loosening dynamic. Measured reach of the 1.10 bound under
    #    coherent editing: it stops discriminating below ~x1.04 of coarse
    #    |S11| inflation.
    # 2) "Has this row been ALTERED?" -- an EXACT pin of the committed value,
    #    which catches x1.005 (recomputed colpow 1.0418), far tighter than any
    #    defensible tolerance, with zero over-fit risk: a legitimate
    #    regeneration re-pins it deliberately in the same commit, exactly as
    #    _PIN_TRACE_SHA256 works. Same guarantee class as the digest --
    #    alteration becomes deliberate and visible, not impossible.
    assert fixture["gated_rfx"]["max_colpow"] <= 1.02
    assert fixture["gated_rfx"]["max_colpow"] == pytest.approx(
        _PIN_GATED_MAX_COLPOW, abs=1e-6)
    coarse = fixture.get("coarse_diagnostic")
    if coarse:
        assert coarse["max_colpow"] <= 1.10, coarse["max_colpow"]
        assert coarse["max_colpow"] == pytest.approx(
            _PIN_COARSE_MAX_COLPOW, abs=1e-6)


# --------------------------------------------------------------------------- #
# Hard numeric pins — filled from the committed fixture, never re-tuned.
# --------------------------------------------------------------------------- #
_PIN_F0_GATE_MHZ = 19.0
_PIN_GATED_MAX_COLPOW = 1.0065
_PIN_COARSE_MAX_COLPOW = 1.0315
_PIN_F0_ENV_MHZ = 12.1230
_PIN_TRACE_SHA256 = (
    "25faa7447a1451f81c578fbbd73b0c7c256e2dbba4fae46696a79c885eff3044")
