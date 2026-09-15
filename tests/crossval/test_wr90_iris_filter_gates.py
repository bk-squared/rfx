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
    gated: they move ~22-40 MHz per cell of lattice rounding against f0's
    ~2.4 MHz, so a gate on them would pin the mesh choice, not the solver.
    (#931 note: the original reason was that the iris-thickness leg of the
    node-plane convention was UNSETTLED at the half-cell level -- the builder
    assumed (t_c - 1)*dx, four FDTD runs measured (t_c - 0.68)*dx, and cv18
    used t_c*dx for the same object. The lattice ownership contract settles
    it at t_c*dx for both cases. What is left is lattice rounding, which is
    smaller but still dominant for these two quantities; re-gating them needs
    its own pre-declaration, not this migration.)
  * GATED (setup, not physics): the ring-down, feed-clearance and
    absorber-depth witnesses must each hold to one frequency bin. A resonant
    band read off an unsettled or absorber-limited run is not a measurement.
  * REPORTED, never gated: worst in-band return loss (the reference's own two
    solvers disagree by 0.7 dB on it and on which ripple peak is worst),
    individual ripple levels, every zero DEPTH, the coarse a/60 rung, phase.
  * THE COMPARATOR'S INPUTS ARE GATED TOO, and this is the lesson of this
    stage: the oracle must be fed the geometry that was BUILT, not the geometry
    that was drawn. Under the pre-#931 realization those were two different
    things, and confusing them biased f0 by +107.5 MHz -- FIVE times the
    reference's own 21.9 MHz CST-vs-HFSS spread. The envelope-times-1.5 rule
    does not catch that, because it bounds SCATTER and this is BIAS: it
    launders the bias into a ~162 MHz "measured" gate, 46% of the passband,
    which pins nothing. The lattice ownership contract removes the gap at its
    source -- drawn IS realized (design note §1.2), and the builder's
    t_c = round(t/dx) + 1 / L_c = round(L/dx) - 1 compensations are deleted
    rather than re-tuned. The electrical cell counts are still re-derived HERE
    from the committed node indices, and the oracle's SENSITIVITY to one cell
    of geometry is still measured rather than quoted, because that sensitivity
    is what makes the f0 gate mean anything.

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


# --------------------------------------------------------------------------- #
# #931 lattice ownership — the migration gate for this case
# --------------------------------------------------------------------------- #
# Case 19 carried the largest compensation family in the repo: the builder drew
# t_c = round(t/dx) + 1 cells of iris and L_c = round(L/dx) - 1 cells of cavity
# so that the OLD realization -- one wall per masked cell at that cell's lower
# node plane, the far face never zeroed -- would land the ELECTRICAL dimensions
# on nominal. Under the ownership contract a volume realizes walls on both
# drawn faces and the realized thickness is the drawn thickness (design note
# §1.2), so both compensations are deleted from the builder and the oracle
# takes the drawn value.
#
# The fixture is regenerated by the crossval-D migration
# (`19_wr90_iris_filter_aghanim.py --write-fixture`, VESSL run named
# `rfx-931-post-cv19`). Until that artifact is ingested the committed fixture
# still records the compensated geometry, so the identity tests below cannot
# pass and cannot be made to pass by editing a number. They SKIP on the old
# fixture, naming the run, and go live the moment the new one lands.
_MIGRATION_RUN = ("VESSL rfx-931-post-cv19 — "
                  "validation/crossval/19_wr90_iris_filter_aghanim.py "
                  "--write-fixture on the migrated builder (crossval-D)")


def _fixture_is_post_931(fx) -> bool:
    """True when the committed fixture was produced by the migrated builder.

    The discriminator is the compensation itself, not a schema flag: post-#931
    the electrical counts ARE the drawn counts. On the pre-#931 artifact they
    differ by exactly the (+1, -1) pair (8 vs 9 cells of iris, 56/62 vs 55/61
    cells of cavity). A regenerated fixture that drops the ``drawn_*`` keys
    altogether also reads as post-#931, which is the right answer.
    """
    eg = fx["electrical_geometry"]
    t = int(eg["iris_thickness_cells"])
    cav = [int(v) for v in eg["cavity_cells"]]
    t_drawn = int(eg.get("drawn_iris_thickness_cells", t))
    cav_drawn = [int(v) for v in eg.get("drawn_cavity_cells", cav)]
    return t == t_drawn and cav == cav_drawn


def _require_post_931(fx) -> None:
    if not _fixture_is_post_931(fx):
        pytest.skip(
            "committed cv19 fixture still carries the pre-#931 compensation "
            "(t_c = round(t/dx) + 1, L_c = round(L/dx) - 1); regenerate it "
            f"first: {_MIGRATION_RUN}")


# The numeric pins below the fold (_PIN_F0_ENV_MHZ, _PIN_F0_GATE_MHZ,
# _PIN_TRACE_SHA256, the colpow pair) must be re-pinned from the regenerated
# artifact in the same commit that ingests it -- never re-tuned, never widened.
# Flip this flag in that commit; until then the pin checks skip on a post-#931
# fixture and say why, instead of going red for a reason no one can fix by
# editing a test.
#
# DONE: cv19 pass 2 (VESSL 369367259297, output issue931-post-cv19-20260907T194954Z,
# rc 0 on 2026-09-07, log ends "RESULT: ALL CHECKS PASSED") is the committed
# fixture, and the four pins below are read back out of it. Only one of them
# moved: the f0 envelope 12.1230 -> 12.1219 MHz, which leaves the gate at 19.0
# because ceil(12.1219 x 1.5 = 18.18285) = 19.0. The digest moved because every
# trace was re-solved. The two colpow pins did not move at all.
_PINS_REPINNED_FOR_931 = True


def _require_repinned(fx) -> None:
    if _fixture_is_post_931(fx) and not _PINS_REPINNED_FOR_931:
        pytest.skip(
            "the fixture is the regenerated #931 artifact but the hard "
            "numeric pins in this file still hold the compensated-geometry "
            "values. Re-pin them from the new artifact and set "
            f"_PINS_REPINNED_FOR_931 = True in the SAME commit ({_MIGRATION_RUN}).")


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
            # The three deltas are stored rounded to 0.01 MHz, so the identity
            # can only be checked to the record's own precision: three
            # independent roundings at half a unit each = 0.015. The unrounded
            # identity is exact, and the recomputation just above already
            # checks each field against the traces at 5e-3; a tighter bound
            # here is a claim about the rounding grid, not about the case.
            # cv19 pass 2 lands the coarse rung exactly on that edge:
            # 15.24 - 24.51 = -9.27 against a recorded -9.26, from the
            # unrounded 15.2447 - 24.5050 = -9.2603.
            assert row["d_bw_mhz"] == pytest.approx(
                row["d_hi_mhz"] - row["d_lo_mhz"], abs=1.5e-2)


def _zeros_interpolated(curve, freqs, lo, hi):
    """Reflection-zero frequencies with the grid quantisation removed.

    The committed `zeros` are raw grid samples, which is correct for the COUNT --
    an integer is quantisation-insensitive -- but not for comparing frequencies:
    on a 10 MHz grid two traces whose zeros are physically 12 MHz apart can land
    one or two bins apart, reading 10 or 20 MHz. Measured here: sampled diffs
    20.0/20.0/10.0 MHz against interpolated 15.9/16.8/11.2, so quantisation was
    contributing about 3 MHz of the discrepancy and the interpolated values sit
    where the f0 residual (+12.12 MHz) says they should.

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
    residual-dominated -- 0.02 MHz spread against a 12.12 MHz residual, measured
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
    nodes = _aperture_wall_nodes(fixture["gated_rfx"])
    aps_mm = fixture["reference"]["apertures_mm"]
    # #931: these are the two innermost realized WALL planes, so the aperture
    # is hi - lo cells exactly and lo is the fin's inner face. Until #931 the
    # committed pair was the first and last OPEN node (lo = fin_c + 1,
    # hi = fin_c + d_c - 1), which is the same geometry counted one cell in
    # from each wall.
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
        assert lo == fin_c, ("aperture wall is not where the builder puts it",
                             lo, fin_c, d_mm)
        assert hi == lo + d_c, ("realized aperture width does not match the "
                                "reference", hi, lo + d_c, d_mm)


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
    _require_repinned(fixture)
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
    # The derived relation is contract-independent and stays live at every
    # stage: whatever the envelope measures, the gate is that envelope through
    # the ONE shared policy (tests/_gate_policy.gate_from_envelope), never a
    # number typed beside it.
    assert g["f0_gate_mhz"] == gate_from_envelope(
        max(g["f0_measured_envelope_mhz"], 1e-9), quantum=1)
    # band edges and bandwidth stay REPORTED. #931 settled the iris-thickness
    # convention, so the ORIGINAL reason (a ~1/3-cell comparator-input
    # ambiguity) is gone -- but the quantity it justified has not become
    # gateable by that alone. Edges and bandwidth still move ~22-40 MHz per
    # cell of lattice rounding against f0's ~2.4 MHz, i.e. they are dominated
    # by where the reference dimensions fall between node lines rather than by
    # solver error, and gating them would pin the mesh choice. Re-gating them
    # is now possible in principle and needs its own pre-declaration and
    # sensitivity measurement, which is a separate piece of work, not a
    # side effect of this migration.
    assert "edge_gate_mhz" not in g and "bw_gate_mhz" not in g, (
        "edges/BW were re-gated; #931 settled the thickness convention but "
        "their ~22-40 MHz-per-cell lattice sensitivity against f0's ~2.4 MHz "
        "still needs its own pre-declaration before a gate means anything")
    _require_repinned(fixture)
    assert g["f0_gate_mhz"] == _PIN_F0_GATE_MHZ
    assert g["f0_measured_envelope_mhz"] == pytest.approx(_PIN_F0_ENV_MHZ, abs=1e-4)


def test_edge_and_bw_evidence_is_committed_and_the_gate_is_refused_on_purpose(fixture):
    """The edges/BW posture, decided by measurement and recorded as arithmetic.

    Before #931 this case gave one reason for not gating band edges and
    bandwidth: a ~1/3-cell ambiguity about what the lattice actually built,
    which made the comparator input uncertain. The contract removed that
    reason, and the migration then had a second one -- a gate is
    round-UP(measured envelope x 1.5) and no such envelope existed until the
    record was regenerated. Pass 2 supplies it, so both of the old reasons are
    discharged and the question had to be answered rather than deferred.

    Answer: still NOT gated, for a THIRD reason that the envelope does not
    touch. The nine-configuration population is single-mesh -- every member is
    a/90 -- and the axes it varies (guide height, run length, port standoff,
    absorber depth) are the axes these two observables are insensitive to. They
    move ~22-40 MHz per cell of lattice rounding against f0's ~2.4 MHz, and the
    a/60 diagnostic rung reads +24.5 / +15.2 MHz on the same quantities, which
    is the scale of the term this population cannot see. A 1.5x lock over a
    population blind to the dominant term locks the mesh choice, not the
    solver. Re-gating needs its own pre-declaration and a cross-mesh
    sensitivity measurement.

    What IS committed is the arithmetic of the gate that is not applied, so a
    future pre-declaration starts from a number someone checked rather than
    re-measuring from scratch -- and so that "not gated" cannot quietly become
    "not measured".
    """
    _require_post_931(fixture)
    g = fixture["gates"]
    pop = g["edge_bw_envelope_population"]
    # the same population as the f0 envelope, member for member and in order:
    # a subset would let a would-be gate be quoted off a friendlier set
    assert [r["config"] for r in pop] == [
        r["config"] for r in g["f0_envelope_population"]]
    assert len(pop) == 9
    # envelopes are the max over that population, not a typed-in number
    assert g["edge_measured_envelope_mhz"] == pytest.approx(
        max(max(abs(r["d_lo_mhz"]), abs(r["d_hi_mhz"])) for r in pop), abs=1e-4)
    assert g["bw_measured_envelope_mhz"] == pytest.approx(
        max(abs(r["d_bw_mhz"]) for r in pop), abs=1e-4)
    # and the would-be gates are that envelope through the ONE shared policy
    would = g["edge_bw_gate_would_be_mhz"]
    assert would["applied"] is False
    assert would["edges"] == gate_from_envelope(
        g["edge_measured_envelope_mhz"], quantum=1)
    assert would["bw"] == gate_from_envelope(
        g["bw_measured_envelope_mhz"], quantum=1)
    # the refusal must carry its reason with it; a bare applied=false would let
    # the posture drift back to "nobody got round to it"
    assert "single-mesh" in would["why_not"], would["why_not"]
    # and it must stay a would-be: the live gate keys are still absent
    assert "edge_gate_mhz" not in g and "bw_gate_mhz" not in g


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


def test_zero_count_gate_is_robust_across_a_half_cell_of_geometry(fixture):
    """The gated integer must survive half a cell of geometric uncertainty.

    Joint-review N3 asked this question of an ambiguity that no longer exists.
    The iris-thickness electrical leg used to be an unsettled CONVENTION --
    the builder assumed ``(t_c - 1)*dx`` while four FDTD runs measured a flat
    ``(t_c - 0.68)*dx`` offset, and cv18 used ``t_c*dx`` for the same physical
    object -- so the sweep ran 8.00 to 8.50 cells to show the gated zero count
    did not depend on which convention the comparator picked. The contract
    settles it: the electrical thickness is ``t_c*dx``, exactly, and both
    cases now say so.

    What remains uncertain is not the convention but the LATTICE: a reference
    dimension lands within half a cell of a node line, and that half cell is
    real. So the sweep is re-centred on the realized thickness and spans
    ``±0.5`` cell around it. The property is unchanged and still worth
    gating -- an integer that moved inside half a cell of rasterization would
    be an artefact of where the mesh fell, not a structural count.
    """
    _require_post_931(fixture)
    rows = fixture["iris_thickness_zero_count_sweep"]["rows"]
    eg = fixture["electrical_geometry"]
    t_c = float(eg["iris_thickness_cells"])
    gated_count = len(fixture["oracle_rasterized_band"]["zeros"])
    lo = float(rows[0]["t_elec_cells"])
    hi = float(rows[-1]["t_elec_cells"])
    assert lo == pytest.approx(t_c - 0.5, abs=1e-9), (
        "the sweep is not centred on the realized iris thickness; it still "
        "spans the retired convention's band")
    assert hi == pytest.approx(t_c + 0.5, abs=1e-9), (lo, hi, t_c)
    assert len(rows) >= 11, len(rows)
    assert all(r["zeros"] == gated_count for r in rows), rows
    # the sweep must actually exercise the input: bandwidth moves across the
    # band while the gated integer stays fixed
    assert abs(rows[0]["bw_hz"] - rows[-1]["bw_hz"]) > 10e6
    cfg = fixture["config"]
    freqs = np.asarray(cfg["freqs_hz"], dtype=float)
    dx = A / cfg["gated_cells_per_a"]
    aps, offs = _aps_offs(fixture, dx)
    mid = rows[len(rows) // 2]
    mine = _band([_filter_s11(A, aps, offs, [mid["t_elec_cells"] * dx] * 5,
                              [c * dx for c in eg["cavity_cells"]], f)
                  for f in freqs], freqs)
    assert len(mine["zeros"]) == mid["zeros"]
    assert mine["f0"] == pytest.approx(mid["f0_hz"], abs=2e6)
    assert mine["bw"] == pytest.approx(mid["bw_hz"], abs=3e6)


def test_electrical_geometry_is_rederived_from_committed_node_indices(fixture):
    """Re-derive the oracle's lengths from the realized wall planes, independently.

    #931 lattice ownership contract: a PEC volume drawn on node planes
    realizes tangential walls at BOTH faces and shorts every normal edge
    between them, so the distance between an iris's two bounding wall planes
    IS its drawn cell count, and the clear space between consecutive irises IS
    the drawn cavity. This test asserts that identity from the committed node
    indices, with no rule of its own.

    Until #931 the far face was never a wall, so a cavity drawn with L_c cells
    realized (L_c + 1)*dx and an iris drawn t_c cells realized (t_c - 1)*dx;
    the case compensated in the drawn counts and this test hard-coded the
    -1/+1. That is what made a rule change invisible: the pin agreed with the
    compensation instead of with the geometry.
    #931 §1.2: an iris drawn ``t_c`` cells thick realizes tangential walls on
    BOTH drawn faces, so its electrical thickness is ``t_c*dx``; the clear
    cavity between two irises is the ``L_c*dx`` that was drawn. Drawn equals
    realized, on every region type, with no ``±1``.

    That is not a re-tuning of the old identities, it is their deletion. The
    old rule stood one wall per masked cell at that cell's LOWER node plane
    and never zeroed the far face, which made an iris one cell electrically
    thin and a cavity one cell electrically long -- opposite signs, which is
    why total length was conserved only as ``span - 1``. Under the contract
    the two faces of every region are walls and the total closes on ``span``.
    """
    _require_post_931(fixture)
    eg = fixture["electrical_geometry"]
    row = fixture["gated_rfx"]
    x_runs = [tuple(r) for r in row["iris_wall_nodes"]]
    assert len(x_runs) == 5

    th_cells = [hi - lo for lo, hi in x_runs]
    cav_cells = [x_runs[i + 1][0] - x_runs[i][1] for i in range(4)]
    assert th_cells == [eg["iris_thickness_cells"]] * 5, (th_cells, eg)
    assert cav_cells == list(eg["cavity_cells"]), (cav_cells, eg)

    drawn_t = int(eg.get("drawn_iris_thickness_cells", eg["iris_thickness_cells"]))
    drawn_L = [int(v) for v in eg.get("drawn_cavity_cells", eg["cavity_cells"])]
    assert th_cells == [drawn_t] * 5, (
        "an iris realizes the thickness it is drawn with (#931 §1.2); a "
        "(t_c - 1) electrical thickness is the deleted compensation")
    assert cav_cells == drawn_L, (
        "a cavity realizes the clear span it is drawn with; a (L_c + 1) "
        "electrical length is the deleted compensation")

    # Total length is now plain addition, and so is the outer extent: five
    # irises plus four cavities, first wall plane to last. The pre-#931 form
    # carried a `span - 1` and a face-continuity argument to close.
    span = drawn_t * 5 + sum(drawn_L)
    assert sum(th_cells) + sum(cav_cells) == span, (
        "total length not conserved: under the contract the faces of adjacent "
        "regions are the SAME wall planes, so the parts sum to the whole "
        "exactly -- the old rule's 'span - 1' was the missing far face")
    assert x_runs[-1][1] - x_runs[0][0] == span, "outer extent not conserved"


def test_no_compensation_is_applied_anywhere_in_the_case(fixture, script_src):
    """t_c = round(t/dx) and L_c = round(L/dx) — the +1/-1 is gone, and stays gone.

    Replaces ``test_drawn_counts_are_the_electrical_space_compensation``, which
    asserted ``round(t/dx) + 1`` and ``round(L/dx) - 1`` as the rule: the
    builder drew a thicker iris and a shorter cavity so that a realization
    which lost one wall per region would land the electrical dimensions on
    nominal. Two errors chosen to cancel. Under the contract there is nothing
    to compensate, so the pin inverts: the drawn counts must be the plain
    roundings, the realized geometry must equal them, the only residual is the
    honest half cell of rounding onto the lattice, and the SOURCE must not
    reintroduce a compensating term.
    """
    _require_post_931(fixture)
    eg = fixture["electrical_geometry"]
    ref = fixture["reference"]
    cfg = fixture["config"]
    dx = A / cfg["gated_cells_per_a"]
    drawn_t = int(eg.get("drawn_iris_thickness_cells", eg["iris_thickness_cells"]))
    drawn_L = [int(v) for v in eg.get("drawn_cavity_cells", eg["cavity_cells"])]
    assert drawn_t == round(ref["iris_thickness_mm"] * 1e-3 / dx)
    assert drawn_L == [round(v * 1e-3 / dx) for v in ref["cavities_mm"]]
    assert eg.get("compensation", "none").lower().startswith("none")

    # the realized dimensions are the drawn ones, on every leg
    assert eg["iris_thickness_cells"] == drawn_t
    assert list(eg["cavity_cells"]) == drawn_L
    if "drawn_aperture_cells" in eg:
        assert list(eg["aperture_cells"]) == list(eg["drawn_aperture_cells"])

    # The transverse aperture follows the same rule: the fins' inner faces are
    # walls, so the clear aperture is the gap BETWEEN them and the old
    # (n_open + 1) node count -- which counted the fin face the rule never
    # zeroed -- is gone with it. cv18 drew the same WR-90 inductive iris with
    # a different convention (aperture - 1, thickness with no + 1); one rule
    # now covers both, which is the cross-case inconsistency the contract
    # removes. Read from the wall planes (schema 2 ``aperture_wall_nodes``, or
    # the schema-1 open-node pair widened to its bounding walls); the bound is
    # the producer's own rounding, dx/2. A full-dx bound once admitted an
    # outer-aperture +1-node mutation that would have improved d_lo from
    # 17.08 to 6.76 MHz.
    for (lo, hi), d_mm in zip(_aperture_wall_nodes(fixture["gated_rfx"]),
                              ref["apertures_mm"]):
        realized_mm = (hi - lo) * dx * 1e3
        assert abs(realized_mm - d_mm) <= 0.5 * dx * 1e3 + 1e-9, (realized_mm, d_mm)

    # No compensating term may come back into the builder. Checked on the
    # SOURCE OF `rasterized_geometry` only — the historical prose elsewhere in
    # the script quotes the retired rule on purpose, and a whole-file grep
    # would refuse the history along with the mechanism.
    tree = ast.parse(script_src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "rasterized_geometry")
    assigned = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            assigned.setdefault(node.targets[0].id, ast.unparse(node.value))
    for name in ("t_c", "L_c", "d_c"):
        expr = assigned.get(name)
        assert expr is not None, name
        assert "+ 1" not in expr and "- 1" not in expr, (
            "a drawn-count compensation reappeared in the builder", name, expr)

def test_one_cell_of_geometry_moves_f0_more_than_the_solver_spread(fixture):
    """The comparator's sensitivity is measured here, not asserted in prose.

    This test used to measure the COST of the compensation: it ran the oracle
    on the realized counts and again on the drawn counts and confirmed the
    +107.5 MHz f0 bias between them -- five times the reference's own 21.9 MHz
    CST-vs-HFSS spread. Under the contract drawn IS realized, so that pair of
    evaluations is the same evaluation and the bias is exactly zero; keeping
    the old assertion would have made it vacuous, and loosening its 2x-spread
    bound would have hidden that.

    What survives, and is the property the gate actually rests on: the oracle
    must be SENSITIVE to one cell of geometry. So the second evaluation is now
    the retired convention itself -- an iris one cell electrically thin and a
    cavity one cell electrically long, which is exactly what the pre-#931 rule
    realized -- and the measured f0 shift must still exceed twice the paper's
    own solver spread. If it does not, a one-cell realization error is inside
    the reference's noise and no f0 gate here means anything.
    """
    eg = fixture["electrical_geometry"]
    cfg = fixture["config"]
    freqs = np.asarray(cfg["freqs_hz"], dtype=float)
    dx = A / cfg["gated_cells_per_a"]
    aps, offs = _aps_offs(fixture, dx)

    t_cells = eg["iris_thickness_cells"]
    cav_cells = [int(c) for c in eg["cavity_cells"]]
    real = _band([_filter_s11(A, aps, offs, [t_cells * dx] * 5,
                              [c * dx for c in cav_cells], f)
                  for f in freqs], freqs)
    # The pre-#931 realization, re-expressed: every region loses its far wall,
    # so each iris is a cell thin and each cavity a cell long.
    old_rule = _band([_filter_s11(A, aps, offs, [(t_cells - 1) * dx] * 5,
                                  [(c + 1) * dx for c in cav_cells], f)
                      for f in freqs], freqs)
    shift_mhz = (old_rule["f0"] - real["f0"]) / 1e6
    spread_mhz = fixture["reference"]["digitized_scalars"][
        "solver_spread_f0_hz"] / 1e6
    assert abs(shift_mhz) > 2 * spread_mhz, (
        f"one cell of realization moves f0 by {shift_mhz:.1f} MHz against a "
        f"{spread_mhz:.1f} MHz reference solver spread — below 2x, this "
        "case's f0 gate cannot distinguish a geometry error from the "
        "reference's own disagreement")


def test_the_pre_931_drawn_vs_realized_confusion_is_recorded_as_history(fixture):
    """The +107.5 MHz cost stays in the record; the mechanism that caused it does not.

    Replaces ``test_using_drawn_counts_would_bias_f0_and_is_recorded_as_such``,
    whose premise is now vacuous: drawn == realized, so evaluating the oracle
    on "the drawn counts" and "the realized geometry" gives the same number by
    construction, and the test would pass at bias 0 while asserting 107.5.
    (The sensitivity property it used to carry lives on in the test above.)

    What survives is the historical figure and the reason it mattered — a
    +107.5 MHz f0 bias is five times the reference's own CST-vs-HFSS spread,
    and an envelope-times-1.5 gate bounds SCATTER, not BIAS. The check here is
    that the record still says so, and that the two legs really are identical
    now.
    """
    _require_post_931(fixture)
    eg = fixture["electrical_geometry"]
    cfg = fixture["config"]
    freqs = np.asarray(cfg["freqs_hz"], dtype=float)
    dx = A / cfg["gated_cells_per_a"]
    aps, offs = _aps_offs(fixture, dx)

    t_cells = eg["iris_thickness_cells"]
    cav_cells = [int(c) for c in eg["cavity_cells"]]
    real = _band([_filter_s11(A, aps, offs, [t_cells * dx] * 5,
                              [c * dx for c in cav_cells], f)
                  for f in freqs], freqs)
    drawn_t = int(eg.get("drawn_iris_thickness_cells", t_cells))
    drawn_L = [int(v) for v in eg.get("drawn_cavity_cells", cav_cells)]
    drawn = _band([_filter_s11(A, aps, offs, [drawn_t * dx] * 5,
                               [c * dx for c in drawn_L], f)
                   for f in freqs], freqs)
    assert drawn["f0"] == real["f0"], (
        "drawn and realized geometry no longer agree — the contract is broken "
        "somewhere upstream of this fixture")

    spread_mhz = fixture["reference"]["digitized_scalars"][
        "solver_spread_f0_hz"] / 1e6
    recorded = eg["cost_of_using_intended_counts_mhz"]
    assert abs(recorded) > 2 * spread_mhz
    assert "HISTORICAL" in eg["cost_note"]


def _aperture_wall_nodes(row):
    """The five apertures as (lo, hi) WALL-plane index pairs, either schema.

    schema 2 (crossval-D regeneration) commits ``aperture_wall_nodes``
    directly. schema 1 committed ``aperture_nodes`` = the first/last OPEN node
    between two fins; under the contract the fins' inner faces -- both of them
    walls -- are at ``lo - 1`` and ``hi + 1``, so the pair is widened to its
    bounding walls. Same geometry, one index convention fewer.
    """
    if "aperture_wall_nodes" in row:
        return [tuple(p) for p in row["aperture_wall_nodes"]]
    return [(lo - 1, hi + 1) for lo, hi in row["aperture_nodes"]]


def _aps_offs(fixture, dx):
    """Apertures and their left offsets, from the committed WALL-plane indices.

    #931: an aperture is the distance between its two bounding realized wall
    planes, so the width is (hi - lo)*dx and the left offset is lo*dx. The
    pre-#931 form read a first/last OPEN node pair and added the two half
    cells back by hand ((hi - lo + 2), (lo - 1)); ``_aperture_wall_nodes``
    now does that translation for a schema-1 fixture, so the arithmetic here
    is the contract's, whichever record is committed.
    """
    aps, offs = [], []
    for lo, hi in _aperture_wall_nodes(fixture["gated_rfx"]):
        aps.append((hi - lo) * dx)
        offs.append(lo * dx)
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

    DETECTION POWER, AND TWO HONEST NEGATIVES. See
    `test_the_unitarity_witness_fires_on_loss_and_on_the_historical_defect`.
    What U1 and U2 are demonstrated to catch is LOSS: U1 from Im(eps_r) ~
    7e-14, where the gate it replaces is blind until Im(eps_r) ~ 1e-6 — four
    decades less sensitive — and U2 from Im(eps_r) ~ 1e-9, where the absorbed
    power dominates the roundoff in both residuals. What they do NOT catch:

      * a LOSSLESS GEOMETRY off-by-one. The first aperture at 42 cells instead
        of 40 is a different but still perfectly lossless structure, and
        u_refined stays at 1.93e-14. No unitarity witness can see it. The live
        r=2 anchor at the bottom of this test is what does, at |delta| =
        9.553e-03 against its 1e-5 gate, 955x over.
      * the historical MISSING-/h defect, for the same reason: it turns the
        structure into a total reflector (|S11| = 0.99999980, |S21| = 6.28e-04),
        which is also perfectly lossless. Both residuals collapse to
        cancellation dust — 2.2204e-16 on macOS arm64, exactly 0.0 on Linux
        x86-64 — so whether U2's ratio happens to land under 100 is a property
        of the build, not of the defect. `empty_s11` = 1.0 is that defect's only
        witness, which is why the empty-guide comparison below is load-bearing
        and was not touched.

    The three witnesses are not interchangeable: the anchor gates the geometry,
    unitarity gates losslessness, empty_s11 gates port transparency.
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
    # Written as a product, not a quotient: `u_refined` can be exactly 0.0 on
    # some builds (it did come back 0.0 from Linux x86-64 CI on a degenerate
    # structure), and an infinite ratio is the strongest possible pass of this
    # check, not a crash.
    assert u["unitarity_raw"] > _U_RATIO_FLOOR * u["unitarity_refined"], (
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

    (a) LOSS -- the class both U1 and U2 are shown to catch. A lossy fill
        eps_r = 1 + i*Im is injected by scaling the frequency by sqrt(eps_r),
        which is exactly what enters k, the interior operator and the port DtN.
        u_refined tracks the absorbed power linearly, 1.4223e+02 * Im, so U1's
        1e-11 bound puts the detection threshold at Im(eps_r) ~ 7.0e-14.

        At Im = 1e-13, u_refined = 1.4737e-11 and U1 fires with 1.47x margin,
        while u_raw = 2.0864e-08 is nowhere near the 1e-6 gate this replaces --
        that gate does not fire until Im(eps_r) ~ 1e-6 (u_raw = 1.4222e-04); at
        1e-9 it is still passing at 1.5338e-07. Four decades of sensitivity,
        against the one defect unitarity exists to catch.

        1.47x is thin, so it is worth being precise about what it is thin
        against. u_refined here is absorbed power, not cancellation dust: over
        the four `permc_spec` orderings it reads 1.4676e-11 / 1.4737e-11 /
        1.4700e-11 / 1.4806e-11 -- 0.88% total spread, where u_raw at the same
        point spreads 0.85 decades. A build cannot move a signal that stiff by
        the 32% it would take to drop under the bound, and it cannot collapse
        it to zero the way it can a structurally-zero residual (see (b)).
        Im = 1e-9 is asserted as well, where u_refined = 1.4223e-07 clears the
        bound by 14223x with 0.0002% spread and U2 fires too, at ratio 1.078
        (0.97 to 2.12 across orderings) -- that is U2's falsifier.

    (b) THE HISTORICAL DEFECT, AND THE CORRECTION THAT THIS TEST'S FIRST
        REVISION GOT WRONG. The missing /h in `discrete_gamma` is the solver's
        one real bug, per condition 4 of its contract. **`empty_s11` is its only
        witness. The unitarity witness does not catch it on any platform, and
        the first revision of this test claimed otherwise on the strength of a
        Mac-only measurement.**

        What the defect actually produces is a TOTAL REFLECTOR: the loaded
        solve gives |S11| = 0.9999998026 and |S21| = 6.2829e-04. A total
        reflector is perfectly lossless, so |S11|^2 + |S21|^2 - 1 = -6.44e-15
        is satisfied by construction and BOTH residuals are cancellation dust
        at the last bit. There is no signal for any unitarity witness to see --
        this is structural, not a threshold that could be tightened.

        The dust is then whatever the arithmetic leaves, which is exactly what
        varies by build:

            venv/build      u_raw        u_refined    ratio    U2?
            macOS arm64     6.4393e-15   2.2204e-16   29.0     fires
              (both venvs, COLAMD; other orderings 61.5 / 7.0 / 18.0 --
               u_refined is 1 eps in all four, so the "ratio" is an ulp count)
            Linux x86-64    5.7732e-15   0.0          inf      does NOT fire
              (PR #887 CI, run 33723124437, shard fast-suite (1))

        The Mac ratio of 29 was one ulp count landing under a threshold of 100;
        MMD_AT_PLUS_A already gives 61.5 on the same machine, and Linux gives
        exactly 0.0 for u_refined, i.e. an infinite ratio. So this test asserts
        `empty_s11` = 1.0 -- which holds everywhere and is the real guard --
        plus the total-reflector structure and the dust magnitude, which are
        the platform-independent reasons the unitarity witness is blind. It
        asserts NOTHING about U1/U2 firing here, in either direction, because
        the answer is genuinely build-dependent noise.

        U1 and U2 are still both carried: (a) shows U1 firing where U2 does not
        (Im = 1e-13) and both firing at Im = 1e-9, so neither is redundant.

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
    scipy 1.17.1 and on jax 0.6.2 / numpy 2.2.6 / scipy 1.15.3 -- BUT both are
    macOS arm64 / Accelerate. Two local venvs agreeing is evidence about
    library versions, not about platforms, and (b) is the case that proves the
    difference matters. Every assertion here is therefore written to hold on a
    quantity that is a signal rather than on one that is arithmetic dust.
    """
    fd_mod = _fdfd_module()
    a, freq, base, r, aps, cav, t, marg = _GATED_SELF_TEST

    def fires(u):
        # product form, never a quotient -- see (b): `u_refined` is exactly 0.0
        # on Linux x86-64 for the missing-/h structure, and an infinite ratio
        # means U2 does NOT fire rather than raising ZeroDivisionError.
        return (u["unitarity_refined"] >= _U_REFINED_BOUND
                or u["unitarity_raw"] <= _U_RATIO_FLOOR * u["unitarity_refined"])

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

    # (a2) loss at Im(eps_r) = 1e-9, where both quantities are signal-dominated
    # and U2 fires too. This is U2's falsifier; see (b) for why the missing-/h
    # defect is NOT.
    loud = fd_mod.refined_unitarity(a, freq * cmath.sqrt(complex(1.0, 1e-9)),
                                    base, r, aps, cav, t, marg)
    assert loud["unitarity_refined"] >= _U_REFINED_BOUND, loud
    assert loud["unitarity_raw"] <= _U_RATIO_FLOOR * loud["unitarity_refined"], (
        "U2 no longer fires on a loss large enough that the absorbed power "
        "dominates the LU roundoff in BOTH residuals; U2 has lost the only "
        "detection power that is demonstrated for it", loud)

    # (b) the historical missing-/h defect. `empty_s11` is its ONLY witness --
    # see the docstring. The unitarity witness is asserted BLIND here, in the
    # form that is true on every platform: the defective structure is a total
    # reflector, so the power identity holds by construction and both residuals
    # are cancellation dust.
    original = fd_mod.discrete_gamma
    try:
        fd_mod.discrete_gamma = lambda lam, k, h: original(lam, k, h) * h
        e11, _, _ = fd_mod.solve(a, freq, base, r, aps, cav, t, marg, empty=True)
        s11, s21, _ = fd_mod.solve(a, freq, base, r, aps, cav, t, marg)
        broken = fd_mod.refined_unitarity(a, freq, base, r, aps, cav, t, marg)
    finally:
        fd_mod.discrete_gamma = original
    assert abs(e11) == pytest.approx(1.0, abs=1e-6), (
        "the missing-/h defect no longer reflects a full wave off the empty "
        "guide. empty_s11 is the ONLY witness this defect has, so if this "
        "assertion stops holding the defect has become undetectable", e11)
    # why unitarity cannot see it, asserted rather than argued
    assert abs(s11) == pytest.approx(1.0, abs=1e-5) and abs(s21) < 1e-3, (
        "the missing-/h structure is no longer a total reflector, so the "
        "reasoning below about why the unitarity witness is blind to it no "
        "longer applies and the falsifier needs re-deriving", s11, s21)
    assert broken["unitarity_raw"] < 1e-13 and broken["unitarity_refined"] < 1e-13, (
        "the missing-/h defect now produces a unitarity residual above "
        "cancellation dust -- if that is real, U1/U2 may cover this class "
        "after all and the docstring's claim that empty_s11 is its only "
        "witness must be re-derived", broken)

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

    The f0 residual is +12.12 MHz at a/90 and +19.87 MHz at a/60. Converted to a
    cavity-length offset by asking the oracle what offset nulls each one -- a
    measurement, not an application of a sensitivity coefficient -- the pre-#931 diagnostic read
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
                   "as-snapped", "experimental", "regression lock"):
        assert phrase in low, phrase
    if _fixture_is_post_931(fixture):
        # #931: the claim_scope no longer derives lengths from a local rule,
        # so what is pinned is the contract's source of truth and the fact
        # that realized equals drawn (crossval-D refreshes the literal).
        for phrase in ("realized_pec_edge_masks", "realized == drawn"):
            assert phrase in low, phrase
    if _fixture_is_post_931(fixture):
        # The retired phrase names the pre-contract cavity leg, (L_c + 1)*dx
        # "between the bounding zeroed node planes". A post-#931 record must
        # not offer it as the rule -- but it may, and this one does, keep it
        # inside the labelled HISTORY passage, because the 107.5 MHz
        # drawn-vs-realized bias and the -0.68-cell thickness fit are still
        # committed numbers and stop meaning anything without it. A flat
        # "not in" ban would force the record to delete its own evidence, so
        # what is checked is POSITION: the current rule is stated first, and
        # every occurrence of a retired phrase falls after the HISTORY mark.
        flat = " ".join(low.split())
        hist = flat.find("history,")
        assert hist != -1, (
            "no HISTORY mark in the claim scope, so a retired phrase has "
            "nowhere legitimate to sit")
        assert -1 < flat.find("realized == drawn") < hist, (
            "the contract's rule must be stated before the history it replaced")
        for phrase in ("bounding zeroed node planes", "round(t/dx) + 1",
                       "round(l/dx) - 1"):
            at = flat.find(phrase)
            assert at == -1 or at > hist, (
                "the claim scope states the deleted compensation as current "
                "rather than as history", phrase, at, hist)


def test_public_carriers_quote_fixture_population_and_settling(fixture):
    """Public prose reads the record, including newly populated settling data."""
    population = [r["d_f0_mhz"] for r in fixture["gates"]["f0_envelope_population"]]
    spread = max(population) - min(population)
    for relative in ("validation/README.md", "docs/public/guide/benchmarks.mdx"):
        row = next(line for line in (_REPO_ROOT / relative).read_text().splitlines()
                   if "|" in line and "19_wr90_iris_filter_aghanim" in line)
        assert f"{spread:.2f} MHz" in row, relative
        assert "107.5 MHz" in row and ("Before #931" in row or "before #931" in row)
    public = (_REPO_ROOT / "docs/public/guide/benchmarks.mdx").read_text()
    row = next(line for line in public.splitlines()
               if "|" in line and "19_wr90_iris_filter_aghanim" in line)
    settling = fixture["gated_rfx"]["settling_db"]
    assert len(settling) == 2 and settling[0] == settling[1]
    assert f"{settling[0]:.2f} dB" in row.replace("−", "-"), row
    assert "not populated" not in row


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
    """The facts a future edit must not quietly drop.

    #931: the three pinned phrases that named the deleted compensation --
    "bounding zeroed node planes", "round(t/dx) + 1", "round(L/dx) - 1" --
    are replaced by pins on what the case does INSTEAD: it reads the realized
    edge set and states the identity. The compensation itself is refused on
    the builder's SOURCE (AST, not a whole-file grep) in
    :func:`test_no_compensation_is_applied_anywhere_in_the_case`, because the
    script keeps the retired rule in its history prose on purpose.
    """
    for phrase in (
            "realized_pec_edge_masks",
            "realized == drawn",
            "no compensation",
            "#931",
            "NOT a monotone",
            "EXTERIOR to the requested domain",
            "interpolated in dB",
    ):
        assert phrase in script_src, phrase
    # The retired rule may survive as HISTORY prose; it may not survive as
    # the builder's derivation. The AST refusal lives in
    # test_no_compensation_is_applied_anywhere_in_the_case; here the builder
    # is required to read its geometry back through the one realized-edge
    # function (its shared reader, _wr90_iris_realized, calls
    # realized_wall_planes) rather than re-derive it from cell arithmetic
    # (design note §1.7).
    assert "_wr90_iris_realized" in script_src or "realized_wall_planes" in script_src, (
        "the cv19 builder does not read its geometry back through the "
        "realized-edge reader")


def test_operating_point_is_grid_exact_on_every_row(fixture):
    for row in _rows(fixture):
        cells = row["cells_per_a"]
        dx_mm = round(A / cells * 1e3, 4)
        assert row["dx_mm"] == dx_mm, (row["cells_per_a"], row["dx_mm"])
        for lo, hi in _aperture_wall_nodes(row):
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
    coarse = fixture.get("coarse_diagnostic")
    if coarse:
        assert coarse["max_colpow"] <= 1.10, coarse["max_colpow"]
    # The physical bounds above hold at every stage; the exact pins below are
    # measurements of the pre-#931 geometry and are re-pinned with the fixture.
    _require_repinned(fixture)
    assert fixture["gated_rfx"]["max_colpow"] == pytest.approx(
        _PIN_GATED_MAX_COLPOW, abs=1e-6)
    if coarse:
        assert coarse["max_colpow"] == pytest.approx(
            _PIN_COARSE_MAX_COLPOW, abs=1e-6)


# --------------------------------------------------------------------------- #
# Hard numeric pins — filled from the committed fixture, never re-tuned.
# --------------------------------------------------------------------------- #
_PIN_F0_GATE_MHZ = 19.0                # ceil(12.1219 x 1.5) at quantum 1
_PIN_GATED_MAX_COLPOW = 1.0065         # unchanged by the redraw
_PIN_COARSE_MAX_COLPOW = 1.0315        # unchanged by the redraw
_PIN_F0_ENV_MHZ = 12.1219              # was 12.1230 on the compensated geometry
_PIN_TRACE_SHA256 = (                  # re-solved traces, cv19 pass 2
    "65934f75e51e92dc0ecc9ade1853ad82170bb3bd4b30c94492c4846fa84b482b")
