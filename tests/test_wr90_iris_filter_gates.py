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
opinion. The independence in this case comes from elsewhere, and those axes are
re-run here rather than trusted from generation time:
  * the N=1 centred limit must reduce to the ODD-mode single-iris formulation,
    which is the object PR #480 confirmed against a formulation-independent
    FDFD solver at 5.8e-4;
  * the L -> 0 collapse must turn two thin irises into one thick one;
  * lossless unitarity must hold at every evaluation, and mirror symmetry must
    hold for the reversed cascade.
A cascade error generically breaks unitarity or the collapse limit; an overlap
error breaks the single-iris reduction.

Posture, carrying every lesson from #475/#476/#480 plus this stage's own:
  * GATED: band edges and bandwidth of the -10 dB |S11| passband, rfx vs the
    oracle evaluated on the AS-REALIZED geometry; plus the structural
    reflection-zero COUNT, which is an integer and depth-independent.
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
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _REPO_ROOT / "tests/fixtures/wr90_iris_filter/fixture.json"
_ARTIFACT = _REPO_ROOT / "validation/crossval/_19_iris_filter_results/rfx.json"
_SCRIPT = _REPO_ROOT / "validation/crossval/19_wr90_iris_filter_aghanim.py"

C0 = 299792458.0
MU0 = 4e-7 * np.pi
A = 22.86e-3


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
    return dict(lo=lo, hi=hi, f0=0.5 * (lo + hi), bw=hi - lo, zeros=zeros)


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
    row = fixture["gated_rfx"]
    payload = json.dumps({"s11": row["s11"], "s21": row["s21"],
                          "oracle_s11": row["oracle_s11"]},
                         sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(payload.encode()).hexdigest() == _PIN_TRACE_SHA256, (
        "committed traces changed; if this was a deliberate regeneration, "
        "update _PIN_TRACE_SHA256 in the same commit as the new fixture")


def test_gates_are_hard_pinned_and_equal_recomputed_envelopes(fixture):
    """Hard pins AND the derived relation. Either alone is self-ratifying.

    #475's lesson: a derived-only assert passes after any coherent edit of gate
    and envelope together, so the literal values are pinned as well.
    """
    g = fixture["gates"]
    assert g["edge_gate_mhz"] == _PIN_EDGE_GATE_MHZ
    assert g["bw_gate_mhz"] == _PIN_BW_GATE_MHZ
    assert g["edge_measured_envelope_mhz"] == pytest.approx(_PIN_EDGE_ENV_MHZ, abs=1e-9)
    assert g["bw_measured_envelope_mhz"] == pytest.approx(_PIN_BW_ENV_MHZ, abs=1e-9)
    for gate, env in ((g["edge_gate_mhz"], g["edge_measured_envelope_mhz"]),
                      (g["bw_gate_mhz"], g["bw_measured_envelope_mhz"])):
        assert gate == math.ceil(max(env, 1e-9) * 1.5), (gate, env)


def test_script_live_gate_constants_match_fixture(fixture, script_src):
    """The live script constants, and the self-check that enforces them."""
    g = fixture["gates"]
    for name, want in (("GATE_EDGE_MHZ", g["edge_gate_mhz"]),
                       ("GATE_BW_MHZ", g["bw_gate_mhz"])):
        m = re.search(rf"^{name} = ([0-9.]+)", script_src, re.M)
        assert m, f"{name} not found as a live module constant"
        assert float(m.group(1)) == float(want), (name, m.group(1), want)
    assert "abs(gate - required) > 1e-9" in script_src, (
        "the write-fixture self-check that forces gate == ceil(env*1.5) is gone")


# --------------------------------------------------------------------------- #
# The comparator's INPUTS — the failure this stage actually hit.
# --------------------------------------------------------------------------- #
def test_electrical_geometry_is_rederived_from_committed_node_indices(fixture):
    """Re-derive the oracle's lengths from the rasterised metal, independently.

    The electrical length of a region is the distance between its bounding
    zeroed node planes, so a cavity drawn with L_c cells of clear space is
    (L_c + 1)*dx and an iris drawn t_c cells thick is (t_c - 1)*dx. Only that
    pairing conserves the cascade's total electrical length.
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
        assert abs(realized_mm - d_mm) <= dx * 1e3, (realized_mm, d_mm)


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

    for tag, d in (("lo", rfx["lo"] - oracle["lo"]),
                   ("hi", rfx["hi"] - oracle["hi"])):
        assert abs(d) / 1e6 <= g["edge_gate_mhz"], (tag, d / 1e6)
    assert abs(rfx["bw"] - oracle["bw"]) / 1e6 <= g["bw_gate_mhz"]
    assert len(rfx["zeros"]) == len(oracle["zeros"]), (
        "structural reflection-zero count differs from the oracle")

    # Zero FREQUENCIES, not just the count. The case declares zero frequencies
    # meaningful and zero depths not values, so the frequencies are gated at the
    # edge tolerance. This is also the only pin on the SHAPE of the committed
    # rfx trace: a mutation battery found that a 3-bin cyclic shift of the trace
    # survived the two band-edge scalars alone, because the shift happened to
    # move the existing +17.1 MHz offset to -12.9 MHz, still inside the gate.
    assert len(row_s11(fixture)) == len(freqs), "trace length != frequency grid"
    for got, want in zip(rfx["zeros"], oracle["zeros"]):
        assert abs(got - want) / 1e6 <= g["edge_gate_mhz"], (
            "reflection-zero frequency moved beyond the edge gate",
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
    assert {200.0, 400.0, 800.0} <= set(ring)
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
    assert absb["deep"]["cpml_cells"] > absb["gated"]["cpml_cells"]


def test_b_invariance_witness_is_measured_not_assumed(fixture):
    binv = {r["b_cells"]: r for r in fixture["b_invariance_witness"]}
    assert set(binv) == {4, 8}
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
    for value in (f"{g['edge_gate_mhz']:g}", f"{g['bw_gate_mhz']:g}"):
        assert value in scope, value
    # Framing rules this stage must not quietly drop. Matched case-insensitively
    # because capitalisation is emphasis, not content.
    low = scope.lower()
    for phrase in ("topology first", "not exonerated", "snapped",
                   "as-snapped", "experimental", "regression lock",
                   "bounding zeroed node planes"):
        assert phrase in low, phrase


def test_non_gated_quantities_are_declared_non_gated(fixture, script_src):
    posture = fixture["gates"]["posture"]
    for phrase in ("worst-case RL", "ripple levels", "zero depths", "phase"):
        assert phrase in posture, phrase
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
    assert fixture["gated_rfx"]["max_colpow"] <= 1.02
    coarse = fixture.get("coarse_diagnostic")
    if coarse:
        assert coarse["max_colpow"] <= 1.10, coarse["max_colpow"]


# --------------------------------------------------------------------------- #
# Hard numeric pins — filled from the committed fixture, never re-tuned.
# --------------------------------------------------------------------------- #
_PIN_EDGE_GATE_MHZ = 26.0
_PIN_BW_GATE_MHZ = 15.0
_PIN_EDGE_ENV_MHZ = 17.08
_PIN_BW_ENV_MHZ = 9.99
_PIN_TRACE_SHA256 = (
    "79dfcee9a7454d5229d1873c1008e0411796755f420155f2270124b293d8b964")
