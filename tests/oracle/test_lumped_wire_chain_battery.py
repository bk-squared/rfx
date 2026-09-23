"""Replay of the lumped / wire port chain battery. Reads the fixture, re-derives
every number that decides a verdict from the stored complex S11, and holds it
to the bar written in this file.

No FDTD runs here. The measurement is
``scripts/diagnostics/lumped_wire_chain_battery_measure.py``, pre-declared in
``docs/design_notes/lumped_wire_chain_battery_predeclaration.md``; the bar is
the v2.0 one in ``docs/design_notes/chain_closure_contract.md`` — magnitude
within 2 dB, frequencies within 1 %, AD against FD within 0.05, forward
identity at rtol 1e-5 / atol 1e-7 — plus the numbers this family's
pre-declaration adds in place of reciprocity and power closure: passivity
``max_f |S11| <= 1.02`` on a settled record, the matched control held to
``-20 dB``, and the record-length substitute for a settling witness, 0.2 dB.

Nothing that decides a verdict is read from the artifact under test. The bars,
the set of records the battery declares, the commit they were measured at, the
gradient legs that have no derivative to compare and the cell size the ladder
recommends are constants in this file, and the fixture is asserted to agree
with them. Every derived quantity is recomputed HERE from the stored complex
S11 by an implementation written independently of the assembler's — phase
crossings from the zeros of Im S11, closed-form derivatives from this file's
own closed forms — and the stored summaries are checked against the
recomputation, so a hand-edited fixture or a drifting assembler reds instead of
replaying.

The closed forms the battery is refereed against are unit-tested here against a
hand-computed value first, so a comparison against them is a comparison against
arithmetic somebody checked rather than against the driver's own code.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

FIXTURE = (Path(__file__).resolve().parents[1] / "fixtures"
           / "lumped_wire_chain_battery" / "fixture.json")

C0 = 299792458.0
ETA0 = 376.730313668
DB_FLOOR = 1e-300
CLAIMS_RUNG_UM = 250
COARSEST_RUNG_UM = 1000
RUNGS_UM = (1000, 500, 250)
KINDS = ("lumped", "wire")
DUTS = ("short", "open", "res_half", "res_double", "matched")
REFLECTING_DUTS = ("short", "open", "res_half", "res_double")

# The bar: the contract's v2.0 numbers and the two this family's
# pre-declaration adds. Written here so that the artifact under test cannot
# move it — the fixture's `bar` block and every per-record copy of a threshold
# are asserted equal to these numbers, and every verdict below reads THESE.
BAR = {
    "magnitude_db": 2.0,
    "frequency_frac": 0.01,
    "passivity_max": 1.02,
    "matched_floor_db": -20.0,
    "ad_fd_rel": 0.05,
    "identity_rtol": 1e-5,
    "identity_atol": 1e-7,
    "record_doubling_db": 0.2,
}
# The float64 finite difference's resolving-power floor, in ULPs of the loss.
MIN_FD_ULP_SPAN = 1.0e4

# The channel as the pre-declaration draws it. The referee is rebuilt from
# these, and the fixture's own declared block is asserted to agree.
LINE_M = 30e-3
N_H_CELLS = {"lumped": 1, "wire": 4}
EPS_R_FILL = 2.2
R_OVER_ZC = {"res_half": 0.5, "res_double": 2.0, "matched": 1.0}
AD_BAND_HZ = (4e9, 6e9)
AD_MID_HZ = 5e9

# Every record the battery declares. A missing record fails; it never skips.
SOLVE_KEYS = frozenset(f"{k}_{d}_{um}um" for k in KINDS for d in DUTS
                       for um in RUNGS_UM)
LADDER_KEYS = frozenset(f"{k}_{d}" for k in KINDS for d in DUTS)
ADFD_OBJECTIVES = {
    "adfd-r": ("band_mean_s11_sq", "re_s11_at_mid_band"),
    "adfd-eps": ("band_mean_s11_sq", "re_s11_at_mid_band"),
    "adfd-eps-res": ("band_mean_s11_sq",),
}
ADFD_RUNGS_UM = {"adfd-r": (COARSEST_RUNG_UM,), "adfd-eps": RUNGS_UM,
                 "adfd-eps-res": RUNGS_UM}
ADFD_KEYS = frozenset(f"{k}_{leg}_{um}um" for k in KINDS
                      for leg, rungs in ADFD_RUNGS_UM.items() for um in rungs)
IDENTITY_KINDS = frozenset(KINDS)
NOT_IN_CHAIN_KINDS = frozenset(KINDS)

# The commit every measured record names. Re-measuring the battery moves this
# on purpose.
RECORDS_COMMIT = "c3936b3b859b6d8dda9dae0b0fa1ff43111cf1f7"
# The one block that names another: `port_kind_ab` lifts the known-load record
# committed on main, which its producer wrote at this commit. It is carried
# because a battery job re-ran that producer at RECORDS_COMMIT and reproduced
# it; the reproduction ships beside it and is compared below.
KNOWN_LOAD_RECORD_COMMIT = "018225e6ad07857eb4540d1df071ece38e8949eb"

# The legs with no derivative to compare: band-mean |S11|^2 on the filled
# SHORT, per port kind and cell size. A lossless termination referenced to the
# line's own Zc gives |S11| = 1 at every frequency, so that objective is the
# constant 1. The test re-derives this from each leg's stored S11; no other leg
# may carry the label, and every other leg is compared.
DEGENERATE_CASES = frozenset((f"{k}_adfd-eps_{um}um", "band_mean_s11_sq")
                             for k in KINDS for um in RUNGS_UM)
# |S11|^2 within this of 1 at every bin reads as lossless: the contract's
# column-power tolerance for the settling substitute on a lossless load.
LOSSLESS_POWER_TOL = 1e-3

# The ladder's decision, re-derived below from the stored S11 and pinned here by
# name: the coarsest cell size from which every finer one stays inside the bar
# against the finest.
DECIDING_FLAGS = {
    "reflecting": ("passivity_within_bar", "magnitude_within_2dB_vs_finest",
                   "crossing_within_1pct_vs_finest"),
    "matched": ("passivity_within_bar", "matched_floor_within_bar"),
}
RECOMMENDED_RUNG_UM = {"short": 500, "open": 500, "res_half": 500,
                       "res_double": 500, "matched": 1000}

# How far this file's re-derivations may sit from the assembler's stored
# summaries. Neither is a bar; both are the size of a difference in method.
# Crossings: the assembler interpolates along the unwrapped angle and this file
# along Im S11, which place one crossing up to 3.0e-5 apart (relative) on the
# records this ships with. Doubling: the driver took log10 of float32
# magnitudes; recomputed in float64 the shift moves by up to 7.8e-6 dB.
CROSSING_METHOD_TOL = 2e-4
DOUBLING_DB_TOL = 1e-4

# The window a first-order sequence's successive ratio falls in. A quantity
# converging at first order in the cell size halves each time the mesh halves,
# so the ratio sits near 0.5; the window is wide enough for the second-order
# term that is still present at these cell sizes and narrow enough to exclude a
# sequence that is not converging at all (a ratio near 1).
FIRST_ORDER_RATIO_WINDOW = (0.35, 0.75)


# ---------------------------------------------------------------------------
# independent arithmetic — deliberately not imported from the driver
# ---------------------------------------------------------------------------

def _complex(block) -> np.ndarray:
    return (np.asarray(block["real"], dtype=float)
            + 1j * np.asarray(block["imag"], dtype=float))


def _complex_s11(row) -> np.ndarray:
    """The known-load record's ``s11_real`` / ``s11_imag`` pair."""
    return (np.asarray(row["s11_real"], dtype=float)
            + 1j * np.asarray(row["s11_imag"], dtype=float))


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), DB_FLOOR))


def zin_line(zc, beta, length, z_load):
    """``Zc (ZL + j Zc tan(beta L)) / (Zc + j ZL tan(beta L))``; ``z_load=None``
    is an open. Written from the transmission-line equation rather than
    imported, so a change to the driver's version shows up here as a
    disagreement instead of moving both sides together."""
    t = np.tan(np.asarray(beta, dtype=float) * length)
    if z_load is None:
        return zc / (1j * t)
    return zc * (z_load + 1j * zc * t) / (zc + 1j * z_load * t)


def s11_of(zin, zref):
    return (zin - zref) / (zin + zref)


def gamma_of(z_load, zc):
    if z_load is None:
        return 1.0 + 0.0j
    return (z_load - zc) / (z_load + zc)


def beta_of(freqs, eps_r=1.0):
    return 2.0 * math.pi * np.asarray(freqs, dtype=float) * math.sqrt(eps_r) / C0


def ds11_dr(zc, beta, length, r_ohm, zref):
    """``dS11/dR`` for a resistive load: ``dS11/dZin * dZin/dR`` with
    ``dZin/dR = Zc^2 (1 + t^2) / (Zc + j R t)^2``, ``t = tan(beta L)``."""
    t = np.tan(np.asarray(beta, dtype=float) * length)
    zin = zc * (r_ohm + 1j * zc * t) / (zc + 1j * r_ohm * t)
    dzin = zc ** 2 * (1.0 + t ** 2) / (zc + 1j * r_ohm * t) ** 2
    return 2.0 * zref * dzin / (zin + zref) ** 2


def s11_and_ds11_deps(zc_air, eps_r, freqs, length, r_ohm, zref):
    """S11 of a line filled with ``eps_r`` and terminated in ``r_ohm``
    (``0.0`` is a short), and ``dS11/d(eps_r)`` with the load and ``Zref``
    held. Chain rule on ``Zc = zc_air / sqrt(eps_r)`` and
    ``beta = omega sqrt(eps_r) / c``:

        dZc/deps = -Zc / (2 eps),   dt/deps = L (1 + t^2) beta / (2 eps)
        Zin = Zc N / D,  N = R + j Zc t,  D = Zc + j R t
    """
    b = beta_of(freqs, eps_r)
    zc = zc_air / math.sqrt(eps_r)
    t = np.tan(b * length)
    dzc = -zc / (2.0 * eps_r)
    dt = length * (1.0 + t ** 2) * b / (2.0 * eps_r)
    num = r_ohm + 1j * zc * t
    den = zc + 1j * r_ohm * t
    dnum = 1j * (dzc * t + zc * dt)
    dden = dzc + 1j * r_ohm * dt
    zin = zc * num / den
    dzin = dzc * num / den + zc * (dnum * den - num * dden) / den ** 2
    return s11_of(zin, zref), 2.0 * zref * dzin / (zin + zref) ** 2


def realized_length_m(dut, rung_um):
    """The line the lattice builds: a short on an E node and a resistor one
    node inboard of the wall sit a whole number of cells from the port, while
    the open's current zero falls on the H half-node, half a cell short of the
    drawn 30 mm (the driver's third deviation)."""
    dx = rung_um * 1e-6
    return LINE_M - 0.5 * dx if dut == "open" else LINE_M


def z_load_of(dut, zc):
    """The declared termination: ``0.0`` a short, ``None`` an open, else R."""
    if dut == "short":
        return 0.0
    if dut == "open":
        return None
    return R_OVER_ZC[dut] * zc


def real_axis_crossings(freqs, s11):
    """Where S11 crosses the real axis: a sign change of Im S11 between two
    bins, placed by linear interpolation of Im S11 and tagged with the sign of
    Re S11 there (+1: the angle is a multiple of 2 pi; -1: an odd multiple of
    pi). Independent of the assembler's walk along the unwrapped angle."""
    f = np.asarray(freqs, dtype=float)
    re, im = np.real(s11), np.imag(s11)
    out = []
    for k in range(len(f) - 1):
        a, b = im[k], im[k + 1]
        if a == 0.0:
            t = 0.0
        elif a * b < 0.0:
            t = a / (a - b)
        else:
            continue
        fk = f[k] + t * (f[k + 1] - f[k])
        rk = re[k] + t * (re[k + 1] - re[k])
        out.append((float(fk), 1 if rk > 0.0 else -1))
    return out


def analytic_crossings(length_m, gamma_sign, f_lo, f_hi):
    """``S11 = Gamma_L exp(-2 j beta L)`` is real where ``2 beta L = m pi``,
    i.e. at ``f_m = m c / (4 L)``, with the sign of ``Gamma_L (-1)^m``."""
    out = []
    m = 1
    while m * C0 / (4.0 * length_m) <= f_hi:
        f = m * C0 / (4.0 * length_m)
        if f >= f_lo:
            out.append((m, f, gamma_sign * (-1) ** m))
        m += 1
    return out


def matched_crossings(freqs, s11, dut, rung_um):
    """Each closed-form crossing on the realized length, paired with the
    nearest measured crossing of the same sign: ``{m: (f_analytic,
    f_measured)}``. Two closed-form crossings sharing one measured partner mean
    the measured phase has slid by a whole crossing period, which nearest-
    neighbour matching would otherwise hide, so that is an assertion here."""
    zc = 1.0
    sign = 1 if gamma_of(z_load_of(dut, zc), zc).real > 0.0 else -1
    meas = real_axis_crossings(freqs, s11)
    an = analytic_crossings(realized_length_m(dut, rung_um), sign,
                            float(freqs[0]), float(freqs[-1]))
    rows = {}
    for m, f_an, s in an:
        same = [f for f, sg in meas if sg == s]
        assert same, f"{dut} {rung_um} um: no measured crossing of sign {s}"
        rows[m] = (f_an, min(same, key=lambda f: abs(f - f_an)))
    partners = [fm for _fa, fm in rows.values()]
    assert len(set(partners)) == len(partners), (
        f"{dut} {rung_um} um: two closed-form crossings matched the same measured "
        f"crossing ({partners}) — the measured phase is shifted by about a whole "
        "crossing period")
    return rows


# ---------------------------------------------------------------------------
# the closed forms, checked against hand arithmetic before anything uses them
# ---------------------------------------------------------------------------

def test_the_closed_form_reduces_to_gamma_times_a_phase_when_zref_is_zc():
    """With the port referenced to the line's own Zc the terminated-line formula
    must collapse to ``Gamma_L exp(-2 j beta L)``. Two ways of writing the same
    physics; if they disagree the referee is wrong before any measurement."""
    zc, length = 100.0, 0.02
    freqs = np.linspace(1e9, 10e9, 37)
    beta = beta_of(freqs)
    for z_load in (0.0, None, 50.0, 200.0, 100.0):
        direct = s11_of(zin_line(zc, beta, length, z_load), zc)
        collapsed = gamma_of(z_load, zc) * np.exp(-2j * beta * length)
        np.testing.assert_allclose(direct, collapsed, rtol=1e-10, atol=1e-12)


def test_the_closed_form_matches_a_hand_computed_value():
    """zc = zref = 100 ohm, R = 50 ohm, beta L = 0.25 rad.

    Gamma_L = (50-100)/(50+100) = -1/3, and exp(-0.5j) = 0.8775825619 -
    0.4794255386j, so S11 = -0.2925275206 + 0.1598085129j and |S11| = 1/3.
    dS11/dR = exp(-2 j beta L) * 2 Zc / (R + Zc)^2
            = (0.8775825619 - 0.4794255386j) * 200/22500
            = 0.0078007339 - 0.0042615603j.
    """
    zc = zref = 100.0
    r = 50.0
    beta_l = 0.25
    beta = beta_l / 0.02
    s = s11_of(zin_line(zc, beta, 0.02, r), zref)
    assert s.real == pytest.approx(-0.2925275206, abs=1e-9)
    assert s.imag == pytest.approx(0.1598085129, abs=1e-9)
    assert abs(s) == pytest.approx(1.0 / 3.0, rel=1e-12)

    ds = ds11_dr(zc, beta, 0.02, r, zref)
    assert ds.real == pytest.approx(0.0078007339, abs=1e-9)
    assert ds.imag == pytest.approx(-0.0042615603, abs=1e-9)

    # And against a central difference of the closed form itself, which is an
    # independent route to the same derivative.
    h = 1e-5
    fd = (s11_of(zin_line(zc, beta, 0.02, r + h), zref)
          - s11_of(zin_line(zc, beta, 0.02, r - h), zref)) / (2 * h)
    assert abs(ds - fd) / abs(ds) < 1e-8


def test_the_permittivity_derivative_matches_a_difference_of_the_closed_form():
    """``s11_and_ds11_deps`` is a chain rule written out by hand; this checks it
    against a central difference of the closed form it differentiates, for a
    short and for a resistor, on the filled line and band the battery uses."""
    freqs = np.linspace(1e9, 10e9, 91)
    zc_air = ETA0
    for r_over_zc in (0.0, 2.0):
        zc = zc_air / math.sqrt(EPS_R_FILL)
        r = r_over_zc * zc
        _s, ds = s11_and_ds11_deps(zc_air, EPS_R_FILL, freqs, LINE_M, r, zc)
        h = 1e-6 * EPS_R_FILL

        def at(e):
            return s11_of(zin_line(zc_air / math.sqrt(e), beta_of(freqs, e),
                                   LINE_M, r), zc)
        fd = (at(EPS_R_FILL + h) - at(EPS_R_FILL - h)) / (2.0 * h)
        np.testing.assert_allclose(ds, fd, rtol=1e-6, atol=1e-9)


def test_an_open_and_a_short_are_all_phase():
    """A lossless reactive termination cannot change the magnitude: the referee
    has to return exactly one for both, or the phase comparisons downstream are
    reading a magnitude error as a phase error."""
    beta = beta_of(np.linspace(1e9, 10e9, 91))
    for z_load in (0.0, None):
        s = s11_of(zin_line(376.73, beta, 0.03, z_load), 376.73)
        np.testing.assert_allclose(np.abs(s), 1.0, rtol=1e-10, atol=1e-12)


def test_the_crossing_finder_reads_the_closed_form_back():
    """The crossing finder, run on the closed form itself, has to return the
    closed-form crossings to far better than the 1 % bar; otherwise a phase
    verdict would be a verdict on the finder."""
    freqs = np.linspace(1e9, 10e9, 91)
    for dut in REFLECTING_DUTS:
        zc = ETA0
        length = realized_length_m(dut, CLAIMS_RUNG_UM)
        s = s11_of(zin_line(zc, beta_of(freqs), length, z_load_of(dut, zc)), zc)
        rows = matched_crossings(freqs, s, dut, CLAIMS_RUNG_UM)
        assert rows, dut
        for f_an, f_meas in rows.values():
            assert abs(f_meas - f_an) / f_an < 1e-3, dut


# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture():
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing; the battery's replay has nothing to replay")
    return json.loads(FIXTURE.read_text())


def _solve(fixture, key):
    assert key in fixture["solves"], f"the battery has no {key} record"
    return fixture["solves"][key]


ALL_SOLVES = sorted(SOLVE_KEYS)
CLAIMS_SOLVES = [f"{k}_{d}_{CLAIMS_RUNG_UM}um" for k in KINDS for d in DUTS]
REFLECTING = [f"{k}_{d}_{CLAIMS_RUNG_UM}um" for k in KINDS for d in REFLECTING_DUTS]
ALL_SOLVES = sorted(SOLVE_KEYS)
ALL_REFLECTING = [f"{k}_{d}_{um}um" for k in KINDS for d in REFLECTING_DUTS
                  for um in RUNGS_UM]


def test_the_bars_are_the_contracts_and_no_record_carries_another(fixture):
    """The thresholds live in this file. The artifact carries them too, beside
    each measurement, for a reader; this asserts that every copy it carries is
    this file's number, so an artifact that loosened its own bar reds here
    rather than passing its own test."""
    assert fixture["bar"] == BAR
    for key, e in fixture["solves"].items():
        assert e["passivity"]["bar"] == BAR["passivity_max"], key
        assert e["record_doubling"]["bar_db"] == BAR["record_doubling_db"], key
        if e["dut"] == "matched":
            assert e["matched_floor"]["bar_db"] == BAR["matched_floor_db"], key
        else:
            assert e["magnitude_vs_analytic"]["bar_db"] == BAR["magnitude_db"], key
            assert e["phase"]["bar_frac"] == BAR["frequency_frac"], key
    for name, block in fixture["adfd"].items():
        assert block["bar"] == BAR["ad_fd_rel"], name
        assert block["min_fd_ulp_span"] == MIN_FD_ULP_SPAN, name
    for kind, ident in fixture["identity"].items():
        assert ident["rtol"] == BAR["identity_rtol"], kind
        assert ident["atol"] == BAR["identity_atol"], kind


def test_the_fixture_carries_every_declared_record(fixture):
    """The battery declares thirty solves, ten ladders, fourteen gradient
    blocks, two identity arms and two not-in-chain observations. A record that
    is absent is a failure here, never a skip further down: an artifact that
    silently lost the lumped leg would otherwise replay green on the rest."""
    assert set(fixture["solves"]) == SOLVE_KEYS
    assert set(fixture["ladder"]) == LADDER_KEYS
    assert set(fixture["adfd"]) == ADFD_KEYS
    for name, block in fixture["adfd"].items():
        assert name == f"{block['kind']}_{block['leg']}_{block['rung_um']}um", name
        assert tuple(c["objective"] for c in block["cases"]) == \
            ADFD_OBJECTIVES[block["leg"]], name
    assert set(fixture["identity"]) == IDENTITY_KINDS
    assert set(fixture["not_in_chain_observations"]) == NOT_IN_CHAIN_KINDS
    assert fixture["pilot"] is not None and fixture["pilot"]["cases"]
    ab = fixture["port_kind_ab"]
    assert ab["present"] is True, "the known-load A/B record was not assembled"
    assert "reproduction" in ab, "the known-load record ships without its reproduction"


def _measured_provenance(fixture):
    for key, e in fixture["solves"].items():
        yield f"solves.{key}", e["provenance"]
    for name, block in fixture["adfd"].items():
        yield f"adfd.{name}", block["provenance"]
    for kind, block in fixture["identity"].items():
        yield f"identity.{kind}", block["provenance"]
    for kind, block in fixture["not_in_chain_observations"].items():
        yield f"not_in_chain_observations.{kind}", block["provenance"]
    yield "pilot", fixture["pilot"]["provenance"]


def test_every_measured_record_names_the_battery_commit(fixture):
    """One commit for the whole measurement, named here. A record stamped with
    another commit was measured on another tree, whatever its numbers say."""
    wrong = {tag: p["commit"] for tag, p in _measured_provenance(fixture)
             if p["commit"] != RECORDS_COMMIT}
    assert not wrong, (
        f"records not measured at {RECORDS_COMMIT}: {wrong}")
    assert len(fixture["assembler_commit"]) == 40
    for tag, p in _measured_provenance(fixture):
        # The fixture does not ship machine paths; it ships the question they
        # answered — did `import rfx` resolve to the run's own tree?
        assert p["rfx_import_tail"] == "rfx/__init__.py", tag
        assert p["rfx_resolved_inside_the_run_tree"] is True, tag
        assert p["compute_run_id"], f"{tag} does not name the run that produced it"
        assert p["jax_version"] and p["numpy_version"], tag


def test_the_known_load_block_is_the_one_exception_and_carries_its_reproduction(fixture):
    """``port_kind_ab`` is the only block that names a commit other than the
    battery's: it lifts the known-load record committed on main. It is carried
    because a battery job re-ran that record's producer at the battery's commit,
    and the reproduction ships beside it; the two have to be the same complex
    numbers, and the magnitudes the block quotes have to be theirs."""
    ab = fixture["port_kind_ab"]
    assert ab["commit"] == KNOWN_LOAD_RECORD_COMMIT
    rep = ab["reproduction"]
    assert rep["commit"] == RECORDS_COMMIT
    assert rep["rfx_resolved_inside_this_tree"] is True
    assert rep["compute_run_id"], "the reproduction names no run"
    assert rep["freqs_hz"] == ab["freqs_hz"]
    assert set(rep["loads"]) == set(ab["loads"])
    for name, row in ab["loads"].items():
        for kind in KINDS:
            record = _complex_s11(row[kind])
            again = _complex_s11(rep["loads"][name][kind])
            np.testing.assert_array_equal(record, again, err_msg=f"{name}/{kind}")
            # The producer's own |S11| is float32 arithmetic on these numbers.
            np.testing.assert_allclose(row[kind]["abs_s11"], np.abs(record),
                                       rtol=0, atol=1e-6, err_msg=f"{name}/{kind}")


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_the_realized_channel_is_the_declared_channel(fixture, key):
    """The measurement asserted this before solving; the fixture has to carry
    the evidence, because a number whose channel was never checked is not one of
    this battery's numbers. The channel itself — gap, width, Zc, load, realized
    length — is held to this file's constants, since the referee is built from
    them."""
    entry = _solve(fixture, key)
    kind, dut, um = entry["kind"], entry["dut"], entry["rung_um"]
    assert key == f"{kind}_{dut}_{um}um"
    d, g, ps = entry["declared"], entry["realized_grid"], entry["port_spec"]
    zc = ETA0 * N_H_CELLS[kind]
    assert d["n_h_cells"] == N_H_CELLS[kind], key
    assert d["n_w_cells"] == 1, key
    assert d["zc_ohm"] == pytest.approx(zc, rel=1e-12), key
    assert d["zref_ohm"] == pytest.approx(zc, rel=1e-12), key
    assert d["dx_m"] == pytest.approx(um * 1e-6, rel=1e-12), key
    assert d["length_m_declared"] == LINE_M, key
    assert d["length_m_realized"] == pytest.approx(realized_length_m(dut, um),
                                                   rel=1e-12), key
    want_load = z_load_of(dut, zc)
    if dut == "open":
        assert d["z_load_ohm"] is None, key
    else:
        assert d["z_load_ohm"] == pytest.approx(want_load, rel=1e-12, abs=0.0), key
    # These fields exist because the driver resolves the port and the load from
    # the positions the SIMULATION carries. An older driver read them back out
    # of its own layout and compared layout to layout, which accepted a port
    # built on node 0 and a load moved a whole cell; a record without them was
    # written by that driver and is not evidence.
    assert "n_ports" in g and "n_rlc_elements" in g, (
        f"{key}: the realized block predates the independent resolution and "
        "cannot say where the port and the load actually landed")
    assert g["grid_shape_nodes"] == [d["n_nodes_x"], 2, d["n_h_cells"] + 1], key
    assert g["n_ports"] == 1, key
    assert g["port_index"] == [d["i_port"], 0, 0], key
    assert g["port_impedance_ohm"] == pytest.approx(zc, rel=1e-9), key
    assert g["port_extent_cells"] == (None if kind == "lumped" else d["n_h_cells"]), key
    assert g["port_excite"] is True, key
    want_pec = {"z_lo", "z_hi"} | ({"x_hi"} if dut == "short" else set())
    want_pmc = ({"x_lo", "y_lo", "y_hi"} | (set() if dut == "short" else {"x_hi"}))
    assert set(g["boundary_faces"]["pec"]) == want_pec, key
    assert set(g["boundary_faces"]["pmc"]) == want_pmc, key
    assert ps["impedance_ohm"] == pytest.approx(zc, rel=1e-9), key
    if kind == "wire":
        assert ps["n_live"] == d["n_h_cells"], key
        assert ps["excite"] is True, key
    want_n = d["n_h_cells"] if d["i_rlc"] is not None else 0
    assert g["n_rlc_elements"] == want_n, key
    if want_n:
        assert g["rlc_indices"] == [[d["i_rlc"], 0, k]
                                    for k in range(d["n_h_cells"])], key
        assert sum(g["rlc_values_ohm"]) == pytest.approx(want_load, rel=1e-9), key
    # The realized length follows from the node layout, not from a stored float.
    assert d["length_m_realized"] == pytest.approx(
        d["length_cells_realized"] * d["dx_m"], rel=1e-12), key


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_every_stored_summary_follows_from_the_stored_s11(fixture, key):
    entry = _solve(fixture, key)
    s11 = _complex(entry["s11"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    assert s11.shape == (len(freqs),)
    assert entry["freqs_hz"] == fixture["freqs_hz"]

    np.testing.assert_allclose(np.abs(s11), entry["abs_s11"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(_db(s11), entry["s11_db"], rtol=0, atol=1e-9)
    assert entry["max_abs_s11"] == pytest.approx(np.abs(s11).max(), rel=1e-12)
    assert entry["min_abs_s11"] == pytest.approx(np.abs(s11).min(), rel=1e-12)
    assert entry["argmax_abs_s11_bin"] == int(np.argmax(np.abs(s11)))
    assert entry["passivity"]["max_abs_s11"] == pytest.approx(
        np.abs(s11).max(), rel=1e-12)
    assert entry["passivity"]["within_bar"] == bool(
        np.abs(s11).max() <= BAR["passivity_max"])

    # The referee, rebuilt here from this file's channel and the realized length.
    kind, dut, um = entry["kind"], entry["dut"], entry["rung_um"]
    zc = ETA0 * N_H_CELLS[kind]
    z_load = z_load_of(dut, zc)
    an = s11_of(zin_line(zc, beta_of(freqs), realized_length_m(dut, um), z_load), zc)
    np.testing.assert_allclose(_complex(entry["referee"]["analytic_realized_length"]),
                               an, rtol=1e-9, atol=1e-12)
    mv = entry["magnitude_vs_analytic"]
    assert mv["abs_gamma_load"] == pytest.approx(abs(gamma_of(z_load, zc)), rel=1e-12)
    if dut == "matched":
        # A dB difference against an identically-zero reference is a difference
        # against the log floor. The artifact must not carry one.
        assert mv["applies"] is False
        assert "max_abs_db_diff" not in mv
        assert "within_bar" not in mv
        assert entry["matched_floor"]["max_db"] == pytest.approx(
            float(_db(s11).max()), abs=1e-9)
        assert entry["matched_floor"]["within_bar"] == bool(
            _db(s11).max() <= BAR["matched_floor_db"])
        return
    worst_db = float(np.abs(_db(s11) - _db(an)).max())
    assert mv["applies"] is True
    assert mv["max_abs_db_diff"] == pytest.approx(worst_db, rel=1e-9)
    assert mv["within_bar"] == bool(worst_db <= BAR["magnitude_db"])

    # The stored phase summary against this file's own crossings, at EVERY rung.
    rows = matched_crossings(freqs, s11, dut, um)
    fracs = {m: abs(fm - fa) / fa for m, (fa, fm) in rows.items()}
    ph = entry["phase"]
    assert ph["n_crossings"] == len(rows), key
    assert [r["analytic_hz"] for r in ph["matched_realized"]] == pytest.approx(
        [fa for fa, _fm in rows.values()], rel=1e-12), key
    assert [r["measured_hz"] for r in ph["matched_realized"]] == pytest.approx(
        [fm for _fa, fm in rows.values()], rel=CROSSING_METHOD_TOL), key
    assert ph["max_frac"] == pytest.approx(max(fracs.values()),
                                           abs=CROSSING_METHOD_TOL), key
    assert ph["within_bar"] == bool(max(fracs.values()) <= BAR["frequency_frac"]), key


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_the_record_length_witness_recomputes_from_both_curves(fixture, key):
    """``forward(port_s11_freqs=...)`` emits no settling witness, so the
    contract's admissible substitute is record-length invariance: the same
    solve with the record doubled, and the |S11| shift between the two held to
    one tenth of the magnitude gate. Both curves ship, and the shift is
    recomputed here from them."""
    entry = _solve(fixture, key)
    rd = entry["record_doubling"]
    assert rd is not None, f"{key} carries no record-doubling witness"
    s = _complex(entry["s11"])
    sd = _complex(rd["s11"])
    assert sd.shape == s.shape, key
    assert rd["n_steps"] > entry["n_steps"], key
    assert not np.array_equal(sd, s), (
        f"{key}: the doubled record is the original to the last bit, which a "
        "second, longer solve does not produce")
    shift = float(np.abs(_db(sd) - _db(s)).max())
    assert rd["max_abs_db_shift"] == pytest.approx(shift, abs=DOUBLING_DB_TOL), key
    assert rd["max_abs_diff"] == pytest.approx(float(np.abs(sd - s).max()),
                                               rel=1e-6, abs=1e-12), key
    assert rd["within_bar"] == bool(shift <= BAR["record_doubling_db"]), key
    assert shift <= BAR["record_doubling_db"], (
        f"{key}: doubling the record moved |S11| by {shift:.5f} dB, above the "
        f"{BAR['record_doubling_db']} dB substitute for a settling witness — the "
        "record was truncated, so this S11 is not interpretable")


@pytest.mark.parametrize("dut", DUTS)
@pytest.mark.parametrize("um", RUNGS_UM)
def test_the_lumped_and_the_wire_port_return_the_same_s11(fixture, dut, um):
    """On this line the two port kinds read the same complex S11: identical, to
    the last bit, at every load and cell size on the records this ships with,
    for the solve and for its doubled record. A difference means one of the
    two lanes changed on its own."""
    a = _solve(fixture, f"lumped_{dut}_{um}um")
    b = _solve(fixture, f"wire_{dut}_{um}um")
    np.testing.assert_array_equal(_complex(a["s11"]), _complex(b["s11"]))
    np.testing.assert_array_equal(_complex(a["record_doubling"]["s11"]),
                                  _complex(b["record_doubling"]["s11"]))


@pytest.mark.parametrize("key", CLAIMS_SOLVES)
def test_the_one_port_is_passive_at_the_claims_rung(fixture, key):
    """What replaces reciprocity and power closure for a one-port: a passive
    load cannot reflect more than it received."""
    entry = _solve(fixture, key)
    measured = float(np.abs(_complex(entry["s11"])).max())
    assert measured <= BAR["passivity_max"], (
        f"{key}: max|S11| = {measured:.6f} exceeds "
        f"{BAR['passivity_max']} on a passive termination")


@pytest.mark.parametrize("key", REFLECTING)
def test_the_magnitude_at_the_claims_rung_matches_the_closed_form(fixture, key):
    """|S11| is |Gamma_L| at every bin on a lossless line referenced to its own
    Zc, so this is the v2 bar's magnitude test with the reference known
    exactly."""
    entry = _solve(fixture, key)
    s11 = _complex(entry["s11"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    zc = ETA0 * N_H_CELLS[entry["kind"]]
    z_load = z_load_of(entry["dut"], zc)
    an = s11_of(zin_line(zc, beta_of(freqs),
                         realized_length_m(entry["dut"], entry["rung_um"]), z_load), zc)
    worst = float(np.abs(_db(s11) - _db(an)).max())
    assert worst <= BAR["magnitude_db"], (
        f"{key}: |S11| is {worst:.4f} dB from the closed form's "
        f"{abs(gamma_of(z_load, zc)):.6f} at worst — the bar is "
        f"{BAR['magnitude_db']} dB")


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_the_one_port_is_passive_at_every_rung(fixture, key):
    """Passivity cannot false-alarm on a coarse mesh: a passive load reflects at
    most what it receives at every cell size, on the solve and on its doubled
    record. The ladder judges the coarse rungs only through the recommendation,
    so this is what holds them to the physics directly (review of PR 1215,
    round 2, mutation n02: a coarse short at |S11| = 1.5 passed without it)."""
    entry = _solve(fixture, key)
    for label, block in (("solve", entry["s11"]),
                         ("doubled record", entry["record_doubling"]["s11"])):
        measured = float(np.abs(_complex(block)).max())
        assert measured <= BAR["passivity_max"], (
            f"{key} ({label}): max|S11| = {measured:.6f} exceeds "
            f"{BAR['passivity_max']} on a passive termination")


@pytest.mark.parametrize("key", ALL_REFLECTING)
def test_the_magnitude_stays_inside_the_bar_at_every_rung(fixture, key):
    """The v2 bar against the exact referee at every cell size, not only at the
    claims rung: |S11| within 2 dB of the closed form on the realized length
    (round-2 mutation n02b: a coarse resistor 6 dB off passed without it). The
    matched line is a deep null and is held to its floor elsewhere."""
    entry = _solve(fixture, key)
    s11 = _complex(entry["s11"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    zc = ETA0 * N_H_CELLS[entry["kind"]]
    z_load = z_load_of(entry["dut"], zc)
    an = s11_of(zin_line(zc, beta_of(freqs),
                         realized_length_m(entry["dut"], entry["rung_um"]), z_load), zc)
    worst = float(np.abs(_db(s11) - _db(an)).max())
    assert worst <= BAR["magnitude_db"], (
        f"{key}: |S11| is {worst:.4f} dB from the closed form at worst — the "
        f"bar is {BAR['magnitude_db']} dB")


@pytest.mark.parametrize("key", REFLECTING)
def test_the_phase_crossings_at_the_claims_rung_land_within_one_percent(fixture, key):
    """The pre-declaration's phase test: the frequencies at which S11 crosses
    the real axis, against the closed form on the realized length, to 1 %."""
    entry = _solve(fixture, key)
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    rows = matched_crossings(freqs, _complex(entry["s11"]), entry["dut"],
                             entry["rung_um"])
    assert rows, f"{key}: no closed-form crossing fell inside the band"
    detail = [(round(fa / 1e9, 5), round(fm / 1e9, 5), round(abs(fm - fa) / fa, 5))
              for fa, fm in rows.values()]
    worst = max(abs(fm - fa) / fa for fa, fm in rows.values())
    assert worst <= BAR["frequency_frac"], (
        f"{key}: the worst phase crossing is {worst * 100:.3f} % from the closed "
        f"form (bar {BAR['frequency_frac'] * 100:.1f} %); "
        f"analytic/measured/frac = {detail}")


@pytest.mark.parametrize("key", REFLECTING)
def test_the_recorded_phase_slope_follows_from_the_stored_s11(fixture, key):
    """The crossing comparison cannot see a conjugated S — both time conventions
    cross the real axis at the same frequencies — so the artifact records the
    slope of the unwrapped angle as well. This re-derives it, and checks it is a
    slope of the S11 beside it rather than a number somebody typed.

    No threshold is asserted: the pre-declaration's phase test is the crossings,
    and a second criterion is the PI's to declare, not this test's to invent.
    """
    entry = _solve(fixture, key)
    s11 = _complex(entry["s11"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    block = entry["phase"]["angle_slope_rad_per_hz"]
    measured = float(np.polyfit(freqs, np.unwrap(np.angle(s11)), 1)[0])
    assert block["measured"] == pytest.approx(measured, rel=1e-9), key
    assert block["same_sign"] == bool(
        block["measured"] * block["analytic_realized_length"] > 0.0), key


@pytest.mark.parametrize("kind", KINDS)
def test_the_matched_control_stays_under_its_floor(fixture, kind):
    """A deep null, held to an upper bound rather than compared in dB rung to
    rung — the PI's 2026-09-21 ruling. This number is the port's own reflection
    floor at that cell size."""
    entry = _solve(fixture, f"{kind}_matched_{CLAIMS_RUNG_UM}um")
    worst_db = float(_db(_complex(entry["s11"])).max())
    assert worst_db <= BAR["matched_floor_db"], (
        f"{kind}: the matched control's worst bin is {worst_db:.3f} dB, above the "
        f"{BAR['matched_floor_db']} dB floor")


# ---------------------------------------------------------------------------
# the ladder, re-derived from the stored S11 at every rung
# ---------------------------------------------------------------------------

def rederive_ladder(fixture, kind, dut):
    """Every rung against the finest, from the stored S11 only: passivity, the
    magnitude and crossing distances to the finest (or the matched floor), the
    flags those give against this file's bar, and the coarsest rung from which
    every finer rung sits inside the bar."""
    freqs = np.asarray(fixture["freqs_hz"], dtype=float)
    curves = {um: _complex(_solve(fixture, f"{kind}_{dut}_{um}um")["s11"])
              for um in RUNGS_UM}
    fine = curves[CLAIMS_RUNG_UM]
    reflecting = dut != "matched"
    if reflecting:
        crossings = {um: matched_crossings(freqs, curves[um], dut, um)
                     for um in RUNGS_UM}
    rows = []
    for um in RUNGS_UM:
        s = curves[um]
        row = {"rung_um": um,
               "max_abs_diff_vs_finest": float(np.abs(np.abs(s) - np.abs(fine)).max()),
               "passivity_within_bar": bool(np.abs(s).max() <= BAR["passivity_max"])}
        if reflecting:
            row["max_db_diff_vs_finest"] = float(np.abs(_db(s) - _db(fine)).max())
            row["magnitude_within_2dB_vs_finest"] = bool(
                row["max_db_diff_vs_finest"] <= BAR["magnitude_db"])
            cur, fin = crossings[um], crossings[CLAIMS_RUNG_UM]
            shared = sorted(set(cur) & set(fin))
            assert shared, f"{kind}_{dut}_{um}um shares no crossing with the finest"
            row["crossing_frac_vs_finest"] = max(
                abs(cur[m][1] - fin[m][1]) / fin[m][1] for m in shared)
            row["crossing_within_1pct_vs_finest"] = bool(
                row["crossing_frac_vs_finest"] <= BAR["frequency_frac"])
            row["lowest_crossing_hz"] = cur[min(cur)][1]
            flags = DECIDING_FLAGS["reflecting"]
        else:
            row["matched_floor_db"] = float(_db(s).max())
            row["matched_floor_abs"] = float(np.abs(s).max())
            row["matched_floor_within_bar"] = bool(
                row["matched_floor_db"] <= BAR["matched_floor_db"])
            flags = DECIDING_FLAGS["matched"]
        row["all_inside_bar"] = all(row[f] for f in flags)
        rows.append(row)
    recommended = next((rows[i]["rung_um"] for i in range(len(rows))
                        if all(r["all_inside_bar"] for r in rows[i:])), None)
    return rows, recommended


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dut", DUTS)
def test_the_ladder_recommends_the_named_cell_size(fixture, kind, dut):
    """The support matrix asks each family for one cell size: the coarsest from
    which the compared quantities stay inside the bar against the finest. It is
    re-derived here from the stored S11 and must be the rung this battery
    reports — 0.5 mm (60 cells per wavelength at 10 GHz) for the four
    reflecting loads, 1.0 mm (30 cells) for the matched floor. Nothing read
    from the fixture takes part in the decision."""
    rows, recommended = rederive_ladder(fixture, kind, dut)
    table = [(r["rung_um"], {k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in r.items() if k != "rung_um"}) for r in rows]
    assert recommended == RECOMMENDED_RUNG_UM[dut], (
        f"{kind}_{dut}: the ladder re-derived from the stored S11 recommends "
        f"{recommended} um, the battery reports {RECOMMENDED_RUNG_UM[dut]} um; "
        f"rows {table}")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dut", DUTS)
def test_the_stored_ladder_is_the_rederived_ladder(fixture, kind, dut):
    """The artifact's ladder block has to say what the re-derivation says:
    the same deciding flags, by name, the same verdict per rung, the same
    recommended rung, and numbers that agree to the method difference."""
    lad = fixture["ladder"][f"{kind}_{dut}"]
    rows, recommended = rederive_ladder(fixture, kind, dut)
    flags = DECIDING_FLAGS["matched" if dut == "matched" else "reflecting"]
    assert lad["rungs_um"] == list(RUNGS_UM)
    assert lad["finest"] == f"{kind}_{dut}_{CLAIMS_RUNG_UM}um"
    assert lad["coarsest_rung_within_bar"] == f"{kind}_{dut}_{recommended}um"
    assert len(lad["rows"]) == len(rows)
    for stored, mine in zip(lad["rows"], rows):
        tag = stored["rung"]
        assert tag == f"{kind}_{dut}_{mine['rung_um']}um"
        assert tuple(stored["deciding_flags"]) == flags, tag
        for flag in flags:
            assert stored[flag] == mine[flag], f"{tag}: {flag}"
        assert stored["all_inside_bar"] == mine["all_inside_bar"], tag
        assert stored["max_abs_diff_vs_finest"] == pytest.approx(
            mine["max_abs_diff_vs_finest"], rel=1e-9, abs=1e-15), tag
        if dut == "matched":
            assert stored["matched_floor_db"] == pytest.approx(
                mine["matched_floor_db"], abs=1e-9), tag
            assert stored["matched_floor_abs"] == pytest.approx(
                mine["matched_floor_abs"], rel=1e-12), tag
        else:
            assert stored["max_db_diff_vs_finest"] == pytest.approx(
                mine["max_db_diff_vs_finest"], rel=1e-9, abs=1e-12), tag
            assert stored["crossing_frac_vs_finest"] == pytest.approx(
                mine["crossing_frac_vs_finest"], abs=CROSSING_METHOD_TOL), tag
    # The sequence the ladder quotes, and the arithmetic done on it.
    if dut == "matched":
        assert lad["ladder_sequence"] == pytest.approx(
            [r["matched_floor_abs"] for r in rows], rel=1e-12)
    else:
        assert lad["ladder_sequence"] == pytest.approx(
            [r["lowest_crossing_hz"] for r in rows], rel=CROSSING_METHOD_TOL)
    seq = lad["ladder_sequence"]
    diffs = [abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)]
    assert lad["successive_diff"] == pytest.approx(diffs, rel=1e-12)
    assert lad["successive_diff_ratio"] == pytest.approx(diffs[1] / diffs[0], rel=1e-12)
    for p in (1, 2):
        assert lad["richardson"][f"order_{p}"] == pytest.approx(
            seq[-1] + (seq[-1] - seq[-2]) / (2.0 ** p - 1.0), rel=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_the_forward_identity_holds_on_the_eps_override_channel(fixture, kind):
    assert kind in fixture["identity"], f"no {kind} forward-identity record"
    ident = fixture["identity"][kind]
    a = _complex(ident["plain_s11"])
    b = _complex(ident["override_s11"])
    assert ident["max_abs_diff"] == pytest.approx(float(np.abs(a - b).max()),
                                                  rel=1e-12, abs=1e-300)
    np.testing.assert_allclose(b, a, rtol=BAR["identity_rtol"],
                               atol=BAR["identity_atol"])


# ---------------------------------------------------------------------------
# the gradient legs
# ---------------------------------------------------------------------------

def _adfd_cases(fixture, *, degenerate=False):
    """Every AD/FD case, split by THIS file's list of degenerate legs rather
    than by the label the artifact carries, so relabelling a leg cannot move it
    out of a comparison."""
    for name, block in fixture["adfd"].items():
        for case in block["cases"]:
            if ((name, case["objective"]) in DEGENERATE_CASES) is bool(degenerate):
                yield name, block, case


def closed_form_grad(fixture, block, objective) -> float:
    """The continuum closed form's derivative of the leg's objective, from this
    file's closed forms: in the total load resistance for the R leg, and in the
    scale theta on the eps_r = 2.2 filling (d/dtheta = eps_r d/d(eps_r) at
    theta = 1) for the permittivity legs. On the drawn length."""
    freqs = np.asarray(fixture["freqs_hz"], dtype=float)
    band = (freqs >= AD_BAND_HZ[0]) & (freqs <= AD_BAND_HZ[1])
    mid = int(np.argmin(np.abs(freqs - AD_MID_HZ)))
    zc_air = ETA0 * N_H_CELLS[block["kind"]]
    if block["leg"] == "adfd-r":
        r = R_OVER_ZC["res_double"] * zc_air
        beta = beta_of(freqs)
        s = s11_of(zin_line(zc_air, beta, LINE_M, r), zc_air)
        ds = ds11_dr(zc_air, beta, LINE_M, r, zc_air)
    else:
        zc = zc_air / math.sqrt(EPS_R_FILL)
        r = 0.0 if block["leg"] == "adfd-eps" else R_OVER_ZC["res_double"] * zc
        s, ds_deps = s11_and_ds11_deps(zc_air, EPS_R_FILL, freqs, LINE_M, r, zc)
        ds = EPS_R_FILL * ds_deps
    if objective == "band_mean_s11_sq":
        return float(np.mean(2.0 * np.real(np.conj(s[band]) * ds[band])))
    return float(np.real(ds[mid]))


def test_the_adfd_block_states_what_its_validity_assert_does_not_cover(fixture):
    """The ULP span says the two loss values are resolved from each other, not
    that the derivative is. An objective whose true derivative is zero gives two
    losses millions of ULPs apart whose difference is round-off, and the span
    passes it. The artifact has to carry that sentence, because a reader who
    sees only `ulp_span >= floor` will read the rel_err beside it as meaningful.
    """
    for name, block in fixture["adfd"].items():
        assert block.get("what_the_ulp_span_does_not_say"), name
        for case in block["cases"]:
            # Every case must carry the closed form's own gradient and the loss,
            # which are what a reader needs to tell a resolved derivative from a
            # resolved pair of losses.
            assert "closed_form" in case, f"{name}/{case['objective']}"
            assert "grad" in case["closed_form"], f"{name}/{case['objective']}"
            assert "loss" in case["ad"], f"{name}/{case['objective']}"


def test_the_objectives_are_taken_where_this_file_says(fixture):
    """The band the mean runs over and the bin Re(S11) is read at, as indices
    into the stored frequencies, must be the ones this file differentiates."""
    freqs = np.asarray(fixture["freqs_hz"], dtype=float)
    band = [int(i) for i in np.flatnonzero((freqs >= AD_BAND_HZ[0])
                                           & (freqs <= AD_BAND_HZ[1]))]
    mid = int(np.argmin(np.abs(freqs - AD_MID_HZ)))
    for name, block in fixture["adfd"].items():
        objs = block["objectives"]
        assert objs["band_mean_s11_sq"]["bins"] == band, name
        if "re_s11_at_mid_band" in objs:
            assert objs["re_s11_at_mid_band"]["bin_index"] == mid, name


def test_the_closed_form_gradients_are_this_files_closed_forms(fixture):
    """The third witness is re-derived here rather than read: an artifact whose
    closed-form gradients were replaced would otherwise move the convergence
    test below while every other check stayed green."""
    for name, block in fixture["adfd"].items():
        for case in block["cases"]:
            mine = closed_form_grad(fixture, block, case["objective"])
            stored = case["closed_form"]["grad"]
            assert stored == pytest.approx(mine, rel=1e-6, abs=1e-12), (
                f"{name}/{case['objective']}: stored closed-form gradient "
                f"{stored:.9e}, this file's closed form {mine:.9e}")


def test_the_gradient_matches_a_float64_finite_difference(fixture):
    """Criterion 3a as the contract defines it: the derivative rfx returns is
    the derivative of the line rfx solves.

    The bar is 0.05 and it is asserted on every leg this file does not list as
    degenerate, at every rung. A degenerate objective has no derivative to agree
    about — see ``test_the_degenerate_legs_are_this_files_and_carry_no_comparison``.
    PI decision of 2026-09-21, in the pre-declaration's addendum.
    """
    ads = list(_adfd_cases(fixture))
    assert len(ads) == sum(len(ADFD_OBJECTIVES[k.split("_")[1]]) for k in ADFD_KEYS) \
        - len(DEGENERATE_CASES)
    for name, block, case in ads:
        tag = f"{name}/{case['objective']}"
        assert case["ad"]["grad_finite"], f"{tag}: the AD gradient is not finite"
        assert np.isfinite(case["ad"]["grad"]), tag
        assert case["fd"]["loss_dtype"] == "float64", tag
        stored = abs(case["fd"]["f_plus"] - case["fd"]["f_minus"])
        mid = abs(0.5 * (case["fd"]["f_plus"] + case["fd"]["f_minus"]))
        ulp = float(np.spacing(np.asarray(mid, dtype=np.float64)))
        assert case["fd"]["ulp_span"] == pytest.approx(stored / ulp, rel=1e-9), tag
        # The comparator's resolving power is read BEFORE its verdict: a
        # reference that cannot resolve the quantity turns rel_err into noise.
        assert stored / ulp >= MIN_FD_ULP_SPAN, (
            f"{tag}: the float64 FD reference spans {stored / ulp:.3e} "
            f"ULP, below the {MIN_FD_ULP_SPAN:.0e} floor — its "
            "disagreement with AD would say nothing about the gradient")
        g_fd = ((case["fd"]["f_plus"] - case["fd"]["f_minus"])
                / (2.0 * case["fd"]["h"]))
        assert case["fd"]["grad"] == pytest.approx(g_fd, rel=1e-9), tag
        rel = abs(case["ad"]["grad"] - g_fd) / abs(g_fd)
        assert case["rel_err"] == pytest.approx(rel, rel=1e-9), tag
        assert rel <= BAR["ad_fd_rel"], f"{tag}: AD vs FD rel_err {rel:.4f}"


def test_the_closed_form_distance_shrinks_with_the_cell(fixture):
    """The closed form is a third witness, not a bar. PI decision, 2026-09-21.

    AD against FD is what carries criterion 3a, and it is asserted at 0.05 next
    door. The closed form is a different object from both: it differentiates the
    CONTINUUM line while AD and FD differentiate the LATTICE one, and at these
    cell sizes the solved line's phase is itself only converged to a fraction of
    a percent. Holding a continuum formula to 5 % against a lattice derivative
    was a fault of the original declaration — the addendum in
    ``docs/design_notes/lumped_wire_chain_battery_predeclaration.md`` records
    that and what replaces it.

    What replaces it is the statement the third witness can actually support:
    the distance to the closed form SHRINKS as the mesh refines, and shrinks at
    a rate consistent with first order. The families are named here — the
    mid-band Re(S11) on the filled short and the band mean on the filled
    resistor, per port kind — and every one of them must be present at all
    three cell sizes. The closed form is this file's, not the stored one.

    On the measurement this ships with the ratios are 0.557 and 0.530 on the
    resistive permittivity leg, and 0.728 and 0.646 on the mid-band one.
    """
    families = [(k, leg, obj) for k in KINDS
                for leg, obj in (("adfd-eps", "re_s11_at_mid_band"),
                                 ("adfd-eps-res", "band_mean_s11_sq"))]
    lo, hi = FIRST_ORDER_RATIO_WINDOW
    problems = []
    for kind, leg, objective in families:
        seq = []
        for um in RUNGS_UM:
            block = fixture["adfd"][f"{kind}_{leg}_{um}um"]
            case = next(c for c in block["cases"] if c["objective"] == objective)
            a, b = case["ad"]["grad"], closed_form_grad(fixture, block, objective)
            seq.append(abs(a - b) / max(abs(a), abs(b), 1e-300))
        tag = f"{kind}/{leg}/{objective}"
        if not (seq[0] > seq[1] > seq[2]):
            problems.append(
                f"{tag}: the distance to the closed form does not shrink with "
                f"the cell: {seq[0]:.5f} -> {seq[1]:.5f} -> {seq[2]:.5f}")
            continue
        ratios = [seq[1] / seq[0], seq[2] / seq[1]]
        if not all(lo <= r <= hi for r in ratios):
            problems.append(
                f"{tag}: successive ratios {ratios[0]:.4f}, {ratios[1]:.4f} are "
                f"not both inside the first-order window [{lo}, {hi}]; the "
                f"distances are {seq[0]:.5f} -> {seq[1]:.5f} -> {seq[2]:.5f}")
    assert not problems, (
        "the closed-form witness does not converge like a discretisation "
        "offset:\n" + "\n".join(problems))


def test_the_degenerate_legs_are_this_files_and_carry_no_comparison(fixture):
    """Band-mean |S11|^2 on the filled SHORT is identically 1, so its derivative
    is zero and AD and FD both return residue about it. Those records stay in
    the artifact as measured and are held to nothing.

    Which legs those are is written in this file, and re-derived: each listed
    leg's own stored S11 has |S11|^2 within ``LOSSLESS_POWER_TOL`` of 1 at every
    bin, and every OTHER band-mean leg's does not, so no leg can be excused by
    editing a label. The label the artifact carries has to match the list, the
    closed form's derivative (this file's) has to vanish on the listed legs,
    and the listed legs carry no verdict.
    """
    labelled = set()
    for name, block in fixture["adfd"].items():
        for obj, spec in block["objectives"].items():
            if spec.get("degenerate_on_this_dut"):
                labelled.add((name, obj))
                assert spec.get("why"), f"{name}/{obj}: marked degenerate with no reason"
    assert labelled == DEGENERATE_CASES, (
        f"legs labelled degenerate {sorted(labelled)} are not this file's "
        f"{sorted(DEGENERATE_CASES)}")
    for name, block in fixture["adfd"].items():
        s = _complex(block["s11_at_theta0"])
        power_off = float(np.abs(np.abs(s) ** 2 - 1.0).max())
        for case in block["cases"]:
            tag = f"{name}/{case['objective']}"
            if (name, case["objective"]) in DEGENERATE_CASES:
                assert power_off <= LOSSLESS_POWER_TOL, (
                    f"{tag}: |S11|^2 is {power_off:.3e} from 1 somewhere in band, "
                    "so the objective is not constant and this leg must be compared")
                assert abs(closed_form_grad(fixture, block, case["objective"])) <= 1e-12, tag
                verdicts = [k for k in _walk_keys(case)
                            if k in ("within_bar", "passed", "ok", "verdict")]
                assert not verdicts, f"{tag}: the record carries a verdict key {verdicts}"
            elif case["objective"] == "band_mean_s11_sq":
                assert power_off > LOSSLESS_POWER_TOL, (
                    f"{tag}: |S11|^2 is within {LOSSLESS_POWER_TOL} of 1 at every "
                    "bin, which would make this objective constant — a leg the "
                    "battery compares cannot be one")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_the_openems_record_is_carried_as_context_only(fixture):
    """The recorded openEMS comparison is on a different fixture, so the battery
    must not have compared anything against it. What is checkable is that the
    record keeps its own scope limits attached and that no measured entry
    references it."""
    ctx = fixture.get("openems_context")
    assert ctx is not None
    assert ctx["producers"], "the context block names no producer"
    assert ctx.get("note"), "the context block does not say what it is"
    for name, rec in ctx["records"].items():
        assert rec.get("claim_scope"), f"{name} carries no claim_scope"
        assert rec.get("completion_decision"), f"{name} carries no completion_decision"
        assert rec.get("_committed_as"), (
            f"{name} does not say where it came from or what was edited to "
            "commit it")
    # Nothing measured may point at it: a referee is named in `referee`, and
    # this record is not one.
    for key, entry in fixture["solves"].items():
        blob = json.dumps(entry).lower()
        assert "openems" not in blob, (
            f"{key} references the openEMS record; the pre-declaration carries "
            "it as context and compares nothing against it")


def test_the_port_kind_ab_block_is_internally_consistent(fixture):
    """The A/B the lumped leg turns on: one cell of one line declared two ways,
    in front of three loads whose reflection is an exact number.

    This checks the block says what its own numbers say — the closed-form
    column is |Gamma_L| for each load, and each port kind's distance from it is
    the distance its stored curve actually has. It asserts no verdict: which
    port kind is inside the bar is decided by the battery's own gates on the
    battery's own solves, not here.
    """
    ab = fixture["port_kind_ab"]
    assert ab["present"] is True, (
        "the A/B record was not produced; run its producer with no arguments")
    assert ab["producer"].endswith("lumped_port_known_load_line.py")
    zc = ab["channel"]["zc_ohm"]
    assert ab["channel"]["zref_ohm"] == pytest.approx(zc, rel=1e-12), (
        "the A/B only collapses to |Gamma_L| when Zref is the line's own Zc")
    assert set(ab["loads"]) == {"half_zc", "matched", "double_zc"}, sorted(ab["loads"])
    for name, row in ab["loads"].items():
        r = row["r_over_zc"]
        assert row["r_ohm"] == pytest.approx(r * zc, rel=1e-12), name
        assert row["closed_form_abs_s11"] == pytest.approx(
            abs((r - 1.0) / (r + 1.0)), rel=1e-12), name
        for kind in KINDS:
            side = row[kind]
            worst = max(abs(v - row["closed_form_abs_s11"]) for v in side["abs_s11"])
            assert side["max_abs_from_closed_form"] == pytest.approx(worst, rel=1e-9), (
                f"{name}/{kind}")
            assert len(side["abs_s11"]) == len(ab["freqs_hz"]), f"{name}/{kind}"


def test_the_not_in_chain_block_gates_nothing(fixture):
    """``run(compute_s_params=True)`` is not in the v2.0 chain for this family.

    The block is carried because the two paths disagreed on this channel and a
    number nobody wrote down gets measured again. What is checked is that its
    derived statement follows from its own stored arrays, and that it is not
    quietly being used as a gate: it carries no bar and no pass/fail.
    """
    obs = fixture["not_in_chain_observations"]
    for kind, block in obs.items():
        assert "not in the v2.0 chain" in block["scope"].lower(), kind
        blob = json.dumps(block).lower()
        for word in ("bar", "within_bar", "pass", "fail"):
            assert f'"{word}"' not in blob, (
                f"{kind}: the not-in-chain block carries a {word!r} key, which "
                "would make an observation into a gate")
        li = block["run_path_load_independence"]
        runs = {c["dut"]: _complex(c["run_s11"]) for c in block["cases"]}
        assert set(runs) == set(DUTS), kind
        ref = next(iter(runs.values()))
        for dut, v in runs.items():
            assert li["identical_to_first_dut"][dut] == bool(np.array_equal(v, ref)), dut
        assert li["all_identical"] == all(li["identical_to_first_dut"].values())
        assert li["run_abs_max"] == pytest.approx(
            max(float(np.abs(v).max()) for v in runs.values()), rel=1e-12)


def test_the_degenerate_objective_evidence_follows_from_the_records(fixture):
    """The ULP-span floor is computed on the two LOSS values, so it passes on an
    objective whose derivative is zero and whose loss difference is residue.

    The block states that with the numbers beside it. This checks the numbers
    are the ones in the records, and that the rows it lists are exactly this
    file's degenerate legs — a block that quietly listed a healthy leg would be
    an excuse rather than evidence.
    """
    ev = fixture["degenerate_objective_evidence"]
    assert {(row["leg"], "band_mean_s11_sq") for row in ev["rows"]} == DEGENERATE_CASES
    for row in ev["rows"]:
        block = fixture["adfd"][row["leg"]]
        case = [c for c in block["cases"] if c["objective"] == "band_mean_s11_sq"][0]
        assert row["ulp_span"] == pytest.approx(case["fd"]["ulp_span"], rel=1e-12)
        assert row["span_above_floor"] == bool(case["fd"]["ulp_span"] >= MIN_FD_ULP_SPAN)
        assert row["closed_form_grad"] == pytest.approx(
            case["closed_form"]["grad"], rel=1e-12, abs=1e-30)


def test_the_fitted_electrical_length_follows_from_its_own_short(fixture):
    """Every AD leg fits the line's own electrical length from a short it solved
    at that rung. This re-derives the fit from the stored short and checks the
    stated length is the one that fit gives."""
    freqs = np.asarray(fixture["freqs_hz"], dtype=float)
    for name, block in fixture["adfd"].items():
        fit = block.get("fitted_electrical_length")
        assert fit, f"{name} carries no fitted electrical length"
        s11 = _complex(fit["s11_short"])
        slope, intercept = np.polyfit(freqs, np.unwrap(np.angle(s11)), 1)
        # Two checks with two different jobs. The length must follow from the
        # STORED slope exactly — that is arithmetic and catches a hand edit.
        exact = -fit["slope_rad_per_hz"] * C0 / (4.0 * math.pi * math.sqrt(fit["eps_r"]))
        assert fit["length_m"] == pytest.approx(exact, rel=1e-15), name
        # The stored slope must come from the stored S11, which is a looser
        # question: the driver takes the angle of a complex64 array, so its
        # unwrap and polyfit run in float32 while this re-derivation runs in
        # float64. The two agree to about float32 epsilon (1.5e-08 on the legs
        # measured here) and no tighter. A bound of 1e-9 would not be a tighter
        # test of the fit, only a test of which dtype ran it — and `rel` alone
        # would not even catch that, because pytest.approx keeps a 1e-12
        # ABSOLUTE floor and these slopes are of order 1e-09.
        assert fit["slope_rad_per_hz"] == pytest.approx(float(slope), rel=1e-6,
                                                        abs=0.0), name
        assert fit["frac_from_declared"] == pytest.approx(
            abs(fit["length_m"] - fit["declared_length_m"]) / fit["declared_length_m"],
            rel=1e-12), name
        # The fit's own residual has to be carried: a straight line is an
        # assumption about the record, and the lattice's dispersion is not
        # linear in frequency.
        assert fit["rms_residual_rad"] > 0.0, name
        assert "least squares" in fit["fit"], name


def test_perturbing_the_stored_s11_breaks_its_own_summary(fixture):
    """A summary that no longer follows from the S11 beside it has to be
    visible to the checks above: a magnitude change moves the stored magnitude
    summaries, and a phase-only change, which no magnitude check can see, moves
    the crossings the phase test reads."""
    key = CLAIMS_SOLVES[0]
    entry = _solve(fixture, key)
    s11 = _complex(entry["s11"]) * 1.01            # 0.086 dB, far inside any bar
    assert not np.allclose(_db(s11), entry["s11_db"], rtol=0, atol=1e-9)
    assert not np.allclose(np.abs(s11), entry["abs_s11"], rtol=0, atol=1e-12)
    assert np.abs(s11).max() != pytest.approx(entry["max_abs_s11"], rel=1e-12)
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    rotated = _complex(entry["s11"]) * np.exp(1j * 0.05 * freqs / freqs[-1])
    before = real_axis_crossings(freqs, _complex(entry["s11"]))
    after = real_axis_crossings(freqs, rotated)
    assert before and after
    assert [f for f, _ in before] != [f for f, _ in after]
