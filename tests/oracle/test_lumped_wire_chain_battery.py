"""Replay of the lumped / wire port chain battery. Reads the fixture, re-derives
every assembled number from the stored S11, and compares the result against the
bar.

No FDTD runs here. The measurement is
``scripts/diagnostics/lumped_wire_chain_battery_measure.py``, pre-declared in
``docs/design_notes/lumped_wire_chain_battery_predeclaration.md``; the bar is
the v2.0 one in ``docs/design_notes/chain_closure_contract.md`` — magnitude
within 2 dB, frequencies within 1 %, AD against FD within 0.05, forward
identity at rtol 1e-5 / atol 1e-7 — plus the two numbers this family's
pre-declaration adds in place of reciprocity and power closure: passivity
``max_f |S11| <= 1.02`` on a settled record, and the matched control held to
``-20 dB``.

Every derived quantity is recomputed HERE from the stored complex S11 by an
implementation written independently of the assembler's, so a fixture whose
summary numbers were edited by hand — or an assembler whose arithmetic drifts —
reds instead of replaying. ``test_perturbing_the_stored_s11_breaks_its_own_
summary`` runs that as a mutation: it revives the defect (a wrong S11 behind a
right-looking summary) with the checks left exactly as they ship.

The closed forms the battery is refereed against are unit-tested here against a
hand-computed value first, so a comparison against them is a comparison against
arithmetic somebody checked rather than against the driver's own code.
"""
import cmath
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


# ---------------------------------------------------------------------------
# independent arithmetic — deliberately not imported from the driver
# ---------------------------------------------------------------------------

def _complex(block) -> np.ndarray:
    return (np.asarray(block["real"], dtype=float)
            + 1j * np.asarray(block["imag"], dtype=float))


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


def crossings_of(freqs, s11):
    """Frequencies where the unwrapped angle passes a multiple of pi, by linear
    interpolation. Independent of the assembler's walk."""
    ang = np.unwrap(np.angle(s11))
    out = []
    for k in range(len(ang) - 1):
        a, b = ang[k], ang[k + 1]
        lo, hi = (a, b) if a < b else (b, a)
        m_lo = int(math.floor(lo / math.pi))
        m_hi = int(math.ceil(hi / math.pi))
        for m in range(m_lo, m_hi + 1):
            level = m * math.pi
            if (a - level) * (b - level) < 0.0:
                f = freqs[k] + (level - a) / (b - a) * (freqs[k + 1] - freqs[k])
                out.append((m, float(f)))
            elif a == level:
                out.append((m, float(freqs[k])))
    return sorted(set(out), key=lambda r: r[1])


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

    t = math.tan(beta_l)
    zin = zc * (r + 1j * zc * t) / (zc + 1j * r * t)
    dzin = zc ** 2 * (1.0 + t ** 2) / (zc + 1j * r * t) ** 2
    ds = 2.0 * zref * dzin / (zin + zref) ** 2
    assert ds.real == pytest.approx(0.0078007339, abs=1e-9)
    assert ds.imag == pytest.approx(-0.0042615603, abs=1e-9)

    # And against a central difference of the closed form itself, which is an
    # independent route to the same derivative.
    h = 1e-5
    fd = (s11_of(zin_line(zc, beta, 0.02, r + h), zref)
          - s11_of(zin_line(zc, beta, 0.02, r - h), zref)) / (2 * h)
    assert abs(ds - fd) / abs(ds) < 1e-8


def test_an_open_and_a_short_are_all_phase():
    """A lossless reactive termination cannot change the magnitude: the referee
    has to return exactly one for both, or the phase comparisons downstream are
    reading a magnitude error as a phase error."""
    beta = beta_of(np.linspace(1e9, 10e9, 91))
    for z_load in (0.0, None):
        s = s11_of(zin_line(376.73, beta, 0.03, z_load), 376.73)
        np.testing.assert_allclose(np.abs(s), 1.0, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE} has not been measured yet")
    return json.loads(FIXTURE.read_text())


def _solve(fixture, key):
    entry = fixture["solves"].get(key)
    if entry is None:
        pytest.skip(f"the battery has no {key} record")
    return entry


ALL_SOLVES = [f"{k}_{d}_{um}um" for k in KINDS for d in DUTS for um in RUNGS_UM]
CLAIMS_SOLVES = [f"{k}_{d}_{CLAIMS_RUNG_UM}um" for k in KINDS for d in DUTS]
REFLECTING = [f"{k}_{d}_{CLAIMS_RUNG_UM}um" for k in KINDS
              for d in ("short", "open", "res_half", "res_double")]


def test_the_fixture_names_its_own_provenance(fixture):
    assert fixture["schema"] == "rfx.lumped_wire_chain_battery"
    assert fixture["predeclaration"].endswith(
        "lumped_wire_chain_battery_predeclaration.md")
    assert len(fixture["assembler_commit"]) == 40
    assert fixture["deviations"], "a fixture that lists no deviation from its "\
        "pre-declaration is claiming there were none"
    for key, entry in fixture["solves"].items():
        p = entry["provenance"]
        assert len(p["commit"]) == 40, f"{key} carries no resolvable commit"
        # The fixture does not ship machine paths; it ships the question they
        # answered — did `import rfx` resolve to the run's own tree?
        assert p["rfx_import_tail"] == "rfx/__init__.py", key
        assert p["rfx_resolved_inside_the_run_tree"] is True, key
        assert p["compute_run_id"], f"{key} does not name the run that produced it"
        assert p["jax_version"] and p["numpy_version"], key
        assert entry["preflight_text"] is not None, key
        assert entry["wall_s"] > 0.0, key


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_the_realized_channel_is_the_declared_channel(fixture, key):
    """The measurement asserted this before solving; the fixture has to carry
    the evidence, because a number whose channel was never checked is not one of
    this battery's numbers."""
    entry = fixture["solves"].get(key)
    if entry is None:
        pytest.skip(f"no {key} record")
    d, g, ps = entry["declared"], entry["realized_grid"], entry["port_spec"]
    assert g["grid_shape_nodes"] == [d["n_nodes_x"], 2, d["n_h_cells"] + 1], key
    assert g["port_index"] == [d["i_port"], 0, 0], key
    want_pec = {"z_lo", "z_hi"} | ({"x_hi"} if entry["dut"] == "short" else set())
    assert set(g["boundary_faces"]["pec"]) == want_pec, key
    assert ps["impedance_ohm"] == pytest.approx(d["zref_ohm"], rel=1e-9), key
    if entry["kind"] == "wire":
        assert ps["n_live"] == d["n_h_cells"], key
        assert ps["excite"] is True, key
    if d["i_rlc"] is not None:
        assert g["rlc_indices"] == [[d["i_rlc"], 0, k]
                                    for k in range(d["n_h_cells"])], key
        assert len(g["rlc_values_ohm"]) == d["n_h_cells"], key
        assert sum(g["rlc_values_ohm"]) == pytest.approx(d["r_total_ohm"], rel=1e-9), key
    # The realized length follows from the node layout, not from a stored float.
    assert d["length_m_realized"] == pytest.approx(
        d["length_cells_realized"] * d["dx_m"], rel=1e-12), key


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_every_stored_summary_follows_from_the_stored_s11(fixture, key):
    entry = fixture["solves"].get(key)
    if entry is None:
        pytest.skip(f"no {key} record")
    s11 = _complex(entry["s11"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    assert s11.shape == (len(freqs),)

    np.testing.assert_allclose(np.abs(s11), entry["abs_s11"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(_db(s11), entry["s11_db"], rtol=0, atol=1e-9)
    assert entry["max_abs_s11"] == pytest.approx(np.abs(s11).max(), rel=1e-12)
    assert entry["min_abs_s11"] == pytest.approx(np.abs(s11).min(), rel=1e-12)
    assert entry["argmax_abs_s11_bin"] == int(np.argmax(np.abs(s11)))
    assert entry["passivity"]["max_abs_s11"] == pytest.approx(
        np.abs(s11).max(), rel=1e-12)

    # The referee, rebuilt here from the declared channel and the realized length.
    d = entry["declared"]
    z_load = 0.0 if entry["dut"] == "short" else d["z_load_ohm"]
    an = s11_of(zin_line(d["zc_ohm"], beta_of(freqs), d["length_m_realized"], z_load),
                d["zref_ohm"])
    np.testing.assert_allclose(_complex(entry["referee"]["analytic_realized_length"]),
                               an, rtol=1e-9, atol=1e-12)
    mv = entry["magnitude_vs_analytic"]
    assert mv["abs_gamma_load"] == pytest.approx(abs(gamma_of(z_load, d["zc_ohm"])),
                                                 rel=1e-12)
    if entry["dut"] == "matched":
        # A dB difference against an identically-zero reference is a difference
        # against the log floor. The artifact must not carry one.
        assert mv["applies"] is False
        assert "max_abs_db_diff" not in mv
        assert "within_bar" not in mv
    else:
        assert mv["applies"] is True
        assert mv["max_abs_db_diff"] == pytest.approx(
            float(np.abs(_db(s11) - _db(an)).max()), rel=1e-9)


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_the_record_length_witness_is_present_and_recomputes(fixture, key):
    """``forward(port_s11_freqs=...)`` emits no settling witness, so the
    contract's admissible substitute is record-length invariance. A solve with
    no doubled record has no witness at all, which is a harder failure than a
    wide one."""
    entry = fixture["solves"].get(key)
    if entry is None:
        pytest.skip(f"no {key} record")
    rd = entry["record_doubling"]
    assert rd is not None, f"{key} carries no record-doubling witness"
    assert rd["n_steps"] > entry["n_steps"], key
    assert rd["max_abs_db_shift"] <= rd["bar_db"], (
        f"{key}: doubling the record moved |S11| by {rd['max_abs_db_shift']:.5f} dB, "
        f"above the {rd['bar_db']} dB substitute for a settling witness — the "
        "record was truncated, so this S11 is not interpretable")


@pytest.mark.parametrize("key", CLAIMS_SOLVES)
def test_the_one_port_is_passive_at_the_claims_rung(fixture, key):
    """What replaces reciprocity and power closure for a one-port: a passive
    load cannot reflect more than it received."""
    entry = _solve(fixture, key)
    measured = float(np.abs(_complex(entry["s11"])).max())
    assert measured <= fixture["bar"]["passivity_max"], (
        f"{key}: max|S11| = {measured:.6f} exceeds "
        f"{fixture['bar']['passivity_max']} on a passive termination")


@pytest.mark.parametrize("key", REFLECTING)
def test_the_magnitude_at_the_claims_rung_matches_the_closed_form(fixture, key):
    """|S11| is |Gamma_L| at every bin on a lossless line referenced to its own
    Zc, so this is the v2 bar's magnitude test with the reference known
    exactly."""
    entry = _solve(fixture, key)
    s11 = _complex(entry["s11"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    d = entry["declared"]
    z_load = 0.0 if entry["dut"] == "short" else d["z_load_ohm"]
    an = s11_of(zin_line(d["zc_ohm"], beta_of(freqs), d["length_m_realized"], z_load),
                d["zref_ohm"])
    worst = float(np.abs(_db(s11) - _db(an)).max())
    assert worst <= fixture["bar"]["magnitude_db"], (
        f"{key}: |S11| is {worst:.4f} dB from the closed form's "
        f"{abs(gamma_of(z_load, d['zc_ohm'])):.6f} at worst — the bar is "
        f"{fixture['bar']['magnitude_db']} dB")


@pytest.mark.parametrize("key", REFLECTING)
def test_the_phase_crossings_at_the_claims_rung_land_within_one_percent(fixture, key):
    """The pre-declaration's phase test: the frequencies at which angle(S11)
    crosses 0 and pi, against the closed form, to 1 %."""
    entry = _solve(fixture, key)
    s11 = _complex(entry["s11"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    d = entry["declared"]
    z_load = 0.0 if entry["dut"] == "short" else d["z_load_ohm"]
    phi = cmath.phase(gamma_of(z_load, d["zc_ohm"]))
    slope = 2.0 * 2.0 * math.pi * d["length_m_realized"] / C0
    meas = crossings_of(freqs, s11)
    assert meas, f"{key}: the measured angle crosses no multiple of pi in band"
    worst = 0.0
    detail = []
    partners = []
    for m in range(-4000, 4000):
        f_an = (phi - m * math.pi) / slope
        if not (freqs[0] <= f_an <= freqs[-1]):
            continue
        same = [f for mm, f in meas if (mm - m) % 2 == 0]
        if not same:
            continue
        best = min(same, key=lambda f: abs(f - f_an))
        frac = abs(best - f_an) / f_an
        detail.append((f_an, best, frac))
        partners.append(best)
        worst = max(worst, frac)
    assert detail, f"{key}: no analytic crossing fell inside the measured band"
    # Nearest-neighbour matching alone cannot see a shift of a whole crossing
    # period: the curve would alias onto its neighbour and every distance would
    # come back small. Two analytic crossings sharing one measured partner is
    # exactly that, so it is a failure rather than a small number.
    assert len(set(partners)) == len(partners), (
        f"{key}: two analytic crossings matched the same measured crossing "
        f"({partners}) — the measured phase is shifted by about a whole crossing "
        "period, which nearest-neighbour matching would otherwise hide")
    assert worst <= fixture["bar"]["frequency_frac"], (
        f"{key}: the worst phase crossing is {worst * 100:.3f} % from the closed "
        f"form (bar {fixture['bar']['frequency_frac'] * 100:.1f} %); "
        f"analytic/measured/frac = {[(round(a / 1e9, 5), round(b / 1e9, 5), round(c, 5)) for a, b, c in detail]}")


@pytest.mark.parametrize("key", REFLECTING)
def test_the_recorded_phase_slope_follows_from_the_stored_s11(fixture, key):
    """The crossing comparison cannot see a conjugated S — both time conventions
    cross multiples of pi at the same frequencies — so the artifact records the
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
    s11 = _complex(entry["s11"])
    worst_db = float(_db(s11).max())
    assert worst_db == pytest.approx(entry["matched_floor"]["max_db"], abs=1e-9)
    assert worst_db <= fixture["bar"]["matched_floor_db"], (
        f"{kind}: the matched control's worst bin is {worst_db:.3f} dB, above the "
        f"{fixture['bar']['matched_floor_db']} dB floor")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dut", DUTS)
def test_the_ladder_follows_from_the_stored_curves(fixture, kind, dut):
    lad = fixture["ladder"].get(f"{kind}_{dut}")
    if lad is None:
        pytest.skip(f"no {kind}_{dut} ladder")
    assert lad["rungs_um"] == list(RUNGS_UM), lad["rungs_um"]
    fine = fixture["solves"][lad["finest"]]
    fine_db = np.asarray(fine["s11_db"], dtype=float)
    fine_abs = np.asarray(fine["abs_s11"], dtype=float)
    for row in lad["rows"]:
        entry = fixture["solves"][row["rung"]]
        cur_abs = np.asarray(entry["abs_s11"], dtype=float)
        assert row["max_abs_diff_vs_finest"] == pytest.approx(
            float(np.abs(cur_abs - fine_abs).max()), rel=1e-9), row["rung"]
        if dut == "matched":
            # No dB comparison against a deep null, rung to rung or otherwise.
            assert "max_db_diff_vs_finest" not in row, row["rung"]
            assert row["matched_floor_abs"] == pytest.approx(
                float(cur_abs.max()), rel=1e-12), row["rung"]
        else:
            cur = np.asarray(entry["s11_db"], dtype=float)
            assert row["max_db_diff_vs_finest"] == pytest.approx(
                float(np.abs(cur - fine_db).max()), rel=1e-9), row["rung"]
            assert row["magnitude_within_2dB_vs_finest"] == bool(
                np.abs(cur - fine_db).max() <= fixture["bar"]["magnitude_db"])
        # The recommendation follows from the NAMED flags, so a renamed key
        # cannot drop out of the decision unnoticed.
        assert row["deciding_flags"], row["rung"]
        assert row["all_inside_bar"] == all(bool(row[f]) for f in row["deciding_flags"])
    qualifying = [r["rung"] for r in lad["rows"] if r["all_inside_bar"]]
    assert lad["coarsest_rung_within_bar"] == (qualifying[0] if qualifying else None)


@pytest.mark.parametrize("kind", KINDS)
def test_the_ladder_recommends_a_cell_size(fixture, kind):
    """The support matrix asks each family for one cell size. A ladder in which
    no rung sits inside the bar against the finest recommends none, and that is
    the failure this states rather than hides."""
    lads = {k: v for k, v in fixture["ladder"].items() if k.startswith(kind + "_")}
    if not lads:
        pytest.skip(f"no {kind} ladder")
    missing = [k for k, v in lads.items() if v["coarsest_rung_within_bar"] is None]
    assert not missing, (
        f"{kind}: no cell size sits inside the bar for {missing} — the battery "
        "recommends no resolution for this port kind")


@pytest.mark.parametrize("kind", KINDS)
def test_the_forward_identity_holds_on_the_eps_override_channel(fixture, kind):
    ident = fixture["identity"].get(kind)
    if ident is None:
        pytest.skip(f"no {kind} forward-identity stage")
    a = _complex(ident["plain_s11"])
    b = _complex(ident["override_s11"])
    assert ident["max_abs_diff"] == pytest.approx(float(np.abs(a - b).max()), rel=1e-12)
    np.testing.assert_allclose(b, a, rtol=ident["rtol"], atol=ident["atol"])


def _adfd_cases(fixture):
    for name, block in fixture.get("adfd", {}).items():
        for case in block["cases"]:
            yield name, block, case


def test_the_adfd_block_states_what_its_validity_assert_does_not_cover(fixture):
    """The ULP span says the two loss values are resolved from each other, not
    that the derivative is. An objective whose true derivative is zero gives two
    losses millions of ULPs apart whose difference is round-off, and the span
    passes it. The artifact has to carry that sentence, because a reader who
    sees only `ulp_span >= floor` will read the rel_err beside it as meaningful.
    """
    if not fixture.get("adfd"):
        pytest.skip("no AD/FD stage is assembled")
    for name, block in fixture["adfd"].items():
        assert block.get("what_the_ulp_span_does_not_say"), name
        for case in block["cases"]:
            # Every case must carry the closed form's own gradient and the loss,
            # which are what a reader needs to tell a resolved derivative from a
            # resolved pair of losses.
            assert "closed_form" in case, f"{name}/{case['objective']}"
            assert "grad" in case["closed_form"], f"{name}/{case['objective']}"
            assert "loss" in case["ad"], f"{name}/{case['objective']}"


def test_the_gradient_matches_a_float64_finite_difference(fixture):
    ads = list(_adfd_cases(fixture))
    if not ads:
        pytest.skip("no AD/FD stage is assembled")
    for name, block, case in ads:
        tag = f"{name}/{case['objective']}"
        assert case["ad"]["grad_finite"], f"{tag}: the AD gradient is not finite"
        assert case["fd"]["loss_dtype"] == "float64", tag
        stored = abs(case["fd"]["f_plus"] - case["fd"]["f_minus"])
        mid = abs(0.5 * (case["fd"]["f_plus"] + case["fd"]["f_minus"]))
        ulp = float(np.spacing(np.asarray(mid, dtype=np.float64)))
        assert case["fd"]["ulp_span"] == pytest.approx(stored / ulp, rel=1e-9), tag
        # The comparator's resolving power is read BEFORE its verdict: a
        # reference that cannot resolve the quantity turns rel_err into noise.
        assert case["fd"]["ulp_span"] >= block["min_fd_ulp_span"], (
            f"{tag}: the float64 FD reference spans {case['fd']['ulp_span']:.3e} "
            f"ULP, below the {block['min_fd_ulp_span']:.0e} floor — its "
            "disagreement with AD would say nothing about the gradient")
        g_fd = ((case["fd"]["f_plus"] - case["fd"]["f_minus"])
                / (2.0 * case["fd"]["h"]))
        assert case["fd"]["grad"] == pytest.approx(g_fd, rel=1e-9), tag
        rel = abs(case["ad"]["grad"] - g_fd) / abs(g_fd)
        assert case["rel_err"] == pytest.approx(rel, rel=1e-9), tag
        assert rel <= block["bar"], f"{tag}: AD vs FD rel_err {rel:.4f}"


def test_the_pairwise_gradient_falsifier(fixture):
    """The pre-declaration's fourth falsifier, exactly as written: 'If AD, FD and
    the closed-form derivative disagree by more than 5 % pairwise where the FD is
    interpretable, criterion 3a is open.'

    The closed form is a different object from AD and FD — it differentiates the
    continuum line, they differentiate the lattice — so this is the leg that can
    separate a gradient defect from a discretisation offset. It asserts the
    falsifier rather than a verdict.
    """
    ads = list(_adfd_cases(fixture))
    if not ads:
        pytest.skip("no AD/FD stage is assembled")
    failures = []
    for name, block, case in ads:
        pw = case["pairwise"]
        if not pw["interpretable"]:
            continue
        if pw["max_rel"] > pw["bar"]:
            failures.append(
                f"{name}/{case['objective']}: {pw['rel']} with grads {pw['grads']} "
                f"and bin context {case.get('bin_context')}")
    assert not failures, (
        "AD, FD and the closed-form derivative disagree by more than the bar:\n"
        + "\n".join(failures))


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


def test_perturbing_the_stored_s11_breaks_its_own_summary(fixture):
    """The mutation that revives the defect these checks exist for: a summary
    that no longer follows from the S11 beside it. Only the stored complex
    numbers move; every check above is the shipped one."""
    key = next((k for k in CLAIMS_SOLVES if k in fixture["solves"]), None)
    if key is None:
        pytest.skip("no claims-rung record")
    entry = fixture["solves"][key]
    s11 = _complex(entry["s11"])
    s11 = s11 * 1.01                       # 0.086 dB, far inside any bar
    assert not np.allclose(_db(s11), entry["s11_db"], rtol=0, atol=1e-9)
    assert not np.allclose(np.abs(s11), entry["abs_s11"], rtol=0, atol=1e-12)
    assert np.abs(s11).max() != pytest.approx(entry["max_abs_s11"], rel=1e-12)
    # And a phase-only perturbation, which the magnitude checks cannot see: it
    # has to move the crossings the phase test reads.
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    rotated = _complex(entry["s11"]) * np.exp(1j * 0.05 * freqs / freqs[-1])
    before = crossings_of(freqs, _complex(entry["s11"]))
    after = crossings_of(freqs, rotated)
    assert before and after
    assert [f for _, f in before] != [f for _, f in after]
