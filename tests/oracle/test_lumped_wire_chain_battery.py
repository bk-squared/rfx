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
    assert g["port_impedance_ohm"] == pytest.approx(d["zref_ohm"], rel=1e-9), key
    assert g["port_extent_cells"] == (None if entry["kind"] == "lumped"
                                      else d["n_h_cells"]), key
    assert g["port_excite"] is True, key
    want_pec = {"z_lo", "z_hi"} | ({"x_hi"} if entry["dut"] == "short" else set())
    want_pmc = ({"x_lo", "y_lo", "y_hi"}
                | (set() if entry["dut"] == "short" else {"x_hi"}))
    assert set(g["boundary_faces"]["pec"]) == want_pec, key
    assert set(g["boundary_faces"]["pmc"]) == want_pmc, key
    assert ps["impedance_ohm"] == pytest.approx(d["zref_ohm"], rel=1e-9), key
    if entry["kind"] == "wire":
        assert ps["n_live"] == d["n_h_cells"], key
        assert ps["excite"] is True, key
    want_n = d["n_h_cells"] if d["i_rlc"] is not None else 0
    assert g["n_rlc_elements"] == want_n, key
    if want_n:
        assert g["rlc_indices"] == [[d["i_rlc"], 0, k]
                                    for k in range(d["n_h_cells"])], key
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


# The window a first-order sequence's successive ratio falls in. A quantity
# converging at first order in the cell size halves each time the mesh halves,
# so the ratio sits near 0.5; the window is wide enough for the second-order
# term that is still present at these cell sizes and narrow enough to exclude a
# sequence that is not converging at all (a ratio near 1).
FIRST_ORDER_RATIO_WINDOW = (0.35, 0.75)


def _is_degenerate(block, case) -> bool:
    """Whether this case's objective has no derivative to compare.

    Band-mean |S11|^2 on a SHORT-terminated line referenced to its own Zc is
    identically 1, so its derivative with respect to anything is zero and both
    AD and FD return round-off. The driver marks that on the objective; the
    marking is what every gate below reads, so a leg cannot be excused from a
    comparison without the artifact saying it is degenerate.
    """
    return bool(block["objectives"][case["objective"]].get("degenerate_on_this_dut"))


def _adfd_cases(fixture, *, degenerate=False):
    """Every AD/FD case. ``degenerate`` selects which half: the comparisons run
    on the non-degenerate cases, and the degenerate ones are checked for being
    left alone."""
    for name, block in fixture.get("adfd", {}).items():
        for case in block["cases"]:
            if _is_degenerate(block, case) is bool(degenerate):
                yield name, block, case


def _closed_form_distance(case) -> float:
    """Distance between the solver's gradient and the closed form's, symmetric
    and normalised by the larger of the two. Re-derived here from the two
    stored gradients rather than read out of the assembler's `pairwise` block.
    """
    a, b = case["ad"]["grad"], case["closed_form"]["grad"]
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _adfd_ladders(fixture):
    """Group the non-degenerate cases into (port kind, leg, objective) families
    that were measured at all three cell sizes, with the closed-form distance at
    each. A family measured at one rung has no sequence and is not returned."""
    fams: dict = {}
    for name, block, case in _adfd_cases(fixture):
        key = (block["kind"], block["leg"], case["objective"])
        fams.setdefault(key, {})[block["rung_um"]] = _closed_form_distance(case)
    return {k: v for k, v in fams.items() if set(v) == set(RUNGS_UM)}


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
    """Criterion 3a as the contract defines it: the derivative rfx returns is
    the derivative of the line rfx solves.

    The bar is 0.05 and it is asserted on every NON-degenerate leg at every
    rung. A degenerate objective is excluded because there is no derivative
    there to agree about — see
    ``test_the_degenerate_records_carry_no_comparison``, which holds those
    records to being left alone rather than to a number. PI decision of
    2026-09-21, in the pre-declaration's addendum.
    """
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
    a rate consistent with first order. A gradient that disagreed with the
    continuum for a reason other than discretisation would not do that — it
    would sit at a ratio near 1, which is what the window excludes.

    Both ratios are re-derived from the two stored gradients at each rung, never
    read out of the assembler's own `pairwise` block. On the measurement this
    ships with they are 0.557 and 0.530 on the resistive permittivity leg, and
    0.728 and 0.646 on the mid-band one.
    """
    fams = _adfd_ladders(fixture)
    if not fams:
        pytest.skip("no AD leg was measured at all three rungs")
    lo, hi = FIRST_ORDER_RATIO_WINDOW
    problems = []
    for (kind, leg, objective), dist in sorted(fams.items()):
        seq = [dist[um] for um in RUNGS_UM]          # 1000, 500, 250 um
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


def test_the_degenerate_records_carry_no_comparison(fixture):
    """Band-mean |S11|^2 on a SHORT is identically 1, so its derivative is zero
    and both AD and FD return round-off about it. Those records stay in the
    artifact as measured and are held to nothing.

    What is asserted is that they are left alone on purpose rather than by
    accident: the objective says so, the closed form's own derivative really is
    zero, and the record carries no verdict. The last clause is checked by
    asking the two gates' own selectors whether they picked the case up — so a
    leg cannot lose its marking and quietly slip back under a bar, and cannot
    keep its marking while a bar is applied anyway.
    """
    degenerate = list(_adfd_cases(fixture, degenerate=True))
    if not degenerate:
        pytest.skip("no degenerate objective was measured")
    gated = {(n, c["objective"]) for n, _b, c in _adfd_cases(fixture)}
    laddered = set()
    for (kind, leg, objective) in _adfd_ladders(fixture):
        laddered.add((kind, leg, objective))
    for name, block, case in degenerate:
        tag = f"{name}/{case['objective']}"
        assert block["objectives"][case["objective"]]["degenerate_on_this_dut"] is True, tag
        assert block["objectives"][case["objective"]].get("why"), (
            f"{tag}: marked degenerate with no reason recorded")
        # The closed form's derivative for a constant objective is zero, and the
        # record has to show that rather than assert it.
        assert abs(case["closed_form"]["grad"]) <= 1e-12, (
            f"{tag}: the closed form's derivative is "
            f"{case['closed_form']['grad']:.6e}, which is not zero — then the "
            "objective is not constant and this record should be compared")
        # Neither gate may have selected it.
        assert (name, case["objective"]) not in gated, (
            f"{tag}: a degenerate case reached the AD-vs-FD gate")
        assert (block["kind"], block["leg"], case["objective"]) not in laddered, (
            f"{tag}: a degenerate case reached the convergence gate")
        # And it carries no pass/fail of its own.
        verdicts = [k for k in _walk_keys(case)
                    if k in ("within_bar", "passed", "ok", "verdict")]
        assert not verdicts, f"{tag}: the record carries a verdict key {verdicts}"


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
    ab = fixture.get("port_kind_ab")
    assert ab is not None, "the fixture carries no port-kind A/B block"
    assert ab["present"] is True, (
        "the A/B record was not produced; run its producer with no arguments")
    assert ab["producer"].endswith("lumped_port_known_load_line.py")
    assert len(ab["commit"]) == 40, "the A/B record names no resolvable commit"
    zc = ab["channel"]["zc_ohm"]
    assert ab["channel"]["zref_ohm"] == pytest.approx(zc, rel=1e-12), (
        "the A/B only collapses to |Gamma_L| when Zref is the line's own Zc")
    assert set(ab["loads"]) == {"half_zc", "matched", "double_zc"}, sorted(ab["loads"])
    for name, row in ab["loads"].items():
        r = row["r_over_zc"]
        assert row["r_ohm"] == pytest.approx(r * zc, rel=1e-12), name
        assert row["closed_form_abs_s11"] == pytest.approx(
            abs((r - 1.0) / (r + 1.0)), rel=1e-12), name
        for kind in ("lumped", "wire"):
            side = row[kind]
            worst = max(abs(v - row["closed_form_abs_s11"]) for v in side["abs_s11"])
            assert side["max_abs_from_closed_form"] == pytest.approx(worst, rel=1e-9), (
                f"{name}/{kind}")
            assert len(side["abs_s11"]) == len(ab["freqs_hz"]), f"{name}/{kind}"


def test_the_not_in_chain_block_gates_nothing(fixture):
    """``run(compute_s_params=True)`` is not in the v2.0 chain for this family.

    The block is carried because the two paths disagree on this channel and a
    number nobody wrote down gets measured again. What is checked is that its
    derived statement follows from its own stored arrays, and that it is not
    quietly being used as a gate: it carries no bar and no pass/fail.
    """
    obs = fixture.get("not_in_chain_observations")
    if not obs:
        pytest.skip("no not-in-chain observations are assembled")
    for kind, block in obs.items():
        assert "not in the v2.0 chain" in block["scope"].lower(), kind
        blob = json.dumps(block).lower()
        for word in ("bar", "within_bar", "pass", "fail"):
            assert f'"{word}"' not in blob, (
                f"{kind}: the not-in-chain block carries a {word!r} key, which "
                "would make an observation into a gate")
        li = block["run_path_load_independence"]
        runs = {c["dut"]: _complex(c["run_s11"]) for c in block["cases"]}
        ref = next(iter(runs.values()))
        for dut, v in runs.items():
            assert li["identical_to_first_dut"][dut] == bool(np.array_equal(v, ref)), dut
        assert li["all_identical"] == all(li["identical_to_first_dut"].values())
        assert li["run_abs_max"] == pytest.approx(
            max(float(np.abs(v).max()) for v in runs.values()), rel=1e-12)


def test_the_degenerate_objective_evidence_follows_from_the_records(fixture):
    """The ULP-span floor is computed on the two LOSS values, so it passes on an
    objective whose derivative is zero and whose loss difference is round-off.

    The block states that with the numbers beside it. This checks the numbers
    are the ones in the records, and that every row it lists really is an
    objective the driver marked degenerate — a block that quietly listed a
    healthy leg would be an excuse rather than evidence.
    """
    ev = fixture.get("degenerate_objective_evidence")
    if ev is None or not ev["rows"]:
        pytest.skip("no degenerate objective was measured")
    for row in ev["rows"]:
        block = fixture["adfd"][row["leg"]]
        assert block["objectives"]["band_mean_s11_sq"]["degenerate_on_this_dut"] is True, (
            f"{row['leg']} is listed as degenerate but its own record does not "
            "say so")
        case = [c for c in block["cases"] if c["objective"] == "band_mean_s11_sq"][0]
        assert row["ulp_span"] == pytest.approx(case["fd"]["ulp_span"], rel=1e-12)
        assert row["span_above_floor"] == bool(
            case["fd"]["ulp_span"] >= block["min_fd_ulp_span"])
        assert row["closed_form_grad"] == pytest.approx(
            case["closed_form"]["grad"], rel=1e-12, abs=1e-30)


def test_the_fitted_electrical_length_follows_from_its_own_short(fixture):
    """Every AD leg fits the line's own electrical length from a short it solved
    at that rung. This re-derives the fit from the stored short and checks the
    stated length is the one that fit gives."""
    legs = [(n, b) for n, b in fixture.get("adfd", {}).items()
            if b.get("fitted_electrical_length")]
    if not legs:
        pytest.skip("no AD leg carries a fitted electrical length")
    for name, block in legs:
        fit = block["fitted_electrical_length"]
        s11 = _complex(fit["s11_short"])
        freqs = np.asarray(fixture["freqs_hz"], dtype=float)
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
