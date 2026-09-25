"""Replay of the coaxial chain battery. Reads the fixture, re-derives every
assembled number from the stored S, and compares the result against the bar.

No FDTD runs here. The measurement is
``scripts/diagnostics/coax_chain_battery_measure.py``, pre-declared in
``docs/design_notes/coax_chain_battery_predeclaration.md``; the bar is the v2.0
one in ``docs/design_notes/chain_closure_contract.md`` — magnitude within 2 dB,
frequencies within 1 %, column power at most 1.02, reciprocity at most 0.02, AD
against FD within 0.05, forward identity at rtol 1e-5 / atol 1e-7.

Every derived quantity is recomputed HERE from the stored complex S by an
implementation written independently of the assembler's — the coaxial ``Z_TEM``,
the TEM section's ``S11``/``S21``, the parabola through the reflection zero, the
column power — so a fixture whose summary numbers were edited by hand, or an
assembler whose arithmetic drifts, reds instead of replaying.
``test_perturbing_the_stored_s_breaks_its_own_summary`` runs that as a mutation:
it revives the defect (a wrong S behind a right-looking summary) with the checks
left exactly as they ship.

The PI's 2026-09-21 deep-null ruling is encoded rather than described: a
quantity near zero by construction is not compared in dB. The bead's |S11| is
compared to the referee only where the ANALYTIC |S11| is above the null level;
inside a zero's core the verdict is the zero's FREQUENCY. On the dx ladder a
quantity that is deep on the finest rung or in the closed form at every bin
(the thru's S11 and S22) is read on the -20 dB bound, never in dB, and a
quantity deep only in a zero's core is compared in dB outside it (the leader's
L2, 2026-09-23).

P2 (PI 2026-09-23): criterion 1(2) is the identity within the traced path
(the bead in two containers, rtol 1e-5 / atol 1e-7); the untraced-against-traced
thru arm compares two functions by design and is held to its measured envelope.
Of 2026-09-24 (PI, R1): settling is judged by the contract's amplitude
substitute.

Of 2026-09-25 (the pre-declaration's addendum, issue 1218): both coax lanes
absorb on all three axes, so the open termination is judged again like the
short and the loads — magnitude within the bar, passivity 1.02, record-length
invariance — and nothing is left unjudged, which ``test_every_dut_is_judged``
holds. The open's exclusion of 2026-09-23 (P1) and the loads' closed-can
footprint note of 2026-09-24 lapsed with the fix, and a fixture that still
carries either reds here. Passivity and the thru's -20 dB bound are checked at
every rung, the ladder's zeros are re-derived from each rung's S, and the
recommended cell size is pinned at 4 annulus cells for every DUT.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests import _electrical_length as EL

FIXTURE = (Path(__file__).resolve().parents[1] / "fixtures" / "coax_chain_battery"
           / "fixture.json")

C0 = 299792458.0
MU0 = 4.0e-7 * math.pi
EPS0 = 1.0 / (MU0 * C0 * C0)
DB_FLOOR = 1e-300
CLAIMS_RUNG = 9
TWO_PORT_DUTS = ("bead", "thru")
ONE_PORT_DUTS = ("short", "open", "r25", "r100")
DUTS = TWO_PORT_DUTS + ONE_PORT_DUTS
RUNGS = (4, 6, 9)
ALL_SOLVES = [f"{dut}_rung{r}" for dut in DUTS for r in RUNGS]


# ---------------------------------------------------------------------------
# independent arithmetic — deliberately not imported from the assembler or rfx
# ---------------------------------------------------------------------------

def _complex(block) -> np.ndarray:
    return (np.asarray(block["real"], dtype=float)
            + 1j * np.asarray(block["imag"], dtype=float))


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), DB_FLOOR))


def _z_tem(a: float, b: float, eps_r: float) -> float:
    """``sqrt(L'/C')`` for a coaxial TEM line, written from the per-metre
    constants rather than imported, so a change to the library function shows up
    here as a disagreement instead of moving both sides together."""
    return math.log(b / a) / (2.0 * math.pi) * math.sqrt(MU0 / (EPS0 * eps_r))


def _tem_section(freqs, *, a, b, eps_fill, eps_scale, length,
                 d1=None, d2=None):
    """``S`` of a lossless TEM section between two matched lines, at the section
    faces and, when the two line lengths are given, at the feed planes."""
    freqs = np.asarray(freqs, dtype=float)
    z1 = _z_tem(a, b, eps_fill)
    z2 = _z_tem(a, b, eps_fill * eps_scale)
    gam = (z2 - z1) / (z2 + z1)
    beta1 = 2.0 * np.pi * freqs * math.sqrt(eps_fill) / C0
    beta2 = 2.0 * np.pi * freqs * math.sqrt(eps_fill * eps_scale) / C0
    e1 = np.exp(-1j * beta2 * length)
    den = 1.0 - gam ** 2 * e1 ** 2
    s11 = gam * (1.0 - e1 ** 2) / den
    s21 = (1.0 - gam ** 2) * e1 / den
    if d1 is None or d2 is None:
        return gam, s11, s21
    return (gam,
            s11 * np.exp(-2j * beta1 * d1),
            s21 * np.exp(-1j * beta1 * (d1 + d2)))


def _vertex(freqs, y, k):
    """Parabola vertex through bins k-1, k, k+1, from the three-point Lagrange
    form rather than the assembler's rearrangement of it."""
    x0, x1, x2 = float(freqs[k - 1]), float(freqs[k]), float(freqs[k + 1])
    y0, y1, y2 = float(y[k - 1]), float(y[k]), float(y[k + 1])
    d0 = (y2 - y1) / (x2 - x1)
    d1 = (y1 - y0) / (x1 - x0)
    a = (d0 - d1) / (x2 - x0)
    b = d1 - a * (x1 + x0)
    return -b / (2.0 * a)


def _zero_hz(freqs, s11, k):
    """The battery's estimator for the reflection zero: the parabola vertex on
    ``|S11|^2``, which is the quantity a simple zero makes quadratic."""
    return _vertex(freqs, np.abs(s11) ** 2, k)


def _column_power(S):
    n_ports, _, n_f = S.shape
    out = np.zeros((n_ports, n_f))
    for j in range(n_ports):
        for k in range(n_f):
            out[j, k] = sum(abs(S[i, j, k]) ** 2 for i in range(n_ports))
    return out


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


def _radii(fixture):
    line = fixture["line"]
    return line["pin_radius_m"], line["outer_radius_m"], line["fill_eps_r"]


# The rulings this fixture is judged under, restated rather than read from the
# fixture, so a fixture judged under the lapsed ones reds instead of passing.
RULINGS_IN_FORCE = {
    "deep_null_pi_2026_09_21", "identity_within_the_traced_path_pi_2026_09_23",
    "bead_whole_cells_leader_2026_09_23", "deep_quantity_by_the_bound_leader_2026_09_23",
    "settling_by_the_amplitude_substitute_pi_2026_09_24",
    "column_power_half_not_applied_leader_2026_09_24",
    "open_judged_all_axis_absorption_leader_2026_09_25",
}
RULINGS_LAPSED = {"open_not_judged_pi_2026_09_23", "loads_closed_can_footprint_pi_2026_09_24"}


# ---------------------------------------------------------------------------
# provenance and geometry
# ---------------------------------------------------------------------------

def test_the_fixture_names_its_own_provenance(fixture):
    assert fixture["schema"] == "rfx.coax_chain_battery"
    assert fixture["predeclaration"].endswith("coax_chain_battery_predeclaration.md")
    assert len(fixture["assembler_commit"]) == 40
    for key, entry in fixture["solves"].items():
        p = entry["provenance"]
        assert len(p["commit"]) == 40, f"{key} carries no resolvable commit"
        # The fixture ships no machine paths; it ships the question they
        # answered — did `import rfx` resolve to the run's own tree?
        assert p["rfx_import_tail"] == "rfx/__init__.py", key
        assert p["rfx_resolved_inside_the_run_tree"] is True, key
        assert p["compute_run_id"], f"{key} does not name the run that produced it"
        assert p["jax_version"] and p["numpy_version"], key
        assert entry["preflight_text"] is not None, key
        assert entry["wall_s"] > 0.0, key


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_the_realized_line_is_the_declared_line(fixture, key):
    """The measurement asserted this before solving; the fixture has to carry
    the evidence, because a number whose line was never checked is not one of
    this battery's numbers."""
    entry = _solve(fixture, key)
    dec, real = entry["declared"], entry["realized"]
    rung = entry["rung_annulus_cells"]
    assert real["annulus_cells"] == pytest.approx(rung, abs=1e-9), key
    assert real["dx_m"] == pytest.approx(dec["dx_m"], rel=1e-12), key
    assert real["fill_eps_r_realized"] == pytest.approx(dec["fill_eps_r"], rel=1e-6), key
    assert real["pin_cells_cross_section"] >= 1, key
    assert real["shell_cells_cross_section"] >= 1, key
    # The conductors are shorted E edges, not a conductivity (PR #1169). Three
    # consequences the measurement checked before solving and the fixture has
    # to carry: the wall's inner face is the DECLARED outer radius at every
    # cell size, so the dielectric annulus no longer shrinks with the mesh;
    # nothing is left carrying a conductivity where the conductor is; and the
    # mask the stamper returned is the cross-section this driver replicated.
    assert real["conductor_realization"] == "pec_edge_masks", key
    assert real["shell_inner_radius_m"] == pytest.approx(dec["outer_radius_m"],
                                                         rel=1e-12), key
    assert real["shell_outer_radius_m"] > real["shell_inner_radius_m"], key
    assert real["wall"]["thickness_cells"] >= 1.0, key
    assert real["pec_mask_vs_replicated_mismatch_cells"] == 0, key
    assert real["n_sigma_cells_at_probe_plane"] == 0, key
    assert real["realized_fill_radius_max_m"] <= real["outer_radius_m"], key
    # The layout this driver replicated against the layout the extractor used.
    cc = entry["cross_check"]
    assert cc["annulus_cells_agree"] is True, key
    if entry["lane"] == "two_port":
        assert cc["reference_planes_agree"] is True, key
        assert cc["reference_planes_max_abs_diff_m"] == 0.0, key
    if entry["dut"] == "bead":
        assert real["bead_inside_probe_gap"] is True, key
        assert abs(real["bead_length_realized_m"] - dec["bead_length_m"]) <= real["dx_m"], key
        # L1 (leader, 2026-09-23): the bead is a whole number of annulus widths,
        # so it is a whole number of cells at every rung and every rung builds
        # the same bead. The 6 mm of the first run rasterized to three lengths.
        widths = dec["bead_length_annulus_widths"]
        assert dec["bead_length_m"] == pytest.approx(widths * dec["annulus_m"], rel=1e-12), key
        assert real["bead_n_cells"] == widths * rung, key
        assert real["bead_length_realized_m"] == pytest.approx(dec["bead_length_m"],
                                                               rel=1e-9), key


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_every_stored_summary_follows_from_the_stored_s(fixture, key):
    entry = _solve(fixture, key)
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    if entry["lane"] == "two_port":
        S = _complex(entry["S"])
        assert S.shape == (2, 2, len(freqs))
        np.testing.assert_allclose(_db(S[0, 0, :]), entry["s11_db"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(_db(S[1, 0, :]), entry["s21_db"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(_db(S[1, 1, :]), entry["s22_db"], rtol=0, atol=1e-9)

        col = _column_power(S)
        np.testing.assert_allclose(col, entry["power"]["column_power"],
                                   rtol=1e-12, atol=1e-15)
        assert entry["power"]["max_column_power"] == pytest.approx(col.max(), rel=1e-12)
        np.testing.assert_allclose(1.0 - col, entry["power"]["power_closure"],
                                   rtol=1e-12, atol=1e-15)
        recip = np.abs(S[1, 0, :] - S[0, 1, :])
        np.testing.assert_allclose(recip, entry["power"]["reciprocity_abs"],
                                   rtol=1e-12, atol=1e-18)
        assert entry["power"]["reciprocity_metric"] == pytest.approx(
            recip.max() / np.abs(S).max(), rel=1e-12)
        if entry["dut"] != "bead":
            assert entry["thru_reflection_floor_db"] == pytest.approx(
                float(_db(S[0, 0, :]).max()), abs=1e-9)
            return
        k = int(np.argmin(np.abs(S[0, 0, :])))
        assert entry["reflection_zero_measured"]["bin_index"] == k
        if 0 < k < len(freqs) - 1:
            assert entry["reflection_zero_measured"]["interp_hz"] == pytest.approx(
                _zero_hz(freqs, S[0, 0, :], k), rel=1e-6)
    else:
        g = _complex(entry["S11"])
        assert g.shape == (len(freqs),)
        np.testing.assert_allclose(np.abs(g), entry["abs_s11"], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(_db(g), entry["s11_db"], rtol=0, atol=1e-9)
        assert entry["max_abs_s11"] == pytest.approx(float(np.abs(g).max()), rel=1e-12)


# ---------------------------------------------------------------------------
# criterion 2 — physics gates at the claims rung
# ---------------------------------------------------------------------------

CLAIMS_TWO_PORT = tuple(f"{d}_rung{CLAIMS_RUNG}" for d in TWO_PORT_DUTS)
CLAIMS_ONE_PORT = tuple(f"{d}_rung{CLAIMS_RUNG}" for d in ONE_PORT_DUTS)
# Passivity is a property of every record, not of the claims rung: a coarser
# rung that returns more power than it receives is not a rung to recommend, and
# the recommended cell size (4 annulus cells) is the coarsest of them.
ALL_TWO_PORT = tuple(f"{d}_rung{r}" for d in TWO_PORT_DUTS for r in RUNGS)
ALL_ONE_PORT = tuple(f"{d}_rung{r}" for d in ONE_PORT_DUTS for r in RUNGS)


@pytest.mark.parametrize("key", ALL_TWO_PORT)
def test_raw_column_power_stays_inside_the_passivity_bar(fixture, key):
    entry = _solve(fixture, key)
    measured = _column_power(_complex(entry["S"])).max()
    assert measured <= fixture["bar"]["column_power_max"], (
        f"{key}: max column power {measured:.6f} exceeds "
        f"{fixture['bar']['column_power_max']} on a passive line")


@pytest.mark.parametrize("key", CLAIMS_TWO_PORT)
def test_reciprocity_stays_inside_the_bar(fixture, key):
    entry = _solve(fixture, key)
    S = _complex(entry["S"])
    measured = np.abs(S[1, 0, :] - S[0, 1, :]).max() / np.abs(S).max()
    assert measured <= fixture["bar"]["reciprocity"], (
        f"{key}: max_f |S21 - S12| / max|S| = {measured:.6f}")


@pytest.mark.parametrize("key", ALL_ONE_PORT)
def test_the_one_port_reflection_is_passive(fixture, key):
    """A one-port's column power IS ``|Gamma|^2``, so the contract's 1.02 applies
    to the square and not to the magnitude. An excess is non-physical and is an
    extraction or normalization finding, never physics."""
    entry = _solve(fixture, key)
    g = np.abs(_complex(entry["S11"]))
    measured = float((g ** 2).max())
    k = int(np.argmax(g))
    assert measured <= fixture["bar"]["column_power_max"], (
        f"{key}: column power |Gamma|^2 = {measured:.6f} (|Gamma| = {g[k]:.6f} at "
        f"{np.asarray(entry['freqs_hz'])[k]/1e9:.3f} GHz) exceeds "
        f"{fixture['bar']['column_power_max']} on a passive termination")


@pytest.mark.parametrize("dut", DUTS)
def test_the_record_is_settled_or_record_length_invariant(fixture, dut):
    """Criterion 2's settling witness, or the one substitute the contract admits.

    ``compute_coaxial_two_port`` skips its ring-down witness whenever
    ``eps_scale`` is given, and the one-port result carries none at all, so most
    of this battery has no energy witness. Where it is missing the doubled-record
    arm has to be there AND has to be invariant: a record that still moves when
    the window doubles was truncated, and its S is not interpretable.
    """
    bound = (10 ** (fixture["bar"]["magnitude_db"] / 20.0) - 1.0) / 10.0
    key = f"{dut}_rung{CLAIMS_RUNG}"
    entry = _solve(fixture, key)
    witness = entry.get("settling")
    if witness and witness["has_energy_witness"]:
        worst = max(witness["settling_db"])
        assert worst <= fixture["bar"]["settling_db"], (
            f"{key}: settling {witness['settling_db']} dB is above the "
            f"{fixture['bar']['settling_db']} dB rule — the record was truncated "
            "before the line rang down")
        return
    inv = fixture["record_length_invariance"].get(dut)
    assert inv is not None, (
        f"{key} carries no energy witness ({witness['why']}) and no doubled-record "
        "arm either, so nothing says its record was long enough")
    assert inv["record_ratio"] >= 1.9, inv["record_ratio"]
    assert inv["max_abs_shift"] == pytest.approx(inv["max_abs_shift"]), dut
    assert inv["max_abs_shift"] <= bound, (
        f"{dut}: doubling the record moved max|S| by {inv['max_abs_shift']:.5g}, "
        f"above a tenth of the magnitude bar ({bound:.5g}) — the record is "
        "truncated, not settled")


def _every_place(fixture):
    places = [(f"solves[{k}]", e["dut"], e) for k, e in fixture["solves"].items()]
    places += [(f"record_length_invariance[{d}]", d, e)
               for d, e in fixture["record_length_invariance"].items()]
    places += [(f"ladder[{d}]", d, e) for d, e in fixture["ladder"].items()]
    return places


def test_every_dut_is_judged(fixture):
    """The addendum of 2026-09-25: with both lanes absorbing on all three axes the
    open is judged like every other DUT. Every record, doubled arm and ladder
    carries ``judged: true`` and no reason for not being judged, and the open is
    among them at every rung, in the doubled arm and in the ladder."""
    places = _every_place(fixture)
    assert places
    for where, dut, entry in places:
        assert entry.get("judged") is True, f"{where} is not marked judged"
        assert "judged_reason" not in entry, where
    open_places = {where for where, dut, _ in places if dut == "open"}
    want = ({f"solves[open_rung{r}]" for r in RUNGS}
            | {"record_length_invariance[open]", "ladder[open]"})
    assert want <= open_places, f"the open is missing from {sorted(want - open_places)}"


def test_no_record_carries_the_lapsed_footprint_note(fixture):
    """The loads' closed-can footprint note (R1, 2026-09-24) described the closed
    can; it lapsed with the fix, and no record may carry it."""
    for where, _dut, entry in _every_place(fixture):
        assert "footprint_note" not in entry, where


def test_the_fixture_is_judged_under_the_rulings_in_force(fixture):
    rulings = set(fixture["rulings"])
    assert rulings == RULINGS_IN_FORCE, (
        f"missing {sorted(RULINGS_IN_FORCE - rulings)}, unexpected "
        f"{sorted(rulings - RULINGS_IN_FORCE)}")
    assert not rulings & RULINGS_LAPSED, sorted(rulings & RULINGS_LAPSED)


def test_every_doubling_and_power_span_fact_follows_from_the_stored_s(fixture):
    """The facts R1 and R2 add beside the verdicts: the column-power span of
    every record, and what doubling the record did per bin in dB."""
    def span(S):
        col = _column_power(S) if S.ndim == 3 else np.abs(S) ** 2
        return [float(col.min()), float(col.max())]

    for key, entry in fixture["solves"].items():
        S = _complex(entry["S"] if entry["lane"] == "two_port" else entry["S11"])
        np.testing.assert_allclose(entry["column_power_span"], span(S), rtol=1e-12,
                                   err_msg=key)
    for dut, inv in fixture["record_length_invariance"].items():
        base = fixture["solves"][f"{dut}_rung{CLAIMS_RUNG}"]
        a = _complex(base["S"] if inv["lane"] == "two_port" else base["S11"])
        b = _complex(inv["S_doubled"])
        np.testing.assert_allclose(inv["column_power_span"], [span(a), span(b)],
                                   rtol=1e-12, err_msg=dut)
        freqs = np.asarray(inv["freqs_hz"], dtype=float)
        entries = ({"s11": (0, 0), "s21": (1, 0), "s12": (0, 1), "s22": (1, 1)}
                   if inv["lane"] == "two_port" else {"s11": None})
        assert set(inv["doubling_db_change"]) == set(entries), dut
        level = fixture["bar"]["deep_null_db"]
        for name, ij in entries.items():
            x = a[ij[0], ij[1], :] if ij else a
            y = b[ij[0], ij[1], :] if ij else b
            ch = np.abs(_db(x) - _db(y))
            # a bin where either record is deep is left out (PI 2026-09-21)
            live = (_db(x) > level) & (_db(y) > level)
            rec = inv["doubling_db_change"][name]
            assert [v is None for v in rec["db_change"]] == (~live).tolist(), (dut, name)
            np.testing.assert_allclose(
                [v for v in rec["db_change"] if v is not None], ch[live],
                rtol=1e-9, atol=1e-12, err_msg=f"{dut} {name}")
            assert rec["n_deep_bins_left_out"] == int((~live).sum()), (dut, name)
            if live.any():
                assert rec["max_db_change"] == pytest.approx(float(ch[live].max()), rel=1e-9)
            else:
                assert rec["max_db_change"] is None, (dut, name)
            assert rec["bins_above_0p1_db_hz"] == freqs[live & (ch > 0.1)].tolist(), (dut, name)


def test_the_doubled_record_arm_is_derived_from_its_own_stored_s(fixture):
    """The invariance number has to follow from the two stored matrices, or it
    is a summary nobody can check."""
    for dut, inv in fixture["record_length_invariance"].items():
        base = fixture["solves"][f"{dut}_rung{CLAIMS_RUNG}"]
        a = _complex(base["S"] if inv["lane"] == "two_port" else base["S11"])
        b = _complex(inv["S_doubled"])
        assert inv["max_abs_shift"] == pytest.approx(
            float(np.abs(np.abs(a) - np.abs(b)).max()), rel=1e-9), dut


# ---------------------------------------------------------------------------
# criterion 3(d) — the analytic referee
# ---------------------------------------------------------------------------

def test_the_bead_reflection_zero_matches_the_analytic_tem_section(fixture):
    """Inside a reflection zero's core the verdict is the zero's FREQUENCY, not
    its depth (PI ruling, 2026-09-21)."""
    entry = _solve(fixture, f"bead_rung{CLAIMS_RUNG}")
    a, b, eps_fill = _radii(fixture)
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    S = _complex(entry["S"])
    k = int(np.argmin(np.abs(S[0, 0, :])))
    f_meas = _zero_hz(freqs, S[0, 0, :], k)
    length = entry["realized"]["bead_length_realized_m"]
    eps_scale = fixture["line"]["bead_eps_scale"]
    f_an = C0 / (2.0 * length * math.sqrt(eps_fill * eps_scale))
    assert freqs.min() <= f_an <= freqs.max(), (
        f"the analytic reflection zero {f_an/1e9:.4f} GHz is outside the measured "
        "band — the fixture cannot judge it")
    frac = abs(f_meas - f_an) / f_an
    assert frac <= fixture["bar"]["frequency_frac"], (
        f"reflection zero {f_meas/1e9:.5f} GHz against the analytic {f_an/1e9:.5f} GHz "
        f"is {frac*100:.3f} % — the bar is "
        f"{fixture['bar']['frequency_frac']*100:.1f} %")


def test_the_bead_magnitudes_match_the_analytic_tem_section(fixture):
    """|S11| outside the zero's core and |S21| over the whole band, within the
    2 dB magnitude bar, against the TEM section the bead is."""
    entry = _solve(fixture, f"bead_rung{CLAIMS_RUNG}")
    a, b, eps_fill = _radii(fixture)
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    S = _complex(entry["S"])
    real = entry["realized"]
    gam, an11, an21 = _tem_section(
        freqs, a=a, b=b, eps_fill=eps_fill,
        eps_scale=fixture["line"]["bead_eps_scale"],
        length=real["bead_length_realized_m"],
        d1=real["d_port1_to_bead_m"], d2=real["d_port2_to_bead_m"])
    assert gam == pytest.approx(-1.0 / 3.0, rel=1e-9), (
        "the bead's permittivity ratio of 4 makes the impedance step exactly "
        f"-1/3; the fixture's line gives {gam}")
    core = _db(an11) <= fixture["bar"]["deep_null_db"]
    d11 = np.abs(_db(S[0, 0, :]) - _db(an11))
    d21 = np.abs(_db(S[1, 0, :]) - _db(an21))
    outside = ~core
    assert outside.any(), "every bin is inside a reflection-zero core"
    worst11 = float(d11[outside].max())
    worst21 = float(d21.max())
    assert worst11 <= fixture["bar"]["magnitude_db"], (
        f"|S11| is {worst11:.3f} dB from the analytic TEM section outside the "
        f"zero's core (bar {fixture['bar']['magnitude_db']} dB)")
    assert worst21 <= fixture["bar"]["magnitude_db"], (
        f"|S21| is {worst21:.3f} dB from the analytic TEM section "
        f"(bar {fixture['bar']['magnitude_db']} dB)")


@pytest.mark.parametrize("rung", RUNGS)
@pytest.mark.parametrize("dut", TWO_PORT_DUTS)
def test_each_line_is_as_long_electrically_as_its_closed_form(fixture, dut, rung):
    """The v2 bar's phase item (PI 2026-09-24): the least-squares slope of the
    unwrapped phase of S21 against frequency, over the bins where |S21| is above
    -20 dB, within 1 % of the closed form's slope over the same bins, at every
    rung.

    S21 is referenced to the two feed planes (``compute_coaxial_two_port``,
    ``rfx/sparams/coax.py:886-887`` and ``:1036``), so the closed form spans
    exactly their separation: the bare PTFE line ``exp(-j omega sqrt(eps) L / c)``
    for the thru, and for the bead the same line with the four-annulus-width
    section of 4x permittivity where the record says it sits. ``eps`` is the
    fill the grid realized. The thru has no frequency feature, so nothing else
    in this file sees it grow longer: with every edge permittivity above vacuum
    read 5 % high the solved thru is 2.48 % longer electrically and its |S21|
    moves 0.01 dB.
    """
    key = f"{dut}_rung{rung}"
    entry = _solve(fixture, key)
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    S = _complex(entry["S"])
    real = entry["realized"]
    top, bottom = (float(z) for z in entry["reference_planes_m"])
    L = top - bottom
    assert L == pytest.approx(real["feed_to_feed_m"], rel=1e-12), (
        f"{key}: the reference planes are {L} m apart, the feeds "
        f"{real['feed_to_feed_m']} m — S is not where the record says it is")
    eps = float(real["fill_eps_r_realized"])
    if dut == "thru":
        reference = np.exp(-1j * 2.0 * np.pi * freqs * math.sqrt(eps) * L / C0)
    else:
        a, b, _ = _radii(fixture)
        d1, d2 = real["d_port1_to_bead_m"], real["d_port2_to_bead_m"]
        length = real["bead_length_realized_m"]
        assert d1 + length + d2 == pytest.approx(L, rel=1e-12), key
        _, _, reference = _tem_section(
            freqs, a=a, b=b, eps_fill=eps, eps_scale=fixture["line"]["bead_eps_scale"],
            length=length, d1=d1, d2=d2)
    ratio = EL.electrical_length_ratio(freqs, S[1, 0, :], reference,
                                       EL.transmitting_bins(S[1, 0, :]))
    assert abs(ratio) <= EL.ELECTRICAL_LENGTH_FRAC, (
        f"{key}: S21's phase slope is {ratio * 100:+.3f} % from the closed form's over "
        f"the {L * 1e3:.3f} mm between the feed planes — the line is that much longer "
        f"electrically than it is drawn (bar {EL.ELECTRICAL_LENGTH_FRAC * 100:.0f} %)")


@pytest.mark.parametrize("rung", RUNGS)
def test_the_thru_reflection_stays_below_the_deep_null_bound(fixture, rung):
    """The control. Its |S11| and |S22| are near zero by construction, so they
    are held to an upper bound at every bin rather than compared in dB — the
    line's own reflection floor — at every rung, the recommended one included."""
    key = f"thru_rung{rung}"
    S = _complex(_solve(fixture, key)["S"])
    for name, (i, j) in (("S11", (0, 0)), ("S22", (1, 1))):
        floor = float(_db(S[i, j, :]).max())
        assert floor <= fixture["bar"]["deep_null_db"], (
            f"{key}: the thru's {name} is {floor:.2f} dB at its worst bin, above the "
            f"{fixture['bar']['deep_null_db']} dB bound — the matched feeds are not "
            "matched on this line")


@pytest.mark.parametrize("dut", ONE_PORT_DUTS)
def test_the_one_port_loads_match_their_analytic_reflection(fixture, dut):
    entry = _solve(fixture, f"{dut}_rung{CLAIMS_RUNG}")
    a, b, eps_fill = _radii(fixture)
    z0 = _z_tem(a, b, eps_fill)
    if dut == "short":
        want = 1.0
    elif dut == "open":
        want = 1.0
    else:
        r = entry["declared"]["dut_impedance_ohm"]
        want = abs((r - z0) / (r + z0))
    measured = np.abs(_complex(entry["S11"]))
    worst = float(np.max(np.abs(_db(measured) - _db(want))))
    assert worst <= fixture["bar"]["magnitude_db"], (
        f"{dut}: |Gamma| is {worst:.3f} dB from the analytic {want:.5f} at its worst "
        f"bin (bar {fixture['bar']['magnitude_db']} dB); measured "
        f"[{measured.min():.4f}, {measured.max():.4f}]")


# ---------------------------------------------------------------------------
# criterion 3(c) — the ladder and the recommended cell size
# ---------------------------------------------------------------------------

def _rung_zero_hz(entry):
    """A rung's reflection zero from its own stored S, with this test's
    estimator (the parabola vertex on |S11|^2 around the deepest bin)."""
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    s11 = _complex(entry["S"])[0, 0, :]
    k = int(np.argmin(np.abs(s11)))
    assert 0 < k < len(freqs) - 1, "the reflection zero sits on a band edge"
    return _zero_hz(freqs, s11, k)


def test_the_ladder_converges_and_sets_a_recommended_cell_size(fixture):
    lad = fixture["ladder"].get("bead")
    if lad is None:
        pytest.skip("the bead ladder is not assembled")
    assert lad["rungs_annulus_cells"] == list(RUNGS), lad["rungs_annulus_cells"]
    # The frequency verdict decides the recommended cell size, so the zeros are
    # re-derived from each rung's stored S, not read from the ladder block.
    rungs = [row["rung"] for row in lad["rung_within_bar_vs_finest"]]
    mine = [_rung_zero_hz(fixture["solves"][k]) for k in rungs]
    # rtol 1e-6, as the per-solve check uses: the assembler's vertex assumes a
    # uniform bin step and the stored bins are float32-rounded (6e-8 apart).
    np.testing.assert_allclose(lad["reflection_zero_interp_hz"], mine, rtol=1e-6, atol=0.0)
    fine = mine[-1]
    for row, f in zip(lad["rung_within_bar_vs_finest"], mine):
        within = bool(abs(f - fine) / fine <= fixture["bar"]["frequency_frac"])
        assert row["zero_within_1pct"] is within, (row["rung"], f, fine)
    fs = lad["reflection_zero_interp_hz"]
    diffs = [abs(fs[i + 1] - fs[i]) for i in range(len(fs) - 1)]
    np.testing.assert_allclose(diffs, lad["successive_diff_hz"], rtol=1e-9, atol=1.0)
    ratio = lad["successive_diff_ratio"]
    assert ratio is not None and ratio < 1.0, (
        f"successive reflection-zero differences {diffs} do not shrink (ratio "
        f"{ratio}); on a ladder that is not converging no cell size is recommended")
    rows = lad["rung_within_bar_vs_finest"]
    for row in rows:
        flags = [v for k, v in row.items()
                 if isinstance(v, bool) and k != "all_inside_bar"]
        assert row["all_inside_bar"] == all(flags), row
    qualifying = [r["rung"] for r in rows if r["all_inside_bar"]]
    assert lad["coarsest_rung_within_bar"] == (qualifying[0] if qualifying else None)


# The cell size this battery hands a user: the coarsest rung inside the bar for
# every judged DUT, measured 2026-09-23 at 4 annulus cells. A change that moves
# any judged DUT's recommendation reds here instead of passing on "not None".
RECOMMENDED_RUNG = 4


@pytest.mark.parametrize("dut", DUTS)
def test_the_recommended_cell_size_is_four_annulus_cells(fixture, dut):
    lad = fixture["ladder"].get(dut)
    assert lad is not None, f"the {dut} ladder is not assembled"
    assert lad["coarsest_rung_within_bar"] == f"{dut}_rung{RECOMMENDED_RUNG}", (
        f"{dut}: the coarsest rung inside the bar is {lad['coarsest_rung_within_bar']}, "
        f"not {RECOMMENDED_RUNG} annulus cells")


def _ladder_curve(entry, name):
    if entry["lane"] == "two_port":
        i, j = {"s11": (0, 0), "s21": (1, 0), "s22": (1, 1)}[name]
        return _complex(entry["S"])[i, j, :]
    return _complex(entry["S11"])


def _deep_bins(fixture, dut, name, fine_entry):
    """Written from the closed forms, not read from the assembler: the thru's S11
    and S22 are zero, the bead's S11 and S22 are the TEM section's |S11| (the
    section is symmetric) at the finest rung's realized length, nothing else has
    a deep bin in its closed form; the finest rung's own curve adds its deep bins."""
    level = fixture["bar"]["deep_null_db"]
    deep = _db(_ladder_curve(fine_entry, name)) <= level
    if dut == "thru" and name in ("s11", "s22"):
        deep = np.ones_like(deep, dtype=bool)
    elif dut == "bead" and name in ("s11", "s22"):
        a, b, eps_fill = _radii(fixture)
        _, an11, _ = _tem_section(
            np.asarray(fine_entry["freqs_hz"], dtype=float), a=a, b=b,
            eps_fill=eps_fill, eps_scale=fixture["line"]["bead_eps_scale"],
            length=fine_entry["realized"]["bead_length_realized_m"])
        deep = deep | (_db(an11) <= level)
    return deep


def test_every_ladder_row_follows_from_the_stored_curves(fixture):
    """Every ladder row, and every rung's verdict, from the stored curves. A
    quantity deep at every bin (on the finest rung or in the closed form) is read
    on the -20 dB bound; one deep only in a zero's core is compared in dB outside
    it (PI 2026-09-21; the leader's L2, 2026-09-23)."""
    level = fixture["bar"]["deep_null_db"]
    for dut, lad in fixture["ladder"].items():
        names = ("s11", "s21", "s22") if lad["lane"] == "two_port" else ("s11",)
        have = [row["rung"] for row in lad["s11_vs_finest"]]
        fine_entry = fixture["solves"][have[-1]]
        verdict = {row["rung"]: {} for row in lad["rung_within_bar_vs_finest"]}
        for name in names:
            fine = _ladder_curve(fine_entry, name)
            deep = _deep_bins(fixture, dut, name, fine_entry)
            assert lad["deep_bins"][name] == deep.tolist(), (dut, name)
            mode = "bound" if deep.all() else "db_outside_deep"
            for row in lad[f"{name}_vs_finest"]:
                cur = _ladder_curve(fixture["solves"][row["rung"]], name)
                d = np.abs(_db(cur) - _db(fine))
                where = (dut, name, row["rung"])
                assert row["max_db_diff_vs_finest"] == pytest.approx(
                    float(d.max()), rel=1e-9), where
                assert row["judged_by"] == mode, where
                assert row["n_bins_deep"] == int(deep.sum()), where
                if (~deep).any():
                    assert row["max_db_diff_vs_finest_outside_deep"] == pytest.approx(
                        float(d[~deep].max()), rel=1e-9), where
                else:
                    assert row["max_db_diff_vs_finest_outside_deep"] is None, where
                assert row["max_db_on_rung"] == pytest.approx(
                    float(_db(cur).max()), rel=1e-9), where
                if mode == "bound":
                    ok = bool(_db(cur).max() <= level)
                    assert row["within_deep_null_bound"] is ok, where
                    verdict[row["rung"]][f"{name}_within_{level:g}dB_bound"] = ok
                else:
                    assert row["within_deep_null_bound"] is None, where
                    worst = row["max_db_diff_vs_finest_outside_deep"]
                    verdict[row["rung"]][
                        f"{name}_within_{fixture['bar']['magnitude_db']:g}dB"] = (
                        None if worst is None
                        else bool(worst <= fixture["bar"]["magnitude_db"]))
        for row in lad["rung_within_bar_vs_finest"]:
            for key, want in verdict[row["rung"]].items():
                assert row[key] == want, (dut, row["rung"], key)
            # a deep quantity never carries a dB verdict, and the reverse
            for name in names:
                keys = {k for k in row if k.startswith(f"{name}_within_")}
                assert keys == {k for k in verdict[row["rung"]]
                                if k.startswith(f"{name}_within_")}, (dut, row["rung"], keys)
            flags = [v for k, v in row.items()
                     if isinstance(v, bool) and k != "all_inside_bar"]
            assert row["all_inside_bar"] == all(flags), (dut, row)
        qualifying = [r["rung"] for r in lad["rung_within_bar_vs_finest"]
                      if r["all_inside_bar"]]
        assert lad["coarsest_rung_within_bar"] == (qualifying[0] if qualifying else None), dut


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_the_fitted_propagation_constant_follows_from_the_stored_gamma(fixture, key):
    """``eps_eff = (beta c / omega)^2`` from the extractor's own fit. On a coax
    filled with ONE dielectric the analytic phase constant is the fill's own and
    a partially filled line can only sit below it, so a value above the fill's
    is a statement about the realized line that the fixture has to carry rather
    than round away."""
    entry = _solve(fixture, key)
    mb = entry["measured_beta"]
    g = _complex(entry["gamma"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    axes = tuple(range(g.ndim - 1))
    beta = np.imag(g).mean(axis=axes) if axes else np.imag(g)
    np.testing.assert_allclose(beta, mb["beta_fitted_rad_per_m"], rtol=1e-9, atol=1e-12)
    omega = 2.0 * np.pi * freqs
    np.testing.assert_allclose((beta * C0 / omega) ** 2, mb["eps_eff_fitted"],
                               rtol=1e-9, atol=1e-12)
    _, _, eps_fill = _radii(fixture)
    assert mb["eps_eff_analytic"] == pytest.approx(eps_fill, rel=1e-12)
    np.testing.assert_allclose(omega * math.sqrt(eps_fill) / C0,
                               mb["beta_analytic_rad_per_m"], rtol=1e-9, atol=1e-9)


def test_the_ladder_carries_the_fitted_phase_constant_per_rung(fixture):
    """The ladder's own answer to whether the phase constant's gap refines away.
    Nothing is asserted about the trend here — the rows have to be present and
    to follow from the solves, and the reading is written where a person signs
    it."""
    for dut, lad in fixture["ladder"].items():
        mb = lad.get("measured_beta")
        assert mb is not None, f"the {dut} ladder carries no fitted phase constant"
        rungs = [row["rung"] for row in lad["rung_within_bar_vs_finest"]]
        assert len(mb["mean_eps_eff_fitted"]) == len(rungs), dut
        for i, rung in enumerate(rungs):
            assert mb["mean_eps_eff_fitted"][i] == pytest.approx(
                fixture["solves"][rung]["measured_beta"]["mean_eps_eff_fitted"],
                rel=1e-12), (dut, rung)


def test_the_bead_referee_on_the_fitted_permittivity_uses_that_permittivity(fixture):
    """The third referee reading: the same TEM closed form fed the line's OWN
    fitted eps_eff. It is a consistency witness and not a second opinion — the
    fitted eps_eff and the measured S come from one field — so what is checked
    here is only that it was built from the number it claims."""
    entry = _solve(fixture, f"bead_rung{CLAIMS_RUNG}")
    ref = entry.get("referee_fitted_eps_eff")
    if ref is None:
        pytest.skip("the fitted-permittivity referee is not assembled")
    eps_eff = entry["referee_fitted_eps_eff_value"]
    assert eps_eff == pytest.approx(
        float(np.mean(entry["measured_beta"]["eps_eff_fitted"])), rel=1e-12)
    assert ref["eps_fill"] == pytest.approx(eps_eff, rel=1e-12)
    a, b, _ = _radii(fixture)
    gam, an11, an21 = _tem_section(
        np.asarray(entry["freqs_hz"], dtype=float), a=a, b=b, eps_fill=eps_eff,
        eps_scale=fixture["line"]["bead_eps_scale"],
        length=entry["realized"]["bead_length_realized_m"],
        d1=entry["realized"]["d_port1_to_bead_m"],
        d2=entry["realized"]["d_port2_to_bead_m"])
    np.testing.assert_allclose(np.abs(an11), ref["abs_S11"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.abs(an21), ref["abs_S21"], rtol=1e-9, atol=1e-12)
    # The impedance step does not depend on the fill, only on the ratio.
    assert gam == pytest.approx(-1.0 / 3.0, rel=1e-9)


# ---------------------------------------------------------------------------
# criterion 1(2) — the forward identity
# ---------------------------------------------------------------------------

# P2 (PI 2026-09-23). The thru arm compares the untraced call (the float64 NumPy
# assembly, eps_scale=None) with the traced one (the float32 jnp path,
# _prefer_jnp). Those are two functions by design, and the contract's identity
# clause applies "where the traced and untraced call are the same function", so
# their difference is not held to rtol 1e-5 / atol 1e-7. It is pinned as a
# measured envelope: max |dS| = 6.938e-4 on the thru (identity stage, run
# 369367263527, commit ca6da2b1), times 1.5.
THRU_TRACED_VS_UNTRACED_ENVELOPE = 1.041e-3


def test_the_forward_identity_holds_on_the_eps_scale_channel(fixture):
    ident = fixture.get("identity")
    if ident is None:
        pytest.skip("the forward-identity stage is not assembled")
    assert ident["arms"], "the identity stage recorded no arm"
    assert {arm["dut"] for arm in ident["arms"]} == {"thru", "bead"}, (
        "criterion 1(2) needs both arms: the identity within the traced path and "
        "the envelope between the two paths")
    for arm in ident["arms"]:
        a = _complex(arm["left_S"])
        b = _complex(arm["right_S"])
        assert arm["max_abs"] == pytest.approx(float(np.abs(a - b).max()), rel=1e-12), (
            arm["tag"])
        if arm["left"] == "eps_scale=None":
            # Untraced against traced: two functions by design (P2, PI
            # 2026-09-23), held to the envelope measured on them, not to the
            # contract's identity, which covers the same function called twice.
            assert arm["judged_as"] == "measured_envelope", arm["tag"]
            assert arm["envelope_max_abs"] == THRU_TRACED_VS_UNTRACED_ENVELOPE, arm["tag"]
            worst = float(np.abs(a - b).max())
            assert worst <= THRU_TRACED_VS_UNTRACED_ENVELOPE, (
                f"{arm['tag']}: the untraced and the traced call now differ by "
                f"{worst:.4g} in complex S, outside the {THRU_TRACED_VS_UNTRACED_ENVELOPE:g} "
                "envelope pinned on their measured 6.938e-4 — one of the two paths moved")
            continue
        # The same function twice (the bead in a numpy and in a jnp container,
        # both on the traced path): the contract's identity, unchanged.
        assert arm["judged_as"] == "identity", arm["tag"]
        np.testing.assert_allclose(
            b, a, rtol=ident["rtol"], atol=ident["atol"],
            err_msg=(f"{arm['tag']}: {arm['what']} — the no-op override does not "
                     f"reproduce the call it is a no-op of"))


# ---------------------------------------------------------------------------
# criterion 3(a) — AD against a float64 finite difference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", ("two_port", "one_port"))
def test_the_gradient_matches_a_float64_finite_difference(fixture, lane):
    ad = fixture["adfd"].get(lane)
    if ad is None:
        pytest.skip(f"the {lane} AD/FD stage is not assembled")
    assert ad["cases"], f"{lane}: the AD stage recorded no objective"
    for case in ad["cases"]:
        fd = case["fd"]
        stored = abs(fd["f_plus"] - fd["f_minus"])
        mid = abs(0.5 * (fd["f_plus"] + fd["f_minus"]))
        ulp = float(np.spacing(np.asarray(mid, dtype=np.float64)))
        assert fd["loss_dtype"] == "float64", case["objective"]
        assert fd["ulp_span"] == pytest.approx(stored / ulp, rel=1e-9), case["objective"]
        # The comparator's resolving power is read BEFORE its verdict: a
        # reference that cannot resolve the quantity turns rel_err into noise.
        assert fd["ulp_span"] >= ad["min_fd_ulp_span"], (
            f"{lane}/{case['objective']}: the float64 FD reference spans "
            f"{fd['ulp_span']:.3e} ULP, below the {ad['min_fd_ulp_span']:.0e} floor — "
            "its disagreement with AD would say nothing about the gradient")
        g_fd = (fd["f_plus"] - fd["f_minus"]) / (2.0 * fd["h"])
        assert fd["grad"] == pytest.approx(g_fd, rel=1e-9), case["objective"]
        rel = abs(case["ad"]["grad"] - g_fd) / abs(g_fd)
        assert case["rel_err"] == pytest.approx(rel, rel=1e-9), case["objective"]
        assert case["ad"]["grad"] * g_fd > 0, (
            f"{lane}/{case['objective']}: AD and FD have opposite signs "
            f"({case['ad']['grad']:+.4e} vs {g_fd:+.4e})")
        assert rel <= ad["bar"], (
            f"{lane}/{case['objective']}: AD vs FD rel_err {rel:.4f} "
            f"(bar {ad['bar']})")


# ---------------------------------------------------------------------------
# criterion 3(b) — reference-plane invariance
# ---------------------------------------------------------------------------

def _plane_core(fixture, base_S):
    """Bins where the base run's own |S11| is already at the deep-null level:
    a dB difference there is the difference of two near-zeros."""
    return _db(base_S[0, 0, :]) <= fixture["bar"]["deep_null_db"]


def test_moving_the_dut_leaves_the_magnitudes_alone(fixture):
    pl = fixture.get("plane")
    if pl is None:
        pytest.skip("the reference-plane stage is not assembled")
    a = _complex(pl["base_S"])
    b = _complex(pl["shifted_S"])
    d = np.abs(_db(a) - _db(b))
    np.testing.assert_allclose(d, pl["mag_diff_db"], rtol=1e-9, atol=1e-12)
    outside = ~_plane_core(fixture, a)
    assert outside.any(), "every bin sits in the reflection zero's core"
    worst = float(d[:, :, outside].max())
    assert worst <= pl["bar_magnitude_db"], (
        f"|S| moved {worst:.3f} dB when only the DUT's position moved "
        f"({pl['shift_m']*1e3:.4f} mm), outside the reflection zero's core")


# The contract calls this leg report-only on its first run "against a
# pre-declared 1e-2", and this IS the coax family's first run, so 1e-2 rad is
# what it is pinned against here rather than at a tolerance read off the
# measurement.
PLANE_ROTATION_BOUND_RAD = 1.0e-2


def _wrap(x):
    return np.abs(np.angle(np.exp(1j * np.asarray(x))))


def test_the_plane_shift_rotates_the_phase_by_two_beta_delta(fixture):
    """Moving the DUT by Delta moves port 1's electrical distance to it by
    -Delta and port 2's by +Delta, so S11 and S22 rotate by 2 beta Delta in
    OPPOSITE directions and S21, whose total path is conserved, does not rotate
    at all.

    Both signs are stored because which one the extractor carries is a fact
    about its phase convention, not something this battery gets to choose; what
    is asserted is that ONE of them is right and the other is not, which is what
    a mis-placed reference plane or a non-unit-modulus shift factor breaks.
    """
    pl = fixture.get("plane")
    if pl is None:
        pytest.skip("the reference-plane stage is not assembled")
    a = _complex(pl["base_S"])
    b = _complex(pl["shifted_S"])
    freqs = np.asarray(pl["freqs_hz"], dtype=float)
    _, _, eps_fill = _radii(fixture)
    beta = 2.0 * np.pi * freqs * math.sqrt(eps_fill) / C0
    np.testing.assert_allclose(beta, pl["analytic_beta_rad_per_m"],
                               rtol=1e-12, atol=1e-9)
    pred_analytic = 2.0 * beta * pl["shift_m"]
    np.testing.assert_allclose(pred_analytic, pl["predicted_rotation_rad"],
                               rtol=1e-9, atol=1e-12)
    # The prediction the contract's own wording supports: beta is a property of
    # the line, and this lane MEASURES it (the matrix-pencil fit the extractor
    # already returns as gamma). The analytic omega*sqrt(eps_fill)/c is stored
    # beside it, and the two disagree by however much the realized line differs
    # from the declared one — which is a fact about the line, not about the
    # plane shift this test is checking.
    pred = np.asarray(pl["predicted_rotation_from_fitted_beta_rad"], dtype=float)
    np.testing.assert_allclose(
        pred, 2.0 * np.asarray(pl["fitted_beta_rad_per_m"], dtype=float) * pl["shift_m"],
        rtol=1e-9, atol=1e-12)

    outside = ~_plane_core(fixture, a)
    rot11 = np.angle(b[0, 0, :] * np.conj(a[0, 0, :]))
    rot22 = np.angle(b[1, 1, :] * np.conj(a[1, 1, :]))
    rot21 = np.angle(b[1, 0, :] * np.conj(a[1, 0, :]))
    np.testing.assert_allclose(rot11, pl["rotation_s11_rad"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rot22, pl["rotation_s22_rad"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rot21, pl["rotation_s21_rad"], rtol=1e-9, atol=1e-12)

    # S21's path length is conserved, so its rotation is zero in either
    # convention — nothing to choose, and a residual here is not a sign question.
    worst21 = float(_wrap(rot21)[outside].max())
    assert worst21 <= PLANE_ROTATION_BOUND_RAD, (
        f"S21 rotated {worst21:.4g} rad when the DUT moved, although the sum of "
        "the two port-to-DUT distances did not change")

    # S11 and S22 must rotate by the same amount in OPPOSITE directions.
    plus = float(_wrap(rot11 - pred)[outside].max())
    minus = float(_wrap(rot11 + pred)[outside].max())
    best, other = (plus, minus) if plus <= minus else (minus, plus)
    assert best <= PLANE_ROTATION_BOUND_RAD, (
        f"S11's rotation is {best:.4g} rad from 2*beta_fitted*Delta in the better "
        f"of the two sign conventions (the other reads {other:.4g} rad), above the "
        f"contract's {PLANE_ROTATION_BOUND_RAD} rad — the extractor's reference "
        "plane is not where its result says it is, or the shift factor is not "
        "unit modulus")
    s22_opposite = float(_wrap(rot22 + (pred if plus <= minus else -pred))[outside].max())
    assert s22_opposite <= PLANE_ROTATION_BOUND_RAD, (
        f"S11 rotates one way but S22 does not rotate the other way "
        f"({s22_opposite:.4g} rad); moving the DUT changes the two port-to-DUT "
        "distances by equal and opposite amounts")


def test_a_sign_flipped_plane_rotation_would_be_caught(fixture):
    """The mutation for the check above: keep the stored base run, build the
    shifted run with S11 rotated the WRONG way, and confirm the shipped
    comparison reds on it. Without this, a rotation check that had quietly
    become an identity would pass forever."""
    pl = fixture.get("plane")
    if pl is None:
        pytest.skip("the reference-plane stage is not assembled")
    a = _complex(pl["base_S"])
    pred = np.asarray(pl["predicted_rotation_from_fitted_beta_rad"], dtype=float)
    outside = ~_plane_core(fixture, a)
    rot11 = np.asarray(pl["rotation_s11_rad"], dtype=float)
    plus = float(_wrap(rot11 - pred)[outside].max())
    minus = float(_wrap(rot11 + pred)[outside].max())
    # The rotation the run actually carries, doubled: a plane placed twice as
    # far away. It has to fail the same comparison the shipped test makes.
    wrong = _wrap(2.0 * rot11 - (pred if plus <= minus else -pred))[outside].max()
    assert float(wrong) > PLANE_ROTATION_BOUND_RAD, (
        "doubling the measured rotation still lands inside the bound, so the "
        "comparison cannot see a mis-placed reference plane")
    assert min(plus, minus) < max(plus, minus), (
        "the two sign conventions give identical residuals, so this check cannot "
        "say which one the extractor carries")


# ---------------------------------------------------------------------------
# the mutation that revives the defect these checks exist for
# ---------------------------------------------------------------------------

def test_perturbing_the_stored_s_breaks_its_own_summary(fixture):
    """A summary that no longer follows from the S beside it. Only the stored
    complex numbers move; every check above is the shipped one."""
    entry = fixture["solves"].get(f"bead_rung{CLAIMS_RUNG}")
    if entry is None:
        pytest.skip(f"no bead_rung{CLAIMS_RUNG} record")
    S = _complex(entry["S"])
    S[0, 0, :] *= 1.01                      # 0.086 dB, far inside any bar
    assert not np.allclose(_db(S[0, 0, :]), entry["s11_db"], rtol=0, atol=1e-9)
    assert not np.allclose(_column_power(S), entry["power"]["column_power"],
                           rtol=1e-12, atol=1e-15)
    k = int(np.argmin(np.abs(S[0, 0, :])))
    S[0, 0, k] *= 0.5                       # move the zero's depth
    assert _db(S[0, 0, :])[k] != pytest.approx(
        np.asarray(entry["s11_db"])[k], abs=1e-9)


def test_a_wrong_analytic_length_moves_the_referee_off_the_measurement(fixture):
    """The referee's own falsifier: the TEM section formula reads the bead's
    length, so a bead 10 % longer must put the reflection zero outside the 1 %
    bar. Without this the frequency check could pass on a formula that ignored
    its input."""
    entry = _solve(fixture, f"bead_rung{CLAIMS_RUNG}")
    a, b, eps_fill = _radii(fixture)
    eps_scale = fixture["line"]["bead_eps_scale"]
    length = entry["realized"]["bead_length_realized_m"]
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    S = _complex(entry["S"])
    k = int(np.argmin(np.abs(S[0, 0, :])))
    f_meas = _zero_hz(freqs, S[0, 0, :], k)
    f_wrong = C0 / (2.0 * 1.1 * length * math.sqrt(eps_fill * eps_scale))
    assert abs(f_meas - f_wrong) / f_wrong > fixture["bar"]["frequency_frac"], (
        "a 10 % wrong bead length still lands inside the 1 % bar — the frequency "
        "check is not sensitive to the quantity it claims to test")
    _, an11, _ = _tem_section(freqs, a=a, b=b, eps_fill=eps_fill,
                             eps_scale=eps_scale, length=1.1 * length,
                             d1=entry["realized"]["d_port1_to_bead_m"],
                             d2=entry["realized"]["d_port2_to_bead_m"])
    core = _db(an11) <= fixture["bar"]["deep_null_db"]
    worst = float(np.abs(_db(S[0, 0, :]) - _db(an11))[~core].max())
    assert worst > fixture["bar"]["magnitude_db"], (
        f"a 10 % wrong bead length is still within {worst:.3f} dB of the "
        "measurement — the magnitude check does not see the section's length")
