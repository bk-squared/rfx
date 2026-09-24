"""Replay of the microstrip chain battery. Reads the fixture, re-derives every
assembled number from the stored S, and compares the result against the bar.

No FDTD runs here. The measurement is
``scripts/diagnostics/msl_chain_battery_measure.py``, pre-declared in
``docs/design_notes/msl_chain_battery_predeclaration.md``; the bar is the v2.0
one in ``docs/design_notes/chain_closure_contract.md`` — magnitude within 2 dB,
frequencies within 1 %, column power at most 1.02, reciprocity at most 0.02,
AD against FD within 0.05, forward identity at rtol 1e-5 / atol 1e-7.

Every derived quantity is recomputed HERE from the stored complex S by an
implementation written independently of the assembler's, so a fixture whose
summary numbers were edited by hand — or an assembler whose arithmetic drifts
— reds instead of replaying. ``test_perturbing_the_stored_s_breaks_its_own_
summary`` runs that as a mutation: it revives the defect (a wrong S behind a
right-looking summary) with the checks left exactly as they ship.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "msl_chain_battery" / "fixture.json"

C0 = 299792458.0
CLAIMS_RUNG = "notch_25um"
DB_FLOOR = 1e-300


# ---------------------------------------------------------------------------
# independent arithmetic — deliberately not imported from the assembler
# ---------------------------------------------------------------------------

def _complex(block) -> np.ndarray:
    return (np.asarray(block["real"], dtype=float)
            + 1j * np.asarray(block["imag"], dtype=float))


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), DB_FLOOR))


def _vertex(freqs, y, k):
    """Parabola vertex through bins k-1, k, k+1, written from the three-point
    Lagrange form rather than the assembler's rearrangement of it."""
    x0, x1, x2 = (float(freqs[k - 1]), float(freqs[k]), float(freqs[k + 1]))
    y0, y1, y2 = (float(y[k - 1]), float(y[k]), float(y[k + 1]))
    d0 = (y2 - y1) / (x2 - x1)
    d1 = ((y1 - y0) / (x1 - x0))
    a = (d0 - d1) / (x2 - x0)
    b = d1 - a * (x1 + x0)
    return -b / (2.0 * a)


def _notch_hz(freqs, s21, k):
    """The battery's notch estimator: the parabola vertex on |S21|^2, which is
    the quantity a simple transmission zero makes quadratic."""
    return _vertex(freqs, np.abs(s21) ** 2, k)


def _stopband_edges(freqs, y_db, level, k):
    """Linear crossings of ``level`` bracketing bin ``k``, found by walking out
    from the minimum. Independent of the assembler's version."""
    n = len(y_db)
    lo = k
    while lo > 0 and y_db[lo - 1] <= level:
        lo -= 1
    hi = k
    while hi < n - 1 and y_db[hi + 1] <= level:
        hi += 1
    f_lo = f_hi = None
    if lo > 0:
        f_lo = np.interp(level, [y_db[lo], y_db[lo - 1]], [freqs[lo], freqs[lo - 1]]) \
            if y_db[lo - 1] > y_db[lo] else freqs[lo]
    if hi < n - 1:
        f_hi = np.interp(level, [y_db[hi], y_db[hi + 1]], [freqs[hi], freqs[hi + 1]]) \
            if y_db[hi + 1] > y_db[hi] else freqs[hi]
    return f_lo, f_hi


def _column_power(S):
    n_ports = S.shape[0]
    n_f = S.shape[2]
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


ALL_SOLVES = [f"{dut}_{um}um" for dut in ("notch", "thru") for um in (100, 50, 25)]


def test_the_fixture_names_its_own_provenance(fixture):
    assert fixture["schema"] == "rfx.msl_chain_battery"
    assert fixture["predeclaration"].endswith("msl_chain_battery_predeclaration.md")
    assert len(fixture["assembler_commit"]) == 40
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
def test_the_realized_board_is_the_declared_board(fixture, key):
    """The measurement asserted this before solving; the fixture has to carry
    the evidence, because a number whose board was never checked is not one of
    this battery's numbers."""
    entry = _solve(fixture, key)
    dec, real = entry["declared"], entry["realized"]
    dx = dec["dx_m"]
    assert real["n_pec_volume_cells"] == 0, f"{key}: the metal realized as a volume"
    assert abs(real["sheet_plane_z_m"] - dec["h_sub_m"]) < 1e-12, key
    assert real["substrate_cells_under_strip"] == round(dec["h_sub_m"] / dx), key
    assert abs(real["trace_w_geometric_m"] - dec["w_trace_m"]) <= dx, key
    if entry["dut"] == "notch":
        assert abs(real["stub_len_m"] - dec["l_stub_m"]) <= dx, key
        assert real["trace_n_rows"] == real["stub_n_cols"], key


@pytest.mark.parametrize("key", ALL_SOLVES)
def test_every_stored_summary_follows_from_the_stored_s(fixture, key):
    entry = _solve(fixture, key)
    S = _complex(entry["S"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    assert S.shape == (2, 2, len(freqs))

    np.testing.assert_allclose(_db(S[1, 0, :]), entry["s21_db"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(_db(S[0, 0, :]), entry["s11_db"], rtol=0, atol=1e-9)

    col = _column_power(S)
    np.testing.assert_allclose(col, entry["power"]["column_power"], rtol=1e-12, atol=1e-15)
    assert entry["power"]["max_column_power"] == pytest.approx(col.max(), rel=1e-12)
    np.testing.assert_allclose(1.0 - col, entry["power"]["power_closure"],
                               rtol=1e-12, atol=1e-15)

    recip = np.abs(S[1, 0, :] - S[0, 1, :])
    np.testing.assert_allclose(recip, entry["power"]["reciprocity_abs"],
                               rtol=1e-12, atol=1e-18)
    assert entry["power"]["reciprocity_metric"] == pytest.approx(
        recip.max() / np.abs(S).max(), rel=1e-12)

    if entry["dut"] != "notch":
        return
    mag = np.abs(S[1, 0, :])
    k = int(np.argmin(mag))
    assert entry["notch"]["bin_index"] == k
    assert entry["notch"]["bin_hz"] == pytest.approx(freqs[k], rel=1e-15)
    if 0 < k < len(mag) - 1:
        assert entry["notch"]["interp_on"] == "|S21|^2"
        # 1e-6, not 1e-9: this file's three-point Lagrange form and the
        # assembler's vertex-offset form are the same parabola written two
        # ways, and on the measured record they land 21 Hz apart on 3.808 GHz.
        # 1e-6 still rejects any hand edit above 3.8 kHz.
        assert entry["notch"]["interp_hz"] == pytest.approx(_notch_hz(freqs, S[1, 0, :], k),
                                                            rel=1e-6)
        assert entry["notch"]["interp_hz_on_magnitude"] == pytest.approx(
            _vertex(freqs, mag, k), rel=1e-6)
        assert entry["notch"]["estimator_spread_hz"] == pytest.approx(
            abs(entry["notch"]["interp_hz"] - entry["notch"]["interp_hz_on_magnitude"]),
            rel=1e-9)
    assert entry["notch_depth_db"] == pytest.approx(_db(mag)[k], abs=1e-9)

    f_lo, f_hi = _stopband_edges(freqs, _db(mag), -10.0, k)
    if entry["stopband"]["width_hz"] is not None:
        assert entry["stopband"]["width_hz"] == pytest.approx(f_hi - f_lo, rel=1e-6)

    _, eps_eff = _hj(entry["declared"]["w_trace_m"], entry["declared"]["h_sub_m"],
                     entry["declared"]["eps_r"])
    f_an = C0 / (4.0 * entry["declared"]["l_stub_m"] * math.sqrt(eps_eff))
    assert entry["f_notch_analytic_hz"] == pytest.approx(f_an, rel=1e-9)
    assert entry["notch_vs_analytic_frac"]["interp"] == pytest.approx(
        abs(entry["notch"]["interp_hz"] - f_an) / f_an, rel=1e-9)


def _hj(w, h, eps_r):
    """The zero-thickness quasi-static microstrip closed form, transcribed from
    the reference rather than imported, so a change to the library function
    shows up here as a disagreement instead of moving both sides together."""
    u = w / h
    eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 / u) ** -0.5
    if u <= 1.0:
        z0 = (60.0 / math.sqrt(eps_eff)) * math.log(8.0 / u + u / 4.0)
    else:
        z0 = 120.0 * math.pi / (math.sqrt(eps_eff)
                                * (u + 1.393 + 0.667 * math.log(u + 1.444)))
    return z0, eps_eff


# ---------------------------------------------------------------------------
# the bar
# ---------------------------------------------------------------------------

def test_the_record_at_every_rung_is_settled(fixture):
    for key in ALL_SOLVES:
        entry = fixture["solves"].get(key)
        if entry is None:
            continue
        settling = entry["settling_db"]
        assert settling is not None, f"{key} carries no settling witness"
        assert max(settling) <= fixture["bar"]["settling_db"], (
            f"{key}: settling {settling} dB is above the "
            f"{fixture['bar']['settling_db']} dB rule — the record was truncated "
            "before the structure rang down, so its S is not interpretable")


# The pre-declaration gates criterion 2 at the claims rung and REPORTS the two
# coarser ones ("25 um (others reported)"); their values are in the ladder.
CLAIMS_RUNGS = ("notch_25um", "thru_25um")


@pytest.mark.parametrize("key", CLAIMS_RUNGS)
def test_raw_column_power_stays_inside_the_passivity_bar(fixture, key):
    entry = _solve(fixture, key)
    S = _complex(entry["S"])
    measured = _column_power(S).max()
    assert measured <= fixture["bar"]["column_power_max"], (
        f"{key}: max column power {measured:.6f} exceeds "
        f"{fixture['bar']['column_power_max']} on a passive board, with the raw "
        "extraction the shipped default returns")


@pytest.mark.parametrize("key", CLAIMS_RUNGS)
def test_reciprocity_stays_inside_the_bar(fixture, key):
    entry = _solve(fixture, key)
    S = _complex(entry["S"])
    measured = np.abs(S[1, 0, :] - S[0, 1, :]).max() / np.abs(S).max()
    assert measured <= fixture["bar"]["reciprocity"], (
        f"{key}: max_f |S21 - S12| / max|S| = {measured:.6f}")


def test_the_coarser_rungs_report_their_physics_numbers(fixture):
    """Not gated, but they have to be THERE: a rung whose column power and
    reciprocity were never recorded cannot be the trend the ladder claims."""
    for key in ALL_SOLVES:
        entry = fixture["solves"].get(key)
        if entry is None:
            continue
        S = _complex(entry["S"])
        assert entry["power"]["max_column_power"] == pytest.approx(
            _column_power(S).max(), rel=1e-12), key
        assert entry["power"]["reciprocity_metric"] == pytest.approx(
            np.abs(S[1, 0, :] - S[0, 1, :]).max() / np.abs(S).max(), rel=1e-12), key


# The measured envelope of the reference-plane rotation residual: moving both
# observation planes 500 um leaves S11's phase within this of the 2*beta*Delta
# the line's own fitted beta predicts. The opposite sign misses by 0.488 rad,
# 220 times worse, which is what makes the sign assertion below a real check.
PLANE_ROTATION_RESIDUAL_RAD = 0.005


def test_the_notch_frequency_at_the_claims_rung_matches_the_quarter_wave_value(fixture):
    entry = _solve(fixture, CLAIMS_RUNG)
    S = _complex(entry["S"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    k = int(np.argmin(np.abs(S[1, 0, :])))
    f_meas = _notch_hz(freqs, S[1, 0, :], k)
    _, eps_eff = _hj(entry["declared"]["w_trace_m"], entry["declared"]["h_sub_m"],
                     entry["declared"]["eps_r"])
    f_an = C0 / (4.0 * entry["declared"]["l_stub_m"] * math.sqrt(eps_eff))
    frac = abs(f_meas - f_an) / f_an
    assert frac <= fixture["bar"]["frequency_frac"], (
        f"notch {f_meas/1e9:.5f} GHz against the analytic {f_an/1e9:.5f} GHz is "
        f"{frac*100:.3f} % — the bar is {fixture['bar']['frequency_frac']*100:.1f} %")


def test_the_thru_lines_reflection_stays_under_its_bound(fixture):
    """PI 2026-09-21 (a). The control line's |S11| is a deep null, so it carries
    an upper bound at every bin of every rung instead of a rung-to-rung dB
    comparison — a dB difference on a quantity 30 dB down says nothing about
    the line."""
    lad = fixture["ladder"].get("thru")
    if lad is None:
        pytest.skip("the thru ladder is not assembled")
    floor = lad["s11_reflection_floor"]
    assert "s11_vs_finest" not in lad, (
        "the thru carries a rung-to-rung |S11| dB comparison, which the ruling "
        "removed")
    for row in floor["per_rung"]:
        entry = fixture["solves"][row["rung"]]
        S = _complex(entry["S"])
        measured = float(_db(S[0, 0, :]).max())
        assert row["max_s11_db"] == pytest.approx(measured, abs=1e-9), row["rung"]
        assert row["within_bound"] is bool(measured <= floor["bound_db"])
        assert measured <= floor["bound_db"], (
            f"{row['rung']}: the line reflects {measured:.2f} dB at "
            f"{row['argmax_hz']/1e9:.3f} GHz, above the {floor['bound_db']} dB bound")


def test_the_notch_core_is_the_bins_the_finest_rung_puts_below_the_level(fixture):
    """PI 2026-09-21 (b). The excluded set is not a choice made per rung: it is
    the finest rung's own |S21|, and the fixture has to show it."""
    lad = fixture["ladder"].get("notch")
    if lad is None:
        pytest.skip("the notch ladder is not assembled")
    core = lad["notch_core"]
    fine = fixture["solves"][core["defined_by_rung"]]
    S = _complex(fine["S"])
    freqs = np.asarray(fine["freqs_hz"], dtype=float)
    expect = [int(i) for i in np.flatnonzero(_db(S[1, 0, :]) <= core["level_db"])]
    assert core["bin_indices"] == expect, "the stored core is not the measured one"
    assert core["n_bins"] == len(expect)
    np.testing.assert_allclose(core["bin_hz"], freqs[expect], rtol=0, atol=1e-3)
    for name in ("s21", "s11"):
        for row in lad[f"{name}_vs_finest"]:
            assert row["excluded_core_bin_indices"] == expect, (name, row["rung"])


def test_the_only_bins_the_extractor_declines_to_vouch_for_are_inside_the_null(fixture):
    """At a transmission zero the voltage and current at the port plane are both
    small by construction, so the extractor's low-signal mask fires there — and
    only there. The mask is stored per bin and per port; this asserts it sits
    inside the notch's own core on every mesh of the notch filter, and that the
    thru line, which has no null, carries none at all.

    What it protects: the notch frequency is the vertex of a parabola through
    the minimum bin and its two shoulders, and the shoulders (-27.78 and
    -27.03 dB at the finest mesh) are bins the extractor does vouch for. A mask
    that spread beyond the core would mean the shoulders were in doubt too, and
    with them the frequency this battery reports.
    """
    core = set(fixture["ladder"]["notch"]["notch_core"]["bin_indices"])
    for key in ALL_SOLVES:
        entry = fixture["solves"].get(key)
        if entry is None:
            continue
        rel = np.asarray(entry["reliable"], dtype=bool)
        freqs = np.asarray(entry["freqs_hz"], dtype=float)
        flagged = [sorted(int(k) for k in np.flatnonzero(~rel[p]))
                   for p in range(rel.shape[0])]
        assert flagged[0] == flagged[1], (
            f"{key}: the two ports disagree about which bins are low-signal, "
            f"{flagged[0]} against {flagged[1]} — a null is symmetric in a "
            "two-port measurement of a reciprocal structure")
        union = set(flagged[0])
        if entry["dut"] == "thru":
            assert not union, (
                f"{key}: the control line has no transmission zero, so no bin "
                f"should read low-signal, yet "
                f"{[round(freqs[i]/1e9, 4) for i in sorted(union)]} GHz does")
            continue
        assert union, (
            f"{key}: no bin reads low-signal at all. The notch is a "
            "transmission zero; if the extractor vouches for every bin through "
            "it, the mask is no longer measuring what it used to.")
        assert union <= core, (
            f"{key}: bins "
            f"{[round(freqs[i]/1e9, 4) for i in sorted(union - core)]} GHz read "
            "low-signal outside the notch's -20 dB core, so the doubt is no "
            "longer confined to the null")


def test_the_notch_depth_is_recorded_and_compared_with_nothing(fixture):
    """PI 2026-09-21 (c). Inside the core the verdict is the frequency. The
    depth is evidence; it carries no threshold and no boolean."""
    lad = fixture["ladder"].get("notch")
    if lad is None:
        pytest.skip("the notch ladder is not assembled")
    block = lad["notch_depth_db_per_rung"]
    assert not any(isinstance(v, bool) for row in block["per_rung"] for v in row.values()), (
        "the depth block carries a boolean, i.e. a comparison the ruling removed")
    for row in block["per_rung"]:
        entry = fixture["solves"][row["rung"]]
        S = _complex(entry["S"])
        k = int(np.argmin(np.abs(S[1, 0, :])))
        assert row["depth_db"] == pytest.approx(float(_db(S[1, 0, :])[k]), abs=1e-9)


def test_the_ladder_converges_and_sets_a_recommended_cell_size(fixture):
    lad = fixture["ladder"].get("notch")
    if lad is None:
        pytest.skip("the notch ladder is not assembled")
    assert lad["rungs_um"] == [100, 50, 25], lad["rungs_um"]
    fs = lad["notch_interp_hz"]
    diffs = [abs(fs[i + 1] - fs[i]) for i in range(len(fs) - 1)]
    np.testing.assert_allclose(diffs, lad["successive_diff_hz"], rtol=1e-9, atol=1.0)
    ratio = lad["successive_diff_ratio"]
    assert ratio is not None and ratio < 1.0, (
        f"successive notch-frequency differences {diffs} do not shrink (ratio "
        f"{ratio}); on a ladder that is not converging no cell size is recommended")
    # The recommended cell size is the deliverable of this ladder, so every
    # boolean under it is recomputed here from the stored curves — not read
    # back from the fixture and compared with itself. A row claiming a mesh is
    # inside the bar while its own dB difference says otherwise must fail, and
    # so must a ladder that recommends nothing.
    rows = lad["rung_within_bar_vs_finest"]
    bar_db = fixture["bar"]["magnitude_db"]
    bar_frac = fixture["bar"]["frequency_frac"]
    for i, row in enumerate(rows):
        for name in ("s21", "s11"):
            key = f"{name}_within_2dB"
            if row.get(key) is None:
                continue
            worst = lad[f"{name}_vs_finest"][i]["max_db_diff_vs_finest_outside_notch_core"]
            assert row[key] == bool(worst <= bar_db), (
                f"{row['rung']}: {key} is {row[key]} but its own curve is "
                f"{worst:.3f} dB from the finest mesh against a {bar_db} dB bar")
        if row.get("notch_within_1pct") is not None:
            assert row["notch_within_1pct"] == bool(
                lad["notch_frac_vs_finest"][i] <= bar_frac), row["rung"]
        if row.get("s11_floor_within_bound") is not None:
            floor = lad["s11_reflection_floor"]
            assert row["s11_floor_within_bound"] == bool(
                floor["per_rung"][i]["max_s11_db"] <= floor["bound_db"]), row["rung"]
        flags = [row[k] for k in ("s21_within_2dB", "s11_within_2dB",
                                  "notch_within_1pct", "s11_floor_within_bound")
                 if row.get(k) is not None]
        assert row["all_inside_bar"] == all(flags), row
    qualifying = [r["rung"] for r in rows if r["all_inside_bar"]]
    assert lad["coarsest_rung_within_bar"] == (qualifying[0] if qualifying else None)
    assert lad["coarsest_rung_within_bar"] is not None, (
        "no mesh on this ladder sits inside the bar against the finest one, so "
        "the battery recommends no cell size and the support matrix has nothing "
        "to carry")
    assert lad["coarsest_rung_within_bar"] == "notch_100um", (
        "the recommended cell size moved from 100 um to "
        f"{lad['coarsest_rung_within_bar']}; that is the number the support "
        "matrix carries, so it does not change silently")


def test_the_forward_identity_holds_on_the_eps_override_channel(fixture):
    ident = fixture.get("identity")
    if ident is None:
        pytest.skip("the forward-identity stage is not assembled")
    a = _complex(ident["plain_S"])
    b = _complex(ident["override_S"])
    assert ident["max_abs_diff"] == pytest.approx(np.abs(a - b).max(), rel=1e-12)
    np.testing.assert_allclose(b, a, rtol=ident["rtol"], atol=ident["atol"])


def test_the_gradient_matches_a_float64_finite_difference(fixture):
    ad = fixture.get("adfd")
    if ad is None:
        pytest.skip("the AD/FD stage is not assembled")
    for case in ad["cases"]:
        span = case["fd"]["ulp_span"]
        stored = abs(case["fd"]["f_plus"] - case["fd"]["f_minus"])
        mid = abs(0.5 * (case["fd"]["f_plus"] + case["fd"]["f_minus"]))
        ulp = float(np.spacing(np.asarray(mid, dtype=np.float64)))
        assert case["fd"]["loss_dtype"] == "float64", case["objective"]
        assert span == pytest.approx(stored / ulp, rel=1e-9), case["objective"]
        # The comparator's resolving power is read BEFORE its verdict: a
        # reference that cannot resolve the quantity turns rel_err into noise.
        assert span >= ad["min_fd_ulp_span"], (
            f"{case['objective']}: the float64 FD reference spans {span:.3e} ULP, "
            f"below the {ad['min_fd_ulp_span']:.0e} floor — its disagreement with "
            "AD would say nothing about the gradient")
        g_fd = (case["fd"]["f_plus"] - case["fd"]["f_minus"]) / (2.0 * case["fd"]["h"])
        assert case["fd"]["grad"] == pytest.approx(g_fd, rel=1e-9), case["objective"]
        rel = abs(case["ad"]["grad"] - g_fd) / abs(g_fd)
        assert case["rel_err"] == pytest.approx(rel, rel=1e-9), case["objective"]
        assert rel <= ad["bar"], f"{case['objective']}: AD vs FD rel_err {rel:.4f}"


def test_moving_both_reference_planes_leaves_the_magnitudes_alone(fixture):
    pl = fixture.get("plane")
    if pl is None:
        pytest.skip("the reference-plane stage is not assembled")
    a = _complex(pl["base_S"])
    b = _complex(pl["shifted_S"])
    d = np.abs(_db(a) - _db(b))
    np.testing.assert_allclose(d, pl["mag_diff_db"], rtol=1e-9, atol=1e-12)
    core = np.abs(a[1, 0, :]) <= 10 ** (-20.0 / 20.0)
    outside = ~core
    worst = float(d[:, :, outside].max())
    assert worst == pytest.approx(pl["max_mag_diff_db_outside_notch_core"], rel=1e-9)
    assert worst <= pl["bar_magnitude_db"], (
        f"|S| moved {worst:.3f} dB when only the observation plane moved "
        f"({pl['displacement_m']} m), outside the notch's -20 dB core")


def test_the_plane_shift_rotates_the_phase_by_two_beta_delta(fixture):
    pl = fixture.get("plane")
    if pl is None:
        pytest.skip("the reference-plane stage is not assembled")
    a = _complex(pl["base_S"])
    b = _complex(pl["shifted_S"])
    beta = _complex(pl["base_beta"])
    delta = float(pl["displacement_m"][0])
    pred = 2.0 * np.real(beta) * delta
    np.testing.assert_allclose(pred, pl["predicted_rotation_rad"], rtol=1e-9, atol=1e-12)
    for name, meas in (("S11", np.angle(b[0, 0, :] * np.conj(a[0, 0, :]))),
                       ("S21", np.angle(b[1, 0, :] * np.conj(a[1, 0, :])))):
        stored = pl[f"rotation_{name.lower()}_rad"]
        np.testing.assert_allclose(meas, stored, rtol=1e-9, atol=1e-12,
                                   err_msg=f"{name} rotation")

    # The invariant itself, not just that two stored arrays agree with their
    # own recomputation. Moving the observation plane Delta further down the
    # line removes 2*beta*Delta of round-trip phase from S11, so the measured
    # rotation must match +2*beta*Delta and must NOT match -2*beta*Delta. A
    # flipped sign convention regenerates both stored arrays consistently and
    # would otherwise ship green.
    rot11 = np.angle(b[0, 0, :] * np.conj(a[0, 0, :]))
    resid = np.abs(np.angle(np.exp(1j * (rot11 - pred))))
    resid_flipped = np.abs(np.angle(np.exp(1j * (rot11 + pred))))
    assert float(resid.max()) == pytest.approx(pl["max_rotation_residual_rad"], rel=1e-9)
    assert float(resid_flipped.max()) == pytest.approx(
        pl["max_rotation_residual_opposite_sign_rad"], rel=1e-9)
    assert resid.max() <= PLANE_ROTATION_RESIDUAL_RAD, (
        f"S11 rotates by {resid.max():.5f} rad away from the 2*beta*Delta the "
        f"line's own fitted beta predicts over a {delta*1e6:.0f} um shift; the "
        f"measured envelope is {PLANE_ROTATION_RESIDUAL_RAD} rad")
    assert resid.max() < resid_flipped.max(), (
        f"the measured rotation fits -2*beta*Delta ({resid_flipped.max():.5f} rad) "
        f"at least as well as +2*beta*Delta ({resid.max():.5f} rad) — the sign "
        "convention of the reference-plane shift is not what the fixture says")


def test_perturbing_the_stored_s_breaks_its_own_summary(fixture):
    """The mutation that revives the defect these checks exist for: a summary
    that no longer follows from the S beside it. Only the stored complex
    numbers move; every check above is the shipped one."""
    entry = fixture["solves"].get(CLAIMS_RUNG)
    if entry is None:
        pytest.skip(f"no {CLAIMS_RUNG} record")
    S = _complex(entry["S"])
    S[1, 0, :] *= 1.01                      # 0.086 dB, far inside any bar
    assert not np.allclose(_db(S[1, 0, :]), entry["s21_db"], rtol=0, atol=1e-9)
    assert not np.allclose(_column_power(S), entry["power"]["column_power"],
                           rtol=1e-12, atol=1e-15)
    k = int(np.argmin(np.abs(S[1, 0, :])))
    S[1, 0, k] *= 0.5                       # move the minimum's depth
    assert _db(S[1, 0, :])[k] != pytest.approx(entry["notch_depth_db"], abs=1e-9)
