"""The auxiliary absorber's reflection, as a gate (#888).

The TF/SF auxiliary grids carry normal and oblique plane waves, whose outgoing
field must be absorbed. These locks measure the returned field at each declared
angle and retain the instrument's fit-residual check. With H at the Yee half cell
(#1234), the 1-D depth is 20 and the 2-D depth remains 200; the 82-degree row is now
below the 1e-3 return-amplitude bar on this instrument.

THE ABSENCE THIS FILLS. The auxiliary-echo record invariant (#892, removed
2026-09-21 with the slab family it ran on) bounded WHEN the auxiliary echo
arrives and said so in its own words:
"it does not bound HOW LARGE the echo is ... this guard would pass a rig whose
absorber was ten times worse". Nothing measured the amplitude, so an absorber
reflecting 4 to 6 percent shipped on both TF/SF paths and set the injected
field for every consumer of them. This file measures it.

WHAT IS DECLARED, AND WHERE IT IS VALID. The bars below are the MEASURED
maxima of the shipped absorber at the rig each test runs on, carried through
the shared ``gate_from_envelope`` policy -- never a number chosen to pass.
Two rows are bounds, not envelope locks: the 60-degree fast row and the
80-degree domain row read the rig's own floor at f0 and move with record length,
so they are held between 0.5x the smallest and 2x the largest reading over three
record lengths (rfx-archive rfx/records/20261020-aux-f0-reading/), written here,
not through gate_from_envelope.
They are per angle because the reflection is per angle: the optimum reflection
target inverts between normal and near-grazing incidence, so no single absorber
setting minimises every angle and the derivation is taken at the WORST DECLARED
ANGLE. That angle is **82 degrees**, where cv26 declares its three grazing arms,
not the 70 degrees of its primary-rig cap -- this paragraph said 70 until the
2026-09-13 review of PR #1005, which is the value section 3 of the note derived
at and section 12 corrected. The assertion is
``test_the_target_was_rederived_at_grazing_not_at_seventy`` below, and
``test_the_domain_edge_is_where_the_note_says`` now records the updated
82-degree reading below the bar.

Both angles are quoted from the note's section 12, not read from a file here:
cv26 is PR #924's lane and is NOT in this repository, so there is no
``validation/crossval/26_*`` to grep for ``THETA_GATE_MAX_DEG`` or
``GRAZE_THETA0_DEG``. Nothing in this file reads them at run time.

The declared ANGLE domain comes with a RESOLUTION: every 2-D rig here and in
``tests/_aux_absorber_reflection.py`` runs at 29.98 cells per free-space
wavelength (``dx = 1 mm``, ``lambda_0 = 29.979 mm`` at 10 GHz; rounded to 30
below and in the note), so "meets the bar through 82 degrees" means "at that
mesh". A cells-per-wavelength sweep is not run in this lane and is not filed as
an issue; ``test_the_declared_domain_names_the_resolution_it_was_measured_at``
is what keeps the claim from widening by accident. The 1-D depth law is a
different case and is not covered by it: ``measure_aux_echo_1d`` integrates
cv04's 3-15 GHz band, roughly 20 to 100 cells per wavelength.

Historical derivation: ``docs/design_notes/20260904_aux_absorber_depth_derivation.md``.
That dated note retains its original measurements; the comments below record
each #1234 re-pin at the final configuration.
"""

from __future__ import annotations

import math

import pytest

from rfx.sources.tfsf import init_tfsf
from rfx.sources.tfsf_2d import AUX_CPML_KAPPA_MAX, AUX_N_CPML, init_tfsf_2d
from tests._aux_absorber_reflection import FAST_RIG, FIT_RESID_LIMIT, FULL_RIG, measure_aux_echo_1d, measure_aux_reflection_2d
from tests._gate_policy import gate_from_envelope

# ---------------------------------------------------------------------------
# The profile that SHIPPED before this lane. It is no longer run through the rig (the
# tests that showed it missing the bar were removed under #1127; git history 609becd5):
# the dispatch test below passes it as an OVERRIDE and checks that it arrives.
# tfsf_2d.py before this lane: cpml_order = 4, kappa_max = 7.0,
# sigma_max = 0.8 (m+1)/(eta dx) * kappa_max. That sigma is the SAME number
# _cpml_profile produces at R_asymptotic = exp(-2 * 0.8 * n) for depth n, which
# is how the shipped profile is expressed through the shared law here.
# ---------------------------------------------------------------------------
SHIPPED_2D = {"aux_n_cpml": 30, "aux_cpml_order": 4, "aux_cpml_kappa_max": 7.0,
              "aux_cpml_r_asymptotic": math.exp(-2 * 0.8 * 30)}

# MEASURED maxima of the DECLARED absorber (n = 200, R_asym = 1e-28, re-derived at
# 82 deg -- see the module comment) at the FAST rig. Reproduced by the gate below to
# 5 % (except a row marked "floor" or "f0_bound", an upper bound at the f0 bin); the bar is derived from the measurement through gate_from_envelope, so widening
# it means editing a shared, reviewer-visible object rather than a local literal (#528).
FAST_MEASURED = {
    # 6.4413e-06 -> 3.7855e-07;
    # #1234: the auxiliary H profile at the Yee half cell.
    # The 0-degree reading is the instrument's floor, not the absorber (measured 2026-10-11,
    # Apple silicon, CPU, main 63a8f79b4): it is of the order of the fit residual (3.36e-07
    # against 2.98e-07), it does not follow the absorber depth from 200 down to 50 cells
    # (3.3e-07 .. 3.9e-07), and it moves with the rounding path -- 3.7855e-07 as pinned,
    # 3.3575e-07 on Apple silicon at the pinning commit dce96837e and at main alike (the
    # code did not drift), 3.71e-07 with x64 enabled, 4.12e-07 with a longer record, 4.68e-07
    # with Bloch at t0 = 5 tau (#1831; 100 and 50 cells then read 2.6e-07). A real
    # reflection shows: 30 cells read 1.5e-06, R_asymptotic 1e-6 reads 6.6e-07. So this
    # row ("floor") is held to its bar only and is not reproduced to 5 %.
    0.0:  {"max": 3.7855e-07, "quantum": 1e7, "floor": True},
    # 7.8540e-05 -> 7.5787e-05;
    # #1234: the auxiliary H profile at the Yee half cell.
    30.0: {"max": 7.5787e-05, "quantum": 1e6},
    # 7.5086e-05 -> 7.0387e-05;
    # #1234: the auxiliary H profile at the Yee half cell.
    45.0: {"max": 7.0387e-05, "quantum": 1e6},
    # 60 deg is an upper bound, not a lock (#1845). At f0 this rig reads its own floor, which sits above
    # the auxiliary absorber's reflection: the f0-bin |B/A| falls about as 1/record length (1.71e-05 /
    # 1.08e-05 / 8.06e-06 at 2500 / 3750 / 5000 steps), and cutting the absorber from 200 to 30 cells moves
    # nothing (1.71e-05 -> 1.76e-05). The band maximum (3.25e-04 here) sits at band-edge bins, where the
    # forward wave is weak, and moves with record length, so it is not pinned. The row catches the
    # absorber's target reflection rising to about 1e-6 (R_asymptotic 1e-6 reads 1.61e-04, 4.7x the bound)
    # and nothing finer; a finer test of the absorber needs another rig. Bound: 2x the largest of the three
    # lengths. Records: rfx-archive rfx/records/20261020-aux-f0-reading/.
    60.0: {"f0_bound": 2.0 * max(1.7095e-05, 1.0833e-05, 8.0564e-06),
           "f0_floor": 0.5 * min(1.7095e-05, 1.0833e-05, 8.0564e-06)},
}

# MEASURED maxima at the FULL rig, which does resolve 70 degrees.
FULL_MEASURED = {
    # 5.1338e-05 -> 3.4447e-05;
    # #1234: the auxiliary H profile at the Yee half cell.
    60.0: {"max": 3.4447e-05, "quantum": 1e6},
    # 1.2186e-04 -> 5.1014e-05;
    # #1234: the auxiliary H profile at the Yee half cell.
    70.0: {"max": 5.1014e-05, "quantum": 1e5},
}

# The 1-D path on cv04's own rig and band, now at 20 cells.
# 9.430e-06 -> 6.3843e-06;
# #1234: the auxiliary H profile at the Yee half cell.
CV04_ECHO_MEASURED = 6.3843e-06
CV04_ECHO_QUANTUM = 1e8


def bar(measured: float, quantum: float) -> float:
    return gate_from_envelope(measured, quantum=quantum)


# ==========================================================================
# 1. The declared absorber is what the module says it is
# ==========================================================================

def test_the_1d_path_refuses_a_kappa_it_cannot_realize():
    """TFSFConfig carries b and c but no kappa, so a kappa_max override would
    build a profile the update equations do not implement. It is refused, not
    ignored -- the silent-inconsistency class this lane exists to remove."""
    with pytest.raises(ValueError, match="carries no kappa"):
        init_tfsf(150, 1e-3, 1.9e-12, cpml_layers=20, tfsf_margin=5,
                  aux_cpml_kappa_max=7.0)


def test_the_oblique_dispatch_carries_the_override_instead_of_dropping_it():
    """``init_tfsf`` at an oblique angle hands the work to ``init_tfsf_2d``, and
    the auxiliary-absorber arguments have to survive that hop.

    A forwarded-parameter bug is invisible from the outside -- the call
    succeeds, the default absorber is built, and the caller believes it asked
    for something else. The falsifier is the layout: an override to 30 cells
    must produce a 30-cell auxiliary grid, and the default must not.
    """
    cfg_default, _ = init_tfsf(160, 1e-3, 1.9e-12, ny=160, cpml_layers=10,
                               tfsf_margin=5, angle_deg=30.0, bandwidth=0.1)
    cfg_override, _ = init_tfsf(160, 1e-3, 1.9e-12, ny=160, cpml_layers=10,
                                tfsf_margin=5, angle_deg=30.0, bandwidth=0.1,
                                **SHIPPED_2D)
    assert cfg_default.n_cpml == AUX_N_CPML
    assert cfg_override.n_cpml == SHIPPED_2D["aux_n_cpml"] != AUX_N_CPML
    # and the profile itself, not only the depth: kappa is 1 on the derived
    # absorber and 7 on the one that shipped, so a forwarded override changes
    # the stretching array too, not just its length.
    assert float(cfg_default.kappa_cpml.max()) == pytest.approx(
        AUX_CPML_KAPPA_MAX, rel=1e-6)
    assert float(cfg_override.kappa_cpml.max()) == pytest.approx(
        SHIPPED_2D["aux_cpml_kappa_max"], rel=1e-5)


def test_method_b_refuses_an_override_it_would_not_honour():
    """Method B (``tfsf_oblique_open.py``) builds its own 20-cell auxiliary
    absorber from the pre-#888 heuristic; #888 did not touch it. Accepting an
    aux_* argument on that path and dropping it is the same silent-drop the
    waveform guard beside it exists to prevent, so it raises."""
    with pytest.raises(ValueError, match="methodB"):
        init_tfsf(160, 1e-3, 1.9e-12, ny=160, cpml_layers=10, tfsf_margin=5,
                  angle_deg=30.0, method="methodB", aux_n_cpml=200)


# ==========================================================================
# 2. What it reflects, where the instrument can see
# ==========================================================================

@pytest.mark.parametrize("theta_deg", sorted(FAST_MEASURED))
def test_the_declared_absorber_meets_its_bar(theta_deg):
    m = FAST_MEASURED[theta_deg]
    r = measure_aux_reflection_2d(theta_deg, **FAST_RIG)
    assert r["fit_resid_max"] < FIT_RESID_LIMIT, (
        f"the rig did not resolve {theta_deg} deg (residual "
        f"{r['fit_resid_max']:.2e}); its |B/A| is not a measurement")
    if "f0_bound" in m:
        assert r["f0_bin"] <= m["f0_bound"], (
            f"{theta_deg} deg f0-bin reading {r['f0_bin']:.6e} exceeds bound {m['f0_bound']:.6e}")
    else:
        if not m.get("floor"):
            assert r["max"] == pytest.approx(m["max"], rel=0.05)
        assert r["max"] <= bar(m["max"], m["quantum"])
    if "f0_floor" in m:
        assert r["f0_bin"] >= m["f0_floor"], (
            f"{theta_deg} deg f0-bin reading {r['f0_bin']:.6e} falls below floor {m['f0_floor']:.6e}")


# ==========================================================================
# 3. The 1-D path, on cv04's own rig and band
# ==========================================================================

def test_the_1d_echo_on_cv04s_own_band_meets_its_bar():
    r = measure_aux_echo_1d()
    assert r["n_bins"] > 200
    assert r["band_peak"] == pytest.approx(CV04_ECHO_MEASURED, rel=0.10)
    assert r["band_peak"] <= bar(CV04_ECHO_MEASURED, CV04_ECHO_QUANTUM)


# ==========================================================================
# 4. The angles only the full rig resolves
# ==========================================================================

@pytest.mark.slow
@pytest.mark.parametrize("theta_deg", sorted(FULL_MEASURED))
def test_the_full_rig_resolves_sixty_and_seventy(theta_deg):
    m = FULL_MEASURED[theta_deg]
    r = measure_aux_reflection_2d(theta_deg, **FULL_RIG)
    assert r["fit_resid_max"] < FIT_RESID_LIMIT
    assert r["max"] == pytest.approx(m["max"], rel=0.05)
    assert r["max"] <= bar(m["max"], m["quantum"])


# ==========================================================================
# 5. The validity DOMAIN, in both directions (note section 12.5)
# ==========================================================================
# The first derivation took 70 deg as the worst declared angle; cv26's grazing
# arms are declared at 82. Measured on a 40000-step rig so the grazing bands
# carry bins. With the half-cell H profile the measured rows at 70, 80 and 82
# degrees all meet LEAK_BAR. This samples through 82 degrees at the declared
# resolution; it does not locate a new outside edge beyond 82 degrees.
LEAK_BAR = 1.0e-3
GRAZE_RIG = {"nx": 400, "n_steps": 40000, "n_samp": 96}
DOMAIN_MEASURED = {
    # 1.2462e-04 -> 4.9831e-05;
    # #1234: the auxiliary H profile at the Yee half cell.
    70.0: {"max": 4.9831e-05, "quantum": 1e6, "inside": True},
    # 80 deg is an upper bound, not a lock (#1845). With the shipped absorber the f0 bin reads the rig's own
    # floor: the f0-bin |B/A| reads 1.03e-06 / 1.11e-06 / 1.26e-06 at 40000 / 60000 / 80000 steps, cutting
    # the absorber from 200 to 30 cells moves nothing (1.11e-06), and a target reflection of 1e-40 instead
    # of 1e-28 moves it 4 %. A uniformly worse absorber (higher target reflection) shows at f0 before the band maximum does. Target
    # reflection 1e-28 / 1e-26 / 1e-24 / 1e-22 / 1e-20 at 40000 steps reads f0 1.02e-06 / 4.34e-06 /
    # 1.35e-05 / 3.89e-05 / 1.08e-04 and band maximum 1.35e-04 / 2.37e-04 / 4.63e-04 / 9.31e-04 / 1.86e-03,
    # so the f0 bound fails from 1e-26 on, while a band-maximum bound of 2x the three-length maximum
    # (2.70e-04) would fail only from 1e-24. The band maximum sits at band-edge bins where the forward
    # wave is weak and moves with record length; it is not pinned and carries the domain check against
    # LEAK_BAR. Bound: 2x the largest, floor 0.5x the smallest f0 reading of the three lengths. Records:
    # rfx-archive rfx/records/20261020-aux-f0-reading/ and rfx/records/20261024-aux80-rsweep/.
    80.0: {"f0_bound": 2.0 * max(1.0305e-06, 1.1146e-06, 1.2552e-06),
           "f0_floor": 0.5 * min(1.0305e-06, 1.1146e-06, 1.2552e-06), "inside": True},
    # 1.2282e-03 -> 4.3980e-04; outside -> inside at the unchanged 1e-3 bar.
    # #1234: the auxiliary H profile at the Yee half cell.
    82.0: {"max": 4.3980e-04, "quantum": 1e5, "inside": True},
}


@pytest.mark.slow
@pytest.mark.parametrize("theta_deg", sorted(DOMAIN_MEASURED))
def test_the_domain_edge_is_where_the_note_says(theta_deg):
    from tests._aux_absorber_reflection import bandwidth_for
    m = DOMAIN_MEASURED[theta_deg]
    r = measure_aux_reflection_2d(theta_deg, bw=bandwidth_for(theta_deg), **GRAZE_RIG)
    assert r["fit_resid_max"] < FIT_RESID_LIMIT, (theta_deg, r["fit_resid_max"])
    if "f0_bound" in m:
        assert r["f0_bin"] <= m["f0_bound"], (
            f"{theta_deg} deg f0-bin reading {r['f0_bin']:.6e} exceeds bound {m['f0_bound']:.6e}")
    else:
        assert r["max"] == pytest.approx(m["max"], rel=0.10), (theta_deg, r["max"])
        assert r["max"] <= bar(m["max"], m["quantum"])
    if "f0_floor" in m:
        assert r["f0_bin"] >= m["f0_floor"], (
            f"{theta_deg} deg f0-bin reading {r['f0_bin']:.6e} falls below floor {m['f0_floor']:.6e}")
    if m["inside"]:
        assert r["max"] <= LEAK_BAR, f"{theta_deg} deg is declared INSIDE the domain and reads {r['max']:.3e}"
    else:
        assert r["max"] > LEAK_BAR, (
            f"{theta_deg} deg is declared OUTSIDE the domain but reads {r['max']:.3e} <= LEAK_BAR: "
            "either the absorber improved (re-derive the domain) or the instrument lost resolution")
