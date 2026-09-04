"""The auxiliary absorber's reflection, as a gate (#888).

THE ABSENCE THIS FILLS. ``tests/crossval/test_aux_echo_record_invariant.py``
(#892) bounds WHEN the auxiliary echo arrives and says so in its own words:
"it does not bound HOW LARGE the echo is ... this guard would pass a rig whose
absorber was ten times worse". Nothing measured the amplitude, so an absorber
reflecting 4 to 6 percent shipped on both TF/SF paths and set the injected
field for every consumer of them. This file measures it.

WHAT IS DECLARED, AND WHERE IT IS VALID. The bars below are the MEASURED
maxima of the shipped absorber at the rig each test runs on, carried through
the shared ``gate_from_envelope`` policy -- never a number chosen to pass. They
are per angle because the reflection is per angle: the optimum reflection
target inverts between normal and near-grazing incidence, so no single absorber
setting minimises every angle and the derivation is taken at the WORST DECLARED
ANGLE (70 degrees, cv26's gate cap).

The fast rig cannot resolve 70 degrees -- four bins and a fit residual of 0.24.
That is asserted as a property of the instrument
(``test_the_fast_rig_refuses_the_angle_it_cannot_resolve``) and the angle is
measured on the full rig under ``slow`` instead. An angle the instrument cannot
resolve is NOT-APPLICABLE, never a pass.

Derivation and every number: ``docs/design_notes/20260904_aux_absorber_depth_derivation.md``.
"""

from __future__ import annotations

import math

import pytest

from rfx.sources.tfsf import (
    AUX_CPML_KAPPA_MAX_1D, AUX_CPML_ORDER_1D, AUX_CPML_R_ASYMPTOTIC_1D,
    AUX_N_CPML_1D, AUX_N_MARGIN_1D, AUX_SRC_OFFSET_1D, init_tfsf,
)
from rfx.sources.tfsf_2d import (
    AUX_CPML_KAPPA_MAX, AUX_CPML_ORDER, AUX_CPML_R_ASYMPTOTIC, AUX_N_CPML,
    AUX_N_MARGIN_X, AUX_SRC_OFFSET, init_tfsf_2d,
)
from tests._aux_absorber_reflection import (
    FAST_RIG, FIT_RESID_LIMIT, FULL_RIG, measure_aux_echo_1d,
    measure_aux_reflection_2d, sigma_max_of,
)
from tests._gate_policy import gate_from_envelope

# ---------------------------------------------------------------------------
# The profile that SHIPPED, restated so it can be run as a falsifier.
# tfsf_2d.py before this lane: cpml_order = 4, kappa_max = 7.0,
# sigma_max = 0.8 (m+1)/(eta dx) * kappa_max. That sigma is the SAME number
# _cpml_profile produces at R_asymptotic = exp(-2 * 0.8 * n) for depth n, which
# is how the shipped profile is expressed through the shared law here.
# ---------------------------------------------------------------------------
SHIPPED_2D = {"aux_n_cpml": 30, "aux_cpml_order": 4, "aux_cpml_kappa_max": 7.0,
              "aux_cpml_r_asymptotic": math.exp(-2 * 0.8 * 30)}
SHIPPED_1D = {"aux_n_cpml": 20, "aux_cpml_order": 3,
              "aux_cpml_r_asymptotic": math.exp(-2 * 0.8 * 20)}

# MEASURED maxima of the DECLARED absorber at the FAST rig, and the fit residual
# each came with. Reproduced by the tests below; the gate is derived from the
# measurement through gate_from_envelope, so widening it means editing a shared,
# reviewer-visible object rather than a local literal (#528).
FAST_MEASURED = {
    0.0:  {"max": 3.0382e-06, "resid": 3.2e-07, "quantum": 1e8},
    30.0: {"max": 7.6721e-05, "resid": 5.7e-04, "quantum": 1e7},
    45.0: {"max": 7.2586e-05, "resid": 6.8e-04, "quantum": 1e7},
    60.0: {"max": 3.2100e-04, "resid": 2.0e-03, "quantum": 1e6},
}
# The angle the FAST rig cannot resolve, and what it reads when asked to.
FAST_UNRESOLVED_DEG = 70.0
FAST_UNRESOLVED_RESID = 0.24

# MEASURED maxima at the FULL rig, which does resolve 70 degrees.
FULL_MEASURED = {
    60.0: {"max": 4.6153e-05, "quantum": 1e7},
    70.0: {"max": 2.3357e-04, "quantum": 1e6},
}

# The 1-D path on cv04's own rig and band.
CV04_ECHO_MEASURED = 9.430e-06
CV04_ECHO_QUANTUM = 1e8
# Measured by driving the ORIGINAL origin/main module (b59e1d99) through the same
# padded-twin instrument, in a separate process against the un-parameterised code:
# 5.7846e-02. The override path below reproduces it to five significant figures,
# which is what makes SHIPPED_1D a faithful falsifier rather than an approximation
# of one. NOTE this is NOT the cv04 envelope-decomposition note's 4.4023e-02:
# that note's number is a different statistic (its own in-band two-mode fit), the
# same phenomenon at the same scale measured a different way.
CV04_ECHO_SHIPPED = 5.7846e-02


def bar(measured: float, quantum: float) -> float:
    return gate_from_envelope(measured, quantum=quantum)


# ==========================================================================
# 1. The declared absorber is what the module says it is
# ==========================================================================

def test_the_declared_constants_are_the_derived_ones():
    """Both paths carry the depth and target read off the derivation table,
    and both use the SAME law the 3-D absorber uses."""
    assert (AUX_N_CPML, AUX_CPML_ORDER, AUX_CPML_KAPPA_MAX, AUX_CPML_R_ASYMPTOTIC) \
        == (200, 3, 1.0, 1e-14)
    assert (AUX_N_CPML_1D, AUX_CPML_ORDER_1D, AUX_CPML_KAPPA_MAX_1D,
            AUX_CPML_R_ASYMPTOTIC_1D) == (200, 3, 1.0, 1e-6)


def test_the_deep_tight_absorber_is_GENTLER_than_the_shallow_one_it_replaces():
    """The reason a tighter reflection target is not a tighter absorber.

    sigma_max = -ln(R) (m+1) / (2 eta n dx) falls with depth, so the 200-cell
    R = 1e-14 layer carries LESS sigma than a 30-cell R = 1e-6 one, and two
    orders less than the 148.6 that shipped. Long and gentle beats short and
    steep -- which is why tightening R at fixed depth made the measured echo
    WORSE at every depth in the derivation table.
    """
    declared = sigma_max_of(AUX_N_CPML, AUX_CPML_R_ASYMPTOTIC)
    shallow = sigma_max_of(30, 1e-6)
    # tfsf_2d.py before this lane: 0.8 (m+1)/(eta dx) * kappa_max at m = 4,
    # kappa_max = 7. At dx = 1 mm that is 74.32; the #888 note's 148.647 is the
    # SAME expression at the dx/2 rung, which is where it was measured.
    shipped = 0.8 * (4 + 1) / (376.730313668 * 1e-3) * 7.0
    assert declared == pytest.approx(0.856, rel=1e-2)
    assert shallow == pytest.approx(2.445, rel=1e-2)
    assert shipped == pytest.approx(74.32, rel=1e-2)
    assert 0.8 * (4 + 1) / (376.730313668 * 0.5e-3) * 7.0 == pytest.approx(148.6, rel=1e-2)
    assert declared < shallow < shipped


def test_the_layout_follows_the_depth():
    """n2x, i0_x and the source index are derived from the absorber depth, not
    pinned beside it -- a deeper absorber must move them or it overlaps the
    mapped region."""
    cfg, _ = init_tfsf_2d(150, 4, 1e-3, 2.335e-12, cpml_layers=20, tfsf_margin=5,
                          f0=10e9, bandwidth=0.25, theta_deg=0.0)
    assert int(cfg.n_cpml) == AUX_N_CPML
    assert int(cfg.i0_x) == AUX_N_CPML + AUX_N_MARGIN_X
    assert int(cfg.src_x) == AUX_N_CPML + AUX_SRC_OFFSET
    assert int(cfg.src_x) < int(cfg.i0_x)          # the source stays out of the mapped span
    cfg1, st1 = init_tfsf(150, 1e-3, 1.9e-12, cpml_layers=20, tfsf_margin=5,
                          f0=10e9, bandwidth=0.5)
    assert int(cfg1.n_cpml) == AUX_N_CPML_1D
    assert int(cfg1.i0) == AUX_N_CPML_1D + AUX_N_MARGIN_1D
    assert int(cfg1.src_idx) == AUX_N_CPML_1D + AUX_SRC_OFFSET_1D
    assert int(st1.e1d.shape[0]) == 2 * AUX_N_CPML_1D + 2 * AUX_N_MARGIN_1D + (
        int(cfg1.x_hi) - int(cfg1.x_lo) + 2)


def test_the_1d_path_refuses_a_kappa_it_cannot_realize():
    """TFSFConfig carries b and c but no kappa, so a kappa_max override would
    build a profile the update equations do not implement. It is refused, not
    ignored -- the silent-inconsistency class this lane exists to remove."""
    with pytest.raises(ValueError, match="carries no kappa"):
        init_tfsf(150, 1e-3, 1.9e-12, cpml_layers=20, tfsf_margin=5,
                  aux_cpml_kappa_max=7.0)


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
    assert r["max"] == pytest.approx(m["max"], rel=0.05)
    assert r["max"] <= bar(m["max"], m["quantum"])


def test_the_fast_rig_refuses_the_angle_it_cannot_resolve():
    """The instrument's own validity limit, asserted rather than assumed.

    At 70 degrees the fast rig's band carries four bins and the two-mode fit
    leaves a residual of 0.24 -- two decades over the limit and three over the
    worst residual at any angle it does resolve. Its |B/A| there reads 2.3e-02
    for an absorber the full rig measures at 3.8e-05. A bar applied to that
    number would be judging noise, so the angle is NOT-APPLICABLE here and is
    measured under ``slow`` on the full rig instead.
    """
    r = measure_aux_reflection_2d(FAST_UNRESOLVED_DEG, **FAST_RIG)
    assert r["n_bins"] <= 8
    assert r["fit_resid_max"] > FIT_RESID_LIMIT
    assert r["fit_resid_max"] == pytest.approx(FAST_UNRESOLVED_RESID, rel=0.25)
    worst_resolved = max(m["resid"] for m in FAST_MEASURED.values())
    assert r["fit_resid_max"] > 100.0 * worst_resolved


# ==========================================================================
# 3. Falsifiers -- the bars kill what they were built to kill
# ==========================================================================

@pytest.mark.parametrize("theta_deg", [0.0, 45.0])
def test_the_shipped_absorber_FAILS_the_bar_it_never_had(theta_deg):
    """(B): the profile that shipped, run through the same rig and the same
    bar. It must fail, or the gate does not discriminate."""
    m = FAST_MEASURED[theta_deg]
    r = measure_aux_reflection_2d(theta_deg, **FAST_RIG, **SHIPPED_2D)
    assert r["fit_resid_max"] < FIT_RESID_LIMIT
    assert r["max"] > 1.0e-2
    assert r["max"] > 100.0 * bar(m["max"], m["quantum"])


def test_a_thirty_cell_layer_at_the_declared_target_also_fails():
    """(B), sharper: keeping the derived law but not the depth. Depth is the
    lever the derivation identified, so a shallow layer at the declared target
    must still fail -- otherwise the gate is testing the law, not the fix."""
    r = measure_aux_reflection_2d(45.0, **FAST_RIG, aux_n_cpml=30,
                                  aux_cpml_r_asymptotic=AUX_CPML_R_ASYMPTOTIC)
    assert r["fit_resid_max"] < FIT_RESID_LIMIT
    assert r["max"] > bar(FAST_MEASURED[45.0]["max"], FAST_MEASURED[45.0]["quantum"])


# ==========================================================================
# 4. The 1-D path, on cv04's own rig and band
# ==========================================================================

def test_the_1d_echo_on_cv04s_own_band_meets_its_bar():
    r = measure_aux_echo_1d()
    assert r["n_bins"] > 200
    assert r["band_peak"] == pytest.approx(CV04_ECHO_MEASURED, rel=0.10)
    assert r["band_peak"] <= bar(CV04_ECHO_MEASURED, CV04_ECHO_QUANTUM)


def test_the_shipped_1d_absorber_FAILS_that_bar():
    """(B) on the 1-D path, against the number measured from origin/main itself."""
    r = measure_aux_echo_1d(**SHIPPED_1D)
    assert r["band_peak"] == pytest.approx(CV04_ECHO_SHIPPED, rel=0.10)
    assert r["band_peak"] > 1000.0 * bar(CV04_ECHO_MEASURED, CV04_ECHO_QUANTUM)


# ==========================================================================
# 5. The angles only the full rig resolves
# ==========================================================================

@pytest.mark.slow
@pytest.mark.parametrize("theta_deg", sorted(FULL_MEASURED))
def test_the_full_rig_resolves_sixty_and_seventy(theta_deg):
    m = FULL_MEASURED[theta_deg]
    r = measure_aux_reflection_2d(theta_deg, **FULL_RIG)
    assert r["fit_resid_max"] < FIT_RESID_LIMIT
    assert r["max"] == pytest.approx(m["max"], rel=0.05)
    assert r["max"] <= bar(m["max"], m["quantum"])
