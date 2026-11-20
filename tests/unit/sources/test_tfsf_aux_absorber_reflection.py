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
record lengths, written here,
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

SHIPPED_2D = {"aux_n_cpml": 30, "aux_cpml_order": 4, "aux_cpml_kappa_max": 7.0,
              "aux_cpml_r_asymptotic": math.exp(-2 * 0.8 * 30)}

FAST_MEASURED = {
    0.0:  {"max": 3.7855e-07, "quantum": 1e7, "floor": True},
    30.0: {"max": 7.5787e-05, "quantum": 1e6},
    45.0: {"max": 7.0387e-05, "quantum": 1e6},
    60.0: {"f0_bound": 2.0 * max(1.7095e-05, 1.0833e-05, 8.0564e-06),
           "f0_floor": 0.5 * min(1.7095e-05, 1.0833e-05, 8.0564e-06)},
}

FULL_MEASURED = {
    60.0: {"max": 3.4447e-05, "quantum": 1e6},
    70.0: {"max": 5.1014e-05, "quantum": 1e5},
}

CV04_ECHO_MEASURED = 6.3843e-06
CV04_ECHO_QUANTUM = 1e8


def bar(measured: float, quantum: float) -> float:
    return gate_from_envelope(measured, quantum=quantum)



def test_the_1d_path_refuses_a_kappa_it_cannot_realize():
    """TFSFConfig carries b and c but no kappa, so a kappa_max override would
    build a profile the update equations do not implement. It is refused, not
    ignored -- the silent-inconsistency class this lane exists to remove."""
    with pytest.raises(ValueError, match="carries no kappa"):
        init_tfsf(150, 1e-3, 1.9e-12, cpml_layers=20, tfsf_margin=5,
                  aux_cpml_kappa_max=7.0, dtype='float32')


def test_the_oblique_dispatch_carries_the_override_instead_of_dropping_it():
    """``init_tfsf`` at an oblique angle hands the work to ``init_tfsf_2d``, and
    the auxiliary-absorber arguments have to survive that hop.

    A forwarded-parameter bug is invisible from the outside -- the call
    succeeds, the default absorber is built, and the caller believes it asked
    for something else. The falsifier is the layout: an override to 30 cells
    must produce a 30-cell auxiliary grid, and the default must not.
    """
    cfg_default, _ = init_tfsf(160, 1e-3, 1.9e-12, ny=160, cpml_layers=10,
                               tfsf_margin=5, angle_deg=30.0, bandwidth=0.1, dtype='float32')
    cfg_override, _ = init_tfsf(160, 1e-3, 1.9e-12, ny=160, cpml_layers=10,
                                tfsf_margin=5, angle_deg=30.0, bandwidth=0.1,
                                **SHIPPED_2D, dtype='float32')
    assert cfg_default.n_cpml == AUX_N_CPML
    assert cfg_override.n_cpml == SHIPPED_2D["aux_n_cpml"] != AUX_N_CPML
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
                  angle_deg=30.0, method="methodB", aux_n_cpml=200, dtype='float32')



@pytest.mark.parametrize("theta_deg", sorted(FAST_MEASURED))
def test_the_declared_absorber_meets_its_bar(theta_deg):
    """The 0-degree row is held to its bar only because an earlier measurement placed the reading at the instrument floor rather than the absorber reflection."""
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



def test_the_1d_echo_on_cv04s_own_band_meets_its_bar():
    r = measure_aux_echo_1d()
    assert r["n_bins"] > 200
    assert r["band_peak"] == pytest.approx(CV04_ECHO_MEASURED, rel=0.10)
    assert r["band_peak"] <= bar(CV04_ECHO_MEASURED, CV04_ECHO_QUANTUM)



@pytest.mark.slow
@pytest.mark.parametrize("theta_deg", sorted(FULL_MEASURED))
def test_the_full_rig_resolves_sixty_and_seventy(theta_deg):
    """An earlier measurement resolved the 70-degree reflection on the full rig."""
    m = FULL_MEASURED[theta_deg]
    r = measure_aux_reflection_2d(theta_deg, **FULL_RIG)
    assert r["fit_resid_max"] < FIT_RESID_LIMIT
    assert r["max"] == pytest.approx(m["max"], rel=0.05)
    assert r["max"] <= bar(m["max"], m["quantum"])


LEAK_BAR = 1.0e-3
GRAZE_RIG = {"nx": 400, "n_steps": 40000, "n_samp": 96}
DOMAIN_MEASURED = {
    70.0: {"max": 4.9831e-05, "quantum": 1e6, "inside": True},
    80.0: {"f0_bound": 2.0 * max(1.0305e-06, 1.1146e-06, 1.2552e-06),
           "f0_floor": 0.5 * min(1.0305e-06, 1.1146e-06, 1.2552e-06), "inside": True},
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
