"""Issue #680 — the NU lane must REFUSE boundary='upml', not substitute.

``rfx/nonuniform.py`` has no UPML implementation: its absorber dispatch is
``use_cpml = grid.cpml_layers > 0``, keyed on the layer COUNT, never on the
boundary type. So ``boundary='upml'`` on a graded mesh was accepted and
ignored — measured on otherwise identical 4x4x3 mm ez-dipole configs:
``apply_upml_e`` called 1x on the uniform lane, 0x on the NU lane, with
``sim._boundary`` still reading ``'upml'`` afterwards.

The counting test below is the falsifier that made the old behaviour
visible; it is kept so a future silent re-substitution is caught by a
positive measurement, not only by the absence of an exception.

Graded profiles here use a DIFFERENT ratio per axis on purpose: equal
ratios leave dual_x == dual_y == dual_z at the node, so an axis
permutation would pass.
"""

import numpy as np
import pytest

from rfx import Simulation
from rfx.sources.sources import GaussianPulse


def _graded(n, d0, ratio):
    return d0 * ratio ** np.arange(n, dtype=float)


def _sim(boundary, **profiles):
    sim = Simulation(freq_max=10e9, domain=(4e-3, 4e-3, 3e-3), dx=0.5e-3,
                     boundary=boundary, cpml_layers=6, **profiles)
    sim.add_source(position=(2e-3, 2e-3, 1.5e-3), component="ez",
                   waveform=GaussianPulse(f0=10e9, bandwidth=5e9),
                   amplitude_kind="current")
    return sim


DZ = _graded(8, 0.30e-3, 1.07)
DX = _graded(8, 0.50e-3, 1.03)   # deliberately a different ratio from dz
DY = _graded(8, 0.50e-3, 1.11)   # and different again from dx


@pytest.mark.parametrize("profiles", [
    {"dz_profile": DZ},
    {"dx_profile": DX, "dz_profile": DZ},
    {"dy_profile": DY, "dz_profile": DZ},
])
def test_run_refuses_upml_on_any_graded_axis(profiles):
    sim = _sim("upml", **profiles)
    with pytest.raises(ValueError, match=r"boundary='upml' does not support"):
        sim.run(n_steps=4, skip_preflight=True)


def test_forward_refuses_upml_on_a_graded_mesh():
    sim = _sim("upml", dz_profile=DZ)
    with pytest.raises(ValueError, match=r"boundary='upml' does not support"):
        sim.forward(n_steps=4, skip_preflight=True)


def test_message_names_the_issue_and_the_remedy():
    sim = _sim("upml", dz_profile=DZ)
    with pytest.raises(ValueError) as exc:
        sim.run(n_steps=4, skip_preflight=True)
    msg = str(exc.value)
    assert "#680" in msg
    assert "boundary='cpml'" in msg
    assert "non-uniform" in msg


def test_cpml_on_the_same_graded_mesh_still_runs():
    """The guard is scoped to upml — it must not close the NU CPML lane."""
    sim = _sim("cpml", dz_profile=DZ)
    sim.add_probe((2e-3, 2e-3, 1.5e-3), "ez")
    res = sim.run(n_steps=8, skip_preflight=True)
    ts = np.asarray(res.time_series)
    assert ts.shape[0] == 8
    assert np.all(np.isfinite(ts))


def test_uniform_lane_still_reaches_the_real_upml_kernel():
    """Positive measurement: the uniform lane calls apply_upml_e; the NU
    lane used to call it ZERO times for the same request (#680)."""
    import rfx.boundaries.upml as upml_mod

    calls = {"n": 0}
    original = upml_mod.apply_upml_e

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    upml_mod.apply_upml_e = counting
    try:
        _sim("upml").run(n_steps=4, skip_preflight=True)
        assert calls["n"] >= 1, (
            "uniform lane did not reach apply_upml_e — the counting "
            "falsifier itself is broken, so the NU measurement below "
            "would prove nothing")
        calls["n"] = 0
        with pytest.raises(ValueError, match=r"boundary='upml'"):
            _sim("upml", dz_profile=DZ).run(n_steps=4, skip_preflight=True)
        assert calls["n"] == 0
    finally:
        upml_mod.apply_upml_e = original


# ---------------------------------------------------------------------------
# The preflight advisory (#680 follow-up).
#
# The lane guard above is the enforcement point, but ``sim.preflight()``
# printed "All checks passed" and then ``run()`` refused — a preflight that
# does not know what the runner will do. The repo's convention for a
# dispatch-rejected non-uniform combination is an ADVISORY that explains the
# coming error in advance and covers ``skip_preflight=True``
# (``_validate_cfg_precision_x64``). These pin the matching one.
# ---------------------------------------------------------------------------

_ADVISORY = "upml_nonuniform_lane_unsupported"


def test_preflight_warns_before_the_lane_guard_raises():
    report = _sim("upml", dz_profile=DZ).preflight(strict=False)
    hits = report.by_code(_ADVISORY)
    assert len(hits) == 1, (
        f"expected exactly one NU/UPML advisory; got "
        f"{[getattr(i, 'code', None) for i in report]}")


@pytest.mark.parametrize("boundary,profiles,refused", [
    ("upml", {"dz_profile": DZ}, True),
    ("upml", {"dx_profile": DX}, True),
    ("upml", {"dy_profile": DY}, True),
    ("upml", {}, False),            # uniform lane DOES implement UPML
    ("cpml", {"dz_profile": DZ}, False),   # NU lane's real absorber
    ("cpml", {}, False),
])
def test_advisory_fires_exactly_when_the_run_is_refused(boundary, profiles,
                                                        refused):
    """The advisory and the error are one predicate.

    An advisory that fires on configs that run (or stays silent on configs
    that refuse) is worse than none — it teaches the reader to ignore it.
    So this asserts the two agree case by case rather than asserting the
    warning text in isolation.
    """
    warned = bool(_sim(boundary, **profiles).preflight(strict=False)
                  .by_code(_ADVISORY))
    try:
        _sim(boundary, **profiles).run(n_steps=4, skip_preflight=True)
        raised = False
    except ValueError as exc:
        assert "boundary='upml' does not support" in str(exc)
        raised = True
    assert raised is refused, (
        f"{boundary} {list(profiles)}: expected refused={refused}, "
        f"got {raised}")
    assert warned == raised, (
        f"{boundary} {list(profiles)}: preflight warned={warned} but "
        f"run() raised={raised} — the advisory and the guard disagree")


def test_advisory_carries_its_basis_not_only_its_verdict():
    """A finding a reader cannot audit is an assertion.

    The message must name what was observed, the mechanism, the cost, the
    legitimate alternative, and what would prove the guard stale — the
    last one is what lets a future reader tell a live guard from a
    leftover.
    """
    (hit,) = _sim("upml", dz_profile=DZ).preflight(strict=False) \
        .by_code(_ADVISORY)
    msg = str(hit)   # PreflightIssue subclasses str: it IS its message
    # observed
    assert "apply_upml_e ran 1x" in msg and "0x on the non-uniform" in msg
    assert "sim._boundary still read 'upml'" in msg
    # mechanism
    assert "use_cpml = grid.cpml_layers > 0" in msg
    assert "rfx/nonuniform.py" in msg
    # cost
    assert "differ in reflection" in msg
    # alternative
    assert "boundary='cpml'" in msg
    assert "uniform lane" in msg
    # falsifier
    assert "stale if" in msg
    assert "apply_upml_* call site appears outside rfx/simulation.py" in msg
    # and the issue it came from
    assert "#680" in msg


def test_skip_preflight_does_not_bypass_the_lane_guard():
    """The advisory's own claim about skip_preflight, pinned."""
    with pytest.raises(ValueError, match=r"boundary='upml' does not support"):
        _sim("upml", dz_profile=DZ).run(n_steps=4, skip_preflight=True)
