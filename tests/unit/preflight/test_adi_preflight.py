"""Preflight advisory for the 3D ADI large-timestep accuracy envelope.

HISTORY (W3.8 / OPT-C1): until 2026-07-13 ``adi_step_3d`` was an LOD split
with artificial diffusion and the ``adi_3d_accuracy`` warning flagged the
whole 3D path as KNOWN-INACCURATE unconditionally. The scheme is now the
full Zheng–Chen–Zhang two-sub-step 3D ADI (issue #338 follow-up); the
adjudication test
``test_review_tier1_validation_battery.py::test_optc1_adi_3d_cavity_eigenfrequency``
passes its 2% gate at 2x CFL (measured 1.2%) and its former strict-xfail
marker is removed.

What this file now locks is the ENVELOPE advisory that replaced the old
blanket warning: dispersion error of the implicit scheme grows ~dt^2, so at
~15 cells/wavelength the <2% eigenfrequency envelope holds only up to ~2x
CFL. The ``adi_3d_accuracy`` warning must fire on ``solver='adi'`` + a 3D
grid when ``adi_cfl_factor > 2`` (note the constructor DEFAULTS are
``mode='3d'`` and ``adi_cfl_factor=5.0``, so a bare ``Simulation(
solver='adi')`` is advised), and stay silent at ``adi_cfl_factor <= 2``, on
the validated 2D TMz path, and on the explicit Yee solver.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from rfx import Box, PolylineWire, Simulation


def _codes(issues):
    return {getattr(i, "code", "") for i in issues}


def test_adi_3d_large_cfl_fires_envelope_advisory():
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", mode="3d", solver="adi", adi_cfl_factor=5.0,
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" in _codes(issues), (
        f"3D ADI at 5x CFL must get the dt^2-envelope advisory; "
        f"issues: {issues!r}"
    )
    severities = {
        getattr(i, "code", ""): getattr(i, "severity", "")
        for i in issues
    }
    # WARNING severity, not error — large-dt 3D ADI is a legitimate stiff-
    # mesh tool; only wavelength-scale accuracy degrades.
    assert severities["adi_3d_accuracy"] == "warning"


def test_adi_default_cfl_factor_fires_envelope_advisory():
    """Defaults are mode='3d' AND adi_cfl_factor=5.0 -> must still advise."""
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", solver="adi",
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" in _codes(issues)


def test_adi_3d_within_envelope_is_silent():
    """At adi_cfl_factor <= 2 the 3D scheme meets its 2% gate — no advisory
    (measured 1.2% on the tier-1 adjudication cavity at 2x CFL)."""
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", mode="3d", solver="adi", adi_cfl_factor=2.0,
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" not in _codes(issues), (
        f"3D ADI at <=2x CFL is inside the validated envelope and must NOT "
        f"warn; issues: {issues!r}"
    )


def test_adi_2d_tmz_is_silent():
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, 0.01),
        boundary="pec", mode="2d_tmz", solver="adi", adi_cfl_factor=5.0,
    )
    sim.add_source((0.01, 0.01, 0.0), "ez")
    sim.add_probe((0.012, 0.01, 0.0), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" not in _codes(issues), (
        f"validated 2D TMz ADI path must NOT warn; issues: {issues!r}"
    )


def test_yee_3d_is_silent():
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", mode="3d", solver="yee",
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" not in _codes(issues), (
        f"explicit Yee solver must NOT get the ADI advisory; issues: {issues!r}"
    )


# #931 P1: a finite 200-step trace at factor 1 is not a stability bound.
# The 4000-step sheet witness grows to 4.3e6 (3D) / 1.8e19 (TMz).
GUARD = "adi_interior_pec_unsupported"


def _conductor_sim(mode="3d", kind="sheet", **kwargs):
    sim = Simulation(freq_max=15e9, domain=(.02, .02, .02), dx=.001,
                     boundary="pec", solver="adi", mode=mode, **kwargs)
    if kind == "wire":
        sim.add(PolylineWire(((.010, .005, 0), (.010, .015, 0)),
                             radius=.0002), material="pec")
    else:
        thickness = {"sheet": 0, "volume1": .001, "volume3": .003}[kind]
        sim.add(Box((.004, .010, 0), (.016, .010 + thickness, .020)),
                material="pec")
    sim.add_source((.010, .005, 0), "ez")
    sim.add_probe((.010, .015, 0), "ez")
    return sim


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("kind", ["sheet", "volume1", "volume3", "wire"])
def test_default_adi_conductor_is_named_error_and_cannot_solve(mode, kind, monkeypatch):
    """No factor argument: pin the shipped default, including skipped preflight."""
    sim = _conductor_sim(mode, kind)
    assert sim._adi_cfl_factor == 5.0
    issues = [i for i in sim.preflight() if i.code == GUARD]
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "solver='yee'" in issues[0]
    with pytest.raises(ValueError, match=GUARD):
        sim.preflight(strict=True)

    def forbidden_solve(*args, **kwargs):
        pytest.fail("ADI solve started before the conductor refusal")

    monkeypatch.setattr("rfx.adi.run_adi_3d", forbidden_solve)
    monkeypatch.setattr("rfx.adi.run_adi_2d", forbidden_solve)
    for skip in (False, True):
        with pytest.raises(ValueError, match=GUARD):
            sim.run(n_steps=200, skip_preflight=skip)
    with pytest.raises(ValueError, match=GUARD):
        sim.forward(n_steps=200)


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("factor", [.5, 1., 2.])
def test_reducing_factor_is_not_a_supported_interior_pec_remedy(mode, factor):
    sim = _conductor_sim(mode, adi_cfl_factor=factor)
    assert GUARD in _codes(sim.preflight())
    with pytest.raises(ValueError, match=GUARD):
        sim.forward(n_steps=1)


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
def test_no_interior_pec_and_loss_only_material_do_not_get_conductor_error(mode):
    sim = Simulation(freq_max=15e9, domain=(.02, .02, .02), dx=.001,
                     boundary="pec", solver="adi", mode=mode)
    sim.add_source((.010, .005, 0), "ez")
    assert GUARD not in _codes(sim.preflight())
    sim.add_material("loss_only", eps_r=2, sigma=.1)
    sim.add(Box((.004, .010, 0), (.016, .013, .020)), material="loss_only")
    assert GUARD not in _codes(sim.preflight())


def test_custom_pec_material_uses_production_classification():
    sim = Simulation(freq_max=15e9, domain=(.02, .02, .02), dx=.001,
                     boundary="pec", solver="adi")
    sim.add_material("metal", eps_r=1, sigma=1e9)
    sim.add(Box((.004, .010, 0), (.016, .013, .020)), material="metal")
    sim.add_source((.010, .005, 0), "ez")
    assert GUARD in _codes(sim.preflight())


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("override", ["pec_mask_override", "pec_occupancy_override"])
def test_forward_call_time_conductors_cannot_bypass_guard(mode, override):
    sim = Simulation(freq_max=15e9, domain=(.02, .02, .02), dx=.001,
                     boundary="pec", solver="adi", mode=mode)
    sim.add_source((.010, .005, 0), "ez")
    mask = jnp.zeros(sim._build_grid().shape)
    if override == "pec_mask_override":
        mask = mask.astype(bool)
    pattern = GUARD if override == "pec_mask_override" else "pec_occupancy_override"
    # Even an empty supplied mask is rejected structurally under tracing.
    with pytest.raises(ValueError, match=pattern):
        sim.forward(n_steps=1, **{override: mask})
    with pytest.raises(ValueError, match=pattern):
        jax.jit(lambda m: sim.forward(n_steps=1, **{override: m}).time_series)(mask)


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
def test_low_level_adi_refuses_masks_before_scan_or_step(mode):
    from rfx.adi import adi_step_2d, adi_step_3d, run_adi_2d, run_adi_3d
    is_3d = mode == "3d"
    a = jnp.zeros((4, 4, 4) if is_3d else (4, 4))
    mask = a.astype(bool)
    kwargs = {"pec_edge_masks": (mask, mask, mask)} if is_3d else {"ez_pec_mask": mask}
    args = [a] * (6 if is_3d else 3) + [jnp.ones_like(a), a, 1e-12]
    args += [.001] * (3 if is_3d else 2)
    step = adi_step_3d if is_3d else adi_step_2d
    run = run_adi_3d if is_3d else run_adi_2d
    with pytest.raises(ValueError, match=GUARD):
        step(*args, **kwargs)
    with pytest.raises(ValueError, match=GUARD):
        run(*args, n_steps=0, **kwargs)


def test_accuracy_advisory_scopes_the_stability_guarantee():
    sim = _conductor_sim()
    accuracy = next(i for i in sim.preflight() if i.code == "adi_3d_accuracy")
    assert "homogeneous, lossless" in accuracy
    assert "interior PEC projection (refused)" in accuracy
    assert "unconditionally stable" not in accuracy
