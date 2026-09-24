"""A subgrid refinement on a non-uniform mesh is refused, not dropped (#1240).

``add_refinement(z_range, ratio)`` asks for a finer mesh in a z slab. The
non-uniform lane has no subgrid and never read the refinement: on main
7b2fc25d the issue's 12 mm PEC box (1 mm cells, 60 steps) gave a probe
bit-identical to the unrefined box (peak 1.079170e6 both), with no warning and
a preflight that passed, while the same refinement on a uniform mesh moves the
peak to 6.611695e4. The same drop held for ``forward()``, for
``forward(distributed=True)`` and for a mesh that auto-meshing made
non-uniform.

The refusal is decided on the RESOLVED mesh (``_uses_nonuniform_mesh``, the
property ``_dispatch_plan`` picks the lane with) and sits in three places:

* ``preflight()``: an error finding, code ``nonuniform_refinement``, so
  ``run()``/``forward()`` stop on it by default;
* ``_dispatch_plan``: every non-uniform lane, including the distributed ones;
* ``run_nonuniform_path``: the lane itself, which the graded branch of
  ``compute_waveguide_s_matrix`` reaches without ``_dispatch_plan``.

Only ``run()`` has a subgridded lane, and on a UNIFORM mesh the other
entry points dropped a refinement the same way: ``forward()`` (bit-identical,
peak 1.079171e6 both), ``topology_optimize()`` (bit-identical loss history),
the ``vmap_material_sweep()`` batched kernel and the uniform
``compute_waveguide_s_matrix()`` scan (both bit-identical). Each refuses it
now through one shared check. ``forward()`` and ``topology_optimize()`` both
enter the uniform forward lane (``_forward_from_materials``), which refuses it
once for both; ``optimize()`` goes through ``forward()``, and
``differentiable_material_fit()`` runs its own uniform scan.

The ADI lane never dropped a refinement: ``_validate_adi_configuration``
refuses it. Locked here too.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

N_STEPS = 60
_REFUSED = "subgridding on a non-uniform mesh"


def _box(refine, *, graded=True, validation="production", solver="yee",
         boundary="pec"):
    """The #1240 box: 12 mm, 1 mm cells, Ez current source and Ez probe."""
    kw = dict(freq_max=10e9, domain=(0.012, 0.012, 0.012), dx=1e-3,
              boundary=boundary, solver=solver)
    if graded:
        # One cell size, so the mesh is the uniform one; the profile alone
        # sends the model to the non-uniform lane, as in the issue.
        kw["dx_profile"] = np.full(12, 1e-3)
    sim = Simulation(**kw)
    sim.add_source((0.004, 0.006, 0.006), "ez",
                   waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
                   amplitude_kind="current")
    sim.add_probe((0.008, 0.006, 0.006), "ez")
    if refine:
        sim.add_refinement(z_range=(0.004, 0.008), ratio=2,
                           validation=validation)
    return sim


def _auto_graded(refine):
    """No dx= and a 0.2 mm slab: auto-meshing invents a dz_profile."""
    sim = Simulation(freq_max=15e9, domain=(0.020, 0.020, 0.004),
                     boundary="pec")
    sim.add_material("slab", eps_r=4.0)
    sim.add(Box((0.0, 0.0, 0.0), (0.020, 0.020, 0.0002)), material="slab")
    sim.add_source((0.006, 0.010, 0.002), "ez",
                   waveform=GaussianPulse(f0=7e9, bandwidth=0.8),
                   amplitude_kind="current")
    sim.add_probe((0.014, 0.010, 0.002), "ez")
    if refine:
        sim.add_refinement(z_range=(0.001, 0.003), ratio=2,
                           validation="research")
    return sim


def _assert_names_the_refusal(message):
    assert _REFUSED in message, message
    assert "no subgrid" in message, message
    assert "uniform base mesh" in message and "dz_profile" in message, message


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


# ---------------------------------------------------------------------------
# preflight()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("validation", ["production", "research", "off"])
def test_preflight_reports_the_refinement_as_an_error(validation):
    report = _quiet(lambda: _box(True, validation=validation).preflight())
    errors = [i for i in report if i.severity == "error"]
    assert [i.code for i in errors] == ["nonuniform_refinement"], list(report)
    _assert_names_the_refusal(str(errors[0]))


def test_preflight_is_silent_about_it_on_a_uniform_mesh():
    report = _quiet(lambda: _box(True, graded=False).preflight())
    assert "nonuniform_refinement" not in [i.code for i in report]


# ---------------------------------------------------------------------------
# run() and forward(), with and without preflight
# ---------------------------------------------------------------------------

def test_run_stops_on_the_preflight_error():
    with pytest.raises(ValueError, match="blocking error") as exc:
        _quiet(lambda: _box(True).run(n_steps=N_STEPS))
    _assert_names_the_refusal(str(exc.value))


def test_run_refuses_with_preflight_skipped():
    with pytest.raises(NotImplementedError) as exc:
        _quiet(lambda: _box(True).run(n_steps=N_STEPS, skip_preflight=True))
    _assert_names_the_refusal(str(exc.value))


def test_run_until_decay_refuses():
    sim = _box(True, boundary="cpml")
    with pytest.raises(NotImplementedError, match=_REFUSED):
        _quiet(lambda: sim.run(until_decay=1e-3, skip_preflight=True))


def test_run_with_wire_port_s_parameters_refuses():
    """A wire port is the one port the graded lane extracts S from."""
    sim = Simulation(freq_max=10e9, domain=(0.012, 0.012, 0.012), dx=1e-3,
                     boundary="pec", dx_profile=np.full(12, 1e-3))
    sim.add_port((0.004, 0.006, 0.005), "ez", impedance=50.0, extent=0.002,
                 waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
    sim.add_refinement(z_range=(0.004, 0.008), ratio=2)
    with pytest.raises(NotImplementedError, match=_REFUSED):
        _quiet(lambda: sim.run(n_steps=N_STEPS, compute_s_params=True,
                               skip_preflight=True))


def test_forward_refuses():
    with pytest.raises(ValueError, match=_REFUSED):
        _quiet(lambda: _box(True).forward(n_steps=N_STEPS))
    with pytest.raises(NotImplementedError, match=_REFUSED):
        _quiet(lambda: _box(True).forward(n_steps=N_STEPS,
                                          skip_preflight=True))


def test_distributed_forward_refuses():
    """The distributed graded forward lane never reaches run_nonuniform_path,
    so only the _dispatch_plan refusal covers it."""
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2, "requires the root conftest's two CPU devices"
    with pytest.raises(NotImplementedError, match=_REFUSED):
        _quiet(lambda: _box(True).forward(
            n_steps=N_STEPS, distributed=True, devices=devices,
            skip_preflight=True))


def test_distributed_run_names_the_graded_mesh():
    """#1241 refuses a refinement on the multi-device lanes; its message
    advises a single-device run, which on a graded mesh is refused too, so
    the graded-mesh reason comes first."""
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2, "requires the root conftest's two CPU devices"
    with pytest.raises(NotImplementedError, match=_REFUSED):
        _quiet(lambda: _box(True).run(n_steps=N_STEPS, devices=devices,
                                      skip_preflight=True))


# ---------------------------------------------------------------------------
# A mesh that only auto-meshing made non-uniform
# ---------------------------------------------------------------------------

def test_auto_meshed_graded_mesh_is_refused():
    sim = _auto_graded(True)
    assert _quiet(lambda: sim._uses_nonuniform_mesh), \
        "the fixture must resolve to the non-uniform lane"
    assert all(sim._declared_mesh[name] is None
               for name in ("_dx_profile", "_dy_profile", "_dz_profile")), \
        "the fixture must declare no profile"
    report = _quiet(sim.preflight)
    assert "nonuniform_refinement" in [i.code for i in report
                                       if i.severity == "error"]
    with pytest.raises(NotImplementedError, match=_REFUSED):
        _quiet(lambda: sim.run(n_steps=N_STEPS, skip_preflight=True))


# ---------------------------------------------------------------------------
# The graded waveguide S-matrix branch, which bypasses _dispatch_plan
# ---------------------------------------------------------------------------

def test_graded_waveguide_s_matrix_refuses():
    """WR-90 on a graded z mesh (the #811 fixture). compute_waveguide_s_matrix
    runs no preflight and calls run_nonuniform_path directly."""
    dz = np.concatenate([np.full(10, 0.40e-3), np.full(3, 0.52e-3),
                         np.full(2, 0.70e-3), np.full(4, 0.80e-3)])
    sim = Simulation(
        freq_max=12.4e9, domain=(0.06, 0.02286, 0.01016), dx=1e-3,
        dz_profile=dz,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=8)
    for x_position, direction, name in ((0.012, "+x", "wg1"),
                                        (0.048, "-x", "wg2")):
        sim.add_waveguide_port(
            x_position, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=np.linspace(8.2e9, 12.4e9, 3), f0=10.3e9, bandwidth=0.5,
            name=name)
    sim.add_refinement(z_range=(0.002, 0.006), ratio=2)
    with pytest.raises(NotImplementedError, match=_REFUSED):
        _quiet(lambda: sim.compute_waveguide_s_matrix(n_steps=1,
                                                      normalize="flux"))


# ---------------------------------------------------------------------------
# Entry points with no subgridded lane, on a uniform mesh
# ---------------------------------------------------------------------------

_NO_SUBGRID = "no subgridded lane"


def test_uniform_forward_refuses():
    with pytest.raises(NotImplementedError, match=_NO_SUBGRID) as exc:
        _quiet(lambda: _box(True, graded=False).forward(n_steps=N_STEPS))
    assert "takes effect in run(), on a uniform mesh" in str(exc.value)


def test_optimize_refuses():
    from rfx.optimize import DesignRegion, optimize

    region = DesignRegion(corner_lo=(0.005, 0.005, 0.005),
                          corner_hi=(0.007, 0.007, 0.007), eps_range=(1.0, 4.0))
    with pytest.raises(NotImplementedError, match=_NO_SUBGRID):
        _quiet(lambda: optimize(_box(True, graded=False), region,
                                lambda r: -jnp.sum(r.time_series ** 2),
                                n_iters=1, n_steps=8, verbose=False,
                                skip_preflight=True))


def test_topology_optimize_refuses():
    """It enters the forward lane without forward() or _dispatch_plan."""
    from rfx.topology import TopologyDesignRegion, topology_optimize

    region = TopologyDesignRegion(corner_lo=(0.005, 0.005, 0.005),
                                  corner_hi=(0.007, 0.007, 0.007),
                                  material_bg="air", material_fg="fr4",
                                  beta_projection=1.0)
    with pytest.raises(NotImplementedError, match=_NO_SUBGRID):
        _quiet(lambda: topology_optimize(
            _box(True, graded=False), region,
            lambda r: -jnp.sum(r.time_series ** 2), n_iterations=1,
            learning_rate=0.05, beta_schedule=[(0, 1.0)], verbose=False,
            skip_preflight=True))


def test_uniform_vmap_material_sweep_refuses():
    """The batched kernel; the graded fallback goes through run() instead."""
    from rfx.vmap_sweep import vmap_material_sweep

    with pytest.raises(NotImplementedError, match=_NO_SUBGRID):
        _quiet(lambda: vmap_material_sweep(_box(True, graded=False), "eps_r",
                                           [1.0, 2.0], n_steps=8))


def _lumped_port_box(refine, eps_r=2.0):
    """A lumped port sends vmap_material_sweep to its sequential run() path.

    12 mm PEC box, 1 mm cells, a 2 mm substrate on the floor, 50 ohm port;
    the refinement covers z = 0 to 8 mm.
    """
    sim = Simulation(freq_max=10e9, domain=(0.012, 0.012, 0.012), dx=1e-3,
                     boundary="pec")
    sim.add_material("diel", eps_r=eps_r)
    sim.add(Box((0.0, 0.0, 0.0), (0.012, 0.012, 0.002)), material="diel")
    sim.add_port((0.004, 0.006, 0.004), "ez", impedance=50.0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
    sim.add_probe((0.008, 0.006, 0.004), "ez")
    if refine:
        sim.add_refinement(z_range=(0.0, 0.008), ratio=2)
    return sim


def _sweep_trace(**kw):
    from rfx.vmap_sweep import vmap_material_sweep

    result = _quiet(lambda: vmap_material_sweep(
        _lumped_port_box(True), "diel.eps_r", [3.0], **kw))
    return np.asarray(result.time_series)[0]


def _run_trace(refine, **kw):
    return np.asarray(_quiet(lambda: _lumped_port_box(refine, eps_r=3.0)
                             .run(**kw).time_series))


def test_vmap_sequential_fallback_still_solves_the_refinement():
    """The fallback calls run(), which has the subgridded lane: not refused,
    and each swept value is the refined run() of that value."""
    swept = _sweep_trace(n_steps=N_STEPS)
    direct = _run_trace(True, n_steps=N_STEPS)
    np.testing.assert_allclose(swept, direct, rtol=1e-6, atol=0.0)
    unrefined = _run_trace(False, n_steps=N_STEPS)
    assert not np.allclose(swept, unrefined, rtol=0.1), \
        "the swept trace must be the refined solve, not the coarse one"


def test_vmap_sequential_fallback_covers_the_duration_run_covers():
    """With n_steps left to num_periods, the fallback's trace spans the same
    time as run()'s: run() resolves the count on the subgridded lane (fine
    steps), where a coarse count would cover 1/ratio of it."""
    swept = _sweep_trace(num_periods=2.0)
    direct = _run_trace(True, num_periods=2.0)
    assert swept.shape == direct.shape, (swept.shape, direct.shape)
    np.testing.assert_allclose(swept, direct, rtol=1e-6, atol=0.0)


def test_vmap_sequential_fallback_stacks_an_auto_meshed_sweep():
    """No refinement, no dx=: auto-meshing reads the swept permittivity, so
    each value resolves its own cell and step count (eps_r 2 and 3 gave 105
    and 140 steps at 2 periods). The sweep hands every copy the base model's
    count, so the traces stack as on main instead of failing to."""
    from rfx.vmap_sweep import vmap_material_sweep

    def model():
        sim = Simulation(freq_max=10e9, domain=(0.012, 0.012, 0.012),
                         boundary="pec")
        sim.add_material("diel", eps_r=2.0)
        sim.add(Box((0.0, 0.0, 0.0), (0.012, 0.012, 0.004)), material="diel")
        sim.add_port((0.004, 0.006, 0.006), "ez", impedance=50.0,
                     waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
        sim.add_probe((0.008, 0.006, 0.006), "ez")
        return sim

    n_base = _quiet(lambda: model()._build_grid().num_timesteps(
        num_periods=2.0))
    result = _quiet(lambda: vmap_material_sweep(model(), "eps_r", [2.0, 3.0],
                                                num_periods=2.0))
    assert np.asarray(result.time_series).shape == (2, n_base, 1)


def test_lumped_wire_scan_driver_refuses():
    """It calls the uniform forward lane directly, without forward(); the
    lane's own refusal covers it."""
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan

    with pytest.raises(NotImplementedError, match=_NO_SUBGRID):
        _quiet(lambda: compute_lumped_wire_s_matrix_via_scan(
            _lumped_port_box(True), np.linspace(3e9, 7e9, 3), n_steps=8))


def test_waveguide_port_reference_model_with_a_refinement_refuses():
    """A per-port reference model is built on the uniform grid too."""
    def two_port(refine):
        sim = Simulation(freq_max=10e9, domain=(0.12, 0.04, 0.02),
                         boundary="cpml", cpml_layers=10, dx=0.004)
        common = dict(mode=(1, 0), mode_type="TE",
                      freqs=np.linspace(4.5e9, 8e9, 3), f0=6e9,
                      ref_offset=3, probe_offset=8)
        sim.add_waveguide_port(0.01, direction="+x", name="a", **common)
        sim.add_waveguide_port(0.11, direction="-x", name="b", **common)
        if refine:
            sim.add_refinement(z_range=(0.006, 0.014), ratio=2)
        return sim

    refs = [two_port(True), two_port(False)]
    with pytest.raises(NotImplementedError, match=_NO_SUBGRID):
        _quiet(lambda: two_port(False).compute_waveguide_s_matrix(
            n_steps=10, normalize="flux", port_reference_sims=refs))


def test_direct_run_uniform_refuses():
    """rfx.runners.run_uniform is exported; run() never sends it a refined
    model, a direct call must not solve one without the refinement."""
    from rfx.runners import run_uniform

    with pytest.raises(NotImplementedError, match=_NO_SUBGRID):
        _quiet(lambda: run_uniform(_box(True, graded=False), n_steps=8))


def test_uniform_waveguide_s_matrix_refuses():
    sim = Simulation(
        freq_max=8e9, domain=(0.06, 0.02286, 0.01016), dx=3e-3,
        boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=8)
    for x_position, direction, name in ((0.012, "+x", "wg1"),
                                        (0.048, "-x", "wg2")):
        sim.add_waveguide_port(
            x_position, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=np.linspace(6.6e9, 7.8e9, 3), f0=7.2e9, bandwidth=0.3,
            name=name)
    sim.add_refinement(z_range=(0.003, 0.007), ratio=2)
    with pytest.raises(NotImplementedError, match=_NO_SUBGRID):
        _quiet(lambda: sim.compute_waveguide_s_matrix(n_steps=1))


def test_differentiable_material_fit_refuses():
    from rfx.differentiable_material_fit import differentiable_material_fit

    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = Simulation(freq_max=5e9, domain=(0.024, 0.009, 0.009),
                         dx=1e-3, boundary="pec")
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
        sim.add(Box((0.010, 0.0, 0.0), (0.016, 0.009, 0.009)),
                material="dut")
        sim.add_port((0.003, 0.0045, 0.0045), "ez",
                     waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
        sim.add_probe((0.020, 0.0045, 0.0045), component="ez")
        sim.add_refinement(z_range=(0.003, 0.006), ratio=2)
        return sim

    with pytest.raises(NotImplementedError, match=_NO_SUBGRID):
        _quiet(lambda: differentiable_material_fit(
            factory, np.zeros((1, 1, 3), complex),
            np.linspace(2.0e9, 4.0e9, 3), n_debye_poles=1,
            n_iterations=1, learning_rate=0.0, verbose=False))


# ---------------------------------------------------------------------------
# ADI: refused before #1240, locked
# ---------------------------------------------------------------------------

def test_adi_refuses_a_refinement():
    with pytest.raises(ValueError, match="does not support subgridding"):
        _quiet(lambda: _box(True, graded=False, solver="adi").run(
            n_steps=N_STEPS, skip_preflight=True))


# ---------------------------------------------------------------------------
# Control: the uniform mesh still runs the subgrid
# ---------------------------------------------------------------------------

def test_uniform_mesh_refinement_still_reaches_the_subgridded_lane():
    """The issue's table, uniform row: the refinement changes the probe.
    Measured on main 7b2fc25d: peak 1.079172e6 unrefined, 6.611695e4 refined."""
    def peak(refine):
        ts = _quiet(lambda: _box(refine, graded=False, validation="research")
                    .run(n_steps=N_STEPS).time_series)
        return float(np.max(np.abs(np.asarray(ts))))

    unrefined, refined = peak(False), peak(True)
    assert abs(refined - unrefined) > 0.5 * unrefined, (unrefined, refined)
