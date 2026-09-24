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

The ADI lane never dropped a refinement: ``_validate_adi_configuration``
refuses it. Locked here too.
"""

from __future__ import annotations

import warnings

import jax
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
