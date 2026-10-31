"""Traced dz must not silently cut a ground at an absorbing face.

The clear Box's traced and concrete records share the per-step cross-trace
bar: 9 float32 ULP at the record peak. Expected faces come from the drawing,
without continuation helpers.
"""

import warnings

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._traced_refusal import TracedStructureRefusal
from rfx.core.jax_utils import is_tracer
from rfx.model import NominalProfile
from rfx.sources import GaussianPulse

DX = 1 / 1024
PROFILE = np.full(8, DX)


def _scene(dz, reaching, near_face=False):
    sim = Simulation(domain=(8*DX, 8*DX, 8*DX), dx=DX, freq_max=20e9,
                     boundary="cpml", cpml_layers=2, dz_profile=dz,
                     nominal_profiles={"z": PROFILE} if is_tracer(dz) else None,
                     dt=1e-12, dt_min_cell=DX)
    sim.add(Box((0. if reaching else 2*DX, 2*DX, 2*DX),
                (6*DX, 6*DX, (7.9 if near_face else 3)*DX)), material="pec")
    sim.add_port((4*DX, 4*DX, 4*DX), component="ez",
                 waveform=GaussianPulse(f0=10e9, bandwidth=1.))
    sim.add_probe((4*DX, 4*DX, 5*DX), "ez")
    return sim


def _record(dz, reaching, near_face=False):
    return _scene(dz, reaching, near_face).forward(
        n_steps=12, checkpoint=False, skip_preflight=True).time_series


def test_reaching_box_refused_before_stepping(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a conductor cut at the absorber reached stepping")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(TracedStructureRefusal) as caught:
        jax.jit(lambda dz: _record(dz, True))(PROFILE)
    message = str(caught.value)
    for expected in ("x-lo", "Cannot continue", "ground cut at the absorber entrance",
                     "reaches or ends within one cell of", "mesh axis z is traced",
                     "dz_profile", "concrete NumPy array", "for this run"):
        assert expected in message
    assert "dx_profile" not in message
    assert "dy_profile" not in message
    assert "#" not in message


def test_clear_box_record_matches_concrete_without_continuation_warning():
    concrete = np.asarray(_record(PROFILE, False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        traced = np.asarray(jax.jit(lambda dz: _record(dz, False))(PROFILE))
    assert not any("conductor continuation" in str(w.message) for w in caught)
    assert np.any(traced != 0)
    peak = np.max(np.abs(concrete))
    difference = np.max(np.abs(traced.astype(np.float64) - concrete))
    ulp = float(np.spacing(np.float32(peak)))
    print(f"clear Box record: max difference = {difference / ulp:.9g} float32 ULP at peak")
    assert difference <= 9 * ulp


def test_concrete_reaching_box_still_runs_and_continues():
    from rfx.runners.nonuniform import assemble_materials_nu

    sim = _scene(PROFILE, True)
    grid = sim._build_nonuniform_grid()
    pec = np.asarray(assemble_materials_nu(sim, grid, pec_sheets=[], pec_wires=[])[3])
    # x-lo has two absorber layers; y=3*DX, z=2*DX are inside the Box.
    assert pec[:2, 5, 4].all()
    assert np.isfinite(np.asarray(_record(PROFILE, True))).all()


def test_box_ending_one_tenth_cell_short_on_traced_axis_refused(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a conductor ending in the outermost cell reached stepping")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(TracedStructureRefusal, match="absorbing face\\(s\\) z-hi"):
        jax.jit(lambda dz: _record(dz, False, near_face=True))(PROFILE)


@pytest.mark.parametrize("nominal", [False, True], ids=["no_nominal", "nominal_edge"])
def test_e_three_dx_edge_cell_refuses_box_at_one_point_two_dx(nominal, monkeypatch):
    # z-lo is 1.2*DX from the face, inside its 3*DX edge cell.
    def forbidden(*args, **kwargs):
        pytest.fail("the Box inside the 3*DX edge cell reached stepping")

    monkeypatch.setattr(jax.lax, "scan", forbidden)

    @jax.jit
    def traced(dz):
        sim = Simulation(domain=(8*DX, 8*DX, 24*DX), dx=DX, freq_max=20e9,
                         boundary="cpml", cpml_layers=2, dz_profile=dz,
                         nominal_profiles={"z": np.full(8, 3*DX)} if nominal else None,
                         dt=1e-12, dt_min_cell=DX)
        sim.add(Box((2*DX, 2*DX, 1.2*DX), (6*DX, 6*DX, 12*DX)), material="pec")
        return sim.forward(n_steps=1, checkpoint=False, skip_preflight=True).time_series

    with pytest.raises(TracedStructureRefusal) as caught:
        traced(np.full(8, 3*DX))
    message = str(caught.value)
    if nominal:
        assert "absorbing face(s) z-lo" in message
        assert "within one cell" in message
        assert "nominal_profiles" not in message
    else:
        assert "Declare nominal_profiles for axis z, or keep it concrete" in message


def test_f_windowed_nominal_profile_allows_clear_box():
    from rfx.geometry.smoothing import continued_conductor_shape

    # Each bound is three DX from its face, beyond the one-DX edge cells.
    box = Box((3*DX, 3*DX, 3*DX), (5*DX, 5*DX, 5*DX))

    @jax.jit
    def traced(dx):
        sim = Simulation(domain=(8*DX, 8*DX, 8*DX), dx=DX, freq_max=20e9,
                         boundary="cpml", cpml_layers=2, dx_profile=dx,
                         nominal_profiles={"x": NominalProfile(np.ones(8)*DX, traced_cells=(2, 6))},
                         dt=1e-12, dt_min_cell=DX)
        assert continued_conductor_shape(sim, sim._build_nonuniform_grid(), box) is box
        return dx

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np.testing.assert_array_equal(traced(PROFILE), PROFILE)
    assert not any("conductor continuation" in str(w.message) for w in caught)
