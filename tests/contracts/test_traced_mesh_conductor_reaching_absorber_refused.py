"""Traced dz must not silently cut a ground at an absorbing face.

The clear Box's record is compared bit for bit with main's traced run at
1f88121fc (CPU, float32). Expected faces are read from the drawing, without
continuation helpers. The reference was captured before applying the change.
"""

import warnings

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._traced_refusal import TracedStructureRefusal
from rfx.sources import GaussianPulse

DX = 1 / 1024
PROFILE = np.full(8, DX)
MAIN_TRACED_RECORD_BITS = np.array([
    0, 0, 0, 3121557267, 3145517095, 3161232885, 3172858286, 3182021456,
    3189564386, 3196189601, 3200701867, 3205454082,
], dtype=np.uint32).reshape(12, 1)


def _scene(dz, reaching):
    sim = Simulation(domain=(8*DX, 8*DX, 8*DX), dx=DX, freq_max=20e9,
                     boundary="cpml", cpml_layers=2, dz_profile=dz,
                     dt=1e-12, dt_min_cell=DX)
    sim.add(Box((0. if reaching else 2*DX, 2*DX, 2*DX),
                (6*DX, 6*DX, 3*DX)), material="pec")
    sim.add_port((4*DX, 4*DX, 4*DX), component="ez",
                 waveform=GaussianPulse(f0=10e9, bandwidth=1.))
    sim.add_probe((4*DX, 4*DX, 5*DX), "ez")
    return sim


def _record(dz, reaching):
    return _scene(dz, reaching).forward(
        n_steps=12, checkpoint=False, skip_preflight=True).time_series


def test_reaching_box_refused_before_stepping(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a conductor cut at the absorber reached stepping")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(TracedStructureRefusal) as caught:
        jax.jit(lambda dz: _record(dz, True))(PROFILE)
    message = str(caught.value)
    for expected in ("x-lo", "Cannot continue", "ground cut at the absorber entrance",
                     "dz_profile", "concrete NumPy array", "for this run"):
        assert expected in message
    assert "#" not in message


def test_clear_box_record_equals_main_traced_without_continuation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        traced = np.asarray(jax.jit(lambda dz: _record(dz, False))(PROFILE))
    assert not any("conductor continuation" in str(w.message) for w in caught)
    assert np.any(traced != 0)
    np.testing.assert_array_equal(traced.view(np.uint32), MAIN_TRACED_RECORD_BITS)


def test_concrete_reaching_box_still_runs_and_continues():
    from rfx.runners.nonuniform import assemble_materials_nu

    sim = _scene(PROFILE, True)
    grid = sim._build_nonuniform_grid()
    pec = np.asarray(assemble_materials_nu(sim, grid, pec_sheets=[], pec_wires=[])[3])
    # x-lo has two absorber layers; y=3*DX, z=2*DX are inside the Box.
    assert pec[:2, 5, 4].all()
    assert np.isfinite(np.asarray(_record(PROFILE, True))).all()
