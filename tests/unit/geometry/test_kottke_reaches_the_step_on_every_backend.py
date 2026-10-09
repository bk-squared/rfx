"""`subpixel_smoothing='kottke_pec'` must reach the E update on every backend.

In a PEC-walled box with no absorber, no periodic axis and no PEC shape, the
GPU step may take the baked H+E fast path, whose coefficients are built from
the scalar realized permittivity. Its eligibility test looked at the Stage-1
tensor (``aniso_eps``) but not at the inverse tensor ``kottke_pec`` builds, so
the smoothing could be dropped there with no message. The invariant below
holds on any backend: the smoothed run differs from the unsmoothed one, and
the two smoothing stages agree with each other far better than either agrees
with the staircase.
"""
import warnings

import numpy as np
import pytest

from rfx import Simulation, Sphere



def _trace(smoothing):
    sim = Simulation(freq_max=20e9, domain=(0.024, 0.020, 0.016), dx=1e-3,
                     boundary="pec", cpml_layers=0)
    sim.add_material("ball", eps_r=6.0)
    # centre and radius off the lattice: every surface cell is partly filled
    sim.add(Sphere((0.0117, 0.0103, 0.0081), 0.00437), material="ball")
    sim.add_source((0.004, 0.005, 0.004), "ez", amplitude_kind="current")
    sim.add_probe((0.019, 0.014, 0.011), "ez")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=400, subpixel_smoothing=smoothing,
                         compute_s_params=False, skip_preflight=True)
    return np.asarray(result.time_series)[:, 0].astype(np.float64)


def _judge(plain, stage1, kottke):
    peak = np.max(np.abs(plain))
    assert peak > 0
    moved = np.max(np.abs(kottke - plain)) / peak
    between = np.max(np.abs(kottke - stage1)) / peak
    stage1_moved = np.max(np.abs(stage1 - plain)) / peak
    # CPU reference: stage 1 moves the probe trace by 0.1148 of its peak,
    # kottke_pec by 0.1148, and the two differ by 3.0e-5. A dropped tensor
    # gives moved == 0 exactly.
    assert stage1_moved > 1e-2, stage1_moved
    assert moved > 1e-2, (moved, between, stage1_moved)
    assert between < 0.5 * moved, (moved, between, stage1_moved)


@pytest.mark.gpu_gate
def test_kottke_smoothing_changes_the_field_and_agrees_with_stage_one():
    """On whatever backend this runs (the merge train runs it on a GPU)."""
    _judge(_trace(False), _trace(True), _trace("kottke_pec"))


def test_the_baked_fast_path_is_not_taken_with_an_inverse_tensor(monkeypatch):
    """The same judge with the step told it is on a GPU: runs in the CPU lane.

    Only the backend QUERY is replaced, so the fast-path eligibility takes its
    GPU branch while the arithmetic stays on this machine. Before the fix the
    kottke run built the baked scalar coefficients and returned the unsmoothed
    trace bit for bit.
    """
    import jax
    import rfx.simulation as simulation
    built = []
    original = simulation.precompute_coeffs

    def spy(*args, **kwargs):
        built.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(simulation, "precompute_coeffs", spy)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    plain = _trace(False)
    assert built, "control: the plain run takes the baked fast path here"
    built.clear()
    kottke = _trace("kottke_pec")
    assert not built, "kottke_pec took the baked scalar-coefficient path"
    built.clear()
    stage1 = _trace(True)
    assert not built
    _judge(plain, stage1, kottke)
