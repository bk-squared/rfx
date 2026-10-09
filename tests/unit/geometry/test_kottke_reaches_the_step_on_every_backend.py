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


def _told_it_is_on_a_gpu(monkeypatch):
    import jax
    import rfx.simulation as simulation
    built = []
    original = simulation.precompute_coeffs

    def spy(*args, **kwargs):
        built.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(simulation, "precompute_coeffs", spy)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    return built


def test_the_occupancy_tensor_of_forward_is_not_dropped_either(monkeypatch):
    """forward() with RFX_PEC_OCC_KOTTKE=1 turns the occupancy into the same
    inverse tensor and hands the step no occupancy, so nothing else kept it
    off the baked path: the CONDUCTOR was dropped (probe 0.79 of its peak
    away from the general path)."""
    import jax.numpy as jnp

    def trace(occupancy):
        sim = Simulation(freq_max=20e9, domain=(0.024, 0.020, 0.016), dx=1e-3,
                         boundary="pec", cpml_layers=0)
        sim.add_source((0.004, 0.005, 0.004), "ez", amplitude_kind="current")
        sim.add_probe((0.019, 0.014, 0.011), "ez")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.forward(n_steps=300, pec_occupancy_override=occupancy,
                                 skip_preflight=True, checkpoint=False)
        return np.asarray(result.time_series)[:, 0].astype(np.float64), sim

    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "1")
    empty, sim = trace(None)
    occ = np.zeros(sim._build_grid().shape, np.float32)
    occ[10:14, 8:12, 6:10] = 0.7
    occ[11:13, 9:11, 7:9] = 1.0
    general, _ = trace(jnp.asarray(occ))
    built = _told_it_is_on_a_gpu(monkeypatch)
    told, _ = trace(jnp.asarray(occ))
    assert not built, "the occupancy tensor took the baked scalar-coefficient path"
    peak = np.max(np.abs(empty))
    assert np.max(np.abs(general - empty)) / peak > 1e-2      # the conductor matters here
    assert np.array_equal(told, general)


def test_a_waveguide_port_is_not_dropped_by_the_baked_path(monkeypatch):
    """Reachable only through rfx.simulation.run (the public port requires an
    absorber), but the same hand-kept exclusion list: the port's E and H
    corrections sit outside the baked step, so the port radiated nothing."""
    import jax.numpy as jnp
    import rfx.simulation as simulation
    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.sources.waveguide_port import WaveguidePort, init_waveguide_port

    def peak():
        grid = Grid(freq_max=12e9, domain=(0.060, 0.02286, 0.01016), dx=1.27e-3, cpml_layers=0)
        port = WaveguidePort(x_index=8, y_slice=(0, grid.shape[1]), z_slice=(0, grid.shape[2]),
                             a=0.02286, b=0.01016, mode=(1, 0), mode_type="TE", direction="+x")
        cfg = init_waveguide_port(port, grid.dx, jnp.linspace(8e9, 11e9, 5), f0=10e9,
                                  dft_total_steps=200, dt=grid.dt, grid=grid)
        probe = simulation.make_probe(grid, (0.030, 0.0114, 0.005), "ez")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = simulation.run(grid, init_materials(grid.shape), 200, boundary="pec",
                                    waveguide_ports=[cfg], probes=[probe])
        return float(np.max(np.abs(np.asarray(result.time_series))))
    general = peak()
    built = _told_it_is_on_a_gpu(monkeypatch)
    told = peak()
    assert general > 0
    assert not built and told == general
