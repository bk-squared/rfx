"""A waveguide port's time record does not depend on how long the run is.

M2 moves retained slots 1:N to 0:N-1 with their physical stamps. The
historical exclusion of the final completed step is unchanged: slot N-1
is now the empty padding slot, and a write for completed step N is dropped.
"""

import numpy as np

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def _guide():
    kw = {}
    sim = Simulation(
        freq_max=12.4e9, domain=(0.06, 0.02286, 0.01016), dx=1e-3,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=8, **kw)
    sim.add_waveguide_port(0.012, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=np.linspace(8.2e9, 12.4e9, 3), f0=10.3e9,
                           bandwidth=0.5, name="wg1")
    return sim


def test_out_of_range_step_leaves_the_record_unchanged():
    from rfx.sources.waveguide_port import update_waveguide_port_probe
    from rfx.core.yee import init_state

    sim = _guide()
    result = sim.run(n_steps=40)
    cfg = next(iter(result.waveguide_ports.values()))
    n_t = cfg.v_probe_t.shape[0]
    grid = result.grid
    state = init_state((grid.nx, grid.ny, grid.nz))._replace(step=n_t)
    out = update_waveguide_port_probe(cfg, state, grid.dt, grid.dx)
    # M2: same samples, new addresses; the last retained slot moved from N-1 to N-2.
    assert np.any(np.asarray(cfg.v_ref_t)[-2:-1] != 0) or np.any(np.asarray(cfg.v_probe_t)[-2:-1] != 0)
    assert np.asarray(cfg.v_ref_t)[-1] == np.asarray(cfg.v_probe_t)[-1] == 0
    for name in ("v_probe_t", "v_ref_t", "i_probe_t", "i_ref_t", "v_inc_t"):
        np.testing.assert_array_equal(np.asarray(getattr(out, name)),
                                      np.asarray(getattr(cfg, name)), err_msg=name)
    assert int(out.n_steps_recorded) <= n_t


def test_a_source_still_on_at_the_end_is_not_read_as_ended():
    """The unwritten last slot is not a sample: the count excludes it, and a
    reader of the raw record sees a drive that is still on, not one that fell
    to zero on the last step."""
    from rfx.measurement.modal import recorded
    from rfx.probes.settling import source_end_step
    from rfx.sources.waveguide_port import settling_db_from_port_records

    sim = _guide()
    result = sim.run(n_steps=40)       # the pulse is still rising at step 40
    cfg = next(iter(result.waveguide_ports.values()))
    n_t = cfg.v_inc_t.shape[0]
    assert int(cfg.n_steps_recorded) == n_t - 1
    drive = np.asarray(recorded(cfg, "v_inc_t"))
    assert drive.shape == (n_t - 1,) and drive[-1] != 0
    assert source_end_step([(drive, 0.0)], n_t, cfg.dt) is None
    _, detail = settling_db_from_port_records([cfg], return_detail=True)
    assert detail["status"] == "undetermined"
    assert "source end is unavailable" in detail["reason"], detail["reason"]


def test_before_first_step_leaves_every_record_and_count_unchanged():
    from rfx.sources.waveguide_port import update_waveguide_port_probe
    from rfx.core.yee import init_state

    result = _guide().run(n_steps=2)
    cfg = next(iter(result.waveguide_ports.values()))
    grid = result.grid
    state = init_state((grid.nx, grid.ny, grid.nz))._replace(step=0)
    # Nonzero fields make an erroneous last-slot V/I write observable too.
    state = state._replace(ey=state.ey + 1, hz=state.hz + 1)
    names = ('v_probe_t', 'v_ref_t', 'i_probe_t', 'i_ref_t', 'v_inc_t')
    before = {name: np.asarray(getattr(cfg, name)).copy() for name in names}
    count = np.asarray(cfg.n_steps_recorded).copy()
    out = update_waveguide_port_probe(cfg, state, grid.dt, grid.dx)
    for name in names:
        np.testing.assert_array_equal(np.asarray(getattr(out, name)), before[name], err_msg=name)
    np.testing.assert_array_equal(np.asarray(out.n_steps_recorded), count)


def test_the_first_completed_step_writes_slot_zero():
    """Step 1 is the first sample and lands in slot 0 (a guard of `> 1` would drop it)."""
    from rfx.sources.waveguide_port import update_waveguide_port_probe
    from rfx.core.yee import init_state

    result = _guide().run(n_steps=2)
    cfg = next(iter(result.waveguide_ports.values()))
    grid = result.grid
    blank = cfg._replace(v_inc_t=cfg.v_inc_t * 0, n_steps_recorded=cfg.n_steps_recorded * 0)
    state = init_state((grid.nx, grid.ny, grid.nz))._replace(step=1)
    out = update_waveguide_port_probe(blank, state, grid.dt, grid.dx)
    arg = (float(grid.dt) - float(cfg.src_t0)) / float(cfg.src_tau)
    expected = float(cfg.src_amp) * (-2.0 * arg) * np.exp(-(arg ** 2))    # the drive at t = 1 dt
    assert expected != 0
    np.testing.assert_allclose(np.asarray(out.v_inc_t)[0], expected, rtol=1e-5)
    assert int(out.n_steps_recorded) == 1
