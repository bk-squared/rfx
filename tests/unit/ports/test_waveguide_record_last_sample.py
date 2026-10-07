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
