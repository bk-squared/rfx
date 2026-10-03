"""A waveguide port's time record does not depend on how long the run is.

The E update advances ``state.step`` before the port samples its modal V/I,
so the last scan step addresses one slot past the record. That write is
dropped; clamping it into the last slot used to put the NEXT step's sample
there, so the last sample of a record was a sample one step later than its slot.
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
    assert np.any(np.asarray(cfg.v_ref_t)[-1:] != 0) or np.any(np.asarray(cfg.v_probe_t)[-1:] != 0)
    for name in ("v_probe_t", "v_ref_t", "i_probe_t", "i_ref_t", "v_inc_t"):
        np.testing.assert_array_equal(np.asarray(getattr(out, name)),
                                      np.asarray(getattr(cfg, name)), err_msg=name)
    assert int(out.n_steps_recorded) <= n_t
