"""Guard: run(compute_s_params=True) for lumped/wire ports rejects periodic axes (#206).

The lumped/wire S-parameter extractor runs a SEPARATE eager FDTD re-run
(rfx/probes/probes.py extract_s_matrix / extract_s_matrix_wire) that does NOT
apply periodic boundaries, while the main run does. So a sim with periodic axes
+ a lumped/wire port + compute_s_params=True would silently extract S-parameters
under the wrong (non-periodic) boundary conditions -- witnessed: setting
set_periodic_axes() changed the returned |S11| by <1.5e-4 (float32 graph noise),
i.e. the periodic BC was silently ignored.

Rather than support periodicity in the eager extractor (an unrequested feature),
the preflight now fails loudly -- mirroring the non-uniform + lumped-port
NotImplementedError guard. These tests pin that contract and confirm the guard
does not over-fire on the ordinary non-periodic path.
"""

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse


def _sim(periodic_axes, extent=None):
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1.0e-3,
        boundary="cpml", cpml_layers=4,
    )
    if periodic_axes:
        sim.set_periodic_axes(periodic_axes)
    sim.add_port(
        position=(0.01, 0.01, 0.01), component="ez", impedance=50.0,
        waveform=GaussianPulse(f0=5e9, bandwidth=0.9), extent=extent,
    )
    return sim


def test_periodic_lumped_compute_s_params_raises():
    """Periodic axes + single-cell lumped port + compute_s_params must raise."""
    sim = _sim("y", extent=None)
    with pytest.raises(NotImplementedError, match="periodic"):
        sim.run(n_steps=50, compute_s_params=True)


def test_periodic_wire_compute_s_params_raises():
    """Periodic axes + wire port (extent=) + compute_s_params must raise."""
    sim = _sim("xy", extent=0.004)
    with pytest.raises(NotImplementedError, match="periodic"):
        sim.run(n_steps=50, compute_s_params=True)


def test_nonperiodic_lumped_compute_s_params_ok():
    """Guard must NOT over-fire: the ordinary non-periodic path still runs."""
    sim = _sim("", extent=None)
    res = sim.run(n_steps=50, compute_s_params=True)
    assert res.s_params is not None
    assert np.all(np.isfinite(np.asarray(res.s_params)))
