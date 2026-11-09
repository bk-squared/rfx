"""A magnetic SAT interface preserves x/y transposition of public probe records."""
import sys

import numpy as np

from rfx import Box, GaussianPulse, Simulation


DX = 1e-3
N_STEPS = 160
TRANSPOSE_BAR = 1e-5


def _record(*, transpose=False, ramp=False):
    def xyz(x, y, z):
        return tuple(DX * v for v in ((y, x, z) if transpose else (x, y, z)))

    sim = Simulation(freq_max=25e9, domain=xyz(9, 9, 9), dx=DX,
                     boundary="pec", cpml_layers=0)
    sim.add_refinement((3 * DX, 6 * DX), ratio=2, validation="research")
    if ramp:
        # Nine cell-wide pieces of the 1..3 ramp cross the low SAT face.
        # Transposing each Box and both observation sites builds the reference
        # independently through Simulation, without reading SAT coefficients.
        for i in range(9):
            name = f"mu-{i}"
            sim.add_material(name, eps_r=1., mu_r=1. + i / 4.)
            sim.add(Box(xyz(i, 0, 2), xyz(i + 1, 9, 5)), material=name)
    else:
        sim.add_material("uniform-mu", eps_r=1., mu_r=2.)
        sim.add(Box(xyz(0, 0, 0), xyz(9, 9, 9)), material="uniform-mu")
    component = "ey" if transpose else "ex"
    sim.add_source(xyz(2, 3, 4), component,
                   waveform=GaussianPulse(25e9), amplitude_kind="field")
    sim.add_probe(xyz(3, 2, 3.5), component)
    result = sim.run(n_steps=N_STEPS, compute_s_params=False)
    trace = np.asarray(result.time_series, dtype=np.float64).reshape(N_STEPS)
    # Result.grid is the executed fine grid, with both endpoint nodes.
    assert result.grid.shape == (19, 19, 7)
    assert np.isfinite(trace).all()
    assert np.max(np.abs(trace)) > 0, "transposition needs a nonzero probe record"
    return trace, result.grid.shape


def _compare(*, ramp):
    a, mesh = _record(ramp=ramp)
    b, mesh_b = _record(transpose=True, ramp=ramp)
    assert mesh_b == mesh
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    delta = np.abs(a - b)
    error = float(np.max(delta))
    row = "ramp" if ramp else "uniform-mu-2-control"
    worst = int(np.argmax(delta))
    measurement = (
        f"dual/{row}: relative={error / peak:.12g}, bar={TRANSPOSE_BAR:.12g}, "
        f"max_abs_difference={error:.12g}, peak={peak:.12g}, "
        f"absolute_bar={TRANSPOSE_BAR * peak:.12g}, step={worst + 1}, "
        f"A={a[worst]:.12g}, B={b[worst]:.12g}, steps={N_STEPS}, "
        f"coarse_cells=(9, 9, 9), fine_nodes={mesh}"
    )
    print(measurement, file=sys.stderr)
    assert error <= TRANSPOSE_BAR * peak, measurement
    return a


def test_sat_probe_records_are_xy_transposes():
    # No existing subgrid x/y transposition parity test was found in tests/.
    # Design B3.7 and this function's spec fix the image bar at 1e-5 of peak.
    # The public refinement API always uses dual projection (Design A39).
    # Control first: stop on an asymmetric uniform-mu pair; no fine-grid fallback.
    control = _compare(ramp=False)
    ramp = _compare(ramp=True)
    peak = max(float(np.max(np.abs(control))), float(np.max(np.abs(ramp))))
    contrast = float(np.max(np.abs(ramp - control))) / peak
    print(f"dual/mu-contrast: relative={contrast:.12g}, "
          f"bar=>{TRANSPOSE_BAR:.12g}", file=sys.stderr)
    assert contrast > TRANSPOSE_BAR, "the probe must see the magnetic contrast"
