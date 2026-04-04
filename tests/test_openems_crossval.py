"""Cross-validation: rfx vs openEMS for PEC rectangular cavity.

Compares TM110 resonant frequency from both simulators against
the analytical solution.

Requires: openems (apt install openems python3-openems)
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

# numpy deprecation fix for openEMS v0.0.35
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'complex'):
    np.complex = complex

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse

# Reuse FFT helper from meep crossval
from tests.test_meep_crossval import fft_peak_freq

# Physical cavity dimensions (meters)
CAVITY_A = 0.1
CAVITY_B = 0.1
CAVITY_D = 0.05

# Analytical TM110
F_ANALYTICAL = (C0 / 2.0) * np.sqrt((1 / CAVITY_A) ** 2 + (1 / CAVITY_B) ** 2)


def run_rfx_cavity() -> float:
    """Run rfx PEC cavity and return peak frequency via FFT."""
    grid = Grid(freq_max=5e9, domain=(CAVITY_A, CAVITY_B, CAVITY_D),
                dx=0.001, cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=F_ANALYTICAL, bandwidth=0.8)
    src_i = grid.nx // 3
    src_j = grid.ny // 3
    src_k = grid.nz // 2

    probe_i = 2 * grid.nx // 3
    probe_j = 2 * grid.ny // 3
    probe_k = grid.nz // 2

    num_steps = grid.num_timesteps(num_periods=120)
    dt = grid.dt
    dx = grid.dx

    time_series = np.zeros(num_steps)
    for n in range(num_steps):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        ez = state.ez.at[src_i, src_j, src_k].add(pulse(t))
        state = state._replace(ez=ez)
        time_series[n] = float(state.ez[probe_i, probe_j, probe_k])

    return fft_peak_freq(time_series, dt, F_ANALYTICAL * 0.5, F_ANALYTICAL * 1.5)


def run_openems_cavity() -> float:
    """Run openEMS PEC cavity and return resonant frequency via FFT probe.

    Uses the same cavity dimensions and comparable resolution to rfx.
    """
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS

    unit = 1e-3  # mm
    a_mm = CAVITY_A / unit  # 100 mm
    b_mm = CAVITY_B / unit  # 100 mm
    d_mm = CAVITY_D / unit  # 50 mm

    # Frequency parameters
    f0 = F_ANALYTICAL
    fc = f0 * 0.8  # bandwidth

    # Create FDTD object
    FDTD = openEMS(NrTS=60000, EndCriteria=0)  # run all steps, no early stop
    FDTD.SetGaussExcite(f0, fc)
    # PEC boundary on all 6 faces
    FDTD.SetBoundaryCond(['PEC', 'PEC', 'PEC', 'PEC', 'PEC', 'PEC'])

    # Create structure
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)

    # Mesh: uniform 1mm grid (matches rfx dx=0.001)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', np.linspace(0, a_mm, int(a_mm) + 1))
    mesh.AddLine('y', np.linspace(0, b_mm, int(b_mm) + 1))
    mesh.AddLine('z', np.linspace(0, d_mm, int(d_mm) + 1))

    # Source: Ez point source at ~1/3, snapped to mesh nodes (integer mm)
    src_x = round(a_mm / 3.0)
    src_y = round(b_mm / 3.0)
    src_z = round(d_mm / 2.0)

    # Ez excitation needs 1-cell z-extent to have a z-directed edge
    exc = CSX.AddExcitation('source', exc_type=0, exc_val=[0, 0, 1])  # Ez
    exc.AddBox([src_x, src_y, src_z], [src_x, src_y, src_z + 1])

    # Probe: Ez time-domain at ~2/3, snapped to mesh nodes
    probe_x = round(2.0 * a_mm / 3.0)
    probe_y = round(2.0 * b_mm / 3.0)
    probe_z = round(d_mm / 2.0)

    # Voltage probe: integrate Ez over 1 cell in z (point box gives zero)
    probe = CSX.AddProbe('ez_probe', p_type=0)
    probe.AddBox([probe_x, probe_y, probe_z], [probe_x, probe_y, probe_z + 1])

    # Run in temp directory
    sim_dir = tempfile.mkdtemp(prefix='rfx_openems_')
    try:
        FDTD.Run(sim_dir, verbose=0)

        # openEMS saves probe as tab-separated text: sim_dir/ez_probe
        # Header lines start with '%', columns are t(s) and voltage
        probe_file = os.path.join(sim_dir, 'ez_probe')
        if not os.path.exists(probe_file):
            pytest.fail(f"openEMS probe file not found in {sim_dir}: {os.listdir(sim_dir)}")

        data = np.loadtxt(probe_file, comments='%')
        t_arr = data[:, 0]
        ez_arr = data[:, 1]

        dt_oems = t_arr[1] - t_arr[0]
        f_oems = fft_peak_freq(ez_arr, dt_oems, F_ANALYTICAL * 0.5, F_ANALYTICAL * 1.5)
        return f_oems
    finally:
        shutil.rmtree(sim_dir, ignore_errors=True)


@pytest.mark.slow
def test_rfx_vs_openems_cavity():
    """rfx and openEMS should agree on TM110 resonance within 0.5%."""
    f_rfx = run_rfx_cavity()
    f_oems = run_openems_cavity()

    err_rfx = abs(f_rfx - F_ANALYTICAL) / F_ANALYTICAL
    err_oems = abs(f_oems - F_ANALYTICAL) / F_ANALYTICAL
    err_cross = abs(f_rfx - f_oems) / f_oems

    print(f"\n{'='*55}")
    print("PEC Cavity TM110 Cross-Validation (rfx vs openEMS)")
    print(f"{'='*55}")
    print(f"Analytical:    {F_ANALYTICAL / 1e9:.6f} GHz")
    print(f"rfx:           {f_rfx / 1e9:.6f} GHz  (err: {err_rfx*100:.3f}%)")
    print(f"openEMS:       {f_oems / 1e9:.6f} GHz  (err: {err_oems*100:.3f}%)")
    print(f"rfx vs openEMS: {err_cross*100:.3f}%")
    print(f"{'='*55}")

    assert err_rfx < 0.005, f"rfx error {err_rfx*100:.2f}% exceeds 0.5%"
    assert err_oems < 0.005, f"openEMS error {err_oems*100:.2f}% exceeds 0.5%"
    assert err_cross < 0.01, f"Cross-validation gap {err_cross*100:.2f}% exceeds 1%"
