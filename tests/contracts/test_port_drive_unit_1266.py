"""Absolute voltage-port drive parity, with asymmetric non-grid lengths.

Power is the monitor's rectangular-window Fourier component in W:
2 * Re(integral(E_DFT x H_DFT*) dA) / T**2, where DFT includes dt.
This compares absolute quantities, without fitting a source normalization.
Each relative-error limit is 2x the measured relative error from an earlier measurement at that test's h (metres).
"""
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.probes import flux_spectrum
from rfx.sources import GaussianPulse

QUARTER_CELL_SIZE_M = .00025

HALF_CELL_SIZE_M = .0005

COARSE_CELL_SIZE_M = .001

DOMAIN = (.0173, .0147, .0162)
FREQS = np.array([4e9, 6e9, 8e9])
RECORD = 8e-10
WIRE_TRACE_RTOL = {
    COARSE_CELL_SIZE_M: 2 * 8.721312769921497e-5,
    HALF_CELL_SIZE_M: 2 * 5.532864452106878e-5,
    QUARTER_CELL_SIZE_M: 2 * 1.916419751069043e-5,
}
POWER_REFINEMENT_RTOL = {
    False: {
        COARSE_CELL_SIZE_M: 2 * 1.1071336963967921e-4,
        HALF_CELL_SIZE_M: 2 * 1.4035944945799651e-5,
        QUARTER_CELL_SIZE_M: 2 * 8.776083758108314e-6,
    },
    True: {
        COARSE_CELL_SIZE_M: 2 * 9.291038004658156e-5,
        HALF_CELL_SIZE_M: 2 * 1.2314657926748717e-5,
        QUARTER_CELL_SIZE_M: 2 * 8.028748179073175e-6,
    },
}


def model(nu, wire, dx=COARSE_CELL_SIZE_M, *, graded=False):
    profiles = {}
    if nu:
        for axis, length in zip('xyz', DOMAIN):
            profiles[f'd{axis}_profile'] = np.full(int(np.ceil(length / dx)), dx)
        if graded:
            dz = profiles['dz_profile']
            start = int(round(.010 / dx))
            groups = int(round(.004 / (4 * dx)))
            profiles['dz_profile'] = np.r_[dz[:start],
                np.tile([1.25 * dx, 1.5 * dx, 1.25 * dx], groups),
                dz[start + 4 * groups:]]
    assert not graded or np.all(np.maximum(profiles['dz_profile'][1:]/profiles['dz_profile'][:-1], profiles['dz_profile'][:-1]/profiles['dz_profile'][1:]) <= 1.25), 'Grading must keep adjacent cell ratios at most 1.25.'
    sim = Simulation(freq_max=20e9, domain=tuple(float(np.sum(profiles[f"d{axis}_profile"])) for axis in "xyz") if nu else DOMAIN, dx=dx,
        cpml_layers=int(round(.003 / dx)),
        boundary=BoundarySpec(x=Boundary('pec', 'cpml'),
            y=Boundary('cpml', 'cpml'), z=Boundary('cpml', 'cpml')),
        **profiles)
    sim.add_port((.0063, .0052, .0044), 'ez', impedance=50.,
        waveform=GaussianPulse(f0=8e9, bandwidth=.9),
        extent=.0023 if wire else None)
    sim.add_probe((.0071, .0053, .0042), 'ez')
    sim.add_flux_monitor(axis='x', coordinate=.0101, freqs=FREQS, name='power')
    return sim


def measure(nu, wire, dx=COARSE_CELL_SIZE_M, *, graded=False):
    sim = model(nu, wire, dx, graded=graded)
    dt = sim._build_nonuniform_grid().dt if nu else sim._build_grid().dt
    steps = int(round(RECORD / dt))
    result = sim.run(n_steps=steps, compute_s_params=False, skip_preflight=True)
    power = 2 * np.asarray(flux_spectrum(result.flux_monitors['power'],
                                       exact_f64=True)) / (steps * dt)**2
    return np.asarray(result.time_series), power


@pytest.mark.parametrize('wire', [False, True], ids=['lumped', 'wire'])
def test_equal_cells_absolute_port_drive(wire):
    uniform, p_uniform = measure(False, wire)
    graded, p_graded = measure(True, wire)
    peak = np.max(np.abs(uniform))
    ulp_error = np.max(np.abs(graded - uniform)) / np.spacing(np.float32(peak))
    power_error = np.max(np.abs(p_graded - p_uniform)) / np.max(np.abs(p_uniform))
    print(f'wire={wire}: trace_peak={peak}, trace_ULP={ulp_error}, power_rel={power_error}')
    assert peak > 0 and np.max(np.abs(p_uniform)) > 0
    assert ulp_error <= 9
    assert power_error <= 1e-4


@pytest.mark.parametrize('wire', [False, True], ids=['lumped', 'wire'])
@pytest.mark.parametrize('dx', [COARSE_CELL_SIZE_M, HALF_CELL_SIZE_M, QUARTER_CELL_SIZE_M], ids=['h', 'h2', 'h4'])
def test_graded_absolute_port_drive(wire, dx):
    """Lumped near-field placement at the source differs between refinements."""
    uniform, p_uniform = measure(False, wire, dx)
    graded, p_graded = measure(True, wire, dx, graded=True)
    trace_error = np.max(np.abs(graded - uniform)) / np.max(np.abs(uniform))
    power_error = np.max(np.abs(p_graded - p_uniform)) / np.max(np.abs(p_uniform))
    print(f'wire={wire}, dx={dx}: trace_rel={trace_error}, power_rel={power_error}')
    if wire:
        assert trace_error <= max(1e-4, WIRE_TRACE_RTOL[dx])
    assert power_error <= max(1e-4, POWER_REFINEMENT_RTOL[wire][dx])
