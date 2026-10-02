"""B3b mirrored-lattice judges, independent of legacy half-cell wall locks."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rfx import Simulation, Box, DebyePole, GaussianPulse
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.probes import flux_spectrum
from rfx.harminv import harminv


def record(name, data):
    directory = os.environ.get("RFX_B3_OUTPUT")
    if directory:
        Path(directory, name + '.json').write_text(json.dumps(data, indent=2) + '\n')


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_asymmetric_low_high_mirror(graded, entry):
    traces, separations = [], []
    for high in (False, True):
        profile = np.r_[1., np.linspace(.8, 1.2, 22), 1.] * .001
        if high:
            profile = profile[::-1].copy()
        sim = Simulation(freq_max=20e9, domain=(.024, .0137, .0053), dx=.001,
                         boundary=BoundarySpec(x=Boundary("pec", "pmc") if high else
                                               Boundary("pmc", "pec"), y="pec", z="pec"),
                         cpml_layers=0,
                         **({'dx_profile': profile} if graded else {}))
        def reflect(x):
            return .024-x if high else x
        sim.add_source((reflect(.0073), .0042, .0021), 'ez', amplitude_kind='field',
                       waveform=GaussianPulse(f0=12e9, bandwidth=.8))
        sim.add_probe((reflect(.0171), .0093, .0021), 'ez')
        kw = dict(n_steps=512, skip_preflight=True)
        if entry == 'run':
            kw['compute_s_params'] = False
        traces.append(np.asarray(getattr(sim, entry)(**kw).time_series)[:, 0])
        grid = sim._build_realized_grid()
        modes = harminv(traces[-1][128:], grid.dt, 9e9, 14e9,
                        min_Q=1, max_modes=16, sv_threshold=1e-3, decimate="auto")
        # Mixed PMC/PEC x walls have a quarter-wave fundamental; y is PEC.
        fundamental = min(modes, key=lambda m: m.freq).freq
        ly = (grid.ny - 1) * .001
        separations.append(float(.5 / np.sqrt((2*fundamental/299792458.)**2 - 1/ly**2)))
    relative = np.linalg.norm(traces[0]-traces[1])/np.linalg.norm(traces[1])
    record(f'mirror-{entry}-{graded}', dict(relative_l2=float(relative), derived_separations_m=separations))
    np.testing.assert_allclose(separations[0], separations[1], rtol=512*np.finfo(np.float32).eps)
    # Reflected discrete lattices: accumulated float32 arithmetic, not
    # a continuum-discretization tolerance. 512 epsilon is the record bound.
    assert relative <= 512*np.finfo(np.float32).eps


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("material", ['dielectric', 'debye'])
def test_face_material_and_crossing_flux_against_full_mirror(graded, material):
    traces, fluxes = [], []
    for full in (False, True):
        length = .024 if full else .012
        profile = np.r_[1., [.9, 1.1]*5, 1.] * .001
        if full:
            profile = np.r_[profile[::-1], profile]
        sim = Simulation(freq_max=20e9, domain=(length, .009, .004), dx=.001,
                         boundary=BoundarySpec(x='pec' if full else Boundary('pmc','pec'),
                                               y='cpml', z='pec'), cpml_layers=4,
                         **({'dx_profile': profile} if graded else {}))
        sim.add_material('medium', eps_r=4.,
                         debye_poles=[DebyePole(2.,20e-12)] if material=='debye' else None)
        sim.add(Box((-.002,-.002,-.002),(length+.002,.011,.006)), material='medium')
        for x in ((.008,.016) if full else (.004,)):
            sim.add_source((x,.003,.002), 'ez', amplitude_kind='field',
                           waveform=GaussianPulse(f0=10e9, bandwidth=.8))
        sim.add_probe((.012 if full else 0.,.006,.002),'ez')
        sim.add_flux_monitor(axis='y', coordinate=.006, freqs=np.array([8e9,10e9,12e9]), name='crossing')
        result=sim.run(n_steps=256, skip_preflight=True, compute_s_params=False,
                       subpixel_smoothing=material=='dielectric')
        traces.append(np.asarray(result.time_series)[:,0])
        fluxes.append(np.asarray(flux_spectrum(result.flux_monitors['crossing'], exact_f64=True)))
    relative=np.linalg.norm(traces[0]-traces[1])/np.linalg.norm(traces[1])
    flux_relative=np.linalg.norm(2*fluxes[0]-fluxes[1])/np.linalg.norm(fluxes[1])
    record(f'material-flux-{graded}-{material}', dict(trace_relative=float(relative),
           flux_relative=float(flux_relative),half_flux=fluxes[0].tolist(),full_flux=fluxes[1].tolist()))
    assert relative <= 256*np.finfo(np.float32).eps
    assert flux_relative <= 512*np.finfo(np.float32).eps


@pytest.mark.parametrize("mode,component", [("2d_tmz", "ez"), ("2d_tez", "ey")])
@pytest.mark.parametrize("high", [False, True])
def test_two_d_magnetic_face_against_full_mirror(mode, component, high):
    """In-plane image survives both 2-D masks, including the TEz z-hi mask."""
    traces = []
    for full in (False, True):
        length = .024 if full else .012
        boundary = BoundarySpec(x="pec" if full else
                                (Boundary("pec", "pmc") if high else Boundary("pmc", "pec")),
                                y="pec", z="pec" if mode == "2d_tmz" else "pmc")
        sim = Simulation(freq_max=20e9, domain=(length, .009, .001), dx=.001,
                         boundary=boundary, cpml_layers=0, mode=mode)
        sources = (.008, .016) if full else ((.008,) if high else (.004,))
        for x in sources:
            sim.add_source((x, .003, 0.), component, amplitude_kind="field",
                           waveform=GaussianPulse(f0=10e9, bandwidth=.8))
        sim.add_probe((.012 if full or high else 0., .006, 0.), component)
        result = sim.run(n_steps=256, skip_preflight=True, compute_s_params=False)
        traces.append(np.asarray(result.time_series)[:, 0])
    relative = np.linalg.norm(traces[0]-traces[1])/np.linalg.norm(traces[1])
    record(f"two-d-{mode}-{high}", dict(relative_l2=float(relative)))
    assert np.max(np.abs(traces[1])) > 0
    assert relative <= 256*np.finfo(np.float32).eps


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("high", [False, True])
def test_magnetic_face_opposite_cpml_against_full_mirror(graded, high):
    """The image and opposite normal absorber coexist on the same axis."""
    traces = []
    for full in (False, True):
        profile = np.r_[1., [.9, 1.1]*3, np.ones(5)] * .001
        if full:
            profile = np.r_[profile[::-1], profile]
        elif high:
            profile = profile[::-1].copy()
        sim = Simulation(freq_max=20e9, domain=(.024 if full else .012, .009, .004),
                         dx=.001, cpml_layers=4,
                         boundary=BoundarySpec(x="cpml" if full else
                                               (Boundary("cpml", "pmc") if high else Boundary("pmc", "cpml")),
                                               y="pec", z="pec"),
                         **({"dx_profile": profile} if graded else {}))
        sources = (.008, .016) if full else ((.008,) if high else (.004,))
        for x in sources:
            sim.add_source((x, .003, .002), "ez", amplitude_kind="field",
                           waveform=GaussianPulse(f0=10e9, bandwidth=.8))
        sim.add_probe((.012 if full or high else 0., .006, .002), "ez")
        result = sim.run(n_steps=256, skip_preflight=True, compute_s_params=False)
        traces.append(np.asarray(result.time_series)[:, 0])
    relative = np.linalg.norm(traces[0]-traces[1])/np.linalg.norm(traces[1])
    record(f"opposite-cpml-{graded}-{high}", dict(relative_l2=float(relative)))
    assert np.max(np.abs(traces[1])) > 0
    assert relative <= 256*np.finfo(np.float32).eps
