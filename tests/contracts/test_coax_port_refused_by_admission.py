"""Uniform model and spectral lanes are admitted; other lanes name their refusal."""
import pytest
import numpy as np

from rfx import Simulation
from rfx.runners._admission import admit
from tests.contracts.test_coax_port_model_and_drive import scene, capture


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_uniform_reaches_runner(monkeypatch, entry):
    model = capture(monkeypatch, scene(), entry)
    assert model['sources'] and model['mag_sources']


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_without_coax_runs_one_step(mesh, entry):
    profiles = {'dx_profile': np.full(12, .001)} if mesh == 'graded' else {}
    sim = Simulation(freq_max=10e9, domain=(.012, .012, .012), dx=.001,
                     boundary='pec', **profiles)
    sim.add_probe((.008, .006, .006), 'ez')
    result = getattr(sim, entry)(n_steps=1, skip_preflight=True)
    assert result.time_series.shape == (1, 1)
    assert np.isfinite(np.asarray(result.time_series)).all()


@pytest.mark.parametrize('caller', ['run()', 'forward()'])
def test_uniform_s_request_admitted(caller):
    from rfx.runners._admission import admit_s_request
    admit_s_request(scene(), caller)


GATE_CASES = (
    ('precision', 'float32'), ('stencil_order', 'stencil_order=2'), ('mode', "mode='3d'"),
    ('periodic', 'periodic'), ('boundary', 'CPML on all six faces'), ('zero_cpml', 'CPML on all six faces'),
    ('partial_cpml', 'CPML on all six faces'), ('partial_zero_cpml', 'CPML on all six faces'),
    ('tfsf', 'TF/SF'), ('count', 'exactly one coax'),
    ('face', "face='top' or 'bottom'"), ('mixed', 'other S-port family'),
    ('conformal_pec', 'conformal_pec'), ('subpixel_smoothing', 'subpixel_smoothing'),
    ('eps_override', 'eps_override'), ('sigma_override', 'sigma_override'),
    ('short', '67 line cells for the shielded source end'), ('lateral', 'absorber.*1 cell'),
    ('ntff_shell', 'NTFF box.*shell'), ('ntff_pin', 'NTFF box.*reference and pin'),
    ('nonuniform', 'add_coaxial_port'), ('distributed', 'add_coaxial_port'),
    ('distributed_nu', 'add_coaxial_port'), ('subgridded', 'add_coaxial_port'),
)


def check_refusal(sim, lane, options, pattern):
    with pytest.raises(NotImplementedError, match=pattern) as caught:
        admit(sim, lane, run_args=options, grid=sim._build_grid())
    assert lane in str(caught.value)
    assert 'add_coaxial_port' in str(caught.value)


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('gate,pattern', GATE_CASES)
def test_admission_conditions(entry, gate, pattern):
    sim = scene()
    options = {}
    lane = ('run' if entry == 'run' else 'fwd') + '_uniform'
    if gate in ('nonuniform', 'distributed', 'distributed_nu', 'subgridded'):
        if gate == 'subgridded' and entry == 'forward':
            pytest.skip('forward() has no subgridded lane')
        lane = ('run' if entry == 'run' else 'fwd') + '_' + gate
    elif gate in ('precision', 'stencil_order', 'mode', 'periodic', 'boundary', 'zero_cpml'):
        attribute, value = {'precision': ('_precision', 'float64'), 'stencil_order': ('_stencil_order', 4),
            'mode': ('_mode', '2d_tmz'), 'periodic': ('_periodic_axes', 'x'),
            'boundary': ('_boundary', 'pec'), 'zero_cpml': ('_cpml_layers', 0)}[gate]
        setattr(sim, attribute, value)
    elif gate == 'partial_cpml':
        from rfx.boundaries.spec import BoundarySpec
        sim._boundary_spec = BoundarySpec(x='pec', y='cpml', z='cpml')
    elif gate == 'partial_zero_cpml':
        from rfx.boundaries.spec import Boundary, BoundarySpec
        sim._boundary_spec = BoundarySpec(x=Boundary('cpml', 'cpml', lo_thickness=0),
                                         y='cpml', z='cpml')
    elif gate == 'tfsf':
        sim.add_tfsf_source()
    elif gate == 'count':
        sim.add_coaxial_port((.006, .006, .002))
    elif gate == 'mixed':
        sim.add_port((.008, .006, .004), 'ez')
    elif gate in ('conformal_pec', 'subpixel_smoothing', 'eps_override', 'sigma_override'):
        options[gate] = True
    elif gate in ('face', 'short', 'lateral'):
        fields = {'face': {'face': 'left'}, 'short': {'position': (.006, .006, .025)},
                  'lateral': {'position': (.0025, .006, .002)}}[gate]
        sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(**fields)
    elif gate == 'ntff_shell':
        sim._ntff = ((.004, .001, .001), (.011, .011, .039))
    elif gate == 'ntff_pin':
        sim._ntff = ((.001, .001, .003), (.011, .011, .039))
    check_refusal(sim, lane, options, pattern)
