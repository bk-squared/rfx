"""Review regressions: boundary flux, clocks, availability and missing owners."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from rfx.measurement import measurement_plan
from tests.contracts.path_equivalence.builders import FREQS, build
from tests.contracts.test_measurement_plan import SetupCaptured, judge, setup


@pytest.mark.parametrize('mesh', ['uniform', 'constant', 'graded'])
@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
@pytest.mark.parametrize('boundary', [False, True])
def test_flux_actual_sampler_all_axes(mesh, axis, boundary):
    row = ('_flux_monitors', 'flux')
    lane = 'run_uniform' if mesh == 'uniform' else 'run_nonuniform'
    sim = build(row, lane, graded=mesh == 'graded')
    grid = sim._build_realized_grid()
    sim._flux_monitors[0] = replace(sim._flux_monitors[0], axis=axis,
                                    coordinate=grid.node_of('xyz'.index(axis), 0 if boundary else 3))
    actual = setup(sim)
    judge(measurement_plan(sim, n_steps=12, path=lane), actual, row, mesh != 'uniform')


@pytest.mark.parametrize('row,change', [
    (('_dft_planes', 'dft_plane'), 'stamp'),
    (('_waveguide_ports', 'waveguide_port'), 'slot'),
    (('_ports', 'lumped_port'), 'availability'),
    (('_ports', 'wire_port'), 'reference_stage'),
    (('_ntff', 'ntff_box'), 'drop'),
    (('_current_moments', 'block_moments'), 'drop'),
])
def test_review_metadata_mutations(row, change):
    sim = build(row, 'run_uniform')
    actual = setup(sim)
    plan = measurement_plan(sim, n_steps=12, frequencies=FREQS)
    owner = next(o for o in plan.owners if o.kind != 'probe')
    if change == 'stamp':
        channel = replace(owner.channels[0], e_time_offset=1.)
        owner = replace(owner, channels=(channel,))
    elif change == 'slot':
        owner = replace(owner, channels=tuple(replace(c, slot_offset=1) for c in owner.channels))
    elif change == 'availability':
        owner = replace(owner, availability='sampled')
    elif change == 'reference_stage':
        owner = replace(owner, channels=tuple(replace(c, sample_stage='post-injection')
                                             if c.name == 'V_ref' else c for c in owner.channels))
    mutated = replace(plan, owners=tuple(owner if o.id == owner.id else o for o in plan.owners
                                        if change != 'drop' or o.id != owner.id))
    with pytest.raises((AssertionError, KeyError)):
        judge(mutated, actual, row, False)


def test_uniform_lumped_run_second_scan_availability():
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    sim = build(('_ports', 'lumped_port'), 'run_uniform')
    actual = {}
    def stop(grid, *args, **kwargs):
        actual.update(kwargs, grid=grid)
        raise SetupCaptured
    # First prove run(compute_s_params=True) reaches the second-scan driver.
    with patch('rfx.simulation.run', return_value=SimpleNamespace(wire_port_sparams=None)), \
         patch('rfx.probes.sparam_driver.compute_lumped_wire_s_matrix_via_scan', side_effect=SetupCaptured) as driver:
        with pytest.raises(SetupCaptured):
            sim.run(n_steps=12, skip_preflight=True, compute_s_params=True, s_param_freqs=FREQS)
    assert driver.call_count == 1
    # Then capture that driver's actual production-scan setup, including V_ref.
    with patch('rfx.simulation.run', stop), pytest.raises(SetupCaptured):
        compute_lumped_wire_s_matrix_via_scan(sim, jnp.asarray(FREQS), n_steps=12)
    plan = measurement_plan(sim, n_steps=12, frequencies=FREQS)
    judge(plan, actual, ('_ports', 'lumped_port'), False)


@pytest.mark.parametrize('forward', [False, True])
def test_current_moment_supported_z_graded(forward):
    import numpy as np
    from tests.contracts.path_equivalence.builders import DX
    row = ('_current_moments', 'block_moments')
    lane = 'fwd_nonuniform' if forward else 'run_nonuniform'
    sim = build(row, lane, graded=True)
    # The production monitor admits z grading, but refuses in-plane grading.
    sim._dx_profile = np.full_like(sim._dx_profile, DX)
    sim._dy_profile = np.full_like(sim._dy_profile, DX)
    actual = setup(sim, forward)
    judge(measurement_plan(sim, n_steps=12, path=lane), actual, row, True)


@pytest.mark.parametrize('mesh', ['uniform', 'constant', 'graded'])
@pytest.mark.parametrize('component', ['ex', 'hx'])
def test_raw_plane_and_probe_stamps(mesh, component):
    row = ('_dft_planes', 'dft_plane')
    lane = 'run_uniform' if mesh == 'uniform' else 'run_nonuniform'
    sim = build(row, lane, graded=mesh == 'graded')
    sim._dft_planes[0] = replace(sim._dft_planes[0], component=component)
    sim._probes[0] = replace(sim._probes[0], component=component)
    actual = setup(sim)
    judge(measurement_plan(sim, n_steps=12, path=lane), actual, row, mesh != 'uniform')


@pytest.mark.parametrize('mesh', ['uniform', 'constant', 'graded'])
@pytest.mark.parametrize('forward', [False, True])
@pytest.mark.parametrize('row,kind', [(('_msl_ports', 'msl_port'), 'msl'),
                                     (('_coaxial_ports', 'coax_port'), 'coax')])
def test_run_calculator_owner_availability(mesh, forward, row, kind):
    from tests.contracts.measurement_oracles import judge_metadata
    lane = ('fwd_' if forward else 'run_')+('uniform' if mesh == 'uniform' else 'nonuniform')
    sim = build(row, lane, graded=mesh == 'graded')
    if kind == 'coax':
        with pytest.raises(NotImplementedError, match='is not wired into Simulation'):
            setup(sim, forward)
        actual = {}
    else:
        actual = setup(sim, forward)
    plan = measurement_plan(sim, n_steps=12, path=lane)
    owner = next(o for o in plan.owners if o.kind == kind)
    # MSL feed setup has no calculator projector; coax refuses before the scan.
    assert not actual.get('dft_planes')
    assert not actual.get('wire_port_sparams')
    assert not actual.get('lumped_port_sparams')
    assert owner.channels == ()
    judge_metadata(plan, actual)
    with pytest.raises(AssertionError):
        judge_metadata(replace(plan, owners=tuple(replace(o, availability='sampled') if o == owner else o
                                                  for o in plan.owners)), actual)
