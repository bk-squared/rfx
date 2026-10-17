"""Item 5's independently sampled one-device law C and spectrum reference."""
import json
import os
from pathlib import Path

import jax
import numpy as np

from rfx import Box, Simulation
from rfx._grid_metric import field_index
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from tests.unit.runners.test_distributed_edge_averaged_materials import GRADED_1302, WAVEFORM, _mm


def scene(kind, *, tau_factor=1., cells=True, port=False, sigma=0.):
    """Scene T: the pole block continues through both x CPML faces."""
    sim = Simulation(freq_max=10e9, domain=(24e-3, 12e-3, 12e-3), dx=1e-3,
                     boundary='cpml', cpml_layers=6, dx_profile=GRADED_1302)
    if port:
        sim.add_port(_mm(6, 6, 6), 'ez', impedance=50., waveform=WAVEFORM)
    else:
        sim.add_source(_mm(6, 6, 6), 'ez', waveform=WAVEFORM, amplitude_kind='current')
    for pos, comp in [((19, 6, 6), 'ez'), ((22, 6, 6), 'ez'),
                      ((2, 4, 7), 'hy'), ((13, 6, 6), 'ez')]:
        sim.add_probe(_mm(*pos), comp)
    poles = {}
    if kind in ('debye', 'mixed'):
        poles['debye_poles'] = [DebyePole(delta_eps=1., tau=1e-11 * tau_factor)]
    if kind in ('lorentz', 'mixed'):
        poles['lorentz_poles'] = [LorentzPole(omega_0=2*np.pi*8e9, delta=2*np.pi*1e9,
                                            kappa=(2*np.pi*8e9)**2)]
    sim.add_material('m', eps_r=4., sigma=sigma, **poles)
    sim.add(Box((-1., 3e-3, 3e-3), (1., 9e-3, 9e-3)), material='m')
    if cells:
        for pos, comp in all_e_cells(sim):
            sim.add_probe(pos, comp)
    return sim


def all_e_cells(sim):
    """One probe per distinct Ex/Ey/Ez cell in the physical domain (law.py)."""
    grid, dom = sim._build_nonuniform_grid(), sim._domain
    axes = [np.arange(0., d + 1e-12, 5e-5) for d in dom]
    seen, out = set(), []
    for comp in ('ex', 'ey', 'ez'):
        indices = [sorted({field_index(grid, tuple(v if a == ax else dom[a]/2
                    for a in range(3)), comp)[ax]: v for v in axes[ax]}.items())
                   for ax in range(3)]
        for _, x in indices[0]:
            for _, y in indices[1]:
                for _, z in indices[2]:
                    key = (comp, *field_index(grid, (x, y, z), comp))
                    if key not in seen:
                        seen.add(key)
                        out.append(((x, y, z), comp))
    return out


def measure(one, multi, dt):
    one, multi = np.asarray(one, np.float64), np.asarray(multi, np.float64)
    assert one.shape == multi.shape and one.shape[1] > 4
    assert np.isfinite(one).all() and np.isfinite(multi).all()
    difference = np.abs(multi[:, 4:] - one[:, 4:]).max(axis=1)
    allowance = 1.2e-7 * np.cumsum(np.abs(one[:, 4:]).max(axis=1))
    ratio = np.divide(difference, allowance, out=np.where(difference > 0, np.inf, 0.),
                      where=allowance > 0)
    kernel = np.exp(-2j*np.pi*np.outer(np.linspace(2e9, 10e9, 20),
                                     np.arange(len(one))*dt)) * dt
    expected, actual = kernel @ one[:, :4], kernel @ multi[:, :4]
    return dict(law_ratio=float(ratio.max()), worst_step=int(ratio.argmax()) + 1,
                spectral=float(np.abs(actual - expected).max() / np.abs(expected).max()),
                e_cells=one.shape[1] - 4)


def record(row):
    print('ITEM5 ' + json.dumps(row), flush=True)
    target = os.environ.get('RFX_ITEM5_REPORT')
    if target:
        with Path(target).open('a') as stream:
            stream.write(json.dumps(row) + '\n')


def check(row, *, spectral_only=False):
    if not spectral_only:
        assert row['law_ratio'] <= 1., f"law C: {row}"
    assert row['spectral'] <= 1e-6, f"bar S: {row}"


def snapshot_parity(kind, lengths, monkeypatch, *, entry, scene_builder=None):
    """Measure the existing x-lo pole box with lawS.py's probe arrangement."""
    from rfx.runners import _distributed_nu_wrapped as runner
    from tests.unit.runners.test_distributed_edge_averaged_materials import _debye_box, _devices
    sim = (scene_builder or _debye_box)('cpml', 'fwd', kind)
    assert len(sim._probes) == 2
    sim.add_probe(_mm(12, 6, 6), 'ez')
    sim.add_probe(_mm(14, 5, 7), 'hy')
    assert len(sim._probes) == 4
    assert len({(p.component, tuple(p.position)) for p in sim._probes}) == 4
    for position, component in all_e_cells(sim):
        sim.add_probe(position, component)
    dt = float(sim._build_nonuniform_grid().dt)
    original, calls = runner.build_setup, []
    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'build_setup', observe)
    for n in lengths:
        one = np.asarray(sim.forward(n_steps=n, skip_preflight=True, checkpoint=False).time_series)
        calls.clear()
        multi = np.asarray(sim.forward(n_steps=n, skip_preflight=True, checkpoint=False,
            distributed=True, devices=_devices(2)).time_series)
        assert calls, (kind, entry, n, 'wrapped setup not reached')
        row = dict(box='debye_box_cpml_fwd', kind=kind, entry=entry, devices=2,
                   n=n, **measure(one, multi, dt))
        record(row)
        check(row)


def parity(kind, n, monkeypatch):
    from rfx.runners import _distributed_nu_wrapped as runner
    sim = scene(kind)
    dt = float(sim._build_nonuniform_grid().dt)
    original, calls = runner.build_setup, []
    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'build_setup', observe)
    for entry in ('forward', 'run'):
        def evaluate(devices=None):
            if entry == 'run':
                return sim.run(n_steps=n, skip_preflight=True, devices=devices).time_series
            kw = {} if devices is None else dict(distributed=True, devices=devices)
            return sim.forward(n_steps=n, skip_preflight=True, checkpoint=False, **kw).time_series
        one = np.asarray(evaluate())
        for devices in ((2, 3) if entry == 'forward' else (2,)):
            calls.clear()
            multi = np.asarray(evaluate(jax.devices('cpu')[:devices]))
            assert calls, (kind, entry, devices, 'wrapped setup not reached')
            row = dict(kind=kind, entry=entry, devices=devices, n=n, **measure(one, multi, dt))
            record(row)
            check(row)


def override_parity():
    """Scene T mixed override, against actual one-device solves (#1834)."""
    import jax.numpy as jnp
    sim = scene('mixed')
    grid = sim._build_nonuniform_grid()
    eps = jnp.asarray(sim._assemble_materials_nu(grid)[0].eps_r)
    def forward(model, e, devices=None):
        kw = {} if devices is None else dict(distributed=True, devices=jax.devices('cpu')[:devices])
        return model.forward(eps_override=e, n_steps=400, checkpoint=False,
                             skip_preflight=True, **kw).time_series
    one = np.asarray(forward(sim, eps))
    for devices in (1, 2, 3):
        multi = np.asarray(forward(sim, eps, devices))
        row = dict(kind='mixed', boundary='cpml', entry='override_forward', n=400,
                   devices=devices, **measure(one, multi, float(grid.dt)))
        record(row)
        check(row)

    # These four declared probes define the gradient objective; the added
    # all-cell probes above only observe law C and do not change the model.
    sim = scene('mixed', cells=False)
    def objective(e, devices=None):
        return jnp.sum(forward(sim, e, devices)**2)
    one = np.asarray(jax.grad(objective)(eps), np.float64)
    for devices in (1, 2, 3):
        multi = np.asarray(jax.grad(lambda e: objective(e, devices))(eps), np.float64)
        assert np.isfinite(one).all() and np.isfinite(multi).all() and np.any(one)
        relative = float(np.abs(multi - one).max() / np.abs(one).max())
        row = dict(kind='mixed', boundary='cpml', entry='override_gradient', n=400,
                   devices=devices, relative=relative)
        record(row)
        assert relative <= 1e-4, row
