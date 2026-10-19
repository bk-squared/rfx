"""Item 5 gradient, traced entry paths, and lumped-port spectrum.

The port-S bar is calibrated per host. The multi-device S differs from one device
by float32 rounding formed inside the compiled step: it moves with the host and
the XLA CPU flags, 2 and 3 devices give identical values, and it does not grow
from 400 to 800 steps (rfx-archive rfx/records/20261019-item5-port-s-bar,
78108cc99: 0.91-1.87e-6 of the |S| peak, 1.04-1.23x one device's own
jit-vs-eager spread on every host). So the bar is that unit, measured here on the
same scene and host: multi-device <= 2 x the one-device jit-vs-eager spread,
which itself stays <= 1e-4 (the cross-trace bar for a record-summed quantity).
A floor of 9 float32 ULP of the peak keeps the bar defined where jit is bit-equal.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._distributed_dispersive_cpml import check, measure, record, scene

pytestmark = pytest.mark.distributed

N_STEPS = 400
BINS = np.linspace(2e9, 10e9, 20)


def _s(sim, *, devices=None, eps=None):
    kw = {} if devices is None else dict(distributed=True, devices=devices)
    if eps is not None:
        kw['eps_override'] = eps
    return sim.forward(n_steps=N_STEPS, checkpoint=False, skip_preflight=True,
                       port_s11_freqs=BINS, **kw).s_params


def _relative(a, b):
    a, b = np.asarray(a, np.complex128), np.asarray(b, np.complex128)
    return float(np.abs(a - b).max() / np.abs(b).max())


def _eps(sim):
    return jnp.asarray(sim._assemble_materials_nu(sim._build_nonuniform_grid())[0].eps_r)


@lru_cache(maxsize=1)
def _reference():
    """One device's S and this host's bar: 2 x its own jit-vs-eager spread."""
    sim = scene('debye', cells=False, port=True)
    one = np.asarray(_s(sim), np.complex128)
    eps = _eps(sim)
    eager = np.asarray(_s(sim, eps=eps), np.complex128)
    jitted = np.asarray(jax.jit(lambda e: _s(sim, eps=e))(eps), np.complex128)
    spread = _relative(jitted, eager)
    peak = float(np.abs(one).max())
    floor = 9 * float(np.spacing(np.float32(peak))) / peak
    bar = 2 * max(spread, floor)
    record(dict(kind='debye', entry='port_bar', n=N_STEPS, spread=spread, floor=floor, bar=bar,
                override_vs_plain=_relative(eager, one)))
    assert np.isfinite(one).all() and np.any(one)
    assert spread <= 1e-4, spread
    return one, bar


def test_debye_cpml_lumped_port_s():
    one, bar = _reference()
    sim = scene('debye', cells=False, port=True)
    for devices in (2, 3):
        multi = np.asarray(_s(sim, devices=jax.devices('cpu')[:devices]), np.complex128)
        assert np.isfinite(multi).all() and one.shape == multi.shape
        relative = _relative(multi, one)
        row = dict(kind='debye', entry='port_forward', n=N_STEPS, devices=devices,
                   relative=relative, bar=bar)
        record(row)
        assert relative <= bar, row


def _seam_column(sim, devices):
    nx = sim._build_nonuniform_grid().nx
    return -(-nx // devices)


@pytest.mark.parametrize('mutation', ['seam_eps_10%', 'skipped_e_exchange', 'tau_0.01%'])
def test_port_s_bar_catches(mutation, monkeypatch):
    """Each defect the bar exists for lands above it on this host."""
    one, bar = _reference()
    if mutation == 'tau_0.01%':
        changed = np.asarray(_s(scene('debye', tau_factor=1.0001, cells=False, port=True)),
                             np.complex128)
    else:
        sim = scene('debye', cells=False, port=True)
        eps = None
        if mutation == 'seam_eps_10%':
            # A 10 % permittivity error confined to the first owned x column of
            # rank 1 (2 devices), across the pole block (36 cells); an owned-cell
            # coefficient defect at a material face is of order one. S11 at this
            # port moves by about 1e-4 of its peak per unit relative eps in that
            # column (Mac: 0.1 -> 9.3e-6, 0.5 -> 4.9e-5), so errors below ~2 % sit
            # in the rounding floor. Drawn as geometry: forward refuses
            # eps_override on graded multi-device port runs with Debye live edges.
            from rfx import Box
            from rfx.materials.debye import DebyePole
            grid = sim._build_nonuniform_grid()
            edges = np.concatenate(([0.], np.cumsum(np.asarray(grid.dx_arr, np.float64))))
            seam = _seam_column(sim, 2)
            x0, x1 = (edges[i] - edges[grid.pad_x_lo] for i in (seam, seam + 1))
            sim.add_material('m_seam', eps_r=4. * 1.1,
                             debye_poles=[DebyePole(delta_eps=1., tau=1e-11)])
            sim.add(Box((x0, 3e-3, 3e-3), (x1, 9e-3, 9e-3)), material='m_seam')
        else:
            from rfx.runners import distributed_nu
            monkeypatch.setattr(distributed_nu, '_exchange_e_ghosts_nu',
                                lambda st, *args, **kwargs: st)
        changed = np.asarray(_s(sim, devices=jax.devices('cpu')[:2], eps=eps), np.complex128)
    relative = _relative(changed, one)
    record(dict(kind='debye', entry='port_mutation', mutation=mutation, relative=relative, bar=bar))
    assert relative > bar, (mutation, relative, bar)


def test_run_lumped_s_retains_one_device_refusal():
    """Item 5 leaves the one-device graded lumped-S run API unchanged."""
    sim = scene('debye', cells=False, port=True)
    for devices in (None, jax.devices('cpu')[:2], jax.devices('cpu')[:3]):
        with pytest.raises(NotImplementedError, match='only for add_port') as exc:
            sim.run(n_steps=400, devices=devices, skip_preflight=True,
                    compute_s_params=True, s_param_freqs=np.linspace(2e9, 10e9, 20))
        record(dict(kind='debye', entry='port_run', devices=1 if devices is None else len(devices),
                    status='refuses', reason=str(exc.value)))
