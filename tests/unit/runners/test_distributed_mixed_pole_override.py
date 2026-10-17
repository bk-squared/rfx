"""rfx #1834: a block with both Debye and Lorentz poles under forward(eps_override=...).

On the graded multi-device wrapped step the override path scales the point source with the mixed update
coefficient. Its Debye beta was unpacked rows-first, so the pole sum ran along x and the source came out at
0.478 of its value (probe traces 35-52 % of peak, the eps gradient 84-100 %). The override here is the drawn
permittivity, so one device with and without it is the same model. Bars: the accumulated rounding law on every
E cell (1.2e-7 x the running sum of max|E|), the probe spectra within 1e-6 of their largest bin, the gradient
within 1e-4 (rfx-archive records/20261017-item5-dispersive).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx._grid_metric import field_index
from tests.unit.runners.test_distributed_nu_wrapped_step import pole_scene

pytestmark = pytest.mark.distributed

N_STEPS = 400
N_NAMED = 4  # pole_scene's two E and two H probes come first


def _all_e_cells(sim):
    """One probe position per distinct Ex/Ey/Ez cell of the physical domain."""
    grid, dom = sim._build_nonuniform_grid(), sim._domain
    axes = [np.arange(0., d + 1e-12, 5e-5) for d in dom]
    seen, out = set(), []
    for comp in ('ex', 'ey', 'ez'):
        indices = [sorted({field_index(grid, tuple(v if a == ax else dom[a] / 2 for a in range(3)), comp)[ax]: v
                           for v in axes[ax]}.items()) for ax in range(3)]
        for _, x in indices[0]:
            for _, y in indices[1]:
                for _, z in indices[2]:
                    key = (comp, *field_index(grid, (x, y, z), comp))
                    if key not in seen:
                        seen.add(key)
                        out.append(((x, y, z), comp))
    return out


def _scene(cells):
    sim = pole_scene('mixed')
    if cells:
        for position, component in _all_e_cells(sim):
            sim.add_probe(position, component)
    return sim


def _forward(sim, eps, devices=None):
    kw = {} if devices is None else dict(distributed=True, devices=jax.devices('cpu')[:devices])
    return sim.forward(eps_override=eps, n_steps=N_STEPS, checkpoint=False, skip_preflight=True, **kw).time_series


def _drawn_eps(sim):
    return jnp.asarray(sim._assemble_materials_nu(sim._build_nonuniform_grid())[0].eps_r)


def test_mixed_pole_override_fields_match_one_device():
    sim = _scene(cells=True)
    eps, dt = _drawn_eps(sim), float(sim._build_nonuniform_grid().dt)
    one = np.asarray(_forward(sim, eps), np.float64)
    assert np.isfinite(one).all() and one.shape[1] > N_NAMED
    kernel = np.exp(-2j * np.pi * np.outer(np.linspace(2e9, 10e9, 20), np.arange(len(one)) * dt)) * dt
    expected = kernel @ one[:, :N_NAMED]
    allowance = 1.2e-7 * np.cumsum(np.abs(one[:, N_NAMED:]).max(axis=1))
    for devices in (1, 2, 3):
        multi = np.asarray(_forward(sim, eps, devices), np.float64)
        assert np.isfinite(multi).all(), devices
        difference = np.abs(multi[:, N_NAMED:] - one[:, N_NAMED:]).max(axis=1)
        ratio = np.divide(difference, allowance, out=np.where(difference > 0, np.inf, 0.), where=allowance > 0)
        spectral = float(np.abs(kernel @ multi[:, :N_NAMED] - expected).max() / np.abs(expected).max())
        assert ratio.max() <= 1., (devices, float(ratio.max()), int(ratio.argmax()) + 1)
        assert spectral <= 1e-6, (devices, spectral)


def test_mixed_pole_override_gradient_matches_one_device():
    sim = _scene(cells=False)
    eps = _drawn_eps(sim)

    def objective(e, devices=None):
        return jnp.sum(_forward(sim, e, devices) ** 2)

    one = np.asarray(jax.grad(objective)(eps), np.float64)
    assert np.isfinite(one).all() and np.any(one)
    for devices in (1, 2, 3):
        multi = np.asarray(jax.grad(lambda e: objective(e, devices))(eps), np.float64)
        assert np.isfinite(multi).all(), devices
        relative = float(np.abs(multi - one).max() / np.abs(one).max())
        assert relative <= 1e-4, (devices, relative)
