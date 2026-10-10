"""Judge 2: observe the coefficients used by one actual E/absorber step."""
import importlib.util
import json

import jax
import numpy as np
import pytest

from rfx import Box
from rfx.core.yee import EPS_0
from tests.contracts._absorber_curl_scene import build


def edge_mean(cells, component):
    """Independent numpy four-cell mean, exterior cells replicated."""
    cells = np.asarray(cells, dtype=np.float64)
    a, b = [axis for axis in range(3) if axis != component]
    prev_a = np.take(cells, np.maximum(np.arange(cells.shape[a]) - 1, 0), axis=a)
    prev_b = np.take(cells, np.maximum(np.arange(cells.shape[b]) - 1, 0), axis=b)
    prev_ab = np.take(prev_a, np.maximum(np.arange(cells.shape[b]) - 1, 0), axis=b)
    return (cells + prev_a + prev_b + prev_ab) / 4


def coefficient_scene(owner, lane='uniform', face='x_lo'):
    sim = build(owner, face, nu='graded' in lane)
    sim.add_material('lossless', eps_r=3.25)
    bounds = ((0., 6e-3, 0.), (24e-3, 8e-3, 12e-3)) if face == 'x_lo' else (
        (6e-3, 0., 0.), (8e-3, 24e-3, 12e-3))
    sim.add(Box(*bounds), material='lossless')
    return sim


def observe(monkeypatch, sim):
    import rfx.core.yee as yee
    import rfx.boundaries.cpml as cpml
    import rfx.materials.debye as debye
    import rfx.materials.lorentz as lorentz
    jax.clear_caches()
    observed = {'absorber': {}, 'update': {}}

    def capture(kind, c, value):
        jax.debug.callback(lambda v: observed[kind].__setitem__(c, np.asarray(v).copy()), value)

    original = yee.e_component_coeffs

    def plain(*args, **kwargs):
        ca, cb = original(*args, **kwargs)
        for c in range(3):
            capture('update', c, cb[c])
        return ca, cb

    monkeypatch.setattr(yee, 'e_component_coeffs', plain)
    for module, name in ((debye, 'debye_e_component'), (lorentz, 'lorentz_e_component')):
        original_component = getattr(module, name)

        def component(coeffs, c, *args, _original=original_component, **kwargs):
            capture('update', c, coeffs.cb[c])
            return _original(coeffs, c, *args, **kwargs)

        monkeypatch.setattr(module, name, component)
    original_mixed = lorentz.mixed_e_component_coeffs

    def mixed(db, lr, c, dt):
        out = original_mixed(db, lr, c, dt)
        capture('update', c, out[1])
        return out

    monkeypatch.setattr(lorentz, 'mixed_e_component_coeffs', mixed)
    counter = 0
    if importlib.util.find_spec('rfx.boundaries.electric_coefficient') is not None:
        import rfx.boundaries.electric_coefficient as coefficients
        original_absorber = coefficients.corrected_coefficient

        def absorber(*args, **kwargs):
            nonlocal counter
            out = original_absorber(*args, **kwargs)
            capture('absorber', counter % 3, out)
            counter += 1
            return out

        monkeypatch.setattr(coefficients, 'corrected_coefficient', absorber)
    else:
        # Observe the unpatched base at the same multiplication operand.
        original_absorber = cpml._ce_from_inv_si

        def absorber(*args):
            nonlocal counter
            out = original_absorber(*args)
            capture('absorber', counter % 3, out)
            counter += 1
            return out

        monkeypatch.setattr(cpml, '_ce_from_inv_si', absorber)
    result = sim.run(n_steps=1, skip_preflight=True)
    jax.block_until_ready(result.time_series)
    jax.effects_barrier()
    return observed


def coefficient_ulp_bar(owner):
    # ce_today/(1+loss) and the update's one-expression Cb differ by rounding
    # of two extra operations; measured stand-alone on 200,000 random cells
    # (eps 1-12, sigma 1e-3-1e3): max 6 ULP, 0.9 % above 2
    # (independent review, 2026-10-10). A restored defect is 1e6 ULP.
    return 2 if owner in ('debye', 'mixed') else 8


def ulps(a, b):
    a = np.asarray(a, np.float32).view(np.uint32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.uint32).astype(np.int64)
    return np.abs(a - b)


@pytest.mark.parametrize('owner', ['plain', 'dielectric', 'debye', 'lorentz', 'mixed'])
@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_step_uses_owner_curl_coefficient(monkeypatch, owner, lane, record_property):
    sim = coefficient_scene(owner, lane)
    grid = sim._build_realized_grid()
    assemble = sim._assemble_materials_nu if lane == 'graded' else sim._assemble_materials
    mats, ds, _, *_ = assemble(grid)
    observed = observe(monkeypatch, sim)
    assert set(observed['update']) == set(observed['absorber']) == {0, 1, 2}
    values = []
    for c in range(3):
        eps = edge_mean(mats.eps_r, c)
        sigma = edge_mean(mats.sigma, c)
        # Written owner formula, independent numpy arithmetic; never a product coefficient helper.
        denominator = eps * EPS_0 + sigma * float(grid.dt) / 2
        if ds is not None:
            mask = ds[1][0] if isinstance(ds[1], (list, tuple)) else ds[1]
            fraction = edge_mean(mask, c)
            denominator += EPS_0 * 50. * float(grid.dt) / (2 * 5e-12 + float(grid.dt)) * fraction
        expected = np.asarray(float(grid.dt) / denominator, np.float32)
        actual = observed['absorber'][c][:6]
        update = observed['update'][c][:6]
        difference = int(ulps(actual, update).max())
        formula_difference = int(ulps(actual, expected[:6]).max())
        values.append(dict(component=c, update_ulps=difference, formula_ulps=formula_difference,
                           lossless_bits=int(actual[2, 7, 3].view(np.uint32))))
        # Literal SI coefficient bits captured from the
        # base tree (00a7328dc; unchanged on a8a23248d),
        # at x=2 in the pad, lossless epsilon=3.25 region y=7, z=3.
    record_property('judge2', json.dumps(dict(owner=owner, lane=lane, components=values)))
    print(lane, owner, json.dumps(values))
    for value in values:
        c = value['component']
        assert value['update_ulps'] <= coefficient_ulp_bar(owner), f'{owner} component {c}: absorber/update {value["update_ulps"]} float32 ULP'
        assert value['formula_ulps'] <= coefficient_ulp_bar(owner), f'{owner} component {c}: written Cb formula {value["formula_ulps"]} float32 ULP'
        assert value['lossless_bits'] == LOSSLESS_BITS[c]


LOSSLESS_BITS = (1032302835, 1032302835, 1032302835)
