"""Judge 2: observe actual distributed face multipliers on both active copies."""
import importlib.util
import json

import jax
import numpy as np
import pytest

from rfx.core.yee import EPS_0
from tests.contracts.test_absorber_curl_coefficients import (
    LOSSLESS_BITS, coefficient_scene, edge_mean, ulps,
)


def observe_faces(monkeypatch, sim, lane, face):
    """Spy the returned face multipliers inside the actual rank-local step."""
    import rfx.boundaries.cpml as cpml
    import rfx.runners.distributed_v2 as uniform
    import rfx.runners.distributed_nu as graded

    jax.clear_caches()
    observed = {}
    active = {0: 1, 2: 2} if face == 'x_lo' else {4: 0, 6: 2}
    context = {}

    def capture(index, value):
        if index not in active:
            return
        component = active[index]

        def save(rank, array):
            observed[int(rank), component] = np.asarray(array).copy()

        jax.debug.callback(save, context['rank'], value)

    module, name = (graded, '_apply_cpml_e_local_nu') if 'graded' in lane else (
        uniform, '_apply_cpml_e_distributed')
    original_step = getattr(module, name)

    def step(*args, **kwargs):
        context.update(rank=kwargs['rank'], counter=0)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(module, name, step)
    if importlib.util.find_spec('rfx.boundaries.electric_coefficient') is not None:
        import rfx.boundaries.electric_coefficient as coefficients
        original_faces = coefficients.slab_face_coefficients

        def faces(*args, **kwargs):
            out = original_faces(*args, **kwargs)
            for index in active:
                capture(index, out[index])
            return out

        monkeypatch.setattr(coefficients, 'slab_face_coefficients', faces)
    else:
        # Exact base tree: the same twelve operands are formed inline.
        original_ce = cpml._ce_si

        def coefficient(*args):
            out = original_ce(*args)
            if 'rank' in context:
                capture(context['counter'], out)
                context['counter'] += 1
            return out

        monkeypatch.setattr(cpml, '_ce_si', coefficient)

    result = sim.run(n_steps=1, skip_preflight=True, devices=jax.devices('cpu')[:2])
    jax.block_until_ready(result.time_series)
    jax.effects_barrier()
    return observed


@pytest.mark.parametrize('owner', ['plain', 'dielectric', 'debye', 'lorentz', 'mixed'])
@pytest.mark.parametrize('lane', ['distributed_uniform', 'distributed_graded'])
@pytest.mark.parametrize('face', ['x_lo', 'y_lo'])
def test_distributed_face_uses_owner_coefficient(monkeypatch, owner, lane, face, record_property):
    # The two faces exercise all three E components with an active psi/kappa
    # correction: x drives Ey/Ez, y drives Ex/Ez. No inactive-face surrogate.
    assert len(jax.devices('cpu')) >= 2
    sim = coefficient_scene(owner, lane, face)
    grid = sim._build_realized_grid()
    assemble = sim._assemble_materials_nu if 'graded' in lane else sim._assemble_materials
    mats, ds, _, *_ = assemble(grid)
    observed = observe_faces(monkeypatch, sim, lane, face)
    components = (1, 2) if face == 'x_lo' else (0, 2)
    assert set(observed) == {(rank, c) for rank in range(2) for c in components}
    values = []
    for c in components:
        denominator = edge_mean(mats.eps_r, c) * EPS_0 + edge_mean(mats.sigma, c) * float(grid.dt) / 2
        if ds is not None:
            mask = ds[1][0] if isinstance(ds[1], (list, tuple)) else ds[1]
            denominator += EPS_0 * 50. * float(grid.dt) / (2 * 5e-12 + float(grid.dt)) * edge_mean(mask, c)
        expected = np.asarray(float(grid.dt) / denominator, np.float32)
        if face == 'x_lo':
            # Only rank zero applies the x-low face. Its returned operand has
            # already been sliced past the one-cell communication ghost.
            actual, expected = observed[0, c], expected[:6]
            literal = actual[2, 7, 3]
        else:
            # y-low applies on both x ranks; remove communication ghosts and
            # stitch their real cells in the same order as the global grid.
            actual = np.concatenate([observed[rank, c][1:-1] for rank in range(2)])[:grid.shape[0]]
            expected = expected[:, :6]
            literal = actual[7, 2, 3]
        difference = int(ulps(actual, expected).max())
        values.append(dict(component=c, formula_ulps=difference,
                           lossless_bits=int(literal.view(np.uint32))))
    record_property('judge2', json.dumps(dict(owner=owner, lane=lane, face=face, components=values)))
    print(lane, face, owner, json.dumps(values))
    for value in values:
        assert value['formula_ulps'] <= 2, f'{lane}/{face}/{owner} component {value["component"]}: written Cb formula {value["formula_ulps"]} float32 ULP'
        assert value['lossless_bits'] == LOSSLESS_BITS[value['component']]
