"""Public-entry scenes and C4's observation point for the one-PEC contract."""
import sys

import numpy as np

from rfx import Box, Simulation, Sphere
from rfx.boundaries.spec import Boundary, BoundarySpec


def scene(name, *, conformal=True, ports=1, snap='declared'):
    short = name == 'P'
    a, b, length, dx, pads = ((.04, .02, .12, .003, 10) if short else
                             (.02286, .01016, .14, .00127, 16))
    sim = Simulation(freq_max=10e9 if short else 12.4e9, domain=(length, a, b),
                     dx=dx, precision='float32', cpml_layers=pads, snap=snap,
                     boundary=BoundarySpec(x='cpml',
                         y=Boundary('pec', 'pec', conformal=conformal),
                         z=Boundary('pec', 'pec', conformal=conformal and short)))
    if name == 'S':
        shapes = [Sphere((.070 + a / 108, .01273, .00548), .0025)]
    elif short:
        shapes = [Box((.084, 0., 0.), (.087, a, b))]
    elif name == 'G':
        shapes = [Box((lo, .00762, .00254), (hi, .01524, .00762))
                  for lo, hi in ((.06985, .07112), (.07239, .07366))]
    else:
        shapes = [Box((.06985, .00762, .00254), (.07366, .01524, .00762))]
    for shape in shapes:
        sim.add(shape, material='pec')
    f0 = 6e9 if short else 10e9
    for position, direction, label in ((.010 if short else .03048, '+x', 'left'),
                                        (.099 if short else .11176, '-x', 'right'))[:ports]:
        sim.add_waveguide_port(position, direction=direction, mode=(1, 0), mode_type='TE',
                               freqs=[f0], f0=f0, bandwidth=.5, name=label)
    return sim


def capture_setups(monkeypatch):
    """No public read exposes post-restrict_node_pec operands; wrap as C4 did."""
    import jax
    from rfx.stepping import uniform_setup
    from rfx.geometry import conformal, conformal_cells

    captured = []
    fractions = []
    original = uniform_setup._build_step_setup
    original_fractions = conformal_cells.compute_conformal_fractions

    def observe_fractions(grid, shapes, **kwargs):
        result = original_fractions(grid, shapes, **kwargs)
        fractions.append((grid, result))
        return result

    monkeypatch.setattr(conformal, 'compute_conformal_fractions', observe_fractions)
    monkeypatch.setattr(conformal_cells, 'compute_conformal_fractions', observe_fractions)

    def observe(grid, materials, **kwargs):
        setup = original(grid, materials, **kwargs)
        ctx = setup.ctx_kwargs
        realized = ctx['materials']
        tensor = realized.components.tensor
        masks = ctx['pec_edge_masks']
        weights = None if tensor is None else tensor.conformal_weights
        areas = next((a for g, (w, a) in reversed(fractions)
                      if g is grid and weights is not None and
                      all(np.array_equal(x, y) for x, y in zip(w, weights))), None)
        captured.append(dict(grid=grid,
            masks=tuple(np.zeros(grid.shape, bool) for _ in range(3)) if masks is None else
                  tuple(np.asarray(a).copy() for a in masks),
            weights=None if tensor is None or tensor.conformal_weights is None else
                    tuple(np.asarray(a).copy() for a in tensor.conformal_weights),
            materials=tuple(np.asarray(a).copy() for a in jax.tree.leaves(realized)),
            areas=None if areas is None else tuple(np.asarray(a).copy() for a in areas)))
        return setup

    for name, module in list(sys.modules.items()):
        if name.startswith('rfx.') and module is not None:
            for key, value in list(vars(module).items()):
                if value is original:
                    monkeypatch.setattr(module, key, observe)
    return captured
