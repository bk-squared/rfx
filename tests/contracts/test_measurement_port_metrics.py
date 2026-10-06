"""Port projectors read primal voltage lengths and transverse dual contours."""
from types import SimpleNamespace

import numpy as np
import pytest

from rfx.probes.probes import _metric_ampere_loop


@pytest.mark.parametrize('component,axes', [('ex', (1, 2)), ('ey', (2, 0)), ('ez', (0, 1))])
def test_ampere_contour_uses_both_transverse_duals(component, axes):
    # Independent linear H fields make each backward difference exactly one.
    indices = np.indices((4, 4, 4))
    state = SimpleNamespace(hx=indices[2]+indices[1],
                            hy=indices[0]+indices[2], hz=indices[1]+indices[0])
    lengths = (2., 3., 5.)
    grid = SimpleNamespace(duals=lambda axis: np.full(4, lengths[axis]))
    got = _metric_ampere_loop(state, (2, 2, 2), component, grid, (False,)*3)
    assert got == lengths[axes[1]] - lengths[axes[0]]
