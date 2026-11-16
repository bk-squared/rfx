"""Literal boundary cases for coaxial extraction status."""

import numpy as np
import pytest

from rfx.sparams._coax_status import coax_extraction_status


@pytest.mark.parametrize(
    "annulus_cells,residual,traced,expected",
    [
        pytest.param(3.4, [1.0], False, "under_resolved", id="under-resolved-concrete"),
        pytest.param(3.4, None, True, "under_resolved", id="under-resolved-traced"),
        pytest.param(3.5, None, True, "differentiable", id="resolved-traced"),
        pytest.param(3.5, [0.0, 0.1], False, "passed", id="residual-at-threshold"),
        pytest.param(3.5, [0.0, 0.1000001], False, "contaminated", id="residual-above-threshold"),
        pytest.param(3.5, [1.0], True, "differentiable", id="traced-ignores-residual"),
    ],
)
def test_coax_extraction_status(annulus_cells, residual, traced, expected):
    recurrence_residual = None if residual is None else np.asarray(residual)
    assert coax_extraction_status(
        annulus_cells, recurrence_residual, traced=traced,
    ) == expected
