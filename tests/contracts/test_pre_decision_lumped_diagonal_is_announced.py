"""A lane still on the pre-decision lumped diagonal has to say so.

The uniform lumped lane moved to the driven terminal reflection on 2026-09-21
(scripts/diagnostics/lumped_port_known_load_line.py). One lane did not: the
experimental subgridded runner, whose sampling slot depends on the runtime
``inject_sources_before_e_coupling`` flag, so no single dt/2 offset is
derivable there. It still returns a lumped S-matrix built the old way — on a
known load, the reciprocal of the physical reflection.

The public run must emit its dedicated warning when it returns that
diagnostic S-matrix.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from rfx.probes.probes import PreDecisionLumpedDiagonalWarning
from tests.unit.sparams.test_plain_source_refusal import _subgrid_model


def test_the_warning_is_its_own_category_under_user_warning():
    """Its own class, so a caller can filter or escalate just this one."""
    assert issubclass(PreDecisionLumpedDiagonalWarning, UserWarning)
    assert PreDecisionLumpedDiagonalWarning is not UserWarning

    with pytest.warns(PreDecisionLumpedDiagonalWarning):
        warnings.warn("probe", PreDecisionLumpedDiagonalWarning)


def test_the_subgridded_lumped_s_matrix_block_still_carries_the_warning():
    """A returned diagnostic S-matrix is accompanied by its typed warning."""
    sim = _subgrid_model(source=False)
    with pytest.warns(PreDecisionLumpedDiagonalWarning) as captured:
        result = sim.run(n_steps=8, compute_s_params=True, skip_preflight=True)

    warning_count = sum(
        issubclass(warning.category, PreDecisionLumpedDiagonalWarning)
        for warning in captured
    )
    print(
        f"subgridded lumped S: dedicated_warning_count={warning_count} bar>=1; "
        f"s_params_present={int(result.s_params is not None)} bar=1",
        file=sys.stderr,
    )
    assert result.s_params is not None, "subgridded run returned no diagnostic S-matrix"
