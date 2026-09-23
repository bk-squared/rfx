"""A lane still on the pre-decision lumped diagonal has to say so.

The uniform lumped lane moved to the driven terminal reflection on 2026-09-21
(scripts/diagnostics/lumped_port_known_load_line.py). One lane did not: the
experimental subgridded runner, whose sampling slot depends on the runtime
``inject_sources_before_e_coupling`` flag, so no single dt/2 offset is
derivable there. It still returns a lumped S-matrix built the old way — on a
known load, the reciprocal of the physical reflection.

That is an acceptable state for a diagnostic. Returning it SILENTLY is not,
and a warning nothing pins is one edit away from being silent: this scan is
here because a `warnings.warn` has no test unless someone writes one, and the
subgrid lane is expensive enough that nothing drives it end to end.

The scan is static on purpose — it costs milliseconds and it fails for the one
reason that matters, the guard disappearing from the block that returns the
number.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

from rfx.probes.probes import PreDecisionLumpedDiagonalWarning

_SUBGRIDDED = (Path(__file__).resolve().parents[2]
               / "rfx" / "runners" / "subgridded.py")


def test_the_warning_is_its_own_category_under_user_warning():
    """Its own class, so a caller can filter or escalate just this one."""
    assert issubclass(PreDecisionLumpedDiagonalWarning, UserWarning)
    assert PreDecisionLumpedDiagonalWarning is not UserWarning

    with pytest.warns(PreDecisionLumpedDiagonalWarning):
        warnings.warn("probe", PreDecisionLumpedDiagonalWarning)


def test_the_subgridded_lumped_s_matrix_block_still_carries_the_warning():
    """The block that assembles the diagnostic S-matrix warns before it.

    Pinned against the source rather than a solve: driving this lane end to
    end needs a full subgrid run, which is why it had no coverage to begin
    with.
    """
    source = _SUBGRIDDED.read_text()
    assert "lumped_sparam_v_dft_f" in source, (
        "the subgridded runner no longer assembles a lumped S-matrix — if "
        "that lane is gone, delete this test with it")

    tree = ast.parse(source)
    warns_here = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "warn"
        and any(isinstance(a, ast.Name)
                and a.id == "PreDecisionLumpedDiagonalWarning"
                for a in node.args)
    ]
    assert warns_here, (
        "rfx/runners/subgridded.py returns a lumped S-matrix on the "
        "pre-decision convention without raising "
        "PreDecisionLumpedDiagonalWarning. The number is knowingly not the "
        "physical reflection; the lane has to say so.")
