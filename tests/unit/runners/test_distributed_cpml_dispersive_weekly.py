"""Weekly CPML dispersive forward parity on the R15l thinned prefix set."""
import pytest

from tests._distributed_cpml_cross_trace import WEEKLY_PREFIXES, cross_trace
from tests.unit.runners.test_distributed_edge_averaged_materials import (
    _cpml_1302_box,
    _debye_box,
    _devices,
)

pytestmark = [pytest.mark.distributed, pytest.mark.slow]


@pytest.mark.parametrize("boundary,lane", [("cpml", "fwd")])
@pytest.mark.parametrize("kind", ["debye", "lorentz", "mixed"])
def test_a_dispersive_model_matches_one_device(boundary, lane, kind):
    """Use the per-step 9-ULP cross-trace bar (rfx CLAUDE.md), since R15g
    replaces the vacuum-relative gate that collapsed with an exact reference.

    The mixed (Debye+Lorentz) row is not bit-equal (#1535): the slab path forms
    the ADE coefficients inside the loop, one device before it. It read
    1.92/7.25/7.25 ULP at 200/400/800 steps on Linux x86 (Mac 6.04), a 1.75-ULP
    margin; a red here is answered by forming them before the loop, not a mark.
    """
    cross_trace(lambda: _debye_box(boundary, lane, kind), lane, _devices(2),
                f"{kind}/{boundary}/{lane}", prefixes=WEEKLY_PREFIXES)


@pytest.mark.parametrize("lane,kind", [
    pytest.param(lane, kind, id=f"{kind}-{lane}")
    for lane in ("fwd", "fwd_graded") for kind in ("debye", "lorentz")
])
def test_a_dispersive_cpml_box_stays_finite_and_matches_one_device(lane, kind):
    """Use the per-step 9-ULP cross-trace bar (rfx CLAUDE.md), since R15g
    replaces the vacuum-relative gate that collapsed with an exact reference.
    """
    cross_trace(lambda: _cpml_1302_box(kind, lane), lane, _devices(2),
                f"cpml1302/{kind}/{lane}", prefixes=WEEKLY_PREFIXES)
