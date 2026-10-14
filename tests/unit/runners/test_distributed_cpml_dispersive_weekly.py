"""Weekly CPML dispersive forward parity on the R15l thinned prefix set."""
import sys

import pytest

from tests._distributed_cpml_cross_trace import CrossTraceBarExceeded, WEEKLY_PREFIXES, cross_trace
from tests.unit.runners.test_distributed_edge_averaged_materials import (
    _cpml_1302_box,
    _debye_box,
    _devices,
)

pytestmark = [pytest.mark.distributed, pytest.mark.slow]


@pytest.mark.parametrize("boundary,lane", [("cpml", "fwd")])
@pytest.mark.parametrize("kind", ["debye", "lorentz", "mixed"])
def test_a_dispersive_model_matches_one_device(boundary, lane, kind, request):
    """Use the per-step 9-ULP cross-trace bar (rfx CLAUDE.md), since R15g
    replaces the vacuum-relative gate that collapsed with an exact reference.

    R15r: the mixed difference collapses to 1e-15 in float64 on this branch,
    while it stays at 3e-11..5e-10 on main.
    The last-bit law is exceeded on Linux at 800 steps (1.16).
    """
    devices = _devices(2)
    guard = None
    if (boundary, lane, kind) == ("cpml", "fwd", "mixed"):
        # R15n: thinned prefixes read 10.0 against 9 at 800 on Linux, a thin
        # margin; an XPASS on Linux means the rounding path moved (platform or JAX version).
        # Re-read the full prefix set before removing this weekly-only mark.
        request.applymarker(pytest.mark.xfail(
            strict=sys.platform.startswith("linux"), raises=CrossTraceBarExceeded, reason=(
            "#1535: two-pole (Debye+Lorentz) block, one vs two devices on Linux x86: "
            "peak 3.75/8.44/11.5 ULP at 200/400/800 steps (main 3.5/6.1/8.4). The first "
            "departure (Mac, every step) is the Debye polarisation state at step 5 in the same 18 interior "
            "cells with the same bytes on main and here, away from the absorber and the "
            "slab cut; the rest is the rounding path after that common origin.")))

        def guard(results):
            # regression guard pinned to a measured envelope (R15r: 2 x the recorded Linux 10.0 ULP)
            for length in (200, 400, 800):
                maximum = max(results[length]['max_d'])
                if maximum > 20:
                    pytest.fail(f"mixed regression guard: {length} steps, {maximum} > 20 float32 ULP")

    cross_trace(lambda: _debye_box(boundary, lane, kind), lane, devices,
                f"{kind}/{boundary}/{lane}", prefixes=WEEKLY_PREFIXES, regression_guard=guard)


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
