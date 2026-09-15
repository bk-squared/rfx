"""#1041/#1055 physics gate: a source in rank d's FIRST real cell must reach
rank d-1. Parametrised over BOTH distributed runners.

``rfx/runners/distributed_v2.py`` used to exchange the E ghost rows BEFORE
injecting sources, in both scan bodies. A soft source written into rank d's
first real cell was therefore absent from rank d-1's right ghost for one step,
and rank d-1's H update at its last real cell consumed a pre-injection E plane.
``distributed_nu.py`` had the same defect and fixed it in ``ac782d4f`` (#931
T3) by making the E exchange the last stage of the E half-step; #1041 measured
the same thing on v2 and moved it there too.

``rfx/runners/distributed.py`` -- the legacy ``jax.pmap`` lane -- had the
identical order in its own two scan bodies and reproduced v2's pre-fix seam
error TO FOUR DIGITS (#1041 ran it as a witness and found it was not one).
#1055 measured it on its own harness and applied the same reorder, so this
file drives both runners: ``runner="v2"`` through ``sim.run(devices=...)``,
the public dispatch, and ``runner="v1"`` through a direct
``rfx.runners.distributed.run_distributed`` call, which since #1049 is the
only way v1 is reachable at more than one device. ``NX_CELLS`` yields an EVEN
``nx``, which v1 requires (it refuses an odd ``nx`` rather than padding it),
so both runners see the same 2-rank decomposition and the same seam plane.

The reference is the SAME model on ONE device through the uniform lane, which
is the only lane in the repo without a seam. It is not a tolerance-free
comparison: the distributed and single-device lanes evaluate the same physics
through different kernels and different float32 fusions, so the interior-source
control below measures what that lane difference costs when NO source sits at a
seam, and the gate is set above it.

GATE DERIVATION (scripts/diagnostics/issue1041_v2_step_order.py and its #1055
sibling issue1055_v1_step_order.py, 2026-09-15, 2 virtual CPU devices, jax
0.10.2, float32, 300 steps; max|multi - single| divided by the single-device
probe peak):

    v2  fixture       pre-fix order    post-fix order   lane floor (interior)
    seam / pec, r0    1.859e-01        4.858e-06        5.100e-06
    seam / pec, r1    1.131e-01        3.915e-06        7.450e-05
    seam / cpml, r0   1.653e-01        1.894e-06        1.629e-06
    seam / cpml, r1   1.039e-01        1.948e-06        2.529e-05

    v1  fixture       pre-fix order    post-fix order   lane floor (interior)
    seam / pec, r0    1.859e-01        0.000e+00        0.000e+00
    seam / pec, r1    1.131e-01        0.000e+00        0.000e+00
    seam / cpml, r0   1.653e-01        1.579e-06        2.417e-06
    seam / cpml, r1   1.039e-01        1.580e-06        2.894e-05

v1's PEC body has a floor of exactly zero on this geometry: the pmap lane
reproduces the single-device uniform lane BIT-FOR-BIT there -- on the
interior-source control, on the mirror placement, and, once the order is
fixed, at the seam. Its CPML body does not (2.417e-06 / 2.894e-05 on the
control), and the corrected seam sits just under that. Both are properties of
this fixture, not promises about the lane -- the #1038 lock records v1 and v2
differing by 2.794e-09 on a different model -- so the gate below is NOT
tightened for v1: the same 1e-3 reads both runners, and the interior control
still MEASURES the floor per runner instead of assuming it.

THE DEFECT IS ONE-SIDED, and the third test here pins that. Only the RIGHT E
ghost is live: rank d-1's H at its LAST REAL cell consumes it. A rank's LEFT
E ghost feeds only its own H at that same index, and the H exchange overwrites
that H with the neighbour's authoritative value before anything reads it. So a
source in rank 0's LAST real cell was never wrong -- measured bit-identical
between the two orderings, 1.469e-06 / 1.008e-05 (pec) and 1.180e-06 /
3.485e-06 (cpml) against the single-device lane either way. That is also why
the #1038 bit-identity lock stays 13/13 green through both changes: every
``distributed_v2_*`` fixture in it puts its source exactly there, and so do
both ``distributed_v1_*`` fixtures -- ``distributed_v1_cpml_small`` (nx=34,
nx_per_rank=17, source at global node 16 = rank 0's last real cell) and
``distributed_v1_cpml_wide`` (nx=60, nx_per_rank=30, source at global node 29
= rank 0's last real cell).

``GATE = 1e-3`` is 13x the worst lane floor (7.450e-05, the interior-source
control's rank-1 probe on the same geometry) and 206x the worst post-fix seam
error (4.858e-06); the pre-fix ordering sits 186x ABOVE it. The gate is set
from the floor rather than from the post-fix seam error alone because the
post-fix seam error IS the floor -- pinning it tighter would pin float32
fusion, not the step order. Verified to FAIL on the pre-fix ordering by
checking out ``e7725d89:rfx/runners/distributed_v2.py`` and running this file:
both seam cases red at 1.859e-01 / 1.653e-01. The v1 rows were verified the
same way against ``6210e9fe:rfx/runners/distributed.py`` (#1055): both seam
cases red at the same 1.859e-01 / 1.653e-01, which is the four-digit agreement
that identified the two defects as one.
"""

# Simulate 2 devices on CPU. Must be set BEFORE importing JAX.
import os  # noqa: I001

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import jax  # noqa: E402

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2,
    reason=(
        "the #1041 seam gate needs >=2 devices. Run with "
        "XLA_FLAGS=--xla_force_host_platform_device_count=2. "
        "tests/unit/runners/test_device_count_sentinel.py FAILS rather than "
        "skips in that environment, so this skip is not a silent hole."
    ),
)

from rfx import Simulation  # noqa: E402

DX = 1e-3
NX_CELLS = 31        # -> nx = 32 nodes, so nx_per_rank = 16 and pad_x = 0
NYZ_CELLS = 15       # -> ny = nz = 16
SEAM_X = 16e-3       # global x node 16 == rank 1's FIRST real cell
SEAM_LO_X = 15e-3    # global x node 15 == rank 0's LAST real cell
INTERIOR_X = 8e-3    # 8 cells from the seam, inside rank 0
PROBE_LO_X = 12e-3   # 4 cells inside rank 0
PROBE_HI_X = 20e-3   # 4 cells inside rank 1
CENTER = 8e-3
N_STEPS = 300

GATE = 1e-3


def _build(boundary):
    kw = dict(freq_max=15e9,
              domain=(NX_CELLS * DX, NYZ_CELLS * DX, NYZ_CELLS * DX),
              dx=DX, boundary=boundary,
              cpml_layers=6 if boundary == "cpml" else 0)
    sim = Simulation(**kw)
    sim.add_probe((PROBE_LO_X, CENTER, CENTER), "ez")
    sim.add_probe((PROBE_HI_X, CENTER, CENTER), "ez")
    return sim


def _run_v2(sim):
    """The public dispatch: rfx/api/_execute.py sends this to distributed_v2."""
    return sim.run(n_steps=N_STEPS, devices=jax.devices()[:2])


def _run_v1(sim):
    """The legacy pmap lane. Since #1049 it is not re-exported by
    ``rfx.runners``, and ``sim.run(devices=...)`` never reaches it at 2
    devices, so the only way to drive it here is the full-path import."""
    from rfx.runners.distributed import run_distributed as v1
    return v1(sim, n_steps=N_STEPS, devices=jax.devices()[:2])


RUNNERS = {"v2": _run_v2, "v1": _run_v1}


def _rel_per_probe(source_x, boundary, runner):
    """max|multi - single| / peak(|single|) for each of the two probes."""
    sim_multi = _build(boundary)
    sim_multi.add_source((source_x, CENTER, CENTER), "ez")
    ts_multi = np.asarray(RUNNERS[runner](sim_multi).time_series)

    sim_single = _build(boundary)
    sim_single.add_source((source_x, CENTER, CENTER), "ez")
    ts_single = np.asarray(sim_single.run(n_steps=N_STEPS).time_series)

    assert ts_multi.shape == ts_single.shape == (N_STEPS, 2), (
        f"shapes: multi {ts_multi.shape}, single {ts_single.shape}")
    peaks = np.max(np.abs(ts_single), axis=0)
    assert np.all(peaks > 1e-6), (
        f"vacuous fixture: single-device probe peaks {peaks}")
    return np.max(np.abs(ts_multi - ts_single), axis=0) / peaks


@pytest.mark.parametrize("runner", ["v2", "v1"])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_seam_cell_source_reaches_the_neighbouring_rank(boundary, runner):
    """The gate. A source in rank 1's first real cell, probed from rank 0.

    Pre-fix this read 1.859e-01 (pec) / 1.653e-01 (cpml) on the rank-0
    probe, with the first divergence at step 4 -- the causal arrival of the
    source's own wavefront, not an accumulated drift. Both runners had the
    same order and read the same four digits.
    """
    rel = _rel_per_probe(SEAM_X, boundary, runner)
    assert rel[0] < GATE, (
        f"runner={runner} boundary={boundary}: source at the seam, probe 4 "
        f"cells into rank 0 deviates from the single-device lane by "
        f"{rel[0]:.3e} (gate {GATE:.0e}). The E ghost exchange must be the "
        "LAST stage of the E half-step (#1041 for v2, #1055 for v1).")
    assert rel[1] < GATE, (
        f"runner={runner} boundary={boundary}: source at the seam, probe "
        f"inside rank 1 deviates by {rel[1]:.3e} (gate {GATE:.0e})")


@pytest.mark.parametrize("runner", ["v2", "v1"])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_interior_source_control_sets_the_floor(boundary, runner):
    """The control the gate is derived from, and a floor-drift witness.

    With the source 8 cells from the seam the step order cannot matter (the
    exchanged rows are the neighbours' first/last REAL cells, which no source
    touches), and the two orderings were measured bit-identical here. What is
    left is the lane difference itself: 5.100e-06 / 7.450e-05 (pec) and
    1.629e-06 / 2.529e-05 (cpml) on v2; on v1, 0.000e+00 (pec -- bit-identical
    to the single-device lane) and 2.417e-06 / 2.894e-05 (cpml). If this test
    ever approaches GATE the gate above has stopped separating a step-order
    defect from lane noise.
    """
    rel = _rel_per_probe(INTERIOR_X, boundary, runner)
    assert np.all(rel < GATE / 10), (
        f"runner={runner} boundary={boundary}: the interior-source lane "
        f"floor is {rel}, "
        f"within a decade of the seam gate {GATE:.0e}. Re-derive the gate "
        "before trusting test_seam_cell_source_reaches_the_neighbouring_rank.")


@pytest.mark.parametrize("runner", ["v2", "v1"])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_last_cell_source_is_the_side_that_was_never_wrong(boundary, runner):
    """The mirror placement, and the reason the #1038 lock did not move.

    A source in rank 0's LAST real cell is exchanged into rank 1's LEFT
    ghost, whose only consumer is rank 1's H at that index -- which the H
    exchange overwrites with rank 0's authoritative H before the next E
    update reads it. Measured bit-identical between the two orderings. If
    this ever reds while the seam test above is green, the left ghost has
    acquired a consumer and the asymmetry documented in ``step_fn_cpml``
    stage 7 no longer holds.
    """
    rel = _rel_per_probe(SEAM_LO_X, boundary, runner)
    assert np.all(rel < GATE / 10), (
        f"runner={runner} boundary={boundary}: source at rank 0's last real "
        f"cell deviates by {rel} from the single-device lane; this placement "
        "is supposed to be insensitive to the E-exchange hook point "
        "(#1041, #1055).")
