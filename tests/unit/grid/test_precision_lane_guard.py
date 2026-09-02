"""Issue #630 follow-up: precision="mixed"/"float64" must fail LOUD, not
silently degrade to float32, on lanes whose runner does not thread
field_dtype.

`field_dtype` is threaded only by `rfx/runners/uniform.py`. The
non-uniform-mesh, distributed, distributed-NU, and subgridded runners have
zero `field_dtype` occurrences (grepped as part of the #630 follow-up
review), so before this fix a non-uniform-mesh or distributed simulation
with `precision="mixed"`/`"float64"` would silently run float32 fields
while reporting no error at all -- the SILENT_WRONG class this repo removes
on sight. `_dispatch_plan` (the single lane-decision point for `run()` and
`forward()`) now rejects the combination with `NotImplementedError` before
any compute runs, on every affected lane. `_validate_cfg_precision_x64`
also warns in advance for the non-uniform-mesh case (the distributed case
is a call-time kwarg invisible to preflight, so the dispatch-time raise is
the ONLY enforcement point there -- see that check's docstring).

These tests are deliberately NOT `pytest.mark.gpu`: they only exercise the
dispatch-time guard (a Python-level raise before any FDTD compute), so they
run on CPU in the default local suite. (`tests/unit/grid/test_mixed_precision.py` used
to be module-wide `gpu`-marked, which is why it could not serve as that
coverage; issue #644 removed that mark after measuring the file at 6.5 s on
CPU, so it now runs in the default lane too.)

Also covers the `forward()` + CPML precision matrix, which issue #644 turned
from a guarded-off combination into a working one -- see the module section
below.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation


def _nu_sim(*, precision: str) -> Simulation:
    dz = np.full(20, 1.0e-3)
    sim = Simulation(
        freq_max=5.0e9, domain=(0.02, 0.02, 0.0), dz_profile=dz,
        boundary="pec", precision=precision,
    )
    sim.add_source(
        position=(0.01, 0.01, 0.005), component="ez",
        amplitude_kind="current",
    )
    return sim


def _uniform_sim(*, precision: str) -> Simulation:
    sim = Simulation(
        freq_max=5.0e9, domain=(0.02, 0.02, 0.02), boundary="pec",
        precision=precision,
    )
    sim.add_source(
        position=(0.01, 0.01, 0.01), component="ez",
        amplitude_kind="current",
    )
    return sim


@pytest.mark.parametrize("precision", ["mixed", "float64"])
def test_nonuniform_run_rejects_nonfloat32_precision(precision):
    sim = _nu_sim(precision=precision)
    with pytest.raises(NotImplementedError, match="run_nonuniform"):
        sim.run(n_steps=5, skip_preflight=True)


@pytest.mark.parametrize("precision", ["mixed", "float64"])
def test_nonuniform_forward_rejects_nonfloat32_precision(precision):
    sim = _nu_sim(precision=precision)
    with pytest.raises(NotImplementedError, match="fwd_nonuniform"):
        sim.forward(n_steps=5, skip_preflight=True)


def test_nonuniform_precision_preflight_warns_in_advance():
    """The preflight warning fires BEFORE the dispatch-time raise -- it is
    an advisory, not the enforcement point (that is _dispatch_plan)."""
    sim = _nu_sim(precision="mixed")
    report = sim.preflight(strict=False)
    hits = report.by_code("precision_nonuniform_lane_unsupported")
    assert len(hits) == 1, (
        f"expected exactly one nonuniform-lane precision warning; got "
        f"{[getattr(i, 'code', None) for i in report]}"
    )


def test_distributed_forward_rejects_nonfloat32_precision():
    """distributed=True is a call-time forward() kwarg invisible to
    preflight (Simulation-construction-time); _dispatch_plan is the only
    place this can be caught, so this is a distinct code path from the
    non-uniform-mesh test above, not a duplicate."""
    sim = _nu_sim(precision="float64")  # DP3: distributed forward is NU-only
    with pytest.raises(NotImplementedError, match="run_distributed|fwd_distributed_nu"):
        sim.forward(distributed=True, n_steps=5, skip_preflight=True)


def test_subgridded_run_rejects_nonfloat32_precision():
    sim = _uniform_sim(precision="mixed")
    # Minimal refinement dict -- only the lane-dispatch guard is under
    # test here, not subgridding physics, so the other fields don't need
    # to describe a valid refinement region.
    sim._refinement = {
        "z_range": (0.005, 0.015), "ratio": 2, "xy_margin": 0.0,
        "tau": None, "validation": None, "topology": None,
    }
    with pytest.raises(NotImplementedError, match="run_subgridded"):
        sim.run(n_steps=5, skip_preflight=True)


@pytest.mark.parametrize("precision", ["float32", "mixed", "float64"])
def test_uniform_lane_unaffected_by_the_new_guard(precision):
    """Negative control: the uniform lane is the ONE lane this knob is
    documented to support, so none of it should raise here. float64
    needs x64 enabled or it silently downcasts -- irrelevant to this
    guard (a separate, already-covered concern), so only dtype-reachable
    aspects are checked, not the float64 numeric path itself."""
    sim = _uniform_sim(precision=precision)
    result = sim.run(n_steps=3, skip_preflight=True)
    assert result is not None


# ---------------------------------------------------------------------------
# Issue #644, RESOLVED: precision="mixed" + CPML now works through forward().
#
# It was a pre-existing, NOT-#630-caused defect (measured identical via run()
# on the pre- and post-#630 trees): the CPML psi_* carry followed
# field_dtype=float16 while the CPML coefficients are hard-pinned float32, so
# `psi = b*psi + c*curl` promoted to float32 and broke the lax.scan carry
# contract. #630 made forward() honour precision= for the first time, which
# newly exposed it there, and #630 shipped a temporary forward() guard raising
# NotImplementedError rather than leak a raw JAX carry-dtype TypeError.
#
# #644 fixed the root cause -- the psi_* arrays are ACCUMULATION state and are
# now allocated at promote_types(field_dtype, float32), so they never sit
# below float32 -- and REMOVED that guard. These tests were the guard's
# regression pins; they are kept (rather than deleted) and inverted to assert
# SUCCESS, because the two cases they encode are exactly the two that must
# keep working: a plain scalar boundary="cpml", and a per-face BoundarySpec
# carrying only ONE cpml face (the guard read self._boundary_spec, so the
# per-face case is a genuinely distinct path, not a restatement).
#
# Caveat worth knowing when reading these: "mixed" + CPML is correct but
# raises the absorber's residual floor (~-76 dB -> ~-59.5 dB on the fixture
# in tests/unit/grid/test_mixed_precision.py). See the precision= docstring in
# rfx/api/__init__.py. These tests pin reachability, not absorber quality --
# tests/unit/grid/test_mixed_precision.py owns the numeric assertions.
# ---------------------------------------------------------------------------

def _cpml_sim(*, precision: str) -> Simulation:
    sim = Simulation(
        freq_max=5.0e9, domain=(0.02, 0.02, 0.02), boundary="cpml",
        precision=precision,
    )
    sim.add_source(
        position=(0.01, 0.01, 0.01), component="ez",
        amplitude_kind="current",
    )
    return sim


def test_forward_mixed_cpml_works():
    """Was NotImplementedError (the #630-era guard), and a raw lax.scan carry
    TypeError before that. Both are fixed at the root -- see issue #644."""
    sim = _cpml_sim(precision="mixed")
    result = sim.forward(n_steps=5, skip_preflight=True)
    assert result is not None


def test_forward_mixed_cpml_per_face_boundary_spec_also_works():
    """A per-face BoundarySpec with only ONE cpml axis is a distinct path.

    The removed guard read ``self._boundary_spec`` (always normalized, even
    from a legacy scalar ``boundary=``), so this case tripped it separately
    from the all-faces-cpml scalar case. The fix is in ``init_cpml``, which
    every cpml face routes through, so this must now work too -- keeping the
    case pins that the psi dtype policy is not somehow scalar-boundary-only.

    ``cpml_layers=4`` is load-bearing and deliberate, NOT a tolerance being
    loosened to get green. This test previously used the default 16 layers
    and passed only because the guard raised BEFORE any CPML compute ran --
    it never exercised the per-face CPML path at all. With the guard gone the
    fixture turned out to be broken independently of precision: a per-face
    spec whose PEC axes are narrower than ``cpml_layers`` (here 8 cells vs 16
    layers) dies in ``apply_cpml_e`` with a SHAPE error,
    ``mul got incompatible shapes for broadcasting: (16,1,1) vs (8,8,40)``,
    because the x/y face slices ``[:n]`` still run on unpadded axes. MEASURED
    on the unpatched parent commit: that combination fails identically at
    ``precision="float32"`` through BOTH ``run()`` and ``forward()``, so it is
    a pre-existing per-face CPML geometry defect, not a precision defect and
    not caused by issue #644. It was reported as issue #647 and is now FIXED
    (``rfx/boundaries/cpml.py`` clamps the CPML scratch buffer per axis; the
    #647 regression lock lives in
    ``tests/unit/boundaries/test_boundary_spec_cpml_budget.py``), so 16 layers no longer
    crashes. ``cpml_layers=4`` stays pinned anyway: this test is about the
    psi dtype policy, and 4 layers is the smallest fixture that reaches the
    CPML compute -- moving it now would only make the dtype assertion depend
    on the buffer-clamp path as well. Verified red-then-green: at 4 layers
    this raises the #644 lax.scan carry TypeError on the parent commit and
    passes here.
    """
    from rfx.boundaries.spec import BoundarySpec, Boundary

    sim = Simulation(
        freq_max=5.0e9, domain=(0.02, 0.02, 0.02), cpml_layers=4,
        boundary=BoundarySpec(x="pec", y="pec", z=Boundary(lo="cpml", hi="cpml")),
        precision="mixed",
    )
    sim.add_source(
        position=(0.01, 0.01, 0.01), component="ez",
        amplitude_kind="current",
    )
    result = sim.forward(n_steps=5, skip_preflight=True)
    assert result is not None


def test_forward_float64_cpml_works():
    # Scoped x64, per this repo's own rule (CLAUDE.md, and see this
    # package's own precision docstring): never flip jax_enable_x64
    # process-globally in a test -- it leaks to every test scheduled after
    # this one in the same pytest-split worker. Use the repo's version-
    # robust scoped context (jax.experimental.enable_x64 upstream, or the
    # tests/_x64_compat.py shim on newer JAX that removed it), matching
    # the idiom in tests/unit/autodiff/test_coax_two_port_ad.py.
    try:
        from jax import enable_x64
    except ImportError:  # older JAX (< ~0.4.31)
        from tests._x64_compat import enable_x64

    sim = _cpml_sim(precision="float64")
    with enable_x64():
        result = sim.forward(n_steps=5, skip_preflight=True)
    assert result is not None


def test_forward_float32_cpml_works():
    sim = _cpml_sim(precision="float32")
    result = sim.forward(n_steps=5, skip_preflight=True)
    assert result is not None
