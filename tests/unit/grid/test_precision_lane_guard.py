"""Single-device graded Yee accepts float64 and retains mixed refusals.

Distributed and subgridded precision refusals keep their existing messages.
Uniform mixed and float64 CPML coverage remains unchanged.
Every domain length written as ``N * (c / 5 GHz / 20)`` is a whole number of cells of
dx, so the solved domain equals the declared one.
"""

from __future__ import annotations

from contextlib import nullcontext
from tests._x64_compat import enable_x64

import numpy as np
import pytest

from rfx import Simulation


def _nu_sim(*, precision: str) -> Simulation:
    dz = np.full(20, 1.0e-3)
    sim = Simulation(
        freq_max=5.0e9, domain=(7 * (299792458.0 / 5.0e9 / 20.0), 7 * (299792458.0 / 5.0e9 / 20.0), 0.0), dx=299792458.0 / 5.0e9 / 20.0, dz_profile=dz,
        boundary="pec", precision=precision,
    )
    sim.add_source(
        position=(0.01, 0.01, 0.005), component="ez",
        amplitude_kind="current",
    )
    return sim


def _uniform_sim(*, precision: str) -> Simulation:
    sim = Simulation(
        freq_max=5.0e9, domain=(7 * (299792458.0 / 5.0e9 / 20.0), 7 * (299792458.0 / 5.0e9 / 20.0), 7 * (299792458.0 / 5.0e9 / 20.0)), dx=(299792458.0 / 5.0e9 / 20.0), boundary="pec",
        precision=precision,
    )
    sim.add_source(
        position=(0.01, 0.01, 0.01), component="ez",
        amplitude_kind="current",
    )
    return sim


@pytest.mark.parametrize("precision", ["mixed"])
def test_nonuniform_run_rejects_nonfloat32_precision(precision):
    sim = _nu_sim(precision=precision)
    with enable_x64() if precision == "float64" else nullcontext(), pytest.raises(NotImplementedError, match="run_nonuniform"):
        sim.run(n_steps=5, skip_preflight=True)


@pytest.mark.parametrize("precision", ["mixed"])
def test_nonuniform_forward_rejects_nonfloat32_precision(precision):
    sim = _nu_sim(precision=precision)
    with enable_x64() if precision == "float64" else nullcontext(), pytest.raises(NotImplementedError, match="fwd_nonuniform"):
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
    non-uniform-mesh test above, not a duplicate.

    The fixture is the non-uniform model because distributed forward runs only on the
    non-uniform mesh.
    """
    sim = _nu_sim(precision="float64")
    with enable_x64(), pytest.raises(NotImplementedError, match="run_distributed|fwd_distributed_nu"):
        sim.forward(distributed=True, n_steps=5, skip_preflight=True)


def test_subgridded_run_rejects_nonfloat32_precision():
    """The refinement dict is minimal: only the lane-dispatch guard is under test, not
    subgridding physics, so the other fields need not describe a valid refinement
    region."""
    sim = _uniform_sim(precision="mixed")
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
    with enable_x64() if precision == "float64" else nullcontext():
        result = sim.run(n_steps=3, skip_preflight=True)
    assert result is not None


def _cpml_sim(*, precision: str) -> Simulation:
    sim = Simulation(
        freq_max=5.0e9, domain=(7 * (299792458.0 / 5.0e9 / 20.0), 7 * (299792458.0 / 5.0e9 / 20.0), 7 * (299792458.0 / 5.0e9 / 20.0)), dx=(299792458.0 / 5.0e9 / 20.0), boundary="cpml",
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
        freq_max=5.0e9, domain=(7 * (299792458.0 / 5.0e9 / 20.0), 7 * (299792458.0 / 5.0e9 / 20.0), 7 * (299792458.0 / 5.0e9 / 20.0)), dx=(299792458.0 / 5.0e9 / 20.0), cpml_layers=4,
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
    try:
        from jax import enable_x64
    except ImportError:
        from tests._x64_compat import enable_x64

    sim = _cpml_sim(precision="float64")
    with enable_x64():
        result = sim.forward(n_steps=5, skip_preflight=True)
    assert result is not None


def test_forward_float32_cpml_works():
    sim = _cpml_sim(precision="float32")
    result = sim.forward(n_steps=5, skip_preflight=True)
    assert result is not None


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_nonuniform_float64_accepted(entry):
    with enable_x64():
        sim = _nu_sim(precision="float64")
        sim.add_probe((.01, .01, .005), "ez")
        assert not sim.preflight(strict=False).by_code("precision_nonuniform_lane_unsupported")
        result = getattr(sim, entry)(n_steps=5, skip_preflight=True)
        assert result.time_series.dtype == np.float64
