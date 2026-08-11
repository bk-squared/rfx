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
run on CPU in the default local suite -- `tests/test_mixed_precision.py`'s
numeric mixed-precision tests are module-wide `gpu`-marked and would
otherwise be the only place this behavior is covered, deselected from the
default local run.

Also covers a second, related precision guard added in the same follow-up:
issue #644 (precision="mixed" + CPML, a pre-existing defect #630 newly made
reachable from forward() -- see the module section below for the full
mechanism).
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
# Issue #644: precision="mixed" + CPML is a pre-existing, NOT-#630-caused
# defect (measured identical via run() on the pre- and post-#630 trees; the
# CPML psi_* carry follows field_dtype=float16 but the CPML coefficients are
# hard-pinned to float32, breaking the lax.scan carry contract). #630 made
# forward() honour precision= for the first time (previously it silently
# ignored precision= and always ran float32), which newly makes this
# pre-existing defect reachable from forward() -- these tests pin that the
# NEW reachability surfaces a clear rfx-layer NotImplementedError instead of
# degrading to a raw JAX carry-dtype TypeError, and that float32/float64
# (unaffected -- promote_types(float64, float32) == float64 matches
# storage, no carry-dtype mismatch) and UPML (measured: no analogous
# hard-float32 psi carry) are untouched by this guard. Fixing #644 itself
# (giving psi_* and the CPML coefficients one consistent dtype policy) is
# explicitly out of scope here -- see issue #644.
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


def test_forward_mixed_cpml_raises_clean_notimplementederror_not_raw_typeerror():
    sim = _cpml_sim(precision="mixed")
    with pytest.raises(NotImplementedError, match="#644"):
        sim.forward(n_steps=5, skip_preflight=True)


def test_forward_mixed_cpml_per_face_boundary_spec_also_caught():
    """The guard reads self._boundary_spec (always normalized, even from a
    legacy scalar boundary=), so a per-face BoundarySpec with only ONE
    CPML face must trip it too -- not just the all-faces-cpml scalar case."""
    from rfx.boundaries.spec import BoundarySpec, Boundary

    sim = Simulation(
        freq_max=5.0e9, domain=(0.02, 0.02, 0.02),
        boundary=BoundarySpec(x="pec", y="pec", z=Boundary(lo="cpml", hi="cpml")),
        precision="mixed",
    )
    sim.add_source(
        position=(0.01, 0.01, 0.01), component="ez",
        amplitude_kind="current",
    )
    with pytest.raises(NotImplementedError, match="#644"):
        sim.forward(n_steps=5, skip_preflight=True)


def test_forward_float64_cpml_is_unaffected_by_the_644_guard():
    import jax
    jax.config.update("jax_enable_x64", True)
    sim = _cpml_sim(precision="float64")
    result = sim.forward(n_steps=5, skip_preflight=True)
    assert result is not None


def test_forward_float32_cpml_is_unaffected_by_the_644_guard():
    sim = _cpml_sim(precision="float32")
    result = sim.forward(n_steps=5, skip_preflight=True)
    assert result is not None
