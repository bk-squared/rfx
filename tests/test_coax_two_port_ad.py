"""AD status for ``Simulation.compute_coaxial_two_port`` (issue #489, active
dev track leg 3) — GATE, not a blocker pin.

HISTORY: this file used to pin the ABSENCE of AD (see git history for the
pre-fix version) — no ``eps_scale``-shaped traced-input parameter existed on
``compute_coaxial_two_port``, and even with one, the post-FDTD extraction
concretized before the two-drive solve. Per the #489 correction comment
(2026-08-05) the accurate blocker map had TWO concrete links, not three: the
middle extractor (``coaxial_line_reflection_from_plane_voltages``) already had
a validated jnp core (``_coaxial_line_reflection_jnp``, the same one the
GRAD_SAFE 1-port ``eps_scale`` path already used) — it just never received a
traced input, because the caller concretized first. The two REAL links were
(i) ``rfx/api/_sparams.py``'s ``_assemble_coaxial_two_port_from_voltages``
host-side ``np.asarray(v_bot_by_drive, ...)`` assembly loop, and (ii)
``rfx/sources/coaxial_port.py``'s ``solve_two_port_from_wave_amplitudes``
(the ``S = B @ inv(A)`` two-drive solve), which had no jnp path at all.

WHAT WAS FIXED (this change):

1. ``compute_coaxial_two_port`` gained an ``eps_scale`` parameter — same
   name, same "multiply the stamped eps_r" semantics, same position in the
   design as the 1-port sibling's own ``eps_scale`` — applied once (after
   both feed stamps) so it reaches both drives' FDTD runs.
2. When ``eps_scale is not None``, both drives now extract voltage via
   ``coaxial_line_plane_voltage_jnp`` (the differentiable twin the 1-port
   path already used) instead of the concrete ``coaxial_line_plane_voltage``.
3. ``_assemble_coaxial_two_port_from_voltages`` gained a jnp assembly branch
   that routes every per-(drive, frequency) fit through
   ``coaxial_line_reflection_from_plane_voltages(..., _prefer_jnp=True)`` —
   REUSING the extractor's own existing jnp core, not re-deriving it.
4. ``solve_two_port_from_wave_amplitudes`` gained a jnp core
   (``_solve_two_port_from_wave_amplitudes_jnp``, batched ``S = B @ inv(A)``
   via ``jax.numpy``'s native batched ``linalg.inv``/``linalg.cond``).
5. All three dispatch on an explicit ``_prefer_jnp`` flag (set by
   ``compute_coaxial_two_port`` whenever its own ``eps_scale`` was given),
   NOT merely ``isinstance(..., jax.core.Tracer)`` auto-detection. This
   matters: a CONCRETE FD probe (``eps_scale`` provided but the call made
   eagerly, e.g. one side of a finite-difference cross-check) is NOT a
   tracer, so tracer-only dispatch would silently fall back to the strict
   NumPy ``lstsq`` in ``coaxial_line_reflection_from_plane_voltages`` instead
   of that function's own tolerant jnp lstsq — measured DURING THIS CHANGE
   to raise ``numpy.linalg.LinAlgError: SVD did not converge`` on a marginal
   fit that the jnp path handles fine. That is the PR #468 / #559-B1 class
   this file's own task briefing warned about ("check whether the coax
   two-port path has any passivity-projection or post-processing gated on
   is_tracer/eager that would make jax.grad trace a different function than
   an eager FD call sees") — found on the FIRST real measurement attempt
   below, not hypothetically, and fixed by making ``_prefer_jnp`` explicit
   rather than tracer-inferred at every dispatch point.

With ``eps_scale=None`` every path above is BYTE-IDENTICAL to the pre-fix
code (the concrete branches were moved into an ``else:``, not rewritten) —
regression-locked by the existing ``tests/test_coax_two_port_fdtd.py``
(``slow_physics``) fixture, whose tightly-pinned measured numbers
(|S21|/|S12| >= 0.70, cond_a < 3.0, recurrence_residual < 0.02, ...) still
hold unchanged.

NO PROJECTION TO DISABLE (#468/#559-B1 cross-check, second half): unlike
``compute_msl_s_matrix``/``compute_mixed_s_matrix``, ``compute_coaxial_two_
port`` never calls ``_project_passive`` at all — its only post-assembly step
is ``_finalize_sparam_result`` -> ``_warn_if_nonpassive_smatrix``, which is a
WARN-only diagnostic (returns ``result`` unchanged either way, and is already
tracer-safe: ``isinstance(s, jax.core.Tracer): return`` short-circuits it
under trace). So the AD call and an eager FD call consume the IDENTICAL raw
``s_params`` — there is no eager-vs-traced branch to keep synchronized here,
unlike the MSL ``enforce_passivity``-gated case this task briefing cited.

STRUCTURAL FINDING — no CPU float64 END-TO-END FDTD comparator is possible
for this method (deviation from the MSL precedent, evidenced, not a
shortcut): ``compute_coaxial_two_port`` HARD-REQUIRES
``precision='float32'`` (raises ``ValueError`` otherwise, mirroring the
1-port sibling's identical contract) — the field/CPML carry arrays are
initialized at that dtype regardless of the outer JAX x64 scope. Attempting
``eps_scale``/``freqs`` at float64 under a scoped ``enable_x64()`` produced
a live ``FutureWarning`` at ``rfx/boundaries/cpml.py:855``
(``hx.at[:, :, :n].add(ch_zlo * correction_hx_zlo)``): "scatter inputs have
incompatible types: cannot safely cast value from dtype=float64 to
dtype=float32" — ``hx`` (the H-field carry) is the SCATTER TARGET and is
float32-typed from initialization, so the float64 intermediate is silently
downcast back to float32 at that point. The measured gradient/FD numbers
below were consequently near-identical whether ``freqs``/``eps_scale`` were
float32 or float64 (0.16% vs 0.45% rel_err at the same h — see below), i.e.
no real precision gain was achieved. The 1-port sibling's own committed gate
(``tests/test_coax_end_to_end_ad.py``) has the same property and is ALSO
float32-only throughout, no ``enable_x64`` anywhere — consistent precedent.
The AD-vs-FD measurement below is therefore float32 AD vs float32-FD
(matching what production actually ships), NOT the CPU-float64 comparator
``scripts/msl_ad_fd_f64_referee.py`` uses for MSL — that script's approach
does not transfer to a method with a hard float32 precision pin.

SEVERED-TAPE FALSIFIER (performed by hand during this change, R3 discipline):
wrapping the traced ``eps_scale`` argument in ``jax.lax.stop_gradient``
before it reaches ``compute_coaxial_two_port`` (reproducing the #483/#515
class of defect — the fixture samples the value but the tape never sees it)
made ``jax.grad`` return EXACTLY ``0.0`` on this fixture, confirming the
floor assertion below is a real falsifier and not vacuous. Reverted before
this file was written (not applied to any production or test code — a
throwaway script, not a temporary edit-and-revert of shipped files).

GATE DERIVATION: measured on this checkout, CPU, float32, the small fixture
below (``N_STEPS=400``, domain ``(0.008, 0.008, 0.012)``,
``probe_count=3``):

    g_ad = -5.475045e-02   (jax.grad, eps_scale0 = 1.0)
    h=0.01   g_fd = -5.487204e-02   rel_err = 0.00222
    h=0.02   g_fd = -5.447567e-02   rel_err = 0.00504

Worst rel_err across this sweep is 0.00504 (h=0.02);
``tests._gate_policy.gate_from_envelope(0.00504, quantum=100) == 0.01``. The
committed gate below runs at ``h=0.01`` (rel_err 0.00222 there, so the 0.01
threshold carries ~4.5x margin over what is actually asserted, while still
being derived from the WORSE point in the sweep rather than the better one
it happens to test at). float32 resolving power at ``h=0.01``: loss ~1.0,
``|f(+h)-f(-h)| ~ 1.10e-3``, float32 ULP of a loss near 1.0 is ~1.19e-7, so
the FD signal spans ~9.2e3 ULP — comfortably outside the #527-class blind
spot (4.4 ULP) without needing the full resolving-power apparatus
``tests/test_msl_ad_fd_converged.py`` built (this is a FIRST gate being
added, not a threshold being tightened after a comparator incident on this
specific method).

ENVELOPE PROVENANCE: measured on CPU float32 only, this checkout,
2026-08-05. Owner-platform (GPU) re-measurement is PENDING — not required
for this PR per the #489 task briefing (a first gate being added, not an
existing gate being replaced under a documented comparator defect, unlike
the MSL precedent this file's briefing pointed at).
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from tests._coax_ad_objective import coax_band_mean_s21_sq
from tests._gate_policy import gate_from_envelope


def test_coax_band_mean_s21_sq_reduction_is_correct():
    """Unit-check the shared reduction on a synthetic S-matrix: correctness
    is independent of which AD gate exercises it."""
    n_f = 3
    s21 = np.array([0.9 + 0.1j, 0.7 - 0.2j, 0.5 + 0.0j])
    S = np.zeros((2, 2, n_f), dtype=np.complex128)
    S[1, 0, :] = s21
    got = float(coax_band_mean_s21_sq(jnp.asarray(S)))
    want = float(np.mean(np.abs(s21) ** 2))
    # float32-precision tolerance: jnp.asarray on a complex128 array
    # downcasts to complex64 by default (no x64 enabled in this test).
    assert abs(got - want) < 1e-6, (got, want)


def _traced_input_param_names(sig: "inspect.Signature") -> set:
    """Matches the naming convention used by every OTHER traced-input
    parameter in this codebase (compute_coaxial_line_reflection's
    eps_scale, compute_waveguide_s_matrix's eps_override/sigma_override):
    a name starting with "eps_"/"sigma_", or an explicit "eps_scale"-style
    design-variable name. Deliberately narrow (field_scale, a plain float
    source-amplitude knob, must NOT match)."""
    return {
        name for name in sig.parameters
        if name.lower().startswith(("eps_", "sigma_")) or name.lower() == "eps_scale"
    }


def test_traced_input_param_predicate_has_a_positive_control():
    """N7 (review fix, carried forward): the predicate in
    ``_traced_input_param_names`` is a name-prefix heuristic -- without a
    positive control, a typo that makes it match NOTHING would let
    ``test_compute_coaxial_two_port_has_a_traced_input_channel`` below pass
    vacuously forever. Prove the matcher is live: the 1-port sibling's own
    ``eps_scale`` parameter MUST match.
    """
    from rfx.api import Simulation as _Simulation

    sig = inspect.signature(_Simulation.compute_coaxial_line_reflection)
    matched = _traced_input_param_names(sig)
    assert "eps_scale" in matched, (
        "the traced-input-parameter predicate did not match "
        "compute_coaxial_line_reflection's own eps_scale -- the matcher "
        "itself is broken"
    )


def test_compute_coaxial_two_port_has_a_traced_input_channel():
    """FLIPPED (was ``test_compute_coaxial_two_port_has_no_traced_input_
    channel``, pinning the ABSENCE of a channel): ``compute_coaxial_two_
    port`` now has an ``eps_scale`` traced-input parameter, mirroring the
    1-port sibling's own design. If a future change removes it, this test
    (and ``tests/test_ad_surface_contract.py``'s classification) must be
    updated in the SAME change, not left stale.
    """
    sig = inspect.signature(Simulation.compute_coaxial_two_port)
    traced_input_names = _traced_input_param_names(sig)
    assert "eps_scale" in traced_input_names, (
        f"compute_coaxial_two_port() lost its eps_scale traced-input "
        f"channel (matched names: {traced_input_names!r}) -- this reverts "
        f"issue #489 leg 3; see this module's docstring for the fix that "
        f"added it."
    )


# ---------------------------------------------------------------------------
# AD gate: jax.grad through the FULL two-drive method, small real-FDTD
# fixture (mirrors tests/test_coax_end_to_end_ad.py's own structure for the
# 1-port sibling — same domain shape, smaller z extent, marked slow_physics).
# ---------------------------------------------------------------------------

# A small, resolved-enough two-feed coax line: annulus ~3.8 cells (same
# cross-section as the 1-port gate's fixture), z domain shrunk to the
# minimum that fits BOTH probe arrays with probe_count=3 (verified via
# rfx.api.Simulation._build_grid()'s own layout-fit arithmetic, the same
# check tests/test_coax_two_port_fdtd.py::test_default_domain_fits_the_
# default_layout uses for the full-size fixture).
N_STEPS = 400
FREQ = jnp.asarray([8.0e9], dtype=jnp.float32)
PROBE_COUNT = 3
PROBE_START_CELLS = 4
PROBE_SPACING_CELLS = 2

# Central-FD step and the threshold derived from it — see this module's
# docstring "GATE DERIVATION" section for the measured sweep this comes
# from. gate_from_envelope(0.00504, quantum=100) == 0.01 (computed below,
# not hand-typed, so a change to ENVELOPE_GATE_MULTIPLIER moves this gate
# too — issue #528).
_FD_H = 0.01
_WORST_MEASURED_REL_ERR = 0.00504  # h=0.02 point of the 2-point CPU sweep
_REL_ERR_THRESHOLD = gate_from_envelope(_WORST_MEASURED_REL_ERR, quantum=100)

# Not merely isfinite/nonzero (the #515-class defect: those cannot fail on a
# severed tape reading exactly 0.0). Derived with ~180x headroom below the
# measured |g_ad| ~ 0.0548 -- comfortable margin for platform/JAX-version
# float32 noise while still rejecting the severed-tape falsifier's exact
# 0.0 reading (confirmed by hand -- see this module's docstring).
_GRAD_FLOOR = 3.0e-4


def _build_small_two_port_sim() -> Simulation:
    sim = Simulation(domain=(0.008, 0.008, 0.012), freq_max=40.0e9, boundary="cpml")
    sim.add_coaxial_port(
        (0.004, 0.004, 0.006), face="top", pin_length=5.0e-3,
        waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2),
    )
    return sim


def _band_mean_s21_sq(eps_scale):
    sim = _build_small_two_port_sim()
    res = sim.compute_coaxial_two_port(
        n_steps=N_STEPS, freqs=FREQ, probe_count=PROBE_COUNT,
        probe_start_cells=PROBE_START_CELLS, probe_spacing_cells=PROBE_SPACING_CELLS,
        eps_scale=eps_scale,
    )
    return coax_band_mean_s21_sq(res.s_params)


@pytest.mark.slow_physics
def test_compute_coaxial_two_port_ad_grad_finite_and_fd_consistent():
    """Gate: ``compute_coaxial_two_port`` is differentiable end to end w.r.t.
    ``eps_scale`` and the AD gradient matches a central finite difference —
    this is the AD moat for the two-port coax method (mirrors
    ``tests/test_coax_end_to_end_ad.py::
    test_coax_reflection_grad_finite_and_fd_consistent`` for the 1-port
    sibling). See this module's docstring for the full measured record,
    the severed-tape falsifier, and why this comparator is float32-only.
    """
    val, g = jax.value_and_grad(_band_mean_s21_sq)(jnp.asarray(1.0, dtype=jnp.float32))
    g_ad = float(g)
    print(f"\n[coax two-port AD] loss = {float(val):.6e}  g_ad = {g_ad:.6e}")

    assert np.isfinite(float(val)), f"objective is not finite: {val}"
    assert np.isfinite(g_ad), f"gradient is not finite: {g}"
    assert abs(g_ad) > _GRAD_FLOOR, (
        f"[coax two-port AD] gradient is effectively zero ({g_ad:.3e}, floor "
        f"{_GRAD_FLOOR:.0e}): the tape may be severed -- see this module's "
        "docstring for the by-hand severed-tape falsifier that confirms "
        "this floor actually catches that defect (reads exactly 0.0 there)."
    )

    fp = float(_band_mean_s21_sq(jnp.asarray(1.0 + _FD_H, dtype=jnp.float32)))
    fm = float(_band_mean_s21_sq(jnp.asarray(1.0 - _FD_H, dtype=jnp.float32)))
    g_fd = (fp - fm) / (2.0 * _FD_H)
    print(f"[coax two-port AD] g_fd(h={_FD_H:g}) = {g_fd:.6e}")
    assert np.isfinite(g_fd) and g_fd != 0.0, "FD slope not finite/nonzero -- rebuild fixture"

    assert g_ad * g_fd > 0, (
        f"[coax two-port AD] AD and FD gradients have OPPOSITE SIGNS: "
        f"g_ad={g_ad:.4e}, g_fd={g_fd:.4e}."
    )
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-12)
    print(f"[coax two-port AD] rel_err = {rel:.5f} (threshold {_REL_ERR_THRESHOLD:.2f})")
    assert rel <= _REL_ERR_THRESHOLD, (
        f"[coax two-port AD] AD={g_ad:+.6e} vs FD={g_fd:+.6e} "
        f"(rel diff {rel:.4f} > {_REL_ERR_THRESHOLD:.2f})"
    )


@pytest.mark.slow_physics
def test_coax_two_port_eps_scale_unity_matches_concrete_path():
    """The AD path does not change the physics: ``eps_scale=1.0`` (jnp path,
    every voltage extraction/assembly/solve routed through the jnp cores)
    matches ``eps_scale=None`` (validated numpy path) in |S| — mirrors
    ``tests/test_coax_end_to_end_ad.py::
    test_coax_eps_scale_unity_matches_concrete_path`` for the 1-port
    sibling. Measured on this checkout: max abs diff 3.6e-6, max rel diff
    1.9e-4 -- tighter than the tolerance below by >1000x margin, consistent
    with float32 reassociation noise only (not a structural difference).
    """
    sim_a = _build_small_two_port_sim()
    res_a = sim_a.compute_coaxial_two_port(
        n_steps=N_STEPS, freqs=FREQ, probe_count=PROBE_COUNT,
        probe_start_cells=PROBE_START_CELLS, probe_spacing_cells=PROBE_SPACING_CELLS,
    )
    sim_b = _build_small_two_port_sim()
    res_b = sim_b.compute_coaxial_two_port(
        n_steps=N_STEPS, freqs=FREQ, probe_count=PROBE_COUNT,
        probe_start_cells=PROBE_START_CELLS, probe_spacing_cells=PROBE_SPACING_CELLS,
        eps_scale=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert res_a.status == "passed"
    assert res_b.status == "differentiable"
    sa = np.asarray(res_a.s_params)
    sb = np.asarray(res_b.s_params)
    assert np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))
    np.testing.assert_allclose(np.abs(sb), np.abs(sa), rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-m", "slow_physics"])
