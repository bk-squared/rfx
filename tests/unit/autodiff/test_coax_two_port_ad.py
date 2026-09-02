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
regression-locked by the existing ``tests/unit/sparams/test_coax_two_port_fdtd.py``
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

CORRECTION (adversarial review, 2026-08-05, BLOCKING): the first version of
this file claimed "no CPU float64 comparator is possible for this method"
because ``compute_coaxial_two_port`` hard-requires ``precision='float32'``.
That claim was WRONG, and was found wrong empirically, not by re-reading the
code harder. ``precision='float32'`` governs only the FDTD field/CPML CARRY
arrays, which do stay float32 -- exactly like MSL's own f64 referee ("the
shipped gate's AD is f32, so f32 is the default",
``scripts/msl_ad_fd_f64_referee.py``). The LOSS dtype instead follows the
DFT plane accumulator, which ``rfx/probes/probes.py``'s
``init_dft_plane_probe`` keys off ``jax.config.x64_enabled`` INDEPENDENTLY
of ``self._precision``::

    _cdtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64

A scoped ``enable_x64()`` around the forward call is sufficient --
``eps_scale``/``freqs`` do not even need to be passed as float64 themselves.
The first attempt got this wrong two ways: it forced ``eps_scale``/``freqs``
to float64 explicitly (unnecessary, and actively counterproductive: it made
``materials.eps_r`` float64 while the field carry stayed float32, producing
a harmless-but-misleading ``FutureWarning`` at ``rfx/boundaries/cpml.py``'s
CPML scatter), and then judged "no real precision gain" by comparing NUMBERS
across two runs instead of checking ``loss.dtype`` directly -- the actual,
decisive check (``scripts/coax_two_port_ad_fd_f64_referee.py`` asserts it
every run). See that script for the full mechanism writeup and a
regenerating measurement; ``docs/guides/sparameter_support_matrix.{md,json}``
are corrected to match.

SEVERED-TAPE FALSIFIER (performed by hand during this change, R3 discipline,
re-confirmed after the N_STEPS settling fix below): wrapping the traced
``eps_scale`` argument in ``jax.lax.stop_gradient`` before it reaches
``compute_coaxial_two_port`` (reproducing the #483/#515 class of defect --
the fixture samples the value but the tape never sees it) made ``jax.grad``
return EXACTLY ``0.0`` on this fixture, confirming the floor assertion below
is a real falsifier and not vacuous. Not applied to any production or test
code -- a throwaway script, not a temporary edit-and-revert of shipped
files.

GATE DERIVATION (measured on this checkout, CPU, this fixture, N_STEPS=600 --
see ``scripts/coax_two_port_ad_fd_f64_referee.py`` to regenerate):

    g_ad (float32, as shipped) = -1.2053344399e-02

    h          g_fd (float64)          rel_err   ULP span
    2.0e-04    -1.2253229603e-02       0.01631    2.21e+10
    5.0e-04    -1.2316619146e-02       0.02138    5.55e+10
    1.0e-03    -1.2181618882e-02       0.01053    1.10e+11
    2.0e-03    -1.1992383331e-02       0.00508    2.16e+11
    5.0e-03    -1.1993261017e-02       0.00501    5.40e+11
    1.0e-02    -1.1596279883e-02       0.03941    1.04e+12
    2.0e-02    -1.0302271013e-02       0.16997    1.86e+12

R5 (inspect the full trace, not a single point): this is NOT flat -- it has
real structure, not scatter. ``h=2e-3`` and ``h=5e-3`` agree with each other
to 0.007% (-1.19924e-2 vs -1.19933e-2), the classic signature of a converged
central-difference estimate with both truncation and floor noise negligible
-- that pair anchors the "true" local slope at ~-1.1993e-2, giving rel_err
~0.5% against AD. Below that window (``h<=1e-3``) the estimate moves the
OTHER direction and NON-monotonically as ``h`` shrinks further (2e-4 reads
CLOSER to the converged value than 5e-4 does) -- the signature of a floor
noise source, not systematic truncation; consistent with the field carry
staying float32 throughout (per the CORRECTION above, the loss dtype does
not eliminate float32 rounding accumulated over 600 real Yee steps, it only
stops the LOSS's OWN arithmetic from adding quantization on top). Above the
window (``h>=1e-2``) the estimate drifts monotonically AWAY from the
converged value as ``h`` grows -- consistent with ordinary O(h^2) central-FD
truncation from real curvature (a multi-percent dielectric perturbation is
not small on this annulus), not a comparator defect. ULP span is >=2e10 at
every point in the entire swept range (loss dtype is float64 throughout,
verified every run) -- the degradation at both ends is NOT a resolving-power
problem in the ``_MIN_FD_ULP_SPAN`` sense below.

The committed gate therefore runs at ``h=2e-3`` (inside the converged
window) and derives its threshold from the WORST rel_err over the three
POINTS INSIDE that window (``h`` in ``{1e-3, 2e-3, 5e-3}``), not the full
7-point diagnostic sweep printed above -- excluding ``h in {2e-4, 5e-4}``
(floor-noise-dominated, not a property of AD accuracy) and ``h in {1e-2,
2e-2}`` (truncation-dominated, expected for a finite step of that size, also
not a property of AD accuracy at ``eps_scale=1.0``) is a step-size-selection
judgment, made and disclosed BEFORE reading it as license to loosen anything
-- it makes the gate TIGHTER (2%) than deriving from the full 7-point sweep
would (``gate_from_envelope(0.16997, quantum=100) == 0.26``, i.e. 26%, not
merely the raw 17% worst-rel_err before the 1.5x multiplier), not looser.
Worst of the 3-point window is 0.01053 (h=1e-3);
``tests._gate_policy.gate_from_envelope(0.01053, quantum=100) == 0.02``. At
the gate's own ``h=2e-3`` the measured rel_err is 0.00508, so the 2%
threshold carries ~3.9x margin over what is actually asserted.

RESOLVING-POWER FLOOR (#529 discipline, ``_MIN_FD_ULP_SPAN`` below): every
measured ULP span in the table above is >= 2e10, ~2e6x this file's own
floor (1e4, matching ``tests/unit/autodiff/test_msl_ad_fd_converged.py``'s own value) --
``_MIN_FD_ULP_SPAN * _REL_ERR_THRESHOLD = 1e4 * 0.02 = 200 >= 100``, this
repo's own bidirectional anchor (a floor admitting less than 100x combined
margin is not worth having --
``tests/unit/autodiff/test_msl_ad_fd_converged.py::test_comparator_floor_rejects_the_f32_
reference_that_caused_527``). A prior VERSION of this gate that compared f32
AD against an f32-computed FD reference (before this correction) measured
``span x threshold = 92.06`` on the (then-unsettled, N_STEPS=400) fixture --
under this repo's own 100 floor; the float64 reference here does not have
that problem (min span 2.21e10 even in the excluded small-h region).

ENVELOPE PROVENANCE: measured on CPU, this checkout, 2026-08-05 (git SHA in
the PR that landed this correction). Owner-platform (GPU) re-measurement is
PENDING -- not required for this PR per the #489 task briefing (a first gate
being added, not an existing gate being replaced under a documented
comparator defect).

SETTLING FIX (adversarial review F6): the fixture below now runs
``N_STEPS=600`` (was 400) -- the concrete (``eps_scale=None``) twin's own
``settling_db`` at 400 steps read ``[-37.13, -37.70]`` dB, short of this
repo's -40 dB ring-down rule; 600 steps clears it with margin
(``[-61.36, -63.88]`` dB, reprinted by
``scripts/coax_two_port_ad_fd_f64_referee.py``'s own forward sanity check
every run). All numbers in this docstring are measured on the SETTLED
(600-step) fixture, per this issue's own instruction to fix settling before
re-measuring the gate.
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
    (and ``tests/unit/autodiff/test_ad_surface_contract.py``'s classification) must be
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
# fixture (mirrors tests/unit/autodiff/test_coax_end_to_end_ad.py's own structure for the
# 1-port sibling — same domain shape, smaller z extent, marked slow_physics).
# ---------------------------------------------------------------------------

# A small, resolved-enough two-feed coax line: annulus ~3.8 cells (same
# cross-section as the 1-port gate's fixture), z domain shrunk to the
# minimum that fits BOTH probe arrays with probe_count=3 (verified via
# rfx.api.Simulation._build_grid()'s own layout-fit arithmetic, the same
# check tests/unit/sparams/test_coax_two_port_fdtd.py::test_default_domain_fits_the_
# default_layout uses for the full-size fixture).
#
# N_STEPS=600 (not the originally-committed 400 -- adversarial review of
# this PR measured the concrete twin's own settling_db at N_STEPS=400 as
# [-37.13, -37.70] dB, short of this repo's -40 dB ring-down rule; 600
# clears it with margin: [-61.36, -63.88] dB, verified below and by
# scripts/coax_two_port_ad_fd_f64_referee.py's own forward sanity check).
N_STEPS = 600
FREQ = jnp.asarray([8.0e9], dtype=jnp.float32)
PROBE_COUNT = 3
PROBE_START_CELLS = 4
PROBE_SPACING_CELLS = 2

# Central-FD step (inside the converged window -- see this module's
# docstring "GATE DERIVATION" section) and the threshold derived from the
# WORST rel_err over the 3-point window {1e-3, 2e-3, 5e-3}, not the full
# 7-point diagnostic sweep printed there. gate_from_envelope(0.01053,
# quantum=100) == 0.02 (computed below, not hand-typed, so a change to
# ENVELOPE_GATE_MULTIPLIER moves this gate too — issue #528).
_FD_H = 2.0e-3
_WORST_MEASURED_REL_ERR = 0.01053  # h=1e-3, worst of the 3-point window
_REL_ERR_THRESHOLD = gate_from_envelope(_WORST_MEASURED_REL_ERR, quantum=100)

# Resolving-power floor (#529/#527 discipline): a central difference whose
# f(+h)-f(-h) spans too few ULPs of the loss dtype is quantized to noise no
# matter how exact the solver is -- see tests/unit/autodiff/test_msl_ad_fd_converged.py's
# own _MIN_FD_ULP_SPAN for the incident this class of check exists to catch.
# 1e4 matches that file's value (this gate's own measured spans are all
# >=2e10, ~2e6x headroom -- no tighter floor is needed here, and reusing the
# established number avoids inventing a second one with no basis). The
# bidirectional anchor _MIN_FD_ULP_SPAN * _REL_ERR_THRESHOLD >= 100 (that
# file's own discipline) holds here: 1e4 * 0.02 = 200.
_MIN_FD_ULP_SPAN = 1.0e4

# COMMITTED as a module-level assert, not left as a comment (adversarial
# review, 2026-08-05): tests/unit/autodiff/test_msl_ad_fd_converged.py enforces this same
# invariant in an executable check
# (test_comparator_floor_rejects_the_f32_reference_that_caused_527); a
# comment alone would let a future editor tighten _REL_ERR_THRESHOLD below
# 100/_MIN_FD_ULP_SPAN without ever being told to also raise the
# resolving-power floor. Runs at collection time (no simulation needed),
# so it fires even if nobody runs the slow_physics gate test itself.
assert _MIN_FD_ULP_SPAN * _REL_ERR_THRESHOLD >= 100.0, (
    f"_MIN_FD_ULP_SPAN ({_MIN_FD_ULP_SPAN:.0e}) combined with "
    f"_REL_ERR_THRESHOLD ({_REL_ERR_THRESHOLD:.2f}) gives only "
    f"{_MIN_FD_ULP_SPAN * _REL_ERR_THRESHOLD:.0f}x combined margin, under "
    "this repo's own 100x floor (tests/unit/autodiff/test_msl_ad_fd_converged.py). "
    "Raise _MIN_FD_ULP_SPAN before tightening _REL_ERR_THRESHOLD further."
)

# Not merely isfinite/nonzero (the #515-class defect: those cannot fail on a
# severed tape reading exactly 0.0). Derived with ~40x headroom below the
# measured |g_ad| ~ 0.01205 -- comfortable margin for platform/JAX-version
# float32 noise while still rejecting the severed-tape falsifier's exact
# 0.0 reading (confirmed by hand on this exact N_STEPS=600 fixture -- see
# this module's docstring).
_GRAD_FLOOR = 3.0e-4


def _fd_ulp_span(f_plus: float, f_minus: float, dtype) -> float:
    """Resolving power of a central difference, in ULPs of ``dtype``.

    Mirrors ``tests/unit/autodiff/test_msl_ad_fd_converged.py::_fd_ulp_span`` exactly (not
    imported from there -- that helper is local to that file, no shared
    module exists for it yet). ``dtype`` must be the dtype the LOSS was
    computed in, not the Python float container both values arrived in as:
    ``float(jnp_scalar)`` always widens to a Python float, so keying off the
    value alone would silently measure float64 resolution even for a
    float32 loss -- the exact defect that let the #527 comparator pass the
    check it should have failed.
    """
    ulp = float(np.spacing(np.asarray(abs(0.5 * (f_plus + f_minus)), dtype=dtype)))
    return abs(f_plus - f_minus) / ulp


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
    ``eps_scale`` and the float32 AD gradient (as shipped) matches a central
    finite difference computed with a float64 LOSS (scoped ``enable_x64()``
    -- on this gate's production config the FDTD fields stay float32
    STORAGE throughout; only the DFT-accumulator-and-downstream math gains
    precision, exactly like MSL's own referee). Storage vs arithmetic
    (issue #630): before #630 the Yee update's compute dtype was pinned to
    float32 for any field storage dtype, so this gate's floor was never a
    field-storage question in the first place; #630 makes compute dtype
    follow storage (a no-op here, since storage IS float32 by default).
    This is the AD moat for the two-port coax method (mirrors
    ``tests/unit/autodiff/test_coax_end_to_end_ad.py::
    test_coax_reflection_grad_finite_and_fd_consistent`` for the 1-port
    sibling, upgraded with the float64-loss + resolving-power discipline
    ``tests/unit/autodiff/test_msl_ad_fd_converged.py`` established after issue #527). See
    this module's docstring for the full measured 7-point sweep, the
    severed-tape falsifier, and the step-size-selection rationale behind
    ``_FD_H``/the envelope window. Regenerate/extend the sweep with
    ``scripts/coax_two_port_ad_fd_f64_referee.py``.
    """
    try:
        from jax import enable_x64
    except ImportError:  # older JAX (< ~0.4.31)
        from tests._x64_compat import enable_x64

    val, g = jax.value_and_grad(_band_mean_s21_sq)(jnp.asarray(1.0, dtype=jnp.float32))
    g_ad = float(g)
    print(f"\n[coax two-port AD] loss (f32) = {float(val):.6e}  g_ad = {g_ad:.6e}")

    assert np.isfinite(float(val)), f"objective is not finite: {val}"
    assert np.isfinite(g_ad), f"gradient is not finite: {g}"
    assert abs(g_ad) > _GRAD_FLOOR, (
        f"[coax two-port AD] gradient is effectively zero ({g_ad:.3e}, floor "
        f"{_GRAD_FLOOR:.0e}): the tape may be severed -- see this module's "
        "docstring for the by-hand severed-tape falsifier that confirms "
        "this floor actually catches that defect (reads exactly 0.0 there)."
    )

    # eps_scale is deliberately left float32 (its natural dtype) -- only the
    # global x64 flag is scoped on. See this module's CORRECTION docstring
    # section for why that alone is sufficient (the DFT accumulator dtype
    # follows jax.config.x64_enabled, independent of the FDTD field
    # precision) and why forcing eps_scale to float64 is unnecessary and
    # counterproductive.
    with enable_x64():
        fp = _band_mean_s21_sq(jnp.asarray(1.0 + _FD_H, dtype=jnp.float32))
        fm = _band_mean_s21_sq(jnp.asarray(1.0 - _FD_H, dtype=jnp.float32))
        assert fp.dtype == jnp.float64 and fm.dtype == jnp.float64, (
            f"[coax two-port AD] FD reference did NOT run in float64 (got "
            f"{fp.dtype}). JAX truncates a float64 request to float32 when "
            "x64 is off -- check jax.experimental.enable_x64 and the "
            "jax.config.x64_enabled keying in rfx/probes/probes.py. Do NOT "
            "read rel_err below as physics if this fires."
        )
        fp, fm = float(fp), float(fm)
    g_fd = (fp - fm) / (2.0 * _FD_H)
    print(f"[coax two-port AD] g_fd(h={_FD_H:g}, float64) = {g_fd:.6e}")
    assert np.isfinite(g_fd) and g_fd != 0.0, "FD slope not finite/nonzero -- rebuild fixture"

    # Resolving-power floor MUST precede the accuracy gate (#527 discipline
    # -- see this module's docstring "RESOLVING-POWER FLOOR" section): a
    # reference that cannot resolve the quantity it judges turns every
    # verdict into noise.
    fd_ulp_span = _fd_ulp_span(fp, fm, np.float64)
    print(f"[coax two-port AD] FD resolving power: {fd_ulp_span:.3g} ULP "
          f"of the loss (floor {_MIN_FD_ULP_SPAN:.0e})")
    assert fd_ulp_span >= _MIN_FD_ULP_SPAN, (
        f"[coax two-port AD] the FD REFERENCE cannot resolve this gradient: "
        f"f(+h)-f(-h) spans only {fd_ulp_span:.3g} ULP of the loss (need "
        f">= {_MIN_FD_ULP_SPAN:.0e}). This is a COMPARATOR failure, not an "
        "AD failure -- do not touch the extractor or the tape on the "
        "strength of it."
    )

    assert g_ad * g_fd > 0, (
        f"[coax two-port AD] AD and FD gradients have OPPOSITE SIGNS: "
        f"g_ad={g_ad:.4e}, g_fd={g_fd:.4e}."
    )
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-12)
    print(f"[coax two-port AD] rel_err = {rel:.5f} (threshold {_REL_ERR_THRESHOLD:.2f})")
    assert rel <= _REL_ERR_THRESHOLD, (
        f"[coax two-port AD] AD={g_ad:+.6e} vs FD={g_fd:+.6e} "
        f"(rel diff {rel:.4f} > {_REL_ERR_THRESHOLD:.2f}) -- a genuine "
        "gradient accuracy finding, do not loosen the gate; investigate the "
        "AD path instead."
    )


@pytest.mark.slow_physics
def test_coax_two_port_eps_scale_unity_matches_concrete_path():
    """The AD path does not change the physics: ``eps_scale=1.0`` (jnp path,
    every voltage extraction/assembly/solve routed through the jnp cores)
    matches ``eps_scale=None`` (validated numpy path) in |S| and phase —
    mirrors ``tests/unit/autodiff/test_coax_end_to_end_ad.py::
    test_coax_eps_scale_unity_matches_concrete_path`` for the 1-port
    sibling.

    MEASURED MARGINS (on this checkout, this fixture, N_STEPS=600 --
    corrected 2026-08-05: an earlier version of this docstring claimed
    ">1000x margin", which adversarial review measured wrong by 25-130x; the
    true margin is honestly reported here, not re-asserted from memory):
    max abs diff ``4.46e-6``, max rel diff ``1.824e-4``. Against the
    ``rtol=1e-3`` bound alone that is a ``5.48x`` margin; against the full
    ``rtol*|a|+atol`` combined bound used by ``assert_allclose`` below, the
    worst-case (smallest) margin across every element is ``70.75x``. Both
    are consistent with float32 reassociation noise (the jnp and numpy
    voltage-extraction/assembly paths do the same arithmetic in a different
    op order), not a structural difference -- but the margin is single-to-
    low-double-digit, not the ">1000x" the previous claim implied, so this
    tolerance is not slack to spend elsewhere. Max phase diff (via
    ``angle(sb) - angle(sa)``, wrapped) is ``2.675e-4`` rad, asserted below
    against a ``1e-3`` rad bound (~3.7x margin).
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

    max_phase_diff_rad = float(np.max(np.abs(np.angle(np.exp(1j * (np.angle(sb) - np.angle(sa)))))))
    print(f"\n[coax two-port unity] max phase diff = {max_phase_diff_rad:.4e} rad")
    assert max_phase_diff_rad < 1.0e-3, (
        f"[coax two-port unity] eps_scale=1.0 phase diverges from the "
        f"concrete path by {max_phase_diff_rad:.4e} rad (bound 1e-3) -- "
        "measured 2.675e-4 rad on this checkout; a jump here would signal "
        "a structural (not reassociation-noise) difference between the "
        "jnp and numpy paths."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-m", "slow_physics"])
