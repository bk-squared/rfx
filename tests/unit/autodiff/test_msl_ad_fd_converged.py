"""MSL-FD-TIGHT: Converged tight AD-vs-FD cross-check for end-to-end gradient.

MSL-FD-TIGHT (2026-05-25) adds a slow-marked test that runs
compute_msl_s_matrix(eps_override=...) at num_periods=20 (converged DFT)
and asserts jax.grad agrees with a central finite-difference to a tight
tolerance. The reference itself runs in float64 and must clear a
resolving-power floor before its verdict is read — see issue #527.

This converts the "AD tape flows + roughly right" evidence from
test_sparam_ad_end_to_end.py (num_periods=3, rel_err=16%) into
"AD gradient is accurate" evidence.

OBJECTIVE (issue #530, 2026-08-04): the differentiated quantity is band-mean
``|S21|**2`` (``tests._msl_ad_objective.msl_band_mean_s21_sq``), shared with
the #515 AD smoke (``tests/unit/autodiff/test_msl_sparam_ad.py``) so the two tests
differentiate the identical reduction. This REPLACES the prior
``sum_ij|S_ij|**2`` objective — see the docstring of
``test_msl_ad_fd_converged_tight`` below for the full replacement
rationale and the historical numbers it supersedes.

Geometry mirrors _build_msl_sim() in test_sparam_ad_end_to_end.py exactly.
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests._x64_compat import enable_x64  # SCOPED x64 — never flip it at module level

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from tests._gate_policy import gate_from_envelope
from tests._msl_ad_objective import msl_band_mean_s21_sq

# ---------------------------------------------------------------------------
# Geometry — identical to _build_msl_sim() in test_sparam_ad_end_to_end.py
# ---------------------------------------------------------------------------

_MSL_EPS_R = 3.66
_MSL_H_SUB = 254e-6
_MSL_W_TRACE = 600e-6
_MSL_DX = 80e-6
_MSL_L_LINE = 6e-3
_MSL_PORT_MARGIN = 2e-3
_MSL_F_MAX = 5e9


def _build_msl_sim() -> Simulation:
    """Tiny MSL thru-line sim (2 ports, minimal domain)."""
    lx = _MSL_L_LINE + 2 * _MSL_PORT_MARGIN
    ly = _MSL_W_TRACE + 2 * (2 * _MSL_H_SUB + 8 * _MSL_DX)
    lz = _MSL_H_SUB + 0.5e-3

    sim = Simulation(
        freq_max=_MSL_F_MAX,
        domain=(lx, ly, lz),
        dx=_MSL_DX,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml",
            y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )

    sim.add_material("ro4350b", eps_r=_MSL_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _MSL_H_SUB)), material="ro4350b")

    y_centre = ly / 2.0
    trace_y_lo = y_centre - _MSL_W_TRACE / 2.0
    trace_y_hi = y_centre + _MSL_W_TRACE / 2.0
    sim.add(
        Box((0.0, trace_y_lo, _MSL_H_SUB), (lx, trace_y_hi, _MSL_H_SUB + _MSL_DX)),
        material="pec",
    )

    sim.add_msl_port(
        position=(_MSL_PORT_MARGIN, y_centre, 0.0),
        width=_MSL_W_TRACE,
        height=_MSL_H_SUB,
        direction="+x",
        impedance=50.0,
    )
    sim.add_msl_port(
        position=(_MSL_PORT_MARGIN + _MSL_L_LINE, y_centre, 0.0),
        width=_MSL_W_TRACE,
        height=_MSL_H_SUB,
        direction="-x",
        impedance=50.0,
    )
    return sim


# ---------------------------------------------------------------------------
# Converged tight AD-vs-FD test
# ---------------------------------------------------------------------------

# Number of periods for converged DFT — must be >= 20 per MSL-FD-TIGHT spec.
_NUM_PERIODS = 20
_N_FREQS = 8
_FD_H = 1e-3
# Derived via tests._gate_policy.gate_from_envelope (issue #530; no more
# hand-picked literals for this gate — see the "GATE REBUILT" docstring
# section below for the measured envelope and the full derivation):
#   gate_from_envelope(0.0146, quantum=100) == 0.03
# 0.0146 is the WORST (largest) rel_err observed across the owner-platform
# h-sweep (VESSL 369367251813, gpu-rtx4090) — not just the single value at
# this test's own h=1e-3 (0.0026) — a deliberately conservative choice.
# Nothing about the derivation formula requires the sweep-max; the reason to
# use it anyway is that h-sensitivity in a central-difference FD reference IS
# comparator uncertainty (which side of the truncation/evaluation-noise
# trade-off a given h lands on), not run-to-run measurement noise the 1.5x
# multiplier alone would cover — so folding the worst observed h into the
# envelope is folding in a real, disclosed source of uncertainty rather than
# a decorative tie-breaker. This gives 0.03/0.0026 = 11.5x margin over the
# rel_err actually read at the gate's own h=1e-3.
_REL_ERR_THRESHOLD = 0.03

# Minimum resolving power the FD reference must have before its disagreement
# with AD means anything: |f(+h) - f(-h)| expressed in ULPs of the loss.
# The loss is a float, so it can only move in whole ULPs — a difference N ULPs
# wide is quantised to ~1/N relative resolution no matter how exact the solver
# is. The shipped float32 comparator ran at N = 4.4 under the OLD objective
# (issue #527), i.e. ~23% resolution against the THEN-current 10% gate; under
# the CURRENT #530 objective the shipped f32 comparator would run at N = 53.8
# (measured this fixture, gate's h). 1e4 ULP is 0.01% resolution, 300x inside
# the CURRENT 0.03 gate's bound (recomputed for the #530 threshold; the
# float64 comparator measures 2.9e10 on this fixture, 2.4e9 was the OLD
# objective's figure).
_MIN_FD_ULP_SPAN = 1.0e4


def _fd_ulp_span(f_plus: float, f_minus: float, dtype) -> float:
    """Resolving power of a central difference, in ULPs of ``dtype``.

    ``dtype`` is the dtype the LOSS was computed in, not the container the
    values arrived in. ``float(jnp_scalar)`` always yields a Python float, so
    keying off the value alone silently measures float64 even for a float32
    loss — which is how the first version of this check passed on the exact
    configuration it exists to reject (PR #529 review).

    The gate and its falsifier both call this, so the test exercises the
    expression the gate actually evaluates rather than a re-derivation of it.
    """
    ulp = float(np.spacing(np.asarray(abs(0.5 * (f_plus + f_minus)), dtype=dtype)))
    return abs(f_plus - f_minus) / ulp


def _closest_divisor(n: int, target: int) -> int:
    """Divisor of ``n`` nearest ``target`` (for checkpoint_segments, which must
    divide n_steps exactly — see forward(checkpoint_segments=) issue #73)."""
    best = 1
    for d in range(1, int(n ** 0.5) + 1):
        if n % d == 0:
            for cand in (d, n // d):
                if abs(cand - target) < abs(best - target):
                    best = cand
    return best


# G-AD-CHECKPOINT (2026-05-26): un-skipped. compute_msl_s_matrix now forwards
# checkpoint_every into forward(), so the reverse-mode AD tape is segmented
# (scan-of-scan remat) instead of storing the entire num_periods=20 trajectory.
# Memory scales ~sqrt(n_steps); the OOM (EXIT 137) that forced the prior skip is
# removed. Marked gpu+slow: still a heavy converged run, owned by the VESSL
# physics harness, excluded from the default CPU suite.
@pytest.mark.gpu
@pytest.mark.slow
def test_msl_ad_fd_converged_tight():
    """MSL-FD-TIGHT: converged (num_periods=20) AD gradient matches FD within
    a measured envelope (currently 3%; see GATE REBUILT below for the
    derivation — the value is _REL_ERR_THRESHOLD, do not hardcode it here).

    G-AD-CHECKPOINT (un-skipped 2026-05-26): the num_periods=20 reverse-AD tape
    is now segmented via checkpoint_segments → forward(), so it runs within
    memory budget instead of being OOM-killed. Marked gpu+slow (VESSL-owned).

    R5 instrumentation: prints g_ad, g_fd, rel_err, and forward |S| range.
    If forward |S| is outside [0, 1.2], the test fails explicitly rather than
    silently reporting a gradient on an exploded impedance.

    If rel_err stays above _REL_ERR_THRESHOLD this test will fail
    (deliberately — do NOT loosen the gate to force a pass; report as a
    gradient accuracy finding instead).

    GATE REBUILT (issue #530, 2026-08-04) — OBJECTIVE REPLACED.
    Everything below this point describes the CURRENT gate. The
    "issue #477 / #483 / #527" sections further down are HISTORY: they
    describe how the *previous* objective (``sum_ij|S_ij|**2``) and its
    comparator were debugged, and their numbers are SUPERSEDED — kept only
    so the record of why the objective changed is not lost.

    Why the old objective had to go (full derivation: issue #530, PI decision
    linked from #530's final comment): for a passive network S^dag S <= I,
    so each column of S has norm <= 1 and sum_ij|S_ij|^2 <= 2 per frequency —
    16 over the old gate's 8 bins. Its measured loss was 16.00599: 99.96% a
    passivity-pinned STRUCTURAL CONSTANT, with the differentiated signal
    riding on the remaining 0.037%. That is why the gate went blind when
    PR #516 moved |S| closer to unitary (g_ad collapsed 50x, −2.1143e-01 to
    −4.2425e-03, issue #527). An extractor fix that shrinks |S11| would
    shrink THIS objective's gradient too — band-mean ``|S21|**2`` is not
    immune to that shape of change, only better positioned against it: what
    is MEASURED (not narrated) is that the level dropped 16x on this
    fixture (16.00599 -> 0.99787211), cutting the loss's float32 ULP 32x
    (1.9073e-06 -> 5.9605e-08) and lifting f32 resolving power from 4.45 to
    53.8 ULP at the gate's h — and the residue from unity (~2.5e-3, order
    |S11|**2 with |S11| ~ 0.05 here) is now a physical observable
    (reflected power), not a unitarity-violation artifact. If a future
    extractor fix shrinks this signal again, that risk is CONTAINED by the
    f64 comparator's 2.9e6x resolving-power headroom above
    ``_MIN_FD_ULP_SPAN`` (measured this run) and by the resolving-power
    floor assert below (issue #527's fix) reporting a comparator failure
    loudly instead of a gradient defect silently — not eliminated by a
    claim that this objective cannot go blind.

    WHAT DRIVES THE GRADIENT — RESOLVED, issue #560, 2026-08-06 (see
    ``tests/_msl_ad_objective.py`` for the full statement and self-
    correction — an earlier draft of both this docstring and that module
    claimed the objective "moves directly with... guided wavelength via
    beta"; that mechanism was UNMEASURED, and a sign witness in this
    fixture — the wave split's FROZEN Hammerstad-Jensen Z0 reference,
    against which g_ad > 0 matches reflection REDUCTION toward that fixed
    reference as alpha grows, whereas a beta/standing-wave channel would
    have no particular sign preference — pointed at a reference-plane
    mismatch mechanism instead. The decisive probe (anchor the wave split
    on a FROZEN per-frequency-band FITTED ``z0`` — measured at alpha=1,
    held constant, same discipline as the analytic anchor — instead of the
    frozen analytic ``z0_hj``, then re-measure ``|g_ad|``) has now been
    run: ``scripts/diagnostics/msl_ad_z0_anchor_probe.py``, full log
    ``scripts/diagnostics/msl_ad_z0_anchor_probe_run_20260806.md``.
    Measured on this exact fixture (CPU, float32, same dtype/discipline as
    the gate): ``|g_ad|`` collapsed from ``1.602236e-03`` (production,
    frozen analytic anchor — bit-identical across 2 repeats in the same
    process) to ``6.885110e-05`` (frozen fitted anchor — this HEADLINE
    value is from an un-repeated run, killed by a background-task duration
    limit before its own repeat printed; the value that IS 2/2
    bit-identical-confirmed is a CLI-rounded-anchor rerun, ``6.884444e-05``,
    agreeing with the headline to 4 significant figures). PRIMARY
    criterion (issue #560's own qualitative wording, "drops toward the
    FD-unresolvable floor" applied literally): the estimated FD signal for
    g_b at the gate's h is only ~1.16 ULP of a float32 loss — below the
    4.449 ULP issue #527 measured for the RETIRED objective's f32
    comparator and declared untrustworthy, i.e. g_b is noise-floor by this
    repo's own established standard. SECONDARY check (this PR's own
    pre-declared threshold, NOT a quote from #560 — an earlier draft of
    this docstring wrongly attributed "5x" to the issue body, which
    contains no such number): ~23.3x, 4.6x past that self-declared 5x
    bar. VERDICT: the reference-plane mismatch (mechanism 2) is the
    DOMINANT channel behind d(loss)/d(alpha) on this fixture — the
    beta/reflection-physics reading is NOT supported as the dominant
    explanation. This does not touch the accuracy gate below: AD and FD
    still differentiate the identical function, so rel_err/PASS is
    unaffected either way — only the physical story attached to the
    gradient's magnitude changes. Separately, anchor B's own loss exceeded
    1 (a passivity violation, expected/attributed to the raw unprojected
    eps_override channel — see the probe's run log) — evidence the fitted
    anchor is not self-evidently "more correct," so whether
    ``compute_msl_s_matrix``'s PRODUCTION wave split should anchor on it is
    a SEPARATE, undecided design question this PR does not settle. Band-mean
    ``|S21|**2`` (``tests._msl_ad_objective.
    msl_band_mean_s21_sq``, shared with the #515 AD smoke) is computed from
    the exact same ``compute_msl_s_matrix`` call the old objective used;
    only the post-call reduction changed.

    MEASURED (owner platform: gpu-rtx4090, VESSL 369367251813, branch
    ``msl-ad-band-mean-s21-objective`` @ ``0acbfb54f16958813bcbd2e413b992cff036cb98``,
    harness ``scripts/diagnostics/msl_ad_band_mean_s21_owner_measurement.py``
    — built standalone because this measurement predates the objective's
    merge to main, so it clones the branch instead of using the mounted
    primary checkout, which stays on ``main``). The raw logs live only under
    the primary checkout's gitignored ``.omx/`` (per-repo runtime scratch);
    the TRACKED copy, including the actual pytest gate's own PASS output
    (VESSL 369367251827, "3 passed in 138.72s"), is
    ``scripts/diagnostics/msl_ad_band_mean_owner_measurement/owner_runs_20260804.md``:

        loss = 0.99787211            g_ad (f32, as shipped) = 1.602933e-03

        h          g_fd                rel_err   FD resolving power (ULP)
        3.0e-04    1.6267607988e-03    0.0146    8.79e+09
        1.0e-03    1.6071476933e-03    0.0026    2.90e+10   <- gate's h
        2.0e-03    1.6034690085e-03    0.0003    5.78e+10
        5.0e-03    1.6012954754e-03    0.0010    1.44e+11
        1.0e-02    1.6026462683e-03    0.0002    2.89e+11

    f64 FD spread over this h-sweep: 1.583% (tight — the new objective's FD
    reference is well-behaved, unlike the old objective's 2017%/3.4%/0.97%
    scatter history under #527). Resolving power at every h clears
    ``_MIN_FD_ULP_SPAN`` (1e4) by 6+ orders of magnitude, so
    ``_MIN_FD_ULP_SPAN`` itself is UNCHANGED — the new objective did not need
    a bigger resolving-power floor, it needed to stop riding on a constant.

    THRESHOLD DERIVATION: ``gate_from_envelope(0.0146, quantum=100) == 0.03``
    (``tests._gate_policy``). The envelope fed in is 0.0146 — the WORST rel_err
    observed across the h-sweep above, not just the 0.0026 the gate structurally
    reads at its own h=1e-3 — a deliberately conservative choice (R5: inspect
    the full trace, not one point). The reason to prefer the sweep-max over
    the single gate-h value: h-sensitivity in a central-difference reference
    IS comparator uncertainty (which side of the truncation/evaluation-noise
    trade-off a given h happens to land on), a real source of doubt about the
    reference distinct from the 1.5x multiplier's run-to-run margin, so
    folding it into the envelope is honest rather than decorative. This puts
    ``_MIN_FD_ULP_SPAN * _REL_ERR_THRESHOLD = 300`` (comfortably above the
    check's own 100 minimum — see
    ``test_comparator_floor_rejects_the_f32_reference_that_caused_527``,
    which ALSO documents that this self-check now bounds any FUTURE
    tightening of this gate from below: it requires
    ``_REL_ERR_THRESHOLD >= 100 / _MIN_FD_ULP_SPAN = 0.01`` for a fixed
    ``_MIN_FD_ULP_SPAN``, so this gate cannot be tightened below 0.01
    without first raising the resolving-power floor). At the gate's own
    h=1e-3, the resulting margin is explicit: 0.03 / 0.0026 = 11.5x.

    RESOLVING-POWER FALSIFIER (#529 discipline — the gate must be shown to
    actually catch a defect, not just pass on the happy path). The same
    measurement run also built the AD objective with ``eps_override`` FROZEN
    at a concrete alpha computed BEFORE tracing (an issue-#483-CLASS defect:
    "the fixture sampled the override under FD while the tape saw it
    frozen" — here reproduced deliberately by never letting the traced
    ``alpha`` argument enter the computation at all). ``jax.grad`` of a
    function that ignores its argument returns exactly 0.0, so at every h in
    the sweep: ``g_ad(defect) = 0.000000e+00`` and ``rel_err = 1.0000`` — the
    gate reds at 33x the 0.03 threshold. Separately (this repo, CPU, fast):
    the #515 smoke's OLD uniform-Hy/Hz-field construction, under this SAME
    new objective, also reads grad = 0.0 exactly — see
    ``tests/unit/autodiff/test_msl_sparam_ad.py::test_compute_msl_s_matrix_ad_smoke_has_finite_gradient``'s
    docstring for that falsifier record. Both confirm the gate has resolving
    power against real defects, not just margin against comparator noise.

    COST: AD (f32, correct) 71.6s, AD (f32, planted defect) 19.1s, FD (f64)
    ~41-47s per h-row (two forwards each) — all measured on VESSL
    369367251813; total script wall-time ~308s (job start 14:11:27 to script
    completion 14:16:35 UTC on 2026-08-04).

    --- HISTORY BELOW (superseded numbers; issues #477, #483, #527) ---

    OWNERSHIP + measured comparator envelope (issue #477 root-cause,
    2026-07-28 — all numbers measured, main @ 98c5e33, OLD objective):
    This is a GPU-lane gate (gpu marker; VESSL harness). Do NOT treat a CPU
    execution as a main-health signal: the FD comparator carries an f32
    evaluation-noise envelope that straddles the old 0.10 gate across
    platforms while the AD side is platform-stable to 5 digits
    (g_ad CPU −2.1104e-01 vs GPU −2.1105e-01; g_fd CPU −2.3746e-01 vs GPU
    −2.3079e-01 at the same h=1e-3 → rel_err CPU 0.1113 / GPU 0.0855).
    A CPU h-sweep (h ∈ [3e-4, 1e-2]) shows a 27% NON-monotone FD scatter —
    evaluation noise, not h² truncation — so rel_err here compares AD
    against a ±3–5%-noisy reference. Whether AD carries a genuine ~10%
    systematic vs the true derivative was initially INDETERMINATE at f32;
    the f64 referee (unblocked by the accumulator-dtype fix) then measured
    a genuine ~13.7% AD systematic — ATTRIBUTED AND FIXED (issue #483):
    the auto-eps_r_sub launch fixture sampled the override under FD while
    the tape saw it frozen. Post-fix the same converged f64 referee reads
    rel_err = 0.00011 (f64 AD −2.11425e-01 vs FD −2.11449e-01) — the MSL
    eps_override gradient now matches the repo's best lanes. Full record:
    issues #477 and #483.

    COMPARATOR REBUILT (issue #527, 2026-08-01 — all numbers measured, OLD
    objective): PR #516 moved |S11| from 0.2233 to ~0.05, and since
    sum|S|² was pinned near the passivity value 2·n_freqs = 16 (measured loss
    16.00599, i.e. 0.037% of signal on top of a structural constant), the
    gradient it differentiated collapsed 50x, from −2.1143e-01 to
    −4.2425e-03. The loss magnitude — and therefore its float32 ULP,
    1.9073e-06 — did not move. The FD signal 2h·|g| fell from ~222 ULP to
    4.4 ULP and the comparator stopped resolving:

        f32 comparator, gate settings, gate's h    rel_err 0.8519
        f64 comparator, gate settings, gate's h    rel_err 0.0331

    Both measured ON THIS LANE (gpu-rtx4090, VESSL 369367250775). A 4-point
    CPU run of the same referee reads rel_err {0.0035, 0.0053, 0.0040,
    0.0044} over h ∈ {1e-3, 2e-3, 5e-3, 1e-2} with a 0.972% h-spread — the
    GPU's f64 FD was the noisier of the two platforms, so 0.0331 was the
    conservative figure. The reference's own h-spread fell from 2017% (f32)
    to 3.4% (GPU) / 0.97% (CPU), and its resolving power rose from 4.4 to
    2.31e+09 ULP. This comparator machinery (the scoped-x64 FD reference,
    ``_fd_ulp_span``, ``_MIN_FD_ULP_SPAN``) is UNCHANGED by the #530 objective
    swap — only the loss/gradient MAGNITUDES it was measuring changed.

    (An earlier draft of this block offered "every sweep value was an exact
    integer multiple of ULP/(2h)" as the proof. That is a tautology — ANY two
    float32 values in the same binade differ by an integer number of ULPs, and
    200k random well-resolved pairs reproduce it. It shows only that the values
    were float32. The load-bearing fact is the MAGNITUDE of the span, which is
    what this block and _MIN_FD_ULP_SPAN now use.)

    The FD reference therefore runs in float64 under a SCOPED enable_x64(), and
    the resolving-power assert below runs BEFORE the accuracy gate so that a
    comparator failure is reported as a comparator failure. That assert is the
    check whose absence let #527 be filed against the AD path — the honest
    reading of a 0.85 rel_err on a 4.4-ULP reference is "the instrument is
    broken", not "the gradient is wrong". Harness:
    scripts/msl_ad_fd_f64_referee.py.
    """
    t_start = time.perf_counter()

    sim = _build_msl_sim()
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    # G-AD-CHECKPOINT: the uniform forward path uses checkpoint_segments
    # (issue #73; checkpoint_every is NU-only). The segment count must DIVIDE
    # n_steps exactly — padding is rejected because it would shift the DFT
    # accumulator windows. Pick the divisor nearest sqrt(n_steps) so backward
    # memory scales ~sqrt(n_steps)*carry instead of n_steps*carry (the OOM cause).
    n_steps = int(grid.num_timesteps(num_periods=_NUM_PERIODS))
    checkpoint_segments = _closest_divisor(n_steps, int(np.sqrt(n_steps)))
    # compute_msl_s_matrix passes n_steps=None, so forward() re-derives the SAME
    # grid.num_timesteps(num_periods) value used here; the segment count must
    # divide it exactly or the segmented scan hard-errors (simulation.py). Assert
    # the invariant locally so a future Courant/n_steps change fails loudly here.
    assert n_steps % checkpoint_segments == 0, (
        f"checkpoint_segments={checkpoint_segments} does not divide n_steps={n_steps}"
    )
    print(f"\n[MSL-FD-TIGHT] n_steps={n_steps}, "
          f"checkpoint_segments={checkpoint_segments} (~sqrt={np.sqrt(n_steps):.1f})")

    def objective(alpha: jnp.ndarray) -> jnp.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.compute_msl_s_matrix(
                n_freqs=_N_FREQS,
                num_periods=_NUM_PERIODS,
                eps_override=eps_base * alpha,
                checkpoint_segments=checkpoint_segments,
            )
        # Band-mean |S21|^2 (issue #530) — a smooth scalar that depends on
        # eps at every grid cell, and is NOT passivity-pinned the way
        # sum_ij|S_ij|^2 is (see module docstring / test docstring).
        return msl_band_mean_s21_sq(result.S)

    alpha0 = jnp.float32(1.0)

    # --- Forward sanity gate (R5) -------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd_result = sim.compute_msl_s_matrix(
            n_freqs=_N_FREQS,
            num_periods=_NUM_PERIODS,
            eps_override=eps_base * alpha0,
            checkpoint_segments=checkpoint_segments,
        )
    S_fwd = np.asarray(fwd_result.S)
    s_vals = np.abs(S_fwd)
    s_min = float(np.min(s_vals))
    s_max = float(np.max(s_vals))
    print(f"\n[MSL-FD-TIGHT] forward |S| range: [{s_min:.4f}, {s_max:.4f}]")

    assert s_max <= 1.2, (
        f"[MSL-FD-TIGHT] Forward |S|_max = {s_max:.4f} exceeds 1.2 — "
        "physically implausible. Gradient on an exploded impedance is meaningless. "
        "Check MSL forward path or geometry."
    )
    assert s_max > 0.0, (
        "[MSL-FD-TIGHT] Forward |S| = 0 everywhere — likely a broken forward pass."
    )

    # --- AD gradient ---------------------------------------------------------
    t_ad_start = time.perf_counter()
    loss_val, g = jax.value_and_grad(objective)(alpha0)
    t_ad = time.perf_counter() - t_ad_start

    g_ad = float(g)
    print(f"[MSL-FD-TIGHT] loss = {float(loss_val):.6e}")
    print(f"[MSL-FD-TIGHT] g_ad = {g_ad:.6e}  (AD wall-time: {t_ad:.1f}s)")

    assert jnp.isfinite(g), f"[MSL-FD-TIGHT] AD gradient is not finite: {g}"
    assert abs(g_ad) > 1e-10, (
        f"[MSL-FD-TIGHT] AD gradient is effectively zero ({g_ad:.3e}): "
        "tape may still be broken."
    )

    # --- Central finite-difference, float64 reference -------------------------
    # The two loss evaluations run under a SCOPED x64 context. Never flip x64 at
    # module level: it is process-global, flips at pytest collection, and reds
    # every same-process pytest-split shard. rfx/probes/probes.py already keys
    # its DFT accumulator dtype off jax.config.x64_enabled, so x64 reaches S.
    #
    # `grid` and `checkpoint_segments` are computed OUTSIDE this context and
    # reused inside it. Verified identical under both configs — shape
    # (142, 54, 19), n_steps 26226, cseg 141, same dt and dx — and a divergence
    # could not be silent anyway: a mismatched grid.shape would blow up on the
    # eps_override broadcast, and a cseg that stopped dividing n_steps hard-errors
    # in the segmented scan (plus the local assert above).
    t_fd_start = time.perf_counter()
    with enable_x64():
        sim64 = _build_msl_sim()
        eps64 = jnp.ones(grid.shape, dtype=jnp.float64)

        def objective64(alpha):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = sim64.compute_msl_s_matrix(
                    n_freqs=_N_FREQS, num_periods=_NUM_PERIODS,
                    eps_override=eps64 * alpha,
                    checkpoint_segments=checkpoint_segments,
                )
            return msl_band_mean_s21_sq(r.S)

        # Keep the ARRAYS. float() would widen them to Python floats — always
        # float64 — and the resolving-power check below would then measure the
        # ULP of the container instead of the dtype the loss was computed in.
        # That was a real defect in the first version of this block: fed the
        # pre-#527 float32 configuration it read 2.4e+09 ULP instead of 4.4 and
        # sailed through the assert it exists to trip (PR #529 review).
        #
        # Materialize each forward BEFORE dispatching the next. float() blocks;
        # .dtype does not. Issuing both first would let two f64 field sets hold
        # device buffers concurrently under JAX async dispatch — harmless here
        # (~10^2 MB, forward-only, no AD tape) but a deviation from the strictly
        # sequential profile the referee measured on the lane, and there is no
        # reason to pay for it.
        def _f64_loss(alpha):
            arr = objective64(jnp.float64(alpha))
            assert arr.dtype == jnp.float64, (
                "[MSL-FD-TIGHT] the FD reference did NOT run in float64 "
                f"(got {arr.dtype}). JAX truncates a float64 request to "
                "float32 when x64 is off, so the scoped context failed to "
                "engage — check jax.experimental.enable_x64 and the "
                "jax.config.x64_enabled keying in rfx/probes/probes.py. Do NOT "
                "read the rel_err below as physics: an f32 reference here spans "
                "only ~54 ULP (measured on this #530 objective/fixture; far "
                f"below the {_MIN_FD_ULP_SPAN:.0e}-ULP resolving-power floor) "
                f"and cannot be trusted against the {_REL_ERR_THRESHOLD:.2f} gate."
            )
            return float(arr), arr.dtype

        f_plus, loss_dtype = _f64_loss(1.0 + _FD_H)
        f_minus, _ = _f64_loss(1.0 - _FD_H)
    t_fd = time.perf_counter() - t_fd_start
    g_fd = (f_plus - f_minus) / (2.0 * _FD_H)
    print(f"[MSL-FD-TIGHT] g_fd = {g_fd:.6e}  (FD wall-time: {t_fd:.1f}s, h={_FD_H})")

    assert abs(g_fd) > 1e-10, (
        f"[MSL-FD-TIGHT] FD gradient is effectively zero ({g_fd:.3e}): "
        "objective may be constant w.r.t. alpha at num_periods={_NUM_PERIODS}."
    )

    # --- COMPARATOR VALIDITY (must precede the accuracy gate) ----------------
    # A reference that cannot resolve the quantity it is judging turns every
    # verdict into noise. This is the check whose absence made #527 read as a
    # gradient defect for four investigation rounds: the f32 comparator was
    # running at 4.4 ULP and the gate reported 0.85 rel_err as if it were
    # physics. Assert the reference's resolving power BEFORE comparing to AD,
    # so a comparator failure is reported as a comparator failure.
    #
    # The ULP MUST be taken at loss_dtype, not at the Python float the values
    # were widened into. The loss can only move in whole ULPs OF THE DTYPE IT
    # WAS COMPUTED IN, so that is the quantity that bounds the resolution.
    fd_ulp_span = _fd_ulp_span(f_plus, f_minus, loss_dtype)
    print(f"[MSL-FD-TIGHT] FD reference resolving power: "
          f"{fd_ulp_span:.3g} ULP of the loss (floor {_MIN_FD_ULP_SPAN:.0e})")
    assert fd_ulp_span >= _MIN_FD_ULP_SPAN, (
        f"[MSL-FD-TIGHT] the FD REFERENCE cannot resolve this gradient: "
        f"f(+h)-f(-h) spans only {fd_ulp_span:.3g} ULP of the loss "
        f"(need >= {_MIN_FD_ULP_SPAN:.0e}). This is a COMPARATOR failure, not "
        "an AD failure — do not touch the extractor or the tape on the "
        "strength of it. Raise h, raise the loss precision, or pick an "
        "objective with more dynamic range. See issue #527."
    )

    # --- Accuracy gate -------------------------------------------------------
    rel_err = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
    # The "tighten to 0.05 if it lands there" variable this block used to build
    # was never read by the assert below — the gate printed "threshold: 0.05"
    # while enforcing 0.10. Removed rather than wired up: at the measured GPU
    # rel_err 0.0331 against a 3.4% reference h-spread, enforcing 0.05 would be
    # 1.5x margin, i.e. a silent tightening of a live gate. Print what is
    # actually enforced (PR #529 review).
    t_total = time.perf_counter() - t_start
    print(f"[MSL-FD-TIGHT] rel_err = {rel_err:.4f} "
          f"(threshold: {_REL_ERR_THRESHOLD:.2f}, enforced below)")
    print(f"[MSL-FD-TIGHT] sign agreement: g_ad={g_ad:.4e} g_fd={g_fd:.4e}")
    print(f"[MSL-FD-TIGHT] total wall-time: {t_total:.1f}s")
    print(f"[MSL-FD-TIGHT] num_periods={_NUM_PERIODS}, n_freqs={_N_FREQS}")

    assert g_ad * g_fd > 0, (
        f"[MSL-FD-TIGHT] AD and FD gradients have OPPOSITE SIGNS: "
        f"g_ad={g_ad:.4e}, g_fd={g_fd:.4e}. "
        "This is a gradient accuracy failure, not a tolerance issue."
    )

    assert rel_err <= _REL_ERR_THRESHOLD, (
        f"[MSL-FD-TIGHT] AD gradient inaccurate at num_periods={_NUM_PERIODS}: "
        f"g_ad={g_ad:.4e}, g_fd={g_fd:.4e}, rel_err={rel_err:.4f} > {_REL_ERR_THRESHOLD}. "
        "This is a genuine gradient accuracy finding — do not loosen the gate. "
        "Investigate: (1) DFT window vs transient drain, (2) JAX float32 precision, "
        "(3) port extractor AD path for residual non-differentiable ops."
    )

    print("[MSL-FD-TIGHT] PASS")


def test_comparator_floor_rejects_the_f32_reference_that_caused_527():
    """The resolving-power floor must reject the comparator #527 shipped.

    ``_MIN_FD_ULP_SPAN`` is only worth having if it fires on the exact
    configuration that made this gate report a comparator artefact as a
    gradient defect. Two things this test is careful about, both of which the
    first version got wrong (PR #529 review):

    * it calls ``_fd_ulp_span`` — the SAME expression the gate evaluates —
      rather than re-deriving the arithmetic, so green here says something
      about the gate's own code path;
    * it passes the DTYPE, because ``float(jnp_scalar)`` is always a Python
      float and a value-only check reads float64 ULP for a float32 loss. Fed
      that way the first version read 2.4e+09 ULP for the f32 reference
      instead of 4.4 and passed.

    Fast and deterministic — no simulation. It replays the MEASURED numbers
    from the gate's own fixture UNDER THE RETIRED ``sum_ij|S_ij|^2``
    OBJECTIVE (issue #530 replaced it with band-mean ``|S21|**2`` — these are
    NOT current-gate numbers, they are the #527 comparator-floor incident
    this test regression-locks): loss 16.00599, g_ad -4.2425e-03 on CPU;
    16.005951 / -4.236159e-03 on gpu-rtx4090, VESSL 369367250775.
    """
    loss, g_true = 16.00599, 4.2425e-03
    f_plus, f_minus = loss + _FD_H * g_true, loss - _FD_H * g_true

    span32 = _fd_ulp_span(f_plus, f_minus, np.float32)
    span64 = _fd_ulp_span(f_plus, f_minus, np.float64)

    assert span32 < 10.0, (
        f"sanity: the f32 comparator should span a handful of ULP, got {span32:.3g}"
    )
    assert span32 < _MIN_FD_ULP_SPAN, (
        f"the floor ({_MIN_FD_ULP_SPAN:.0e}) does NOT reject the float32 "
        f"comparator that caused #527 ({span32:.3g} ULP)."
    )
    assert span64 >= _MIN_FD_ULP_SPAN, (
        f"the floor ({_MIN_FD_ULP_SPAN:.0e}) rejects the float64 comparator too "
        f"({span64:.3g} ULP) — it is set so high the gate can never run."
    )

    # ANCHOR THE FLOOR TO THE GATE, not to the one number it must reject.
    # A span of N ULP quantises the difference to ~1/N relative resolution, so
    # judging a threshold T needs 1/N << T. Requiring N*T >= 100 buys 100x
    # margin. Without this, a floor of 5.0 satisfies every assert above while
    # giving 20% resolution — inadequate against ANY realistic gate,
    # including the CURRENT 3% one (PR #529 review measured that the
    # unanchored version admitted any floor in (4.449, 2.388e5), 4.7 decades,
    # against the THEN-current 10% gate; those specific historical bounds are
    # not recomputed for the #530 objective here, only the "10%" label is).
    #
    # This assert is bidirectional: with _MIN_FD_ULP_SPAN fixed at 1e4, it
    # also lower-bounds any FUTURE tightening of _REL_ERR_THRESHOLD at
    # 100 / 1e4 = 0.01 — a future editor cannot silently tighten this gate
    # below 0.01 without first raising the resolving-power floor to match.
    assert _MIN_FD_ULP_SPAN * _REL_ERR_THRESHOLD >= 100.0, (
        f"floor {_MIN_FD_ULP_SPAN:.0e} gives only "
        f"{1.0 / _MIN_FD_ULP_SPAN:.2e} relative resolution against a "
        f"{_REL_ERR_THRESHOLD} gate; need span*threshold >= 100."
    )
    assert span64 / _MIN_FD_ULP_SPAN > 1.0e4, (
        f"float64 headroom above the floor has shrunk to {span64/_MIN_FD_ULP_SPAN:.3g}x"
    )


def test_fd_ulp_span_is_dtype_sensitive_not_container_sensitive():
    """The span helper must key off the loss dtype, not the Python container.

    Direct regression lock on the PR #529 blocking defect. Both arguments are
    Python floats in every call — as they are in the gate, after
    ``float(jnp_scalar)`` — so if the helper ever goes back to inferring
    precision from the value it will read the same number for both dtypes and
    this fails.

    The replayed ``loss``/``g_true`` below are UNDER THE RETIRED
    ``sum_ij|S_ij|^2`` OBJECTIVE (issue #530 replaced it with band-mean
    ``|S21|**2``) — this test locks the ``_fd_ulp_span`` helper's dtype
    sensitivity, a property independent of which objective the gate
    differentiates, so the historical numbers are fine to keep as the fixed
    replay input; they are not a claim about the current gate's fixture.
    """
    loss, g_true = 16.00599, 4.2425e-03
    f_plus, f_minus = loss + _FD_H * g_true, loss - _FD_H * g_true

    span32 = _fd_ulp_span(f_plus, f_minus, np.float32)
    span64 = _fd_ulp_span(f_plus, f_minus, np.float64)
    assert span64 / span32 > 1.0e6, (
        f"the helper is not dtype-sensitive: f32 {span32:.4g} vs f64 "
        f"{span64:.4g} ULP for identical Python-float inputs. It is measuring "
        "the container, so it cannot tell a float32 reference from a float64 "
        "one — the exact defect that let the first version of this gate pass "
        "on the configuration it exists to reject."
    )
    assert abs(span32 - 4.449) < 0.01, (
        f"f32 span drifted from the measured 4.449 ULP to {span32:.4g}. This "
        f"number is a function of _FD_H (currently {_FD_H:g}) as well as the "
        "loss and gradient, so if you just changed h this is expected — but "
        "the recorded measurement no longer applies and the docstring numbers "
        "above need re-measuring, not just this constant nudged."
    )
