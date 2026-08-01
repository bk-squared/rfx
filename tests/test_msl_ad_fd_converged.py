"""MSL-FD-TIGHT: Converged tight AD-vs-FD cross-check for end-to-end gradient.

MSL-FD-TIGHT (2026-05-25) adds a slow-marked test that runs
compute_msl_s_matrix(eps_override=...) at num_periods=20 (converged DFT)
and asserts jax.grad agrees with a central finite-difference to a tight
tolerance (rel_err <= 0.10). The reference itself runs in float64 and must
clear a resolving-power floor before its verdict is read — see issue #527.

This converts the "AD tape flows + roughly right" evidence from
test_sparam_ad_end_to_end.py (num_periods=3, rel_err=16%) into
"AD gradient is accurate" evidence.

Geometry mirrors _build_msl_sim() in test_sparam_ad_end_to_end.py exactly.
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64  # SCOPED x64 — never flip it at module level

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

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
# The one tolerance, printed and enforced. A "tighten to 0.05" variable used
# to be computed here and never read by the assert, so the gate printed 0.05
# while enforcing 0.10 (PR #529 review).
_REL_ERR_THRESHOLD = 0.10

# Minimum resolving power the FD reference must have before its disagreement
# with AD means anything: |f(+h) - f(-h)| expressed in ULPs of the loss.
# The loss is a float, so it can only move in whole ULPs — a difference N ULPs
# wide is quantised to ~1/N relative resolution no matter how exact the solver
# is. The shipped float32 comparator ran at N = 4.4 (issue #527), i.e. ~23%
# resolution against a 10% gate. 1e4 ULP is 0.01% resolution, 1000x inside the
# gate's bound; the float64 comparator measures 2.4e9.
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
    """MSL-FD-TIGHT: converged (num_periods=20) AD gradient matches FD to <=10%.

    G-AD-CHECKPOINT (un-skipped 2026-05-26): the num_periods=20 reverse-AD tape
    is now segmented via checkpoint_segments → forward(), so it runs within
    memory budget instead of being OOM-killed. Marked gpu+slow (VESSL-owned).

    R5 instrumentation: prints g_ad, g_fd, rel_err, and forward |S| range.
    If forward |S| is outside [0, 1.2], the test fails explicitly rather than
    silently reporting a gradient on an exploded impedance.

    If rel_err stays >10% this test will fail (deliberately — do NOT loosen
    the gate to force a pass; report as a gradient accuracy finding instead).

    OWNERSHIP + measured comparator envelope (issue #477 root-cause,
    2026-07-28 — all numbers measured, main @ 98c5e33):
    This is a GPU-lane gate (gpu marker; VESSL harness). Do NOT treat a CPU
    execution as a main-health signal: the FD comparator carries an f32
    evaluation-noise envelope that straddles the 0.10 gate across platforms
    while the AD side is platform-stable to 5 digits
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

    COMPARATOR REBUILT (issue #527, 2026-08-01 — all numbers measured):
    The sentence that used to end this block — "what remains in THIS f32 gate
    is only the #477 comparator-noise envelope (±3–5%), so the 0.10 threshold
    now carries real margin" — was false, and PR #516 exposed it. That fix
    moved |S11| from 0.2233 to ~0.05, and since sum|S|² is pinned near the
    passivity value 2·n_freqs = 16 (measured loss 16.00599, i.e. 0.037% of
    signal on top of a structural constant), the gradient it differentiates
    collapsed 50x, from −2.1143e-01 to −4.2425e-03. The loss magnitude — and
    therefore its float32 ULP, 1.9073e-06 — did not move. The FD signal
    2h·|g| fell from ~222 ULP to 4.4 ULP and the comparator stopped resolving:

        f32 comparator, gate settings, gate's h    rel_err 0.8519
        f64 comparator, gate settings, gate's h    rel_err 0.0331

    Both measured ON THIS LANE (gpu-rtx4090, VESSL 369367250775), so the number
    quoted is the one the gate will actually read. A 4-point CPU run of the same
    referee reads rel_err {0.0035, 0.0053, 0.0040, 0.0044} over
    h ∈ {1e-3, 2e-3, 5e-3, 1e-2} with a 0.972% h-spread — the GPU's f64 FD is
    the noisier of the two platforms, so 0.0331 is the conservative figure. The
    reference's own h-spread fell from 2017% (f32) to 3.4% (GPU) / 0.97% (CPU),
    and its resolving power rose from 4.4 to 2.31e+09 ULP.

    COST, measured on the lane, since a float64 reference sounds expensive and
    is not: f32 AD 59.1s, f64 FD 41.7s and 37.7s per h (two forwards each),
    i.e. ~19s per f64 forward against ~15s for f32. This FDTD is
    memory-bandwidth-bound, so the RTX4090's 1/64 float64 throughput barely
    shows; the whole gate runs in ~140s. On CPU the same forward takes ~500s,
    which is why the decision was made from a measurement on the lane the gate
    actually runs in rather than from a local timing.

    A 7-point h-sweep on this fixture (VESSL 369367250742) put the span between
    1 and 76 ULP across h ∈ [3e-4, 2e-2] — at h = 5e-4 the whole measurement
    was ONE single ULP — while the FD value scattered 2017% and changed sign.
    Raising h does not fix it: 1% resolution needs h > 0.0225, where h²
    truncation takes over.

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

    The 0.10 threshold is UNCHANGED. Nothing here loosens a gate.

    WHAT THIS GATE DOES NOT ESTABLISH (issue #530). Passivity bounds
    sum_ij|S_ij|^2 at 2 per frequency, so 16 over 8 bins — and the measured loss
    is 16.00599, which sits 0.037% ABOVE that ceiling. The residue this gate
    differentiates is therefore dominated by extraction/numerical error, not by
    physical loss. That is a valid AD-vs-FD consistency test — AD must match the
    FD of whatever the code computes, and it does, to 0.4% on CPU — but it is a
    weak test of anything physical, and its signal will shrink again the next
    time an extractor fix lands. The comparator floor above now makes that
    failure mode loud instead of silent. Choosing an objective with real dynamic
    range is tracked separately; it needs its own evidence, not a rider on a
    comparator repair.
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
        S = result.S
        # Sum of |S|^2 over all matrix entries and all frequency bins —
        # a smooth scalar that depends on eps at every grid cell.
        return jnp.real(jnp.sum(jnp.abs(S) ** 2))

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
            return jnp.real(jnp.sum(jnp.abs(r.S) ** 2))

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
                "~4 ULP and cannot resolve the 10% gate."
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
    from the gate's own fixture (loss 16.00599, g_ad -4.2425e-03 on CPU;
    16.005951 / -4.236159e-03 on gpu-rtx4090, VESSL 369367250775).
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
    # giving 20% resolution against a 10% gate — the review measured that the
    # unanchored version admitted any floor in (4.449, 2.388e5), 4.7 decades.
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
