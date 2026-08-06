#!/usr/bin/env python3
"""MSL band-mean |S21|^2 AD-gradient channel-attribution probe (issue #560).

THE OPEN QUESTION (issue #530 -> #559 review -> #560)
------------------------------------------------------
``test_msl_ad_fd_converged_tight`` differentiates band-mean ``|S21|**2``
(``tests._msl_ad_objective.msl_band_mean_s21_sq``) w.r.t. ``alpha``
(``eps_override = eps_base * alpha``) on the gate's own MSL thru fixture and
measures ``g_ad = +1.602933e-03`` (owner platform gpu-rtx4090, VESSL
369367251813/369367251827; tracked in
``scripts/diagnostics/msl_ad_band_mean_owner_measurement/owner_runs_20260804.md``).
Two candidate mechanisms for that gradient's magnitude were never
distinguished (adversarial review of PR #559, filed as #560):

  1. GENUINE PHYSICS: alpha shifts the guided wavelength via beta, which
     shifts the standing-wave pattern the line's electrical length puts at
     each port -- a real beta/reflection channel.
  2. REFERENCE-PLANE ARTIFACT: ``compute_msl_s_matrix``'s wave split
     (``a = (V + Z0*I)/2``, ``b = (V - Z0*I)/2``) uses the FROZEN analytic
     Hammerstad-Jensen ``z0_hj`` -- computed once from the REGISTERED
     substrate permittivity, before the alpha-dependent FDTD trace even
     starts, so it never moves with alpha -- rather than the per-frequency
     FITTED ``z0`` the same function's N-probe extractor (``extract_msl_nprobe``)
     also produces. As alpha changes the line's true Zc, the frozen z0_hj's
     drift relative to Zc(alpha) can itself synthesize an alpha-dependent
     apparent reflection/transmission split with nothing to do with beta.

The sign witness (g_ad > 0, i.e. increasing eps_r_sub REDUCES apparent
reflection) is SUGGESTIVE of mechanism 2 -- a reference-mismatch channel has
a natural sign preference; a pure beta/standing-wave channel would not --
but issue #560 is explicit that suggestive is not decisive, and specifies
the probe below as the thing that would settle it.

THE DECISIVE PROBE (issue #560's own text)
--------------------------------------------
Anchor the wave split on the per-frequency FITTED z0 instead of the frozen
analytic z0_hj, and re-run ONE ``jax.grad`` call of the SAME objective on
the SAME fixture. If |g_ad| collapses, the frozen-reference gap was
supplying most of the sensitivity (mechanism 2 settled). If |g_ad| stays
comparable, the reference gap is not dominant and the beta/reflection
channel survives (mechanism 1 supported, though issue #560 itself notes a
self-consistent z0_fit-based split changes MORE than one term at a time, so
even a "stays comparable" verdict does not fully PROVE mechanism 1 -- it
only fails to falsify it).

DECISION RULE (pre-declared before this script's first run)
--------------------------------------------------------------
Let g_a = AD gradient under the PRODUCTION anchor (frozen analytic z0_hj,
as shipped) and g_b = AD gradient under the SAME objective/fixture but with
the anchor swapped for a FROZEN (alpha-independent, same discipline as the
production anchor: a single constant baked in before the alpha-dependent
trace) per-port z0 measured from a preliminary alpha=1.0 forward run.
ratio = |g_a| / |g_b|.

    ratio >= 5   -> COLLAPSE. The frozen-reference normalization gap is the
                    dominant channel (mechanism 2). Do not read a future
                    eps_override gradient on this objective as "the line's
                    electrical response to substrate permittivity" without
                    that caveat.
    ratio <= 2   -> NO COLLAPSE. The reference-plane channel is not
                    dominant; the beta/reflection-physics reading survives
                    (mechanism 1) at least as the leading explanation.
    2 < ratio < 5 -> AMBIGUOUS. Report both numbers; leave #560 open with
                    the measurement recorded rather than force a verdict.

Why 5x / 2x and not other numbers: 5x is issue #560's OWN proposed
threshold ("say >5x smaller" -- quoted verbatim from the issue body) for
"collapse", so re-deriving a different number here would silently redefine
the question #560 asked. 2x (not "close to 1") is the complementary side:
inside a factor of 2 is the same order of magnitude, which given this is a
single-run attribution measurement (not a precision comparator with a
measured noise floor) is a defensible boundary for "did not move
materially" without claiming a false precision the single-run design cannot
support. The gap between 2x and 5x is deliberately left as an honest
gray zone rather than picked to force a clean verdict either way.

WHAT THIS PROBE DOES **NOT** CHANGE, EITHER WAY (issue #530 / #527 history)
-------------------------------------------------------------------------------
``test_msl_ad_fd_converged_tight``'s pass/fail status, its 0.03 threshold,
and its validity as an AD-vs-FD comparator are UNAFFECTED by this probe's
outcome. The gate differentiates ``compute_msl_s_matrix(...).S`` through
``jax.grad`` on one side and a central finite difference on the other --
the SAME function on both sides, whatever that function's dominant physical
channel turns out to be. #527 already established (as a SEPARATE, prior
incident) that a comparator can fail for reasons that have nothing to do
with gradient correctness (there: the float32 loss ran out of ULP
resolving power after PR #516 shrank the differentiated signal 50x -- a
COMPARATOR defect, not a gradient defect). This probe is a different
question again: not "is the AD gradient numerically correct" (answered,
rel_err 0.0026 against a 2.9e10-ULP f64 reference) but "what physical story
explains why it has the magnitude it does". Whatever this probe finds, the
gate keeps doing its job; only the INTERPRETATION attached to a passing
gate's gradient changes.

HOW THE PROBE IS IMPLEMENTED (injection point)
--------------------------------------------------
Issue #560's text names ``rfx/api/_sparams.py``'s ``a_fwd_d``/``b_ref_d``
computation (the driven-port block, ~line 2942-2946) as where to swap the
anchor. Reading the function end to end shows that block is NOT actually
where the injection needs to happen for THIS 2-port fixture: ``a_fwd_d``
and ``b_ref_d`` only ever feed (1) ``S[driven, driven]`` and (2) the
off-diagonal single-ratio fallback ``S[j, driven] = b_out_p / alpha_d`` --
and BOTH are unconditionally overwritten by the issue-#507 multi-drive
solve (``S = B @ A^-1`` via ``msl_solve_s_from_waves``) whenever every port
has been driven, which happens on every 2-port run including this one (and,
under jax tracing, ``cond_a`` is always ``None`` so the ``_bad`` guard that
could in principle keep the fallback never fires -- the multi-drive solve
result unconditionally overwrites S). Patching only ``a_fwd_d``/``b_ref_d``
would therefore be a NO-OP on the actual objective value: it would change
numbers nothing downstream reads.

The quantity that DOES reach the final S (and therefore the objective) is
the ``wave_a``/``wave_b`` matrices the multi-drive solve consumes, built by
(rfx/api/_sparams.py ~2978-2985)::

    for j in range(n_ports):
        z0_j = z0_hj_per_port[j]
        wave_a[driven][j] = 0.5 * (v0_j + z0_j * i_j)
        wave_b[driven][j] = 0.5 * (v0_j - z0_j * i_j)

-- which reuses the SAME per-port constant ``z0_hj_per_port[j]``, computed
once per port (not per drive) from ``hammerstad_jensen_z0_eps_eff(width,
height, eps_r_ref)`` with ``eps_r_ref`` read from the REGISTERED substrate
material (``self._assemble_materials(grid)``, which does not take
``eps_override`` -- confirmed by reading the call site), so it is frozen
w.r.t. alpha regardless of which specific line consumes it. Anchoring on
the fitted z0 therefore means anchoring the ORIGIN of ``z0_hj_per_port``,
not the vestigial ``a_fwd_d``/``b_ref_d`` intermediate -- a broader (and,
for this fixture, the ONLY effective) swap than the issue text's literal
line numbers, but the same swap in spirit: every consumer of the frozen
analytic reference moves to the fitted one together, self-consistently per
port.

``hammerstad_jensen_z0_eps_eff`` is imported LOCALLY inside
``compute_msl_s_matrix`` (``from rfx.sources.msl_eigenmode import
hammerstad_jensen_z0_eps_eff``, re-executed on every call), so
``unittest.mock.patch.object`` on ``rfx.sources.msl_eigenmode``'s module
attribute intercepts every call with ZERO edits to
``rfx/api/_sparams.py`` and zero change to any production default. The
patch wrapper below calls the REAL function to obtain ``eps_eff`` (used
only to seed ``beta0_per_port``, the extractor's beta-scan anchor) and
passes it through UNCHANGED -- only ``z0`` is replaced -- so the probe
isolates the z0 reference channel alone; the beta anchor is identical
between anchor A and anchor B.

Why a SCALAR anchor, not the full 8-bin fitted array: ``z0_hj_per_port``
is built via ``z0_hj_per_port.append(float(z0_hj))``
(rfx/api/_sparams.py ~2638) -- the production code hard-casts to a Python
float the instant the call returns. There is no per-frequency channel for
this constant anywhere downstream; array-valued spoofing would silently
diverge from what the production code path actually is. The frozen anchor
used here is therefore the frequency-band MEAN of the REAL part of the
fitted Z0 (``result.Z0[port, :]``) from a preliminary, un-patched,
alpha=1.0 forward run of the identical fixture -- computed and printed for
both ports, applied per-port by call order (``hammerstad_jensen_z0_eps_eff``
is called once per port, in ``entries`` order, inside
``compute_msl_s_matrix``, so a call-count closure correctly attributes each
call to its port).

DTYPE / PLATFORM
-------------------
float32 throughout (the gate's own AD dtype -- "as shipped"), CPU (no CUDA
jaxlib installed in this environment; ``jax.devices() == [CpuDevice(id=0)]``).
This is a channel-ATTRIBUTION question (does the gradient survive an
anchor swap, order-of-magnitude comparison), not a precision comparator, so
CPU is adequate -- issue #560 says so explicitly. The committed owner-
platform number (gpu-rtx4090, VESSL) is quoted for context but this
script's own g_a is the one compared against g_b (both measured on the
SAME platform/dtype/fixture in the SAME process), which is the apples-to-
apples comparison the ratio actually needs.

FALSIFIER / DETERMINISM CHECK (R3)
--------------------------------------
The FDTD forward pass and its AD tape are deterministic (no RNG anywhere in
this fixture). Each anchor's ``jax.value_and_grad`` call is run TWICE; if
either anchor's (loss, g) pair is not reproduced exactly, that is a defect
in the harness (e.g. a leaking monkeypatch, a stale JAX cache) and the
result must not be trusted -- the probe is its own falsifier, per the task
framing: no report is issued from a harness that fails its own
determinism check.

USAGE
-----
    python scripts/diagnostics/msl_ad_z0_anchor_probe.py
    python scripts/diagnostics/msl_ad_z0_anchor_probe.py --out-json PATH

Writes a JSON summary next to this script by default
(``msl_ad_z0_anchor_probe.json``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402

_COLLAPSE_RATIO = 5.0
_SURVIVE_RATIO = 2.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - provenance-only
        return f"<unavailable: {exc}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-periods", type=int, default=None,
                     help="default: the gate's own _NUM_PERIODS (20)")
    ap.add_argument("--n-freqs", type=int, default=None,
                     help="default: the gate's own _N_FREQS (8)")
    ap.add_argument("--repeats", type=int, default=2,
                     help="determinism check: repeat each anchor's AD call "
                          "this many times and require an exact match")
    ap.add_argument("--out-json", type=str, default=None,
                     help="default: msl_ad_z0_anchor_probe.json next to "
                          "this script")
    ap.add_argument("--only", choices=["both", "a", "b"], default="both",
                     help="run only one anchor (for re-confirming determinism "
                          "on a single anchor without re-paying the other "
                          "anchor's + the preliminary forward's cost)")
    ap.add_argument("--z0-fit-ohm", type=float, nargs="+", default=None,
                     help="skip the preliminary forward pass and use these "
                          "precomputed per-port band-mean real(Z0) values "
                          "(ohm) as anchor B's frozen constant instead -- "
                          "for re-running anchor B alone from a prior run's "
                          "printed 'fitted z0' values")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    import rfx
    print(f"rfx        : {rfx.__file__}")
    print(f"git SHA    : {_git_sha()}")
    print(f"jax        : {jax.__version__}   devices: {jax.devices()}")
    print("dtype      : float32 (AD, as shipped)   platform: CPU")
    if not str(Path(rfx.__file__).resolve()).startswith(str(REPO)):
        print(f"FATAL: imported rfx is not this checkout ({REPO})")
        return 2

    from rfx.sources import msl_eigenmode as _hj_mod
    from test_msl_ad_fd_converged import (  # noqa: E402
        _N_FREQS,
        _NUM_PERIODS,
        _build_msl_sim,
        _closest_divisor,
    )
    from tests._msl_ad_objective import msl_band_mean_s21_sq  # noqa: E402

    num_periods = args.num_periods if args.num_periods is not None else _NUM_PERIODS
    n_freqs = args.n_freqs if args.n_freqs is not None else _N_FREQS

    sim_probe = _build_msl_sim()
    grid = sim_probe._build_grid()
    n_steps = int(grid.num_timesteps(num_periods=num_periods))
    cseg = _closest_divisor(n_steps, int(np.sqrt(n_steps)))
    assert n_steps % cseg == 0
    print(f"fixture    : num_periods={num_periods} n_freqs={n_freqs} "
          f"(gate defaults: {_NUM_PERIODS}, {_N_FREQS})")
    print(f"grid       : {grid.shape}  n_steps={n_steps}  checkpoint_segments={cseg}")

    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    alpha0 = jnp.float32(1.0)

    # -- Preliminary concrete forward run (production, unpatched) -----------
    # Establishes: (1) the frozen z0_hj analytic anchor per port (for
    # cross-reference/printing only -- the production path recomputes this
    # itself), (2) the FITTED z0 per port at alpha0, which becomes the
    # anchor-B frozen constant, (3) which S-assembly path is active on this
    # fixture (expected: multi_drive_solve). Skipped when --z0-fit-ohm is
    # given (re-confirming one anchor's determinism without re-paying this
    # pass's + the other anchor's cost).
    assembly_path = None
    if args.z0_fit_ohm is not None:
        z0_fit_per_port = list(args.z0_fit_ohm)
        print("\n--- preliminary forward SKIPPED (--z0-fit-ohm given) ---")
        print(f"  using precomputed z0_fit_per_port = {z0_fit_per_port}")
    else:
        print("\n--- preliminary forward (production, unpatched, alpha=1.0) ---",
              flush=True)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fwd = sim_probe.compute_msl_s_matrix(
                n_freqs=n_freqs, num_periods=num_periods,
                eps_override=eps_base * alpha0, checkpoint_segments=cseg,
            )
        print(f"  wall-time: {time.perf_counter()-t0:.1f}s")
        assembly_path = fwd.assembly
        print(f"  assembly path: {fwd.assembly!r} "
              "(expected 'multi_drive_solve' -- see script header)")
        z0_fit_per_port = [
            float(np.mean(np.real(np.asarray(fwd.Z0[p, :])))) for p in range(fwd.Z0.shape[0])
        ]
        s_abs = np.abs(np.asarray(fwd.S))
        print(f"  forward |S| range: [{float(s_abs.min()):.4f}, {float(s_abs.max()):.4f}]")
        for p, z0f in enumerate(z0_fit_per_port):
            z0_im = float(np.mean(np.imag(np.asarray(fwd.Z0[p, :]))))
            print(f"  port {p}: fitted z0 (band-mean, real) = {z0f:.4f} ohm "
                  f"(band-mean imag = {z0_im:.4f} ohm)")

    # -- Objective (identical to the gate's) ---------------------------------
    def _make_objective(sim):
        def objective(alpha: jnp.ndarray) -> jnp.ndarray:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sim.compute_msl_s_matrix(
                    n_freqs=n_freqs, num_periods=num_periods,
                    eps_override=eps_base * alpha,
                    checkpoint_segments=cseg,
                )
            return msl_band_mean_s21_sq(result.S)
        return objective

    def _run_ad(sim):
        loss, g = jax.value_and_grad(_make_objective(sim))(alpha0)
        return float(loss), float(g)

    run_a = args.only in ("both", "a")
    run_b = args.only in ("both", "b")

    # -- Anchor A: production, frozen analytic z0_hj -------------------------
    a_runs = []
    if run_a:
        print("\n--- anchor A: production (frozen analytic Hammerstad-Jensen z0_hj) ---",
              flush=True)
        for r in range(args.repeats):
            sim_a = _build_msl_sim()
            t0 = time.perf_counter()
            loss_a, g_a = _run_ad(sim_a)
            dt = time.perf_counter() - t0
            a_runs.append((loss_a, g_a))
            print(f"  run {r}: loss = {loss_a:.8f}   g_ad = {g_a:.6e}   ({dt:.1f}s)")

    # -- Anchor B: frozen FITTED z0 (measured at alpha0, held constant) -----
    def _make_frozen_z0_patch(z0_by_port):
        """Wrap hammerstad_jensen_z0_eps_eff: keep eps_eff (beta anchor)
        from the REAL analytic function; replace z0 with a precomputed,
        alpha-independent constant, attributed to a port by call order
        (compute_msl_s_matrix calls this once per port, in registration
        order, before the alpha-dependent FDTD trace starts)."""
        orig = _hj_mod.hammerstad_jensen_z0_eps_eff
        state = {"i": 0}

        def _patched(w, h, eps_r):
            _z0_orig, eps_eff_orig = orig(w, h, eps_r)
            idx = state["i"] % len(z0_by_port)
            state["i"] += 1
            return float(z0_by_port[idx]), eps_eff_orig

        return _patched

    b_runs = []
    if run_b:
        print("\n--- anchor B: frozen fitted z0 (measured at alpha=1.0, held constant) ---",
              flush=True)
        for r in range(args.repeats):
            sim_b = _build_msl_sim()
            with mock.patch.object(
                _hj_mod, "hammerstad_jensen_z0_eps_eff",
                _make_frozen_z0_patch(z0_fit_per_port),
            ):
                t0 = time.perf_counter()
                loss_b, g_b = _run_ad(sim_b)
                dt = time.perf_counter() - t0
            b_runs.append((loss_b, g_b))
            print(f"  run {r}: loss = {loss_b:.8f}   g_ad = {g_b:.6e}   ({dt:.1f}s)")

    # -- Determinism check (R3 falsifier) ------------------------------------
    a_det = all(a_runs[0] == r for r in a_runs[1:]) if a_runs else None
    b_det = all(b_runs[0] == r for r in b_runs[1:]) if b_runs else None
    if a_runs:
        print(f"\n[determinism] anchor A exact match across {len(a_runs)} runs: {a_det}")
    if b_runs:
        print(f"[determinism] anchor B exact match across {len(b_runs)} runs: {b_det}")
    if (a_runs and not a_det) or (b_runs and not b_det):
        print("\nFATAL: harness is non-deterministic -- the probe cannot be "
              "trusted. Not issuing a verdict.")
        return 3

    out_path = (
        Path(args.out_json) if args.out_json
        else Path(__file__).resolve().with_suffix(".json")
    )

    if not (run_a and run_b):
        # Partial run (--only a / --only b): print/save what was measured and
        # stop here -- there is nothing to compute a ratio/verdict against.
        partial = {
            "issue": 560, "partial_run": True, "only": args.only,
            "git_sha": _git_sha(), "jax_version": jax.__version__,
            "z0_fit_per_port_ohm_real_band_mean": z0_fit_per_port,
            "a_runs": a_runs, "a_deterministic": a_det,
            "b_runs": b_runs, "b_deterministic": b_det,
        }
        out_path.write_text(json.dumps(partial, indent=2) + "\n")
        print(f"\nPartial run ({args.only!r}) complete. Wrote {out_path}")
        return 0

    loss_a, g_a = a_runs[0]
    loss_b, g_b = b_runs[0]
    ratio = abs(g_a) / (abs(g_b) + 1e-300)
    same_sign = (g_a > 0) == (g_b > 0)

    print("\n=== RESULT ===")
    print(f"  g_a (production, frozen analytic z0_hj)      = {g_a:.6e}  (loss {loss_a:.8f})")
    print(f"  g_b (frozen fitted z0, held at alpha0 value)  = {g_b:.6e}  (loss {loss_b:.8f})")
    print(f"  |g_a| / |g_b|                                 = {ratio:.3f}")
    print(f"  same sign                                     = {same_sign}")

    if ratio >= _COLLAPSE_RATIO:
        verdict = "collapse_mechanism2_dominant"
        headline = (
            f"COLLAPSE (ratio {ratio:.2f} >= {_COLLAPSE_RATIO:g}x): the "
            "frozen-reference normalization gap (mechanism 2) is the "
            "dominant channel behind d(band-mean|S21|^2)/d(alpha) on this "
            "fixture."
        )
    elif ratio <= _SURVIVE_RATIO:
        verdict = "survives_mechanism1_supported"
        headline = (
            f"NO COLLAPSE (ratio {ratio:.2f} <= {_SURVIVE_RATIO:g}x): the "
            "reference-plane channel is not dominant; the beta/reflection-"
            "physics reading (mechanism 1) survives as the leading "
            "explanation."
        )
    else:
        verdict = "ambiguous"
        headline = (
            f"AMBIGUOUS (ratio {ratio:.2f}, between {_SURVIVE_RATIO:g}x and "
            f"{_COLLAPSE_RATIO:g}x): neither mechanism is settled by this "
            "measurement alone."
        )
    print(f"\n  VERDICT: {headline}")

    payload = {
        "issue": 560,
        "rfx_file": str(rfx.__file__),
        "git_sha": _git_sha(),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "dtype": "float32",
        "platform": "cpu",
        "fixture": {
            "num_periods": num_periods,
            "n_freqs": n_freqs,
            "grid_shape": list(grid.shape),
            "n_steps": n_steps,
            "checkpoint_segments": cseg,
        },
        "assembly_path": assembly_path,
        "z0_fit_per_port_ohm_real_band_mean": z0_fit_per_port,
        "repeats": args.repeats,
        "anchor_a_production_frozen_analytic_z0_hj": {
            "loss": loss_a, "g_ad": g_a, "all_runs": a_runs, "deterministic": a_det,
        },
        "anchor_b_frozen_fitted_z0": {
            "loss": loss_b, "g_ad": g_b, "all_runs": b_runs, "deterministic": b_det,
        },
        "ratio_abs_ga_over_abs_gb": ratio,
        "same_sign": same_sign,
        "decision_rule": {
            "collapse_ratio_threshold": _COLLAPSE_RATIO,
            "survive_ratio_threshold": _SURVIVE_RATIO,
        },
        "verdict": verdict,
        "headline": headline,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
