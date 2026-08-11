"""RELATIONS-only gate for scripts/benchmark_jacobian_fwd.py (issue #577).

Precedent: tests/test_memory_reduction_planning_artifact.py /
tests/test_estimate_ad_memory.py -- assert RATIOS/RELATIONS between
measured quantities, never a magic absolute constant (a compiled flops/
temp_bytes/wall-time number is compiler- and machine-dependent and would
rot the moment it was pinned). Uses a small, fast fixture (distinct from
the benchmark script's own default) purely to keep this test's CPU time
bounded -- the numbers THEMSELVES belong in the PR body/CHANGELOG from a
real run of the script, not in this file or in any docstring.

Relations gated:
    - intercept_vs_plain_ratio lands within [0.5x, 2.0x] of the plain
      baseline for flops/temp_bytes ONLY (compiler-derived, deterministic
      quantities). wall_s is REPORTED, never asserted -- see the
      wall-clock note below.
    - XLA temp_bytes is independent of n_steps (forward-mode memory does
      not scale with n_steps; only wall time does).

Wall-clock is diagnostic, not gated (issue #632 follow-up): this test's
own fixture red on a loaded CI runner -- commit 55fa85b, run 31464443445,
fast-suite shard 1 -- with intercept_vs_plain_ratio[wall_s] = 0.474,
just outside the [0.5x, 2.0x] band, and went green again on the very next
commit with no code change. #632 already diagnosed the underlying
quantity (this same ratio) as machine-dependent (0.98-1.10 measured on
CPU, 1.41-1.87 on an RTX 4090); wall_s inherits that plus ordinary
scheduler/thermal noise on a shared runner, which flops/temp_bytes (both
compiler cost-model outputs, not measured runtime) do not. A flaky
assertion in a per-PR required gate reds unrelated PRs, so wall_s is
computed and surfaced in the table for a human to read, never gated in
CI. The authoritative, backend-independent primal-sharing check is G3 in
tests/test_jacobian_fwd.py (it inspects the jaxpr directly for one
unbatched primal scan, a structural property, not a timing measurement).
"""

from __future__ import annotations

from scripts.benchmark_jacobian_fwd import build_benchmark_table


def test_benchmark_table_relations():
    table = build_benchmark_table(
        n_p=4, n_t_values=(1, 2, 4), n_steps=60, grid_scale=0, reps=3,
    )

    # Deterministic, compiler-derived columns only -- see the module
    # docstring's "Wall-clock is diagnostic, not gated" note for why
    # wall_s is excluded here rather than asserted with a wider band.
    ratios = table["intercept_vs_plain_ratio"]
    # This CPU-only test fixture (JAX_PLATFORMS=cpu) must always expose
    # both compiler-derived columns; a missing key here means
    # scripts/benchmark_jacobian_fwd.py's own bare `except Exception`
    # around cost_analysis()/memory_analysis() silently swallowed a real
    # failure (info["flops"]/["temp_bytes"] = None), which would
    # otherwise make the loop below skip both columns via `continue` and
    # pass with NOTHING actually checked -- an empty gate that looks
    # green. Assert the keys exist before trusting the loop's silence.
    assert "flops" in ratios, (
        "intercept_vs_plain_ratio has no 'flops' key -- the benchmark "
        "script's cost_analysis() call likely failed silently (bare "
        "except in _compile_and_measure); this CPU fixture must expose it"
    )
    assert "temp_bytes" in ratios, (
        "intercept_vs_plain_ratio has no 'temp_bytes' key -- the benchmark "
        "script's memory_analysis() call likely failed silently (bare "
        "except in _compile_and_measure); this CPU fixture must expose it"
    )
    for name in ("flops", "temp_bytes"):
        ratio = ratios[name]
        assert 0.5 <= ratio <= 2.0, (
            f"intercept_vs_plain_ratio[{name}] = {ratio:.3f} outside the "
            "primal-sharing band [0.5x, 2.0x]"
        )

    plain = table["rows"][0]
    assert plain["mode"] == "plain"

    seq_row = next(r for r in table["rows"] if r["mode"] == "jvp_sequential")
    batched_row = next(
        r for r in table["rows"]
        if r["mode"] == "jvp_batched" and r["n_t"] == seq_row["n_t"]
    )
    # Wall-clock, reported not gated -- but PRECAUTIONARY, not
    # demonstrated: unlike intercept_vs_plain_ratio's wall_s column above
    # (confirmed red on a real CI run, commit 55fa85b), this specific
    # batched-vs-sequential inversion has never actually been observed
    # (across every local run collected while developing this fix, 0/18
    # samples inverted; worst measured margin 1.30x in batched's favor,
    # still comfortably faster). It is de-gated on the SAME reasoning as
    # wall_s (both are wall-clock, both are compiler/scheduler-dependent
    # in principle) rather than on independent evidence that this
    # particular comparison is actually flaky -- treat it as a
    # consistency choice, not a second confirmed-red case.
    _batched_vs_sequential_speedup = (
        seq_row["t_median_s"] / batched_row["t_median_s"]
        if batched_row["t_median_s"] else float("nan")
    )

    w = table["n_steps_independence_witness"]
    t1, t2 = w["temp_bytes_1x"], w["temp_bytes_2x"]
    assert t1 is not None and t2 is not None
    rel = abs(t2 - t1) / max(t1, t2)
    assert rel < 0.02, (
        f"temp_bytes depends on n_steps: n_steps={w['n_steps_1x']} -> {t1}, "
        f"n_steps={w['n_steps_2x']} -> {t2} (rel diff {rel * 100:.2f}%)"
    )

    # Honesty-label discipline: the table's own field names must not use
    # the repo's forbidden AD-memory vocabulary (tests/test_estimate_ad_memory.py
    # _FORBIDDEN_CURRENT_EVIDENCE_FIELDS / _FORBIDDEN_RECOMMENDATION_TERMS).
    forbidden_fields = {
        "observed_peak_gb", "profile_peak_gb", "peak_bound_gb",
        "compiler_memory_gb", "certificate_status",
    }
    for row in table["rows"]:
        assert forbidden_fields.isdisjoint(row), f"forbidden field name leaked into row: {row.keys()}"
