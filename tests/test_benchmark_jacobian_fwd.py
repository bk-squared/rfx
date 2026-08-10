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
      baseline for flops/temp_bytes/wall time (the primal-sharing
      witness; band matches the repo's own established tolerance in
      tests/test_estimate_ad_memory.py::test_segmented_estimate_within_tolerance).
    - batched (batch_tangents=True) wall time is faster than sequential
      (batch_tangents=False) at the same n_t.
    - XLA temp_bytes is independent of n_steps (forward-mode memory does
      not scale with n_steps; only wall time does).
"""

from __future__ import annotations

from scripts.benchmark_jacobian_fwd import build_benchmark_table


def test_benchmark_table_relations():
    table = build_benchmark_table(
        n_p=4, n_t_values=(1, 2, 4), n_steps=60, grid_scale=0, reps=3,
    )

    ratios = table["intercept_vs_plain_ratio"]
    for name, ratio in ratios.items():
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
    assert batched_row["t_median_s"] < seq_row["t_median_s"], (
        f"batched wall time {batched_row['t_median_s']:.4f}s is not faster "
        f"than sequential {seq_row['t_median_s']:.4f}s at n_t={seq_row['n_t']}"
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
