"""Hand-calculated oracles; no timing, rfx import, or accelerator execution."""

import math

import pytest

from scripts.perf.step_cost_census import (
    classify, compiled_opcode_counts, cost_record, finish_report, log_slope, mad_sigma, pair_statistics,
    structural_counts, subtract_counts,
)


def test_median_mad_interleaved_pair_and_seconds_per_step():
    # A median=4, absolute deviations 3,2,1,0,1,2,3 -> MAD=2.
    # B median=9, absolute deviations 6,4,2,0,2,4,6 -> MAD=4.
    a, b = list(range(1, 8)), list(range(3, 16, 2))
    result = pair_statistics(a, b)
    assert result["base_runs_seconds"] == a
    assert result["row_runs_seconds"] == b
    assert result["base_median_seconds"] == 4
    assert result["row_median_seconds"] == 9
    assert result["t_base_seconds"] == 4 / 200
    assert result["t_row_seconds"] == 9 / 200
    assert result["sigma_base_seconds"] == pytest.approx(1.4826 * 2 / 200)
    assert result["sigma_row_seconds"] == pytest.approx(1.4826 * 4 / 200)
    noise = 1.4826 * math.sqrt(2**2 + 4**2) / 200
    assert result["sigma_delta_seconds"] == pytest.approx(noise)
    assert result["delta_seconds"] == pytest.approx(5 / 200)
    assert result["floored_delta_seconds"] == pytest.approx(3 * noise)


def test_mad_robust_to_one_outlier():
    # Median=10, deviations sorted 0,1,1,2,2,3,990; MAD=2.
    assert mad_sigma([7, 8, 9, 10, 11, 12, 1000]) == pytest.approx(2 * 1.4826)


def test_negative_delta_is_floored():
    result = pair_statistics([10, 11, 12, 13, 14, 15, 16], [1, 2, 3, 4, 5, 6, 7], steps=1)
    assert result["delta_seconds"] == -9
    assert result["floored_delta_seconds"] == pytest.approx(3 * 1.4826 * math.sqrt(8))


def test_visible_delta_is_not_changed_by_floor():
    result = pair_statistics([1]*7, [5]*7, steps=1)
    assert result["delta_seconds"] == result["floored_delta_seconds"] == 4


def test_slope_uses_least_squares_not_endpoint_ratio():
    # log C = 0,1,3 times ln 2; log y = 0,2,2 times ln 2.
    # Both means=4/3; covariance=8/3, variance=14/3 -> slope=4/7.
    assert log_slope([1, 2, 8], [1, 4, 4]) == pytest.approx(4 / 7)


def test_undefined_log_has_no_invented_epsilon():
    assert log_slope([1, 2, 4], [0, 0, 0]) is None
    assert log_slope([1, 1, 1], [1, 2, 3]) is None


def _points(deltas=(1, 2, 4), footprints=(1, 1, 1), noise=.01):
    # Hand-set synthetic C=1,2,4, independently of the physical N labels.
    return [dict(N=n, C=c, F=f, delta_seconds=d, floored_delta_seconds=max(d, 3*noise),
                 sigma_delta_seconds=noise, sigma_base_seconds=noise,
                 t_base_seconds=1, status="ok")
            for n, c, f, d in zip((56, 88, 144), (1, 2, 4), footprints, deltas)]


ZERO = {"scatter": 0, "dynamic-update-slice": 0, "copy": 0, "fusion": 0}


def test_domain_cost_is_flagged_by_a():
    result = classify(_points(), ZERO, ZERO)
    assert result["s"] == pytest.approx(1)
    assert result["s_F"] == pytest.approx(0)
    assert result["e"] == pytest.approx(1)
    assert result["growth_seconds"] == 3
    assert result["flag"] == "a"


def test_realized_footprint_slope_is_fitted_not_forced_to_two_thirds():
    # F=C and delta=C -> s=s_F=1, e=0; scaled endpoint growth=0.
    result = classify(_points(footprints=(1, 2, 4)), ZERO, ZERO)
    assert result["s_F"] == pytest.approx(1)
    assert result["e"] == pytest.approx(0)
    assert result["growth_seconds"] == 0
    assert result["flag"] == "none"


def test_timing_growth_threshold_is_strict():
    points = _points(deltas=(1, 2, 4), noise=1)
    # Floored values=3,3,4 give s=ln(4/3)/ln(4), below .3 anyway;
    # choose positive floors independently to exercise the endpoint clause.
    for p, value in zip(points, (1, 2, 4)):
        p["floored_delta_seconds"] = value
    result = classify(points, ZERO, ZERO)
    assert result["growth_seconds"] == 3 * points[-1]["sigma_delta_seconds"]
    assert result["flag"] == "none"


def test_excess_below_threshold_is_not_flagged():
    # delta=C**.25 -> e=.25, below .3 despite observable endpoint growth.
    result = classify(_points(deltas=(1, 2**.25, 4**.25), noise=1e-6), ZERO, ZERO)
    assert result["e"] == pytest.approx(.25)
    assert result["flag"] == "none"


def test_below_noise_is_judged_only_by_structure():
    result = classify(_points(deltas=(-2, 0, 1), noise=1), ZERO, {**ZERO, "copy": 1})
    assert result["below_noise"] is True
    assert result["a"] is False
    assert result["flag"] == "b"


def test_constant_positive_structure_is_listed_not_flagged():
    counts = {**ZERO, "copy": 3}
    result = classify(_points(deltas=(1, 1, 1)), counts, counts)
    assert result["b"] is False
    assert result["flag"] == "none"


@pytest.mark.parametrize("kind", tuple(ZERO))
def test_each_structure_kind_can_flag_and_negative_base_deltas_are_valid(kind):
    f, f2 = {**ZERO, kind: -2}, {**ZERO, kind: -1}
    assert classify(_points(), f, f2)["flag"] == "a/b"


def test_untrusted_timing_is_disabled_table_wide():
    points1, points2 = _points(), _points()
    points2[-1]["sigma_base_seconds"] = .050001
    for points in (points1, points2):
        points[1]["structure"] = {"F": {"minus_base": ZERO}, "2F": {"minus_base": ZERO}}
    report = {"rows": {"base": {"sizes": points1}, "control": {"sizes": points2}}}
    finish_report(report)
    assert report["timing_trusted"] is False
    assert all(r["statistics"]["timing_clause_raw"] for r in report["rows"].values())
    assert all(r["statistics"]["flag"] == "none" for r in report["rows"].values())
    assert report["control_flagged"] is False


def test_five_percent_equality_remains_trusted():
    points = _points()
    points[-1]["sigma_base_seconds"] = .05
    points[1]["structure"] = {"F": {"minus_base": ZERO}, "2F": {"minus_base": ZERO}}
    report = {"rows": {"base": {"sizes": points}}}
    finish_report(report)
    assert report["timing_trusted"] is True


def test_structural_counts_inspect_outputs_and_include_exact_c():
    lowered = '''
    %0 = "stablehlo.scatter"(%big, %i, %u) ({
      stablehlo.return %arg : tensor<f32>
    }) : (tensor<8x8xf32>, tensor<i32>, tensor<f32>) -> tensor<8x8xf32>
    %1 = "stablehlo.scatter"(%big, %i, %u) ({
      stablehlo.return %arg : tensor<f32>
    }) : (tensor<100x100xf32>, tensor<i32>, tensor<f32>) -> tensor<2x2xf32>
    %2 = stablehlo.dynamic_update_slice %a, %b, %i, %j : (tensor<8x8xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8xf32>
    %3 = stablehlo.dynamic_update_slice %a, %b, %i, %j : (tensor<7x8xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<7x8xf32>
    '''
    compiled = '''
    %a = f32[8,8]{1,0} copy(%big)
    %b = f32[2,2]{1,0} copy(%big)
    %c = (f32[8,8]{1,0}, f32[1]{0}) fusion(%a), kind=kLoop
    %d = f32[2,2]{1,0} fusion(%big), kind=kLoop
    '''
    assert structural_counts(lowered, compiled, 64) == {
        "scatter": 1, "dynamic-update-slice": 1, "copy": 1, "fusion": 1}
    assert subtract_counts({k: 2 for k in ZERO}, {k: 3 for k in ZERO}) == {k: -1 for k in ZERO}


@pytest.mark.parametrize("runs", [[1]*6, [1]*8])
def test_exact_seven_runs_required(runs):
    with pytest.raises(ValueError, match="7\\+7"):
        pair_statistics(runs, [1]*7)


def test_invalid_times_rejected():
    with pytest.raises(ValueError, match="finite"):
        pair_statistics([float("nan")]*7, [1]*7)


def test_missing_size_is_not_silently_classified():
    with pytest.raises(ValueError, match="three Design sizes"):
        classify(_points()[:2], ZERO, ZERO)


def test_all_compiled_opcodes_outputs_exclusions_and_nested_bodies():
    compiled = '''
    %fused_computation (x: f32[8,8]) -> f32[8,8] {
      %x = f32[8,8]{1,0} parameter(0)
      %c = f32[8,8]{1,0} constant({...})
      %b = f32[8,8]{1,0} bitcast(%x)
      %g = f32[8,8]{1,0} get-tuple-element(%tuple), index=0
      %t = (f32[8,8], f32[8,8]) tuple(%x, %c)
      %small = f32[2,2]{1,0} gather(%x, %index)
      %large = f32[8,8]{1,0} gather(%x, %index)
      %larger = f32[9,8]{1,0} gather(%x, %index)
      %slice = f32[8,8]{1,0} dynamic-update-slice(%x, %small, %i)
      ROOT %result = f32[8,8]{1,0} add(%large, %x)
    }
    ENTRY %main {
      %fusion = f32[8,8]{1,0} fusion(%input), kind=kLoop, calls=%fused_computation
      %loop = (s32[], f32[8,8]{1,0}) while(%init), body=%body, condition=%cond
      %small_tuple = (f32[4,8], f32[4,8]) custom-call(%input), custom_call_target="small"
      ROOT %out = f32[8,8]{1,0} optimization-barrier(%fusion)
    }
    '''
    # Two large gathers, one each of add/DUS/fusion/while/barrier.
    # Small outputs (even when their tuple sum is 64) and five excluded
    # opcodes contribute zero. The fused body's add counts separately.
    assert compiled_opcode_counts(compiled, 64) == {
        "add": 1, "dynamic-update-slice": 1, "fusion": 1, "gather": 2,
        "optimization-barrier": 1, "while": 1,
    }


def test_new_opcode_and_disappearing_opcode_are_zero_filled():
    assert subtract_counts({"gather": 5, "add": 2}, {"gather": 3, "copy": 1}) == {
        "add": 2, "copy": -1, "gather": 2}
    result = classify(_points(deltas=(1, 1, 1)), {"copy": 5}, {"gather": 1})
    assert result["growing_opcodes"] == ["gather"]
    assert result["flag"] == "b"


def test_c_domain_bytes_are_primary_even_when_timing_is_untrusted_and_below_noise():
    points = _points(deltas=(0, 0, 0), noise=100)
    # B_feature=116,132,164 minus B_base=100 -> ΔB=16,32,64.
    # C=1,2,4 and F=8,8,8 -> s_c=1, s_F=0, e_c=1.
    for p, feature in zip(points, (116, 132, 164)):
        p.update(F=8, delta_bytes_accessed=feature - 100)
    result = classify(points, {}, {"gather": 24}, timing_trusted=False)
    assert result["s_c"] == pytest.approx(1)
    assert result["e_c"] == pytest.approx(1)
    assert result["s_F"] == pytest.approx(0)
    assert result["below_noise"] is True
    assert result["a"] is False
    assert result["flag"] == "b/c"


def test_c_uses_fitted_footprint_slope_and_least_squares():
    points = _points(deltas=(1, 1, 1), footprints=(1, 2, 8))
    for p, cells, value in zip(points, (1, 2, 8), (1, 4, 4)):
        p.update(C=cells, delta_bytes_accessed=value)
    # log C=0,1,3; log ΔB=0,2,2 -> slope=(8/3)/(14/3)=4/7.
    # F=C -> s_F=1; e_c=-3/7.
    result = classify(points, {}, {})
    assert result["s_c"] == pytest.approx(4 / 7)
    assert result["e_c"] == pytest.approx(-3 / 7)
    assert result["c"] is False


@pytest.mark.parametrize("deltas", [(0, 0, 0), (-1, 1, 2), (1, 0, 4)])
def test_c_nonpositive_byte_deltas_have_no_log_floor(deltas):
    points = _points(deltas=(1, 1, 1))
    for p, delta in zip(points, deltas):
        p["delta_bytes_accessed"] = delta
    result = classify(points, {}, {})
    assert result["s_c"] is None
    assert result["e_c"] is None
    assert result["c"] is False


def test_c_constant_bytes_and_excess_below_threshold_are_not_flagged():
    for values, expected in (((16, 16, 16), 0), ((1, 2**.25, 4**.25), .25)):
        points = _points(deltas=(1, 1, 1))
        for p, value in zip(points, values):
            p["delta_bytes_accessed"] = value
        result = classify(points, {}, {})
        assert result["e_c"] == pytest.approx(expected)
        assert result["flag"] == "none"


@pytest.mark.parametrize("wrapped", [False, True])
def test_cost_record_retains_bytes_without_dividing_by_scan_length(wrapped):
    class Executable:
        def cost_analysis(self):
            record = {"bytes accessed": 32768, "flops": 4096}
            return [record] if wrapped else record
    assert cost_record(Executable()) == {
        "bytes_accessed": 32768, "raw": {"bytes accessed": 32768, "flops": 4096}}


def test_cost_record_refuses_ambiguous_multiple_device_dictionaries():
    class Executable:
        def cost_analysis(self):
            return [{"bytes accessed": 16}, {"bytes accessed": 16}]
    with pytest.raises(ValueError, match="one executable"):
        cost_record(Executable())
