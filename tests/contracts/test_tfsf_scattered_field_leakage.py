"""Vacuum TF/SF leakage and public run/forward probe-record equality."""

import sys
import time

import numpy as np
import pytest

from rfx import Simulation


STEPS = 240
LEAKAGE_BAR = 1e-4
KINDS = ("normal", "bloch", "methodB", "closed")
SF_POSITIONS = ((.001, .006, .005), (.001, .001, .005))
TF_POSITION = (.009, .006, .005)


@pytest.fixture(scope="module", params=KINDS)
def plane_wave(request):
    # Cache returned public records only, so each entry starts a fresh solve.
    return request.param, {}


def _record(kind, entry, layers, records):
    key = entry, layers
    if key not in records:
        sim = Simulation(freq_max=30e9, domain=(.0243, .0187, .0141),
                         dx=.001, cpml_layers=layers)
        sim.add_tfsf_source(
            f0=15e9, bandwidth=.15, margin=3,
            angle_deg=20 if kind in ("bloch", "methodB") else 0,
            method="methodB" if kind == "methodB" else "bloch",
            closed_box=kind == "closed",
        )
        # Keep an interior transverse sample and an edge sample: a spurious
        # transverse absorber diffracts at the edge before reaching the interior.
        for position in SF_POSITIONS:
            sim.add_probe(position, "ez")
        sim.add_probe(TF_POSITION, "ez")
        grid = sim.freeze_mesh()
        sf = tuple(tuple(int(grid.index_of(a, p)) for a, p in enumerate(position))
                   for position in SF_POSITIONS)
        tf = tuple(int(grid.index_of(a, p)) for a, p in enumerate(TF_POSITION))
        # The low injection plane is pad + margin. The scattered probe lies
        # between it and the absorber. All probes avoid every symmetry plane.
        lo = tuple(layers + 3 for _ in range(3))
        hi = tuple(n - layers - 4 for n in grid.shape)
        assert all(grid.pad_x_lo <= idx[0] < lo[0] for idx in sf)
        assert all(a < i < b for a, i, b in zip(lo, tf, hi))
        assert all(2*i != n - 1 for idx in (*sf, tf) for i, n in zip(idx, grid.shape))
        options = {"compute_s_params": False} if entry == "run" else {"checkpoint": False}
        started = time.perf_counter()
        result = getattr(sim, entry)(n_steps=STEPS, skip_preflight=True, **options)
        record = np.asarray(result.time_series)
        elapsed = time.perf_counter() - started
        assert record.shape == (STEPS, 3)
        assert np.isfinite(record).all()
        records[key] = record
        print(f"B1.3 {kind}/{entry}/cpml{layers}: mesh={grid.shape} "
              f"sf_index={sf} tf_index={tf} injection_bounds={lo, hi} "
              f"steps={STEPS} solve_wall_s={elapsed:.9g}", file=sys.stderr)
    return records[key]


@pytest.mark.parametrize("entry", ("run", "forward"))
def test_vacuum_scattered_field_leakage_and_lane_parity(plane_wave, entry):
    kind, records = plane_wave
    # The depth-doubled public model keeps the source, probes and 240-step
    # record fixed. Both absorbers must satisfy the same absolute leakage bar.
    for layers in (4, 8):
        row = f"{kind}/{entry}/cpml{layers}"
        record = _record(kind, entry, layers, records)
        peak = float(np.max(np.abs(record[:, -1])))
        scattered_peaks = np.max(np.abs(record[:, :-1]), axis=0)
        scattered = float(np.max(scattered_peaks))
        assert peak > 0, f"{row}: total-field record must be nonzero"
        leakage = scattered / peak
        print(f"B1.3 {row}: leakage={leakage:.12g} bar={LEAKAGE_BAR:.12g} "
              f"scattered_peak={scattered:.12g} total_peak={peak:.12g} "
              f"probe_ratios={(scattered_peaks / peak).tolist()} "
              f"expected_scattered=0 steps={STEPS}", file=sys.stderr)
        if entry == "forward":
            reference = _record(kind, "run", layers, records)
            wide = np.result_type(record.dtype, np.float64)
            difference = float(np.max(np.abs(record.astype(wide) - reference)))
            ulps = difference / float(np.spacing(np.float32(np.max(np.abs(reference)))))
            print(f"B1.3 {row}: run_forward_peak_ulps={ulps:.12g} bar=0 "
                  f"max_difference={difference:.12g}", file=sys.stderr)
        assert leakage <= LEAKAGE_BAR, (
            f"{row}: scattered/total={leakage:.12g} > {LEAKAGE_BAR}; "
            f"scattered={scattered:.12g}, total={peak:.12g}, steps={STEPS}")
        if entry == "forward":
            # Existing pair: test_run_forward_one_pipeline.py:525 (PW-A,
            # PW-B, PW-oblique, PW-closed), run_forward_census.py:786-789.
            # It requires equal bytes, stronger than the default 9 peak ULP.
            # Bloch also uses bytes at unit/autodiff/test_tfsf_forward_measurement.py:91.
            assert record.dtype == reference.dtype and record.tobytes() == reference.tobytes(), (
                f"{row}: run/forward records differ by {ulps:.12g} peak ULP; bar=0")
