"""MSL feeds on identical cells: uniform run records are the reference (#2619)."""
import json
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation


CELL = 2.0**-12
STEPS = 512
FREQS = np.array([2e9, 6e9, 10e9])


def _line(mode, graded):
    d = CELL
    sim = Simulation(
        snap="declared", freq_max=20e9, domain=(48*d, 24*d, 12*d),
        dx=d, boundary="cpml", cpml_layers=8,
        **({"dz_profile": np.full(12, d)} if graded else {}))
    sim.add_material("substrate", eps_r=2.2)
    sim.add(Box((0, 0, 0), (48*d, 24*d, 4*d)), material="substrate")
    sim.add(Box((0, 0, 0), (48*d, 24*d, 0)), material="pec")
    sim.add(Box((8*d, 9*d, 4*d), (40*d, 15*d, 4*d)), material="pec")
    for name, x, direction in (("p1", 8*d, "+x"), ("p2", 40*d, "-x")):
        sim.add_msl_port(
            position=(x, 12*d, 0), width=6*d, height=4*d,
            direction=direction, impedance=50., eps_r_sub=2.2,
            name=name, mode=mode, n_probe_offset=6,
            n_probe_spacing=3, n_probes=3)
    return sim


def _solve(mode, graded, monkeypatch):
    from rfx.model import conductors

    captured = []
    releases = []
    original = Simulation.run
    original_kernel = conductors.at_kernel

    def record_kernel(*args, **kwargs):
        result = original_kernel(*args, **kwargs)
        releases.append(result[0].released_port_edges)
        return result

    def record(self, **kwargs):
        result = original(self, **kwargs)
        captured.append(result)
        return result

    with monkeypatch.context() as patch, warnings.catch_warnings():
        patch.setattr(Simulation, "run", record)
        patch.setattr(conductors, "at_kernel", record_kernel)
        warnings.simplefilter("ignore")
        result = _line(mode, graded).compute_msl_s_matrix(
            freqs=FREQS, n_steps=STEPS)
    assert len(captured) == 2
    assert len(releases) == 2
    return captured, np.asarray(result.S), releases


def _measurement(name, reference, actual, kind):
    reference, actual = np.asarray(reference), np.asarray(actual)
    assert reference.shape == actual.shape, name
    assert np.isfinite(reference).all() and np.isfinite(actual).all(), name
    peak = float(np.max(np.abs(reference)))
    difference = float(np.max(np.abs(
        actual.astype(np.complex128) - reference.astype(np.complex128))))
    ulp = float(np.spacing(np.float32(peak)))
    bar = 9 * ulp if kind == "step" else 1e-4 * peak
    return dict(record=name, kind=kind, peak=peak, difference=difference,
                ulps=difference / ulp,
                relative=difference / peak if peak else (0. if difference == 0 else float("inf")),
                bar=bar, passed=difference <= bar)


@pytest.mark.parametrize("mode", ["laplace", "uniform"])
def test_msl_constant_profile(mode, monkeypatch, capsys):
    uniform, s_uniform, uniform_releases = _solve(mode, False, monkeypatch)
    graded, s_graded, graded_releases = _solve(mode, True, monkeypatch)
    measurements = []
    metrics = []
    for drive, (reference, actual) in enumerate(zip(uniform, graded, strict=True)):
        assert reference.grid.shape == actual.grid.shape
        # Compare the metrics at the float32 stepping precision; the two
        # host CFL calculations can differ below that representation.
        np.testing.assert_array_equal(np.float32(reference.dt), np.float32(actual.dt))
        metrics.append(dict(drive=drive, uniform_dt=float(reference.dt),
                            graded_dt=float(actual.dt), cell_differences=[]))
        for axis in range(3):
            ref_cells, got_cells = reference.grid.cells(axis), actual.grid.cells(axis)
            np.testing.assert_array_equal(ref_cells, got_cells)
            metrics[-1]["cell_differences"].append(float(np.max(np.abs(ref_cells - got_cells))))
        # Independently enumerate every V-plane and both sides of each H-plane
        # consumed by the S extraction, for both ports and both driven runs.
        names = {
            f"_msl_run{drive}_p{port}_{component}"
            for port in range(2)
            for component in ("ez0", "ez1", "ez2", "hy", "hy_left", "hz", "hz_left")
        }
        assert set(reference.dft_time_records) == set(actual.dft_time_records) == names
        for name in sorted(names):
            ref, got = reference.dft_time_records[name], actual.dft_time_records[name]
            assert ref.shape[0] == got.shape[0] == STEPS
            measurements.append(_measurement(name, ref, got, "step"))
        # Include the point witnesses the port path also retains.
        for probe in range(reference.time_series.shape[1]):
            measurements.append(_measurement(
                f"drive{drive}.witness{probe}", reference.time_series[:, probe],
                actual.time_series[:, probe], "step"))
    for receiver in range(2):
        for drive in range(2):
            measurements.append(_measurement(
                f"S{receiver+1}{drive+1}", s_uniform[receiver, drive],
                s_graded[receiver, drive], "accumulated"))
    with capsys.disabled():
        print("\nMSL_COMPARISON " + json.dumps(dict(
            mode=mode, metrics=metrics, records=measurements,
            frequencies=FREQS.tolist(), graded_s_real=s_graded.real.tolist(),
            graded_s_imag=s_graded.imag.tolist(),
            release_counts=dict(uniform=list(map(len, uniform_releases)),
                                graded=list(map(len, graded_releases))))))
    failures = [row for row in measurements if not row["passed"]]
    assert not failures, json.dumps(failures, indent=2)
    # Release metadata controls exclusion of design metal around the feed.
    # Compare the actual kernel products with the uniform runner's products.
    assert all(uniform_releases)
    assert graded_releases == uniform_releases
