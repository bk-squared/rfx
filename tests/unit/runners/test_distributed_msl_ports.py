"""B1 field and accumulator bars, using public one-device records as references.

All runs use the declared default Gaussian pulse, unchanged geometry, frequencies,
observers and record length. Probe scale is always the one-device 2100-step peak.
No normalization, phase alignment, time shift or fitted gain is applied to data.
"""
import json
from time import perf_counter

import jax
import numpy as np
import pytest

from tests._multi_device_scene import host_devices
from tests.unit.runners._msl_multidevice_scene import FREQS, build_scene, report_realized


def observed_scene(name, lane, mode, n):
    scene = build_scene(name + ("_graded" if lane == "graded" else ""), mode=mode)
    sim = scene.sim
    grid = sim._build_realized_grid()
    realized = report_realized(scene, n)
    ports = realized["ports"]
    if name == "Y1":
        for port in ports:
            assert port["feed"]["x_indices"] == list(range(23, 33))
            assert port["feed"]["counts_per_rank"] == ([24, 16] if n == 2 else [0, 40, 0])
    else:
        assert [p["feed"]["x_indices"] for p in ports] == [[18], [58]]
        assert [p["feed"]["x_owners"] for p in ports] == [[0], [n - 1]]
    if name == "T2":
        assert [p["x_owners"][0] for p in ports[0]["ladder"]] == (
            [0, 0, 1, 1, 1] if n == 2 else [1] * 5)
    if name == "T1L":
        assert realized["lumped_ports"][0]["x_owners"] == [1]

    # Observer indices come from the declaration and grid coordinates, not the
    # realization implementation or its source/profile/reduction helpers.
    axis = "y" if name == "Y1" else "x"
    width_axis = "x" if axis == "y" else "y"
    centre = .0039 if axis == "y" else .004
    width = .0018 if axis == "y" else .0012
    wc = grid.index_of(width_axis, centre)
    lo, hi = (grid.index_of(width_axis, centre + sign * width / 2) for sign in (-1, 1))
    zlo, zhi = grid.index_of("z", 0.), grid.index_of("z", .0008)
    points = []
    for feed in (.002, .010):
        pos = (centre, feed, .0004) if axis == "y" else (feed, centre, .0004)
        points.append(pos)
    points.append(points[-1])  # explicit far-port observer
    for seam in realized["seams"]:
        for side in (-1, 1):
            points.append((grid.node_of("x", seam + side),
                           .006 if axis == "y" else centre, .0004))
    for pos in points:
        sim.add_probe(pos, "ez")
    planes = [(f"p{i + 1}", axis, feed) for i, feed in enumerate((.002, .010))]
    if name == "T2":
        seam = realized["seams"][0]
        planes.extend((f"seam{side}", "x", grid.node_of("x", seam + side))
                      for side in (-1, 0))
    for label, normal, coordinate in planes:
        for component in ("ez", "hx" if normal == "y" else "hy", "hz"):
            sim.add_dft_plane_probe(axis=normal, coordinate=coordinate,
                                   component=component, freqs=FREQS,
                                   name=f"{label}_{component}")
    dz = np.array([grid.node_of("z", k + 1) - grid.node_of("z", k)
                   for k in range(grid.shape[2] - 1)])
    dw = np.array([grid.node_of(width_axis, k + 1) - grid.node_of(width_axis, k)
                   for k in range(grid.shape[0 if axis == "y" else 1] - 1)])
    return sim, (axis, wc, lo, hi, zlo, zhi, dz, dw)


def vi(result, geometry, port):
    """Literal centre-column voltage and four-leg right-handed Ampere contour."""
    axis, wc, lo, hi, zlo, zhi, dz, dw = geometry
    ez = np.asarray(result.dft_planes[f"{port}_ez"].accumulator)
    hw = np.asarray(result.dft_planes[f"{port}_{'hx' if axis == 'y' else 'hy'}"].accumulator)
    hz = np.asarray(result.dft_planes[f"{port}_hz"].accumulator)
    voltage = (ez[:, wc, zlo:zhi] * dz[zlo:zhi]).sum(axis=1)
    current = ((hw[:, lo:hi + 1, zhi - 1] - hw[:, lo:hi + 1, zhi])
               * dw[lo:hi + 1]).sum(axis=1)
    current += (hz[:, hi, zhi] - hz[:, lo - 1, zhi]) * dz[zhi]
    if axis == "y":
        current = -current  # cyclic transverse pair (z,x), not reflected (x,z)
    if port == "p1":
        current = -current
    return voltage, current


@pytest.fixture(scope="module", params=[2, 3], ids=lambda n: f"n{n}")
def device_count(request):
    devices = host_devices(request.param)  # before ANY reference solve
    return request.param, devices


@pytest.fixture(scope="module", params=[
    (scene, lane, mode) for scene in ("T1", "T2", "Y1", "T1L")
    for lane in ("uniform", "graded") for mode in ("laplace", "uniform")
], ids=lambda case: "-".join(case))
def reference(request, device_count):
    n, devices = device_count
    name, lane, mode = request.param
    sim, geometry = observed_scene(name, lane, mode, n)
    ref = sim.run(n_steps=2100, compute_s_params=False, skip_preflight=True)
    jax.block_until_ready(ref.time_series)
    yield sim, geometry, ref, devices, dict(scene=name, lane=lane, mode=mode, n=n)
    jax.clear_caches()


@pytest.mark.parametrize("length", [1050, 1400, 2100])
def test_msl_fields_and_planes(reference, length):
    sim, geometry, longest, devices, metadata = reference
    expected = longest if length == 2100 else sim.run(
        n_steps=length, compute_s_params=False, skip_preflight=True)
    started = perf_counter()
    actual = sim.run(n_steps=length, devices=devices, compute_s_params=False,
                     skip_preflight=True)
    jax.block_until_ready((actual.time_series, actual.dft_planes))
    report = dict(metadata, length=length, seconds=perf_counter() - started,
                  probes=[], planes={}, vi={})
    a, b = np.asarray(actual.time_series), np.asarray(expected.time_series)
    peak = np.max(np.abs(np.asarray(longest.time_series)), axis=0)
    delta = np.max(np.abs(a - b), axis=0)
    assert np.all(peak > 0), peak
    for d, p in zip(delta, peak):
        ulp = float(d / np.spacing(np.float32(p)))
        report["probes"].append(dict(delta=float(d), peak=float(p), relative=float(d / p),
                                      ulp=ulp, hold=ulp > 9))
    for name, plane in expected.dft_planes.items():
        p = float(np.max(np.abs(np.asarray(plane.accumulator))))
        d = float(np.max(np.abs(np.asarray(actual.dft_planes[name].accumulator)
                                - np.asarray(plane.accumulator))))
        report["planes"][name] = dict(delta=d, peak=p, relative=d / p if p else None)
    for port in ("p1", "p2"):
        for label, av, bv in zip(("V", "I"), vi(actual, geometry, port), vi(expected, geometry, port)):
            report["vi"][f"{port}_{label}"] = dict(
                delta=np.abs(av - bv).tolist(), reference=np.abs(bv).tolist(),
                relative=(np.abs(av - bv) / np.abs(bv)).tolist())
    print("B1 " + json.dumps(report), flush=True)
    assert np.all(delta <= 9e-6 * peak), report
    for name, row in report["planes"].items():
        assert row["peak"] > 0, (name, row)
        assert row["delta"] <= 1e-4 * row["peak"], (name, row)
    for name, row in report["vi"].items():
        assert np.all(np.asarray(row["delta"]) <= 1e-5 * np.asarray(row["reference"])), (name, row)


@pytest.mark.parametrize("script", ["check_layer_structure", "check_file_size_ratchet",
                                    "check_changelog_fragment"])
def test_msl_repository_checks(script):
    """Run the required PR checks through labrun's named-pytest-file route."""
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    command = [sys.executable, f"scripts/ci/{script}.py", "--base", "831180f12", "--head", "HEAD"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    print("CHECK", command, "exit", result.returncode, result.stdout, result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
