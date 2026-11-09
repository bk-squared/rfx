"""PMC image plane and cavity observables through the public solver paths.

The low magnetic wall is on E node zero: H[-1] = -H[0]. Its image
therefore has 26 physical x cells, with the shared E node at index 13.
Closed forms use the physical cell counts; pole media use the public
Simulation mirror model, independently of the curl implementation.
"""

import sys

import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, Simulation, lorentz_pole
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.harminv import matrix_pencil

H = 1e-3
CELLS = (13, 10, 3)
C = 299792458.0
STEPS = 2400
IMAGE_STEPS = 300


def _cavity(mesh="uniform", feature="plain", wall="pmc", image=False):
    a, b, d = (n * H for n in CELLS)
    interior = np.linspace(.87, 1.13, CELLS[0] - 2)
    interior *= (a - 2 * H) / interior.sum()
    widths = np.r_[H, interior, H]
    options = {}
    if mesh == "graded":
        options["dx_profile"] = np.r_[widths[::-1], widths] if image else widths
    sim = Simulation(
        freq_max=30e9, domain=((2 if image else 1) * a, b, d), dx=H,
        boundary=BoundarySpec(x=Boundary(lo="pec" if image else wall, hi="pec"),
                              y="pec", z="pec"), cpml_layers=0, **options)
    shift = a if image else 0.
    source_x = 2 * H if mesh == "uniform" else float(sum(widths[:2]))
    probe_x = 7 * H if mesh == "uniform" else float(sum(widths[:7]))
    pulse = GaussianPulse(f0=16e9, bandwidth=1.2)
    sim.add_source((shift + source_x, 3 * H, H), "ez", waveform=pulse,
                   amplitude_kind="field")
    if image:
        sim.add_source((shift - source_x, 3 * H, H), "ez", waveform=pulse,
                       amplitude_kind="field")
    sim.add_probe((shift + probe_x, 6 * H, H), "ez")
    if feature in ("lumped", "wire"):
        extra = {"extent": H} if feature == "wire" else {}
        sim.add_port((3 * H, 4 * H, H), "ez", impedance=1e9,
                     excite=False, **extra)
    if feature in ("debye", "lorentz", "mixed"):
        material = dict(eps_r=2.4)
        if feature in ("debye", "mixed"):
            material["debye_poles"] = [DebyePole(delta_eps=1.1, tau=1e-11)]
        if feature in ("lorentz", "mixed"):
            material["lorentz_poles"] = [lorentz_pole(delta_eps=.6, omega_0=8e10, delta=2e9)]
        sim.add_material("block", **material)
        # Keep the curl_cases pole block; its upper z face extends past this
        # thinner cavity. The doubled model reflects both material and source.
        sim.add(Box((shift + 4.3e-3, 2.2e-3, 1.4e-3),
                    (shift + 8.6e-3, 6.9e-3, 5.3e-3)), material="block")
        if image:
            sim.add(Box((shift - 8.6e-3, 2.2e-3, 1.4e-3),
                        (shift - 4.3e-3, 6.9e-3, 5.3e-3)), material="block")
    return sim


def _solve(sim, lane="run", steps=IMAGE_STEPS, **kwargs):
    if lane == "run":
        kwargs["compute_s_params"] = False
    result = getattr(sim, lane)(n_steps=steps, skip_preflight=True, **kwargs)
    trace = np.asarray(result.time_series)[:, 0]
    assert np.max(np.abs(trace)) > 0, "probe is not excited"
    return trace, result


def _image_error(actual, reference):
    return float(np.max(np.abs(actual - reference)) / np.max(np.abs(reference)))


def test_plain_pmc_image_plane():
    """Establish the on-node mirror before using it for feature rows."""
    half, result = _solve(_cavity())
    image, doubled = _solve(_cavity(image=True))
    pec, _ = _solve(_cavity(wall="pec"))
    assert tuple(result.grid.shape) == (14, 11, 4)
    assert tuple(doubled.grid.shape) == (27, 11, 4)
    error = _image_error(half, image)
    separation = _image_error(half, pec)
    report = (f"plain on-node mirror: error={error:.12g}, bar=1e-5; "
              f"PEC separation={separation:.12g}, bar=1e-5; steps={IMAGE_STEPS}; "
              f"mesh={result.grid.shape}/{doubled.grid.shape}")
    print(report, file=sys.stderr)
    assert error <= 1e-5, report
    assert separation > 1e-5, report


def _lane_equal(actual, reference, label, poles=False):
    difference = np.abs(actual.astype(float) - reference.astype(float))
    peak = np.max(np.abs(reference))
    if poles:
        bound = 1.2e-7 * np.cumsum(np.abs(reference), dtype=float)
        ratio = float(np.max(difference / np.maximum(bound, np.finfo(float).tiny)))
        report = f"{label}: accumulated-law ratio={ratio:.12g}, bar=1"
        print(report, file=sys.stderr)
        assert np.all(difference <= bound), report
    else:
        ulps = float(np.max(difference) / np.spacing(np.float32(peak)))
        report = f"{label}: peak ULP={ulps:.12g}, bar=9"
        print(report, file=sys.stderr)
        assert ulps <= 9, report


def _translation_model(wall, copies=1):
    boundary = BoundarySpec(x=wall, y="pec", z="periodic")
    sim = Simulation(freq_max=30e9, domain=(copies * 13 * H, 10 * H, H),
                     dx=H, boundary=boundary, cpml_layers=0)
    component = "ez"
    for i in range(copies):
        sim.add_source(((2 + 13 * i) * H, 3 * H, 0.), component,
                       waveform=GaussianPulse(f0=16e9, bandwidth=1.2),
                       amplitude_kind="field")
    sim.add_probe((7 * H, 6 * H, 0.), component)
    return sim


@pytest.mark.parametrize("lane", ["run", "forward"])
def test_periodic_x_translation(lane):
    actual, _ = _solve(_translation_model("periodic"), lane)
    expected, _ = _solve(_translation_model("periodic", copies=2), lane)
    closed, _ = _solve(_translation_model("pec"), lane)
    _lane_equal(actual, expected, f"periodic translation/{lane}, steps={IMAGE_STEPS}")
    separation = _image_error(actual, closed)
    bar = 9 * float(np.spacing(np.float32(np.max(np.abs(closed))))) / np.max(np.abs(closed))
    report = f"periodic/{lane}: PEC separation={separation:.12g}, bar={bar:.12g}"
    print(report, file=sys.stderr)
    assert separation > bar, report


def _frequency(trace, result, wall, label, mode="3d"):
    shape = (14, 11, 4) if mode == "3d" else (14, 11, 1)
    # Stored Yee arrays include the high wall node. These lengths are
    # physical cell counts, not the number of stored rows.
    assert tuple(result.grid.shape) == shape
    a, b = CELLS[0] * H, CELLS[1] * H
    expected = C / 2 * np.sqrt((1 / ((2 if wall == "pmc" else 1) * a)) ** 2
                              + (1 / b) ** 2)
    if mode == "2d_tez":
        expected = C / ((4 if wall == "pmc" else 2) * a)
    modes = [m for m in matrix_pencil(trace[STEPS // 4:], float(result.grid.dt),
                                     .8 * expected, 1.2 * expected) if m.Q > 30]
    assert modes, f"{label}: no ringdown mode; expected={expected:.12g} Hz"
    measured = max(modes, key=lambda m: m.amplitude).freq
    relative = abs(measured / expected - 1)
    report = (f"{label}: expected={expected:.12g} Hz, measured={measured:.12g} Hz, "
              f"relative={relative:.12g}, bar=.01; steps={STEPS}; "
              f"mesh={result.grid.shape}")
    print(report, file=sys.stderr)
    assert relative <= .01, report
    return measured


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("feature", ["plain", "lumped", "wire"])
def test_pmc_cavity_frequency_and_forward(mesh, feature):
    run, result = _solve(_cavity(mesh, feature), steps=STEPS)
    forward, fresult = _solve(_cavity(mesh, feature), "forward", steps=STEPS)
    pec, presult = _solve(_cavity(mesh, feature, wall="pec"), steps=STEPS)
    _frequency(run, result, "pmc", f"{mesh}/{feature}/run")
    f_pmc = _frequency(forward, fresult, "pmc", f"{mesh}/{feature}/forward")
    f_pec = _frequency(pec, presult, "pec", f"{mesh}/{feature}/PEC control")
    separation = abs(f_pmc / f_pec - 1)
    report = f"{mesh}/{feature}: frequency separation={separation:.12g}, bar=.01"
    print(report, file=sys.stderr)
    assert separation > .01, report
    _lane_equal(forward, run, f"{mesh}/{feature}/forward-run")


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("feature", ["debye", "lorentz", "mixed"])
def test_pole_pmc_image_and_forward(mesh, feature):
    traces = {}
    for lane in ("run", "forward"):
        half, result = _solve(_cavity(mesh, feature), lane)
        image, doubled = _solve(_cavity(mesh, feature, image=True), lane)
        pec, _ = _solve(_cavity(mesh, feature, wall="pec"), lane)
        assert tuple(result.grid.shape) == (14, 11, 4)
        assert tuple(doubled.grid.shape) == (27, 11, 4)
        error = _image_error(half, image)
        separation = _image_error(half, pec)
        label = f"{mesh}/{feature}/{lane}"
        report = (f"{label}: image error={error:.12g}, bar=1e-5; "
                  f"PEC separation={separation:.12g}, bar=1e-5; "
                  f"steps={IMAGE_STEPS}; mesh={result.grid.shape}/{doubled.grid.shape}")
        print(report, file=sys.stderr)
        assert error <= 1e-5, report
        assert separation > 1e-5, report
        traces[lane] = half
    _lane_equal(traces["forward"], traces["run"], f"{mesh}/{feature}/forward-run", poles=True)


def _cavity_2d(mode, wall):
    sim = Simulation(freq_max=30e9, domain=(13 * H, 10 * H, H), dx=H,
                     mode=mode, cpml_layers=0,
                     boundary=BoundarySpec(x=Boundary(lo=wall, hi="pec"),
                                           y="pec", z="periodic"))
    # TEz's lowest mode has Ex=0; excite Ey and observe Hz off symmetry lines.
    source, probe = ("ey", "hz") if mode == "2d_tez" else ("ez", "ez")
    sim.add_source((2 * H, 3 * H, 0.), source,
                   waveform=GaussianPulse(f0=16e9, bandwidth=1.2),
                   amplitude_kind="field")
    sim.add_probe((7 * H, 6 * H, 0.), probe)
    return sim


@pytest.mark.parametrize("lane", ["run", "forward"])
@pytest.mark.parametrize("mode", ["2d_tmz", "2d_tez"])
def test_2d_pmc_cavity_frequency(mode, lane):
    pmc, result = _solve(_cavity_2d(mode, "pmc"), lane, steps=STEPS)
    pec, control = _solve(_cavity_2d(mode, "pec"), lane, steps=STEPS)
    f_pmc = _frequency(pmc, result, "pmc", f"{mode}/{lane}/PMC", mode)
    f_pec = _frequency(pec, control, "pec", f"{mode}/{lane}/PEC", mode)
    separation = abs(f_pmc / f_pec - 1)
    report = f"{mode}/{lane}: frequency separation={separation:.12g}, bar=.01"
    print(report, file=sys.stderr)
    assert separation > .01, report
