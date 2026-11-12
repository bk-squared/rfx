"""PMC image plane and cavity observables through the public solver paths.

The low magnetic wall is on E node zero: H[-1] = -H[0]. Its image
therefore has 26 physical x cells, with the shared E node at index 13.
Closed forms use the physical cell counts; every feature also uses the
public Simulation mirror model, independently of the curl implementation.
"""

import sys

import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, Simulation, Sphere, lorentz_pole
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.harminv import matrix_pencil

H = 1e-3
CELLS = (13, 10, 3)
C = 299792458.0
STEPS = 2400
IMAGE_STEPS = 300


def _cavity(mesh="uniform", feature="plain", wall="pmc", image=False, conformal=False,
            sphere_eps=1.):
    a, b, d = (n * H for n in CELLS)
    interior = np.linspace(.87, 1.13, CELLS[0] - 2)
    interior *= (a - 2 * H) / interior.sum()
    widths = np.r_[H, interior, H]
    options = {}
    if mesh == "graded":
        options["dx_profile"] = np.r_[widths[::-1], widths] if image else widths
    sim = Simulation(
        freq_max=30e9, domain=((2 if image else 1) * a, b, d), dx=H,
        boundary=BoundarySpec(x=Boundary(lo="pec" if image or feature == "conformal" else wall,
                                                hi="pec", conformal=conformal),
                              y=Boundary(lo=wall, hi="pec") if feature == "conformal" else "pec",
                              z="pec"), cpml_layers=0, **options)
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
        sim.add_port((shift + 3 * H, 4 * H, H), "ez", impedance=1e9,
                     excite=False, **extra)
        if image:
            # Reflect about the original x=0 E node (x=a after translation).
            # Ez is tangential: the image has the same sign and +z extent.
            sim.add_port((shift - 3 * H, 4 * H, H), "ez", impedance=1e9,
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
    if feature == "subpixel":
        sim.add_material("sphere", eps_r=sphere_eps)
        sim.add(Sphere(center=(shift + 4.3 * H, 2.2 * H, 1.4 * H),
                       radius=H), material="sphere")
        if image:
            sim.add(Sphere(center=(shift - 4.3 * H, 2.2 * H, 1.4 * H),
                           radius=H), material="sphere")
    if feature == "sheet":
        # An xy sheet includes the PMC E node and its tangential Ey edge.
        # Each free end extends 0.35 of the outside cell beyond node 8.
        end = float(np.sum(widths[:8]) + 0.35 * widths[8]) if mesh == "graded" else 8.35 * H
        # Rs0 = sqrt(pi * f0 * mu0 / sigma_bulk) = eta0 (A17).
        f0, mu0, eta0 = 12e9, 4 * np.pi * 1e-7, 376.730313
        sigma_bulk = np.pi * f0 * mu0 / eta0**2
        sim.add_thin_conductor(
            Box((shift - end if image else 0., 1.65 * H, H),
                (shift + end, 6.35 * H, H)),
            sigma_bulk=sigma_bulk, surface_impedance_f0=f0)
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


def _check_image(mesh, feature, lane, **kwargs):
    half, result = _solve(_cavity(mesh, feature), lane, **kwargs)
    image, doubled = _solve(_cavity(mesh, feature, image=True), lane, **kwargs)
    pec, _ = _solve(_cavity(mesh, feature, wall="pec"), lane, **kwargs)
    assert tuple(result.grid.shape) == (14, 11, 4)
    assert tuple(doubled.grid.shape) == (27, 11, 4)
    error = _image_error(half, image)
    separation = _image_error(half, pec)
    report = (f"{mesh}/{feature}/{lane}: image error={error:.12g}, bar=1e-5; "
              f"PEC separation={separation:.12g}, bar=1e-5; steps={IMAGE_STEPS}; "
              f"mesh={result.grid.shape}/{doubled.grid.shape}")
    print(report, file=sys.stderr)
    assert error <= 1e-5, report
    assert separation > 1e-5, report
    return half


@pytest.mark.parametrize("lane", ["run"])
@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_plain_pmc_image_plane(mesh, lane):
    """Establish the on-node mirror on every plain solver path."""
    _check_image(mesh, "plain", lane)


@pytest.mark.parametrize("lane", ["run"])
@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("feature", ["lumped", "wire"])
def test_port_pmc_image_plane(mesh, feature, lane):
    """Both passive image ports retain the original Ez orientation."""
    _check_image(mesh, feature, lane)


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
        traces[lane] = _check_image(mesh, feature, lane)
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
def test_2d_tmz_preserves_polarization(lane):
    """A z-invariant Ez drive cannot generate Ex or Ey in isotropic vacuum.

    The common six-component step still evaluates dHy/dz and dHx/dz for
    Ex/Ey. A zero-padded z ghost creates cross-polarized E even when the
    existing Ez cavity record is unchanged. Read all three E components
    off the symmetry planes and compare with the zero transverse trace.
    """
    sim = _cavity_2d("2d_tmz", "pmc")
    for component in ("ex", "ey"):
        sim.add_probe((7 * H, 6 * H, 0.), component)
    ez, result = _solve(sim, lane)
    assert tuple(result.grid.shape) == (14, 11, 1)
    actual = np.asarray(result.time_series)
    expected = np.zeros_like(actual)
    expected[:, 0] = ez
    _lane_equal(actual, expected,
                f"2d_tmz/{lane}/polarization, steps={IMAGE_STEPS}, mesh={result.grid.shape}")


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


def test_conformal_node_aligned_pec_matches_plain():
    actual, result = _solve(_cavity(feature="conformal", conformal=True))
    expected, reference = _solve(_cavity(feature="conformal"))
    assert tuple(result.grid.shape) == tuple(reference.grid.shape) == (14, 11, 4)
    _lane_equal(actual, expected, f"conformal/plain, steps={IMAGE_STEPS}, mesh={result.grid.shape}")
    pec, _ = _solve(_cavity(feature="conformal", wall="pec", conformal=True))
    separation = _image_error(actual, pec)
    bar = 9 * float(np.spacing(np.float32(np.max(np.abs(pec))))) / np.max(np.abs(pec))
    report = f"conformal: y_lo PMC/PEC separation={separation:.12g}, bar={bar:.12g}"
    print(report, file=sys.stderr)
    assert separation > bar, report


@pytest.mark.parametrize("feature,mesh,sphere_eps", [("subpixel", "uniform", 1.),
                                                     ("subpixel", "graded", 1.),
                                                     ("conformal", "uniform", 1.),
                                                     ("subpixel", "uniform", 4.),
                                                     ("subpixel", "graded", 4.)])
def test_smoothed_checkpoint_matches_plain_scan(feature, mesh, sphere_eps):
    conformal = feature == "conformal"
    dielectric = sphere_eps == 4.
    steps = IMAGE_STEPS if dielectric else STEPS
    options = {"subpixel_smoothing": True} if not conformal else {}
    traces = {}
    for checkpoint in (False, True):
        sim = _cavity(mesh, feature, conformal=conformal, sphere_eps=sphere_eps)
        trace, result = _solve(sim, steps=steps, checkpoint=checkpoint, **options)
        traces[checkpoint] = trace
        if not conformal and not dielectric:
            _frequency(trace, result, "pmc", f"{mesh}/subpixel/checkpoint={checkpoint}")
    assert tuple(result.grid.shape) == (14, 11, 4)
    _lane_equal(traces[True], traces[False],
                f"{mesh}/{feature}/eps={sphere_eps:g}/checkpoint, steps={steps}, mesh={result.grid.shape}")
    if dielectric:
        unsmoothed, _ = _solve(_cavity(mesh, feature, sphere_eps=sphere_eps),
                               steps=steps, checkpoint=False, subpixel_smoothing=False)
        separation = float(np.max(np.abs(traces[False].astype(float) - unsmoothed.astype(float)))
                           / np.spacing(np.float32(np.max(np.abs(unsmoothed)))))
        report = (f"{mesh}/subpixel/eps={sphere_eps:g}: smoothing on/off separation="
                  f"{separation:.12g} peak ULP, bar=9; steps={steps}; mesh={result.grid.shape}")
        print(report, file=sys.stderr)
        assert separation > 9, report
    elif conformal:
        pec, _ = _solve(_cavity(feature="conformal", wall="pec", conformal=True),
                        steps=STEPS, checkpoint=True)
        separation = _image_error(traces[True], pec)
        bar = 9 * float(np.spacing(np.float32(np.max(np.abs(pec))))) / np.max(np.abs(pec))
        report = f"conformal/checkpoint: y_lo PMC/PEC separation={separation:.12g}, bar={bar:.12g}"
        print(report, file=sys.stderr)
        assert separation > bar, report
    else:
        # Image comparisons pin the PMC ghost even when its frequency shift
        # fits inside the closed-form bar. Exercise both scan variants.
        for checkpoint in (False, True):
            _check_image(mesh, feature, "run", checkpoint=checkpoint, **options)


@pytest.mark.parametrize("lane", ["run"])
@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_sheet_pmc_image(mesh, lane):
    _check_image(mesh, "sheet", lane)


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_sheet_pmc_forward_matches_run(mesh):
    traces = {}
    for lane in ("run", "forward"):
        traces[lane], _ = _solve(_cavity(mesh, "sheet"), lane)
        bare, _ = _solve(_cavity(mesh), lane)
        separation = _image_error(traces[lane], bare)
        bar = 9 * float(np.spacing(np.float32(np.max(np.abs(bare))))) / np.max(np.abs(bare))
        report = f"{mesh}/sheet/{lane}: sheet/no-sheet separation={separation:.12g}, bar={bar:.12g}"
        print(report, file=sys.stderr)
        assert separation > bar, report
    _lane_equal(traces["forward"], traces["run"], f"{mesh}/sheet/forward-run, steps={IMAGE_STEPS}")
