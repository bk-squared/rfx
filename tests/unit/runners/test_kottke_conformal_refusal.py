"""Every public entry either carries the PEC choice or refuses before stepping."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rfx import Cylinder, Simulation, Sphere
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.runners import _admission as A


def _scene(entry, source, *, pec=True, dt=None):
    if entry.startswith("waveguide"):
        # Simulation/boundary/port setup from the validation battery's _build_sim.
        boundary = {"x": "cpml", "y": "pec", "z": "pec"}
        if source == "B":
            boundary["y"] = Boundary("pec", "pec", conformal=True)
        sim = Simulation(freq_max=9e9, domain=(.12, .04, .02), dx=.002,
                         boundary=boundary, cpml_layers=10, dt=dt)
        if pec:
            sim.add(Cylinder((.0601, .0198, .0097), .0033, .008, "z"),
                    material="pec")
        for x, direction, name in ((.01, "+x", "left"), (.09, "-x", "right")):
            sim.add_waveguide_port(
                x, direction=direction, mode=(1, 0), mode_type="TE",
                freqs=np.array([7e9, 8e9, 9e9]), f0=8e9, bandwidth=.25,
                waveform="modulated_gaussian", n_modes=1, name=name)
        return sim
    two_d = entry == "run_2d"
    boundary = (BoundarySpec(x="pec", y=Boundary("pec", "pec", conformal=True),
                             z="pec") if source == "B" else "pec")
    sim = Simulation(
        freq_max=20e9, domain=(.012,) * 3, dx=.001, boundary=boundary,
        cpml_layers=0, mode="2d_tmz" if two_d else "3d",
        **({"dz_profile": np.full(12, .001)}
           if entry in ("graded", "distributed_nu") else {}))
    if pec:
        shape = (Cylinder((.0061, .0058, 0.), .0017, .012, "z") if two_d
                 else Sphere((.0061, .0058, .0057), .0017))
        sim.add(shape, material="pec")
    if entry == "subgridded":
        sim.add_refinement((.008, .012), ratio=2, validation="research")
    if entry in ("lumped", "lumped_scan"):
        sim.add_port((.003, .003, .003), "ez", impedance=50.)
        sim.add_port((.009, .009, .009), "ez", impedance=50.)
    else:
        sim.add_source((.003, .004, 0. if two_d else .003), "ez",
                       amplitude_kind="current")
        sim.add_probe((.008, .004, 0. if two_d else .003), "ez")
    return sim


def _invoke(entry, source, *, pec=True, smoothing="kottke_pec", conformal=True):
    sim = _scene(entry, source, pec=pec)
    options = dict(n_steps=1, subpixel_smoothing=smoothing)
    if source == "K":
        options["conformal_pec"] = conformal
    if entry in ("msl", "mixed"):
        return getattr(sim, f"compute_{entry}_s_matrix")(**options)
    if entry == "lumped_scan":
        from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
        return compute_lumped_wire_s_matrix_via_scan(sim, np.array([10e9]), **options)
    options["skip_preflight"] = True
    if entry.startswith("waveguide"):
        return sim.compute_waveguide_s_matrix(
            normalize=entry == "waveguide_normalized", **options)
    if entry.startswith("distributed"):
        devices = jax.devices("cpu")
        assert len(devices) >= 2
        options["devices"] = devices[:2]
    if entry == "lumped":
        options.update(compute_s_params=True, s_param_freqs=np.array([10e9]))
    return getattr(sim, "forward" if entry == "forward" else "run")(**options)


def _no_step(*args, **kwargs):
    pytest.fail("refused combination reached tensor formation or time stepping")


REFUSED = [
    (entry, source)
    for entry in ("run_3d", "run_2d", "forward", "graded", "distributed_v2",
                  "distributed_nu", "subgridded", "waveguide",
                  "waveguide_normalized", "lumped")
    for source in ("K", "B")
    if not (source == "K" and entry == "forward")
]


@pytest.mark.parametrize("entry,source", REFUSED, ids=[f"{e}-{s}" for e, s in REFUSED])
def test_combination_refused_before_step(monkeypatch, entry, source):
    from rfx import simulation
    from rfx.model import tensor_operands
    from rfx.runners import nonuniform
    from rfx.stepping import uniform_loop

    monkeypatch.setattr(tensor_operands, "form_tensor_operands", _no_step)
    # The uniform runner reads run from rfx.simulation, the waveguide S extractors from uniform_loop.
    monkeypatch.setattr(simulation, "run", _no_step)
    monkeypatch.setattr(uniform_loop, "run", _no_step)
    monkeypatch.setattr(nonuniform, "run_nonuniform", _no_step)
    # The graded lane already rejects conformal_pec before considering smoothing.
    # All other reachable rows name both inputs; no implementation text is imported.
    match = (r"conformal_pec" if entry == "graded" else
             r"(?s)(?=.*subpixel_smoothing)(?=.*kottke_pec)(?=.*conformal_pec)")
    with pytest.raises(NotImplementedError, match=match):
        _invoke(entry, source)


UNREACHABLE = [
    ("forward", "K", "conformal_pec"),
    *[(entry, source, "subpixel_smoothing")
      for entry in ("msl", "mixed") for source in ("K", "B")],
]


@pytest.mark.parametrize("entry,source,keyword", UNREACHABLE,
                         ids=[f"{e}-{s}" for e, s, _ in UNREACHABLE])
def test_keyword_not_accepted(entry, source, keyword):
    with pytest.raises(TypeError, match=rf"unexpected keyword.*{keyword}"):
        _invoke(entry, source)


@pytest.mark.parametrize("source", ["K", "B"])
def test_lumped_scan_refuses_conformal_once(source):
    with pytest.raises(NotImplementedError,
                       match="lumped/wire S-matrix scan does not carry") as caught:
        _invoke("lumped_scan", source)
    assert str(caught.value).count("Boundary(conformal=True) or conformal_pec=True") == 1


CONFORMAL_ROWS = (("_boundary_spec", "conformal"),
                  ("_boundary_spec", "conformal_s_matrix"))


@pytest.mark.parametrize("lane", [
    lane for lane in A.CALCULATORS
    if all(row not in A.ADMITS[lane] for row in CONFORMAL_ROWS)
])
def test_calculator_message_lists_conformal_once(lane):
    text = A.message(lane, CONFORMAL_ROWS, SimpleNamespace(_precision="float32"))
    assert text.count("Boundary(conformal=True) or conformal_pec=True") == 1
    assert "with a lumped/wire S-matrix" in text


@pytest.mark.parametrize("rows", [CONFORMAL_ROWS, CONFORMAL_ROWS[:1], CONFORMAL_ROWS[1:]],
                         ids=["both", "conformal-only", "s-matrix-only"])
def test_run_message_lists_conformal_once(monkeypatch, rows):
    monkeypatch.setattr(A, "refused", lambda *args: rows)
    text = A.message("fwd_uniform", rows, SimpleNamespace(_refinement=None))
    assert text.count("Boundary(conformal=True) or conformal_pec=True") == 1
    assert ("with a lumped/wire S-matrix" in text) == (CONFORMAL_ROWS[1] in rows)


@pytest.mark.parametrize("entry", ["run_3d", "run_2d"])
@pytest.mark.parametrize("smoothing,source,conformal,pec", [
    pytest.param("kottke_pec", "K", False, True, id="kottke-alone"),
    pytest.param(False, "K", True, True, id="conformal-keyword"),
    pytest.param(True, "K", True, True, id="conformal-stage1"),
    pytest.param(False, "B", True, True, id="conformal-boundary"),
    pytest.param("kottke_pec", "K", True, False, id="no-pec-no-weights"),
])
def test_supported_controls_run(entry, smoothing, source, conformal, pec):
    result = _invoke(entry, source, pec=pec, smoothing=smoothing, conformal=conformal)
    assert result.time_series.shape[0] == 1
    assert np.isfinite(np.asarray(result.time_series)).all()


@pytest.mark.parametrize("entry", ["run_3d", "run_2d", "waveguide"])
def test_conformal_faces_still_refuse_without_interior_pec(entry):
    # Conformal boundary faces themselves produce PEC shapes/weights.
    with pytest.raises(NotImplementedError,
                       match=r"(?s)(?=.*subpixel_smoothing)(?=.*conformal_pec)"):
        _invoke(entry, "B", pec=False)


@pytest.mark.parametrize("smoothing,source", [
    pytest.param("kottke_pec", "none", id="kottke-alone"),
    pytest.param(False, "B", id="conformal-alone"),
    pytest.param(True, "B", id="conformal-stage1"),
])
def test_waveguide_controls_run(smoothing, source):
    result = _invoke("waveguide", source, smoothing=smoothing)
    assert result.s_params.shape == (2, 2, 3)
    assert np.isfinite(np.asarray(result.s_params)).all()


def test_cut_cell_floor_admission_paths():
    from rfx.grid import Grid
    from rfx.interop.emitters.openems import plan_openems_projection
    from rfx.interop import design_to_dict
    ordinary = Grid.courant_dt(.001)
    # No PEC shapes: the opt-in itself must refuse, never disappear as a no-op.
    for entry in ('run', 'forward', 'graded', 'distributed', 'subgridded', 'adi', 'kottke'):
        kwargs = dict(freq_max=20e9, domain=(.012,) * 3, dx=.001, boundary='pec')
        if entry != 'run':
            kwargs['dt'] = .5 * ordinary
        if entry == 'graded':
            kwargs['dz_profile'] = np.full(12, .001)
        if entry == 'adi':
            kwargs['solver'] = 'adi'
        sim = Simulation(**kwargs)
        options = dict(n_steps=1, skip_preflight=True, conformal_pec='cut_cell_floor')
        if entry == 'distributed':
            options['devices'] = jax.devices('cpu')[:2]
        if entry == 'subgridded':
            sim.add_refinement((.008, .012), ratio=2, validation='research')
        if entry == 'kottke':
            options['subpixel_smoothing'] = 'kottke_pec'
        with pytest.raises(NotImplementedError, match='cut_cell_floor'):
            getattr(sim, 'forward' if entry == 'forward' else 'run')(**options)
    for entry in ('run', 'forward'):
        sim = Simulation(20e9, (.012,) * 3, dx=.001, boundary='pec', dt=.5 * ordinary)
        with pytest.raises(NotImplementedError, match='cut_cell_floor'):
            getattr(sim, entry)(n_steps=1, skip_preflight=True)
    sim = Simulation(20e9, (.012,) * 3, dx=.001, dt=.4 * ordinary)
    with pytest.raises(ValueError) as caught:
        sim.run(n_steps=1, conformal_pec='cut_cell_floor', skip_preflight=True)
    assert repr(.5 * ordinary) in str(caught.value)
    assert repr(.25 * ordinary) in str(caught.value)
    with pytest.raises(ValueError, match='cut_cell_floor'):
        Boundary('pec', 'pec', conformal='cut_cell_floor')
    sim = Simulation(20e9, (.012,) * 3, dx=.001, dt=.5 * ordinary)
    from rfx.interop import UnsupportedDesignFeature
    with pytest.raises(UnsupportedDesignFeature, match='cut_cell_floor'):
        plan_openems_projection(design_to_dict(sim))


def test_cut_cell_floor_grid_tolerance_and_reference(monkeypatch):
    from rfx.grid import Grid
    from rfx.geometry.conformal_cells import compute_conformal_fractions
    from rfx.model.tensor_operands import form_reference_operands
    from rfx.model.materials import with_components
    from rfx.core.yee import init_materials
    from rfx import Box
    for mode in ('3d', '2d_tmz', '2d_tez'):
        ordinary = .001 / (299792458. * np.sqrt(3. if mode == '3d' else 2.)) * .99
        for q in (.5, .25):
            pin = q * ordinary * (1 + .5e-9)
            sim = Simulation(20e9, (.012,) * 3, dx=.001, boundary='pec', mode=mode, dt=pin)
            grid = sim._build_grid()
            assert grid.dt == pin
            assert grid.num_timesteps(2) == int(np.ceil(2 / 20e9 / pin))
            bad = Simulation(20e9, (.012,) * 3, dx=.001, boundary='pec', mode=mode,
                             dt=q * ordinary * (1 + 2e-9))
            with pytest.raises(ValueError, match='cut_cell_floor'):
                bad._build_grid()
    grid = Grid(20e9, (.012,) * 3, dx=.001, cpml_layers=0)
    grid.dt = .5 * Grid.courant_dt(.001)
    wall = Box((-1.,) * 3, (.0029, 1., 1.))
    weights, _ = compute_conformal_fractions(grid, [wall], clamp=False)
    mats = with_components(init_materials(grid.shape), grid, periodic=(False,) * 3)
    eps, _, reference = form_reference_operands(
        grid, mats, subpixel_smoothing=False, conformal_weights=weights,
        boundary_pec_shapes=[wall], conformal_pec='cut_cell_floor')
    assert float(eps[0][2, 5, 5]) == pytest.approx(10.)
    for normal in (1, 2):
        assert float(reference[1][normal][2, 5, 5]) == .25
    from rfx.model import tensor_operands
    captured = []
    original = tensor_operands.form_reference_operands
    def capture(*args, **kwargs):
        assert kwargs['conformal_pec'] == 'cut_cell_floor'
        captured.append(float(args[0].dt))
        return original(*args, **kwargs)
    monkeypatch.setattr(tensor_operands, 'form_reference_operands', capture)
    for q in (.5, .25):
        sim = _scene('waveguide', 'B', dt=q * Grid.courant_dt(.002))
        result = sim.compute_waveguide_s_matrix(
            n_steps=4, normalize=True, conformal_pec='cut_cell_floor', skip_preflight=True)
        assert np.isfinite(np.asarray(result.s_params)).all()
        assert captured[-1] == q * Grid.courant_dt(.002)
    assert len(captured) == 2
