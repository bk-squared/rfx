"""Discrete source invariants; these are algebra gates, not mesh accuracy claims."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.core.yee import EPS_0, MU_0, init_state
from rfx.sources.sources import CustomWaveform, GaussianPulse
from rfx.stepping.tfsf import (
    apply_tfsf_e,
    apply_tfsf_h,
    update_tfsf_1d,
)
from rfx.sources.tfsf import (
    init_tfsf,
    tfsf_injection_planes,
    update_tfsf_1d_e,
)
from rfx.measurement.incident import (
    measure_normal_incident_spectrum,
)


def _peak_ulp(actual, expected, budget=9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    peak = np.float32(np.max(np.abs(expected)))
    assert np.max(np.abs(actual - expected)) <= budget * np.spacing(peak)


def _curl(fields, forward):
    """Independent full-array curl (test source commutator only)."""
    def diff(a, axis):
        if forward:
            shifted = np.roll(a, -1, axis=axis)
            idx = [slice(None)] * 3
            idx[axis] = -1
            shifted[tuple(idx)] = 0
            return shifted - a
        shifted = np.roll(a, 1, axis=axis)
        idx = [slice(None)] * 3
        idx[axis] = 0
        shifted[tuple(idx)] = 0
        return a - shifted
    x, y, z = fields
    return np.stack((diff(z, 1) - diff(y, 2),
                     diff(x, 2) - diff(z, 0),
                     diff(y, 0) - diff(x, 1)))


@pytest.mark.parametrize("polarization,direction", list(itertools.product(("ez", "ey"), ("+x", "-x"))))
def test_six_face_corrections_equal_mask_curl_commutator(polarization, direction):
    shape = (17, 16, 15)
    dx, dt = 0.001, 1e-12
    cfg, aux = init_tfsf(shape[0], dx, dt, ny=shape[1], nz=shape[2],
                         cpml_layers=2, tfsf_margin=3, closed_box=True,
                         polarization=polarization, direction=direction, n_steps=32)
    rng = np.random.default_rng(891)
    ec, hc = polarization, "hy" if polarization == "ez" else "hz"
    e1d = rng.normal(size=getattr(aux, ec).shape).astype(np.float32)
    h1d = rng.normal(size=getattr(aux, hc).shape).astype(np.float32)
    aux = aux._replace(**{ec: jnp.asarray(e1d), hc: jnp.asarray(h1d)})
    mask = np.zeros(shape)
    mask[cfg.x_lo:cfg.x_hi + 1, cfg.y_lo:cfg.y_hi + 1,
         cfg.z_lo:cfg.z_hi + 1] = 1
    # Physical E(x), H(x+1/2) map to signed projected sublattices.
    ix = cfg.m[0] * np.arange(shape[0]) - cfg.origin
    ih = np.floor(cfg.m[0] * (np.arange(shape[0]) + .5)).astype(int) - cfg.origin
    e = np.zeros((3,) + shape)
    h = np.zeros_like(e)
    e["xyz".index(ec[-1])] = e1d[ix, None, None]
    h["xyz".index(hc[-1])] = h1d[ih, None, None]
    want_h = -(dt / (MU_0 * dx)) * (mask * _curl(e, True) - _curl(mask * e, True))
    want_e = (dt / (EPS_0 * dx)) * (mask * _curl(h, False) - _curl(mask * h, False))
    state = init_state(shape)
    got_h = apply_tfsf_h(state, cfg, aux, dx, dt)
    got_e = apply_tfsf_e(state, cfg, aux, dx, dt)
    _peak_ulp(np.stack([getattr(got_h, c) for c in ("hx", "hy", "hz")]), want_h)
    _peak_ulp(np.stack([getattr(got_e, c) for c in ("ex", "ey", "ez")]), want_e)
    assert set(tfsf_injection_planes(cfg)) == set("xyz")


def _sim(*, polarization="ez", direction="+x", closed_box=True, waveform=None):
    sim = Simulation(freq_max=30e9, domain=(0.018, 0.016, 0.014), dx=0.001,
                     boundary="cpml", cpml_layers=4)
    if waveform is None:
        dt = sim._build_grid().dt
        # Both quadratures, a finite record and no reliance on Gaussian metadata.
        u = jnp.arange(72, dtype=jnp.float32)
        samples = jnp.sin(jnp.pi * u / 71) ** 2 * (
            0.7 * jnp.cos(0.31 * u) - 0.4 * jnp.sin(0.31 * u))
        def drive(t):
            i = jnp.rint(t / dt).astype(jnp.int32)
            return jnp.where((i >= 0) & (i < len(samples)),
                             samples[jnp.clip(i, 0, len(samples) - 1)], 0.0)
        waveform = CustomWaveform(drive)
    sim.add_tfsf_source(f0=15e9, bandwidth=0.8, polarization=polarization,
                        direction=direction, margin=3,
                        waveform=waveform, closed_box=closed_box)
    return sim


def _aux(sim):
    grid = sim._build_grid()
    entry = sim._tfsf
    return init_tfsf(grid.nx, grid.dx, grid.dt, ny=grid.ny, nz=grid.nz,
                     cpml_layers=grid.cpml_layers, tfsf_margin=entry.margin,
                     polarization=entry.polarization, direction=entry.direction,
                     f0=entry.f0, bandwidth=entry.bandwidth,
                     waveform=entry.waveform, closed_box=entry.closed_box, n_steps=160)


@pytest.mark.parametrize("polarization,direction", list(itertools.product(("ez", "ey"), ("+x", "-x"))))
def test_closed_vacuum_replays_real_waveform_inside_and_cancels_all_faces(polarization, direction):
    sim = _sim(polarization=polarization, direction=direction)
    grid = sim._build_grid()
    cfg, aux = _aux(sim)
    bounds = sim.tfsf_box_indices()
    assert bounds == {a: (getattr(cfg, a + "_lo"), getattr(cfg, a + "_hi")) for a in "xyz"}
    # Two inside nodes, each outside face, and all eight exterior corners.
    centre = tuple((lo + hi) // 2 for lo, hi in bounds.values())
    points = [centre, (cfg.x_lo, cfg.y_lo, cfg.z_lo)]
    for axis, (lo, hi) in enumerate(bounds.values()):
        for idx in (lo - 1, hi + 1):
            p = list(centre)
            p[axis] = idx
            points.append(tuple(p))
    points.extend(itertools.product(*[(lo - 1, hi + 1) for lo, hi in bounds.values()]))
    for p in points:
        pos = tuple((i - grid.cpml_layers) * grid.dx for i in p)
        assert grid.position_to_index(pos) == p
        sim.add_probe(pos, component=polarization)
    result = sim.run(n_steps=128)
    trace = np.asarray(result.time_series)
    indices = np.array([cfg.m[0] * p[0] - cfg.origin for p in points[:2]])
    def step(st, n):
        st = update_tfsf_1d(cfg, st, grid.dx, grid.dt, n * grid.dt)
        return st, getattr(st, polarization)[indices]
    _, reference = jax.lax.scan(step, aux, jnp.arange(128))
    _peak_ulp(trace[:, :2], reference)
    peak = np.float32(np.max(np.abs(reference)))
    assert peak > 0.1
    assert np.max(np.abs(trace[:, 2:])) <= 9 * np.spacing(peak)
    freqs = np.linspace(8e9, 22e9, 7)
    times = (np.arange(128) + 1) * grid.dt
    spectrum = np.exp(-2j * np.pi * freqs[:, None] * times) @ trace[:, 0] * grid.dt
    expected = measure_normal_incident_spectrum(cfg, aux, 128, freqs, grid.dt,
                                                reference_index=centre[0])
    assert np.max(np.abs(spectrum - expected)) <= 1e-4 * np.max(np.abs(expected))


@pytest.mark.xfail(strict=True, reason=(
    "issue 1668: the built-in TF/SF incident is an inline copy of the old "
    "formula until absorber routes it to GaussianPulse"))
@pytest.mark.parametrize("polarization,direction", [("ez", "+x"), ("ey", "-x")])
def test_custom_gaussian_preserves_legacy_slab(polarization, direction):
    pulse = GaussianPulse(f0=15e9, bandwidth=0.8)
    records = []
    for waveform in ("differentiated_gaussian", CustomWaveform(pulse)):
        sim = _sim(polarization=polarization, direction=direction,
                   closed_box=False, waveform=waveform)
        sim.add_probe((0.009, 0.008, 0.007), component=polarization)
        records.append(sim.run(n_steps=128))
        assert set(sim.tfsf_box_indices()) == {"x"}
    _peak_ulp(records[1].time_series, records[0].time_series)
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        a, b = (np.asarray(getattr(r.state, component)) for r in records)
        if np.any(a):
            _peak_ulp(b, a)
        else:
            np.testing.assert_array_equal(b, a)


@pytest.mark.parametrize("angle", (0.0, 20.0, 40.0))
def test_methodb_auxiliary_keeps_its_original_waveform(angle):
    from rfx.sources.tfsf_oblique_open import init_tfsf_methodB
    cfg, aux = init_tfsf_methodB(33, 31, 0.001, 1e-12, nz=3,
                                cpml_layers=3, tfsf_margin=3, theta_deg=angle,
                                waveform="modulated_gaussian")
    # Method B shares this E kernel, but has no new optional source fields.
    # From zero fields its Gaussian peak is exactly the source amplitude.
    out = update_tfsf_1d_e(cfg, aux, 0.001, 1e-12, cfg.src_t0)
    expected = np.zeros_like(out.e1d)
    expected[cfg.src_idx] = cfg.src_amp
    np.testing.assert_array_equal(out.e1d, expected)


@pytest.mark.parametrize("entry", ("run", "forward"))
@pytest.mark.parametrize("waveform", [
    CustomWaveform(lambda t: jnp.asarray(1j)),
    CustomWaveform(lambda t: jnp.ones(2)),
    CustomWaveform(lambda t: jnp.where(t > 3e-12, jnp.nan, 0.0)),
])
def test_invalid_custom_samples_refused_before_time_stepping(entry, waveform):
    sim = _sim(waveform=waveform)
    with pytest.raises(ValueError, match="CustomWaveform must return"):
        getattr(sim, entry)(n_steps=8, skip_preflight=True)


@pytest.mark.parametrize("entry", ("run", "forward"))
@pytest.mark.parametrize("kind", ("custom", "box"))
@pytest.mark.parametrize("lane", ("nonuniform", "distributed", "distributed_nu", "subgridded"))
def test_new_input_is_never_dropped_on_unsupported_lanes(entry, kind, lane):
    params = dict(freq_max=30e9, domain=(0.018, 0.016, 0.014), dx=0.001,
                  boundary="cpml", cpml_layers=4)
    if lane in ("nonuniform", "distributed_nu"):
        params["dz_profile"] = [0.001] * 14
    sim = Simulation(**params)
    if lane == "distributed" and kind == "box":
        devices = jax.devices("cpu")[:2]
        if len(devices) < 2:
            pytest.skip("requires two host devices")
        sim.add_tfsf_source(closed_box=True)
        options = dict(devices=devices)
        if entry == "forward":
            options.update(distributed=True, checkpoint=False)
        getattr(sim, entry)(n_steps=2, skip_preflight=True, **options)
        return
    with pytest.raises(NotImplementedError, match=(
            "open-slab" if lane == "distributed" else "CustomWaveform and closed_box|graded multi-device")):
        sim.add_tfsf_source(closed_box=kind == "box", waveform=(
            CustomWaveform(lambda t: jnp.sin(t * 1e11)) if kind == "custom"
            else "differentiated_gaussian"))
        if lane == "subgridded":
            sim.add_refinement(z_range=(0.005, 0.009), ratio=2)
        kwargs = dict(n_steps=2, skip_preflight=True)
        if lane in ("distributed", "distributed_nu"):
            if entry == "run":
                kwargs["devices"] = jax.devices("cpu")[:2]
            else:
                kwargs.update(distributed=True, devices=jax.devices("cpu")[:2])
        getattr(sim, entry)(**kwargs)


@pytest.mark.parametrize("angle", (1e-4, 25.0))
@pytest.mark.parametrize("kind", ("custom", "box"))
def test_new_inputs_refuse_every_nonzero_angle(angle, kind):
    sim = Simulation(freq_max=30e9, domain=(0.018, 0.016, 0.014), dx=0.001,
                     boundary="cpml", cpml_layers=4)
    with pytest.raises(NotImplementedError, match="normal incidence"):
        sim.add_tfsf_source(angle_deg=angle, closed_box=kind == "box", waveform=(
            CustomWaveform(lambda t: t) if kind == "custom" else "differentiated_gaussian"))


def test_custom_source_refuses_decay_without_source_off_contract():
    sim = _sim()
    with pytest.raises(NotImplementedError, match="source-off time"):
        sim.run(until_decay=1e-4, n_steps=16, skip_preflight=True)


@pytest.mark.parametrize("runner", ("nonuniform", "distributed", "subgridded"))
def test_direct_runner_calls_cannot_bypass_source_admission(runner):
    sim = _sim()
    if runner == "distributed":
        from rfx.runners.distributed_v2 import run_distributed
        devices = jax.devices("cpu")[:2]
        if len(devices) < 2:
            pytest.skip("requires two host devices")
        result = run_distributed(sim, n_steps=2, devices=devices)
        assert np.isfinite(np.asarray(result.state.ez)).all()
        return
    with pytest.raises(NotImplementedError, match="CustomWaveform and closed_box"):
        if runner == "nonuniform":
            from rfx.runners.nonuniform import run_nonuniform_path
            run_nonuniform_path(sim, n_steps=2)
        else:
            from rfx.runners.subgridded import run_subgridded_path
            run_subgridded_path(sim, None, None, None, 2)


@pytest.mark.parametrize("entry", ("run", "forward"))
@pytest.mark.parametrize("material", ("dielectric", "pec"))
@pytest.mark.parametrize("axis", (0, 1, 2))
def test_every_box_face_requires_vacuum(entry, material, axis):
    from rfx.geometry.csg import Box
    sim = _sim()
    lo = [0.008, 0.007, 0.006]
    hi = [0.010, 0.009, 0.008]
    # The low total-field node sits at 3 mm from the inner CPML edge.
    lo[axis], hi[axis] = 0.002, 0.004
    if material == "dielectric":
        sim.add_material("target", eps_r=2.0)
        name = "target"
    else:
        name = "pec"
    sim.add(Box(tuple(lo), tuple(hi)), material=name)
    with pytest.raises(ValueError, match="vacuum.*all six faces"):
        getattr(sim, entry)(n_steps=4, skip_preflight=True)


@pytest.mark.parametrize("entry", ("run", "forward"))
@pytest.mark.parametrize("operator", ("surface_sheet", "inductor", "series_rl"))
@pytest.mark.parametrize("on_face", (False, True))
def test_nonvolume_operator_footprint_cannot_cross_source(entry, operator, on_face):
    from rfx.geometry.csg import Box
    sim = _sim()
    x = 0.003 if on_face else 0.009
    if operator == "surface_sheet":
        sim.add_thin_conductor(Box((x, 0.006, 0.005), (x, 0.010, 0.009)),
                               surface_impedance_f0=15e9)
    else:
        sim.add_lumped_rlc((x, 0.008, 0.007), component="ez", L=1e-9,
                           R=50.0 if operator == "series_rl" else 0.0,
                           topology="series" if operator == "series_rl" else "parallel")
    if on_face:
        with pytest.raises(ValueError, match="vacuum.*all six faces"):
            getattr(sim, entry)(n_steps=2, skip_preflight=True)
    else:
        getattr(sim, entry)(n_steps=2, skip_preflight=True)


@pytest.mark.parametrize("kind", ("eps", "occupancy"))
@pytest.mark.parametrize("form", ("whole_grid", "design_box"))
@pytest.mark.parametrize("on_face", (False, True))
def test_forward_design_inputs_cannot_cross_source(kind, form, on_face):
    sim = _sim()
    grid = sim._build_grid()
    x = 0.003 if on_face else 0.008
    corners = ((x, 0.006, 0.005), (x + 0.001, 0.009, 0.008))
    lo, hi = map(grid.position_to_index, corners)
    shape = tuple(b - a + 1 for a, b in zip(lo, hi))
    values = jnp.full(shape, 2.0 if kind == "eps" else 0.5)
    if form == "whole_grid":
        window = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
        value = jnp.full(grid.shape, 1.0 if kind == "eps" else 0.0).at[window].set(values)
        kwargs = {"eps_override" if kind == "eps" else "pec_occupancy_override": value}
    else:
        kwargs = {"design_box": corners, "design_" + kind + "_override": values}
    if on_face:
        with pytest.raises(ValueError, match="vacuum.*all six faces"):
            sim.forward(n_steps=2, skip_preflight=True, **kwargs)
    else:
        sim.forward(n_steps=2, skip_preflight=True, **kwargs)


@pytest.mark.parametrize("on_face", (False, True))
def test_conformal_pec_operator_cannot_cross_source(on_face):
    from rfx.geometry.csg import Box
    sim = _sim()
    x = 0.002 if on_face else 0.008
    sim.add(Box((x, 0.006, 0.005), (x + 0.002, 0.009, 0.008)), material="pec")
    if on_face:
        with pytest.raises(ValueError, match="vacuum.*all six faces"):
            sim.run(n_steps=2, conformal_pec=True, skip_preflight=True)
    else:
        sim.run(n_steps=2, conformal_pec=True, skip_preflight=True)


@pytest.mark.parametrize("entry", ("run", "forward"))
@pytest.mark.parametrize("on_face", (False, True))
def test_port_material_fold_is_checked_after_setup(entry, on_face):
    from rfx.geometry.csg import Box
    sim = _sim(waveform="differentiated_gaussian")
    z = 0.001 if on_face else 0.005
    # Each sheet is clear of the source shell; only the load cells between
    # them cross it. Those conductivities are added during port setup.
    for height in (z, z + 0.004):
        sim.add(Box((0.007 if height == z else 0.009, 0.007, height), (0.011, 0.009, height)), material="pec")
    sim.add_msl_port((0.009, 0.008, z), width=0.002, height=0.004,
                     excite=False, mode="uniform", name="passive")
    if on_face:
        with pytest.raises(ValueError, match="vacuum.*all six faces"):
            getattr(sim, entry)(n_steps=8, skip_preflight=True)
    else:
        getattr(sim, entry)(n_steps=8, skip_preflight=True)


def test_isolated_target_scatters_into_both_open_transverse_axes():
    from rfx.geometry.csg import Box
    fields = []
    for target in (False, True):
        sim = _sim()
        if target:
            sim.add_material("target", eps_r=3.0)
            sim.add(Box((0.008, 0.007, 0.006), (0.010, 0.009, 0.008)), material="target")
        cfg, _ = _aux(sim)
        sim.add_probe((0.009, 0.002, 0.007), component="ez")
        sim.add_probe((0.009, 0.008, 0.002), component="ez")
        fields.append(np.asarray(sim.run(n_steps=160).time_series))
        assert sim.boundary_model().requirements[1].admissible[0].value == "ABSORBER"
        assert cfg.closed_box
    # A source-only box has only rounding leakage. Both transverse
    # directions must carry a target response beyond that algebra floor.
    for axis in range(2):
        assert np.max(np.abs(fields[1][:, axis])) > (
            np.max(np.abs(fields[0][:, axis])) + 9 * np.spacing(np.float32(1.0)))


def test_ordinary_forward_run_and_internal_remat_share_the_source():
    from rfx.geometry.csg import Box
    sim = _sim()
    sim.add_material("target", eps_r=2.0)
    sim.add(Box((0.008, 0.007, 0.006), (0.010, 0.009, 0.008)), material="target")
    sim.add_probe((0.009, 0.002, 0.007), component="ez")
    result = sim.run(n_steps=128)
    fwd = sim.forward(n_steps=128, checkpoint=False, skip_preflight=True)
    _peak_ulp(fwd.time_series, result.time_series)
    for segments in (None, 4):
        remat = sim.forward(n_steps=128, checkpoint=True,
                            checkpoint_segments=segments, skip_preflight=True)
        _peak_ulp(remat.time_series, fwd.time_series)
        np.testing.assert_allclose(jnp.sum(remat.time_series ** 2),
                                   jnp.sum(fwd.time_series ** 2), rtol=1e-4, atol=0)


@pytest.mark.parametrize("kind", ("custom", "box"))
@pytest.mark.parametrize("transform", ("grad", "jit"))
def test_run_external_transforms_refuse_extended_tfsf(kind, transform):
    sim = _sim(closed_box=kind == "box", waveform=(
        "differentiated_gaussian" if kind == "box" else None))

    def trace(eps):
        return sim.run(n_steps=8, skip_preflight=True).time_series * eps

    calls = {
        "grad": lambda: jax.grad(lambda eps: jnp.sum(trace(eps) ** 2))(1.0),
        "jit": lambda: jax.jit(trace)(1.0),
    }
    with pytest.raises(NotImplementedError) as exc:
        calls[transform]()
    assert str(exc.value) == (
        "CustomWaveform and closed_box TFSF cannot be passed through run() "
        "under external JAX transformations; differentiate forward() instead.")


@pytest.mark.parametrize("public_api", ("core_opaque", "core_predicate"))
def test_transform_guard_uses_older_public_jax_exports(monkeypatch, public_api):
    """0.6 exports the pair from core; the 0.4.20 floor exports a predicate."""
    import sys
    from types import ModuleType
    from rfx.runners._admission import _refuse_transformed_extended_tfsf

    # Keep the real installed trace semantics while presenting each older
    # public export layout. Actual 0.6 execution is a separate runtime gate.
    try:
        from jax.extend.core import get_opaque_trace_state as get_state
        from jax.extend.core import take_current_trace as take_trace
    except ImportError:
        try:
            from jax.core import get_opaque_trace_state as get_state
            from jax.core import take_current_trace as take_trace
        except ImportError:
            from contextlib import contextmanager
            from jax.core import trace_state_clean
            in_eager_context = False

            def get_state():
                return in_eager_context or trace_state_clean()

            @contextmanager
            def take_trace():
                nonlocal in_eager_context
                in_eager_context = True
                try:
                    yield
                finally:
                    in_eager_context = False
    old_extension = ModuleType("jax.extend.core")
    old_core = ModuleType("jax.core")
    old_core.__dict__.update(jax.core.__dict__)
    old_core.__dict__.pop("__getattr__", None)
    for name in ("get_opaque_trace_state", "take_current_trace"):
        old_core.__dict__.pop(name, None)
    if public_api == "core_opaque":
        old_core.get_opaque_trace_state = get_state
        old_core.take_current_trace = take_trace
    else:
        def clean():
            caller = get_state()
            with take_trace():
                return caller == get_state()
        old_core.trace_state_clean = clean
    monkeypatch.setitem(sys.modules, "jax.extend.core", old_extension)
    monkeypatch.setitem(sys.modules, "jax.core", old_core)
    entry = _sim()._tfsf

    def call(value):
        _refuse_transformed_extended_tfsf(entry)
        return value ** 2

    assert call(2.0) == 4.0
    for transformed, value in ((jax.grad(call), 2.0), (jax.jit(call), 2.0),
                               (jax.vmap(call), jnp.ones(2))):
        with pytest.raises(NotImplementedError) as exc:
            transformed(value)
        assert str(exc.value) == (
            "CustomWaveform and closed_box TFSF cannot be passed through run() "
            "under external JAX transformations; differentiate forward() instead.")


@pytest.mark.parametrize("entry", ("run", "forward"))
@pytest.mark.parametrize("crossing", (False, True), ids=("inside", "crossing"))
def test_closed_box_pec_sheet_declaration_and_execution(entry, crossing):
    from rfx import Box

    sim = Simulation(freq_max=15e9, domain=(0.036, 0.024, 0.024), dx=1e-3,
                     boundary="cpml", cpml_layers=8)
    sheet = (Box((0.002, 0.010, 0.012), (0.006, 0.014, 0.012)) if crossing else
             Box((0.018, 0.008, 0.008), (0.018, 0.016, 0.016)))
    sim.add(sheet, material="pec")
    sim.add_tfsf_source(f0=8e9, margin=4, direction="+x", polarization="ez",
                        closed_box=True)
    sim.add_probe((0.0215, 0.0135, 0.0105), "ez")
    if crossing:
        with pytest.raises(ValueError, match="requires vacuum on and adjacent to all six faces"):
            getattr(sim, entry)(n_steps=20, skip_preflight=True)
    else:
        result = getattr(sim, entry)(n_steps=20, skip_preflight=True)
        assert result.time_series.shape[0] == 20
        assert np.isfinite(np.asarray(result.time_series)).all()
