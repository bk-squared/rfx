"""#1424 F2 gates: CPU fixtures; measurements use record_property, never stdout."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, gradient_record_length_witness
from tests._x64_compat import enable_x64
from tests.contracts.path_equivalence.comparison import compare
from tests.unit.autodiff.test_design_box_tape import (
    _sim, _box_cells, _eps_design, BOX_LO, BOX_HI, F0, DOMAIN,
)

STEPS = 2400


def fixture(precision="float64", point=True):
    sim = _sim(precision=precision)
    # cutoff=6 suppresses the differentiated pulse's deposited DC charge.
    sim._ports.clear()
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8, cutoff=6))
    grid = sim._build_grid()
    ix = grid.position_to_index((18e-3, 10e-3, 8e-3))
    width = 1 if point else 3
    region = (ix[1], ix[1]+width, ix[2], ix[2]+width)
    sim.add_dft_plane_probe(axis="x", coordinate=18e-3, component="ez",
        freqs=jnp.asarray([0.8*F0, 1.05*F0], dtype=precision), name="out", region=region)
    eps = jnp.asarray(_eps_design(_box_cells(sim)[1]), dtype=precision)
    return sim, eps


def objective(sim, steps=STEPS, mode="autodiff"):
    def loss(eps, sigma):
        result = sim.forward(design_box=(BOX_LO, BOX_HI),
            design_eps_override=eps, design_sigma_override=sigma,
            n_steps=steps, checkpoint_segments=8, gradient=mode, skip_preflight=True)
        spectrum = result.dft_planes["out"].accumulator / result.dt
        # Complex and spatially varying cotangents, multiple nonorthogonal bins.
        weights = jnp.arange(spectrum.size).reshape(spectrum.shape) + 1
        return jnp.sum(weights * jnp.abs(spectrum)**2)
    return loss


def decay(sim, eps, sigma, steps=STEPS):
    """Full-domain peak E and impedance-scaled H, including CPML, every step."""
    import rfx.simulation as runner
    original = runner.make_core_step
    def diagnostic(*args, **kwargs):
        core = original(*args, **kwargs)
        def step(*args):
            carry, probes, extras = core(*args)
            st = carry["fdtd"]
            norm = jnp.max(jnp.stack([jnp.max(jnp.abs(getattr(st, c))) * scale
                for c, scale in zip(("ex", "ey", "ez", "hx", "hy", "hz"),
                                    (1, 1, 1, 376.730313, 376.730313, 376.730313))]))
            return carry, jnp.reshape(norm, (1,)), extras
        return step
    with patch.object(runner, "make_core_step", diagnostic):
        def measure(e, s):
            return sim.forward(design_box=(BOX_LO, BOX_HI), design_eps_override=e,
                design_sigma_override=s, n_steps=steps, checkpoint=False,
                skip_preflight=True).time_series[:, 0]
        record = np.asarray(jax.jit(measure)(eps, sigma))
    peak = float(np.max(record))
    end = float(np.max(record[-64:]))
    return dict(peak=peak, end=end, decay_db=float(20*np.log10(peak/end)))


def errors(actual, reference):
    return [float(np.max(np.abs(np.asarray(a)-np.asarray(b))) / np.max(np.abs(b)))
            for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(reference))]


@contextmanager
def sigma_diagnostic_admission():
    """Keep the unresolved sigma numerical witness below public admission."""
    import rfx.adjoint as adjoint
    original = adjoint.admit_forward_adjoint
    def admit(sim, **kwargs):
        return original(sim, **{**kwargs, "design_sigma": None})
    with patch.object(adjoint, "admit_forward_adjoint", admit):
        yield


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize("point", [True, False])
def test_g1_eps(precision, lossy, point, record_property):
    with enable_x64() if precision == "float64" else nullcontext():
        sim, eps = fixture(precision, point)
        if lossy:
            sim.add_material("fixed_loss", eps_r=1., sigma=0.2)
            sim.add(Box((6e-3, 4e-3, 2e-3), (18e-3, 16e-3, 14e-3)),
                    material="fixed_loss")
        settling = decay(sim, eps, None)
        results = [jax.jit(jax.value_and_grad(objective(sim, mode=mode)))(eps, None)
                   for mode in ("autodiff", "adjoint")]
        err = errors(results[1][1], results[0][1])[0]
        record_property("measurement", f"G1 eps {precision=} {lossy=} {point=} {settling=} error={err}")
        assert settling["decay_db"] >= 100
        # Two traces of one objective summed over the record: float64 keeps 1e-6;
        # float32 takes the cross-trace bar for a summed quantity.
        if precision == "float64":
            np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-6)
        else:
            compare(results[0][0], results[1][0], record="g1_eps.value",
                    kind="accumulated", measurements=[])
        assert err <= (1e-5 if precision == "float64" else 1e-4)


@pytest.mark.parametrize("sigma", [0., 0.2])
def test_design_sigma_refused_at_admission(sigma):
    sim, eps = fixture("float32")
    for value in (sigma, jnp.full_like(eps, sigma), (eps * 0 + sigma,) * 3):
        with patch.object(sim, "_build_grid", side_effect=AssertionError("past admission")):
            with pytest.raises(NotImplementedError, match="#1424.*on lossless design cells the gradient with respect to conductivity has no settled value in a finite record"):
                sim.forward(gradient="adjoint", design_box=(BOX_LO, BOX_HI),
                            design_eps_override=eps, design_sigma_override=value)
    with pytest.raises(NotImplementedError, match="#1424.*on lossless design cells the gradient with respect to conductivity has no settled value in a finite record"):
        jax.grad(objective(sim, mode="adjoint"), argnums=1)(eps, jnp.full_like(eps, sigma))


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("lossy", [
    True,
    pytest.param(False, marks=pytest.mark.xfail(
        strict=True, reason="#1424: on lossless cells dJ/dCa has no settled value in a finite record (static sensitivity)")),
])
@pytest.mark.parametrize("point", [True, False])
def test_g1_sigma(precision, lossy, point, record_property):
    with sigma_diagnostic_admission(), (enable_x64() if precision == "float64" else nullcontext()):
        sim, eps = fixture(precision, point)
        sigma = jnp.full_like(eps, 0.2 if lossy else 0.)
        settling = decay(sim, eps, sigma)
        record_property("measurement", f"G1 {precision=} {lossy=} {point=} {settling=}")
        assert settling["decay_db"] >= 100
        results = [jax.jit(jax.value_and_grad(objective(sim, mode=mode), argnums=(0, 1)))(eps, sigma)
                   for mode in ("autodiff", "adjoint")]
        err = errors(results[1][1], results[0][1])
        record_property("measurement", f"G1 {precision=} {lossy=} {point=} values={[float(r[0]) for r in results]} errors={err}")
        # Two traces of one objective summed over the record: float64 keeps 1e-6;
        # float32 takes the cross-trace bar for a summed quantity.
        if precision == "float64":
            np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-6)
        else:
            compare(results[0][0], results[1][0], record="g1_sigma.value",
                    kind="accumulated", measurements=[])
        assert max(err) <= (1e-5 if precision == "float64" else 1e-4)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_g2(precision, record_property):
    with enable_x64() if precision == "float64" else nullcontext():
        sim, eps = fixture(precision)
        sigma = None
        short = 128
        settling = decay(sim, eps, sigma, short)
        record_property("measurement", f"G2 {precision=} short_decay={settling}")
        assert settling["decay_db"] < 100
        grads = [jax.jit(jax.grad(objective(sim, steps, mode)))(eps, sigma)
                 for steps, mode in ((short, "autodiff"), (short, "adjoint"),
                                     (2*short, "autodiff"))]
        record_property("measurement", f"G2 {precision=} short={short} long={2*short} "
              f"adjoint_vs_short={errors(grads[1], grads[0])} "
              f"adjoint_vs_long={errors(grads[1], grads[2])} "
              f"short_vs_long={errors(grads[0], grads[2])}")
        assert all(np.all(np.isfinite(a)) for g in grads for a in jax.tree.leaves(g))


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_g5_default(precision, record_property):
    with enable_x64() if precision == "float64" else nullcontext():
        sim, eps = fixture(precision)
        sigma = jnp.full_like(eps, 0.2)
        def result(e, explicit):
            options = {"gradient": "autodiff"} if explicit else {}
            r = sim.forward(design_box=(BOX_LO, BOX_HI), design_eps_override=e,
                design_sigma_override=sigma, n_steps=48, skip_preflight=True, **options)
            assert r.adjoint_settling is None
            return r.time_series, r.dft_planes["out"].accumulator
        functions = [lambda e: result(e, False), lambda e: result(e, True)]
        assert str(jax.make_jaxpr(functions[0])(eps)) == str(jax.make_jaxpr(functions[1])(eps))
        for a, b in zip(jax.jit(functions[0])(eps), jax.jit(functions[1])(eps)):
            np.testing.assert_array_equal(a, b)
        record_property("measurement", f"G5 default {precision=} bit_equal=True jaxpr_equal=True")


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_g5_mutation(precision, record_property):
    import rfx.adjoint as adjoint
    with enable_x64() if precision == "float64" else nullcontext():
        sim, eps = fixture(precision)
        sigma = None
        reference = jax.jit(jax.grad(objective(sim)))(eps, sigma)
        def drop_solve(freqs, dt, length, target):
            # Mutation: independently normalized carriers; all off-diagonal
            # leakage and the negative-frequency image are dropped.
            n = jnp.arange(length, dtype=freqs.dtype)
            basis = adjoint._wavelet_basis(freqs, dt, length, n)
            transform = jnp.exp(-2j*jnp.pi*freqs[:, None]*(n+1)*dt)*dt
            diagonal = jnp.diag(transform @ basis)
            return target / diagonal[:, None, None]
        with patch.object(adjoint, "_wavelet_coefficients", drop_solve):
            mutated = jax.jit(jax.grad(objective(sim, mode="adjoint")))(eps, sigma)
        err = errors(mutated, reference)
        record_property("measurement", f"G5 mutation {precision=} {err=}")
        assert max(err) > (1e-5 if precision == "float64" else 1e-4)


@pytest.mark.parametrize("path", ["graded", "distributed", "ringdown", "ports",
    "debye", "lorentz", "kerr", "upml", "occupancy", "whole_grid", "missing_box",
    "tfsf", "precision", "solver", "invalid", "sheet", "current_moments", "stencil",
    "ntff", "h_plane", "no_bins", "boundary_plane", "duplicate", "dc", "nyquist",
    "different_bins", "time_domain"])
def test_g5_refusals(path):
    sim, eps = fixture("float32")
    options = dict(gradient="adjoint", design_box=(BOX_LO, BOX_HI),
                   design_eps_override=eps, n_steps=48, skip_preflight=True)
    match = "adjoint"
    if path == "graded":
        sim._dz_profile = np.full(8, 2e-3)
    elif path == "distributed":
        options["distributed"] = True
    elif path == "ringdown":
        options["ringdown"] = object()
    elif path == "ports":
        sim.add_port(position=(4e-3, 10e-3, 8e-3), component="ez", impedance=50)
    elif path in ("debye", "lorentz", "kerr"):
        from rfx.materials.debye import DebyePole
        from rfx.materials.lorentz import LorentzPole
        extra = {"debye_poles": [DebyePole(1., 1e-11)]} if path == "debye" else (
            {"lorentz_poles": [LorentzPole(1., 2e10, 1e9)]} if path == "lorentz"
            else {"chi3": 1e-20})
        sim.add_material("disp", eps_r=3., **extra)
        sim.add(Box(BOX_LO, BOX_HI), material="disp")
    elif path == "upml":
        sim = _sim(boundary="upml")
        match = "upml"
    elif path == "occupancy":
        options["design_occupancy_override"] = jnp.zeros_like(eps)
    elif path == "whole_grid":
        options["eps_override"] = jnp.ones(sim._build_grid().shape)
    elif path == "missing_box":
        options["design_box"] = None
    elif path == "tfsf":
        sim._ports.clear()
        sim.add_tfsf_source(f0=F0, closed_box=True)
    elif path == "precision":
        sim = _sim(precision="mixed")
    elif path == "current_moments":
        sim.add_current_moment_monitor(BOX_LO, BOX_HI, block_size=4e-3, freqs=[F0])
    elif path == "sheet":
        with pytest.warns(UserWarning, match="Leontovich"):
            sim.add_thin_conductor(Box((10e-3, 8e-3, 8e-3), (14e-3, 12e-3, 8e-3)),
                sigma_bulk=5.8e7, thickness=35e-6, surface_impedance_f0=F0)
        match = "sheet"
    elif path == "stencil":
        sim._stencil_order = 4
        match = "stencil_order"
    elif path == "solver":
        sim._solver = "adi"
    elif path == "invalid":
        options["gradient"] = "unknown"
        match = "gradient must"
    elif path == "ntff":
        sim.add_ntff_box((2e-3, 2e-3, 2e-3), (22e-3, 18e-3, 14e-3), freqs=jnp.array([F0]))
    elif path == "h_plane":
        sim._dft_planes[0] = replace(sim._dft_planes[0], component="hz")
    elif path == "no_bins":
        sim._dft_planes.clear()
    elif path == "boundary_plane":
        sim._dft_plane_regions.clear()
    elif path in ("duplicate", "dc", "nyquist"):
        sim._dft_planes[0] = replace(sim._dft_planes[0], freqs=jnp.array(
            {"duplicate": [F0, F0], "dc": [0.], "nyquist": [1./sim._build_grid().dt]}[path]))
    elif path == "different_bins":
        sim.add_dft_plane_probe(axis="x", coordinate=18e-3, freqs=jnp.array([F0]),
            region=sim._dft_plane_regions["out"])
    elif path == "time_domain":
        def loss(e):
            return jnp.sum(sim.forward(**{**options, "design_eps_override": e}).time_series**2)
        with pytest.raises(NotImplementedError, match="time-domain"):
            jax.grad(loss)(eps)
        return
    with pytest.raises((NotImplementedError, ValueError), match=match):
        sim.forward(**options)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_wavelet_targets(precision, record_property):
    from rfx.adjoint import _wavelet_basis, _wavelet_coefficients
    with enable_x64() if precision == "float64" else nullcontext():
        freqs = jnp.array([0.063, 0.078], dtype=precision)
        target = jnp.array([1+2j, -2+0.3j], dtype=jnp.complex128 if precision == "float64" else jnp.complex64)
        n = jnp.arange(128, dtype=precision)
        coeff = _wavelet_coefficients(freqs, 1., 128, target)
        wave = 2*jnp.real(_wavelet_basis(freqs, 1., 128, n) @ coeff)
        actual = jnp.exp(-2j*jnp.pi*freqs[:, None]*(n+1)) @ wave
        error = float(jnp.max(jnp.abs(actual-target)))
        dc = float(jnp.abs(jnp.sum(wave)))
        record_property("measurement", f"wavelet {precision=} bin_error={error} dc={dc}")
        assert error < (1e-12 if precision == "float64" else 2e-5)
        assert dc < (1e-12 if precision == "float64" else 2e-5)


@pytest.mark.parametrize("flag", ["use_debye", "use_lorentz", "use_kerr", "use_upml",
    "use_design_occupancy", "use_current_moments", "use_sheet_impedance", "use_tfsf",
    "use_waveguide_ports", "use_lumped_rlc", "use_wire_sparams", "use_lumped_sparams",
    "use_ntff", "use_flux_monitors", "use_aniso_inv", "use_conformal",
    "use_pec_occupancy", "use_pmc_faces"])
def test_g5_kernel_refusals(flag):
    from types import SimpleNamespace
    from rfx.adjoint import design_adjoint_scan
    class Context(SimpleNamespace):
        def __getattr__(self, name):
            if name.startswith("use_"):
                return False
            raise AttributeError(name)
    with pytest.raises(NotImplementedError, match=flag):
        design_adjoint_scan(Context(**{flag: True}), {}, ())


@pytest.mark.parametrize("case", ["missing_design", "bloch", "periodic", "anisotropy", "dtype", "stencil"])
def test_g5_kernel_admission(case):
    from types import SimpleNamespace
    from rfx.adjoint import design_adjoint_scan
    class Context(SimpleNamespace):
        def __getattr__(self, name):
            if name.startswith("use_"):
                return False
            raise AttributeError(name)
    ctx = Context(use_design_box=True, bloch=None, periodic=(False,)*3,
                  aniso_eps=None, stencil_order=2)
    initial = {"fdtd": SimpleNamespace(ex=jnp.zeros(1))}
    if case == "missing_design":
        ctx.use_design_box = False
    elif case == "bloch":
        ctx.bloch = (1, 1, 1)
    elif case == "periodic":
        ctx.periodic = (True, False, False)
    elif case == "anisotropy":
        ctx.aniso_eps = (1, 1, 1)
    elif case == "dtype":
        initial["fdtd"].ex = jnp.zeros(1, jnp.float16)
    elif case == "stencil":
        ctx.stencil_order = 4
    with pytest.raises(NotImplementedError, match="adjoint"):
        design_adjoint_scan(ctx, initial, ())


def _raw_coefficient_results(n_steps):
    """#1424 follow-up: differentiate the identical resolved edge arrays."""
    import rfx.adjoint as adjoint
    from rfx.simulation import make_core_step, core_step_invariants

    class Captured(Exception):
        pass

    captured = {}
    def capture(ctx, initial, xs):
        captured.update(ctx=ctx, initial=initial, xs=xs)
        raise Captured

    with enable_x64():
        sim, eps = fixture("float64", True)
        with patch.object(adjoint, "design_adjoint_scan", capture):
            with pytest.raises(Captured):
                objective(sim, steps=n_steps, mode="adjoint")(eps, None)
        ctx, initial, xs = (captured[k] for k in ("ctx", "initial", "xs"))
        coefficients = (ctx.design_box.ca, ctx.design_box.cb)

        def loss(coeffs, mode):
            local = replace(ctx, design_box=ctx.design_box._replace(
                ca=coeffs[0], cb=coeffs[1]))
            if mode == "adjoint":
                last, _ = adjoint.design_adjoint_scan(local, initial, xs)
            else:
                core = make_core_step(local, core_step_invariants(local))
                def step(carry, row):
                    carry, _, _ = core(carry, *row)
                    return carry, None
                def segment(carry, rows):
                    return jax.lax.scan(step, carry, rows)
                rows = jax.tree.map(lambda a: a.reshape((8, n_steps // 8) + a.shape[1:]), xs)
                last, _ = jax.lax.scan(jax.checkpoint(segment), initial, rows)
            spectrum = last["dft_planes"][0] / ctx.dt
            weights = jnp.arange(spectrum.size).reshape(spectrum.shape) + 1
            return jnp.sum(weights * jnp.abs(spectrum)**2)

        results = [jax.jit(jax.value_and_grad(lambda ab: loss(ab, mode)))(coefficients)
                   for mode in ("autodiff", "adjoint")]
        return results, coefficients


def _peak(arrays):
    return max(float(np.max(np.abs(a))) for a in arrays)


def _difference(actual, reference):
    return max(float(np.max(np.abs(a-b))) for a, b in zip(actual, reference))


def test_g1_raw_coefficients_lossless_point(record_property):
    """Finite-record Ca sensitivity moves while the settled overlap stays fixed."""
    with enable_x64():
        short, coefficients = _raw_coefficient_results(2400)
        long, longer_coefficients = _raw_coefficient_results(2408)
        for a, b in zip(jax.tree.leaves(coefficients), jax.tree.leaves(longer_coefficients)):
            np.testing.assert_array_equal(a, b)
        peak_ca = _peak(short[1][1][0])
        adjoint_ca_change = _difference(short[1][1][0], long[1][1][0]) / peak_ca
        autodiff_ca_change = _difference(short[0][1][0], long[0][1][0]) / peak_ca
        cb_errors = [_difference(r[0][1][1], r[1][1][1]) / _peak(r[0][1][1])
                     for r in (short, long)]
        value_errors = [float(abs(r[0][0]-r[1][0]) / abs(r[0][0])) for r in (short, long)]
        record_property("adjoint_ca_change", adjoint_ca_change)
        record_property("autodiff_ca_change", autodiff_ca_change)
        for n, cb, value in zip((2400, 2408), cb_errors, value_errors):
            record_property(f"cb_error_{n}", cb)
            record_property(f"value_error_{n}", value)
        assert adjoint_ca_change <= 1e-6
        assert autodiff_ca_change > 1e-2
        assert max(cb_errors) <= 1e-5
        assert max(value_errors) <= 1e-12


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("steps", [STEPS, 64])
def test_adjoint_settling(precision, steps, record_property):
    with enable_x64() if precision == "float64" else nullcontext():
        sim, eps = fixture(precision)
        def measure(e):
            result = sim.forward(design_box=(BOX_LO, BOX_HI),
                design_eps_override=e, n_steps=steps, gradient="adjoint",
                skip_preflight=True)
            return jnp.sum(jnp.abs(result.dft_planes["out"].accumulator)**2), result.adjoint_settling
        (_, witness), grad = jax.jit(jax.value_and_grad(measure, has_aux=True))(eps)
        record_property("measurement", f"SETTLING {precision=} {steps=} ratio={float(witness):.17g}")
        np.testing.assert_array_equal(jax.jit(measure)(eps)[1], witness)
        assert witness.shape == ()
        assert np.all(np.isfinite(grad))
        assert witness < 1e-5 if steps == STEPS else witness > 1e-1


def test_clearing_drives_suppresses_electric_and_magnetic_sources(monkeypatch):
    """One context field disables both injection stages, even with live xs."""
    import rfx.simulation as runner
    from rfx.core.yee import init_materials, init_state
    from rfx.grid import Grid

    contexts = []
    original = runner.make_core_step
    def capture(ctx, *args, **kwargs):
        contexts.append(ctx)
        return original(ctx, *args, **kwargs)
    monkeypatch.setattr(runner, "make_core_step", capture)
    grid = Grid(freq_max=1e9, domain=(.012,)*3, dx=.001,
                cpml_layers=0, cpml_axes="")
    runner.run(grid, init_materials(grid.shape), 1,
        sources=[runner.SourceSpec(4, 4, 4, "ez", jnp.ones(1))],
        mag_sources=[runner.MagneticSourceSpec(4, 4, 4, "hx", jnp.ones(1)*2)])
    ctx = contexts[-1]
    def step(context):
        carry, _, _ = jax.jit(original(context))(
            {"fdtd": init_state(grid.shape)}, jnp.int32(0), jnp.ones(1), jnp.ones(1)*2)
        return carry["fdtd"]
    driven = step(ctx)
    assert np.max(np.abs(driven.ez)) > 0
    assert np.max(np.abs(driven.hx)) > 0
    suppressed = step(replace(ctx, drives=None))
    for field in suppressed[:6]:
        np.testing.assert_array_equal(field, jnp.zeros_like(field))


def test_adjoint_zero_cotangent_has_no_primal_drive(monkeypatch):
    """Exercise the actual adjoint context replacement and its full scan."""
    import rfx.simulation as runner
    original = runner.make_core_step
    adjoint_contexts = []
    def capture(ctx, *args, **kwargs):
        if kwargs.get("design_hook") is not None and not ctx.use_dft_planes:
            adjoint_contexts.append(ctx)
        return original(ctx, *args, **kwargs)
    monkeypatch.setattr(runner, "make_core_step", capture)
    sim, eps = fixture("float32")
    loss = objective(sim, steps=48, mode="adjoint")
    gradient = jax.jit(jax.grad(lambda e: loss(e, None)*0.))(eps)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))
    assert adjoint_contexts
    assert all(ctx.drives is None for ctx in adjoint_contexts)


def cavity_fixture():
    """Reviewer's scene C, including its one bin and one monitor cell."""
    sim = _sim(boundary="pec", precision="float64")
    sim._ports.clear()
    sim.add_material("fill", eps_r=1.0, sigma=0.023)
    sim.add(Box((0, 0, 0), DOMAIN), material="fill")
    sim.add_source((4e-3, 10e-3, 8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8, cutoff=6))
    g = sim._build_grid()
    ix = g.position_to_index((18e-3, 10e-3, 8e-3))
    sim.add_dft_plane_probe(axis="x", coordinate=18e-3, component="ez",
        freqs=jnp.asarray([1.05*F0]), name="out",
        region=(ix[1], ix[1]+1, ix[2], ix[2]+1))
    return sim, jnp.asarray(_eps_design(_box_cells(sim)[1]))


@pytest.mark.parametrize("steps", [621, 623, 625, 627, 629])
def test_cavity_adjoint_settling(steps, record_property):
    with enable_x64():
        sim, eps = cavity_fixture()
        result = sim.forward(design_box=(BOX_LO, BOX_HI),
            design_eps_override=eps, n_steps=steps, gradient="adjoint",
            skip_preflight=True)
        indicator = float(result.adjoint_settling)
        record_property("indicator", indicator)
        assert indicator > 1e-2


@pytest.mark.parametrize("scene,steps", [("cavity", 625), ("open", 2400)])
def test_adjoint_gradient_record_length_witness(scene, steps, record_property):
    with enable_x64():
        sim, eps = cavity_fixture() if scene == "cavity" else fixture()
        def loss(e, n):
            result = sim.forward(design_box=(BOX_LO, BOX_HI),
                design_eps_override=e, n_steps=n, gradient="adjoint",
                skip_preflight=True)
            return jnp.sum(jnp.abs(result.dft_planes["out"].accumulator / result.dt)**2)
        witness = gradient_record_length_witness(loss, eps, steps, tol=1e-2, factor=2.0)
        record_property("worst", witness.worst)
        record_property("passed", witness.passed)
        if scene == "cavity":
            assert witness.passed is False
            assert witness.worst >= 1e-1
        else:
            assert witness.passed is True
            assert witness.worst <= 1e-4
