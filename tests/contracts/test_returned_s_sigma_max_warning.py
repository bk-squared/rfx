"""Physical anchor: passive S has sigma_max <= 1; validated residual <= 0.006."""

from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.errors import AccuracyWarning
from rfx.probes.probes import warn_if_nonpassive_lumped_s11
from rfx.sparams._common import _finalize_sparam_result
from rfx.sparams.passivity_warning import warn_if_singular_value_exceeds_one


FREQS = np.array([1e9, 2e9, 3e9])


def _matrix(peak):
    # Rank-one symmetric matrix: its singular values are peak and zero.
    # Each column power is peak**2 / 2, so column-power checks cannot see it.
    s = np.full((2, 2, 3), 1.005 / 2, dtype=complex)
    s[:, :, 1] = peak / 2
    return s


def _mixed_exit(monkeypatch, s):
    import rfx.sparams.mixed as mixed
    from tests.unit.sparams.test_mixed_port_sparam import _base_sim, _add_feed, _add_msl

    sim, yc = _base_sim()
    _add_feed(sim, yc)
    _add_msl(sim, yc, n_probe_offset=10, n_probe_spacing=4)
    constructor = mixed.MixedSMatrixResult

    def assembled_result(**kwargs):
        # Supply the synthetic returned S at assembly, before either audit.
        # The extraction itself needs only a single step for this exit test.
        kwargs["S"] = s
        return constructor(**kwargs)

    monkeypatch.setattr(mixed, "MixedSMatrixResult", assembled_result)
    return sim.compute_mixed_s_matrix(
        freqs=FREQS, n_steps=1, skip_preflight=True, magnitude_channel="wave",
    )


@pytest.mark.parametrize("exit_name", ["finalize", "mixed", "lumped"])
@pytest.mark.parametrize("peak, expected_count", [(1.02, 1), (1.005, 0)])
def test_returned_s_exit_contract(monkeypatch, exit_name, peak, expected_count):
    s = _matrix(peak)
    before = s.copy()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if exit_name == "finalize":
            assembled = SimpleNamespace(s_params=s, freqs=FREQS, port_names=("a", "b"))
            result = _finalize_sparam_result(assembled, extractor="contract", strict=False)
            assert result is assembled
        elif exit_name == "mixed":
            result = _mixed_exit(monkeypatch, s)
            assert result.S is s
        else:
            warn_if_nonpassive_lumped_s11(s, FREQS, extractor="contract")
    notices = [w for w in seen if "worst sigma_max" in str(w.message)]
    assert len(notices) == expected_count
    if notices:
        assert notices[0].category is AccuracyWarning
        message = str(notices[0].message)
        assert "1 of 3 frequency bins" in message
        assert "sigma_max = 1.02" in message and "at 2 GHz" in message
        assert ("compute_mixed_s_matrix" if exit_name == "mixed" else "contract") in message
        assert "S is returned as extracted" in message
        assert "cannot scatter more power than it receives" in message
        assert "extraction or record artifact" in message
    np.testing.assert_array_equal(s, before)


@pytest.mark.parametrize("exit_name", ["finalize", "mixed", "lumped"])
@pytest.mark.parametrize("peak, expected_count", [(1.04, 1), (1.005, 0)])
def test_nan_column_exit_contract(monkeypatch, exit_name, peak, expected_count):
    # Mixed extraction refuses the entire lumped/wire drive column.
    s = np.zeros((2, 2, 3), dtype=complex)
    s[:, 0, :] = np.nan
    s[1, 1, :] = [1.005, peak, 1.005]
    before = s.copy()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if exit_name == "finalize":
            assembled = SimpleNamespace(s_params=s, freqs=FREQS, port_names=("a", "b"))
            result = _finalize_sparam_result(assembled, extractor="contract", strict=False)
            assert result is assembled
        elif exit_name == "mixed":
            result = _mixed_exit(monkeypatch, s)
            assert result.S is s
        else:
            warn_if_nonpassive_lumped_s11(s, FREQS, extractor="contract")
    notices = [w for w in seen if "worst sigma_max" in str(w.message)]
    assert len(notices) == expected_count
    if notices:
        assert notices[0].category is AccuracyWarning
        message = str(notices[0].message)
        assert "1 of 3 frequency bins" in message
        assert "sigma_max = 1.04" in message and "at 2 GHz" in message
        assert "lower bound from the finite columns" in message
    np.testing.assert_array_equal(s, before)


@pytest.mark.parametrize("shape", ["square", "observed", "diagonal"])
def test_only_fully_finite_columns_contribute(shape):
    # The large finite entry shares a column with NaN and must be discarded.
    s = np.array([
        [[2.0, np.nan, np.inf], [0.0, 0.0, np.nan]],
        [[np.nan, np.nan, np.inf], [1.04j, 1.005, np.nan]],
    ])
    if shape == "observed":
        s = np.concatenate([s, np.zeros((1, 2, 3))], axis=0)
    elif shape == "diagonal":
        s = np.array([[np.nan, np.inf, np.nan], [1.04j, 1.005, np.nan]])
    before = s.copy()
    with pytest.warns(AccuracyWarning, match="1 of 3 frequency bins") as seen:
        warn_if_singular_value_exceeds_one(s, FREQS, extractor="partial")
    assert len(seen) == 1
    message = str(seen[0].message)
    assert "sigma_max = 1.04" in message and "at 1 GHz" in message
    assert "lower bound from the finite columns" in message
    if shape != "square":
        assert ("observed-column" if shape == "observed" else "diagonal-only") in message
    np.testing.assert_array_equal(s, before)


@pytest.mark.parametrize("rectangular", [False, True])
def test_fully_finite_singular_values_unchanged(monkeypatch, rectangular):
    s = np.array([[0.51, 0.51j], [-0.51j, 0.51]], dtype=complex)[..., None]
    if rectangular:
        s = np.concatenate([s, np.zeros((1, 2, 1))], axis=0)
    svd = np.linalg.svd
    measured = []

    def measure(matrix, **kwargs):
        assert matrix.dtype == np.complex128
        values = svd(matrix, **kwargs)
        measured.extend(values.reshape(-1, values.shape[-1])[:, 0])
        return values

    monkeypatch.setattr(np.linalg, "svd", measure)
    with pytest.warns(AccuracyWarning, match="sigma_max = 1.02") as seen:
        warn_if_singular_value_exceeds_one(s, FREQS[:1], extractor="finite")
    np.testing.assert_allclose(measured, [1.02], rtol=1e-12, atol=0.0)
    assert "lower bound from the finite columns" not in str(seen[0].message)
    assert ("observed-column" in str(seen[0].message)) == rectangular


@pytest.mark.parametrize("shape", ["one", "diagonal", "observed"])
def test_partial_shapes_and_nonfinite_bins(shape):
    s = np.array([0.5, 1.02, np.nan], dtype=complex)
    if shape == "diagonal":
        s = np.stack([s, np.full(3, 0.2)])
    elif shape == "observed":
        s = np.stack([s, np.zeros(3)])[:, None, :]
    with pytest.warns(AccuracyWarning, match="1 of 3 frequency bins") as seen:
        warn_if_singular_value_exceeds_one(s, FREQS, extractor="partial")
    assert len(seen) == 1
    assert ("lower bound" in str(seen[0].message)) == (shape != "one")
    assert "lower bound from the finite columns" not in str(seen[0].message)


def test_tracers_are_skipped_and_conversion_errors_propagate():
    class Unavailable:
        def __array__(self, *args, **kwargs):
            raise ValueError("no host value")

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        for s in (None, np.empty((0,)), np.full((2, 2, 3), np.nan)):
            warn_if_singular_value_exceeds_one(s, FREQS, extractor="unavailable")

        def traced(s):
            warn_if_singular_value_exceeds_one(s, FREQS, extractor="traced")
            return jnp.sum(s)

        np.testing.assert_allclose(jax.grad(traced)(jnp.ones(3)), np.ones(3), atol=1e-7)
    assert not seen
    with pytest.raises(ValueError, match="no host value"):
        warn_if_singular_value_exceeds_one(Unavailable(), FREQS, extractor="unavailable")


NEW_EXITS = (
    "coax", "coax-eager-ad", "distributed-run", "distributed-replay",
    "distributed-graded-run", "graded-forward", "distributed-forward",
    "distributed-graded-forward", "subgridded", "subgridded-diagnostic",
    "ringdown-run", "ringdown-forward",
)


def _port_scene(*, graded=False):
    from rfx import Simulation, GaussianPulse

    sim = Simulation(freq_max=10e9, domain=(.012, .008, .008), dx=.001,
                     boundary="pec", **({"dz_profile": np.full(8, .001)} if graded else {}))
    sim.add_port((.004, .004, .003), "ez", impedance=50.,
                 waveform=GaussianPulse(f0=2e9), extent=.002)
    return sim


def _coax_exit(monkeypatch, s, *, differentiable):
    import rfx.simulation as simulation
    import rfx.sources.coaxial_port as coax
    from rfx import Simulation

    sim = Simulation(domain=(.008, .008, .040), freq_max=40e9, boundary="cpml")
    sim.add_coaxial_port((.004, .004, .020), face="top", pin_length=.005)
    monkeypatch.setattr(simulation, "run", lambda *a, **kw: SimpleNamespace(dft_planes=kw["dft_planes"]))
    for name in ("coaxial_line_plane_voltage", "coaxial_line_plane_voltage_jnp"):
        monkeypatch.setattr(coax, name, lambda *a, **kw: jnp.zeros(3, dtype=complex))
    values = iter(s)
    monkeypatch.setattr(coax, "coaxial_line_reflection_from_plane_voltages",
                        lambda *a, **kw: SimpleNamespace(
                            reflection=next(values), gamma=0j, recurrence_residual=0., fit_residual=0.))
    result = sim.compute_coaxial_line_reflection(
        n_steps=1, freqs=FREQS, **({"eps_scale": 1.0} if differentiable else {}))
    return result.s11


def _new_exit(monkeypatch, exit_name, s):
    from rfx.api._spec import Result

    if exit_name.startswith("coax"):
        return _coax_exit(monkeypatch, s, differentiable=exit_name == "coax-eager-ad")
    if exit_name.startswith("ringdown"):
        from rfx.ringdown import RingdownRun, RingdownForward, RingdownForwardResult

        result = Result(state=None, time_series=np.zeros((2, 1)), s_params=np.zeros_like(s), freqs=FREQS)
        cls = RingdownRun if exit_name == "ringdown-run" else RingdownForward
        plan = object.__new__(cls)
        plan.probe_keys, plan.n_steps = ("internal",), 2
        if exit_name == "ringdown-run":
            monkeypatch.setattr(plan, "_host_report", lambda *a: (s, SimpleNamespace(ok=True), None))
            completed = plan._finish(result, 0)
        else:
            value = RingdownForwardResult(s, s, FREQS, None, (), result.s_params,
                                          None, None, None, None, None)
            monkeypatch.setattr(plan, "_traced", lambda *a: value)
            completed = plan._finish_forward(result, 0)
        assert completed.s_params is result.s_params
        np.testing.assert_array_equal(completed.time_series, np.zeros((2, 0)))
        assert completed.ringdown.s_params is s
        return completed.ringdown.s_params
    if exit_name.startswith("subgridded"):
        from rfx import Simulation, GaussianPulse
        from rfx.runners import subgridded

        sim = Simulation(freq_max=10e9, domain=(.012, .012, .020), dx=.002, boundary="pec")
        sim.add_refinement((0., .014), ratio=2, validation="research")
        for x in (.004, .008):
            sim.add_port((x, .006, .004), "ez", impedance=50., waveform=GaussianPulse(f0=2e9))
        calls = []

        def replay(*a, **kw):
            calls.append(kw.get("diagnostic_lumped_sparam_driven_index_override"))
            return Result(state=None, s_params=s, freqs=FREQS, time_series=np.zeros((1, 0)))

        monkeypatch.setattr(subgridded, "_run_subgridded_once", replay)
        diagnostic = exit_name == "subgridded-diagnostic"
        result = sim.run(n_steps=1, compute_s_params=not diagnostic,
                         s_param_freqs=None if diagnostic else FREQS, skip_preflight=True)
        assert calls == ([None] if diagnostic else [None, 0, 1])
        return result.s_params

    from rfx.measurement import port_s
    graded = "graded" in exit_name
    sim = _port_scene(graded=graded)
    if graded:
        assemble = port_s.assemble_graded_ports_result

        def assembled(*a, **kw):
            result = assemble(*a, **kw)
            result["s_params"] = s
            result["s_param_freqs"] = FREQS
            return result

        monkeypatch.setattr(port_s, "assemble_graded_ports_result", assembled)
    elif exit_name == "distributed-replay":
        from rfx.probes import sparam_driver
        monkeypatch.setattr(port_s, "main_record_port_s", lambda *a, **kw: None)
        monkeypatch.setattr(sparam_driver, "compute_lumped_wire_s_matrix_via_scan",
                            lambda *a, **kw: (s, None))
    else:
        monkeypatch.setattr(port_s, "port_s_from_record", lambda *a, **kw: s)
    options = dict(n_steps=1, skip_preflight=True)
    if exit_name.startswith("distributed"):
        options["devices"] = jax.devices("cpu")[:2]
        assert len(options["devices"]) == 2
    if "forward" in exit_name:
        result = sim.forward(**options, port_s11_freqs=FREQS,
                             distributed=exit_name.startswith("distributed"), checkpoint=False)
    else:
        result = sim.run(**options, compute_s_params=True, s_param_freqs=FREQS)
    return result.s_params


@pytest.mark.parametrize("exit_name", NEW_EXITS)
@pytest.mark.parametrize("peak, expected_count", [(1.02, 1), (1.005, 0)])
def test_new_returned_s_exits(monkeypatch, exit_name, peak, expected_count):
    s = np.array([1.005, peak, 1.005], dtype=complex) if exit_name.startswith("coax") else _matrix(peak)
    before = s.copy()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        returned = _new_exit(monkeypatch, exit_name, s)
    notices = [w for w in seen if "worst sigma_max" in str(w.message)]
    assert len(notices) == expected_count
    if notices:
        assert notices[0].category is AccuracyWarning
        assert "1 of 3 frequency bins" in str(notices[0].message)
        assert "sigma_max = 1.02" in str(notices[0].message)
        assert "at 2 GHz" in str(notices[0].message)
    # Some lanes publish complex64; these exactly represent the input after casting.
    np.testing.assert_array_equal(returned, before.astype(np.asarray(returned).dtype))
    np.testing.assert_array_equal(s, before)


@pytest.mark.parametrize("exit_name", ["finalize", "mixed", "lumped"] + [
    name for name in NEW_EXITS if not name.startswith("coax")])
@pytest.mark.parametrize("peak, expected_count", [(1.04, 1), (1.005, 0)])
def test_no_finite_column_exits(monkeypatch, exit_name, peak, expected_count):
    # Every column contains NaN; the final bin has no finite entry at all.
    s = np.array([[[np.nan, np.nan, np.nan], [1.005, peak * 1j, np.nan]],
                  [[0.2, 0.2, np.nan], [np.nan, np.nan, np.nan]]])
    before = s.copy()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if exit_name == "finalize":
            result = SimpleNamespace(s_params=s, freqs=FREQS, port_names=("a", "b"))
            assert _finalize_sparam_result(result, extractor="contract", strict=False) is result
            returned = result.s_params
        elif exit_name == "mixed":
            returned = _mixed_exit(monkeypatch, s).S
        elif exit_name == "lumped":
            warn_if_nonpassive_lumped_s11(s, FREQS, extractor="contract")
            returned = s
        else:
            returned = _new_exit(monkeypatch, exit_name, s)
    notices = [w for w in seen if "worst sigma_max" in str(w.message)]
    assert len(notices) == expected_count
    if notices:
        assert notices[0].category is AccuracyWarning
        message = str(notices[0].message)
        assert "1 of 3 frequency bins" in message
        assert "sigma_max = 1.04" in message and "at 2 GHz" in message
        assert "lower bound from the finite columns" in message
    np.testing.assert_array_equal(returned, before.astype(np.asarray(returned).dtype))
    np.testing.assert_array_equal(s, before)


def _waveguide_scene(path, frequencies):
    from rfx import Simulation

    sim = Simulation(
        freq_max=20e9, domain=(.030, .020, .012), dx=.002,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=2,
        **({"dz_profile": np.full(6, .002)} if path == "graded" else {}),
    )
    for index, freqs in enumerate(frequencies):
        sim.add_waveguide_port(
            .006 if index == 0 else .024, direction="+x" if index == 0 else "-x",
            f0=10e9, freqs=freqs, probe_offset=2, ref_offset=1,
            name=f"port{index}",
        )
    return sim


@pytest.mark.parametrize("case, path, entry", [
    (case, path, entry)
    for case in ("above", "below", "lead-in", "shared")
    for path, entry in (("uniform", "run"), ("graded", "run"),
                        ("uniform", "forward"), ("graded", "forward"))
] + [("lumped", "graded", "run")])
def test_waveguide_returned_s_contract(monkeypatch, path, entry, case):
    import rfx.api._spec as spec
    import rfx.measurement.waveguide_s as waveguide

    if path == "uniform" and entry == "run" and case == "shared":
        pytest.skip("uniform run refuses several waveguide ports at admission")
    frequencies = [np.array([9e9, 10e9, 11e9])]
    above = case in ("above", "lumped")
    reflections = [np.array([1.005, 1.02 if above else 1.005, .5], dtype=complex)]
    transmissions = [np.full(3, .5, dtype=complex)]
    expected_count, worst, bins, diagonal = int(above), 10, "1 of 3", ""
    if case == "lead-in":
        reflections[0][:] = .5j
        transmissions[0][:] = 1.05j
    elif case == "shared":
        frequencies.append(frequencies[0].copy())
        reflections.append(np.array([1.005, 1.04, .5], dtype=complex))
        transmissions.append(np.full(len(frequencies[1]), .5, dtype=complex))
        reflections[0][0] = 1.02
        bins, diagonal = "2 of 3", " (diagonal-only lower bound of sigma_max)"
        expected_count, worst = 1, 10
    before = [(s11.copy(), s21.copy()) for s11, s21 in zip(reflections, transmissions)]
    sim = _waveguide_scene(path, frequencies)
    if case == "lumped":
        from rfx import GaussianPulse

        for index, x in enumerate((.012, .018)):
            sim.add_port((x, .010, .006), "ez", impedance=50.,
                         waveform=GaussianPulse(f0=10e9), excite=index == 0)
    extracted, assembled, fields = [], [], []
    constructor = spec.WaveguideSParamResult

    def extract(cfg, **kwargs):
        index = len(extracted) % len(reflections)
        extracted.append(cfg)
        return reflections[index], transmissions[index]

    def assemble(**kwargs):
        result = constructor(**kwargs)
        assembled.append(result)
        fields.append(tuple(kwargs.values()))
        return result

    monkeypatch.setattr(waveguide, "extract_waveguide_sparams", extract)
    monkeypatch.setattr(spec, "WaveguideSParamResult", assemble)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = getattr(sim, entry)(
            n_steps=1, skip_preflight=True, **({"checkpoint": False} if entry == "forward" else {}),
        )
    notices = [w for w in seen if "waveguide_sparams" in str(w.message)]
    assert len(notices) == expected_count
    if notices:
        assert notices[0].category is AccuracyWarning
        peak = 1.04 if case == "shared" else 1.02
        assert str(notices[0].message) == (
            f"{entry}() waveguide_sparams: the returned S is not passive at {bins} "
            f"frequency bins, worst sigma_max = {peak}{diagonal} at {worst} GHz. "
            "S is returned as extracted. A passive structure cannot scatter more power "
            "than it receives, so these bins are an extraction or record artifact."
        )
    assert list(result.waveguide_sparams) == [f"port{i}" for i in range(len(frequencies))]
    assert result.flux_monitors is None
    for index, record in enumerate(assembled):
        assert result.waveguide_sparams[f"port{index}"] is record
        assert result.waveguide_ports[f"port{index}"] is extracted[index]
        assert record.s11 is reflections[index]
        assert record.s21 is transmissions[index]
        assert all(value is original for value, original in zip(record, fields[index]))
        np.testing.assert_array_equal(record.freqs, np.asarray(frequencies[index], dtype=np.float32))
        np.testing.assert_array_equal(record.s11, before[index][0])
        np.testing.assert_array_equal(record.s21, before[index][1])


@pytest.mark.parametrize("path", ["uniform", "graded"])
def test_waveguide_forward_gradient_unchanged(monkeypatch, path):
    import rfx.measurement.declared as declared
    import rfx.measurement.waveguide_s as waveguide
    from rfx.core.jax_utils import is_tracer

    sim = _waveguide_scene(path, [np.array([9e9, 10e9, 11e9])] * 2)
    grid = sim._build_nonuniform_grid() if path == "graded" else sim._build_grid()
    eps = jnp.ones(grid.shape, dtype=jnp.float32)
    extract = waveguide.extract_waveguide_sparams
    traced = []
    extracted = []
    helper_calls = []
    helper = declared.warn_if_singular_value_exceeds_one

    def capture(cfg, **kwargs):
        s11, s21 = extract(cfg, **kwargs)
        traced.append(is_tracer(s11))
        extracted.append(s11)
        return s11, s21

    def check(*args, **kwargs):
        helper_calls.append(True)
        return helper(*args, **kwargs)

    def loss(eps_override):
        extracted.clear()
        result = sim.forward(eps_override=eps_override, n_steps=6,
                             skip_preflight=True, checkpoint=False)
        for index, name in enumerate(result.waveguide_sparams):
            assert result.waveguide_sparams[name].s11 is extracted[index]
        return sum(jnp.real(jnp.sum(port.s11)) for port in result.waveguide_sparams.values())

    monkeypatch.setattr(waveguide, "extract_waveguide_sparams", capture)
    monkeypatch.setattr(declared, "warn_if_singular_value_exceeds_one", check)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        actual = np.asarray(jax.grad(loss)(eps))
        monkeypatch.setattr(declared, "warn_if_singular_value_exceeds_one", lambda *a, **kw: None)
        expected = np.asarray(jax.grad(loss)(eps))
    assert traced == [True] * 4
    assert not helper_calls
    assert not [w for w in seen if "waveguide_sparams" in str(w.message)]
    assert np.isfinite(actual).all()
    assert np.any(actual != 0)
    np.testing.assert_array_equal(actual, expected)
