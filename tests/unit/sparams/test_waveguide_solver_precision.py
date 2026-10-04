"""Solver and field-storage regressions, not S-parameter accuracy tests.

A two-port 40 x 20 mm guide is sampled at 8 and 9 GHz. These short runs
inspect the six field arrays initialized by the real core for every drive,
including reference runs; the S-matrix's complex dtype is not that check.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
import rfx.nonuniform as nonuniform_core
import rfx.simulation as uniform_core
import rfx.sources.waveguide_port as waveguide_port
from tests._x64_compat import enable_x64


def _guide(*, precision="float32", solver="yee", nonuniform=False, n_modes=1):
    sim = Simulation(
        freq_max=12e9,
        domain=(0.09, 0.04, 0.02),
        dx=0.005,
        boundary="cpml",
        cpml_layers=4,
        precision=precision,
        solver="yee",
        **({"dx_profile": np.full(18, 0.005)} if nonuniform else {}),
    )
    for position, direction, name in (
        (0.015, "+x", "left"), (0.065, "-x", "right"),
    ):
        sim.add_waveguide_port(
            position, direction=direction, n_modes=n_modes,
            freqs=jnp.array([8e9, 9e9], dtype=jnp.float32),
            f0=8.5e9, bandwidth=0.5, name=name,
        )
    # Restore an ADI setting after declaration to test calculator guards;
    # new ADI + CPML models are refused by the constructor.
    sim._solver = solver
    return sim


def _forbid_field_initialization(*args, **kwargs):
    raise AssertionError("Unsupported waveguide input reached field initialization")


@pytest.mark.parametrize("nonuniform", [False, True], ids=["uniform", "nonuniform"])
def test_adi_refused_before_fields(monkeypatch, nonuniform):
    # The constructor already refuses ADI plus explicit profiles. Set the
    # stored solver after construction to exercise the calculator's NU guard.
    sim = _guide(nonuniform=nonuniform, solver="yee" if nonuniform else "adi")
    if nonuniform:
        sim._solver = "adi"
    monkeypatch.setattr(uniform_core, "init_state", _forbid_field_initialization)
    monkeypatch.setattr(nonuniform_core, "init_state", _forbid_field_initialization)
    with pytest.raises(NotImplementedError, match=r"solver='adi'.*#1300.*solver='yee'"):
        sim.compute_waveguide_s_matrix(n_steps=4, normalize=True)


@pytest.mark.parametrize("precision", ["float64", "mixed"])
def test_nonuniform_precision_still_refused_before_fields(monkeypatch, precision):
    monkeypatch.setattr(nonuniform_core, "init_state", _forbid_field_initialization)
    with enable_x64(True), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sim = _guide(nonuniform=True, precision=precision)
        with pytest.raises(NotImplementedError, match="precision other than 'float32'"):
            sim.compute_waveguide_s_matrix(n_steps=4, normalize=True)


def test_preflight_reports_adi_refusal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = _guide(solver="adi").preflight_sparameters(calculator="waveguide")
    assert not report.ok
    routing_errors = [issue for issue in report.errors if issue.code == "sparam_routing_waveguide"]
    assert len(routing_errors) == 1
    message = str(routing_errors[0])
    assert "solver='adi'" in message
    assert "#1300" in message
    assert "Use solver='yee'" in message


@pytest.mark.parametrize(
    "normalize,n_modes,expected_runs",
    [
        pytest.param(False, 1, 2, id="single_raw"),
        pytest.param(True, 1, 4, id="single_normalized"),
        pytest.param("flux", 1, 4, id="single_flux"),
        pytest.param(False, 2, 4, id="multi_raw"),
        pytest.param("flux", 2, 8, id="multi_flux"),
    ],
)
@pytest.mark.parametrize(
    "precision,expected_dtype",
    [("float32", np.float32), ("float64", np.float64), ("mixed", np.float16)],
    ids=["float32", "float64", "mixed"],
)
def test_core_field_dtype(monkeypatch, normalize, n_modes, expected_runs,
                          precision, expected_dtype):
    initialized = []
    initialize = uniform_core.init_state
    run = uniform_core.run
    make_step = uniform_core.make_core_step
    extract_waves = waveguide_port._s_matrix_port_waves
    # With x64 on, even explicit float32 frequencies keep extraction at
    # least as wide as main, independently of the core field-storage dtype.
    expected_record = np.dtype(np.float64)
    expected_complex = np.dtype(np.complex128)

    def observe_run(*args, **kwargs):
        for cfg in kwargs["waveguide_ports"]:
            for name in ("v_probe_t", "v_ref_t", "i_probe_t", "i_ref_t", "v_inc_t"):
                assert getattr(cfg, name).dtype == expected_record, name
        for monitor in kwargs.get("flux_monitors", ()):
            for name in ("e1_dft", "e2_dft", "h1_dft", "h2_dft"):
                assert getattr(monitor, name).dtype == expected_complex, name
        return run(*args, **kwargs)

    def observe_step(ctx):
        step = make_step(ctx)

        def checked_step(*args, **kwargs):
            result = step(*args, **kwargs)
            for accumulators in result[0].get("flux_monitors", ()):
                assert all(acc.dtype == expected_complex for acc in accumulators)
            return result

        return checked_step

    def observe_waves(*args, **kwargs):
        waves = extract_waves(*args, **kwargs)
        assert all(wave.dtype == expected_complex for wave in waves)
        return waves

    def observe_fields(*args, **kwargs):
        state = initialize(*args, **kwargs)
        dtypes = tuple(getattr(state, name).dtype for name in ("ex", "ey", "ez", "hx", "hy", "hz"))
        initialized.append(dtypes)
        assert all(dtype == np.dtype(expected_dtype) for dtype in dtypes), (
            f"core field dtypes for run {len(initialized)}: {dtypes}; "
            f"precision={precision!r} requires {np.dtype(expected_dtype)}"
        )
        return state

    monkeypatch.setattr(uniform_core, "init_state", observe_fields)
    monkeypatch.setattr(uniform_core, "run", observe_run)
    monkeypatch.setattr(uniform_core, "make_core_step", observe_step)
    monkeypatch.setattr(waveguide_port, "_s_matrix_port_waves", observe_waves)
    try:
        with enable_x64(True), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = _guide(precision=precision, n_modes=n_modes).compute_waveguide_s_matrix(
                n_steps=4, normalize=normalize,
            )
            assert result.s_params.shape == (2 * n_modes, 2 * n_modes, 2)
            assert result.s_params.dtype == expected_complex
        assert len(initialized) == expected_runs
    finally:
        jax.clear_caches()
