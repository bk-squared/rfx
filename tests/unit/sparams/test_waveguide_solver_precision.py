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
from tests._x64_compat import enable_x64


def _guide(*, precision="float32", solver="yee", nonuniform=False, n_modes=1):
    sim = Simulation(
        freq_max=12e9,
        domain=(0.09, 0.04, 0.02),
        dx=0.005,
        boundary="cpml",
        cpml_layers=4,
        precision=precision,
        solver=solver,
        **({"dx_profile": np.full(18, 0.005)} if nonuniform else {}),
    )
    for position, direction, name in (
        (0.015, "+x", "left"), (0.065, "-x", "right"),
    ):
        sim.add_waveguide_port(
            position, direction=direction, n_modes=n_modes,
            freqs=jnp.array([8e9, 9e9]), f0=8.5e9, bandwidth=0.5, name=name,
        )
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
    try:
        with enable_x64(True), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = _guide(precision=precision, n_modes=n_modes).compute_waveguide_s_matrix(
                n_steps=4, normalize=normalize,
            )
            assert result.s_params.shape == (2 * n_modes, 2 * n_modes, 2)
        assert len(initialized) == expected_runs
    finally:
        jax.clear_caches()
