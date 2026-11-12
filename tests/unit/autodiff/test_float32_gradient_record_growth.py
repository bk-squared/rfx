"""Float32 stepping error in a closed cavity and its absorbing-wall control.

Compare the probe DFT and its forward-mode material derivative against
float64. The DFT is formed on the host so its summation adds no float32 error.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


_N = 20
_DX = 1e-3
_CPML_LAYERS = 8
_T_MAX = 4000
_FREQUENCIES = (16e9, 20e9, 24e9)
_READOUTS = tuple(
    tuple(round(t * 2 ** (m / 4)) for m in range(-4, 1))
    for t in (1000, 4000)
)
_FLOOR = 1e-8


def _position(fractions):
    return tuple(round(fraction * _N) * _DX for fraction in fractions)


def _run_arm(boundary, precision):
    """Return the primal and tangent of the public probe time series."""
    use_x64 = precision == "float64"
    with jax.enable_x64(use_x64), jax.default_device(jax.devices()[0]):
        dtype = jnp.float64 if use_x64 else jnp.float32
        sim = Simulation(
            freq_max=24e9, domain=(_N * _DX,) * 3, dx=_DX,
            boundary=boundary, cpml_layers=_CPML_LAYERS,
            precision=precision,
        )
        grid = sim._build_grid()
        # Grid arrays include the two endpoints: N cells have N + 1 nodes.
        interior = tuple(s.stop - s.start - 1 for s in grid.interior)
        assert interior == (_N,) * 3, (
            f"{boundary}/{precision}: realised interior {interior}, expected {_N} cells"
        )
        expected_pad = _CPML_LAYERS if boundary == "cpml" else 0
        assert grid.face_pads == (expected_pad,) * 6
        assert grid.dx == _DX
        dt = float(grid.dt)

        def waveform(t):
            k = t / dt
            return jnp.where(
                k < 199.5,
                jnp.exp(-((k - 100) / 30) ** 2) * jnp.sin(2 * jnp.pi * 20e9 * t),
                0.0,
            )

        sim.add_source(
            position=_position((0.21, 0.40, 0.62)), component="ez",
            waveform=waveform, amplitude_kind="field",
        )
        sim.add_probe(position=_position((0.79, 0.58, 0.36)), component="ez")
        lo = grid.position_to_index(_position((0.30, 0.34, 0.42)))
        hi = grid.position_to_index(_position((0.55, 0.72, 0.72)))
        # Cell volumes between the declared faces; forward owns edge averaging.
        block = tuple(slice(lower, upper) for lower, upper in zip(lo, hi))
        eps_base = jnp.ones(grid.shape, dtype=dtype)

        def probe_series(p):
            eps = eps_base.at[block].set(2 * p)
            return sim.forward(
                eps_override=eps, n_steps=_T_MAX, checkpoint=False,
                distributed=False, skip_preflight=True,
            ).time_series

        p = jnp.asarray(1.0, dtype=dtype)
        series, tangent = jax.jvp(probe_series, (p,), (jnp.ones_like(p),))
        # Host transfer also waits for the entire solve before pytest times it.
        series, tangent = np.asarray(series), np.asarray(tangent)
        expected_dtype = np.dtype(precision)
        assert series.dtype == expected_dtype, (
            f"{boundary}/{precision}: time_series dtype {series.dtype}"
        )
        assert tangent.dtype == expected_dtype, (
            f"{boundary}/{precision}: tangent dtype {tangent.dtype}"
        )
        assert series.shape == tangent.shape == (_T_MAX, 1)
        print(
            f"{boundary}/{precision}: interior={interior}, pads={grid.face_pads}, "
            f"dt={dt:.17g}, time_series={series.dtype}, tangent={tangent.dtype}",
            file=sys.stderr,
        )
        return series[:, 0].astype(np.float64), tangent[:, 0].astype(np.float64), dt


def _errors(arm32, arm64):
    x32, dx32, dt32 = arm32
    x64, dx64, dt64 = arm64
    assert dt32 == dt64
    # Slot n contains E at (n + 1) dt. Use no rfx DFT helpers here.
    times = (np.arange(_T_MAX, dtype=np.float64) + 1) * dt64
    phase = np.exp(-2j * np.pi * np.asarray(_FREQUENCIES)[:, None] * times)
    errors = []
    for readouts in _READOUTS:
        window_errors = []
        for length in readouts:
            def dft(values):
                return np.sum(phase[:, :length] * values[None, :length], axis=1) * dt64

            d32, d64 = dft(x32), dft(x64)
            t32, t64 = dft(dx32), dft(dx64)
            with np.errstate(divide="ignore", invalid="ignore"):
                e_step = np.maximum(
                    np.max(np.abs(t32 - t64)) / np.max(np.abs(t64)),
                    np.max(np.abs(d32 - d64)) / np.max(np.abs(d64)),
                )
            window_errors.append(e_step)
        # np.maximum preserves NaN so the finite-value bar cannot hide it.
        errors.append(float(np.maximum(np.max(window_errors), _FLOOR)))
    return tuple(errors)


@pytest.mark.slow
def test_float32_gradient_record_growth():
    """PEC error grows at least 2x; the CPML control stays below 1.5x."""
    errors = {}
    for level, boundary in (("PEC", "pec"), ("CPML", "cpml")):
        arm32 = _run_arm(boundary, "float32")
        arm64 = _run_arm(boundary, "float64")
        errors[level] = _errors(arm32, arm64)

    pec_1k, pec_4k = errors["PEC"]
    cpml_1k, cpml_4k = errors["CPML"]
    pec_ratio, cpml_ratio = pec_4k / pec_1k, cpml_4k / cpml_1k
    print(
        f"N={_N}, CPML layers={_CPML_LAYERS}: "
        f"e_PEC(1k)={pec_1k:.12e}, e_PEC(4k)={pec_4k:.12e}, "
        f"e_CPML(1k)={cpml_1k:.12e}, e_CPML(4k)={cpml_4k:.12e}, "
        f"PEC ratio={pec_ratio:.12g}, CPML ratio={cpml_ratio:.12g}",
        file=sys.stderr,
    )
    bars = {
        "PEC grows": pec_4k >= 2 * pec_1k,
        "CPML flat": cpml_4k < 1.5 * cpml_1k,
        "levels apart": pec_4k > cpml_4k and bool(np.all(np.isfinite(
            (pec_1k, pec_4k, cpml_1k, cpml_4k)
        ))),
    }
    for name, passed in bars.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}", file=sys.stderr)
    failed = [name for name, passed in bars.items() if not passed]
    assert not failed, "Failed bars: " + ", ".join(failed)
