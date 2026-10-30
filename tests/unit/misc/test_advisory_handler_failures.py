"""Advisories skip traced values, expose defects, and record user waveform failures."""
from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api._result_health import _nonfinite_fields
from rfx.errors import AccuracyWarning
from rfx.measurement.decay import DCNotEvaluated, dc_census, warn_dc_floor
from rfx.probes.probes import _warn_if_flux_subnormal_flush, warn_if_nonpassive_lumped_s11
from rfx.sources.sources import CustomWaveform
from rfx.sparams._common import _warn_if_nonpassive_smatrix


def test_advisory_exception_boundaries():
    class BrokenArray:
        def __array__(self, *args, **kwargs):
            raise TypeError("array conversion defect")

    broken = BrokenArray()
    result = SimpleNamespace(time_series=broken, s_params=None)
    with pytest.raises(TypeError, match="array conversion defect"):
        _nonfinite_fields(result)
    with pytest.raises(TypeError, match="array conversion defect"):
        warn_if_nonpassive_lumped_s11(broken, np.ones(1), extractor="test")
    with pytest.raises(TypeError, match="array conversion defect"):
        _warn_if_nonpassive_smatrix(
            SimpleNamespace(s_params=broken, freqs=np.ones(1)), extractor="test")
    with pytest.raises(TypeError):
        warn_if_nonpassive_lumped_s11(np.array([2.]), np.ones((1, 2)), extractor="test")

    # Malformed concrete metadata must not turn a non-finite S into a clean result.
    result = SimpleNamespace(time_series=None, s_params=np.full((1, 1, 1), np.nan),
                             grid=SimpleNamespace(dx_arr=np.ones(1)),
                             wire_port_sparams=(((), ()),))
    with pytest.raises(IndexError):
        _nonfinite_fields(result)

    def traced(x):
        _warn_if_flux_subnormal_flush(None, x)
        warn_if_nonpassive_lumped_s11(x, x, extractor="test")
        # Concrete S with traced frequencies also skips host materialization.
        warn_if_nonpassive_lumped_s11(np.ones(1), x, extractor="test")
        for s in (x, np.ones((1, 1, 1))):
            _warn_if_nonpassive_smatrix(SimpleNamespace(s_params=s, freqs=x), extractor="test")
        assert _nonfinite_fields(SimpleNamespace(time_series=x, s_params=None)) == []
        # A nested tracer reaches numpy's __array__ conversion, unlike a direct tracer.
        assert _nonfinite_fields(SimpleNamespace(time_series=[x], s_params=None)) == []
        return jnp.sum(x)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(jax.grad(traced)(jnp.ones(1)), np.ones(1))

    def waveform_error(t):
        raise RuntimeError("user waveform failed")

    port = SimpleNamespace(impedance=0., excite=True, position=(0., 0., 0.),
                           component="ez", waveform=CustomWaveform(waveform_error))
    good = SimpleNamespace(impedance=0., excite=True, position=(1., 0., 0.),
                           component="ez", waveform=CustomWaveform(lambda t: jnp.ones_like(t)))
    sim = SimpleNamespace(_ports=[port, good], _msl_ports=[])
    rows = dc_census(sim, dt=1., n_table=4)
    assert rows[0] == DCNotEvaluated("(0.0, 0.0, 0.0) (ez)", "CustomWaveform",
                                    "RuntimeError: user waveform failed")
    assert rows[1][2:] == (1., 4., None)
    with pytest.warns(AccuracyWarning) as caught:
        warn_dc_floor(sim, dt=1., n_table=4)
    assert any("DC not evaluated: RuntimeError: user waveform failed" in str(w.message)
               for w in caught)
    assert any("relative DC 1.00e+00" in str(w.message) for w in caught)
    # MSL ports use the same census, even without an impedance field.
    sim = SimpleNamespace(_ports=[], _msl_ports=[port])
    assert dc_census(sim, dt=1., n_table=4) == rows[:1]
    for signal in (TimeoutError, KeyboardInterrupt, SystemExit):
        def interrupted(t):
            raise signal("cancel census")
        port.waveform = CustomWaveform(interrupted)
        with pytest.raises(signal, match="cancel census"):
            dc_census(sim, dt=1., n_table=4)

    # A failed drive must also remain visible in the post-run census consumer.
    from rfx.core.yee import init_materials, init_state
    from rfx.grid import Grid
    from rfx.measurement.decay import static_remnant_advisory

    grid = Grid(freq_max=4e9, domain=(.02, .02, .02), dx=.002, cpml_layers=4)
    state = init_state(grid.shape)
    state = state._replace(ez=state.ez.at[grid.nx // 2, grid.ny // 2, grid.nz // 2].set(1.))
    with pytest.warns(AccuracyWarning, match="DC not evaluated: RuntimeError: user waveform failed"):
        static_remnant_advisory(state, init_materials(grid.shape), grid, rows[:1])
