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


def test_tracers_and_unavailable_values_are_skipped():
    class Unavailable:
        def __array__(self, *args, **kwargs):
            raise ValueError("no host value")

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        for s in (Unavailable(), None, np.empty((0,)), np.full((2, 2, 3), np.nan)):
            warn_if_singular_value_exceeds_one(s, FREQS, extractor="unavailable")

        def traced(s):
            warn_if_singular_value_exceeds_one(s, FREQS, extractor="traced")
            return jnp.sum(s)

        np.testing.assert_allclose(jax.grad(traced)(jnp.ones(3)), np.ones(3), atol=1e-7)
    assert not seen
