"""Issue #980: ``subtract_flux_monitors`` is the public two-run reference
subtraction for flux monitors.

The recipe (``docs/agent/recipe-rt-measurement.mdx``, critical detail (a))
requires the incident field to be removed from the DFT ACCUMULATORS, not from
the computed fluxes: Poynting flux is bilinear in E and H, so
``flux(sample) - flux(ref)`` keeps the incident/scattered cross terms and is a
different quantity from the scattered flux. Callers used to spell that out with
``_replace`` over four private field names; this helper owns it.

These tests pin (1) bit-identity with the hand-written ``_replace``, (2) that
the helper and the flux difference really do disagree when the cross terms are
nonzero, (3) the shape/metadata contract, (4) tracer-safety under ``jax.grad``,
and (5) the export path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx
from rfx.probes.probes import (
    _FLUX_METADATA_FIELDS,
    FluxMonitor,
    flux_spectrum,
    subtract_flux_monitors,
)


def _monitor(e1, h2, *, n_freqs: int = 2, n1: int = 2, n2: int = 2, **overrides) -> FluxMonitor:
    """A hand-built monitor with only the (e1, h2) Poynting pair populated."""
    shape = (n_freqs, n1, n2)
    c64 = jnp.complex64
    mon = FluxMonitor(
        e1_dft=jnp.full(shape, e1, dtype=c64),
        e2_dft=jnp.zeros(shape, dtype=c64),
        h1_dft=jnp.zeros(shape, dtype=c64),
        h2_dft=jnp.full(shape, h2, dtype=c64),
        freqs=jnp.linspace(1e9, 2e9, n_freqs),
        axis=0, index=7,
        dA=jnp.asarray(1.0, dtype=jnp.float32),
        total_steps=100, window="rect", window_alpha=0.25,
        lo1=0, hi1=n1, lo2=0, hi2=n2,
    )
    return mon._replace(**overrides) if overrides else mon


def test_matches_hand_written_replace_bit_for_bit():
    """The helper must be exactly the expression it replaces in the docs."""
    sample = _monitor(3 + 1j, 5 - 2j)
    ref = _monitor(1 + 0.5j, 2 + 0.5j)

    expected = sample._replace(
        e1_dft=sample.e1_dft - ref.e1_dft,
        e2_dft=sample.e2_dft - ref.e2_dft,
        h1_dft=sample.h1_dft - ref.h1_dft,
        h2_dft=sample.h2_dft - ref.h2_dft,
    )
    got = subtract_flux_monitors(sample, ref)

    assert isinstance(got, FluxMonitor)
    for field in ("e1_dft", "e2_dft", "h1_dft", "h2_dft"):
        assert np.array_equal(
            np.asarray(getattr(got, field)), np.asarray(getattr(expected, field))
        ), f"{field} is not bit-identical to the hand-written _replace"
    for field in _FLUX_METADATA_FIELDS:
        assert np.array_equal(
            np.asarray(getattr(got, field)), np.asarray(getattr(sample, field))
        ), f"{field} must be carried over from sample unchanged"


def test_field_subtraction_differs_from_flux_subtraction():
    """The whole point: the cross terms do not cancel in ``F_s - F_r``.

    With only (e1, h2) populated and dA = 1 over 2x2 cells,
    ``flux = 4 * Re(e1 * conj(h2))``:

        sample      Re((3+1j)(5+2j))     = 13   -> 52
        ref         Re((1+0.5j)(2-0.5j)) =  2.25 ->  9
        difference                              -> 43
        field-level Re((2+0.5j)(3+2.5j)) =  4.75 -> 19

    43 is the scattered flux PLUS 24 of incident/scattered cross terms; 19 is
    the scattered flux. A helper that returned the former would be wrong.
    """
    sample = _monitor(3 + 1j, 5 - 2j)
    ref = _monitor(1 + 0.5j, 2 + 0.5j)

    field_level = np.asarray(flux_spectrum(subtract_flux_monitors(sample, ref)))
    flux_difference = np.asarray(flux_spectrum(sample)) - np.asarray(flux_spectrum(ref))

    np.testing.assert_allclose(field_level, np.full(2, 19.0), rtol=1e-5)
    np.testing.assert_allclose(flux_difference, np.full(2, 43.0), rtol=1e-5)
    assert not np.allclose(field_level, flux_difference), (
        "constructed case has vanishing cross terms — it cannot witness the bug"
    )


@pytest.mark.parametrize("field", ["e1_dft", "e2_dft", "h1_dft", "h2_dft"])
def test_shape_mismatch_raises_value_error(field):
    sample = _monitor(1 + 0j, 1 + 0j)
    ref = _monitor(1 + 0j, 1 + 0j)
    bad = getattr(ref, field)[:, :1, :]
    ref = ref._replace(**{field: bad})

    with pytest.raises(ValueError, match=rf"FluxMonitor\.{field} shape mismatch"):
        subtract_flux_monitors(sample, ref)


@pytest.mark.parametrize("field", _FLUX_METADATA_FIELDS)
def test_every_metadata_mismatch_raises_value_error(field):
    """Covers the full metadata field list, so a new field cannot slip the gate."""
    sample = _monitor(1 + 0j, 1 + 0j)
    value = getattr(sample, field)
    if isinstance(value, str):
        perturbed = "hann" if value != "hann" else "rect"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        perturbed = type(value)(value + 1)
    else:
        # Arrays: scale, do not offset. ``freqs + 1`` is a no-op in float32 at
        # 1e9 (spacing ~64 Hz there), which would make this test vacuous.
        perturbed = value * 2 + 1
    ref = sample._replace(**{field: perturbed})

    with pytest.raises(ValueError, match=rf"FluxMonitor\.{field} differs"):
        subtract_flux_monitors(sample, ref)


def test_frequency_mismatch_message_names_the_plane_contract():
    sample = _monitor(1 + 0j, 1 + 0j)
    ref = sample._replace(freqs=jnp.linspace(3e9, 4e9, 2))
    with pytest.raises(ValueError, match="same plane / frequencies"):
        subtract_flux_monitors(sample, ref)


def test_non_monitor_argument_raises_type_error():
    sample = _monitor(1 + 0j, 1 + 0j)
    with pytest.raises(TypeError, match="expects two FluxMonitor instances"):
        subtract_flux_monitors(sample, sample.e1_dft)


def test_grad_through_subtracted_flux_is_finite():
    """Tracer-safety: the helper stays on the AD tape.

    ``loss(s) = flux(subtract(sample with e1 scaled by s, ref))``
    ``        = 4 * Re((3s - 1) * conj(5 - 2)) = 36 s - 12``, so dloss/ds = 36.
    """
    sample = _monitor(3 + 0j, 5 + 0j)
    ref = _monitor(1 + 0j, 2 + 0j)

    def loss(scale):
        scaled = sample._replace(e1_dft=sample.e1_dft * scale)
        return flux_spectrum(subtract_flux_monitors(scaled, ref))[0]

    value = loss(2.0)
    grad = jax.grad(loss)(2.0)

    assert np.isfinite(float(value)) and np.isfinite(float(grad))
    np.testing.assert_allclose(float(value), 60.0, rtol=1e-5)
    np.testing.assert_allclose(float(grad), 36.0, rtol=1e-5)


def test_jit_of_subtracted_flux_matches_eager():
    """Under jit the array leaves are tracers; metadata checks must not crash."""
    sample = _monitor(3 + 1j, 5 - 2j)
    ref = _monitor(1 + 0.5j, 2 + 0.5j)

    def f(e1_sample, e1_ref):
        return flux_spectrum(
            subtract_flux_monitors(
                sample._replace(e1_dft=e1_sample), ref._replace(e1_dft=e1_ref)
            )
        )

    jitted = np.asarray(jax.jit(f)(sample.e1_dft, ref.e1_dft))
    eager = np.asarray(flux_spectrum(subtract_flux_monitors(sample, ref)))
    np.testing.assert_array_equal(jitted, eager)


def test_exported_alongside_flux_spectrum():
    """Same import surface as ``flux_spectrum`` — top-level ``rfx`` and the module."""
    from rfx import flux_spectrum as top_flux_spectrum  # noqa: F401
    from rfx import subtract_flux_monitors as top_subtract

    assert top_subtract is subtract_flux_monitors
    assert "flux_spectrum" in rfx.__all__
    assert "subtract_flux_monitors" in rfx.__all__
