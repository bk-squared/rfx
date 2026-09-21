"""One function: the S a caller measures IS the S a gradient differentiates.

``compute_msl_s_matrix`` reaches its assembly through two runners — ``run()``
on the plain channel, ``forward()`` on the ``eps_override`` channel — and the
passivity projection can only be applied on the first. It refuses to run
under tracing (clipping singular values at 1 zeroes the objective gradient
wherever the clip bites) and on the ``eps_override`` channel even when
concrete (otherwise a finite-difference cross-check and ``jax.grad`` compare
two different functions, PR #468). While the projection was the DEFAULT, that
exemption was the defect: the returned ``S`` was one function on the channel a
user reads and another on the channel a gradient takes, wherever any bin was
non-passive.

The default is now ``enforce_passivity=False`` (PI, 2026-09-21 — the contract
is ``docs/design_notes/chain_closure_contract.md``), so both channels return
the raw extraction. These tests pin that identity.
"""

from __future__ import annotations

import re
import warnings
from types import MethodType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.geometry.csg import Box
from rfx.probes.probes import DFTPlaneProbe
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

# ---------------------------------------------------------------------------
# Planted-S fixture. Same manufacturing technique as
# tests/unit/autodiff/test_msl_sparam_ad.py::_run_manufactured_assembly —
# plant B = S·A for a well-conditioned drive matrix A, hand the extractor
# V = A + B and I = (A - B)/Z_ref — with two differences this test needs:
# the planted S is scaled until it is comfortably NON-passive (so that
# switching the projection on moves S by an amount no tolerance can absorb),
# and the same synthetic records are served to BOTH run() and forward(), so
# the only thing that can differ between the two channels is what
# compute_msl_s_matrix does with them after assembly.
#
# No FDTD runs here. That is deliberate: the identity under test is a property
# of the post-processing branch, and a fixture whose two channels are fed
# byte-identical records is the one that isolates it.
# ---------------------------------------------------------------------------

U = 2.0**-12
FREQS = np.array([0.8e9, 1.2e9, 1.6e9])
_NAME_RE = re.compile(
    r"_msl_run(?P<run>\d+)_p(?P<port>\d+)_(?:ez\d+|hy|hz)(?:_left)?")


def _planted_s(scale: float) -> np.ndarray:
    """A non-diagonal, frequency-varying two-port S, scaled as a whole."""
    base = np.array([[0.2 + 0.05j, 0.55 - 0.07j],
                     [0.65 + 0.1j, -0.15 + 0.03j]])[:, :, None]
    return scale * base * np.array([1.0, 0.9 + 0.1j, 0.8 - 0.15j])


def _build_planted_sim(scale: float):
    """A two-port MSL sim whose run() and forward() both replay planted V/I."""
    expected = _planted_s(scale)
    a = np.array([[1.0 + 0.1j, 0.2 - 0.05j], [-0.1 + 0.04j, 0.9 - 0.1j]])
    assert np.linalg.cond(a) < 2, "the drive system must stay well conditioned"
    b = np.stack([expected[:, :, k] @ a for k in range(len(FREQS))], axis=-1)
    z_ref = float(hammerstad_jensen_z0_eps_eff(4 * U, 2 * U, 2.0)[0])
    voltage, current = a[:, :, None] + b, (a[:, :, None] - b) / z_ref

    def fake_scan(self, **kwargs):
        """Stand in for BOTH run() and forward(); ignores every kwarg.

        Ignoring ``eps_override`` is the point: with identical records on
        both channels, any difference in the returned S is post-processing.
        """
        grid = self._build_grid()
        planes = {}
        for entry in self._dft_planes:
            match = _NAME_RE.fullmatch(entry.name)
            assert match is not None, f"unexpected probe name {entry.name!r}"
            driven, port = int(match["run"]), int(match["port"])
            index = grid.position_to_index((entry.coordinate, 0, 0))[0]
            region = self._dft_plane_regions[entry.name]
            w_lo, w_hi, z_lo, z_hi = region
            shape = (w_hi - w_lo, z_hi - z_lo)
            if entry.component == "ez":
                field = np.ones(shape) / (2 * U)
                amplitude = voltage[port, driven]
            else:
                # Affine longitudinal variation, so recovering I needs both
                # bracketing H samples; transverse ramps give the Ampere
                # contour a nonzero, analytically known value.
                x_h = (index - grid.pad_x_lo + 0.5) * U
                target = (12 if port == 0 else 28) * U
                factor = 1 + (0.5 + 0.25j) * (x_h - target) / U
                if entry.component == "hy":
                    z_h = np.arange(z_lo, z_hi) - grid.pad_z_lo + 0.5
                    profile = np.broadcast_to(z_h[None, :], shape)
                else:
                    y_h = np.arange(w_lo, w_hi) - grid.pad_y_lo + 0.5
                    profile = np.broadcast_to(0.25 * y_h[:, None], shape)
                sign = 1 if port == 0 else -1
                field = sign * factor * profile / (0.75 * 5 * U)
                amplitude = current[port, driven] * np.exp(
                    -1j * np.pi * FREQS * grid.dt)
            planes[entry.name] = DFTPlaneProbe(
                accumulator=(
                    jnp.asarray(field, dtype=jnp.complex64)[None, :, :]
                    * jnp.asarray(amplitude, dtype=jnp.complex64)[:, None, None]
                ),
                freqs=entry.freqs, component=entry.component, axis=0,
                index=index, total_steps=1, window="rect", window_alpha=0.25,
                region=region,
            )
        return SimpleNamespace(dft_planes=planes, time_series=None)

    sim = Simulation(freq_max=20e9, domain=(40 * U, 12 * U, 8 * U),
                     dx=U, cpml_layers=2, boundary="cpml")
    sim.add_material("substrate", eps_r=2.0)
    sim.add(Box((0, 0, 0), (40 * U, 12 * U, 2 * U)), material="substrate")
    sim.add(Box((0, 0, 0), (40 * U, 12 * U, 0)), material="pec")
    sim.add(Box((0, 4 * U, 2 * U), (40 * U, 8 * U, 2 * U)), material="pec")
    for feed, direction in ((2 * U, "+x"), (38 * U, "-x")):
        sim.add_msl_port(position=(feed, 6 * U, 0), width=4 * U, height=2 * U,
                         direction=direction, mode="uniform", eps_r_sub=2.0,
                         n_probe_offset=10, n_probe_spacing=2, n_probes=3)
    sim.run = MethodType(fake_scan, sim)
    sim.forward = MethodType(fake_scan, sim)
    return sim, expected


def _noop_eps_override(sim):
    """The simulation's OWN eps array — an override that overrides nothing.

    Assembled the way ``forward()`` assembles it (``_assemble_materials`` on
    the built grid), so passing it back changes no cell.
    """
    grid = sim._build_grid()
    materials = sim._assemble_materials(
        grid, sheet_specs=[], pec_sheets=[], pec_wires=[])[0]
    return jnp.asarray(materials.eps_r)


def _sigma_max(S) -> np.ndarray:
    S = np.asarray(S, dtype=np.complex128)
    return np.array([np.linalg.svd(S[:, :, k], compute_uv=False)[0]
                     for k in range(S.shape[2])])


def _compute(sim, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.compute_msl_s_matrix(
            n_steps=1, freqs=FREQS, num_periods=1.0, **kwargs)


# ---------------------------------------------------------------------------
# Criterion 1(2): one function across the channels.
# ---------------------------------------------------------------------------

# Scaled so the planted S is non-passive at every bin: sigma_max runs
# 2.0705 / 1.8749 / 1.6853, i.e. an excess of 1.07 / 0.87 / 0.69. That size is
# what makes the mutation below detectable — see this file's mutation note.
_NONPASSIVE_SCALE = 3.0


def test_eps_override_channel_returns_the_same_s_as_the_plain_call():
    """The no-op override must return the plain call's S, entry for entry.

    MEASURED (this fixture, CPU float32): the two channels agree EXACTLY,
    max|dS| = 0.0 over all 12 complex entries. The assertion keeps the
    contract's rtol=1e-5 / atol=1e-7 rather than demanding bit equality,
    because the contract is "one function", not "one bit pattern".

    MUTATION EVIDENCE (both variants run, both recorded in the PR body):
      (a) the assertion is live at the tolerance it states — scaling one
          channel by 1 + 1e-4 with the code untouched turns it RED,
          "Mismatched elements: 12 / 12 (100%), max absolute difference
          1.97e-04".
      (b) the defect itself: restore ``enforce_passivity: bool = True`` in
          rfx/sparams/msl.py, change nothing else, and the plain call is
          projected while the override call is not — RED with "Mismatched
          elements: 12 / 12 (100%), max absolute difference 1.0141459".
    Both channels are fed byte-identical synthetic records here, so (b)
    cannot be passed by a fixture that happens to be passive.
    """
    sim, planted = _build_planted_sim(_NONPASSIVE_SCALE)
    plain = _compute(sim)
    override = _compute(
        _build_planted_sim(_NONPASSIVE_SCALE)[0],
        eps_override=_noop_eps_override(sim),
    )

    # The fixture is only evidence if its raw S is genuinely non-passive:
    # on a passive one the projection is a no-op and the identity is free.
    assert float(_sigma_max(planted).min()) > 1.05, (
        "the planted S must be non-passive at every bin, or switching the "
        "projection on would move nothing and this test would gate nothing")

    np.testing.assert_allclose(np.asarray(override.S), np.asarray(plain.S),
                               rtol=1e-5, atol=1e-7)
    # Neither channel may quietly keep a projected copy around.
    assert plain.S_raw is None and plain.passivity_correction is None
    assert override.S_raw is None and override.passivity_correction is None


def test_the_returned_s_is_the_planted_one_on_both_channels():
    """Independent oracle: neither channel's S is merely equal to the other.

    Two channels agreeing on a wrong matrix would satisfy the identity test
    above. The planted S is built outside the extractor, so this pins WHICH
    function they agree on: the raw extraction.
    """
    sim, planted = _build_planted_sim(_NONPASSIVE_SCALE)
    plain = _compute(sim)
    override = _compute(
        _build_planted_sim(_NONPASSIVE_SCALE)[0],
        eps_override=_noop_eps_override(sim),
    )
    for label, result in (("plain", plain), ("eps_override", override)):
        np.testing.assert_allclose(
            np.asarray(result.S), planted, rtol=0.0, atol=1e-5,
            err_msg=f"{label} channel did not return the planted S")


def test_passivity_excess_measures_the_returned_matrix_on_both_channels():
    """The bound violation stays measurable without projecting it away.

    ``passivity_excess`` is computed from the singular values of the RAW S,
    and the planted matrix's own singular values are the oracle — this does
    not re-run the extractor's helper against itself.
    """
    sim, planted = _build_planted_sim(_NONPASSIVE_SCALE)
    expected = np.maximum(_sigma_max(planted) - 1.0, 0.0)
    for kwargs in ({}, {"eps_override": _noop_eps_override(sim)}):
        result = _compute(_build_planted_sim(_NONPASSIVE_SCALE)[0], **kwargs)
        assert result.passivity_excess is not None
        np.testing.assert_allclose(np.asarray(result.passivity_excess),
                                   expected, rtol=1e-4, atol=1e-6)


def test_a_passive_extraction_reports_zero_excess_and_says_nothing():
    """No warning, and no excess, when the extraction respects the bound."""
    sim, planted = _build_planted_sim(0.8)
    assert float(_sigma_max(planted).max()) < 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.compute_msl_s_matrix(n_steps=1, freqs=FREQS,
                                          num_periods=1.0)
    assert np.all(np.asarray(result.passivity_excess) == 0.0)
    assert not [str(w.message) for w in caught
                if "is not passive" in str(w.message)]


def test_a_nonpassive_extraction_is_named_bin_by_bin_in_one_warning():
    """Loud without projecting: count, worst sigma_max, and the opt-in."""
    sim, planted = _build_planted_sim(_NONPASSIVE_SCALE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.compute_msl_s_matrix(n_steps=1, freqs=FREQS, num_periods=1.0)
    messages = [str(w.message) for w in caught if "is not passive" in str(w.message)]
    assert len(messages) == 1, "one aggregate warning, not one per bin"
    message = messages[0]
    assert "3 of 3 frequency bins" in message
    assert f"worst sigma_max = {float(_sigma_max(planted).max()):.3f}" in message
    assert "returned exactly as extracted" in message
    assert "enforce_passivity=True" in message
    # The projection warning is the other branch's; it must not also fire.
    # Match its own opening, not the phrase "projected onto the passive set"
    # — this warning names that projection as the remedy and contains it too.
    assert not [m for m in [str(w.message) for w in caught]
                if m.startswith("S-matrix projected onto the passive set")]


def test_enforce_passivity_true_still_projects_and_keeps_the_raw():
    """The opt-in is unchanged: bounded S, raw kept, clip recorded, warned."""
    sim, planted = _build_planted_sim(_NONPASSIVE_SCALE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.compute_msl_s_matrix(n_steps=1, freqs=FREQS,
                                          num_periods=1.0,
                                          enforce_passivity=True)
    assert np.all(_sigma_max(result.S) <= 1.0)
    assert result.S_raw is not None
    np.testing.assert_allclose(np.asarray(result.S_raw), planted,
                               rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result.passivity_correction),
                               np.maximum(_sigma_max(planted) - 1.0, 0.0),
                               rtol=1e-4, atol=1e-6)
    assert any("projected onto the passive set" in str(w.message)
               for w in caught)


def test_enforce_passivity_true_is_still_skipped_under_eps_override():
    """The AD exemption survives the default change, flag or no flag.

    Asking for the projection on the differentiable channel must still not
    get it — that is what keeps a finite-difference cross-check and
    ``jax.grad`` on the same function (PR #468).
    """
    sim, planted = _build_planted_sim(_NONPASSIVE_SCALE)
    result = _compute(_build_planted_sim(_NONPASSIVE_SCALE)[0],
                      eps_override=_noop_eps_override(sim),
                      enforce_passivity=True)
    assert result.S_raw is None and result.passivity_correction is None
    assert float(_sigma_max(result.S).max()) > 1.05
    np.testing.assert_allclose(np.asarray(result.S), planted,
                               rtol=0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# The same identity end to end, through the two REAL runners.
# ---------------------------------------------------------------------------

def _live_thru() -> Simulation:
    """The tiny truncated thru of tests/unit/sparams/test_msl_passivity_enforcement."""
    sim = Simulation(freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    sim.add(Box((0., 0., 0.), (0.012, 0.008, 0.)), material="pec")
    sim.add(Box((0.0, 0.0034, 0.0008), (0.012, 0.0046, 0.0008)), material="pec")
    sim.add_msl_port(position=(0.002, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


@pytest.mark.slow
def test_live_thru_eps_override_matches_the_plain_call():
    """Two runners, one S: ``run()`` and ``forward()`` on the same board.

    SCOPE, stated because it is narrower than the test above. The fast tests
    feed both channels identical records and so isolate the post-processing.
    This one solves twice — the plain call through ``run_uniform``, the
    override call through the ``forward()`` lane — and asks whether the two
    RUNNERS land on the same S once the post-processing no longer differs.

    MEASURED (CPU float32, freqs 2-18 GHz x 4, num_periods=2.0): max|dS| =
    6.664e-08 across the 16 complex entries, no entry outside rtol=1e-5 /
    atol=1e-7. That residual is the two lanes' float32 arithmetic, not the
    projection.

    WHAT THIS TEST CANNOT SEE, measured rather than assumed: restoring
    ``enforce_passivity: bool = True`` moves the plain call's S here by only
    7.630e-06, because this board's raw sigma_max is 1.000000003 to
    1.000000119 — float32 noise around a passive |S21| = 1 thru, not a
    physics artifact — so what the projection removes is dominated by
    ``_project_passive``'s own 64-ULP clipping margin. That is INSIDE the
    tolerance above, so the mutation leaves this test green. The gate on the
    projection is
    ``test_eps_override_channel_returns_the_same_s_as_the_plain_call``; this
    test gates lane agreement and nothing else.
    """
    freqs = jnp.linspace(2e9, 18e9, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = _live_thru().compute_msl_s_matrix(freqs=freqs, num_periods=2.0)
        sim = _live_thru()
        override = sim.compute_msl_s_matrix(
            freqs=freqs, num_periods=2.0,
            eps_override=_noop_eps_override(sim))

    assert plain.S_raw is None, "the plain call must return the raw extraction"
    np.testing.assert_allclose(np.asarray(override.S), np.asarray(plain.S),
                               rtol=1e-5, atol=1e-7)
