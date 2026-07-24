"""Regression tests for the RF-core correctness audit (2026-07-23, Workflow + Codex passes).

Each test pins one confirmed extractor defect so a revert re-reds:
  ①  waveguide _modal_net_power omitted the Yee half-step co-location on the current DFT,
     mixing reactive power into the reported real net power (multimode flux S-matrix).
  #1 (Codex) evanescent waveguide β carried a spurious ×c, making |β| ~c× too large (s⁻¹).
  ⑤⑥ floquet extract/port silently ignored n_modes>1 and TM (TE/(0,0)-hardwired) → fail-loud.
"""
import types

import numpy as np
import pytest

import jax.numpy as jnp

from rfx.sources.waveguide_port import (
    _modal_net_power, _co_located_current_spectrum, _rect_dft, _compute_beta, C0_LOCAL,
)


def test_modal_net_power_applies_halfstep_colocation():
    """① _modal_net_power must co-locate the H-derived current before V·conj(I), so a reactive
    (quadrature) V/I component does NOT leak into the reported real power."""
    dt, f0, n = 3e-12, 6e9, 400
    t = np.arange(n) * dt
    v = np.cos(2 * np.pi * f0 * t)
    i = np.cos(2 * np.pi * f0 * t + np.pi / 3)   # 60° phase ⇒ a real reactive component
    cfg = types.SimpleNamespace(
        v_ref_t=jnp.asarray(v), i_ref_t=jnp.asarray(i),
        freqs=jnp.asarray([f0]), dt=dt, n_steps_recorded=n,
    )
    P = np.asarray(_modal_net_power(cfg))

    v_dft = np.array(_rect_dft(cfg.v_ref_t, cfg.freqs, cfg.dt, cfg.n_steps_recorded))
    i_dft = np.array(_rect_dft(cfg.i_ref_t, cfg.freqs, cfg.dt, cfg.n_steps_recorded))
    i_corr = np.asarray(_co_located_current_spectrum(cfg, i_dft))
    expected = 0.5 * np.real(v_dft * np.conj(i_corr))   # co-located (correct)
    raw = 0.5 * np.real(v_dft * np.conj(i_dft))          # pre-fix (buggy)

    np.testing.assert_allclose(P, expected, rtol=1e-6)
    # discriminating: with a reactive component the co-located power differs materially from the
    # raw product (~10% here). (np.allclose's default atol swamps the ~1e-19 DFT magnitudes, so
    # compare the RELATIVE difference explicitly.)
    rel_diff = float(np.max(np.abs((P - raw) / expected)))
    assert rel_diff > 1e-2, (
        f"co-located net power barely differs from the raw product (rel {rel_diff:.1e}) — the "
        f"half-step correction is not being applied")


def test_evanescent_beta_has_inverse_metre_units():
    """Codex #1: below-cutoff (evanescent) β must be O(k) ~ hundreds of m⁻¹, not ~c× larger.
    A WR-90 TE10 guide (f_c≈6.56 GHz) evaluated at 3 GHz has |β| ≈ 122 m⁻¹; the pre-fix ×C0
    form returned ~3.6e10 (units s⁻¹)."""
    a = 22.86e-3
    f_cutoff = C0_LOCAL / (2 * a)            # WR-90 TE10 cutoff ≈ 6.56 GHz
    freqs = jnp.asarray([3e9])                # below cutoff → evanescent
    beta = np.asarray(_compute_beta(freqs, f_cutoff, dt=1e-12, dx=1e-3))
    b = complex(beta[0])
    assert abs(b.real) < 1e-3, f"evanescent β should be ~pure imaginary, got {b}"
    assert 10.0 < abs(b.imag) < 1.0e4, (
        f"|β_evan| = {abs(b.imag):.3e} m⁻¹ is outside the physical O(10²) range — a spurious "
        f"factor of c (~{C0_LOCAL:.1e}) would put it near 3.6e10")


def test_floquet_extract_rejects_higher_order_modes():
    """⑤ extract_floquet_modes fails loud on n_modes>1 instead of silently returning only (0,0)."""
    from rfx.floquet import init_floquet_dft, extract_floquet_modes
    acc = init_floquet_dft(3, (8, 8))
    with pytest.raises(NotImplementedError, match="specular"):
        extract_floquet_modes(acc, dx=1e-3, Lx=8e-3, Ly=8e-3,
                              freqs=jnp.linspace(5e9, 15e9, 3), n_modes=2)


def test_add_floquet_port_rejects_unimplemented_modes_and_tm():
    """⑥ add_floquet_port fails loud on the silently-wrong n_modes>1 and TM paths."""
    from rfx.api import Simulation

    def _sim():
        return Simulation(freq_max=10e9, domain=(0.015, 0.015, 0.03),
                          boundary="cpml", cpml_layers=8)
    with pytest.raises(NotImplementedError, match="specular"):
        _sim().add_floquet_port(0.008, axis="z", n_modes=3)
    with pytest.raises(NotImplementedError, match="TM"):
        _sim().add_floquet_port(0.008, axis="z", polarization="tm")
    # the supported path still works
    _sim().add_floquet_port(0.008, axis="z", polarization="te", n_modes=1)
