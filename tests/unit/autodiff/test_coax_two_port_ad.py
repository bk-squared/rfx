"""Coax two-port AD gates after the realized TEM source fix (#1356).

The matched lossless thru has constant transmitted power; electrical length
provides the nonzero physical sensitivity. CPU calibration uses float32 field
storage and AD, with scoped float64 DFT/phase arithmetic for the FD referee.

CPU h-sweep (2026-09-30), phase AD = -1.2753862143 rad/eps_scale::

    h       float64 central FD     relative AD/FD error
    5e-4    -1.2754165638           2.37958e-5
    1e-3    -1.2752600614           9.89232e-5
    2e-3    -1.2752724880           8.91780e-5
    5e-3    -1.2753441806           3.29587e-5
    1e-2    -1.2753699495           1.27530e-5

The full measured window is stable to 0.013%; h=2e-3 is interior.
FD loss spans 2.87e12 to 5.74e13 float64 ULPs; field storage remains
float32, so this does not claim float64 FDTD accuracy. The maximum
column-normalized complex path difference is 8.22238e-6 (bar 1e-4).
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from tests._coax_ad_objective import coax_band_mean_s21_sq


def test_coax_band_mean_s21_sq_reduction_is_correct():
    """Unit-check the shared reduction on a synthetic S-matrix: correctness
    is independent of which AD gate exercises it."""
    n_f = 3
    s21 = np.array([0.9 + 0.1j, 0.7 - 0.2j, 0.5 + 0.0j])
    S = np.zeros((2, 2, n_f), dtype=np.complex128)
    S[1, 0, :] = s21
    got = float(coax_band_mean_s21_sq(jnp.asarray(S)))
    want = float(np.mean(np.abs(s21) ** 2))
    # float32-precision tolerance: jnp.asarray on a complex128 array
    # downcasts to complex64 by default (no x64 enabled in this test).
    assert abs(got - want) < 1e-6, (got, want)


def _traced_input_param_names(sig: "inspect.Signature") -> set:
    """Matches the naming convention used by every OTHER traced-input
    parameter in this codebase (compute_coaxial_line_reflection's
    eps_scale, compute_waveguide_s_matrix's eps_override/sigma_override):
    a name starting with "eps_"/"sigma_", or an explicit "eps_scale"-style
    design-variable name. Deliberately narrow (field_scale, a plain float
    source-amplitude knob, must NOT match)."""
    return {
        name for name in sig.parameters
        if name.lower().startswith(("eps_", "sigma_")) or name.lower() == "eps_scale"
    }


def test_traced_input_param_predicate_has_a_positive_control():
    """N7 (review fix, carried forward): the predicate in
    ``_traced_input_param_names`` is a name-prefix heuristic -- without a
    positive control, a typo that makes it match NOTHING would let
    ``test_compute_coaxial_two_port_has_a_traced_input_channel`` below pass
    vacuously forever. Prove the matcher is live: the 1-port sibling's own
    ``eps_scale`` parameter MUST match.
    """
    from rfx.api import Simulation as _Simulation

    sig = inspect.signature(_Simulation.compute_coaxial_line_reflection)
    matched = _traced_input_param_names(sig)
    assert "eps_scale" in matched, (
        "the traced-input-parameter predicate did not match "
        "compute_coaxial_line_reflection's own eps_scale -- the matcher "
        "itself is broken"
    )


def test_compute_coaxial_two_port_has_a_traced_input_channel():
    """FLIPPED (was ``test_compute_coaxial_two_port_has_no_traced_input_
    channel``, pinning the ABSENCE of a channel): ``compute_coaxial_two_
    port`` now has an ``eps_scale`` traced-input parameter, mirroring the
    1-port sibling's own design. If a future change removes it, this test
    (and ``tests/unit/autodiff/test_ad_surface_contract.py``'s classification) must be
    updated in the SAME change, not left stale.
    """
    sig = inspect.signature(Simulation.compute_coaxial_two_port)
    traced_input_names = _traced_input_param_names(sig)
    assert "eps_scale" in traced_input_names, (
        f"compute_coaxial_two_port() lost its eps_scale traced-input "
        f"channel (matched names: {traced_input_names!r}) -- this reverts "
        f"issue #489 leg 3; see this module's docstring for the fix that "
        f"added it."
    )


N_STEPS = 600
FREQ = jnp.asarray([6.0e9, 8.0e9, 10.0e9], dtype=jnp.float32)
PROBE_COUNT = 3
PROBE_START_CELLS = 4
PROBE_SPACING_CELLS = 2
_FD_H = 2.0e-3
# A6000 run 369367266495: AD=-1.2760622501, FD=-1.2760970846, gap=0.002730%; retain the 2% policy.
_REL_ERR_THRESHOLD = 0.02  # Retain the existing AD/FD policy.
_MIN_FD_ULP_SPAN = 1.0e4
assert _MIN_FD_ULP_SPAN * _REL_ERR_THRESHOLD >= 100.0
# The S21 phase itself against the closed form -beta*L: CPU mesh phase errors at
# 6/8/10 GHz 0.0237/0.0287/0.0497 % (0.0206/0.0310/0.0456 % after the declared
# line extent, #1138); repository envelope multiplier 1.5, round up to 0.01 %:
# ceil(0.000497 * 1.5 * 10000) / 10000 = 0.0008 (0.08 %).
# A6000 run 369367266495: 6/8/10 GHz gaps=0.020715/0.030934/0.045568%; CPU envelope still sets the 0.08% bar.
_PHASE_ERROR_BAR = 0.0008
# The AD phase gradient's gap to the closed form -beta*L/(2*eps) depends on
# the record length, so the bar comes from the record-length witness, not
# from the phase error alone. The earlier 0.08 % (1.5x the 0.0497 % mesh phase
# error) was read on this 600-step record only; the same gradient from 1.5x
# and 2x records reads 0.103 % / 0.115 % on the pre-#1138 tree and 0.124 % /
# 0.110 % here (0.105 % at 600 steps), and scaling eps only between the
# reference planes changes it by < 0.003 %. Largest gap 0.1236 %; repository
# envelope multiplier 1.5, round up to 0.01 %:
# ceil(0.001236 * 1.5 * 10000) / 10000 = 0.0019 (0.19 %). A stop_gradient on
# the traced port voltage puts the gradient 7 % off and still fails.
# A6000 run 369367266495: AD=-1.2760622501, closed=-1.2747280763, gap=0.104663%; CPU envelope still sets the 0.19% bar.
_PHASE_GRAD_BAR = 0.0019
# Derived from this fixture: measured |d mean |S21|^2 / d eps_scale|
# = 0.00263521; round up, then multiply by 3. Catch the pre-fix spurious
# power slope from the analytic 1/r drive.
# A6000 run 369367266495: power AD=-0.0003003052843; below the CPU envelope and its 3x bound.
_POWER_GRAD_BOUND = 0.00264 * 3


def _fd_ulp_span(f_plus: float, f_minus: float, dtype) -> float:
    ulp = float(np.spacing(np.asarray(abs(0.5 * (f_plus + f_minus)), dtype=dtype)))
    return abs(f_plus - f_minus) / ulp


def _build_small_two_port_sim() -> Simulation:
    sim = Simulation(domain=(0.008, 0.008, 0.012), freq_max=40.0e9, boundary="cpml")
    sim.add_coaxial_port(
        (0.004, 0.004, 0.006), face="top", pin_length=5.0e-3,
        waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2),
    )
    return sim


def _two_port(eps_scale=None):
    return _build_small_two_port_sim().compute_coaxial_two_port(
        n_steps=N_STEPS, freqs=FREQ, probe_count=PROBE_COUNT,
        probe_start_cells=PROBE_START_CELLS, probe_spacing_cells=PROBE_SPACING_CELLS,
        eps_scale=eps_scale,
    )


def _band_mean_s21_phase(s):
    return jnp.mean(jnp.unwrap(jnp.angle(s[1, 0, :])))


def _band_mean_s21_sq(eps_scale):
    # Retained for the standalone power diagnostic referee.
    return coax_band_mean_s21_sq(_two_port(eps_scale).s_params)


@pytest.fixture(scope="module")
def phase_sensitivity():
    # One forward tape, two pullbacks: the power sanity check reuses the
    # expensive FDTD trace instead of running a separate simulation.
    def objectives(eps):
        s = _two_port(eps).s_params
        return _band_mean_s21_phase(s), coax_band_mean_s21_sq(s)

    values, pullback = jax.vjp(objectives, jnp.float32(1.0))
    phase_grad = float(pullback((jnp.ones_like(values[0]), jnp.zeros_like(values[1])))[0])
    power_grad = float(pullback((jnp.zeros_like(values[0]), jnp.ones_like(values[1])))[0])
    return float(values[0]), phase_grad, power_grad


def _closed_form(result):
    # Returned planes are the top/bottom feed planes, NOT source or probe
    # planes: (nz-pad_z_hi-3-pad_z_lo)*dx and 3*dx respectively.
    # Here z_top=0.0116169577475 m, z_bottom=0.0011242217175 m;
    # L=0.01049273603 m, using the returned values rather than rounded ones.
    from rfx.sources.coaxial_port import PTFE_EPS_R
    from rfx.core.yee import EPS_0, MU_0

    length = abs(float(np.diff(result.reference_planes)[0]))
    beta = 2 * np.pi * np.asarray(FREQ, dtype=np.float64) * np.sqrt(PTFE_EPS_R) * np.sqrt(EPS_0 * MU_0)
    phase = -beta * length
    return phase, float(np.mean(phase / 2.0))  # eps_scale = 1


@pytest.mark.slow_physics
@pytest.mark.highmem
def test_compute_coaxial_two_port_ad_grad_finite_and_fd_consistent(phase_sensitivity):
    """The TEM fix removed spurious loss, making the old power objective flat.
    A relative AD/FD comparison then measured residual noise rather than AD
    accuracy; unwrapped transmission phase has the known nonzero line derivative.
    """
    try:
        from jax import enable_x64
    except ImportError:
        from tests._x64_compat import enable_x64

    value, g_ad, g_power = phase_sensitivity
    assert np.all(np.isfinite([value, g_ad, g_power]))
    with enable_x64():
        fp = _band_mean_s21_phase(_two_port(jnp.float32(1 + _FD_H)).s_params)
        fm = _band_mean_s21_phase(_two_port(jnp.float32(1 - _FD_H)).s_params)
        assert fp.dtype == fm.dtype == jnp.float64
        fp, fm = float(fp), float(fm)
    assert _fd_ulp_span(fp, fm, np.float64) >= _MIN_FD_ULP_SPAN
    g_fd = (fp - fm) / (2 * _FD_H)
    assert np.isfinite(g_fd) and g_fd != 0
    rel = abs(g_ad - g_fd) / abs(g_fd)
    print(f"phase AD={g_ad:.10g}, FD={g_fd:.10g}, relative error={rel:.6g}; power AD={g_power:.10g}")
    assert rel <= _REL_ERR_THRESHOLD
    concrete = _two_port()
    exact_phase, exact_grad = _closed_form(concrete)
    phase_error = np.max(np.abs(np.unwrap(np.angle(concrete.s_params[1, 0])) - exact_phase) / np.abs(exact_phase))
    print(f"planes={concrete.reference_planes}, mesh phase error={phase_error:.6g}, exact gradient={exact_grad:.10g}")
    assert phase_error <= _PHASE_ERROR_BAR
    assert abs(g_ad - exact_grad) / abs(exact_grad) <= _PHASE_GRAD_BAR
    # Matched lossless power is constant; allow three times the measured
    # residual derivative, rather than demanding relative agreement at zero.
    assert abs(g_power) <= _POWER_GRAD_BOUND


@pytest.mark.slow_physics
@pytest.mark.highmem
def test_coax_two_port_eps_scale_unity_matches_concrete_path(phase_sensitivity):
    """The TEM fix drove S22 near zero, where its phase is ill-conditioned.
    The old phase gate amplified tiny complex reassociation errors instead of
    measuring path equivalence; column-normalized complex errors remain meaningful.
    """
    a, b = _two_port(), _two_port(jnp.float32(1.0))
    assert a.status == "passed" and b.status == "differentiable"
    sa, sb = np.asarray(a.s_params), np.asarray(b.s_params)
    assert np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))
    scale = np.max(np.abs(sa), axis=0, keepdims=True)
    assert np.all(scale > 0)
    error = np.abs(sb - sa) / scale
    print(f"column-normalized complex error: {error}; maximum={np.max(error):.10g}")
    # Measured maximum column-normalized complex difference: 8.22238e-6.
    # A6000 run 369367266495: maximum=1.0694297116e-5, below the unchanged 1e-4 bar.
    assert np.all(error <= 1.0e-4)
    # Equal forward values alone cannot detect stop_gradient. Reuse the
    # phase tangent witness to require the differentiable path to stay live.
    _, exact_grad = _closed_form(a)
    assert abs(phase_sensitivity[1] - exact_grad) / abs(exact_grad) <= _PHASE_GRAD_BAR
