"""Soft-source deposited-DC guards (issues #386 / #388).

Measured mechanism (issue #388, 2026-07-19, both uniform and NU lanes):
a soft current source deposits its waveform's discrete time integral as a
net static end-charge ``q = S/((1+loss)*dz_src)``, ``S = sum s(t_n)*dt``.
The remnant's discrete-Poisson self-energy neither decays nor is absorbed
by CPML, so it floors the ``until_decay`` interior-energy criterion.

Locks, in order:
1. ``GaussianPulse``'s new ``cutoff`` parameter is byte-identical to the
   historical waveform at the default ``cutoff=3.0`` (formula replica
   pinned from main @ 19a8063).
2. ``cutoff=4.5`` reduces ``|sum s*dt|`` by the predicted factor
   ``e^-(4.5^2 - 3^2)`` within 10% (scoped x64 — the target sum is ~1e-19
   and float32 accumulation noise would swamp it; NO module-level x64).
3. The run()-path ``until_decay`` DC-floor warning fires exactly per the
   predeclared four-case matrix (issue #388 measured rel_DC values at the
   demo dt = 1.2 ps: 6.0e-5 / ~e-10 / 0.68 / 4.0e-4 vs threshold 1e-3).
4. The ``unresolved_pulse`` preflight advisory (issue #386) fires on the
   absolute-Hz-bandwidth footgun and stays silent on the demo-class
   fractional-bandwidth config.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation, GaussianPulse, ModulatedGaussian

DEMO_DT = 1.2e-12  # the issue-#388 measurement dt (patch_antenna_demo scale)


# ---------------------------------------------------------------------------
# 1. cutoff default byte-identity
# ---------------------------------------------------------------------------

def _main_gaussian_pulse(t, f0, bandwidth, amplitude=1.0):
    """Replica of main@19a8063 GaussianPulse.__call__, op-for-op.

    tau = 1/(f0*bandwidth*pi); t0 = 3.0*tau (the pre-cutoff hardcoded
    value, rfx/sources/sources.py:76-88 on main); then
    amplitude * (-2*arg) * exp(-arg^2) with the identical jnp ops.
    """
    tau = 1.0 / (f0 * bandwidth * math.pi)
    t0 = 3.0 * tau
    arg = (t - t0) / tau
    return amplitude * (-2.0 * arg) * jnp.exp(-(arg**2))


def test_cutoff_default_byte_identity():
    """Default cutoff=3.0 reproduces main's waveform bit-for-bit."""
    t = jnp.asarray(np.arange(10_000) * DEMO_DT)
    pulse = GaussianPulse(f0=2.2e9, bandwidth=1.2)

    assert pulse.cutoff == 3.0
    # t0 must be the exact same float product main computed (3.0 * tau).
    assert pulse.t0 == 3.0 * pulse.tau

    got = np.asarray(pulse(t))
    want = np.asarray(_main_gaussian_pulse(t, 2.2e9, 1.2))
    assert np.array_equal(got, want), (
        "GaussianPulse default-cutoff waveform drifted from main's values"
    )

    # Explicit cutoff=3.0 is the same object semantics as the default.
    explicit = np.asarray(GaussianPulse(f0=2.2e9, bandwidth=1.2, cutoff=3.0)(t))
    assert np.array_equal(got, explicit)


# ---------------------------------------------------------------------------
# 2. cutoff=4.5 cuts the deposited-DC integral by e^-(4.5^2 - 3^2)
# ---------------------------------------------------------------------------

def test_cutoff_45_reduces_deposited_dc_by_predicted_factor():
    """|sum s*dt| ratio (cutoff 4.5 vs 3) matches e^-11.25 within 10%.

    x64 is scoped to this test (jax.experimental.enable_x64 context):
    the cutoff=4.5 sum is ~1e-19 while individual samples are O(1), so a
    float32 accumulation would be pure rounding noise. Never enable x64
    at module level — it is process-global at pytest collection.
    """
    from jax.experimental import enable_x64

    with enable_x64():
        t = jnp.arange(50_000, dtype=jnp.float64) * DEMO_DT
        s3 = np.asarray(GaussianPulse(f0=2.2e9, bandwidth=1.2)(t))
        s45 = np.asarray(
            GaussianPulse(f0=2.2e9, bandwidth=1.2, cutoff=4.5)(t)
        )

    s_int_3 = abs(float(np.sum(s3))) * DEMO_DT
    s_int_45 = abs(float(np.sum(s45))) * DEMO_DT
    assert s_int_3 > 0.0

    measured_ratio = s_int_45 / s_int_3
    predicted_ratio = math.exp(-(4.5**2 - 3.0**2))  # ~1.3e-5, ~5 decades
    assert abs(measured_ratio / predicted_ratio - 1.0) < 0.10, (
        f"measured {measured_ratio:.4e} vs predicted {predicted_ratio:.4e}"
    )


# ---------------------------------------------------------------------------
# 3. until_decay DC-floor warning — predeclared four-case matrix (#388)
# ---------------------------------------------------------------------------

def _dc_floor_fires(waveform) -> list[str]:
    sim = Simulation(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=2e-3,
                     cpml_layers=4)
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=waveform)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim._warn_until_decay_dc_floor(dt=DEMO_DT, n_table=50_000)
    return [str(w.message) for w in caught if "issue #388" in str(w.message)]


@pytest.mark.parametrize(
    "label, waveform, expect_fire",
    [
        # measured rel_DC 6.0e-5 < 1e-3
        ("gaussian_cutoff3",
         GaussianPulse(f0=2.2e9, bandwidth=1.2), False),
        # ~5 decades below the cutoff=3 value
        ("gaussian_cutoff45",
         GaussianPulse(f0=2.2e9, bandwidth=1.2, cutoff=4.5), False),
        # measured rel_DC 0.68 (DC suppression e^-1/bw^2 ~ 0.5 at bw=1.2)
        ("modgaussian_bw12",
         ModulatedGaussian(f0=2.2e9, bandwidth=1.2), True),
        # measured rel_DC 4.0e-4 < 1e-3
        ("modgaussian_bw04",
         ModulatedGaussian(f0=2.2e9, bandwidth=0.4), False),
    ],
)
def test_until_decay_dc_floor_matrix(label, waveform, expect_fire):
    fired = _dc_floor_fires(waveform)
    if expect_fire:
        assert fired, f"{label}: expected the #388 DC-floor warning"
        # Quantitative: the message must carry the rel_DC number.
        assert "relative DC" in fired[0] and "cap-hit" in fired[0]
    else:
        assert not fired, f"{label}: unexpected #388 warning: {fired}"


def test_until_decay_dc_floor_wired_into_run():
    """run(until_decay=...) emits the #388 warning; a fixed-n_steps run
    of the same sim does not (the check binds to until_decay only)."""
    def _sim():
        sim = Simulation(freq_max=36e9, domain=(6e-3, 6e-3, 6e-3),
                         dx=1e-3, cpml_layers=4, boundary="cpml")
        sim.add_source((3e-3, 3e-3, 3e-3), "ez",
                       waveform=ModulatedGaussian(f0=30e9, bandwidth=1.2))
        return sim

    with pytest.warns(UserWarning, match="issue #388"):
        _sim().run(until_decay=1e-3, decay_min_steps=50,
                   decay_max_steps=300)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sim().run(n_steps=20)
    assert not [w for w in caught if "issue #388" in str(w.message)], (
        "#388 DC-floor warning must fire only when until_decay is set"
    )


# ---------------------------------------------------------------------------
# 4. unresolved-pulse preflight advisory (#386)
# ---------------------------------------------------------------------------

def _demo_class_sim(bandwidth):
    """patch_antenna_demo scale: dx=2mm, freq_max=4GHz, f0=2.2GHz."""
    sim = Simulation(freq_max=4e9, domain=(0.06, 0.06, 0.03), dx=2e-3,
                     cpml_layers=4)
    sim.add_source((0.03, 0.03, 0.015), "ez",
                   waveform=GaussianPulse(f0=2.2e9, bandwidth=bandwidth))
    return sim


def test_unresolved_pulse_fires_on_absolute_hz_bandwidth():
    """bandwidth=2.64e9 (absolute Hz for a 1.2 fractional intent) makes
    tau ~ 5e-20 s — a sub-dt spike; the #386 advisory must fire."""
    codes = {getattr(i, "code", None)
             for i in _demo_class_sim(2.64e9).preflight()}
    assert "unresolved_pulse" in codes


def test_unresolved_pulse_silent_on_demo_class_config():
    """The committed demo's config (fractional bandwidth=1.2; tau ~ 0.12ns
    vs dt ~ ps, ratio >> 3) must NOT fire."""
    codes = {getattr(i, "code", None)
             for i in _demo_class_sim(1.2).preflight()}
    assert "unresolved_pulse" not in codes


def _tfsf_sim(bandwidth):
    """TFSF carries a string waveform name; tau comes from f0/bandwidth."""
    sim = Simulation(freq_max=4e9, domain=(0.06, 0.06, 0.03), dx=2e-3,
                     cpml_layers=4, boundary="cpml")
    sim.add_tfsf_source(f0=2.2e9, bandwidth=bandwidth)
    return sim


def test_unresolved_pulse_fires_on_tfsf_absolute_hz_bandwidth():
    """add_tfsf_source(bandwidth=<absolute Hz>) is the same #386 footgun;
    the string-named waveform must not shield it from the check."""
    codes = {getattr(i, "code", None)
             for i in _tfsf_sim(2.64e9).preflight()}
    assert "unresolved_pulse" in codes


def test_unresolved_pulse_silent_on_tfsf_fractional_bandwidth():
    codes = {getattr(i, "code", None)
             for i in _tfsf_sim(1.2).preflight()}
    assert "unresolved_pulse" not in codes
