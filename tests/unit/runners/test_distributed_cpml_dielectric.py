"""Distributed-CPML material-awareness regression (#205).

Before this fix the distributed FDTD runners applied the CPML correction
with a HARDCODED VACUUM coefficient ``dt/eps_0`` (E) / ``dt/mu_0`` (H),
ignoring the local material.  Inside a dielectric of ``eps_r`` the correct
E-coefficient is ``dt/(eps_r*eps_0)`` -- the vacuum value is ``eps_r`` x too
large, so the distributed absorber over-corrects and the field DIVERGES once
the wave reaches the absorber (witnessed: a 2-device eps_r=9 sim blows up to
inf at step ~75 while single-device stays finite at ~3.3e-3).

The single-device path (``rfx/boundaries/cpml.py``) has been material-aware
since #204, so ``sim.run(...)`` is the correct reference and the distributed
path must agree with it.  This test pins that agreement.

Fast-suite (NOT gpu-marked) but multi-device: it sets the CPU host-device
sentinel before importing jax and SKIPS cleanly when <2 devices are
available (e.g. a single-GPU pod, or a shared pytest process where another
module initialised jax first -- the XLA_FLAGS sentinel is process-global and
first-init-wins).  It runs correctly in an isolated invocation.
"""

import os
# Must be set before importing jax so we get >=2 host CPU devices.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import jax
import pytest

from rfx import Box, GaussianPulse, Simulation

requires_multidevice = pytest.mark.skipif(
    jax.device_count() < 2,
    reason=(
        f"distributed CPML test needs >=2 JAX devices (got "
        f"{jax.device_count()}); the XLA host-device-count sentinel only adds "
        "virtual devices on the CPU backend and is ignored if jax was already "
        "initialised by another module first."
    ),
)

# --- run parameters (mirrors the #205 witness) ---
_DOMAIN = (0.02, 0.02, 0.02)   # 10 x 10 x 10 interior cells at dx=2mm
_DX = 0.002
_CPML_LAYERS = 6
_FREQ_MAX = 5e9
_F0 = 3e9
_N_STEPS = 200
_EPS_DIELECTRIC = 9.0


def _build_sim(eps_r: float) -> Simulation:
    """Open CPML cube whose dielectric (eps_r) fills the ENTIRE domain,
    including every CPML face, with a central source radiating outward so
    energy is absorbed THROUGH the dielectric-filled absorber."""
    sim = Simulation(
        freq_max=_FREQ_MAX, domain=_DOMAIN,
        boundary="cpml", cpml_layers=_CPML_LAYERS, dx=_DX,
    )
    sim.add_material("d", eps_r=eps_r)
    sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="d")
    cx, cy, cz = _DOMAIN[0] / 2, _DOMAIN[1] / 2, _DOMAIN[2] / 2
    sim.add_source((cx, cy, cz), "ez", waveform=GaussianPulse(f0=_F0))
    return sim


def _max_abs_e(result) -> float:
    """max|E| over all three E components at the final step (inf if any
    NaN/inf is present so divergence reports as non-finite)."""
    st = result.state
    m = 0.0
    for comp in ("ex", "ey", "ez"):
        arr = np.asarray(getattr(st, comp))
        if not np.isfinite(arr).all():
            return np.inf
        m = max(m, float(np.abs(arr).max()))
    return m


@requires_multidevice
def test_distributed_cpml_dielectric_finite_and_matches_single():
    """eps_r=9 filling the CPML: distributed must be finite AND match
    single-device (pre-#205 the distributed path diverged to inf)."""
    devices = jax.devices()[:2]

    single = _max_abs_e(_build_sim(_EPS_DIELECTRIC).run(n_steps=_N_STEPS))
    multi = _max_abs_e(
        _build_sim(_EPS_DIELECTRIC).run(n_steps=_N_STEPS, devices=devices))

    # (a) distributed result is finite / bounded (the bug produced inf).
    assert np.isfinite(multi), (
        f"distributed dielectric max|E| not finite ({multi}) -- the #205 "
        "vacuum-coefficient divergence has regressed")
    assert multi < 1e3, f"distributed dielectric max|E|={multi} grossly large"

    # (b) distributed ~= single-device (the correct material-aware reference).
    rel = abs(multi - single) / max(abs(single), 1e-30)
    assert rel < 1e-2, (
        f"distributed dielectric max|E|={multi:.6e} disagrees with single "
        f"max|E|={single:.6e} (rel-diff {rel:.3e} >= 1e-2)")


@requires_multidevice
def test_distributed_cpml_responds_to_eps():
    """Sanity: the distributed CPML coefficient actually depends on eps_r --
    a vacuum vs a dielectric fill give distinguishable bounded responses
    (guards against the coefficient being silently constant again)."""
    devices = jax.devices()[:2]
    vac = _max_abs_e(_build_sim(1.0).run(n_steps=_N_STEPS, devices=devices))
    diel = _max_abs_e(
        _build_sim(_EPS_DIELECTRIC).run(n_steps=_N_STEPS, devices=devices))
    assert np.isfinite(vac) and np.isfinite(diel)
    # Different permittivity -> different field magnitude at the same step.
    assert abs(vac - diel) / max(vac, 1e-30) > 1e-3, (
        f"vacuum ({vac:.6e}) and dielectric ({diel:.6e}) responses are "
        "indistinguishable -- eps_r is not influencing the distributed run")
