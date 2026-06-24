"""Legacy pmap distributed-CPML material-awareness regression (#205).

Companion to ``test_distributed_cpml_dielectric.py`` (which guards the LIVE
shard_map runner reached via ``sim.run(devices=...)``).  This one guards the
LEGACY pmap runner ``rfx.runners.distributed.run_distributed``, which is
reached only by direct import (the package re-exports it as
``rfx.runners.run_distributed``; ``sim.run(devices=...)`` routes to
``distributed_v2`` instead).

Before #205 the pmap scan body passed ``None`` to the (since-#227
material-aware) CPML kernels, so inside a dielectric the absorber used the
vacuum coefficient ``dt/eps_0`` -- ``eps_r`` x too strong -- and the field
DIVERGED to inf once the wave reached a dielectric-filled CPML face
(witnessed: eps_r=9, 2 devices -> inf on origin/main, finite after the wire).
This pins that the pmap scan body now threads ``materials_slab.eps_r/.mu_r``.

Note on tolerance: the legacy pmap runner has an inherent ~1% disagreement
with the single-device reference EVEN IN VACUUM (a property of this older
domain-decomposition path; the production v2 runner matches to ~1e-7).  The
fix is not expected to make the dielectric match single-device any better
than vacuum does -- only to remove the divergence and restore vacuum-level
parity.  So we assert (a) finite/bounded and (b) the dielectric agrees with
single-device about as well as vacuum does, rather than a tight absolute gate.

Fast-suite (NOT gpu-marked): runs on every PR via the conftest CPU 2-device
sentinel, like the sibling material-CPML guards.  Skips cleanly on <2 devices.
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import jax
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.runners.distributed import run_distributed as legacy_run

requires_multidevice = pytest.mark.skipif(
    jax.device_count() < 2,
    reason=(
        f"pmap distributed CPML test needs >=2 JAX devices (got "
        f"{jax.device_count()}); the XLA host-device-count sentinel only adds "
        "virtual devices on the CPU backend and is ignored if jax was already "
        "initialised by another module first."
    ),
)

# nx must be a multiple of n_devices for the pmap slabber (no padding, unlike
# v2): domain_x=0.022 / dx=0.002 -> 11 interior + 2*6 cpml + 1 node = 24 cells.
_DOMAIN = (0.022, 0.02, 0.02)
_DX = 0.002
_CPML = 6
_N_STEPS = 200
_EPS = 9.0


def _build(eps_r):
    sim = Simulation(freq_max=5e9, domain=_DOMAIN,
                     boundary="cpml", cpml_layers=_CPML, dx=_DX)
    sim.add_material("d", eps_r=eps_r)
    sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="d")
    c = (_DOMAIN[0] / 2, _DOMAIN[1] / 2, _DOMAIN[2] / 2)
    sim.add_source(c, "ez", waveform=GaussianPulse(f0=3e9))
    return sim


def _maxabs_e(result):
    m = 0.0
    for comp in ("ex", "ey", "ez"):
        a = np.asarray(getattr(result.state, comp))
        if not np.isfinite(a).all():
            return np.inf
        m = max(m, float(np.abs(a).max()))
    return m


@requires_multidevice
def test_pmap_distributed_cpml_dielectric_finite_and_matches_single():
    """eps_r=9 filling the CPML through the LEGACY pmap runner: finite (was
    inf on main) and as close to single-device as the vacuum case is."""
    devs = jax.devices()[:2]

    # Vacuum baseline: how well does the legacy pmap path track single-device
    # when the absorber coefficient is unambiguously correct?
    vac_single = _maxabs_e(_build(1.0).run(n_steps=_N_STEPS))
    vac_pmap = _maxabs_e(legacy_run(_build(1.0), n_steps=_N_STEPS, devices=devs))
    assert np.isfinite(vac_pmap)
    vac_rel = abs(vac_pmap - vac_single) / max(abs(vac_single), 1e-30)

    diel_single = _maxabs_e(_build(_EPS).run(n_steps=_N_STEPS))
    diel_pmap = _maxabs_e(legacy_run(_build(_EPS), n_steps=_N_STEPS, devices=devs))

    # (a) finite + bounded -- the bug produced inf.
    assert np.isfinite(diel_pmap), (
        f"pmap dielectric max|E| not finite ({diel_pmap}) -- the #205 "
        "vacuum-coefficient divergence has regressed on the pmap path")
    assert diel_pmap < 1e3, f"pmap dielectric max|E|={diel_pmap} grossly large"

    # (b) the dielectric matches single-device about as well as vacuum does --
    # i.e. the material fix restored vacuum-level parity (it does NOT and need
    # not beat the legacy path's inherent ~1% offset).
    diel_rel = abs(diel_pmap - diel_single) / max(abs(diel_single), 1e-30)
    # +1e-2 pad sized empirically against the legacy path's inherent ~1.05e-2
    # vacuum-vs-single offset (margin ~= the offset itself); assertion (a)
    # above is the load-bearing inf-regression gate and stands alone.
    assert diel_rel < vac_rel + 1e-2, (
        f"pmap dielectric rel-diff {diel_rel:.3e} is materially worse than the "
        f"vacuum baseline {vac_rel:.3e} -- material-aware coeff suspect")


@requires_multidevice
def test_pmap_distributed_cpml_responds_to_eps():
    """The pmap CPML coefficient must actually depend on eps_r (guards against
    a silent regression to a constant vacuum coefficient)."""
    devs = jax.devices()[:2]
    vac = _maxabs_e(legacy_run(_build(1.0), n_steps=_N_STEPS, devices=devs))
    diel = _maxabs_e(legacy_run(_build(_EPS), n_steps=_N_STEPS, devices=devs))
    assert np.isfinite(vac) and np.isfinite(diel)
    assert abs(vac - diel) / max(vac, 1e-30) > 1e-3, (
        f"vacuum ({vac:.6e}) and dielectric ({diel:.6e}) pmap responses are "
        "indistinguishable -- eps_r is not influencing the pmap run")
