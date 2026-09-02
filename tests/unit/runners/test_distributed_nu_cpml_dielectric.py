"""Distributed NON-UNIFORM CPML material-awareness regression (#205, NU lane).

Before this fix ``rfx/runners/distributed_nu.py`` applied its CPML correction
with a HARDCODED VACUUM coefficient ``dt/eps_0`` (E) / ``dt/mu_0`` (H),
ignoring the local material (``_apply_cpml_e_local_nu`` /
``_apply_cpml_h_local_nu``).  Inside a dielectric of ``eps_r`` the correct
E-coefficient is ``dt/(eps_r*eps_0)`` -- the vacuum value is ``eps_r`` x too
large, so the NON-UNIFORM distributed absorber over-corrects and the field
DIVERGES once an outgoing wave is absorbed THROUGH the dielectric-filled CPML
faces.

WITNESS (origin/main, captured field-level via the runner's ``final_state``):
a small genuinely-non-uniform cube (10 interior cells, cpml=6) fully filled
with eps_r=9 and a Gaussian source near the absorber blows the 2-device
distributed forward to ``inf`` while the single-device NU forward (which has
been material-aware since #208) stays finite (~4.6e7).  After the fix the
distributed result is finite and matches the single-device reference to
< 1e-6 (witnessed rel-diff ~8.8e-8).

REACHABILITY (an important finding in itself): the NU-distributed CPML
kernels are reached ONLY via ``Simulation.forward(distributed=True,
devices=[...], boundary='cpml')`` -- the *differentiable* lane.
``run(devices=...)`` on a non-uniform + CPML grid hard-raises
``NotImplementedError`` (Phase C fence) in ``distributed_v2.py`` before ever
touching ``distributed_nu.py``.  So this test exercises ``forward`` (not
``run``) and ALSO asserts the lane is AD-safe -- the gradient through the
fixed per-face ``eps_r`` slicing must stay finite (the uniform fix in
``distributed.py`` did not need this gate; this lane is differentiated).

Multi-device but CPU-only: it sets the host-device sentinel before importing
jax and SKIPS cleanly when < 2 devices are available (the sentinel is
process-global and first-init-wins).  NOT ``gpu``-marked: like its sibling
material-aware-CPML regression guards (``test_vmap_cpml_dielectric.py``,
``test_nonuniform_cpml_dielectric.py``, ``test_lumped_wire_sparam_cpml_dielectric.py``)
it runs in the fast suite via the conftest CPU 2-device sentinel, so the
regression actually executes on every PR.  (A ``gpu`` mark would deselect it
from the fast suite AND skip it on the single-GPU release suite, i.e. run in
no CI lane.)  ~11 s wall-clock.
"""

import os
# Must be set before importing jax so we get >=2 host CPU devices.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Box, GaussianPulse, Simulation
import rfx.runners.distributed_nu as _dnu
import rfx.runners.nonuniform as _rnu

requires_multidevice = pytest.mark.skipif(
    jax.device_count() < 2,
    reason=(
        f"distributed NU CPML test needs >=2 JAX devices (got "
        f"{jax.device_count()}); the XLA host-device-count sentinel only adds "
        "virtual devices on the CPU backend and is ignored if jax was already "
        "initialised by another module first."
    ),
)

# --- run parameters (mirror the #205 NU witness: small genuinely-NU cube, ---
# --- source near the absorber so energy is absorbed THROUGH the dielectric) -
_DX0 = 2e-3
_CPML = 6
_RATIO = 1.1            # mild grading -> genuinely non-uniform, ratio <= 5
_NX = 22               # 22 total = ~10 interior + 2*6 CPML
_FREQ_MAX = 5e9
_F0 = 3e9
_N_STEPS = 200
_EPS_DIELECTRIC = 9.0


def _graded(n: int, ratio: float) -> np.ndarray:
    """Symmetric cosine-bump graded cell-size profile (edges = _DX0, centre =
    ratio*_DX0) -> grid is genuinely non-uniform so the NU lane is taken."""
    x = np.linspace(-1.0, 1.0, n)
    s = 1.0 + (ratio - 1.0) * (0.5 * (1.0 + np.cos(np.pi * x)))
    return (_DX0 * s).astype(np.float64)


_DXP = _graded(_NX, _RATIO)
_DYP = _graded(_NX, _RATIO)
_DZP = _graded(_NX, _RATIO)
_LX = float(_DXP.sum())
_LY = float(_DYP.sum())
_LZ = float(_DZP.sum())


def _build_sim(eps_r: float) -> Simulation:
    """Open CPML cube on a genuinely non-uniform mesh whose dielectric
    (eps_r) fills the ENTIRE domain, including every CPML face, with a
    central Gaussian source radiating outward so energy is absorbed THROUGH
    the dielectric-filled absorber."""
    sim = Simulation(
        freq_max=_FREQ_MAX, domain=(_LX, _LY, _LZ), dx=_DX0,
        boundary="cpml", cpml_layers=_CPML,
        dx_profile=_DXP, dy_profile=_DYP, dz_profile=_DZP,
    )
    sim.add_material("d", eps_r=eps_r)
    sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="d")
    cx, cy, cz = _LX / 2, _LY / 2, _LZ / 2
    sim.add_source((cx, cy, cz), "ez", waveform=GaussianPulse(f0=_F0))
    sim.add_probe((cx, cy, cz), "ez")
    return sim


def _max_abs_E(state) -> float:
    """max|E| over all three E components of a gathered FDTDState (inf if any
    NaN/inf is present so divergence reports as non-finite)."""
    m = 0.0
    for comp in ("ex", "ey", "ez"):
        arr = np.asarray(getattr(state, comp))
        if not np.isfinite(arr).all():
            return np.inf
        m = max(m, float(np.abs(arr).max()))
    return m


def _capture_distributed_final_state(eps_r: float, devices) -> object:
    """Run the 2-device NU forward and capture the runner's gathered
    ``final_state`` (forward() itself surfaces only the probe time-series).
    """
    cap = {}
    orig = _dnu.run_nonuniform_distributed_pec

    def _wrap(*a, **kw):
        out = orig(*a, **kw)
        cap["state"] = out.get("final_state")
        return out

    _dnu.run_nonuniform_distributed_pec = _wrap
    try:
        _build_sim(eps_r).forward(
            distributed=True, devices=devices, n_steps=_N_STEPS,
            checkpoint=False, skip_preflight=True)
    finally:
        _dnu.run_nonuniform_distributed_pec = orig
    return cap["state"]


def _capture_single_final_state(eps_r: float) -> object:
    """Run the single-device NU forward (material-aware reference, #208) and
    capture its gathered ``state`` (``run_nonuniform`` returns a dict)."""
    cap = {}
    orig = _rnu.run_nonuniform

    def _wrap(*a, **kw):
        r = orig(*a, **kw)
        cap["state"] = r["state"] if isinstance(r, dict) else None
        return r

    _rnu.run_nonuniform = _wrap
    try:
        _build_sim(eps_r).forward(
            distributed=False, n_steps=_N_STEPS,
            checkpoint=False, skip_preflight=True)
    finally:
        _rnu.run_nonuniform = orig
    return cap["state"]


@requires_multidevice
def test_distributed_nu_cpml_dielectric_finite_and_matches_single():
    """eps_r=9 filling the NU CPML: the distributed-NU forward must be finite
    AND match the material-aware single-device NU reference (pre-#205 the
    distributed-NU path diverged to inf)."""
    devices = jax.devices()[:2]

    single = _max_abs_E(_capture_single_final_state(_EPS_DIELECTRIC))
    multi = _max_abs_E(_capture_distributed_final_state(_EPS_DIELECTRIC, devices))

    # (a) distributed result is finite (the pre-#205 bug produced inf).
    assert np.isfinite(multi), (
        f"distributed-NU dielectric max|E| not finite ({multi}) -- the #205 "
        "vacuum-coefficient divergence has regressed in distributed_nu.py")

    # (b) distributed ~= material-aware single-device reference.
    rel = abs(multi - single) / max(abs(single), 1e-30)
    assert rel < 1e-2, (
        f"distributed-NU dielectric max|E|={multi:.6e} disagrees with the "
        f"single-device material-aware reference max|E|={single:.6e} "
        f"(rel-diff {rel:.3e} >= 1e-2)")


@requires_multidevice
def test_distributed_nu_cpml_forward_is_ad_finite():
    """The NU-distributed CPML lane is differentiable (it is reached ONLY via
    forward(distributed=True)); the gradient through the fixed per-face eps_r
    slicing must stay finite and non-trivial (no NaN/inf from the new
    dt/(eps_r*eps_0) coefficient path)."""
    devices = jax.devices()[:2]

    # Offset the probe from the source so the recorded field depends on the
    # propagation through the (eps_r-dependent) medium + absorber.
    def _build_offset():
        sim = Simulation(
            freq_max=_FREQ_MAX, domain=(_LX, _LY, _LZ), dx=_DX0,
            boundary="cpml", cpml_layers=_CPML,
            dx_profile=_DXP, dy_profile=_DYP, dz_profile=_DZP,
        )
        sim.add_source((_LX * 0.45, _LY / 2, _LZ / 2), "ez")
        sim.add_probe((_LX * 0.60, _LY / 2, _LZ / 2), "ez")
        return sim

    shp = _build_offset()._build_nonuniform_grid().shape
    eps0 = jnp.full(shp, _EPS_DIELECTRIC)

    def loss(eps_val):
        res = _build_offset().forward(
            n_steps=120, eps_override=eps_val,
            distributed=True, devices=devices,
            checkpoint=False, skip_preflight=True)
        return jnp.sum(res.time_series ** 2)

    grad = np.asarray(jax.grad(loss)(eps0))
    assert np.isfinite(grad).all(), (
        "gradient through distributed-NU CPML forward is non-finite -- the "
        "#205 per-face eps_r slicing broke AD traceability")
    # Non-trivial: the eps_r-dependent coefficient must actually influence
    # the loss (guards against a silently-detached / zeroed gradient).
    assert float(np.abs(grad).max()) > 0.0, (
        "gradient through distributed-NU CPML forward is identically zero -- "
        "eps_r is not influencing the differentiable distributed run")
