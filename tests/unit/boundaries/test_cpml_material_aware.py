"""Material-aware CPML regressions across every scan body (issues #203-#205).

One fix, seven scan bodies. Every FDTD loop in rfx applies the CPML
correction with a coefficient that must be ``dt/(eps_r*eps_0)`` (E) and
``dt/(mu_r*mu_0)`` (H) for the LOCAL material. Each scan body below once
called ``apply_cpml_e/h`` WITHOUT ``materials=`` (or passed ``None``), so
inside a dielectric the absorber used the free-space coefficient — ``eps_r``
times too strong — and the field diverged to NaN/inf once an outgoing wave
reached a dielectric-filled CPML face. The witness geometry is therefore the
same everywhere: a dielectric that fills the ENTIRE domain (every CPML cell is
dielectric), so all absorption goes THROUGH dielectric-filled CPML.

Sections (one per scan body; each was its own file before tier 3b of the
2026-09 test-corpus reorganisation, see
``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``):

1. ``rfx/nonuniform.py`` step_fn (#205, NU path) — was
   ``test_nonuniform_cpml_dielectric.py``.
2. live shard_map runner via ``sim.run(devices=...)`` (#205) — was
   ``test_distributed_cpml_dielectric.py``.
3. ``rfx/runners/distributed_nu.py`` (#205, NU lane, reached ONLY via
   ``forward(distributed=True)``) — was ``test_distributed_nu_cpml_dielectric.py``.
4. legacy pmap runner ``rfx.runners.distributed.run_distributed`` (#205) —
   was ``test_distributed_pmap_cpml_dielectric.py``.
5. ``rfx/vmap_sweep.py`` batched scan (#205, #637) — was
   ``test_vmap_cpml_dielectric.py``.
6. lumped / wire S-parameter extractor re-runs
   ``rfx.probes.probes.extract_s_matrix[_wire]`` (#203) — was
   ``test_lumped_wire_sparam_cpml_dielectric.py``.
7. subgrid coarse-grid scan bodies (#205, last leg; research lane) — was
   ``test_subgrid_cpml_dielectric.py``.

Multi-device sections (2-4) set the CPU host-device sentinel before importing
jax and SKIP cleanly when < 2 devices are available (the XLA_FLAGS sentinel is
process-global and first-init-wins). They are NOT ``gpu``-marked on purpose:
they run in the fast suite via the conftest CPU 2-device sentinel, so the
regression actually executes on every PR (a ``gpu`` mark would deselect them
from the fast suite AND skip them on the single-GPU release suite, i.e. run
in no CI lane).

The witnesses, run parameters, thresholds and tolerances of every original
file are kept verbatim; only the module-level helper names carry a section
prefix so the seven fixtures can coexist.
"""

import os
# Must be set before importing jax so we get >=2 host CPU devices.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.runners.distributed import run_distributed as legacy_run
import rfx.runners.distributed_nu as _dnu
import rfx.runners.nonuniform as _rnu
from rfx.vmap_sweep import vmap_material_sweep

requires_multidevice = pytest.mark.skipif(
    jax.device_count() < 2,
    reason=(
        f"distributed CPML test needs >=2 JAX devices (got "
        f"{jax.device_count()}); the XLA host-device-count sentinel only adds "
        "virtual devices on the CPU backend and is ignored if jax was already "
        "initialised by another module first."
    ),
)


def _max_abs_e_state(state) -> float:
    """max|E| over all three E components of an FDTDState (inf if any NaN/inf
    is present so divergence reports as non-finite)."""
    m = 0.0
    for comp in ("ex", "ey", "ez"):
        arr = np.asarray(getattr(state, comp))
        if not np.isfinite(arr).all():
            return np.inf
        m = max(m, float(np.abs(arr).max()))
    return m


def _max_abs_e(result) -> float:
    """max|E| at the final step of a run() result (see ``_max_abs_e_state``)."""
    return _max_abs_e_state(result.state)


# ===========================================================================
# 1. Non-uniform scan body (issue #205, nonuniform path)
# ===========================================================================
#
# The non-uniform scan body (rfx/nonuniform.py step_fn) called apply_cpml_h/e
# WITHOUT the materials= argument, so the CPML fell back to free-space
# eps_0/mu_0 coefficients (rfx/boundaries/cpml.py). Witnessed: a non-uniform
# (dz_profile) sim whose dielectric fills the whole domain diverges to all-NaN
# pre-fix at eps_r=4 and eps_r=10, and becomes finite + cleanly absorbing
# (tail/peak ~1e-4) once materials= is threaded into the two CPML calls
# (matching the production uniform scan).


def _nu_full_dielectric_sim(eps_r):
    """Non-uniform z-mesh, CPML boundary, dielectric filling the entire domain
    (every CPML cell is dielectric) -- the geometry that forces all absorption
    through dielectric-filled CPML faces."""
    dz_profile = np.concatenate([
        np.full(8, 0.5e-3), np.full(8, 1.0e-3), np.full(8, 0.5e-3),
    ])
    lz = float(dz_profile.sum())
    sim = Simulation(
        freq_max=10e9, domain=(0.03, 0.03, lz), dx=1.0e-3,
        boundary="cpml", cpml_layers=6, dz_profile=dz_profile,
    )
    sim.add_material("diel", eps_r=eps_r)
    sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="diel")  # fills domain + CPML
    sim.add_source((0.015, 0.015, lz / 2), "ez",
                   waveform=GaussianPulse(f0=4e9, bandwidth=0.7))
    sim.add_probe((0.022, 0.015, lz / 2), "ez")
    return sim


def test_nonuniform_cpml_dielectric_stable_and_absorbing():
    """NU CPML in a dielectric-filled domain must stay finite and absorb.

    Pre-fix (free-space CPML coefficients in the dielectric) this diverged to
    all-NaN; the materials-aware fix makes it finite and cleanly absorbing.
    """
    for eps_r in (4.0, 10.0):
        res = _nu_full_dielectric_sim(eps_r).run(n_steps=2500)
        ts = np.abs(np.asarray(res.time_series).reshape(-1))
        assert ts.size > 0
        assert np.all(np.isfinite(ts)), \
            f"NU CPML diverged (non-finite) in eps_r={eps_r} dielectric (issue #205)"
        peak = float(ts.max())
        tail = float(ts[-200:].max())
        # A working absorber drives the late-time field far below the peak.
        assert tail <= 0.05 * peak, \
            f"NU CPML did not absorb (eps_r={eps_r}): tail/peak={tail / max(peak, 1e-30):.3e}"


# ===========================================================================
# 2. Live shard_map distributed runner via sim.run(devices=...) (issue #205)
# ===========================================================================
#
# Before this fix the distributed FDTD runners applied the CPML correction with
# a HARDCODED VACUUM coefficient dt/eps_0 (E) / dt/mu_0 (H), ignoring the local
# material (witnessed: a 2-device eps_r=9 sim blows up to inf at step ~75 while
# single-device stays finite at ~3.3e-3). The single-device path
# (rfx/boundaries/cpml.py) has been material-aware since #204, so sim.run(...)
# is the correct reference and the distributed path must agree with it.

# --- run parameters (mirrors the #205 witness) ---
_UNI_DOMAIN = (0.02, 0.02, 0.02)   # 10 x 10 x 10 interior cells at dx=2mm
_UNI_DX = 0.002
_UNI_CPML_LAYERS = 6
_UNI_FREQ_MAX = 5e9
_UNI_F0 = 3e9
_UNI_N_STEPS = 200
_UNI_EPS_DIELECTRIC = 9.0


def _uni_build_sim(eps_r: float) -> Simulation:
    """Open CPML cube whose dielectric (eps_r) fills the ENTIRE domain,
    including every CPML face, with a central source radiating outward so
    energy is absorbed THROUGH the dielectric-filled absorber."""
    sim = Simulation(
        freq_max=_UNI_FREQ_MAX, domain=_UNI_DOMAIN,
        boundary="cpml", cpml_layers=_UNI_CPML_LAYERS, dx=_UNI_DX,
    )
    sim.add_material("d", eps_r=eps_r)
    sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="d")
    cx, cy, cz = _UNI_DOMAIN[0] / 2, _UNI_DOMAIN[1] / 2, _UNI_DOMAIN[2] / 2
    sim.add_source((cx, cy, cz), "ez", waveform=GaussianPulse(f0=_UNI_F0))
    return sim


@requires_multidevice
def test_distributed_cpml_dielectric_finite_and_matches_single():
    """eps_r=9 filling the CPML: distributed must be finite AND match
    single-device (pre-#205 the distributed path diverged to inf)."""
    devices = jax.devices()[:2]

    single = _max_abs_e(_uni_build_sim(_UNI_EPS_DIELECTRIC).run(n_steps=_UNI_N_STEPS))
    multi = _max_abs_e(
        _uni_build_sim(_UNI_EPS_DIELECTRIC).run(n_steps=_UNI_N_STEPS, devices=devices))

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
    vac = _max_abs_e(_uni_build_sim(1.0).run(n_steps=_UNI_N_STEPS, devices=devices))
    diel = _max_abs_e(
        _uni_build_sim(_UNI_EPS_DIELECTRIC).run(n_steps=_UNI_N_STEPS, devices=devices))
    assert np.isfinite(vac) and np.isfinite(diel)
    # Different permittivity -> different field magnitude at the same step.
    assert abs(vac - diel) / max(vac, 1e-30) > 1e-3, (
        f"vacuum ({vac:.6e}) and dielectric ({diel:.6e}) responses are "
        "indistinguishable -- eps_r is not influencing the distributed run")


# ===========================================================================
# 3. Distributed NON-UNIFORM runner (issue #205, NU lane)
# ===========================================================================
#
# Before this fix rfx/runners/distributed_nu.py applied its CPML correction
# with a HARDCODED VACUUM coefficient (``_apply_cpml_e_local_nu`` /
# ``_apply_cpml_h_local_nu``). WITNESS (origin/main, captured field-level via
# the runner's ``final_state``): a small genuinely-non-uniform cube (10
# interior cells, cpml=6) fully filled with eps_r=9 and a Gaussian source near
# the absorber blows the 2-device distributed forward to inf while the
# single-device NU forward (material-aware since #208) stays finite (~4.6e7).
# After the fix the distributed result matches the single-device reference to
# < 1e-6 (witnessed rel-diff ~8.8e-8).
#
# REACHABILITY (an important finding in itself): the NU-distributed CPML
# kernels are reached ONLY via ``Simulation.forward(distributed=True,
# devices=[...], boundary='cpml')`` -- the *differentiable* lane.
# ``run(devices=...)`` on a non-uniform + CPML grid hard-raises
# NotImplementedError (Phase C fence) in distributed_v2.py before ever touching
# distributed_nu.py. So these tests exercise ``forward`` (not ``run``) and
# ALSO assert the lane is AD-safe. ~11 s wall-clock.

# --- run parameters (mirror the #205 NU witness: small genuinely-NU cube, ---
# --- source near the absorber so energy is absorbed THROUGH the dielectric) -
_NU_DX0 = 2e-3
_NU_CPML = 6
_NU_RATIO = 1.1            # mild grading -> genuinely non-uniform, ratio <= 5
_NU_NX = 22                # 22 total = ~10 interior + 2*6 CPML
_NU_FREQ_MAX = 5e9
_NU_F0 = 3e9
_NU_N_STEPS = 200
_NU_EPS_DIELECTRIC = 9.0


def _nu_graded(n: int, ratio: float) -> np.ndarray:
    """Symmetric cosine-bump graded cell-size profile (edges = _NU_DX0, centre =
    ratio*_NU_DX0) -> grid is genuinely non-uniform so the NU lane is taken."""
    x = np.linspace(-1.0, 1.0, n)
    s = 1.0 + (ratio - 1.0) * (0.5 * (1.0 + np.cos(np.pi * x)))
    return (_NU_DX0 * s).astype(np.float64)


_NU_DXP = _nu_graded(_NU_NX, _NU_RATIO)
_NU_DYP = _nu_graded(_NU_NX, _NU_RATIO)
_NU_DZP = _nu_graded(_NU_NX, _NU_RATIO)
_NU_LX = float(_NU_DXP.sum())
_NU_LY = float(_NU_DYP.sum())
_NU_LZ = float(_NU_DZP.sum())


def _nu_build_sim(eps_r: float) -> Simulation:
    """Open CPML cube on a genuinely non-uniform mesh whose dielectric
    (eps_r) fills the ENTIRE domain, including every CPML face, with a
    central Gaussian source radiating outward so energy is absorbed THROUGH
    the dielectric-filled absorber."""
    sim = Simulation(
        freq_max=_NU_FREQ_MAX, domain=(_NU_LX, _NU_LY, _NU_LZ), dx=_NU_DX0,
        boundary="cpml", cpml_layers=_NU_CPML,
        dx_profile=_NU_DXP, dy_profile=_NU_DYP, dz_profile=_NU_DZP,
    )
    sim.add_material("d", eps_r=eps_r)
    sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="d")
    cx, cy, cz = _NU_LX / 2, _NU_LY / 2, _NU_LZ / 2
    sim.add_source((cx, cy, cz), "ez", waveform=GaussianPulse(f0=_NU_F0))
    sim.add_probe((cx, cy, cz), "ez")
    return sim


def _nu_capture_distributed_final_state(eps_r: float, devices) -> object:
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
        _nu_build_sim(eps_r).forward(
            distributed=True, devices=devices, n_steps=_NU_N_STEPS,
            checkpoint=False, skip_preflight=True)
    finally:
        _dnu.run_nonuniform_distributed_pec = orig
    return cap["state"]


def _nu_capture_single_final_state(eps_r: float) -> object:
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
        _nu_build_sim(eps_r).forward(
            distributed=False, n_steps=_NU_N_STEPS,
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

    single = _max_abs_e_state(_nu_capture_single_final_state(_NU_EPS_DIELECTRIC))
    multi = _max_abs_e_state(_nu_capture_distributed_final_state(_NU_EPS_DIELECTRIC, devices))

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
            freq_max=_NU_FREQ_MAX, domain=(_NU_LX, _NU_LY, _NU_LZ), dx=_NU_DX0,
            boundary="cpml", cpml_layers=_NU_CPML,
            dx_profile=_NU_DXP, dy_profile=_NU_DYP, dz_profile=_NU_DZP,
        )
        sim.add_source((_NU_LX * 0.45, _NU_LY / 2, _NU_LZ / 2), "ez")
        sim.add_probe((_NU_LX * 0.60, _NU_LY / 2, _NU_LZ / 2), "ez")
        return sim

    shp = _build_offset()._build_nonuniform_grid().shape
    eps0 = jnp.full(shp, _NU_EPS_DIELECTRIC)

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


# ===========================================================================
# 4. Legacy pmap distributed runner (issue #205)
# ===========================================================================
#
# Companion to section 2 (which guards the LIVE shard_map runner reached via
# ``sim.run(devices=...)``). This one guards the LEGACY pmap runner
# ``rfx.runners.distributed.run_distributed``, which is reached only by direct
# import (the package re-exports it as ``rfx.runners.run_distributed``;
# ``sim.run(devices=...)`` routes to ``distributed_v2`` instead).
#
# Before #205 the pmap scan body passed ``None`` to the (since-#227
# material-aware) CPML kernels (witnessed: eps_r=9, 2 devices -> inf on
# origin/main, finite after the wire). This pins that the pmap scan body now
# threads ``materials_slab.eps_r/.mu_r``.
#
# Note on tolerance: the legacy pmap runner has an inherent ~1% disagreement
# with the single-device reference EVEN IN VACUUM (a property of this older
# domain-decomposition path; the production v2 runner matches to ~1e-7). The
# fix is not expected to make the dielectric match single-device any better
# than vacuum does -- only to remove the divergence and restore vacuum-level
# parity. So we assert (a) finite/bounded and (b) the dielectric agrees with
# single-device about as well as vacuum does, rather than a tight absolute gate.

# nx must be a multiple of n_devices for the pmap slabber (no padding, unlike
# v2): domain_x=0.022 / dx=0.002 -> 11 interior + 2*6 cpml + 1 node = 24 cells.
_PMAP_DOMAIN = (0.022, 0.02, 0.02)
_PMAP_DX = 0.002
_PMAP_CPML = 6
_PMAP_N_STEPS = 200
_PMAP_EPS = 9.0


def _pmap_build(eps_r):
    sim = Simulation(freq_max=5e9, domain=_PMAP_DOMAIN,
                     boundary="cpml", cpml_layers=_PMAP_CPML, dx=_PMAP_DX)
    sim.add_material("d", eps_r=eps_r)
    sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="d")
    c = (_PMAP_DOMAIN[0] / 2, _PMAP_DOMAIN[1] / 2, _PMAP_DOMAIN[2] / 2)
    sim.add_source(c, "ez", waveform=GaussianPulse(f0=3e9))
    return sim


@requires_multidevice
def test_pmap_distributed_cpml_dielectric_finite_and_matches_single():
    """eps_r=9 filling the CPML through the LEGACY pmap runner: finite (was
    inf on main) and as close to single-device as the vacuum case is."""
    devs = jax.devices()[:2]

    # Vacuum baseline: how well does the legacy pmap path track single-device
    # when the absorber coefficient is unambiguously correct?
    vac_single = _max_abs_e(_pmap_build(1.0).run(n_steps=_PMAP_N_STEPS))
    vac_pmap = _max_abs_e(legacy_run(_pmap_build(1.0), n_steps=_PMAP_N_STEPS, devices=devs))
    assert np.isfinite(vac_pmap)
    vac_rel = abs(vac_pmap - vac_single) / max(abs(vac_single), 1e-30)

    diel_single = _max_abs_e(_pmap_build(_PMAP_EPS).run(n_steps=_PMAP_N_STEPS))
    diel_pmap = _max_abs_e(legacy_run(_pmap_build(_PMAP_EPS), n_steps=_PMAP_N_STEPS, devices=devs))

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
    vac = _max_abs_e(legacy_run(_pmap_build(1.0), n_steps=_PMAP_N_STEPS, devices=devs))
    diel = _max_abs_e(legacy_run(_pmap_build(_PMAP_EPS), n_steps=_PMAP_N_STEPS, devices=devs))
    assert np.isfinite(vac) and np.isfinite(diel)
    assert abs(vac - diel) / max(vac, 1e-30) > 1e-3, (
        f"vacuum ({vac:.6e}) and dielectric ({diel:.6e}) pmap responses are "
        "indistinguishable -- eps_r is not influencing the pmap run")


# ===========================================================================
# 5. vmap_sweep scan body (issue #205; #637 CPML-padding guard)
# ===========================================================================
#
# ``rfx/vmap_sweep.py``'s batched scan called ``apply_cpml_e/h`` WITHOUT
# ``materials=``. The fix is a passthrough: ``materials`` is already the
# per-batch-element ``MaterialArrays`` inside the vmapped ``run_one``, so
# passing ``materials=materials`` lets vmap batch it transparently. Witness
# measured on the buggy tree: vmap diverges (max|.|~1e24-1e26) while the
# material-aware ``run()`` stays finite (~3e-2).


def _vmap_full_dielectric_cpml_sim(eps_r: float):
    """CPML sim whose dielectric fills the ENTIRE domain (incl. all 6 CPML
    faces), so the absorber sees the dielectric — the geometry that exposes a
    free-space-CPML divergence.

    Hi bound is domain-edge + half a cell (0.021, not 0.02): ``Box``'s
    rasterization is half-open (``[lo, hi)``), so a hi bound landing
    EXACTLY on the domain edge drops that edge node from the box's own
    mask on all three axes here. Unrelated to this section's #205 mechanism,
    but it interacts with issue #627 (``rfx.geometry.rasterize_grid``'s
    hi-face pad vacuum fallback, landed as fce1091): post-#627,
    ``Simulation.run()`` recovers that node via the fallback while this
    section's ``vmap_material_sweep`` call does not reproduce that fallback
    (see #637's CHANGELOG entry, "Overlap with issue #627" — an
    already-disclosed, deliberately out-of-scope gap for #637). Without
    this margin, ``test_vmap_cpml_dielectric_is_finite_and_matches_run``
    would fail for that separate, unrelated reason. Confirmed directly
    the margin stays inside the interior (0 mask cells inside the actual
    CPML pad) before landing this change.
    """
    sim = Simulation(
        freq_max=5e9, domain=(0.02, 0.02, 0.02),
        boundary="cpml", cpml_layers=6, dx=0.002,
    )
    sim.add_material("d", eps_r=eps_r)
    sim.add(Box((0.0, 0.0, 0.0), (0.021, 0.021, 0.021)), material="d")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.006, 0.01, 0.01), "ez")
    return sim


def test_vmap_cpml_dielectric_is_finite_and_matches_run():
    """vmap over a dielectric that fills the CPML must stay finite (material-
    aware absorber) and reproduce the uniform material-aware run() path.

    Pre-#205-fix this diverges to NaN/inf because the vmap scan used free-space
    CPML inside the eps_r dielectric.

    #637 completeness note: this originally asserted ONLY ``ts[1]``
    (eps_r=10.0) against ``run()`` — the base simulation below is built
    with ``eps_r=10.0`` too, so ``ts[1]`` is the one sweep element whose
    value happens to EQUAL the base material. #637 was exactly "a
    material-named sweep's CPML padding stays matched to the base
    simulation instead of the swept value" — for the one element that
    already IS the base value, that padding is correct by construction,
    so this test could never have caught it (or a regression of its
    fix) via ``ts[1]`` alone. ``ts[0]`` (eps_r=4.0, which DIFFERS from
    the base 10.0) is the load-bearing assertion: measured directly,
    pre-#637-fix ``ts[0]`` disagreed with ``run(eps_r=4.0)`` at
    rel=9.65e-03 (the #637 defect signature — the padding stayed
    matched to eps_r=10.0 while the interior correctly swept to 4.0);
    post-fix both ``ts[0]`` and ``ts[1]`` are exactly bit-identical
    (0.0) with their respective ``run()`` references.
    """
    eps_values = np.array([4.0, 10.0])
    n_steps = 300  # buggy tree reaches ~1e26 well before this

    res = vmap_material_sweep(
        _vmap_full_dielectric_cpml_sim(10.0), "d.eps_r", eps_values, n_steps=n_steps,
    )
    ts = np.asarray(res.time_series)  # (2, n_steps, 1)

    # Primary regression guard: a free-space absorber in the dielectric diverges.
    assert np.isfinite(ts).all(), (
        "vmap CPML sweep produced non-finite fields — the dielectric-filled "
        "absorber is not material-aware (issue #205 regression). "
        f"max|.|={np.nanmax(np.abs(ts)):.3e}"
    )
    # The whole batch must be bounded (passive), not just non-NaN.
    assert float(np.max(np.abs(ts))) < 1.0, (
        f"vmap CPML sweep fields are implausibly large ({np.max(np.abs(ts)):.3e}) "
        "— absorber likely mismatched."
    )

    # Correctness: EVERY sweep element must reproduce the material-aware
    # uniform run() path (the supported reference) on the same geometry --
    # eps_r=4.0 (differs from the base 10.0, the #637 CPML-padding
    # regression guard) AND eps_r=10.0 (matches base, kept for continuity
    # with the original #205 pin).
    for idx, ev in enumerate(eps_values):
        ref = np.asarray(
            _vmap_full_dielectric_cpml_sim(float(ev)).run(n_steps=n_steps).time_series
        )
        np.testing.assert_allclose(
            ts[idx], ref, atol=1e-5, rtol=1e-4,
            err_msg=f"vmap CPML sweep (eps_r={ev}) disagrees with "
                    f"material-aware run()",
        )


def test_vmap_cpml_distinct_eps_change_response():
    """Sanity: different eps in the (now finite) CPML sweep give distinct, finite
    responses — guards against a degenerate 'all-clamped-to-the-same' fix."""
    eps_values = np.array([2.0, 8.0])
    res = vmap_material_sweep(
        _vmap_full_dielectric_cpml_sim(8.0), "d.eps_r", eps_values, n_steps=250,
    )
    ts = np.asarray(res.time_series)
    assert np.isfinite(ts).all()
    assert np.max(np.abs(ts[0] - ts[1])) > 0.0, "eps_r had no effect on the sweep"


# ===========================================================================
# 6. Lumped / wire S-parameter extractor re-runs (issue #203)
# ===========================================================================
#
# ``run(compute_s_params=True)`` for a single-cell lumped port (or a wire
# port) runs a *separate* eager FDTD re-run inside
# ``rfx.probes.probes.extract_s_matrix`` / ``extract_s_matrix_wire`` to
# accumulate the port V/I DFTs. Those re-runs called ``apply_cpml_h/e``
# WITHOUT ``materials=``; inside a dielectric the field diverges to float32
# overflow (first NaN ~step 300-400), the V/I DFTs are poisoned, and every
# S-parameter comes back NaN. The production JIT scan body passes
# ``materials=materials`` and is stable — which is why the *same* run with
# ``compute_s_params=False`` is healthy.
#
# ``test_lumped_...`` is the strict #203 regression: it triggers the
# divergence deterministically (dielectric spanning the full transverse
# cross-section so its y/z faces sit in the CPML, run ~700 steps so the
# pre-fix field blow-up reaches NaN), and was confirmed to FAIL on the pre-fix
# code and pass after. ``test_wire_...`` exercises the sibling wire extractor
# on the same geometry; empirically this case stayed finite even pre-fix (the
# single-cell lumped excitation is what seeded the instability), so it is a
# forward-looking finiteness/passivity guard, not a proven before/after
# regression. Both assert only finiteness and passivity (|S| <= 1); the
# absolute |S11| of a lossless dielectric block on an open domain is a
# near-total reflector and is not a validated number.

# Enough steps for the pre-fix CPML divergence to reach NaN (it first appears
# ~step 300-400; fewer steps would let the buggy code pass spuriously).
_LW_N_STEPS = 700


def _lw_dielectric_in_cpml_sim():
    """Small open-domain sim with an eps_r=4 block spanning the transverse
    cross-section, so the dielectric occupies CPML cells (the #203 trigger)."""
    sim = Simulation(
        freq_max=5e9,
        domain=(0.06, 0.03, 0.02),
        dx=1.5e-3,
        boundary="cpml",
        cpml_layers=8,
    )
    sim.add_material("diel", eps_r=4.0)
    # Spans full y/z extent -> the block's transverse faces sit in the CPML.
    sim.add(Box((0.02, 0.0, 0.0), (0.04, 0.03, 0.02)), material="diel")
    return sim


def test_lumped_port_sparam_cpml_dielectric_finite_passive():
    """Single-cell lumped port + CPML + dielectric must give finite, passive S11.

    Pre-fix this returned all-NaN s_params (issue #203 as-filed symptom).
    """
    sim = _lw_dielectric_in_cpml_sim()
    # Single-cell lumped port (extent=None), interior in x (clear of x-CPML).
    sim.add_port(
        position=(0.03, 0.015, 0.01),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
    )
    result = sim.run(n_steps=_LW_N_STEPS, compute_s_params=True)

    assert result.s_params is not None
    s = np.asarray(result.s_params)
    assert s.shape == (1, 1, result.freqs.shape[0])
    assert np.all(np.isfinite(s)), "lumped S-params must be finite (issue #203)"
    max_s11 = float(np.max(np.abs(s[0, 0, :])))
    assert max_s11 <= 1.0 + 1e-3, f"passivity: max|S11|={max_s11:.4f} > 1"


def test_wire_port_sparam_cpml_dielectric_finite_passive():
    """Wire port (extent=) + CPML + dielectric stays finite and passive.

    ``extract_s_matrix_wire`` carried the identical missing-``materials=`` CPML
    omission and is fixed alongside the lumped extractor. This geometry did not
    by itself reproduce the pre-fix lumped divergence (the wire excitation does
    not seed it the same way), so this is a forward-looking finiteness/passivity
    guard on the wire S-param path rather than a proven before/after regression.
    """
    sim = _lw_dielectric_in_cpml_sim()
    sim.add_port(
        position=(0.03, 0.015, 0.01),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
        extent=0.006,
    )
    result = sim.run(n_steps=_LW_N_STEPS, compute_s_params=True)

    assert result.s_params is not None
    s = np.asarray(result.s_params)
    assert np.all(np.isfinite(s)), "wire S-params must be finite (issue #203)"
    max_s11 = float(np.max(np.abs(s[0, 0, :])))
    # RESTORED 2026-08-29 (issue #683 x #764 flip, written provenance —
    # docs/design_notes/issue683_decomposer_flip_predeclaration.md): the
    # lane now samples the physical V/I/V_port POST-injection, so the
    # whole-port driven diagonal is the validated terminal pair and its
    # physical passivity bound is live again (on the PRE-injection
    # interim this fixture measured max|S11| = 6.1968 — keyed-envelope
    # era, history in the git log).  The finiteness guard above — this
    # test's original #203 regression target — stays live.
    assert max_s11 <= 1.0 + 1e-3, (
        f"driven wire diagonal not passive: max|S11|={max_s11:.4f} "
        f"(physical gate restored by the #683 flip)")


# ===========================================================================
# 7. Subgrid coarse-grid scan bodies (issue #205, last leg)
# ===========================================================================
#
# The subgrid coarse-grid scan bodies (``rfx/subgridding/jit_runner.py`` — the
# ``Simulation.add_refinement`` API lane — and the eager
# ``rfx/subgridding/runner.py``) called ``apply_cpml_h/e`` WITHOUT
# ``materials=``. This was the last remaining CPML scan body without
# ``materials=`` (after #203/#204, #208, #224, #227-#229).
#
# WHY ``validation="research"``: the production subgrid envelope structurally
# rejects every CPML-adjacent configuration (measured 2026-07-03:
# ``z_slab_requires_guarded_boundary`` for centered slabs,
# ``boundary_terminated_requires_pec_no_cpml`` + ``subgrid_overlaps_absorber``
# for boundary-touching slabs — even all-vacuum with x/y-only CPML). The buggy
# code path is therefore reachable only through the research lane, which these
# tests exercise deliberately; it is a divergence regression, not a
# physics-accuracy claim for the experimental subgrid lane.
#
# Witness measured on the pre-fix tree (research lane, dielectric filling the
# domain incl. all x/y CPML faces, z=PEC): eps_r=4 and eps_r=10 both diverge to
# all-NaN, while the vacuum control stays finite and absorbing (tail/peak
# ~1e-2). Post-fix all three are finite and absorbing; the vacuum control
# changes only at float32 round-off (~6e-8 absolute on ~1e-3-scale fields).


def _subgrid_dielectric_cpml_sim(eps_r):
    """Subgrid (z-slab refinement) sim with x/y CPML, z PEC, and a dielectric
    filling the entire domain — every x/y CPML cell is dielectric, forcing
    absorption through dielectric-filled CPML on the coarse grid."""
    lz = 0.016
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, lz), dx=1.0e-3,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="cpml", hi="cpml"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=5,
    )
    sim.add_refinement((0.0, 0.006), ratio=2, validation="research")
    if eps_r is not None:
        sim.add_material("diel", eps_r=eps_r)
        sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="diel")
    sim.add_source((0.010, 0.010, 0.003), "ez",
                   waveform=GaussianPulse(f0=4e9, bandwidth=0.7))
    sim.add_probe((0.015, 0.010, 0.003), "ez")
    return sim


def test_subgrid_cpml_dielectric_stable_and_absorbing():
    """Subgrid coarse-grid CPML in a dielectric-filled domain must stay finite
    and absorb. Pre-fix (free-space CPML coefficients in the dielectric) this
    diverged to all-NaN at eps_r=4 and eps_r=10 (issue #205)."""
    for eps_r in (4.0, 10.0):
        res = _subgrid_dielectric_cpml_sim(eps_r).run(n_steps=1500)
        ts = np.abs(np.asarray(res.time_series).reshape(-1))
        assert ts.size > 0
        assert np.all(np.isfinite(ts)), \
            f"subgrid CPML diverged (non-finite) in eps_r={eps_r} dielectric (issue #205)"
        peak = float(ts.max())
        tail = float(ts[-150:].max())
        assert tail <= 0.05 * peak, \
            f"subgrid CPML did not absorb (eps_r={eps_r}): tail/peak={tail / max(peak, 1e-30):.3e}"


def test_subgrid_cpml_vacuum_control_still_absorbing():
    """Vacuum control: the materials-aware coefficient path with all-vacuum
    arrays must behave like the old materials=None fallback (same finite,
    absorbing run — only float32 round-off moves)."""
    res = _subgrid_dielectric_cpml_sim(None).run(n_steps=1500)
    ts = np.abs(np.asarray(res.time_series).reshape(-1))
    assert np.all(np.isfinite(ts))
    peak = float(ts.max())
    tail = float(ts[-150:].max())
    assert tail <= 0.05 * peak, \
        f"subgrid vacuum CPML control regressed: tail/peak={tail / max(peak, 1e-30):.3e}"
