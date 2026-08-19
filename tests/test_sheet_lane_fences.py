"""#677 G9 lane fences — every fence pinned THROUGH its lane's entry point.

Why this file exists (the defect it closes)
-------------------------------------------

Since #677 the surface-impedance (``surface_impedance_f0``) sheet no longer
rides ``materials.sigma``. It is a per-step operator ctx that each lane must
thread explicitly, so a lane that ignores the ctx does not produce a *wrong*
sheet — it produces **no sheet at all**, silently, with every existing
assertion still green. That is the #369 vaporized-metal class, and the
guard against it is the set of loud refusals listed in the #677 commit
message.

Before this file, most of those refusals were pinned only by
``tests/test_sheet_impedance_operator.py::test_g9_refusal_helper_message_names_the_lane``,
which calls ``refuse_f0_sheets([tc], "subgridded (SBP-SAT) run()")``
directly. That test proves the HELPER formats a message. It does not prove
any lane CALLS the helper: delete
``refuse_f0_sheets(self._thin_conductors, "ADI run()")`` from
``rfx/api/_execute.py`` and the helper test stays green while
``sim.run(solver="adi")`` starts silently simulating a sheet-free model.
Same for the ADI/subgridded/distributed/NU-forward/MSL/mixed/optimize/
topology/material-fit/gradient-check/subpixel/multimode call sites.

What each test here proves
--------------------------

Every test enters through the lane's PUBLIC entry point with a live f0
sheet registered, and ``_fence`` checks two things at once:

* the message names THIS lane (each ``refuse_f0_sheets`` call site passes a
  distinct lane string, and each inline fence has distinct wording), and
* the traceback passes through the expected ``(module, function)`` — the
  part a helper unit test cannot give. A message match alone would also
  pass if some sibling lane's fence fired first, or if the raise came from
  the helper called anywhere at all.

``test_every_fence_in_the_source_is_pinned`` closes the class rather than
today's instances: it re-derives the fence inventory from the ``rfx/``
source by AST (``refuse_f0_sheets(...)`` call sites, ``(#677 v1)`` inline
raises, and ``has_f0_sheets(...)`` eligibility fences) and fails if a fence
exists that ``FENCE_REGISTRY`` below does not name a pinning test for — so
a new unpinned lane fence is red the day it is added.

Scope note: five fences are pinned in ``test_sheet_impedance_operator.py``
(UPML, the two dispersive-overlap lanes, both distributed runners, the GPU
baked fast path) and one in ``test_leontovich_sheet_identity.py`` (the vmap
scan-builder ineligibility). They are NOT re-implemented here; the registry
names them where they live and the inventory test asserts those tests still
exist under those names, so a rename cannot silently orphan a fence.
"""

from __future__ import annotations

import ast
import importlib
import os
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Box, DebyePole, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

F0 = 10e9
_SHEET_BOX = Box((4e-3, 4e-3, 10e-3), (16e-3, 16e-3, 10e-3))


# ---------------------------------------------------------------------------
# Fixtures: one f0 sheet, per-lane geometry
# ---------------------------------------------------------------------------

def _sheet(sim, box=_SHEET_BOX):
    """Register one Leontovich sheet (the add-time advisory is not the
    subject here — the lane refusal is)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(box, sigma_bulk=1e4, surface_impedance_f0=F0)
    return sim


def _cube(**sim_kw):
    """20 mm PEC cube, one z-normal sheet at mid-height, one field source."""
    sim_kw.setdefault("boundary", "pec")
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     **sim_kw)
    _sheet(sim)
    sim.add_source((10e-3, 10e-3, 5e-3), "ex",
                   waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    sim.add_probe((10e-3, 10e-3, 14e-3), "ex")
    return sim


def _nu_cube(**sim_kw):
    """Same fixture on the non-uniform lane (uniform dz_profile — the
    fences under test do not depend on the grading ratio)."""
    sim_kw.setdefault("boundary", "pec")
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.0), dx=1e-3,
                     dz_profile=[1e-3] * 20, **sim_kw)
    _sheet(sim)
    sim.add_source((10e-3, 10e-3, 5e-3), "ex",
                   waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    sim.add_probe((10e-3, 10e-3, 14e-3), "ex")
    return sim


def _design_cube():
    """Port-fed PEC cube for the optimize / gradient-check / topology lanes."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     boundary="pec")
    _sheet(sim)
    sim.add_port((5e-3, 10e-3, 10e-3), "ez")
    sim.add_probe((14e-3, 10e-3, 10e-3), "ez")
    return sim


def _wr90(n_modes=1):
    """Coarse WR-90 two-port with a sheet across the guide."""
    sim = Simulation(
        freq_max=8e9, domain=(0.06, 0.02286, 0.01016), dx=3e-3,
        boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=6)
    _sheet(sim, Box((0.02, 0.0, 0.005), (0.04, 0.02286, 0.005)))
    freqs = jnp.linspace(6e9, 7e9, 3)
    for x, d, name in ((0.009, "+x", "l"), (0.051, "-x", "r")):
        sim.add_waveguide_port(x, direction=d, mode=(1, 0), mode_type="TE",
                               freqs=freqs, f0=6.5e9, bandwidth=0.4,
                               name=name, n_modes=n_modes)
    return sim


def _msl_thru():
    """Two-port MSL thru line (same shape as the fast MSL fixtures) with a
    sheet floating in the substrate region."""
    sim = Simulation(freq_max=20e9, domain=(0.012, 0.008, 0.0032), dx=2e-4,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    sim.add(Box((0.0, 0.0034, 0.0008), (0.012, 0.0046, 0.0010)), material="pec")
    _sheet(sim, Box((0.004, 0.003, 0.002), (0.008, 0.005, 0.002)))
    sim.add_msl_port(position=(0.002, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


def _mixed_probe_fed_msl():
    """Probe-fed MSL board for the mixed (wire + MSL) lane. Ladder geometry
    copied from the working fixture in tests/test_mixed_port_sparam.py so
    the earlier port-geometry guards pass and the #677 fence is reached."""
    eps_r, h_sub, w_trace, dx = 3.66, 254e-6, 600e-6, 80e-6
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(freq_max=5e9, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("sub", eps_r=eps_r)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, h_sub)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - w_trace / 2, h_sub),
                (lx, y_c + w_trace / 2, h_sub + dx)), material="pec")
    _sheet(sim, Box((3e-3, 1e-3, 5e-4), (5e-3, 2e-3, 5e-4)))
    sim.add_port(position=(2e-3, y_c, 0.0), component="ez", impedance=50.0,
                 extent=h_sub)
    sim.add_msl_port(position=(5.5e-3, y_c, 0.0), width=w_trace, height=h_sub,
                     direction="-x", impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5),
                     n_probe_offset=10, n_probe_spacing=4)
    return sim


# ---------------------------------------------------------------------------
# The call-site assertion
# ---------------------------------------------------------------------------

def _fence(entry, *, where, match):
    """Run ``entry`` and prove the #677 refusal came FROM ``where``.

    ``where`` is ``(module basename, enclosing function name)`` of the call
    site under test. The frame check is the load-bearing half: matching the
    message alone would also pass if the helper were called from anywhere
    else, which is exactly the hole this file closes.
    """
    with pytest.raises(ValueError, match=match) as ei:
        entry()
    frames = [(os.path.basename(str(t.path)), t.name) for t in ei.traceback]
    assert where in frames, (
        f"the refusal did not pass through {where} — the call site may have "
        f"been dropped and a different fence caught it. frames={frames}")
    return ei


# ---------------------------------------------------------------------------
# refuse_f0_sheets() call sites
# ---------------------------------------------------------------------------

def test_fence_adi_run():
    """Honest scope note for BOTH ADI tests, measured 2026-08-19: with this
    call site deleted the ADI lane still refuses, from a pre-existing guard
    that predates #677 — ``_preflight._validate_adi_configuration`` rejects
    ANY registered thin conductor
    (``solver='adi' does not support thin-conductor corrections yet``). So
    ADI's exposure to a silently sheet-free run was lower than the other
    lanes' even before #677; what these two tests pin is that the #677
    fence is wired and names the right lane, not that it is ADI's only
    net."""
    _fence(lambda: _cube(solver="adi").run(n_steps=4, skip_preflight=True),
           where=("_execute.py", "run"),
           match=r"on the ADI run\(\) lane")


def test_fence_adi_forward():
    """See the scope note on ``test_fence_adi_run``: ADI carries a second,
    older thin-conductor guard behind this fence on both entry points."""
    _fence(lambda: _cube(solver="adi").forward(n_steps=4, skip_preflight=True),
           where=("_execute.py", "_forward_from_materials"),
           match=r"on the ADI forward lane")


def test_fence_subgridded_run():
    def go():
        sim = _cube()
        sim.add_refinement((8e-3, 12e-3), ratio=2)
        sim.run(n_steps=4, skip_preflight=True)
    _fence(go, where=("_execute.py", "run"),
           match=r"on the subgridded \(SBP-SAT\) run\(\) lane")


def test_fence_distributed_multidevice_run():
    """Two device handles select the lane; the fence fires before the runner
    is imported, so this needs no real second device (and therefore no
    process-global ``XLA_FLAGS`` device-count flip at import time)."""
    d = jax.devices()[0]
    _fence(lambda: _cube().run(n_steps=4, skip_preflight=True, devices=[d, d]),
           where=("_execute.py", "run"),
           match=r"on the distributed multi-device run\(\) lane")


def test_fence_distributed_nonuniform_forward():
    def go():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nu_cube().forward(n_steps=4, skip_preflight=True,
                               distributed=True)
    _fence(go, where=("_execute.py", "forward"),
           match=r"on the distributed non-uniform forward\(\) lane")


def test_fence_msl_s_matrix():
    _fence(lambda: _msl_thru().compute_msl_s_matrix(n_steps=4, n_freqs=3),
           where=("_sparams.py", "compute_msl_s_matrix"),
           match=r"on the MSL S-parameter lane")


def test_fence_msl_junction_mixed_s_matrix():
    def go():
        _mixed_probe_fed_msl().compute_mixed_s_matrix(
            freqs=np.linspace(1e9, 4e9, 3), num_periods=1.0,
            skip_preflight=True)
    _fence(go, where=("_sparams.py", "compute_mixed_s_matrix"),
           match=r"on the MSL junction S-parameter lane")


def test_fence_optimize_uniform():
    from rfx.optimize import DesignRegion, optimize

    def go():
        region = DesignRegion(corner_lo=(7e-3, 7e-3, 7e-3),
                              corner_hi=(12e-3, 12e-3, 12e-3),
                              eps_range=(1.0, 4.4))
        optimize(_design_cube(), region,
                 lambda r: -jnp.sum(r.time_series ** 2),
                 n_iters=1, lr=0.01, verbose=False, n_steps=4,
                 skip_preflight=True)
    _fence(go, where=("optimize.py", "optimize"),
           match=r"on the optimize\(\) design lane")


def test_fence_optimize_nonuniform():
    from rfx.optimize import DesignRegion, optimize

    def go():
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.0), dx=1e-3,
                         dz_profile=[1e-3] * 20, boundary="pec")
        _sheet(sim)
        sim.add_port((5e-3, 10e-3, 10e-3), "ez")
        sim.add_probe((14e-3, 10e-3, 10e-3), "ez")
        region = DesignRegion(corner_lo=(7e-3, 7e-3, 7e-3),
                              corner_hi=(12e-3, 12e-3, 12e-3),
                              eps_range=(1.0, 4.4))
        optimize(sim, region, lambda r: -jnp.sum(r.time_series ** 2),
                 n_iters=1, lr=0.01, verbose=False, n_steps=4,
                 skip_preflight=True)
    _fence(go, where=("optimize.py", "optimize"),
           match=r"on the optimize\(\) non-uniform design lane")


def test_fence_gradient_check():
    from rfx.optimize import gradient_check

    def go():
        sim = _design_cube()
        dp = jnp.zeros(sim._build_grid().shape, dtype=jnp.float32)
        gradient_check(sim, dp, lambda r: jnp.sum(r.time_series ** 2),
                       eps=1e-2, n_steps=4)
    _fence(go, where=("optimize.py", "gradient_check"),
           match=r"on the gradient-check lane")


def test_fence_topology_optimize():
    """The fence sits ABOVE ``topology_optimize``'s optional ``import optax``
    precisely so this test binds in an environment without the
    ``[optimization]`` extra — the default CI image."""
    from rfx.topology import TopologyDesignRegion, topology_optimize

    def go():
        region = TopologyDesignRegion(corner_lo=(7e-3, 7e-3, 7e-3),
                                      corner_hi=(12e-3, 12e-3, 12e-3),
                                      material_bg="air", material_fg="fr4",
                                      beta_projection=1.0)
        topology_optimize(_design_cube(), region,
                          lambda r: -jnp.sum(r.time_series ** 2),
                          n_iterations=1, learning_rate=0.05,
                          beta_schedule=[(0, 1.0)], verbose=False,
                          skip_preflight=True)
    _fence(go, where=("topology.py", "topology_optimize"),
           match=r"on the topology-optimization lane")


def test_fence_differentiable_material_fit():
    from rfx.differentiable_material_fit import differentiable_material_fit

    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = Simulation(freq_max=5e9, domain=(0.024, 0.009, 0.009))
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
        sim.add(Box((0.010, 0.0, 0.0), (0.016, 0.009, 0.009)), material="dut")
        _sheet(sim, Box((0.004, 0.002, 0.004), (0.008, 0.007, 0.004)))
        sim.add_port((0.003, 0.0045, 0.0045), "ez",
                     waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
        sim.add_probe((0.020, 0.0045, 0.0045), component="ez")
        sim.add_probe((0.007, 0.0045, 0.0045), component="ez")
        return sim

    def go():
        differentiable_material_fit(
            factory, np.zeros((1, 1, 3), complex),
            np.linspace(2.0e9, 4.0e9, 3), n_debye_poles=1,
            n_iterations=1, learning_rate=0.0, verbose=False)
    # the fence lives in the fit's inner ``forward(p)`` closure
    _fence(go, where=("differentiable_material_fit.py", "forward"),
           match=r"on the differentiable material fit lane")


# ---------------------------------------------------------------------------
# Inline "(#677 v1)" fences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{"subpixel_smoothing": True},
                                {"conformal_pec": True}])
def test_fence_uniform_run_subpixel_or_conformal(kw):
    _fence(lambda: _cube().run(n_steps=4, skip_preflight=True, **kw),
           where=("uniform.py", "run_uniform"),
           match=r"not supported with subpixel_smoothing / conformal_pec")


def test_fence_nonuniform_run_anisotropic_eps():
    """The NU fence is conditioned on ``aniso_eps`` actually being built,
    which needs registered geometry — ``subpixel_smoothing=True`` on a
    geometry-free model leaves the plain isotropic update the sheet
    operator is correct against, and is deliberately NOT refused."""
    def go():
        sim = _nu_cube()
        sim.add_material("diel", eps_r=4.0)
        sim.add(Box((2e-3, 2e-3, 2e-3), (6e-3, 6e-3, 6e-3)), material="diel")
        sim.run(n_steps=4, skip_preflight=True, subpixel_smoothing=True)
    _fence(go, where=("nonuniform.py", "run_nonuniform_path"),
           match=r"anisotropic permittivity")


def test_fence_forward_dispersive_overlap():
    """``run()``'s dispersive fence lives in ``run_uniform``; ``forward()``
    carries its OWN copy in ``_execute.forward``, so the message match is
    not enough to tell them apart — the frame is."""
    def go():
        sim = _cube()
        sim.add_material("water", eps_r=4.0,
                         debye_poles=[DebyePole(delta_eps=2.0, tau=8.3e-12)])
        sim.add(Box((2e-3, 2e-3, 2e-3), (6e-3, 6e-3, 6e-3)), material="water")
        sim.forward(n_steps=4, skip_preflight=True)
    _fence(go, where=("_execute.py", "forward"),
           match=r"dispersive \(Debye/Lorentz\) materials")


def test_fence_waveguide_s_matrix_subpixel():
    def go():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _wr90().compute_waveguide_s_matrix(n_steps=4,
                                               subpixel_smoothing=True)
    _fence(go, where=("_sparams.py", "compute_waveguide_s_matrix"),
           match=r"on the waveguide S-matrix lane")


def test_fence_waveguide_s_matrix_multimode():
    def go():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _wr90(n_modes=2).compute_waveguide_s_matrix(n_steps=4)
    _fence(go, where=("_sparams.py", "compute_waveguide_s_matrix"),
           match=r"on the multimode waveguide S-matrix path")


# ---------------------------------------------------------------------------
# vmap sweep: not a refusal — a fallback that must still APPLY the sheet
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_vmap_sweep_fallback_still_applies_the_sheet():
    """``_build_full_scan_fn`` returns ``(None, None)`` for a sheet-bearing
    sim (pinned at unit level in test_leontovich_sheet_identity.py) so
    ``vmap_material_sweep`` takes the sequential fallback. Completing is not
    the property that matters — a sheet-free sweep would complete too. The
    witness is that the swept fields DIFFER from the same sweep with the
    sheet removed, i.e. the fallback really applied the operator ctx."""
    from rfx.vmap_sweep import vmap_material_sweep

    def build(with_sheet):
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                         boundary="pec")
        sim.add_material("substrate", eps_r=4.0)
        sim.add(Box((2e-3, 2e-3, 2e-3), (18e-3, 18e-3, 9e-3)),
                material="substrate")
        if with_sheet:
            _sheet(sim, Box((4e-3, 4e-3, 12e-3), (16e-3, 16e-3, 12e-3)))
        sim.add_source((10e-3, 10e-3, 5e-3), "ex",
                       waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
        sim.add_probe((10e-3, 10e-3, 16e-3), "ex")
        return sim

    values = np.array([2.0, 6.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = vmap_material_sweep(build(True), "substrate.eps_r", values,
                                 n_steps=200)
        off = vmap_material_sweep(build(False), "substrate.eps_r", values,
                                  n_steps=200)

    ts_on = np.asarray(on.time_series)
    ts_off = np.asarray(off.time_series)
    assert ts_on.shape == ts_off.shape
    assert np.all(np.isfinite(ts_on)), "sheet-bearing sweep went non-finite"
    scale = np.abs(ts_off).max()
    assert scale > 0.0, "sheet-free control recorded no field — bad fixture"
    rel = np.abs(ts_on - ts_off).max() / scale
    assert rel > 1e-3, (
        f"sheet-on and sheet-free sweeps agree to {rel:.2e} — the sequential "
        f"fallback appears to have simulated a sheet-FREE model")


# ---------------------------------------------------------------------------
# Inventory guard: no fence may exist without a named pinning test
# ---------------------------------------------------------------------------

#: ``(rel path, enclosing function, discriminant)`` -> ``(test module, test)``
#:
#: ``discriminant`` is the lane string for ``refuse_f0_sheets`` sites, and a
#: distinctive message fragment for the inline ``(#677 v1)`` fences.
FENCE_REGISTRY: dict[tuple[str, str, str], tuple[str, str]] = {
    # --- refuse_f0_sheets(...) call sites ---
    ("rfx/api/_execute.py", "_forward_from_materials", "ADI forward"):
        (__name__, "test_fence_adi_forward"),
    ("rfx/api/_execute.py", "forward", "distributed non-uniform forward()"):
        (__name__, "test_fence_distributed_nonuniform_forward"),
    ("rfx/api/_execute.py", "run", "distributed multi-device run()"):
        (__name__, "test_fence_distributed_multidevice_run"),
    ("rfx/api/_execute.py", "run", "ADI run()"):
        (__name__, "test_fence_adi_run"),
    ("rfx/api/_execute.py", "run", "subgridded (SBP-SAT) run()"):
        (__name__, "test_fence_subgridded_run"),
    ("rfx/api/_sparams.py", "compute_msl_s_matrix", "MSL S-parameter"):
        (__name__, "test_fence_msl_s_matrix"),
    ("rfx/api/_sparams.py", "compute_mixed_s_matrix",
     "MSL junction S-parameter"):
        (__name__, "test_fence_msl_junction_mixed_s_matrix"),
    ("rfx/differentiable_material_fit.py", "forward",
     "differentiable material fit"):
        (__name__, "test_fence_differentiable_material_fit"),
    ("rfx/optimize.py", "optimize", "optimize() non-uniform design"):
        (__name__, "test_fence_optimize_nonuniform"),
    ("rfx/optimize.py", "optimize", "optimize() design"):
        (__name__, "test_fence_optimize_uniform"),
    ("rfx/optimize.py", "gradient_check", "gradient-check"):
        (__name__, "test_fence_gradient_check"),
    ("rfx/topology.py", "topology_optimize", "topology-optimization"):
        (__name__, "test_fence_topology_optimize"),
    ("rfx/runners/distributed.py", "run_distributed", "distributed (v1) runner"):
        ("tests.test_sheet_impedance_operator",
         "test_g9_distributed_runners_refuse"),
    ("rfx/runners/distributed_v2.py", "run_distributed",
     "distributed (v2) runner"):
        ("tests.test_sheet_impedance_operator",
         "test_g9_distributed_runners_refuse"),

    # --- inline "(#677 v1)" fences ---
    ("rfx/api/_execute.py", "forward", "dispersive (Debye/Lorentz)"):
        (__name__, "test_fence_forward_dispersive_overlap"),
    ("rfx/api/_sparams.py", "compute_waveguide_s_matrix",
     "waveguide S-matrix lane"):
        (__name__, "test_fence_waveguide_s_matrix_subpixel"),
    ("rfx/api/_sparams.py", "compute_waveguide_s_matrix",
     "multimode waveguide S-matrix path"):
        (__name__, "test_fence_waveguide_s_matrix_multimode"),
    ("rfx/runners/uniform.py", "run_uniform", "boundary='upml'"):
        ("tests.test_sheet_impedance_operator", "test_g9_upml_refuses"),
    ("rfx/runners/uniform.py", "run_uniform",
     "subpixel_smoothing / conformal_pec"):
        (__name__, "test_fence_uniform_run_subpixel_or_conformal"),
    ("rfx/runners/uniform.py", "run_uniform", "dispersive (Debye/Lorentz)"):
        ("tests.test_sheet_impedance_operator",
         "test_g9_dispersive_overlap_refuses_uniform_and_nu"),
    ("rfx/runners/nonuniform.py", "run_nonuniform_path",
     "dispersive (Debye/Lorentz)"):
        ("tests.test_sheet_impedance_operator",
         "test_g9_dispersive_overlap_refuses_uniform_and_nu"),
    ("rfx/runners/nonuniform.py", "run_nonuniform_path",
     "anisotropic permittivity"):
        (__name__, "test_fence_nonuniform_run_anisotropic_eps"),

    # --- has_f0_sheets(...) eligibility fences (no raise: a fallback) ---
    ("rfx/vmap_sweep.py", "_build_full_scan_fn", "has_f0_sheets"):
        ("tests.test_leontovich_sheet_identity",
         "test_vmap_parity_o7"),
}

#: The GPU baked fast path excludes sheets through a ctx flag rather than a
#: call the AST scan can see (``rfx/simulation.py``'s ``_fast_eligible``
#: reads ``not _ctx["use_sheet_impedance"]``), so it is registered by hand.
_MANUAL_FENCES: dict[tuple[str, str, str], tuple[str, str]] = {
    ("rfx/simulation.py", "run", "use_sheet_impedance"):
        ("tests.test_sheet_impedance_operator",
         "test_g9_fast_path_excludes_sheets"),
}

_RFX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rfx")


def _enclosing_function(func_spans, lineno):
    cands = [f for f in func_spans if f[0] <= lineno <= f[1]]
    cands.sort(key=lambda f: f[1] - f[0])
    return cands[0][2] if cands else "<module>"


def _scan_source_fences():
    """Re-derive the fence inventory from the ``rfx/`` source by AST."""
    found: dict[tuple[str, str, str], str] = {}
    for dirpath, _dirs, files in os.walk(_RFX_ROOT):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, os.path.dirname(_RFX_ROOT))
            rel = rel.replace(os.sep, "/")
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            if "677" not in src and "f0_sheets" not in src:
                continue
            tree = ast.parse(src, filename=path)
            spans = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for node in ast.walk(tree):
                # refuse_f0_sheets(conductors, "<lane>")
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id.lstrip("_").startswith("refuse_f0")
                        and len(node.args) == 2
                        and isinstance(node.args[1], ast.Constant)):
                    key = (rel, _enclosing_function(spans, node.lineno),
                           node.args[1].value)
                    found[key] = "refuse_f0_sheets"
                # has_f0_sheets(...) used as a lane eligibility fence
                elif (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "has_f0_sheets"
                        and rel != "rfx/materials/thin_conductor.py"):
                    key = (rel, _enclosing_function(spans, node.lineno),
                           "has_f0_sheets")
                    found[key] = "has_f0_sheets"
                # raise ValueError("... (#677 v1) ...")
                elif isinstance(node, ast.Raise):
                    exc = node.exc
                    if not (isinstance(exc, ast.Call) and exc.args
                            and isinstance(exc.args[0], ast.Constant)
                            and isinstance(exc.args[0].value, str)):
                        continue
                    msg = exc.args[0].value
                    if "(#677 v1)" not in msg:
                        continue
                    func = _enclosing_function(spans, node.lineno)
                    disc = [k for (_r, _f, k) in FENCE_REGISTRY
                            if _r == rel and _f == func and k in msg]
                    key = (rel, func, disc[0] if disc else msg)
                    found[key] = "inline"
    return found


def test_every_fence_in_the_source_is_pinned():
    """Every #677 lane fence in ``rfx/`` is named by ``FENCE_REGISTRY``.

    This is the anti-regression that closes the CLASS: adding a new fence
    without a lane-level test is red here, and so is deleting a fence whose
    lane is still registered (the registry would name a call site the
    source no longer has).
    """
    found = _scan_source_fences()
    registered = set(FENCE_REGISTRY)
    missing = sorted(set(found) - registered)
    stale = sorted(registered - set(found))
    assert not missing, (
        "these #677 lane fences exist in rfx/ but no test is registered for "
        f"them — add a lane-level test and a FENCE_REGISTRY row: {missing}")
    assert not stale, (
        "FENCE_REGISTRY names call sites that are no longer in rfx/ — a "
        f"fence was moved or deleted: {stale}")


def test_every_registered_pinning_test_exists():
    """A registry row is only worth something if the test it names is real —
    a rename must not silently orphan a fence."""
    orphans = []
    for key, (mod_name, test_name) in {**FENCE_REGISTRY,
                                       **_MANUAL_FENCES}.items():
        mod = (importlib.import_module(mod_name)
               if mod_name != __name__ else globals())
        have = (test_name in mod if isinstance(mod, dict)
                else hasattr(mod, test_name))
        if not have:
            orphans.append((key, mod_name, test_name))
    assert not orphans, f"registered pinning tests that do not exist: {orphans}"
