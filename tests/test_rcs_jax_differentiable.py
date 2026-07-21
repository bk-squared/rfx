"""Differentiable RCS (``compute_rcs_jax``) — equivalence to the validated numpy path + gradient.

``compute_rcs`` is a numpy orchestrator (not on the AD tape); its monostatic bin is
Mie-validated (#280). ``compute_rcs_jax`` is the differentiable POST-PROCESSOR on
``compute_far_field_jax``, so ``jax.grad`` flows scatterer-ε → RCS for RCS-reduction /
-shaping inverse design.

Three gates (ground truths):
  • ``test_rcs_jax_equals_numpy_on_same_ntff_data`` — jnp RCS reproduces the numpy RCS
    formula on the SAME NTFF data (exact; inherits the Mie validation of the numpy path).
  • ``test_rcs_jax_matches_validated_orchestrator`` — the replicated TFSF+NTFF setup
    reproduces ``compute_rcs(...).monostatic_rcs`` (ties the harness to the validated path).
  • ``test_rcs_jax_gradient_fd_convergence`` — dσ_backscatter/dε is finite and central-FD
    CONVERGES to AD as h→0. σ(ε) sits near a backscatter maximum here (strongly curved), so
    a single large-h FD is unreliable (ch07 doctrine: step-sweep, not single-h).

Harness: docs/research_notes/experiments/i404_oblique_20260720/rcs_jax_validate.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.farfield import NTFFBox, compute_far_field
from rfx.sources.tfsf import init_tfsf
from rfx.simulation import run
from rfx.rcs import compute_rcs, compute_rcs_jax, _incident_spectrum_amplitude

F0, BW = 10e9, 0.5
CPML = 8
N_STEPS = 220
DOMAIN = (0.09, 0.09, 0.09)
DX = 0.003
FREQS = np.array([F0])
TH_B, PH_B = np.array([np.pi / 2]), np.array([np.pi])  # backscatter for +x incidence


def _materials(grid, eps_scale):
    mb = init_materials(grid.shape)
    sx, sy, sz = grid.shape
    block = np.zeros(grid.shape, np.float32)
    block[sx // 2 - 3:sx // 2 + 3, sy // 2 - 3:sy // 2 + 3, sz // 2 - 3:sz // 2 + 3] = 1.0
    return mb._replace(eps_r=mb.eps_r + eps_scale * 3.0 * jnp.asarray(block))  # ε_r=4 @ scale=1


def _setup_run(grid, materials):
    cfg, st = init_tfsf(nx=grid.nx, dx=grid.dx, dt=grid.dt, cpml_layers=CPML, tfsf_margin=3,
                        f0=F0, bandwidth=BW, amplitude=1.0, polarization="ez",
                        direction="+x", angle_deg=0.0)
    fl = getattr(grid, "face_layers", None) or {k: CPML for k in
                 ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")}
    box = NTFFBox.from_grid(
        grid,
        i_lo=max(cfg.x_lo - 1, 1), i_hi=min(cfg.x_hi + 2, grid.nx - 2),
        j_lo=max(fl["y_lo"] + 1, 1), j_hi=min(grid.ny - fl["y_hi"] - 1, grid.ny - 2),
        k_lo=max(fl["z_lo"] + 1, 1), k_hi=min(grid.nz - fl["z_hi"] - 1, grid.nz - 2),
        freqs=jnp.array(FREQS, jnp.float32))
    return run(grid, materials, N_STEPS, boundary="cpml", tfsf=(cfg, st), ntff=box).ntff_data, box


@pytest.fixture(scope="module")
def rcs_run():
    grid = Grid(freq_max=15e9, domain=DOMAIN, dx=DX, cpml_layers=CPML)
    e_inc = _incident_spectrum_amplitude(F0, BW, FREQS, grid.dt, N_STEPS)
    mats = _materials(grid, 1.0)
    ntff_data, box = _setup_run(grid, mats)
    return {"grid": grid, "e_inc": e_inc, "mats": mats, "ntff_data": ntff_data, "box": box}


@pytest.mark.slow
def test_rcs_jax_equals_numpy_on_same_ntff_data(rcs_run):
    """jnp RCS == numpy RCS formula on the SAME NTFF data (inherits the Mie validation)."""
    g, box, e_inc, nd = rcs_run["grid"], rcs_run["box"], rcs_run["e_inc"], rcs_run["ntff_data"]
    ff = compute_far_field(nd, box, g, TH_B, PH_B)
    p = np.abs(np.asarray(ff.E_theta)[:, 0, 0]) ** 2 + np.abs(np.asarray(ff.E_phi)[:, 0, 0]) ** 2
    sig_numpy = 4 * np.pi * p / np.abs(e_inc) ** 2
    sig_jax = np.asarray(compute_rcs_jax(nd, box, g, TH_B, PH_B, e_inc))[:, 0, 0]
    assert abs(sig_jax[0] - sig_numpy[0]) < 1e-4 * abs(sig_numpy[0]), \
        f"σ_jax={sig_jax[0]:.6e} vs σ_numpy={sig_numpy[0]:.6e}"


@pytest.mark.slow
def test_rcs_jax_matches_validated_orchestrator(rcs_run):
    """compute_rcs_jax monostatic reproduces compute_rcs(...).monostatic_rcs (the Mie-validated bin)."""
    g, box, e_inc, nd, mats = (rcs_run["grid"], rcs_run["box"], rcs_run["e_inc"],
                               rcs_run["ntff_data"], rcs_run["mats"])
    res = compute_rcs(g, mats, N_STEPS, f0=F0, bandwidth=BW, freqs=FREQS,
                      theta_obs=TH_B, phi_obs=PH_B)
    sig_jax = float(np.asarray(compute_rcs_jax(nd, box, g, TH_B, PH_B, e_inc))[0, 0, 0])
    mono_jax = 10 * np.log10(max(sig_jax, 1e-30))
    assert abs(float(res.monostatic_rcs[0]) - mono_jax) < 0.05, \
        f"compute_rcs mono={float(res.monostatic_rcs[0]):.3f} vs jax mono={mono_jax:.3f} dBsm"


@pytest.mark.slow
def test_rcs_jax_gradient_fd_convergence(rcs_run):
    """dσ_backscatter/dε is finite and central-FD CONVERGES to AD (ch07 step-sweep)."""
    g, e_inc = rcs_run["grid"], rcs_run["e_inc"]

    def sigma_of(eps_scale):
        nd, bx = _setup_run(g, _materials(g, eps_scale))
        return compute_rcs_jax(nd, bx, g, TH_B, PH_B, e_inc)[0, 0, 0]

    g_ad = float(jax.grad(sigma_of)(1.0))
    assert np.isfinite(g_ad) and abs(g_ad) > 1e-9, f"gradient collapsed/NaN: {g_ad}"
    errs = []
    for h in (0.01, 0.005):  # σ(ε) is curved near ε=1 => small h needed
        g_fd = (float(sigma_of(1.0 + h)) - float(sigma_of(1.0 - h))) / (2 * h)
        errs.append(abs(g_ad - g_fd) / abs(g_ad))
    assert errs[1] < errs[0], f"central-FD not converging to AD: {errs}"
    assert errs[1] < 0.05, f"smallest-h FD error {errs[1]*100:.1f}% (AD={g_ad:.3e})"
