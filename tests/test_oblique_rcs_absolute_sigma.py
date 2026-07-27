"""Absolute oblique sigma calibration + Method-B boundary guards (issue #471).

F7 — the absolute-sigma normalization is the MEASURED 1D-aux incident spectrum
(``measure_incident_spectrum``), not the analytic normal-path pulse. Audit
record (2026-07-27 session, scratchpad ``f7_falsifier_declaration.md`` +
``f7_audit{1,2,3}.py`` outputs, quoted in PR):

* pre-fix inflation +6.58 dB (theta=20 deg) / +4.51 dB (40 deg) on the gate
  grid = waveform mismatch +8.83 dB (the #471 review's analytic numbers,
  reproduced exactly) + a theta-dependent source->aux launch/absorption
  response (-2.25 / -4.32 dB) the review neglected;
* post-fix specular-peak Delta vs the PO uniform-aperture oracle:
  +0.86 dB (20 deg) / +0.85 dB (40 deg), lobe-flat over +/-12 deg,
  settling witness -60.7 / -50.1 dB (R5);
* aperture-height convention: sigma(span4)/sigma(span2) = 4.000 measured
  -> h_eff = (k_hi - k_lo)*dx exactly (no off-by-one node);
* RESOLUTION SENSITIVITY (corrected-G4, pinned thickness 2 mm + aperture
  4 mm): Delta moves +0.86 -> -1.55 dB from dx=2 mm to dx=1 mm — a 2.4 dB
  shift, FAILING the pre-declared 1.0 dB invariance bound. Absolute oblique
  sigma is therefore a +/-2 dB-CLASS number at lambda/30..lambda/60. The
  slow gate below pins |Delta| <= 1.5 dB AT THE FIXED GATE GRID as a
  regression lock, not an accuracy claim.

F5 — the 4-edge Method-B box requires vacuum on ALL FOUR transverse boundary
planes; the pre-#471 validator checked only x planes and ``compute_rcs`` ran
no check at all. Fire/clear pairs below cover both lanes.

F8 — the non-uniform-grid oblique fence gets its own test (both prior fence
tests asserted only the ey path).
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import MaterialArrays, init_materials
from rfx.geometry.csg import Box, rasterize
from rfx.farfield import NTFFBox
from rfx.simulation import run
from rfx.rcs import compute_rcs, compute_rcs_jax, _incident_spectrum_amplitude
from rfx.sources.tfsf_oblique_open import (
    init_tfsf_methodB,
    measure_incident_spectrum,
    validate_vacuum_boundary,
)

F0 = 5e9
LAM = C0 / F0
DX = 0.002
CPML = 10
MARGIN = 10
PLATE_W = 2.2 * LAM
N_STEPS = 700
PEC_SIGMA = 1e7
DOMAIN = (0.11, 0.218, 0.006)
PHI = np.linspace(0.0, 2 * np.pi, 361)
TH_OBS = np.array([np.pi / 2])


def _po_sigma(phi_obs, theta_i_deg, W, h, lam):
    """PO / uniform-aperture bistatic sigma(phi) at theta_obs=pi/2 for a PEC
    plate in the y-z plane (normal-x lit face), ez polarization. Numeric
    aperture integral; the closed form below is its independent check."""
    k = 2 * np.pi / lam
    t = np.radians(theta_i_deg)
    y = np.linspace(-W / 2, W / 2, 1201)
    out = np.zeros_like(phi_obs, dtype=np.float64)
    for i, p in enumerate(phi_obs):
        iy = np.trapz(np.exp(1j * k * (np.sin(p) - np.sin(t)) * y), y)
        out[i] = 4 * np.pi * ((k / (4 * np.pi)) * 2 * np.cos(t) * np.abs(iy) * h) ** 2
    return out


def _gate_grid_and_plate():
    grid = Grid(freq_max=F0 * 3, domain=DOMAIN, dx=DX, cpml_layers=CPML)
    nx, ny, nz = grid.shape
    xc = (nx // 2) * DX
    yc = (ny // 2) * DX
    plate = Box(corner_lo=(xc - 0.5 * DX, yc - PLATE_W / 2, -1.0),
                corner_hi=(xc + 0.5 * DX, yc + PLATE_W / 2, +1.0))
    eps_r, sigma = rasterize(grid, [(plate, 1.0, PEC_SIGMA)])
    n_w = int(np.max(np.sum(np.asarray(sigma) > 0, axis=1)))
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma,
                          mu_r=jnp.ones(grid.shape, dtype=jnp.float32))
    return grid, mats, n_w


# ---------------------------------------------------------------------------
# Oracle self-check (comparator-first: PO must reproduce its closed form
# before it referees any rfx number)
# ---------------------------------------------------------------------------

def test_po_oracle_matches_closed_form():
    """The numeric aperture integral must reproduce
    sigma = 4*pi*(W*h*cos(theta_i)/lambda)^2 at the specular direction to
    <0.1 dB at theta_i in {0, 20, 40} deg (independent algebra: integral vs
    closed form)."""
    W, h = 2.2 * LAM, 2 * DX
    for t_i in (0.0, 20.0, 40.0):
        spec = np.radians(180.0 - t_i)
        num = _po_sigma(np.array([spec]), t_i, W, h, LAM)[0]
        closed = 4 * np.pi * (W * h * np.cos(np.radians(t_i)) / LAM) ** 2
        assert abs(10 * np.log10(num / closed)) < 0.1, (t_i, num, closed)


# ---------------------------------------------------------------------------
# F7 fast: the measured normalization captures what the analytic one cannot
# ---------------------------------------------------------------------------

def test_measured_incident_spectrum_captures_mismatch_and_launch():
    """Envelope pins around the measured facts (gate grid, f0):

    * assumed-vs-measured mismatch 6.58 dB at theta=20 (pinned 4..9) and
      4.51 dB at theta=40 (pinned 2.5..7) — nonzero mismatch IS the F7 bug
      class; a normalization that just re-DFTs the assumed pulse fails the
      lower bound (falsifier);
    * measured vs the analytic ACTUAL waveform (launch response) is a mild
    attenuation, -2.25 dB at theta=20 (pinned -4..+0.5) — the term the
      #471 review neglected.
    """
    grid = Grid(freq_max=F0 * 3, domain=DOMAIN, dx=DX, cpml_layers=CPML)
    nx, ny, nz = grid.shape
    dt = grid.dt
    freqs = np.array([F0])
    bands = {20.0: (4.0, 9.0), 40.0: (2.5, 7.0)}
    for theta, (lo, hi) in bands.items():
        cfg, st = init_tfsf_methodB(nx, ny, DX, dt, nz=nz, cpml_layers=CPML,
                                    tfsf_margin=MARGIN, f0=F0,
                                    polarization="ez", direction="+x",
                                    theta_deg=theta)
        S_meas = measure_incident_spectrum(cfg, st, N_STEPS, freqs, DX)
        S_ass = _incident_spectrum_amplitude(F0, 0.5, freqs, dt, N_STEPS)
        mismatch_db = 20 * np.log10(np.abs(S_meas[0]) / np.abs(S_ass[0]))
        assert lo <= mismatch_db <= hi, (theta, mismatch_db)
        if theta == 20.0:
            times = np.arange(N_STEPS) * dt
            arg = (times - cfg.src_t0) / cfg.src_tau
            s_act = np.cos(2 * np.pi * F0 * (times - cfg.src_t0)) * np.exp(-arg ** 2)
            S_act = np.sum(s_act * np.exp(-1j * 2 * np.pi * F0 * times)) * dt
            launch_db = 20 * np.log10(np.abs(S_meas[0]) / np.abs(S_act))
            assert -4.0 <= launch_db <= 0.5, launch_db


# ---------------------------------------------------------------------------
# F5: vacuum-boundary guard — fire / clear pairs on both lanes
# ---------------------------------------------------------------------------

def _small_oblique_setup(strip_iy):
    grid = Grid(freq_max=F0 * 3, domain=(0.06, 0.08, 0.006), dx=DX,
                cpml_layers=6)
    nx, ny, nz = grid.shape
    eps = jnp.ones(grid.shape, dtype=jnp.float32)
    sig = jnp.zeros(grid.shape, dtype=jnp.float32)
    off = 6 + 6  # cpml + margin -> box y_lo/x_lo
    if strip_iy is not None:
        sig = sig.at[off + 3:nx - off - 3, strip_iy, :].set(PEC_SIGMA)
    mats = MaterialArrays(eps_r=eps, sigma=sig,
                          mu_r=jnp.ones(grid.shape, dtype=jnp.float32))
    return grid, mats, off


def test_compute_rcs_vacuum_guard_fires_on_y_plane():
    """A PEC strip lying exactly on the TFSF y_lo plane must fail loud in
    compute_rcs (issue #471 F5: it previously passed silently — compute_rcs
    ran no vacuum check and the preflight validator checked x planes only)."""
    grid, mats, off = _small_oblique_setup(strip_iy=12)  # iy == y_lo plane
    with pytest.raises(ValueError, match="y_lo"):
        compute_rcs(grid, mats, n_steps=8, f0=F0, bandwidth=0.5,
                    theta_inc=20.0, polarization="ez", theta_obs=TH_OBS,
                    phi_obs=np.array([np.pi]), freqs=np.array([F0]),
                    cpml_layers=6, tfsf_margin=6)


def test_compute_rcs_vacuum_guard_clears_interior_target():
    """Control: the same strip moved to the box interior passes the guard
    (compute_rcs completes a short run without the validator firing)."""
    grid, mats, off = _small_oblique_setup(strip_iy=20)  # interior
    res = compute_rcs(grid, mats, n_steps=8, f0=F0, bandwidth=0.5,
                      theta_inc=20.0, polarization="ez", theta_obs=TH_OBS,
                      phi_obs=np.array([np.pi]), freqs=np.array([F0]),
                      cpml_layers=6, tfsf_margin=6)
    assert np.all(np.isfinite(np.asarray(res.rcs_linear)))


def test_validator_covers_x_planes_too():
    """The unified Method-B validator still rejects the x-plane violation the
    old check caught (no coverage regression from the F5 extension)."""
    grid, mats, off = _small_oblique_setup(strip_iy=None)
    nx, ny, nz = grid.shape
    cfg, _ = init_tfsf_methodB(nx, ny, DX, grid.dt, nz=nz, cpml_layers=6,
                               tfsf_margin=6, f0=F0, polarization="ez",
                               direction="+x", theta_deg=20.0)
    sig = mats.sigma.at[cfg.x_lo, 20:24, :].set(PEC_SIGMA)
    bad = MaterialArrays(eps_r=mats.eps_r, sigma=sig, mu_r=mats.mu_r)
    with pytest.raises(ValueError, match="x_lo"):
        validate_vacuum_boundary(bad, cfg)


def test_simulation_lane_vacuum_guard_fires_on_y_plane():
    """Simulation API lane (the runner-level validator): a dielectric box
    crossing the Method-B y_lo TFSF plane must raise. This is the exact
    verified silent failure of issue #471 F5. skip_preflight targets the
    runner-level validator, which runs regardless."""
    from rfx.api import Simulation

    sim = Simulation(freq_max=10e9, domain=(0.08, 0.10, 0.006), dx=DX,
                     boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=F0, polarization="ez", direction="+x",
                        angle_deg=20.0, method="methodB", margin=3)
    sim.add_material("diel", eps_r=4.0)
    # Simulation Box coords are relative to the PHYSICAL domain origin (CPML
    # pads excluded), so the y_lo plane (padded index cpml+margin) sits at
    # physical y = margin * dx.
    y_plane = 3 * DX
    sim.add(Box((0.036, y_plane - DX, 0.0), (0.05, y_plane + DX, 0.006)),
            material="diel")
    with pytest.raises(ValueError, match="y_lo|vacuum"):
        sim.run(n_steps=2, skip_preflight=True)


# ---------------------------------------------------------------------------
# F8: the non-uniform-grid oblique fence, tested on the ez path
# ---------------------------------------------------------------------------

def test_oblique_nu_grid_fence_fails_loud_on_ez():
    """compute_rcs(theta_inc != 0) on a NonUniformGrid must raise the
    non-uniform NotImplementedError even for the SUPPORTED ez polarization
    (both prior fence tests asserted only the ey rejection, which fires
    earlier and never reaches the grid check)."""
    from rfx.nonuniform import make_nonuniform_grid

    grid = make_nonuniform_grid((0.04, 0.04), np.full(6, DX), DX,
                                cpml_layers=4)
    mats = init_materials(grid.shape)
    with pytest.raises(NotImplementedError, match="non-uniform"):
        compute_rcs(grid, mats, n_steps=8, f0=F0, bandwidth=0.5,
                    theta_inc=20.0, polarization="ez", theta_obs=TH_OBS,
                    phi_obs=np.array([np.pi]), freqs=np.array([F0]))


# ---------------------------------------------------------------------------
# F7 slow: the absolute-sigma gates (regression locks at the gate grid)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("theta_inc", [20.0, 40.0])
def test_absolute_sigma_matches_po_at_gate_grid(theta_inc):
    """PUBLIC-path lock: compute_rcs specular-peak sigma vs the PO
    uniform-aperture oracle (W = rasterized plate width, h = 2*dx) within
    1.5 dB at the FIXED gate grid. Measured 2026-07-27: +0.86 dB (20 deg),
    +0.85 dB (40 deg); settling witness -60.7/-50.1 dB (sibling direct-
    pipeline test measures it on this exact config). This pins the fixed-
    discretization envelope — the measured lambda/30->lambda/60 resolution
    sensitivity is 2.4 dB (module docstring), so this is a regression lock,
    not an accuracy claim."""
    grid, mats, n_w = _gate_grid_and_plate()
    res = compute_rcs(grid, mats, N_STEPS, f0=F0, bandwidth=0.5,
                      theta_inc=theta_inc, polarization="ez",
                      theta_obs=TH_OBS, phi_obs=PHI, freqs=np.array([F0]),
                      cpml_layers=CPML, tfsf_margin=MARGIN, ntff_offset=3)
    sb = int(round(180.0 - theta_inc))
    sigma_peak = float(np.asarray(res.rcs_linear)[0, 0, sb])
    po = _po_sigma(np.array([np.radians(180.0 - theta_inc)]), theta_inc,
                   n_w * DX, 2 * DX, LAM)[0]
    delta_db = 10 * np.log10(sigma_peak / po)
    assert abs(delta_db) <= 1.5, (
        f"absolute sigma vs PO drifted: {delta_db:+.2f} dB at "
        f"theta_inc={theta_inc} (measured +0.86/+0.85 when locked)"
    )


@pytest.mark.slow
def test_sigma_scales_as_aperture_height_squared():
    """z-span convention pin: doubling the NTFF k-span doubles h_eff, so
    sigma must scale by 4.000 (measured exactly at both dx=2 mm and 1 mm).
    Guards the h_eff=(k_hi-k_lo)*dx statement in the compute_rcs docstring
    against a future off-by-one in the box quadrature."""
    grid, mats, _ = _gate_grid_and_plate()
    nx, ny, nz = grid.shape
    dt = grid.dt
    sigs = {}
    for k_half in (1, 2):
        cfg, st = init_tfsf_methodB(nx, ny, DX, dt, nz=nz, cpml_layers=CPML,
                                    tfsf_margin=MARGIN, f0=F0,
                                    polarization="ez", direction="+x",
                                    theta_deg=20.0)
        kz = nz // 2
        box = NTFFBox.from_grid(
            grid, i_lo=cfg.x_lo - 3, i_hi=cfg.x_hi + 3,
            j_lo=cfg.y_lo - 3, j_hi=cfg.y_hi + 3,
            k_lo=kz - k_half, k_hi=kz + k_half,
            freqs=jnp.asarray([F0], jnp.float32))
        res = run(grid, mats, N_STEPS, boundary="cpml", cpml_axes="xy",
                  periodic=(False, False, True), pec_axes="",
                  tfsf=(cfg, st), ntff=box)
        S = measure_incident_spectrum(cfg, st, N_STEPS, np.array([F0]), DX)
        sig = np.asarray(compute_rcs_jax(res.ntff_data, box, grid, TH_OBS,
                                         PHI, S))[0, 0, :]
        sigs[k_half] = sig[int(round(180.0 - 20.0))]
    ratio = sigs[2] / sigs[1]
    assert 3.8 <= ratio <= 4.2, ratio
