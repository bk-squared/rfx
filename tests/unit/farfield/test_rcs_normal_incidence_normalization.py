"""Normal-incidence RCS divides by the incident field the grid actually carries (issue #820).

Physics. sigma = 4 pi |r E_scat|^2 / |E_inc|^2 needs the incident field AT THE TARGET. The
normal-incidence plane wave is launched by a 1-D auxiliary line that ADDS the source waveform
to one E node, a soft source: the wave it sends across the TF/SF plane is
1 / (2 S cos(k~ dx / 2)) of the waveform (S = c dt / dx, k~ the grid wavenumber): -1.14 dB at
40 cells per wavelength and the uniform-grid Courant number, -1.05 dB at 20. ``compute_rcs``
used to divide by the waveform's own DFT; sigma goes as |E_scat / E_inc|^2 and E_scat follows
the launched wave, so every normal-incidence sigma it reported was low by that same number of
dB (1.14 dB at 40 cells per wavelength -- a field-amplitude ratio and the power ratio it
implies have the same size in dB).

Two independent witnesses, neither of which shares code with the normalization:

* the 3-D total field at the centre of an EMPTY domain, recorded by a probe inside the very
  run ``compute_rcs`` makes; the denominator is read back from ``compute_rcs``'s own output
  (sigma and the far field it divided), so whatever the orchestrator divides by is what is
  checked -- not what a helper would return if it were called;
* the closed form of the 1-D soft-source launch, derived from the Yee update equations.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import rfx.rcs as rcs_mod
from rfx.core.yee import MaterialArrays
from rfx.farfield import compute_far_field as _real_compute_far_field
from rfx.grid import Grid, C0
from rfx.simulation import ProbeSpec, run as _real_run
from rfx.sources.tfsf import init_tfsf, measure_normal_incident_spectrum

F0 = 6e9
BW = 0.5
DOMAIN = 0.05              # m, cubic
DX = C0 / F0 / 20          # lambda/20 at f0 -> 38^3 cells with the absorber
CPML = 8
N_STEPS = 300              # the pulse has passed the centre: |E| at the end is 2e-5 of peak
FREQS = np.array([0.8, 1.0, 1.2]) * F0
TOL_DB = 0.01              # measured 1e-5 .. 2e-4 dB on this rig


def _vacuum(grid):
    return MaterialArrays(
        eps_r=jnp.ones(grid.shape, jnp.float32),
        sigma=jnp.zeros(grid.shape, jnp.float32),
        mu_r=jnp.ones(grid.shape, jnp.float32),
    )


def _dft(x, dt):
    t = np.arange(len(x)) * dt
    return np.array([np.sum(x * np.exp(-2j * np.pi * f * t)) * dt for f in FREQS])


def test_compute_rcs_divides_by_the_incident_at_the_domain_centre(monkeypatch):
    """The |E_inc| that compute_rcs divides by equals the incident E the 3-D grid carries at
    the centre of an empty domain, to 0.01 dB, at 0.8 / 1.0 / 1.2 f0. Dividing by the
    waveform instead (the pre-#820 denominator) reads about +1.05 dB here (lambda/20)."""
    grid = Grid(freq_max=F0 * 1.5, domain=(DOMAIN,) * 3, dx=DX, cpml_layers=CPML)
    centre = (grid.nx // 2, grid.ny // 2, grid.nz // 2)
    seen = {}

    def run_with_centre_probe(g, materials, n_steps, **kw):
        assert "probes" not in kw
        res = _real_run(g, materials, n_steps,
                        probes=[ProbeSpec(*centre, "ez")], **kw)
        seen["probe"] = np.asarray(res.time_series, dtype=np.float64)[:, 0]
        return res

    def far_field_recorder(*args, **kwargs):
        ff = _real_compute_far_field(*args, **kwargs)
        seen.setdefault("pattern", ff)      # first call = the observation grid
        return ff

    monkeypatch.setattr(rcs_mod, "run", run_with_centre_probe)
    monkeypatch.setattr(rcs_mod, "compute_far_field", far_field_recorder)

    phi = np.linspace(0.0, np.pi, 13)
    res = rcs_mod.compute_rcs(
        grid, _vacuum(grid), N_STEPS, f0=F0, bandwidth=BW, theta_inc=0.0,
        polarization="ez", theta_obs=np.array([np.pi / 2]), phi_obs=phi,
        freqs=FREQS, boundary="cpml", cpml_layers=CPML,
    )

    # The pulse has left the centre before the record ends (else the DFT windows differ).
    probe = seen["probe"]
    assert abs(probe[-1]) < 1e-3 * np.abs(probe).max()

    # Denominator read back from compute_rcs's own output: sigma = 4 pi p / |E_inc|^2 on the
    # bin with the most (leakage) power, so the division is well above float noise.
    ff = seen["pattern"]
    p = (np.abs(np.asarray(ff.E_theta, np.complex128)) ** 2
         + np.abs(np.asarray(ff.E_phi, np.complex128)) ** 2)[:, 0, :]
    k = np.argmax(p, axis=1)
    rows = np.arange(len(FREQS))
    p_inc_used = 4.0 * np.pi * p[rows, k] / np.asarray(res.rcs_linear)[rows, 0, k]

    p_inc_grid = np.abs(_dft(probe, float(grid.dt))) ** 2
    err_db = 10.0 * np.log10(p_inc_used / p_inc_grid)
    assert np.all(np.abs(err_db) <= TOL_DB), (
        f"compute_rcs divides by an incident {np.round(err_db, 4)} dB away from the field "
        f"the grid carries at the domain centre (f = {FREQS / 1e9} GHz); every "
        "normal-incidence sigma is off by the same number of dB, the other way.")


def test_the_aux_line_launches_the_soft_source_closed_form():
    """measure_normal_incident_spectrum against the 1-D Yee soft-source launch,
    E / waveform = 1 / (2 S cos(k~ dx / 2)) with sin(w dt / 2) = S sin(k~ dx / 2),
    on two meshes. Derivation: eliminate H from the 1-D update; an E source added
    after the update enters the second-order recursion as (s^n - s^{n-1}), and the
    outgoing-wave amplitude at the source node follows."""
    for res_per_lambda, n_steps in ((20, 300), (40, 600)):
        dx = C0 / F0 / res_per_lambda
        grid = Grid(freq_max=F0 * 1.5, domain=(DOMAIN,) * 3, dx=dx, cpml_layers=CPML)
        dt = float(grid.dt)
        cfg, st = init_tfsf(nx=grid.nx, dx=dx, dt=grid.dt, cpml_layers=CPML,
                            tfsf_margin=3, f0=F0, bandwidth=BW)
        launched = measure_normal_incident_spectrum(cfg, st, n_steps, FREQS, grid.dt)

        t = np.arange(n_steps) * dt
        arg = (t - cfg.src_t0) / cfg.src_tau
        waveform = cfg.src_amp * (-2.0 * arg) * np.exp(-arg ** 2)
        s = C0 * dt / dx
        k_grid = 2.0 / dx * np.arcsin(np.sin(np.pi * FREQS * dt) / s)
        expected = 1.0 / (2.0 * s * np.cos(k_grid * dx / 2.0))

        ratio_db = 20.0 * np.log10(np.abs(launched) / np.abs(_dft(waveform, dt)))
        assert np.all(np.abs(ratio_db - 20.0 * np.log10(expected)) <= TOL_DB), (
            res_per_lambda, np.round(ratio_db, 4), np.round(20 * np.log10(expected), 4))
        # and it is not the waveform: 1.0-1.16 dB below it at this Courant number
        assert np.all(ratio_db < -0.9), ratio_db
