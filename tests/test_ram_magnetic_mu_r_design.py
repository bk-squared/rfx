"""Differentiable magnetic-material (mu_r) RAM design vs an analytic magnetic TMM
oracle (Item B).

Magnetic RAM (ferrites) matches at a THINNER layer than a dielectric-only design
because a mu_r>1 layer raises the wave impedance eta=ETA0*sqrt(mu/eps) toward the
free-space match AND shortens the electrical quarter-wave (n=sqrt(eps*mu)). Item B
makes mu_r a DIFFERENTIABLE design variable.

Premise (handover-mandated, verified here): before this work ``forward()`` had NO
mu_r channel — only eps_override/sigma_override — while the Yee H-update genuinely
uses mu_r (``mu = materials.mu_r*MU_0``; ``h -= (dt/mu)*curl``). So mu_r was a live
solver axis with no differentiable entry point (a missing-channel no-op, the
add_lumped_rlc+forward footgun class). This suite pins the newly-wired
``forward(mu_r_override=...)``:

  FAST (no FDTD): the magnetic TMM oracle's physical limits (mu_r=1 reduces to the
    dielectric oracle; two formulations agree; passivity; the physical witness that
    a magnetic layer nulls at a thinner thickness) + that forward() exposes the
    channel (locks it against silent removal).
  SLOW (FDTD): the channel is LIVE (mu_r=2 != mu_r=1 output — not a no-op); AD==FD
    self-consistency; d|Gamma|/dmu_r vs the INDEPENDENT analytic magnetic-TMM
    gradient (sign+order — the real physics gradient check, per the Item A lesson
    that AD==FD alone is a tautology); |Gamma|(f) vs the magnetic TMM band; lossless
    energy conservation; and the NU/distributed fail-loud fence (no silent drop).

Measured (CPU, f0=8 GHz, dx=0.5 mm, eps'=4, sigma=1.4, mu_r=2, ~9-cell layer, on
PEC): channel live (18.7% output change mu 1->2); d|Gamma|/dmu_r AD==FD 0.0%; AD vs
analytic magnetic-TMM gradient sign-match, ratio 0.65-0.78; |Gamma|(f) vs TMM band
mean 3.2%; lossless mean|Gamma|~1.02.
"""
from __future__ import annotations

import inspect
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.probes import fresnel_reflection_coefficient

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * np.pi
ETA0 = float(np.sqrt(MU0 / EPS0))

F0 = 8e9
DX = 0.5e-3
DOMAIN = (0.070, 0.020, 0.0015)
X_IFACE = 0.040
LAYER_D = 0.00469
X_BACK = X_IFACE + LAYER_D
PEC_D = 0.002
PLATEAU = (0.020, 0.036)
BAND = np.linspace(6e9, 10e9, 21)
EPS_R, SIGMA, MU_R = 4.0, 1.4, 2.0


# --------------------------------------------------------------------------- #
# Analytic magnetic TMM oracle (e^{+jwt}). Layers FRONT-to-BACK; PEC backing.
#   n = sqrt(eps_c*mu_c),  eta = ETA0*sqrt(mu_c/eps_c)
# --------------------------------------------------------------------------- #
def _eps_c(eps_r, sigma, freq):
    return eps_r - 1j * sigma / (2.0 * np.pi * freq * EPS0)


def _tmm_mag(freqs, layers, backing="pec", method="impedance"):
    z_load = 0.0 if backing == "pec" else ETA0
    freqs = np.atleast_1d(np.asarray(freqs, float))
    out = np.empty(freqs.shape, complex)
    for i, f in enumerate(freqs):
        k0 = 2.0 * np.pi * f / C0
        if method == "impedance":
            z = complex(z_load)
            for (er, sig, mur, d) in reversed(layers):
                ec = _eps_c(er, sig, f)
                n = np.sqrt(ec * mur)
                eta = ETA0 * np.sqrt(mur / ec)
                t = np.tan(k0 * n * d)
                z = eta * (z + 1j * eta * t) / (eta + 1j * z * t)
            z_in = z
        else:
            M = np.eye(2, dtype=complex)
            for (er, sig, mur, d) in layers:
                ec = _eps_c(er, sig, f)
                n = np.sqrt(ec * mur)
                eta = ETA0 * np.sqrt(mur / ec)
                c, s = np.cos(k0 * n * d), np.sin(k0 * n * d)
                M = M @ np.array([[c, 1j * eta * s], [1j * s / eta, c]], complex)
            z_in = (M[0, 0] * z_load + M[0, 1]) / (M[1, 0] * z_load + M[1, 1])
        out[i] = (z_in - ETA0) / (z_in + ETA0)
    return out


def _tmm_die(freqs, er, sig, d):
    """Dielectric (mu_r=1) reference via the plain impedance recursion."""
    out = []
    for f in np.atleast_1d(np.asarray(freqs, float)):
        k0 = 2 * np.pi * f / C0
        ec = _eps_c(er, sig, f)
        n = np.sqrt(ec)
        eta = ETA0 / n
        t = np.tan(k0 * n * d)
        z = eta * (1j * eta * t) / (eta)  # z_load=0
        out.append((z - ETA0) / (z + ETA0))
    return np.array(out)


# --------------------------------------------------------------------------- #
# FAST — oracle limits + channel exposure.
# --------------------------------------------------------------------------- #
def test_magnetic_tmm_oracle_limits():
    band = np.linspace(6e9, 14e9, 41)
    # mu_r=1 reduces to the dielectric oracle
    g_mag = _tmm_mag(band, [(4.0, 1.4, 1.0 + 0j, 4.5e-3)])
    g_die = _tmm_die(band, 4.0, 1.4, 4.5e-3)
    assert np.max(np.abs(g_mag - g_die)) < 1e-12
    # two formulations agree for a magnetic-lossy stack
    lay = [(9.0, 2.0, 3.0 - 1.5j, 2.0e-3), (2.0, 0.5, 1.5 - 0.3j, 1.5e-3)]
    assert np.max(np.abs(_tmm_mag(band, lay, method="impedance")
                        - _tmm_mag(band, lay, method="transfer"))) < 1e-9
    # passivity
    assert np.max(np.abs(_tmm_mag(band, lay))) < 1.0 + 1e-9

    # PHYSICAL WITNESS: a magnetic layer (mu'=eps'=4, impedance-matched) nulls at a
    # THINNER thickness than the dielectric-only (mu=1) design.
    ds = np.linspace(0.5e-3, 8e-3, 800)
    d_die = ds[int(np.argmin([abs(_tmm_mag([F0], [(4.0, 3.0, 1.0 + 0j, d)])[0]) for d in ds]))]
    d_mag = ds[int(np.argmin([abs(_tmm_mag([F0], [(4.0, 3.0, 4.0 - 1.0j, d)])[0]) for d in ds]))]
    assert d_mag < 0.8 * d_die, f"magnetic null d={d_mag*1e3:.2f}mm not thinner than dielectric {d_die*1e3:.2f}mm"


def test_forward_exposes_mu_r_override():
    """Lock the differentiable channel: forward() must keep mu_r_override so a
    future refactor cannot silently drop it back to a zero-gradient no-op."""
    assert "mu_r_override" in inspect.signature(Simulation.forward).parameters


# --------------------------------------------------------------------------- #
# SLOW — FDTD magnetic reflection vs the magnetic TMM oracle.
# --------------------------------------------------------------------------- #
def _build():
    grid = Grid(freq_max=16e9, domain=DOMAIN, dx=DX, cpml_layers=10)
    xi = grid.position_to_index((X_IFACE, 0.010, 0.00075))[0]
    xb = grid.position_to_index((X_BACK, 0.010, 0.00075))[0]
    xpec = grid.position_to_index((X_BACK + PEC_D, 0.010, 0.00075))[0]
    xprobes = np.arange(PLATEAU[0], PLATEAU[1], DX)
    pidx = np.array([grid.position_to_index((float(xp), 0.010, 0.00075))[0] for xp in xprobes])
    dists = (xi - pidx).astype(np.float64) * DX
    ns = int(12.0 * (2.0 * X_BACK / C0) / grid.dt)
    d_raster = (xb - xi) * DX
    sim = Simulation(freq_max=16e9, domain=DOMAIN, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=F0, bandwidth=0.6, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    for xp in xprobes:
        sim.add_probe((float(xp), 0.010, 0.00075), component="ez")
    return sim, grid.shape, xi, xb, xpec, dists, grid.dt, ns, d_raster


def _mu(shape, xi, xb, mu):
    return jnp.ones(shape, jnp.float32).at[xi:xb, :, :].set(mu)


def _run(sim, shape, xi, xb, xpec, ns, eps_r=1.0, sigma=0.0, mu_r=1.0, pec=True):
    m = np.zeros(shape, bool); m[xb:xpec, :, :] = True
    kw = dict(n_steps=ns, checkpoint=False, skip_preflight=True,
              eps_override=jnp.ones(shape, jnp.float32).at[xi:xb, :, :].set(eps_r),
              sigma_override=jnp.zeros(shape, jnp.float32).at[xi:xb, :, :].set(sigma),
              mu_r_override=_mu(shape, xi, xb, mu_r))
    if pec:
        kw["pec_mask_override"] = jnp.asarray(m)
    return sim.forward(**kw).time_series


@pytest.fixture(scope="module")
def mag_run():
    sim, shape, xi, xb, xpec, dists, dt, ns, d_ras = _build()
    inc = sim.forward(eps_override=jnp.ones(shape, jnp.float32),
                      sigma_override=jnp.zeros(shape, jnp.float32),
                      n_steps=ns, checkpoint=False, skip_preflight=True).time_series
    s_mu1 = _run(sim, shape, xi, xb, xpec, ns, eps_r=4.0, sigma=1.4, mu_r=1.0)
    s_mu2 = _run(sim, shape, xi, xb, xpec, ns, eps_r=4.0, sigma=1.4, mu_r=2.0)
    s_lossless = _run(sim, shape, xi, xb, xpec, ns, eps_r=4.0, sigma=0.0, mu_r=2.0)

    def gband(tot):
        return np.array([abs(complex(fresnel_reflection_coefficient(
            jnp.asarray(tot), jnp.asarray(inc), f0=float(f), dt=dt,
            probe_distances=dists, n_gate=ns))) for f in BAND])

    return {"sim": sim, "shape": shape, "xi": xi, "xb": xb, "xpec": xpec, "dists": dists,
            "dt": dt, "ns": ns, "d_ras": d_ras, "inc": inc,
            "s_mu1": np.asarray(s_mu1), "s_mu2": np.asarray(s_mu2),
            "g_lossy": gband(s_mu2), "g_lossless": gband(s_lossless)}


@pytest.mark.slow
def test_mu_r_channel_is_live(mag_run):
    """forward(mu_r_override=2) must differ from mu_r=1 — proves mu_r reaches the
    H-update (not a silent no-op)."""
    s1, s2 = mag_run["s_mu1"], mag_run["s_mu2"]
    rel = np.max(np.abs(s1 - s2)) / (np.max(np.abs(s1)) + 1e-30)
    assert rel > 1e-3, f"mu_r_override is a no-op (series change {rel:.2e})"


@pytest.mark.slow
def test_mu_r_gradient_ad_vs_fd(mag_run):
    """Autodiff SELF-CONSISTENCY (not physics): jax.grad w.r.t. mu_r matches a FD
    step of the same forward. Physical validation is test_mu_r_gradient_vs_analytic_tmm."""
    sim, shape, xi, xb, xpec = (mag_run[k] for k in ("sim", "shape", "xi", "xb", "xpec"))
    dists, dt, ns, inc = (mag_run[k] for k in ("dists", "dt", "ns", "inc"))
    m = np.zeros(shape, bool); m[xb:xpec, :, :] = True
    pec = jnp.asarray(m)

    def absg(mu):
        tot = sim.forward(
            eps_override=jnp.ones(shape, jnp.float32).at[xi:xb, :, :].set(4.0),
            sigma_override=jnp.zeros(shape, jnp.float32).at[xi:xb, :, :].set(1.4),
            mu_r_override=_mu(shape, xi, xb, mu), pec_mask_override=pec,
            n_steps=ns, checkpoint=True, skip_preflight=True).time_series
        return jnp.abs(fresnel_reflection_coefficient(
            tot, inc, f0=F0, dt=dt, probe_distances=dists, n_gate=ns))

    g_ad = float(jax.grad(absg)(MU_R))
    h = 0.05
    g_fd = (float(absg(MU_R + h)) - float(absg(MU_R - h))) / (2 * h)
    assert np.isfinite(g_ad), "NaN/Inf gradient (P0)"
    assert abs(g_ad) > 1e-4, "zero gradient — dead mu_r channel"
    assert abs(g_ad - g_fd) < 0.05 * abs(g_fd) + 1e-6, f"AD={g_ad:.4e} vs FD={g_fd:.4e}"


@pytest.mark.slow
def test_mu_r_gradient_vs_analytic_tmm(mag_run):
    """PHYSICAL gradient check: FDTD jax.grad vs the INDEPENDENT analytic magnetic-
    TMM gradient (closed-form derivative, not the FDTD's own FD). Sign + order;
    the ratio is ~0.4-2x (the ~half-cell thickness ambiguity and dispersion of a
    ~9-cell layer, same envelope as the |Gamma| agreement)."""
    sim, shape, xi, xb, xpec = (mag_run[k] for k in ("sim", "shape", "xi", "xb", "xpec"))
    dists, dt, ns, inc, d = (mag_run[k] for k in ("dists", "dt", "ns", "inc", "d_ras"))
    m = np.zeros(shape, bool); m[xb:xpec, :, :] = True
    pec = jnp.asarray(m)

    def absg(mu):
        tot = sim.forward(
            eps_override=jnp.ones(shape, jnp.float32).at[xi:xb, :, :].set(4.0),
            sigma_override=jnp.zeros(shape, jnp.float32).at[xi:xb, :, :].set(1.4),
            mu_r_override=_mu(shape, xi, xb, mu), pec_mask_override=pec,
            n_steps=ns, checkpoint=True, skip_preflight=True).time_series
        return jnp.abs(fresnel_reflection_coefficient(
            tot, inc, f0=F0, dt=dt, probe_distances=dists, n_gate=ns))

    g_ad = float(jax.grad(absg)(MU_R))
    hh = 1e-3
    def gt(mu):
        return abs(_tmm_mag([F0], [(4.0, 1.4, mu + 0j, d)])[0])
    g_tmm = (gt(MU_R + hh) - gt(MU_R - hh)) / (2 * hh)
    assert np.sign(g_ad) == np.sign(g_tmm), f"AD sign {g_ad:+.3e} != analytic {g_tmm:+.3e}"
    ratio = abs(g_ad) / (abs(g_tmm) + 1e-12)
    assert 0.4 < ratio < 2.5, f"|AD/analytic|={ratio:.2f} outside the discretization envelope"


@pytest.mark.slow
def test_magnetic_magnitude_vs_tmm(mag_run):
    """FDTD |Gamma|(f) of the PEC-backed magnetic lossy layer tracks the magnetic
    TMM (eta=sqrt(mu/eps)), gated against the +-half-cell thickness band."""
    g_fd = mag_run["g_lossy"]
    g_nom = np.abs(_tmm_mag(BAND, [(EPS_R, SIGMA, MU_R + 0j, LAYER_D)]))
    g_ras = np.abs(_tmm_mag(BAND, [(EPS_R, SIGMA, MU_R + 0j, mag_run["d_ras"])]))
    lo, hi = np.minimum(g_nom, g_ras), np.maximum(g_nom, g_ras)
    err = np.where(g_fd < lo, lo - g_fd, np.where(g_fd > hi, g_fd - hi, 0.0))
    assert np.max(g_fd) < 1.02, "nonphysical |Gamma|>1"
    assert np.mean(err) < 0.05, f"mean dist to magnetic-TMM band = {np.mean(err):.3f}"
    assert np.max(err) < 0.11, f"max dist to magnetic-TMM band = {np.max(err):.3f}"


@pytest.mark.slow
def test_magnetic_lossless_energy_conservation(mag_run):
    """A lossless mu_r!=1 PEC-backed layer still reflects everything: |Gamma|~1 in
    the mean (energy conservation), within the two-run extractor's total-reflector
    ripple envelope."""
    g = mag_run["g_lossless"]
    assert abs(np.mean(g) - 1.0) < 0.06, f"mean |Gamma|={np.mean(g):.3f} (must ~1)"
    assert np.min(g) > 0.82 and np.max(g) < 1.25, \
        f"|Gamma| ripple [{np.min(g):.3f}, {np.max(g):.3f}] beyond extractor envelope"


@pytest.mark.slow
def test_mu_r_override_fenced_on_nonuniform():
    """mu_r_override must fail loud on a non-uniform mesh (it is wired only on the
    uniform lane) rather than being silently dropped to a zero-gradient no-op."""
    sim = Simulation(freq_max=16e9, domain=(0.02, 0.006, 0.002), dx=0.5e-3,
                     dz_profile=np.full(4, 0.5e-3), boundary="cpml", cpml_layers=6, mode="3d")
    sim.add_probe((0.01, 0.003, 0.0005), component="ez")
    shape = sim._build_grid().shape
    with pytest.raises(NotImplementedError, match="mu_r_override"):
        sim.forward(mu_r_override=jnp.ones(shape, jnp.float32), n_steps=10, skip_preflight=True)
