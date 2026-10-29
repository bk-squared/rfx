"""An oblique closed TF/SF box launches the lattice plane wave along its realized direction (uniform mesh).

In an empty closed box the total field is the incident wave. On the Yee lattice that wave is the discrete plane
wave along the realized rational direction m (``Simulation.tfsf_incidence()``):
- wavevector k = q m / dx, where q is the principal root of
  sum_u sin^2(q m_u / 2) = (sin(pi f dt) / S)^2 and S = c dt / dx;
- E perpendicular to sin(k dx / 2), so its part along k-hat is the small lattice tilt between the two vectors;
- the same amplitude as the normal-incidence box of the same declaration.
Its continuum limit is the analytic plane wave: k / k0 - 1 is second order in dx. That limit was measured at
1.0 and 0.5 mm in rfx-archive ``rfx/records/20261029-oblique-closedbox-uniform/``, which also gives the bars below.
The scattered-field leak of the same engine is pinned by
``tests/unit/sources/test_tfsf_discrete_plane_wave.py::test_v1_v2_l1``.
"""
from functools import lru_cache

import numpy as np
import pytest

from rfx import Simulation

C0 = 299792458.0
DX, N_STEPS = 1e-3, 600
FREQS = np.linspace(8e9, 12e9, 9)
# Total-field sample planes (mm), unequal per axis, inside the 4-20 mm box with the half-cell Yee offsets.
PLANES_MM = ((5, 9, 14, 19), (6, 10, 13, 18), (5, 8, 12, 17))
E_OFFSETS = np.array(((.5, 0, 0), (0, .5, 0), (0, 0, .5)))


def lattice_k(m, f, dt):
    """Principal root of the Yee dispersion relation along m, by bisection in float64."""
    m = np.asarray(m, float)
    target = (np.sin(np.pi * f * dt) / (C0 * dt / DX)) ** 2
    lo, hi = 0.0, np.pi / np.max(np.abs(m))
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.sum(np.sin(mid * m / 2) ** 2) < target else (lo, mid)
    return (lo + hi) / 2 * m / DX


@lru_cache(maxsize=None)
def box_wave(theta, phi, pol):
    """Probe spectra at 64 total-field nodes and the realized incidence."""
    sim = Simulation(freq_max=15e9, domain=(.024,) * 3, dx=DX, boundary="cpml", cpml_layers=8)
    sim.add_tfsf_source(f0=10e9, bandwidth=.8, margin=4, closed_box=True,
                        theta_deg=theta, phi_deg=phi, polarization=pol)
    grid = sim._build_grid()
    cpml = int(grid.cpml_layers)
    nodes = np.stack(np.meshgrid(*[cpml + np.array(p) for p in PLANES_MM], indexing="ij"), -1).reshape(-1, 3)
    box = sim.tfsf_box_indices()
    assert np.all(nodes >= [box[a][0] for a in "xyz"]) and np.all(nodes + 1 <= [box[a][1] for a in "xyz"])
    for node in nodes:
        for component in ("ex", "ey", "ez"):
            sim.add_probe(tuple(float((i - cpml) * DX) for i in node), component)
    dt = float(grid.dt)
    series = np.asarray(sim.run(n_steps=N_STEPS, skip_preflight=True).time_series, np.float64)
    series = series.reshape(N_STEPS, len(nodes), 3)
    kernel = np.exp(-2j * np.pi * FREQS[:, None] * (np.arange(N_STEPS) + 1) * dt) * dt
    spectra = np.einsum("fn,njc->fjc", kernel, series)
    positions = (nodes[:, None, :] - cpml + E_OFFSETS) * DX
    return sim.tfsf_incidence(), spectra, positions, dt


def plane_wave_fit(theta, phi, pol):
    """Per bin: residual against the lattice plane wave, its fitted wavevector, and the amplitude vector."""
    info, spectra, positions, dt = box_wave(theta, phi, pol)
    e_hat = np.asarray(info["e_hat"], float)
    co = np.flatnonzero(np.abs(e_hat) >= .05)
    rows = []
    for b, f in enumerate(FREQS):
        k = lattice_k(info["m"], f, dt)
        rotated = spectra[b] * np.exp(1j * positions @ k)
        amp = rotated.mean(axis=0)
        residual = np.max(np.abs(rotated[:, co] - amp[co])) / abs(e_hat @ amp)
        phase = np.angle(rotated[:, co] / amp[co])
        design = np.array([np.concatenate((-positions[j, c], np.eye(len(co))[i]))
                           for i, c in enumerate(co) for j in range(len(positions))])
        k_fit = k + np.linalg.lstsq(design, phase.T.reshape(-1), rcond=None)[0][:3]
        rows.append(dict(k=k, k_fit=k_fit, residual=residual, amp=amp, k0=2 * np.pi * f / C0))
    return info, rows


@pytest.mark.parametrize("theta,phi,pol", ((30., 0., "p"), (45., 30., "s")))
def test_oblique_closed_box_is_the_lattice_plane_wave(theta, phi, pol):
    info, rows = plane_wave_fit(theta, phi, pol)
    _, normal = plane_wave_fit(0., 0., pol)
    k_hat, e_hat, h_hat = (np.asarray(info[name], float) for name in ("k_hat", "e_hat", "h_hat"))
    for row, ref in zip(rows, normal):
        assert row["residual"] <= 1e-4, row["residual"]
        k_fit = row["k_fit"]
        angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(k_fit, k_hat)), k_fit @ k_hat))
        assert angle <= 1e-3, angle
        # The analytic wave's wavenumber is missed by the lattice dispersion only, less than along the axis.
        error = np.linalg.norm(k_fit) / row["k0"] - 1
        assert abs(error) < abs(np.linalg.norm(ref["k_fit"]) / ref["k0"] - 1), error
        co = abs(e_hat @ row["amp"])
        assert abs(h_hat @ row["amp"]) / co <= 1e-2
        kappa = np.sin(row["k"] * DX / 2)
        tilt = np.arccos(min(1.0, kappa @ k_hat / np.linalg.norm(kappa)))
        assert abs(k_hat @ row["amp"]) / co <= tilt + 1e-4
        assert abs(co / abs(np.asarray(normal_e_hat(pol)) @ ref["amp"]) - 1) <= 1e-2


def normal_e_hat(pol):
    return box_wave(0., 0., pol)[0]["e_hat"]
