"""A PEC cavity graded in plane at a cell ratio of 1.40 rings at its discrete model's frequency.

Pins the witness behind the in-plane grading cap of 1.4 (rfx-archive records/20261029-inplane-cap-1p4; design
rfx/plans/nu/20261029-inplane-cap-DESIGN.md). A 40 x 34 x 10 mm air box with PEC walls is graded on x and y with two
fine bands of h/4 per axis, and every step between neighbouring cells is exactly 1.40. Its TM110 mode has the closed
form (c/2) sqrt(1/a^2 + 1/b^2). On a separable box the discrete frequency is exact from the 1-D graded operators: the
eigenvalues of the second difference with primal spacing d[k] on cells and dual spacing (d[k-1] + d[k])/2 on nodes,
PEC ends, plus the leapfrog relation. Measured at h = 1 mm: the solver equals that model to 1.2e-7, and the model sits
2.60e-4 below the closed form (second order over h = 1, 2/3, 1/2 mm in the record).
"""

import math
import warnings

import numpy as np

from rfx import GaussianPulse, Simulation

C0 = 299792458.0
A, B, D = 40e-3, 34e-3, 10e-3
X_EDGES = [0.0, 10e-3, 12e-3, 26e-3, 28e-3, 40e-3]
Y_EDGES = [0.0, 8e-3, 10e-3, 22e-3, 24e-3, 34e-3]
H = 1e-3
RATIO = 1.4


def _profile(edges, h=H, r=RATIO):
    """Fine bands of h/4 on the odd zones; ramps at exactly r from each band; equal fillers; end cells of h."""
    f = h / 4
    ramp = []
    c = f
    while c * r < h - 1e-15:
        c *= r
        ramp.append(c)
    cells = []
    last_zone = len(edges) - 2
    for z in range(len(edges) - 1):
        length = edges[z + 1] - edges[z]
        if z % 2 == 1:
            cells += [f] * round(length / f)
            continue
        after = ramp if z > 0 else []
        before = ramp[::-1] if z < last_zone else []
        ends = [h] * ((z == 0) + (z == last_zone))
        rem = length - sum(after) - sum(before) - sum(ends)
        n = math.ceil(rem / h - 1e-9)
        cells += ([h] if z == 0 else []) + after + [rem / n] * n + before + ([h] if z == last_zone else [])
    return np.asarray(cells)


def _lowest_eigenvalue(widths):
    """Smallest eigenvalue of -d2/dx2 on the graded nodes, PEC (zero) at both ends."""
    d = np.asarray(widths, float)
    dual = 0.5 * (d[:-1] + d[1:])                      # interior nodes 1..N-1
    diag = (1 / d[:-1] + 1 / d[1:]) / dual
    off = -(1 / d[1:-1]) / np.sqrt(dual[:-1] * dual[1:])
    m = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return float(np.linalg.eigvalsh(m)[0])


def test_inplane_1p4_cavity_rings_at_its_discrete_model():
    px, py = _profile(X_EDGES), _profile(Y_EDGES)
    pz = np.full(10, D / 10)
    assert abs(px.sum() - A) < 1e-12 and abs(py.sum() - B) < 1e-12
    for p in (px, py):
        q = p[1:] / p[:-1]
        assert abs(np.max(np.maximum(q, 1 / q)) - RATIO) <= 1e-9

    f_closed = (C0 / 2) * math.sqrt((1 / A) ** 2 + (1 / B) ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=2 * 8.7e9, domain=(0, 0, 0), boundary="pec", dx=H,
                         dx_profile=px, dy_profile=py, dz_profile=pz)
        sim.add_source((13.3e-3, 11.7e-3, 5e-3), "ez", waveform=GaussianPulse(f0=f_closed, bandwidth=0.8),
                       amplitude_kind="current")
        sim.add_probe((29.1e-3, 25.3e-3, 5e-3), "ez")
        dt = float(sim._build_nonuniform_grid().dt)
        result = sim.run(n_steps=math.ceil(4.6e-9 / dt))
    modes = result.find_resonances(freq_range=(4e9, 7e9))
    f_sim = min((float(m.freq) for m in modes), key=lambda f: abs(f - f_closed))

    lam = _lowest_eigenvalue(px) + _lowest_eigenvalue(py)
    f_model = math.asin(C0 * dt * math.sqrt(lam) / 2) / (math.pi * dt)

    assert abs(f_sim / f_model - 1) < 2e-6, (f_sim, f_model)
    assert abs(f_model / f_closed - 1) < 3e-4, (f_model, f_closed)
