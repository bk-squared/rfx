"""Scene P and its independent longitudinal 0-D map (pole design, A1/A3).

The record ends at N/f_cf from t=0; the identification window begins at
source-off. Call build AND run inside enable_x64() for the float64 lane.
"""

import numpy as np

from rfx import Box, GaussianPulse, Simulation, drude_pole, lorentz_pole


EPS0 = 8.8541878128e-12
FP = 1e9
DOMAIN = (0.400e-3, 0.300e-3, 0.250e-3)
DX = 25e-6
SOURCE = (0.125e-3, 0.100e-3, 0.100e-3)
PROBES = ((0.175e-3, 0.125e-3, 0.150e-3),
          (0.100e-3, 0.175e-3, 0.075e-3))
FREQ_MAX = 4e9


def waveform():
    """One differentiated Gaussian for all six cases; f0 is the API parameter."""
    return GaussianPulse(f0=1.4e9, bandwidth=1.0)


def coefficients(nu, dt):
    """Hand float64 formation, matching main's once-cast Drude coefficients."""
    dt = np.float64(dt)
    ddt = np.float64(nu) * 0.5 * dt
    return (2.0 / (1.0 + ddt), -(1.0 - ddt) / (1.0 + ddt),
            EPS0 * (2.0 * np.pi * FP)**2 * dt**2 / (1.0 + ddt))


def map_frequency(a, b, c, Cc, dt):
    """Positive frequency of z**2 - (a - c*Cc)*z - b, without rfx helpers."""
    a, b, c, Cc, dt = map(np.float64, (a, b, c, Cc, dt))
    roots = np.roots(np.array([1.0, -(a - c * Cc), -b], dtype=np.float64))
    return float(np.max(np.abs(np.angle(roots.astype(np.complex128))))
                 / (2.0 * np.pi * dt))


def references(nu, dt):
    exact = coefficients(nu, dt)
    rounded = tuple(float(np.float32(v)) for v in exact)
    return dict(a=float(exact[0]), b=float(exact[1]), c=float(exact[2]),
                a32=rounded[0], b32=rounded[1], c32=rounded[2], Cc=1.0 / EPS0,
                r=rounded[0] + rounded[1] - 1.0,
                f_cf=map_frequency(*exact, 1.0 / EPS0, dt),
                f_main=map_frequency(*rounded, 1.0 / EPS0, dt))


def selected_rates(dt):
    """A1.3/section 4: first match for each sign, k=0..100000 inclusive."""
    found = {}
    for k in range(100001):
        nu = 1e7 + k * 1e4
        a, b, _ = coefficients(nu, dt)
        r = float(np.float32(a)) + float(np.float32(b)) - 1.0
        for sign in (-1, 0, 1):
            if sign not in found and r == sign * 2.0**-24:
                found[sign] = dict(nu=nu, k=k, r=r)
        if len(found) == 3:
            return found
    raise ValueError("selection rule has no match for all three Drude residuals")


def scene(nu, precision, n_periods, waveform):
    """Build the declared PEC box, filled Drude medium, point source and probes."""
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX,
                     boundary="pec", precision=precision)
    sim.add_material("plasma", eps_r=1.0, sigma=0.0,
                     lorentz_poles=[drude_pole(2.0 * np.pi * FP, gamma=nu)])
    sim.add(Box((0.0, 0.0, 0.0), DOMAIN), material="plasma")
    sim.add_source(SOURCE, component="ez", waveform=waveform,
                   amplitude_kind="current")
    for position in PROBES:
        sim.add_probe(position, component="ez")
    grid = sim._build_grid()
    f_cf = references(nu, grid.dt)["f_cf"]
    n_steps = int(np.ceil(n_periods / (f_cf * grid.dt)))
    return sim, grid, n_steps


def lorentz_reference(nu, dt):
    """Hand-built longitudinal map and residual, without coefficient helpers."""
    w0 = 2 * np.pi * 3e9
    ddt = np.float64(nu) / 2 * np.float64(dt)
    restoring = w0**2 * dt**2 / (1 + ddt)
    exact = ((2 - w0**2 * dt**2) / (1 + ddt),
             -(1 - ddt) / (1 + ddt), EPS0 * 2 * w0**2 * dt**2 / (1 + ddt))
    rounded = tuple(float(np.float32(v)) for v in exact)
    return dict(a=float(exact[0]), b=float(exact[1]), c=float(exact[2]),
                a32=rounded[0], b32=rounded[1], c32=rounded[2], Cc=1 / EPS0,
                r=rounded[0] + rounded[1] - 1 + restoring,
                f_cf=map_frequency(*exact, 1 / EPS0, dt),
                f_main=map_frequency(*rounded, 1 / EPS0, dt))


def lorentz_scene(nu):
    pulse = GaussianPulse(f0=5.5e9, bandwidth=1.0)
    sim = Simulation(freq_max=16e9, domain=DOMAIN, dx=DX,
                     boundary="pec", precision="float32")
    sim.add_material("lorentz", eps_r=1., sigma=0.,
                     lorentz_poles=[lorentz_pole(2., 2 * np.pi * 3e9, nu / 2)])
    sim.add(Box((0., 0., 0.), DOMAIN), material="lorentz")
    sim.add_source(SOURCE, component="ez", waveform=pulse, amplitude_kind="current")
    for position in PROBES:
        sim.add_probe(position, component="ez")
    grid = sim._build_grid()
    selected = next((1e7 + k * 1e4 for k in range(100001)
                     if abs(lorentz_reference(1e7 + k * 1e4, float(grid.dt))["r"]) >= 2**-24), None)
    if selected != nu:
        raise ValueError(f"Lorentz selection rule gives {selected}, requested {nu}")
    ref = lorentz_reference(nu, float(grid.dt))
    return sim, grid, int(np.ceil(8 / (ref["f_cf"] * grid.dt))), pulse, ref
