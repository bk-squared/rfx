"""Electric/magnetic cavities, periodic ring, absorber decay and face nodes.

Tolerances below precede the B1 measurements. Conclusion: leader fills.
"""

from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import EPS_0, MU_0
from rfx.geometry.csg import Box
from rfx.harminv import harminv
from rfx.materials.debye import DebyePole
from rfx.sources.sources import GaussianPulse
from tests.contracts.boundary_compare import BoundaryDeparture
from tests.contracts.boundary_fields import field_nodes


pytestmark = pytest.mark.slow_physics
C0 = 299792458.0


def record(name, data):
    output = os.environ.get("RFX_B1_OUTPUT")
    if output:
        Path(output, f"physics-{name}.json").write_text(json.dumps(data, indent=2) + "\n")


def compare_quantity(actual, expected, tolerance, quantity):
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise BoundaryDeparture(f"{quantity}: measured {actual}; expected {expected}; tolerance {tolerance}")


@contextmanager
def energies(grid):
    original = jax.lax.scan
    records = []
    depth = 0

    def scan(function, initial, xs=None, *args, **kwargs):
        nonlocal depth
        if depth or not field_nodes(initial):
            return original(function, initial, xs, *args, **kwargs)
        depth += 1
        try:
            def body(carry, x):
                result, y = function(carry, x)
                state = field_nodes(result)[0]
                energy = .5 * grid.dx**3 * sum(
                    weight * jnp.sum(jnp.abs(getattr(state, c))**2)
                    for c, weight in (("ex", EPS_0), ("ey", EPS_0), ("ez", EPS_0),
                                      ("hx", MU_0), ("hy", MU_0), ("hz", MU_0)))
                return result, (y, energy)
            result, (ys, es) = original(body, initial, xs, *args, **kwargs)
            jax.debug.callback(lambda a: records.append(np.asarray(a)), es)
            return result, ys
        finally:
            depth -= 1

    with patch.object(jax.lax, "scan", scan):
        yield records


def execute(sim, entry, steps, *, smoothing=False):
    if entry == "forward":
        return sim.forward(n_steps=steps, checkpoint=False, skip_preflight=True)
    return sim.run(n_steps=steps, compute_s_params=False, skip_preflight=True,
                   subpixel_smoothing=smoothing)


@lru_cache(maxsize=None)
def resonance(structure, entry, dx):
    if structure == "electric":
        domain = (.024, .024, .024)
        boundary = BoundarySpec.uniform("pec")
        source, probe = (.007, .009, .011), (.017, .013, .015)
        band = (7e9, 11e9)
    elif structure == "magnetic":
        domain = (.024, .020, .004)
        boundary = BoundarySpec(x="pmc", y="pec", z="pec")
        source, probe = (.007, .006, .002), (.017, .013, .002)
        band = (5e9, 12e9)
    else:
        domain = (.024, .002, .002)
        boundary = BoundarySpec(x="periodic", y="periodic", z="pec")
        source, probe = (.005, .001, .001), (.017, .001, .001)
        band = (8e9, 16e9)
    sim = Simulation(freq_max=20e9, domain=domain, dx=dx, boundary=boundary, cpml_layers=0)
    sim.add_source(source, "ez", waveform=GaussianPulse(f0=8.5e9, bandwidth=.8), amplitude_kind="field")
    sim.add_probe(probe, "ez")
    steps = round(2048 * .001 / dx)
    result = execute(sim, entry, steps)
    trace = np.asarray(result.time_series)[:, 0]
    dt = sim._build_grid().dt
    modes = harminv(trace[steps // 4:], dt, *band, min_Q=1, max_modes=24,
                    sv_threshold=1e-3, decimate="auto")
    data = dict(structure=structure, entry=entry, dx_m=dx, steps=steps,
                frequencies_Hz=[float(m.freq) for m in modes],
                amplitudes=[float(abs(m.amplitude)) for m in modes],
                trace_peak_V_per_m=float(np.max(np.abs(trace))))
    record(f"{structure}-{entry}-{dx}", data)
    assert data["trace_peak_V_per_m"] > 0
    return data


def nearest(data, target, window):
    candidates = [f for f in data["frequencies_Hz"] if window[0] < f < window[1]]
    return min(candidates, key=lambda f: abs(f - target), default=float("nan"))


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("dx", [.001, .0005])
def test_electric_cube(entry, dx):
    expected = C0 / (.024 * np.sqrt(2))
    measured = nearest(resonance("electric", entry, dx), expected, (7e9, 11e9))
    # 0.5% allows second-order Yee dispersion at dx=1 mm (24 cells).
    compare_quantity(measured, expected, .005 * expected, "TM110 frequency / Hz")


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("dx", [.001, .0005])
def test_magnetic_cavity_01(entry, dx):
    expected = C0 / (.020 * 2)
    measured = nearest(resonance("magnetic", entry, dx), expected, (6.8e9, 8.2e9))
    # 0.5% allows Yee dispersion, and excludes an 8.05 GHz termination.
    compare_quantity(measured, expected, .005 * expected, "f01 / Hz")


@pytest.mark.xfail(strict=True, raises=BoundaryDeparture, reason="h; magnetic wall separation; fixed in B3")
@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("dx", [.001, .0005])
def test_magnetic_cavity_separation(entry, dx):
    target = C0 / 2 * np.sqrt(1 / .024**2 + 1 / .020**2)
    frequency = nearest(resonance("magnetic", entry, dx), target, (9e9, 10.7e9))
    separation = 1 / np.sqrt((2 * frequency / C0)**2 - 1 / .020**2)
    record(f"separation-{entry}-{dx}", dict(frequency_Hz=frequency, derived_separation_m=float(separation),
                                           formula="1/sqrt((2*f/c)^2-(1/0.020)^2)"))
    # 0.2 mm (0.83%) allows dispersion; the supplied 23.02/23.50 mm fail.
    compare_quantity(separation, .024, .0002, "derived wall separation / m")


@pytest.mark.xfail(strict=True, raises=BoundaryDeparture, reason="e; periodic ring; fixed in B2")
@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("dx", [.001, .0005])
def test_periodic_ring(entry, dx):
    expected = C0 / .024
    measured = nearest(resonance("ring", entry, dx), expected, (8e9, 16e9))
    # 1% allows second-order dispersion, below the extra-node 2--4% shift.
    compare_quantity(measured, expected, .01 * expected, "ring frequency / Hz")


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("dx", [.001, .0005])
def test_cpml_box_energy(entry, dx):
    sim = Simulation(freq_max=20e9, domain=(.012, .010, .008), dx=dx,
                     boundary="cpml", cpml_layers=8)
    # B1 measurement amendment: cutoff=3 yielded -24.98 dB at dx=1 mm.
    # cutoff=4.5 is the source API's option for reducing deposited DC;
    # the -40 dB threshold and run duration stay as predeclared.
    pulse = GaussianPulse(f0=10e9, bandwidth=.8, cutoff=4.5)
    sim.add_source((.006, .005, .004), "ez", waveform=pulse, amplitude_kind="field")
    sim.add_probe((.008, .006, .004), "ez")
    grid = sim._build_grid()
    steps = round(1024 * .001 / dx)
    with energies(grid) as traces:
        result = execute(sim, entry, steps)
        jax.block_until_ready(result.time_series)
        jax.effects_barrier()
    assert len(traces) == 1
    waveform = np.asarray(jax.vmap(pulse)(jnp.arange(steps) * grid.dt))
    end_source = np.flatnonzero(abs(waveform) > np.max(abs(waveform)) * 1e-5)[-1] + 1
    energy = traces[0]
    peak = float(np.max(energy[end_source:]))
    assert peak > 0
    level = float(10 * np.log10(max(float(energy[-1]), np.finfo(float).tiny) / peak))
    record(f"cpml-cutoff4p5-{entry}-{dx}", dict(end_over_post_source_peak_dB=level,
                                     final_J=float(energy[-1]), post_source_peak_J=peak, steps=steps))
    # -40 dB = at most 1e-4 of post-source energy, the repository ringdown witness.
    if not np.isfinite(level) or level >= -40:
        raise BoundaryDeparture(f"CPML end/post-source-peak energy {level} dB; required < -40 dB")


@pytest.mark.xfail(strict=True, raises=BoundaryDeparture, reason="b1/h; material face node; fixed in B3")
@pytest.mark.parametrize("entry,material", [
    ("run", "dielectric"), ("run", "debye"), ("forward", "debye"),
    pytest.param("forward", "dielectric", marks=pytest.mark.skip(
        reason="B1 stopped item: design requires smoothing; forward has no subpixel_smoothing argument")),
])
@pytest.mark.parametrize("dx", [.001, .0005])
def test_material_face_against_full_symmetric_domain(entry, material, dx):
    traces = []
    for full in (False, True):
        length = .024 if full else .012
        spec = BoundarySpec(x="pec" if full else Boundary("pmc", "pec"), y="pec", z="pec")
        sim = Simulation(freq_max=20e9, domain=(length, .008, .004), dx=dx, boundary=spec, cpml_layers=0)
        poles = [DebyePole(2, 20e-12)] if material == "debye" else None
        sim.add_material("medium", eps_r=4, debye_poles=poles)
        sim.add(Box((-dx, -dx, -dx), (length + dx, .008 + dx, .004 + dx)), material="medium")
        locations = (.008, .016) if full else (.004,)
        for x in locations:
            sim.add_source((x, .004, .002), "ez", waveform=GaussianPulse(f0=10e9, bandwidth=.8), amplitude_kind="field")
        sim.add_probe((.012 if full else 0., .004, .002), "ez")
        result = execute(sim, entry, round(512 * .001 / dx), smoothing=material == "dielectric")
        traces.append(np.asarray(result.time_series)[:, 0])
    reference_norm = np.linalg.norm(traces[1])
    assert reference_norm > 0
    relative = float(np.linalg.norm(traces[0] - traces[1]) / reference_norm)
    record(f"{material}-face-{entry}-{dx}", dict(relative_trace_l2=relative,
                                                half_peak_V_per_m=float(np.max(abs(traces[0]))),
                                                full_peak_V_per_m=float(np.max(abs(traces[1])))))
    # Identical mirrored lattice and excitation; 1% permits accumulated float32 error.
    compare_quantity(relative, 0, .01, "relative face trace L2")
