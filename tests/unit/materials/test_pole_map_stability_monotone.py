"""P5a: measure the proposed map corner's monotonicity before implementing it.

The Drude bound is re-derived from kappa dt**2 + x_max <= 4 eps_inf.
The asymmetric x profile also declares the graded grid for the later P1 pin.
Set POLE_MAP_P5A_RESULTS to preserve every sampled growth value as JSON.
"""
import itertools
import json
import os
from pathlib import Path

import numpy as np

from rfx import Simulation
from rfx.model.pole_stability import largest_growth


C0 = 299792458.0
DX = 0.5e-3
DOMAIN = (40e-3, 30e-3, 26e-3)
GRADED_X = np.concatenate(([DX], np.linspace(0.25e-3, 0.75e-3, 78), [DX]))
TOLERANCE = 1e-6


def _budgets():
    common = dict(freq_max=6e9, domain=DOMAIN, dx=DX, cpml_layers=8)
    uniform = Simulation(**common)._build_grid()
    graded = Simulation(**common, dx_profile=GRADED_X)._build_nonuniform_grid()
    for name, grid in (("uniform", uniform), ("graded", graded)):
        cells = [np.asarray(grid.cells(axis), dtype=np.float64) for axis in range(3)]
        # The second graded x cell is finest; its immediate neighbours are coarser.
        # Uniform y/z make this the finest region's local (including-neighbour) budget.
        minima = [float(np.min(widths)) for widths in cells]
        budget = C0**2 * sum(1 / width**2 for width in minima)
        yield name, float(grid.dt), budget, list(grid.shape), minima


def test_pole_map_stability_monotone():
    rows = []
    failures = []
    factors = np.unique(np.concatenate((np.linspace(0, 2, 9), np.linspace(0.8, 1.2, 9))))
    deltas = np.linspace(0, 4e10, 9)
    for lane, dt, budget, shape, minima in _budgets():
        x_max = 4 * dt**2 * budget
        # At z=-1, the undamped Drude boundary is kappa dt**2 + x_max = 4.
        bound = (4 - x_max) / dt**2
        assert bound > 0
        kappas = factors * bound
        for eps, sigma, omega_0 in itertools.product((1., 10.), (0., 0.1), (0., 2 * np.pi * 5e9)):
            growth = np.array([
                [largest_growth(eps, sigma, [(omega_0, delta, kappa)], dt, x_max)
                 for delta in deltas]
                for kappa in kappas
            ])
            kappa_drop = -np.diff(growth, axis=0)
            delta_rise = np.diff(growth, axis=1)
            bad_kappa = np.argwhere(kappa_drop > TOLERANCE)
            bad_delta = np.argwhere(delta_rise > TOLERANCE)
            violations = []
            for axis, indices in (("kappa", bad_kappa), ("delta", bad_delta)):
                for i, j in indices:
                    ni, nj = (i + 1, j) if axis == "kappa" else (i, j + 1)
                    violations.append(dict(
                        axis=axis, kappa_before=float(kappas[i]), kappa_after=float(kappas[ni]),
                        delta_before=float(deltas[j]), delta_after=float(deltas[nj]),
                        growth_before=float(growth[i, j]), growth_after=float(growth[ni, nj]),
                    ))
            row = dict(lane=lane, eps_r=eps, sigma=sigma, omega_0=omega_0, dt=dt,
                       budget=budget, x_max=x_max, shape=shape, minimum_cell_widths=minima,
                       drude_bound=bound, kappas=kappas.tolist(), deltas=deltas.tolist(),
                       growth=growth.tolist(), worst_kappa_drop=float(kappa_drop.max()),
                       worst_delta_rise=float(delta_rise.max()), violations=violations)
            rows.append(row)
            if violations or not np.all(np.isfinite(growth)):
                failures.append(row)
    report = dict(tolerance=TOLERANCE, rows=rows)
    if destination := os.environ.get("POLE_MAP_P5A_RESULTS"):
        Path(destination).write_text(json.dumps(report, indent=2) + "\n")
    assert not failures, json.dumps(failures, indent=2)
