"""Declared graded twins of the existing periodic judges."""
import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.grid import Grid


def profiles(domain, d, periodic, *, automatic=False):
    result = {}
    for axis, length in zip("xyz", domain):
        if axis in periodic or (automatic and axis != "z"):
            continue
        n = round(length / d)
        cells = np.full(n, d)
        if n >= 2:
            i = min(3, n - 2)
            cells[i:i+2] = (.9*d, 1.1*d)
        result[f"d{axis}_profile"] = cells
    if not result:
        result["dz_profile"] = np.full(round(domain[2] / d), d)
    return result


def simulation(graded, freq, domain, **kwargs):
    if graded:
        boundary = kwargs["boundary"]
        periodic = "".join(a for a in "xyz" if getattr(boundary, a).lo == "periodic")
        d = kwargs.get("dx", 299792458.0 / freq / 20)
        kwargs.update(profiles(domain, d, periodic, automatic=kwargs.get("dx") is None))
    return Simulation(freq, (domain[0], domain[1], float(np.sum(kwargs["dz_profile"]))) if "dz_profile" in kwargs else domain, **kwargs)


def grid(graded, freq, domain, *, dx=None, periodic_axes="", **kwargs):
    if not graded:
        return Grid(freq, domain, dx=dx, periodic_axes=periodic_axes, **kwargs)
    boundary = BoundarySpec(**{a: "periodic" if a in periodic_axes else "pec" for a in "xyz"})
    options = {} if dx is None else {"dx": dx}
    return simulation(True, freq, domain, boundary=boundary, cpml_layers=0,
                      **options)._build_nonuniform_grid()
