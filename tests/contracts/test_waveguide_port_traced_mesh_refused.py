"""A waveguide port on a mesh whose cell widths are traced is refused by name (#2183).

A WR-90 section (a = 22.86 mm, b = 10.16 mm, dx = 2.54 mm: 9 x 4 whole cells),
PEC walls, CPML on x, an eps_r 4 slab four cells long and two TE10 ports. One
derivative stretches the slab's four x cells and shrinks the next four by the
same amount (the guide keeps its length); the other does the same on two y
cells across the aperture. The graded port reads its planes, mode aperture,
absorber depth and flux planes from host cell widths, and the flux monitor
refuses traced cells, so no cell-width derivative reaches S. Before the fix the
first host read raised a raw TracerArrayConversionError; admission now refuses
before the first time step, naming the ports and the traced axis. The same
guide on the uniform mesh with its one cell size traced is refused the same way.

Physical anchor: none. This file pins a refusal, not a number a user receives;
the concrete primal still runs (row 3). Imitates
tests/contracts/test_waveguide_multimode_refused.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx._traced_refusal import TracedStructureRefusal
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box

C0 = 299792458.0
DX = 2.54e-3
NX, NY, NZ = 40, 9, 4
S_LO, N_S = 18, 4
D_MIN = DX * (1 - 0.5 / N_S)
DT = 0.9 / (C0 * np.sqrt(1 / D_MIN ** 2 + 2 / DX ** 2))
FREQS = np.linspace(8.5e9, 11.5e9, 4)


def _profile(n, lo, width, delta):
    if is_tracer(delta):
        d = jnp.full((n,), DX)
        d = d.at[lo:lo + width].add(delta / width)
        return d.at[lo + width:lo + 2 * width].add(-delta / width).astype(jnp.float32)
    d = np.full((n,), DX)
    d[lo:lo + width] += float(delta) / width
    d[lo + width:lo + 2 * width] -= float(delta) / width
    return d


def _section(delta, axis):
    profiles = {"x": _profile(NX, S_LO, N_S, delta if axis == "x" else 0.0),
                "y": _profile(NY, 3, 2, delta if axis == "y" else 0.0)}
    sim = Simulation(freq_max=13e9, domain=(NX * DX, NY * DX, NZ * DX), dx=DX,
                     boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                                           y=Boundary(lo="pec", hi="pec"),
                                           z=Boundary(lo="pec", hi="pec")),
                     cpml_layers=8, dx_profile=profiles["x"], dy_profile=profiles["y"],
                     dt=float(DT), dt_min_cell=float(D_MIN))
    sim.add_material("slab", eps_r=4.0)
    sim.add(Box((S_LO * DX, 0, 0), ((S_LO + N_S) * DX, NY * DX, NZ * DX)), material="slab")
    for x, direction, name in ((5 * DX, "+x", "left"), ((NX - 5) * DX, "-x", "right")):
        sim.add_waveguide_port(x, direction=direction, mode=(1, 0), mode_type="TE",
                               freqs=FREQS, f0=10.25e9, bandwidth=0.5, name=name)
    return sim


def _read(sim, entry):
    if entry == "s_matrix":
        r = sim.compute_waveguide_s_matrix(n_steps=2, normalize="flux", skip_preflight=True)
        return jnp.abs(jnp.asarray(r.s_params)[1, 0, 0])
    if entry == "forward":
        return jnp.sum(jnp.abs(sim.forward(n_steps=2, checkpoint=False, skip_preflight=True).time_series))
    return jnp.sum(jnp.abs(jnp.asarray(sim.run(n_steps=2, skip_preflight=True).time_series)))


@pytest.mark.parametrize("axis", ("x", "y"))
@pytest.mark.parametrize("entry", ("s_matrix", "run", "forward"))
def test_traced_cell_widths_refused_by_port_name(monkeypatch, entry, axis):
    def forbidden(*args, **kwargs):
        pytest.fail("a waveguide port on a traced mesh reached lax.scan")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(TracedStructureRefusal) as caught:
        jax.jacfwd(lambda d: _read(_section(d, axis), entry))(0.0)
    message = str(caught.value)
    assert "waveguide port(s) 'left', 'right'" in message
    assert f"whose {axis} cell widths are traced" in message
    assert "eps_override" in message


def test_concrete_primal_still_runs():
    """The refusal reads tracers only: the same section with concrete widths solves."""
    value = float(_read(_section(0.25 * DX, "x"), "s_matrix"))
    assert np.isfinite(value)


def _uniform_guide(dx):
    """The same guide on the uniform mesh, its one cell size traced."""
    sim = Simulation(12e9, (.1016, .02286, .01016), dx=dx,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=8)
    for x, direction, name in ((.02, "+x", "left"), (.08, "-x", "right")):
        sim.add_waveguide_port(x, direction=direction, name=name, f0=10e9, freqs=FREQS)
    return sim


@pytest.mark.parametrize("entry", ("s_matrix", "run"))
def test_traced_scalar_cell_size_refused_by_port_name(monkeypatch, entry):
    def forbidden(*args, **kwargs):
        pytest.fail("a waveguide port with a traced cell size reached lax.scan")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(TracedStructureRefusal) as caught:
        jax.jacfwd(lambda d: _read(_uniform_guide(d), entry))(0.00127)
    message = str(caught.value)
    assert "waveguide port(s) 'left', 'right'" in message
    assert "whose dx cell widths are traced" in message
