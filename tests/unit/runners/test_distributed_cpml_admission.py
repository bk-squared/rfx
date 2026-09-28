"""Off-centre pulses in vacuum boxes and PEC plates must survive a device split.

Compare probe electric fields to one device at 1e-4 of each probe's peak
(-80 dB), allowing float32 update reassociation. Unsupported periodic faces
and absorbers that cross device seams must be refused before stepping.
"""

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

pytestmark = pytest.mark.distributed


def _devices(count):
    devices = jax.devices("cpu")[:count]
    if len(devices) < count:
        pytest.skip(f"requires {count} host devices")
    return devices


def _box(lane, *, x_cells=25, x="cpml", plate=False, layers=8, periodic=None):
    dx = 1e-3
    z_cells = 5 if plate else 24
    boundaries = dict(x=x, y="cpml", z=Boundary(lo="pec", hi="pec") if plate else "cpml")
    if periodic:
        boundaries[periodic] = "periodic"
    profiles = {}
    if lane == "nu":
        profile = np.full(x_cells, dx)
        if x_cells > 4:
            profile[1:4] = np.array([0.95, 0.95, 1.1]) * dx
        else:
            profile[1:3] = np.array([0.95, 1.05]) * dx
        profiles["dx_profile"] = profile
    sim = Simulation(
        freq_max=15e9, domain=(x_cells * dx, 24 * dx, z_cells * dx),
        dx=dx, cpml_layers=layers, boundary=BoundarySpec(**boundaries), **profiles,
    )
    sim.add_source((min(6, x_cells - 2) * dx, 9 * dx, (2 if plate else 10) * dx), "ez")
    for pos in ((min(3, x_cells - 3), 9, 2 if plate else 10),
                (min(12, x_cells - 1), 15, 3 if plate else 10)):
        sim.add_probe(tuple(v * dx for v in pos), "ez")
    return sim


def _run(sim, lane, devices=None, **kwargs):
    if lane == "uniform":
        return sim.run(n_steps=240, **({"devices": devices} if devices else {}), **kwargs)
    return sim.forward(n_steps=240, **({"distributed": True, "devices": devices} if devices else {}), **kwargs)


@pytest.mark.parametrize("axis", tuple("xyz"))
@pytest.mark.parametrize("skip_preflight", (False, True))
def test_periodic_nu_refuses_axis(axis, skip_preflight):
    with pytest.raises(NotImplementedError, match=rf"periodic axes '{axis}': periodic / Bloch boundaries are not supported"):
        _run(_box("nu", periodic=axis), "nu", _devices(2), skip_preflight=skip_preflight)


@pytest.mark.parametrize("lane", ("uniform", "nu"))
@pytest.mark.parametrize("count,x_cells,x_wall,face,width,pad", [
    pytest.param(2, 6, True, "x_hi", 7, 1, id="pec_xlo_2"),
    pytest.param(2, 4, False, "x_lo", 7, 1, id="pec_xhi_2"),
    pytest.param(4, 25, True, "x_hi", 7, 2, id="pec_xlo_4"),
    pytest.param(5, 25, True, "x_hi", 6, 1, id="pec_xlo_5"),
    pytest.param(6, 25, None, "x_lo", 7, 0, id="all_cpml_6"),
    pytest.param(8, 25, None, "x_lo", 6, 6, id="all_cpml_8"),
])
def test_absorber_must_fit_owned_slab(lane, count, x_cells, x_wall, face, width, pad):
    devices = _devices(count)
    x = ("cpml" if x_wall is None else
         Boundary(lo="pec", hi="cpml") if x_wall else
         Boundary(lo="cpml", hi="pec"))
    sim = _box(lane, x_cells=x_cells, x=x)
    rank = count - 1 if face.endswith("hi") else 0
    with pytest.raises(ValueError, match=rf"face {face}: boundary rank {rank} has {width} physical x cells, but 8 absorber layers require at least 8 physical x cells plus 1 exchanged ghost cell .*pad_x={pad}"):
        _run(sim, lane, devices, **({"skip_preflight": True} if lane == "nu" else {}))


@pytest.mark.parametrize("lane", ("uniform", "nu"))
@pytest.mark.parametrize("case,count", [
    ("plate", 2), ("no_x_absorber", 2), ("no_x_absorber", 3),
    ("exact_depth", 2), ("exact_depth", 3),
    ("wall_on_short_last_slab", 2), ("unequal_x_depths", 4),
])
def test_admitted_faces_match_one_device(lane, case, count, record_property):
    devices = _devices(count)
    kw = {}
    if case == "plate":
        kw["plate"] = True
    elif case == "no_x_absorber":
        kw.update(x_cells=12 if count == 2 else 25, x=Boundary(lo="pec", hi="pmc"))
    elif case == "exact_depth":
        kw.update(x_cells=7 if count == 2 else 15, x=Boundary(lo="pec", hi="cpml"))
    elif case == "wall_on_short_last_slab":
        kw.update(x_cells=6, x=Boundary(lo="cpml", hi="pec"))
    else:
        if lane == "nu":
            pytest.skip("#1346: NU does not realize unequal face depths")
        kw.update(x_cells=18, x=Boundary(lo="cpml", hi="cpml", lo_thickness=8, hi_thickness=2))
    with jax.default_device(devices[0]):
        one = np.asarray(_run(_box(lane, **kw), lane).time_series)
        result = _run(_box(lane, **kw), lane, devices)
        many = np.asarray(result.time_series)
    if case == "exact_depth":
        assert result.grid.nx == 8 * count  # exactly eight owned rows per slab
    elif case == "wall_on_short_last_slab":
        assert result.grid.nx == 15  # eight rows, then seven and one alignment row
    elif case == "unequal_x_depths":
        assert result.grid.nx == 29  # eight-row low absorber, five-row high slab
    record_property("realized_nx", int(result.grid.nx))
    peak = np.abs(one).max(axis=0)
    assert np.all(peak > 0) and np.isfinite(many).all()
    relative = np.abs(many - one).max(axis=0) / peak
    record_property("relative_probe_error", relative.tolist())
    assert np.all(relative <= 1e-4), relative
