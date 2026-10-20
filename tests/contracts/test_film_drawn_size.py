"""Drawn film size: literal cell areas and the measured sheet edge model."""
import warnings
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.core.yee import init_materials
from rfx.model.materials import assemble_cells
from rfx.model.thin_conductors import (
    DCFilmAdmissionError, admit_dc_film, dc_film_refusals, fold_thin_conductor,
)


def film(*, snap='strict', aligned=False):
    sim = Simulation(10e9, (.024, .016, .012), dx=.001, boundary='pec', snap=snap)
    # T1: 3.5 x 7 mm drawn; the half-cell low edge selects 3 x 7 cells.
    lo, hi = (.003, .006) if aligned else (.0035, .007)
    shape = Box((lo, .003, .005), (hi, .010, .005))
    sim.add_thin_conductor(shape, sigma_bulk=100., thickness=35e-6)
    return sim, sim._build_realized_grid(), shape


def test_t1_box_area_strict_refusal_and_declared_warning():
    sim, grid, shape = film()
    with pytest.raises(DCFilmAdmissionError, match='dc|DC') as caught:
        assemble_cells(sim, grid)
    assert caught.value.code == 'dc_film_area'
    message = str(caught.value)
    assert float(re.search(r'A_d=([^ ]+)', message)[1]) == pytest.approx(2.45e-5, abs=1e-18)
    assert float(re.search(r'A_r=([^ ]+)', message)[1]) == pytest.approx(2.1e-5, abs=1e-18)
    assert 'e=sqrt(A_r/A_d)-1=-0.0741799002274' in message
    assert 0 in dc_film_refusals(sim, grid)
    report = sim.preflight(check_ntff=False, check_resolution=False)
    assert any(i.code == 'dc_film_area' and i.severity == 'error' for i in report)
    sim._snap = 'declared'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        admitted = admit_dc_film(shape, grid, snap='declared')
    assert admitted.declared_area == pytest.approx(2.45e-5, abs=1e-18)
    assert admitted.realized_area == pytest.approx(2.1e-5, abs=1e-18)
    assert any(getattr(w.message, 'code', '') == 'dc_film_area'
               and w.message.severity == 'warning' for w in caught)


def test_t2_on_lattice_area_and_traced_fold():
    sim, grid, shape = film(aligned=True)
    admitted = admit_dc_film(shape, grid)
    assert admitted.declared_area == pytest.approx(2.1e-5, abs=1e-18)
    assert admitted.realized_area == pytest.approx(2.1e-5, abs=1e-18)
    cells = assemble_cells(sim, grid)[0]
    # A traced graded coordinate must preserve the pre-existing Box fold.
    def traced(profile):
        model = Simulation(10e9, (.024, .016, .012), dx=.001,
                           boundary='pec', dx_profile=profile)
        model.add_thin_conductor(shape, sigma_bulk=100., thickness=35e-6)
        g = model._build_realized_grid()
        return fold_thin_conductor(g, model._thin_conductors[0], init_materials(g.shape))[0].sigma_film[0]
    actual = jax.jit(traced)(jnp.full(24, .001))
    np.testing.assert_allclose(actual, cells.sigma_film[0], rtol=2e-7, atol=0)


@pytest.mark.parametrize('snap,severity', [('strict', 'error'), ('declared', 'warning')])
def test_t3_f0_sheet_effective_size(snap, severity):
    sim = Simulation(10e9, (.024, .016, .012), dx=.001, boundary='pec', snap=snap)
    # Closed nodes 4..7 span 3 mm; solved span is 3.7 mm vs 3.5 mm drawn.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim.add_thin_conductor(Box((.0035, .003, .005), (.007, .010, .005)),
                               sigma_bulk=5.8e7, surface_impedance_f0=5e9)
    report = sim.preflight(check_ntff=False, check_resolution=False)
    issues = [i for i in report if i.code == 'sheet_effective_size']
    assert len(issues) == 1
    assert issues[0].severity == severity
    assert 'add_thin_conductor(surface_impedance_f0=...)' in str(issues[0])
    assert '+5.71%' in str(issues[0])


@pytest.mark.parametrize('graded', [False, True])
def test_t4_whole_plane_on_cpml_grid(graded):
    kw = {'dy_profile': .001 * np.array([1.] * 4 + [.8, 1.2] * 4 + [1.] * 4)} if graded else {}
    sim = Simulation(10e9, (.024, .016, .012), dx=.001, boundary='cpml', cpml_layers=8, **kw)
    shape = Box((.010, -1., -1.), (.010, 1., 1.))
    sim.add_thin_conductor(shape, sigma_bulk=100., thickness=35e-6)
    grid = sim._build_realized_grid()
    admitted = admit_dc_film(shape, grid)
    # 16 + 2*8 mm by 12 + 2*8 mm, including CPML padding.
    assert admitted.declared_area == pytest.approx(.000896, abs=1e-16)
    assert admitted.realized_area == pytest.approx(.000896, abs=1e-16)
    assert abs(np.sqrt(admitted.realized_area / admitted.declared_area) - 1) <= 1e-12
    assemble_cells(sim, grid)
