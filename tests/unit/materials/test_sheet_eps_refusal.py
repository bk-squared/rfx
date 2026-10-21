"""Sheet declarations refuse permittivity that their update cannot use."""

import jax
import pytest

from rfx import Box, Simulation
from rfx.model.materials import assemble_cells
from rfx.model.thin_conductors import DCFilmAdmissionError


@pytest.mark.parametrize('mode, eps, outcome', [
    pytest.param('pinned', 1.2, 'refused', id='pinned-nonunit'),
    pytest.param('pec', 1.2, 'refused', id='pec-nonunit'),
    pytest.param('f0', 1.2, 'refused', id='f0-nonunit'),
    pytest.param('pinned', 1.2, 'traced', id='pinned-traced'),
    pytest.param('pec', 1.2, 'traced', id='pec-traced'),
    pytest.param('f0', 1.2, 'traced', id='f0-traced'),
    pytest.param('pinned', None, 'admitted', id='pinned-default'),
    pytest.param('pec', None, 'admitted', id='pec-default'),
    pytest.param('f0', None, 'admitted', id='f0-default'),
    pytest.param('pinned', 1, 'admitted', id='pinned-int'),
    pytest.param('pec', 1, 'admitted', id='pec-int'),
    pytest.param('f0', 1, 'admitted', id='f0-int'),
    pytest.param('dc', 1.0, 'admitted', id='dc-unit'),
    pytest.param('dc', 1.2, 'assembly-refused', id='dc-nonunit'),
])
def test_sheet_eps_refusal(mode, eps, outcome):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001,
                     boundary='pec')

    def declare(value):
        kwargs = {} if value is None else {'eps_r': value}
        if mode == 'pinned':
            sim.add_pinned_sheet(plane_index=4, i_range=(2, 6), j_range=(2, 6),
                                 **kwargs)
        else:
            if mode == 'f0':
                kwargs['surface_impedance_f0'] = 10e9
            sim.add_thin_conductor(Box((.002, .002, .004), (.006, .006, .004)),
                                   sigma_bulk=1e4 if mode == 'dc' else 5.8e7,
                                   **kwargs)
        return 0.0

    if outcome == 'refused':
        with pytest.raises(ValueError, match='eps_r'):
            declare(eps)
    elif outcome == 'traced':
        with pytest.raises(ValueError, match='eps_r'):
            jax.grad(declare)(eps)
    else:
        declare(eps)
        record = (sim._pinned_sheets if mode == 'pinned' else sim._thin_conductors)[0]
        if outcome == 'assembly-refused':
            assert record.eps_r == 1.2
            with pytest.raises(DCFilmAdmissionError) as error:
                assemble_cells(sim, sim._build_grid())
            assert error.value.code == 'dc_film_permittivity'
        else:
            assert record.eps_r == 1.0
