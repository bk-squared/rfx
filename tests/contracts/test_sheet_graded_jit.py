"""A concrete PEC sheet remains geometry under an outer graded run trace."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, GaussianPulse, Simulation
from rfx.boundaries.pec import SheetFootprint, SheetSpec
from rfx.errors import RfxRefusal
from rfx.geometry.rasterize_grid import (
    coords_from_uniform_grid, refuse_vaporized_sheets, sheet_spec_from_shape,
)


def assert_trace_records(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    peak = np.max(np.abs(expected))
    assert peak > 0
    assert np.max(np.abs(actual - expected)) <= 9 * np.spacing(np.float32(peak)), 'per-step'
    assert np.max(np.abs(actual.sum(axis=0) - expected.sum(axis=0))) <= 1e-4 * peak, 'sum'


@pytest.mark.parametrize('shape', (
    Box((.003, .003, .003), (.009, .007, .003)),
    Cylinder((.006, .005, .003), radius=.002, height=0., axis='z'),
), ids=('box', 'disc'))
def test_graded_sheet_run_under_user_jit(shape):
    profile = np.array([.001] * 4 + [.0009, .0011] * 2 + [.001] * 4)
    sim = Simulation(freq_max=10e9, domain=(.012, .010, .010), dx=.001,
                     dx_profile=profile, boundary='pec')
    sim.add_thin_conductor(shape)
    sim.add_source((.005, .005, .005), 'ez', amplitude_kind='current',
                   waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    sim.add_probe((.005, .005, .005), 'ez')

    def run():
        return sim.run(n_steps=24, skip_preflight=True).time_series

    # Trace first so a preceding eager call cannot supply cached geometry.
    compiled = jax.jit(run)()
    assert_trace_records(compiled, run())


def test_trace_check_rejects_changed_record():
    expected = np.ones((4, 1), dtype=np.float32)
    with pytest.raises(AssertionError, match='per-step'):
        assert_trace_records(expected * 2, expected)


@pytest.mark.parametrize('geometry', ('concrete', 'zero-edge', 'traced-footprint', 'traced-join'))
def test_zero_edge_plane_check_under_user_jit(geometry):
    sim = Simulation(freq_max=1e6, domain=(4., 4., 4.), dx=1., boundary='pec')
    coords = coords_from_uniform_grid(sim._build_grid())

    def check(displacement):
        # Rasterize inside the user trace, before any eager geometry cache.
        hi = 1.1 if geometry == 'zero-edge' else 2.1
        sheet = sheet_spec_from_shape(Box((.9, .9, 2.), (hi, hi, 2.)), coords)
        assert isinstance(sheet.footprint, SheetFootprint)
        sheets = [sheet]
        if geometry == 'traced-footprint':
            sheets = [SheetSpec(2, 2, jnp.asarray(np.asarray(sheet.footprint))
                                & (displacement > 0))]
        elif geometry == 'traced-join':
            # Concrete footprints can still have a traced physical join.
            # Neither singleton owns an edge; the join decides their union.
            sheets = []
            for x in (1, 2):
                footprint = np.zeros(coords.shape, dtype=bool)
                footprint[x, 1, 2] = True
                drawn = ((x - .5 + displacement, x + .5 + displacement, 1e-6),
                         (.5, 1.5, 1e-6), None)
                sheets.append(SheetSpec(2, 2, footprint, drawn=drawn))
        refuse_vaporized_sheets(sheets)
        return displacement

    if geometry == 'zero-edge':
        with pytest.raises(RfxRefusal, match='realizes ZERO PEC edges'):
            jax.jit(check)(0.)
    else:
        assert float(jax.jit(check)(0.)) == 0.
