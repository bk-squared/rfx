"""Declaration-time messages, warning locations, and admission ownership."""
import inspect
import math
import warnings

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.geometry.csg import Box, OrientedBox
class NoMask:
    pass


class NoBounds:
    def mask_on_coords(self, x, y, z):
        return True


def simulation():
    return Simulation(freq_max=10e9, domain=(6e-3, 6e-3, 3e-3), dx=1e-3)


def sheet():
    return Box((1e-3, 1e-3, 1e-3), (5e-3, 5e-3, 1e-3))


@pytest.mark.parametrize('shape,kwargs,expected', [
    (sheet(), {'surface_impedance_f0': 0.0},
     'add_thin_conductor: surface_impedance_f0=0.0 must be a positive frequency '
     'in Hz (band centre for the Leontovich surface resistance).'),
    (sheet(), {'surface_impedance_f0': -1e9},
     'add_thin_conductor: surface_impedance_f0=-1000000000.0 must be a positive frequency '
     'in Hz (band centre for the Leontovich surface resistance).'),
    (sheet(), {'surface_impedance_f0': 1e10, 'sigma_bulk': 0.0},
     'add_thin_conductor: sigma_bulk=0.0 must be > 0 when surface_impedance_f0 is set '
     '— Rs0 = sqrt(pi*f0*mu0/sigma_bulk) is undefined otherwise.'),
    (NoMask(), {'surface_impedance_f0': 1e10},
     'add_thin_conductor: surface_impedance_f0 requires a shape that implements '
     'mask_on_coords(x, y, z) — the sheet is rasterized from that mask on both lanes, '
     'and NoMask does not provide it.'),
    (NoBounds(), {'surface_impedance_f0': 1e10},
     'add_thin_conductor: surface_impedance_f0 requires a shape with an axis-aligned '
     'bounding box (Box corner_lo/corner_hi, or Shape.bounding_box()) — the sheet '
     'NORMAL is read from it, and on a graded mesh that normal decides which dual '
     'spacing normalizes the fold. NoBounds provides neither.'),
    # Two faults at once: the f0 check comes first, and every check runs
    # before the record is built (ThinConductor refuses an OrientedBox
    # with NotImplementedError).
    (NoMask(), {'surface_impedance_f0': 0.0},
     'add_thin_conductor: surface_impedance_f0=0.0 must be a positive frequency '
     'in Hz (band centre for the Leontovich surface resistance).'),
    (OrientedBox((3e-3, 3e-3, 1.5e-3), (2e-3, 2e-3, 1e-3), np.eye(3)),
     {'surface_impedance_f0': 0.0},
     'add_thin_conductor: surface_impedance_f0=0.0 must be a positive frequency '
     'in Hz (band centre for the Leontovich surface resistance).'),
    (sheet(), {'surface_impedance_f0': 0.0, 'sigma_bulk': 0.0},
     'add_thin_conductor: surface_impedance_f0=0.0 must be a positive frequency '
     'in Hz (band centre for the Leontovich surface resistance).'),
    (NoMask(), {'surface_impedance_f0': 1e10, 'sigma_bulk': 0.0},
     'add_thin_conductor: sigma_bulk=0.0 must be > 0 when surface_impedance_f0 is set '
     '— Rs0 = sqrt(pi*f0*mu0/sigma_bulk) is undefined otherwise.'),
], ids=['R1', 'R2', 'R3', 'R4', 'R5', 'R6_f0_first', 'R7_before_record',
        'R8_f0_before_sigma', 'R9_sigma_before_mask'])
def test_refusals(shape, kwargs, expected):
    sim = simulation()
    with pytest.raises(ValueError) as exc:
        sim.add_thin_conductor(shape, **kwargs)
    assert str(exc.value) == expected
    assert len(sim._thin_conductors) == 0


@pytest.mark.parametrize('kwargs,thickness_text', [
    ({}, ' thickness is not used'),
    ({'thickness': 17e-6}, ' You passed thickness=1.7e-05 m, which is not used'),
], ids=['W1', 'W2'])
def test_pec_warning(kwargs, thickness_text):
    sim = simulation()
    expected = (
        'add_thin_conductor: sigma_bulk=5.80e+07 S/m is at or above the 1e+06 S/m '
        'PEC threshold, so this is a LOSSLESS PEC sheet.' + thickness_text +
        ' here, and neither is eps_r. Every metal is above this threshold — a lower '
        'sigma_bulk would be a different material, not thinner copper. '
        'For conductor loss pass surface_impedance_f0=<band centre Hz> '
        '(Leontovich sheet).'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        line = inspect.currentframe().f_lineno + 1
        sim.add_thin_conductor(sheet(), **kwargs)
    assert len(caught) == 1
    w = caught[0]
    from rfx.errors import ConfigurationWarning
    assert w.category is ConfigurationWarning
    assert str(w.message) == expected
    assert w.filename == __file__
    assert w.lineno == line
    assert len(sim._thin_conductors) == 1


def test_leontovich_warning_W3():
    sim = simulation()
    f0, sigma = 1e10, 5.8e7
    rs0 = math.sqrt(math.pi * f0 * 1.25663706212e-6 / sigma)
    expected = (
        'add_thin_conductor: surface_impedance_f0=10000000000.0 Hz — Leontovich '
        f'surface-resistance sheet with Rs0 = {rs0:.4g} ohm/sq. Rs is FREQUENCY-FLAT '
        'within the run, frozen at f0, with relative band error |sqrt(f/f0)-1|; '
        'the reactive part Xs = Rs (internal inductance) is NOT modeled, '
        'so resonances read slightly high.'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        line = inspect.currentframe().f_lineno + 1
        sim.add_thin_conductor(sheet(), surface_impedance_f0=f0, sigma_bulk=sigma)
    assert len(caught) == 1
    w = caught[0]
    from rfx.errors import AccuracyWarning
    assert w.category is AccuracyWarning
    assert str(w.message) == expected
    assert w.filename == __file__
    assert w.lineno == line
    assert len(sim._thin_conductors) == 1


def test_traced_warning_W4():
    sim = simulation()
    lines = []

    def declare(f0):
        lines.append(inspect.currentframe().f_lineno + 1)
        sim.add_thin_conductor(sheet(), surface_impedance_f0=f0)
        return f0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        jax.make_jaxpr(declare)(1e10)
    assert len(caught) == 1
    w = caught[0]
    from rfx.errors import AccuracyWarning
    assert w.category is AccuracyWarning
    prefix, rest = str(w.message).split('surface_impedance_f0=', 1)
    tracer, suffix = rest.split(' Hz — ', 1)
    assert tracer
    assert prefix == 'add_thin_conductor: '
    assert suffix == (
        'Leontovich surface-resistance sheet with Rs0 = sqrt(pi*f0*mu0/sigma_bulk) '
        '(traced value not evaluated here). Rs is FREQUENCY-FLAT within the run, '
        'frozen at f0, with relative band error |sqrt(f/f0)-1|; the reactive part '
        'Xs = Rs (internal inductance) is NOT modeled, so resonances read slightly high.'
    )
    assert w.filename == __file__
    assert w.lineno == lines[0]
    assert len(sim._thin_conductors) == 1
