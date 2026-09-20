"""A structure declared out to a padded face must be rasterized to it (#1070).

``extend_cpml_pad_materials`` fills a hi pad by replicating the interior-edge
column outward, and its #627a fallback looks exactly one column further in when
that edge is vacuum. The docstring states the bound -- the half-open
``[lo, hi)`` Box rule costs the hi face "deterministically one node ... never
more". Grid sizing could invent a SECOND empty node: ``ceil`` of a ratio one
ULP above an integer allocated a cell no declared Box reaches. The fallback
then found vacuum in both columns it inspects, gave up, and filled the whole
absorber with vacuum.

The assertion here pins the bound the fallback documents, not the arithmetic
that broke it, so any other route to an unfilled interior node at a pad seam is
caught too. Build-only.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.geometry.rasterize_grid import PadFillShortfall

#: The #1070 rig: RO4003C at h = 0.787 mm, dx = h/4, cpml_layers 8. +10h is
#: the one pad value whose x ratio is BOTH one ULP above an integer AND has
#: ``232*dx`` reproducing the declared length exactly, which is what costs the
#: second node. +8h and +12h round up too but lose only one node, so #627a
#: repairs them and they are not red rigs.
H = 0.787e-3
DX = H / 4.0
EPS_R = 3.38
CPML = 8


def _rig(pad_h: int = 10, *, hi_shrink_cells: float = 0.0):
    dom_x = 38 * H + 2 * (pad_h * H)
    dom_y = 23 * H + 2 * (pad_h * H)
    dom_z = 16 * H
    sim = Simulation(freq_max=10e9, domain=(dom_x, dom_y, dom_z), dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    box_hi_x = dom_x - hi_shrink_cells * DX
    sim.add(Box((0.0, 0.0, 6 * H), (box_hi_x, dom_y, 8 * H)),
            material="ro4003c")
    return sim


def _size_by_plain_ceil(monkeypatch) -> None:
    """Put the pre-#1070 sizing back, keeping everything else."""
    import rfx.grid as grid_mod

    monkeypatch.setattr(grid_mod, "cells_spanning",
                        lambda length, dx, **kw: int(math.ceil(length / dx)))


def test_the_rig_that_found_this_is_silent_now() -> None:
    """Green-after, and the measurement behind it: the +10h x pad carries the
    substrate rather than vacuum."""
    sim = _rig(10)
    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    assert list(grid.shape) == [249, 189, 81], list(grid.shape)
    k = eps.shape[2] // 2
    zs = [z for z in range(eps.shape[2])
          if abs(eps[eps.shape[0] // 2, eps.shape[1] // 2, z] - EPS_R) < 1e-12]
    k = zs[len(zs) // 2]
    assert float(eps[-1, eps.shape[1] // 2, k]) == pytest.approx(EPS_R)


def test_it_fires_when_the_grid_is_sized_the_old_way(monkeypatch) -> None:
    """Red-before, reached by restoring only the sizing.

    This is the two-node shortfall: ``ceil`` invents node 233 and node 232
    sits exactly on the box's hi face, which the half-open rule excludes.
    """
    _size_by_plain_ceil(monkeypatch)
    sim = _rig(10)
    grid = sim._build_grid()
    assert list(grid.shape) == [250, 189, 81], (
        "the old sizing is not in place; this fixture proves nothing")
    with pytest.raises(PadFillShortfall) as excinfo:
        sim._assemble_materials(grid)
    message = str(excinfo.value)
    assert "ro4003c" in message
    assert "x-hi" in message
    assert "2 interior nodes short" in message, message


@pytest.mark.parametrize("pad_h", [6, 8, 12])
def test_the_pad_values_that_lose_only_one_node_stay_silent(
        monkeypatch, pad_h: int) -> None:
    """+8h and +12h round up as well, and #627a repairs them.

    Silent even with the old sizing, which is the measurement that says the
    extra cell alone is not the defect.
    """
    _size_by_plain_ceil(monkeypatch)
    sim = _rig(pad_h)
    sim._assemble_materials(sim._build_grid())


@pytest.mark.parametrize("shrink", [0.3, 1.0, 3.0])
def test_a_declared_air_gap_before_the_absorber_is_left_alone(
        shrink: float) -> None:
    """The common case, and the one an unbounded backward scan would break: a
    box that does not reach the face was never declared to, so its vacuum is
    the declared structure and not a shortfall."""
    sim = _rig(10, hi_shrink_cells=shrink)
    sim._assemble_materials(sim._build_grid())


def test_an_air_gap_is_left_alone_under_the_old_sizing_too(monkeypatch) -> None:
    """Same, with the defect's arithmetic in place: the check keys on what was
    DECLARED, so it cannot be tripped by the grid being one cell too long."""
    _size_by_plain_ceil(monkeypatch)
    sim = _rig(10, hi_shrink_cells=3.0)
    sim._assemble_materials(sim._build_grid())


def test_a_box_landing_exactly_on_the_face_is_not_a_shortfall() -> None:
    """The documented one-node case, stated as its own test.

    ``domain = n*dx`` exactly and the box reaches the face: the half-open rule
    drops that last node, #627a repairs it, and the assertion must not call
    it a defect.
    """
    dom = 64 * DX
    sim = Simulation(freq_max=10e9, domain=(dom, dom, 16 * DX), dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_material("slab", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0.0, 0.0, 4 * DX), (dom, dom, 8 * DX)), material="slab")
    grid = sim._build_grid()
    sim._assemble_materials(grid)
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    k = eps.shape[2] // 2
    zs = [z for z in range(eps.shape[2])
          if abs(eps[eps.shape[0] // 2, eps.shape[1] // 2, z] - EPS_R) < 1e-12]
    k = zs[len(zs) // 2]
    assert float(eps[-1, eps.shape[1] // 2, k]) == pytest.approx(EPS_R)


def test_the_check_is_skipped_where_no_pad_will_be_filled() -> None:
    """A PEC-walled domain has no pad to replicate into, so the question does
    not arise and the check must not invent it."""
    sim = _rig(10)
    sim._boundary = "pec"
    sim._assemble_materials(sim._build_grid())
