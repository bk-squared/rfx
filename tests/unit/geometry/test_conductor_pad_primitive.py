"""Build-only checks of the C2 layer-copy primitive, without port windows."""
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pec import SheetSpec, WireSpec
from rfx.geometry.rasterize_grid import (
    box_at_realized_faces,
    extend_cpml_pad_conductors,
)
from rfx.materials.thin_conductor import SheetImpedanceSpec


SHAPE = (10, 11, 12)


def _expected(array, pads, cell_axes=(), normal=None):
    # Direct index map into the original interior, independent of the
    # implementation's sequential slice writes. Terminal cell-storage
    # planes are outside the physical lattice and keep their input values.
    out = np.array(array, copy=True)
    for index in np.ndindex(array.shape):
        if any(index[a] == array.shape[a] - 1 for a in cell_axes):
            continue
        source = list(index)
        for axis in range(3):
            if axis == normal:
                continue
            lo, hi = pads[2 * axis:2 * axis + 2]
            end = array.shape[axis] - int(axis in cell_axes)
            if lo and index[axis] < lo:
                source[axis] = lo
            elif hi and index[axis] >= end - hi:
                source[axis] = end - hi - 1
        out[index] = array[tuple(source)]
    return out


def _input(pads, cell_axes=(), normal=None):
    a = np.zeros(SHAPE, dtype=bool)
    region = []
    for axis, n in enumerate(SHAPE):
        lo, hi = pads[2 * axis:2 * axis + 2]
        region.append(slice(lo, n - hi - int(axis in cell_axes)))
    if normal is not None:
        region[normal] = slice(SHAPE[normal] // 2, SHAPE[normal] // 2 + 1)
    a[tuple(region)] = True
    # An interior hole makes the test sensitive to the transverse profile.
    hole = tuple(n // 2 for n in SHAPE)
    a[hole] = False
    return a


@pytest.mark.parametrize("face", range(6))
@pytest.mark.parametrize("kind", ["cells", "sheet", "impedance", "wire"])
def test_each_face_and_lattice(kind, face):
    pads = [0] * 6
    pads[face] = 2
    axis = face // 2
    normal = (axis + 1) % 3
    cell_axes = (0, 1, 2) if kind == "cells" else ()
    original = _input(pads, cell_axes, normal if kind in ("sheet", "impedance") else None)
    cells, sheets, wires, specs = None, (), (), ()
    if kind == "cells":
        cells = jnp.asarray(original)
    elif kind == "sheet":
        sheets = (SheetSpec(normal, SHAPE[normal] // 2, jnp.asarray(original), "plate"),)
    elif kind == "impedance":
        specs = (SheetImpedanceSpec(jnp.asarray(original), normal, 3.0,
                                   jnp.asarray(original, dtype=jnp.float32) * 7.0,
                                   SHAPE[normal] // 2),)
    else:
        wire_inputs = tuple(_input(pads, (c,)) for c in range(3))
        wires = (WireSpec(tuple(jnp.asarray(a) for a in wire_inputs), "filament"),)
    got_cells, got_sheets, got_wires, got_specs = extend_cpml_pad_conductors(
        cells, sheets, wires, specs, *pads)
    if kind == "cells":
        np.testing.assert_array_equal(got_cells, _expected(original, pads, (0, 1, 2)))
        assert not np.array_equal(got_cells, original)
    elif kind == "sheet":
        np.testing.assert_array_equal(got_sheets[0].footprint, _expected(original, pads, normal=normal))
        assert got_sheets[0].plane == sheets[0].plane
        assert got_sheets[0].name == "plate"
    elif kind == "impedance":
        expected = _expected(original, pads, normal=normal)
        np.testing.assert_array_equal(got_specs[0].mask, expected)
        np.testing.assert_array_equal(got_specs[0].sigma_sheet, expected * 7.0)
        assert got_specs[0].g_sheet == 3.0
    else:
        for c, output in enumerate(got_wires[0].edges):
            np.testing.assert_array_equal(output, _expected(wire_inputs[c], pads, (c,)))
        assert got_wires[0].name == "filament"


@pytest.mark.parametrize("zero_face", [None, 0, 1, 2, 3, 4, 5])
def test_corners_zero_pad_and_terminal_storage(zero_face):
    pads = [2, 1, 1, 2, 2, 1]
    if zero_face is not None:
        pads[zero_face] = 0
    rng = np.random.default_rng(801)
    original = rng.random(SHAPE) > 0.5
    got, _, _, _ = extend_cpml_pad_conductors(jnp.asarray(original), (), (), (), *pads)
    np.testing.assert_array_equal(got, _expected(original, pads, (0, 1, 2)))
    for axis in range(3):
        np.testing.assert_array_equal(np.take(np.asarray(got), -1, axis=axis), np.take(original, -1, axis=axis))


@pytest.mark.parametrize("axis", range(3))
def test_cell_mirror_uses_physical_cells(axis):
    pads = [2, 1, 1, 2, 2, 1]
    original = _input(pads, (0, 1, 2))
    physical = tuple(slice(0, n - 1) for n in SHAPE)
    mirrored = original.copy()
    mirrored[physical] = np.flip(original[physical], axis=axis)
    flipped_pads = list(pads)
    flipped_pads[2 * axis:2 * axis + 2] = pads[2 * axis:2 * axis + 2][::-1]
    a, _, _, _ = extend_cpml_pad_conductors(jnp.asarray(original), (), (), (), *pads)
    b, _, _, _ = extend_cpml_pad_conductors(jnp.asarray(mirrored), (), (), (), *flipped_pads)
    np.testing.assert_array_equal(a, _expected(original, pads, (0, 1, 2)))
    np.testing.assert_array_equal(np.asarray(b)[physical], np.flip(np.asarray(a)[physical], axis=axis))


@pytest.mark.parametrize("normal", range(3))
def test_sheet_normal_is_not_extruded(normal):
    pads = [2] * 6
    original = _input(pads, normal=normal)
    # Put the sheet on a face plane. The normal pad still stays empty.
    original = np.roll(original, 2 - SHAPE[normal] // 2, axis=normal)
    sheet = SheetSpec(normal, 2, jnp.asarray(original))
    _, sheets, _, _ = extend_cpml_pad_conductors(None, (sheet,), (), (), *pads)
    expected = _expected(original, pads, normal=normal)
    np.testing.assert_array_equal(sheets[0].footprint, expected)
    assert np.asarray(sheets[0].footprint).sum() > original.sum()
    assert not np.take(sheets[0].footprint, [0, 1], axis=normal).any()


@pytest.mark.parametrize("kind", ["sheet", "impedance", "wire"])
@pytest.mark.parametrize("zero_face", [None, 0, 1, 2, 3, 4, 5])
def test_other_lattices_corners_and_zero_pad(kind, zero_face):
    pads = [2, 1, 1, 2, 2, 1]
    if zero_face is not None:
        pads[zero_face] = 0
    rng = np.random.default_rng(801)
    original = rng.random(SHAPE) > 0.5
    if kind == "wire":
        wire = WireSpec((jnp.asarray(original),) * 3)
        _, _, wires, _ = extend_cpml_pad_conductors(None, (), (wire,), (), *pads)
        for component, edges in enumerate(wires[0].edges):
            np.testing.assert_array_equal(edges, _expected(original, pads, (component,)))
    else:
        for normal in range(3):
            mask = np.zeros_like(original)
            plane = SHAPE[normal] // 2
            selection = [slice(None)] * 3
            selection[normal] = plane
            mask[tuple(selection)] = original[tuple(selection)]
            if kind == "sheet":
                sheet = SheetSpec(normal, plane, jnp.asarray(mask))
                _, sheets, _, _ = extend_cpml_pad_conductors(None, (sheet,), (), (), *pads)
                np.testing.assert_array_equal(sheets[0].footprint, _expected(mask, pads, normal=normal))
            else:
                sigma = mask * rng.uniform(1.0, 7.0, SHAPE).astype(np.float32)
                spec = SheetImpedanceSpec(jnp.asarray(mask), normal, 3.0,
                                          jnp.asarray(sigma), plane)
                _, _, _, specs = extend_cpml_pad_conductors(None, (), (), (spec,), *pads)
                np.testing.assert_array_equal(specs[0].mask, _expected(mask, pads, normal=normal))
                np.testing.assert_array_equal(specs[0].sigma_sheet, _expected(sigma, pads, normal=normal))


@pytest.mark.parametrize("kind", ["sheet", "impedance", "wire"])
@pytest.mark.parametrize("axis", range(3))
def test_other_lattices_mirror(kind, axis):
    pads = [2, 1, 1, 2, 2, 1]
    flipped_pads = list(pads)
    flipped_pads[2 * axis:2 * axis + 2] = pads[2 * axis:2 * axis + 2][::-1]
    rng = np.random.default_rng(801)

    def flip(array, cell_axes=()):
        physical = tuple(slice(0, n - int(a in cell_axes)) for a, n in enumerate(SHAPE))
        result = array.copy()
        result[physical] = np.flip(array[physical], axis=axis)
        return result

    if kind == "wire":
        originals = tuple(rng.random(SHAPE) > 0.5 for _ in range(3))
        mirrored = tuple(flip(a, (c,)) for c, a in enumerate(originals))
        wire = WireSpec(tuple(jnp.asarray(a) for a in originals))
        mirror = WireSpec(tuple(jnp.asarray(a) for a in mirrored))
        _, _, a, _ = extend_cpml_pad_conductors(None, (), (wire,), (), *pads)
        _, _, b, _ = extend_cpml_pad_conductors(None, (), (mirror,), (), *flipped_pads)
        for c in range(3):
            np.testing.assert_array_equal(a[0].edges[c], _expected(originals[c], pads, (c,)))
            np.testing.assert_array_equal(b[0].edges[c], flip(np.asarray(a[0].edges[c]), (c,)))
    else:
        for normal in range(3):
            plane = SHAPE[normal] // 2
            mirrored_plane = SHAPE[normal] - 1 - plane if axis == normal else plane
            mask = np.zeros(SHAPE, dtype=bool)
            selection = [slice(None)] * 3
            selection[normal] = plane
            mask[tuple(selection)] = (rng.random(SHAPE) > 0.5)[tuple(selection)]
            if kind == "sheet":
                sheet = SheetSpec(normal, plane, jnp.asarray(mask))
                mirror = SheetSpec(normal, mirrored_plane, jnp.asarray(flip(mask)))
                _, a, _, _ = extend_cpml_pad_conductors(None, (sheet,), (), (), *pads)
                _, b, _, _ = extend_cpml_pad_conductors(None, (mirror,), (), (), *flipped_pads)
                np.testing.assert_array_equal(a[0].footprint, _expected(mask, pads, normal=normal))
                np.testing.assert_array_equal(b[0].footprint, flip(np.asarray(a[0].footprint)))
                assert b[0].plane == mirrored_plane
            else:
                sigma = mask * rng.uniform(1.0, 7.0, SHAPE).astype(np.float32)
                spec = SheetImpedanceSpec(jnp.asarray(mask), normal, 3.0,
                                          jnp.asarray(sigma), plane)
                mirror = replace(spec, mask=jnp.asarray(flip(mask)),
                                 sigma_sheet=jnp.asarray(flip(sigma)), plane=mirrored_plane)
                _, _, _, a = extend_cpml_pad_conductors(None, (), (), (spec,), *pads)
                _, _, _, b = extend_cpml_pad_conductors(None, (), (), (mirror,), *flipped_pads)
                np.testing.assert_array_equal(a[0].mask, _expected(mask, pads, normal=normal))
                np.testing.assert_array_equal(a[0].sigma_sheet, _expected(sigma, pads, normal=normal))
                np.testing.assert_array_equal(b[0].mask, flip(np.asarray(a[0].mask)))
                np.testing.assert_array_equal(b[0].sigma_sheet, flip(np.asarray(a[0].sigma_sheet)))
                assert b[0].plane == mirrored_plane


@pytest.mark.parametrize("sheet", [False, True])
def test_box_primitive_fractional_realized_face(sheet):
    sim = Simulation(freq_max=1e9, domain=(5.4e-3, 6e-3, 6e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=2)
    grid = sim._build_grid()
    original = Box((0, 2e-3, 2e-3), (5.4e-3, 4e-3, 2e-3 if sheet else 4e-3))
    got = box_at_realized_faces(original, grid, sim._unresolved_domain)
    assert got.corner_hi[0] == 6e-3
    assert original.corner_hi[0] == 5.4e-3
    assert got.corner_lo[2] == original.corner_lo[2]
    assert got.corner_hi[2] == original.corner_hi[2]
    # A deliberate gap remains a gap; the declaration decides attachment.
    gap = replace(original, corner_hi=(5.3e-3, *original.corner_hi[1:]))
    assert box_at_realized_faces(gap, grid, sim._unresolved_domain) is gap
