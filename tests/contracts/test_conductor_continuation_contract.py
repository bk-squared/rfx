"""Absorber columns must contain the adjacent solved cross-section."""
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

# Importable on its own, not only after a sibling module has put tests/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _example_fidelity_lib as lib  # noqa: E402
from _conductor_continuation_contract import assembled_arrays, violations  # noqa: E402


CASES = [(path, builder, variant)
         for path, entry in sorted(lib.CLASSIFICATION.items()) if entry.kind == "audited"
         for builder in entry.builders for variant in builder.variants]

# Entry indices in the committed drawings, not a port-classification algorithm.
HELD_FACES = {
    "validation/crossval/06b_msl_notch_filter_uniform.py": [(1, "x-lo"), (1, "x-hi")],
    "validation/crossval/07_sheen_lpf.py": [(1, "x-lo"), (3, "x-hi")],
    "validation/tmtt_paper/msl_stub_notch_tuning.py": [(1, "x-lo"), (1, "x-hi")],
}


def _assert_empty_sheet_faces(grid, arrays, rows):
    for sheet_index, face in rows:
        axis = "xyz".index(face[0])
        pad = getattr(grid, "pad_"+face.replace("-", "_"))
        assert pad > 0
        footprint = arrays[f"sheet_{sheet_index}"]
        layers = range(pad) if face.endswith("lo") else range(grid.shape[axis]-pad, grid.shape[axis])
        assert not np.take(footprint, list(layers), axis=axis).any(), (sheet_index, face)


@pytest.mark.parametrize("path,builder,variant", CASES,
                         ids=[f"{p}:{b.fn}:{v.label}" for p, b, v in CASES])
def test_absorbing_columns_equal_the_face(path, builder, variant):
    with lib.build_only():
        module = lib.load_module(path)
        result = getattr(module, builder.fn)(**variant.kwargs(module))
        sim = result if builder.result_index is None else result[builder.result_index]
        try:
            grid, arrays, poles, nodes = assembled_arrays(sim)
            rows = HELD_FACES.get(path, ())
            bad, _ = violations(sim, grid, arrays, poles, nodes, held_faces=rows)
            assert not bad, bad
            # These three drawings put the dielectric first, then PEC sheets.
            _assert_empty_sheet_faces(grid, arrays, [(i-1, face) for i, face in rows])
        finally:
            jax.clear_caches()


def test_reached_conductor_is_a_live_array_witness():
    from rfx import Box, Simulation
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2)
    sim.add(Box((0., 2., 2.), (8., 6., 6.)), material="pec")
    grid, arrays, poles, nodes = assembled_arrays(sim)
    bad, exceptions = violations(sim, grid, arrays, poles, nodes)
    assert not exceptions
    assert not bad, bad
    assert arrays["pec_mask"][:2].any()
    assert arrays["fenced_y"][:2].any()
    assert (arrays["tensor_y"][:2] == 0.).any()
    # A single removed pad cell is visible without re-running any helper.
    arrays["pec_mask"] = arrays["pec_mask"].copy()
    arrays["pec_mask"][0, 4, 4] = False
    bad, _ = violations(sim, grid, arrays, poles, nodes)
    assert any(item["array"] == "pec_mask" and item["axis"] == 0
               and item["side"] == 0 and item["differences"] == 1 for item in bad)


def test_occupied_patch_ground_continues_on_both_high_faces():
    module = lib.load_module("tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py")
    sim = module._build(n=2, pad_h=10, cpml=4)
    try:
        # This oracle calls run() with subpixel_smoothing=False. Its actual
        # step receives staircase materials and PEC edges, no tensor.
        grid, arrays, poles, nodes = assembled_arrays(sim, include_smoothed=False)
        bad, exceptions = violations(sim, grid, arrays, poles, nodes)
        assert not exceptions
        assert not bad, bad
        for axis in (0, 1):
            assert arrays["pec_mask"].take(-2, axis=axis).any()
    finally:
        jax.clear_caches()


def test_port_exception_is_only_the_terminal_entry():
    import numpy as np
    from rfx import Box, Simulation
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2, pec_faces={"z_lo"})
    sim.add(Box((0., 2., 3.), (8., 3., 3.)), material="pec")
    sim.add_port((4., 2., 0.), component="ez", extent=3.)
    sim.add(Box((.1, 5., 3.), (7.9, 7., 5.)), material="pec")
    grid, arrays, poles, nodes = assembled_arrays(sim)
    rows = [(0, "x-lo"), (0, "x-hi")]
    bad, exceptions = violations(sim, grid, arrays, poles, nodes, held_faces=rows)
    assert not bad, bad
    assert {item[2] for item in exceptions} == {"port-terminal:pec"}
    assert arrays["sheet_0"][2].any()
    assert not arrays["sheet_0"][:2].any()
    assert arrays["pec_mask"][:2].any()
    # Corrupt the other entry outside the terminal sheet's transverse support.
    arrays["pec_mask"] = arrays["pec_mask"].copy()
    occupied = np.argwhere(arrays["pec_mask"][0])
    j, k = occupied[0]
    arrays["pec_mask"][0, j, k] = False
    _assert_empty_sheet_faces(grid, arrays, rows)
    bad, _ = violations(sim, grid, arrays, poles, nodes, held_faces=rows)
    assert any(b["array"] == "pec_mask" and b["differences"] == 1 for b in bad)


def fed_patch(kind="wire"):
    from rfx import Box
    module = lib.load_module("tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py")
    sim = module._build(n=2, pad_h=10, cpml=4)
    ground, patch = sim._geometry[0].shape, sim._geometry[2].shape
    x, y = [(lo+hi)/2 for lo, hi in zip(patch.corner_lo[:2], patch.corner_hi[:2])]
    z = ground.corner_hi[2]
    top = patch.corner_lo[2]
    if kind == "thin":
        sim._geometry.pop(2)
        sim.add_thin_conductor(Box((*patch.corner_lo[:2], top),
                                   (*patch.corner_hi[:2], top)),
                               sigma_bulk=5.8e7, thickness=1e-6)
    if kind == "nu":
        sim._dz_profile = np.full(32, module.H/2)
    if kind == "coax":
        sim.add_coaxial_port((x, y, z), face="bottom", pin_length=top-z)
    elif kind == "lumped":
        # The one-cell lump touches the ground; the two-cell substrate and
        # patch stay as declared in the oracle.
        sim.add_port((x, y, z), component="ez")
    else:
        sim.add_port((x, y, z), component="ez", extent=top-z)
    return sim


@pytest.mark.parametrize("kind", ["wire", "lumped", "coax", "nu", "thin"])
def test_fed_patch_ground_fills_all_lateral_absorbers(kind):
    sim = fed_patch(kind)
    try:
        grid, arrays, poles, nodes = assembled_arrays(sim, include_smoothed=False)
        # Literal cell layer z=5h .. 5h+dx in the unchanged n=2 drawing.
        k = grid.pad_z_lo + 10
        cells = arrays["pec_mask"]
        assert cells[:-1, :-1, k].all()
        bad, exceptions = violations(sim, grid, arrays, poles, nodes)
        assert not exceptions
        assert not bad, bad
    finally:
        jax.clear_caches()


def test_wire_fed_line_strip_empty_and_ground_full():
    from rfx import Box, Simulation
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2)
    sim.add(Box((0., 0., 1.), (8., 8., 1.)), material="pec")
    sim.add(Box((0., 3., 4.), (8., 5., 4.)), material="pec")
    sim.add_port((2., 4., 1.), component="ez", extent=3.)
    sim.add_port((6., 4., 1.), component="ez", extent=3.)
    grid, arrays, poles, nodes = assembled_arrays(sim)
    assert arrays["sheet_0"][:, :, 3].all()
    rows = [(1, "x-lo"), (1, "x-hi")]
    _assert_empty_sheet_faces(grid, arrays, rows)
    bad, _ = violations(sim, grid, arrays, poles, nodes, held_faces=rows)
    assert not bad, bad


@pytest.mark.parametrize("kind", ("partial_reference", "equal", "wide_signal", "dangling_lumped"))
def test_port_pair_holds_signal_and_no_wider_shared_reference(kind):
    from rfx import Box, Simulation
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2)
    if kind == "equal":
        sim.add(Box((0., 1., 4.), (8., 2., 4.)), material="pec")
        sim.add(Box((0., 5., 4.), (8., 6., 4.)), material="pec")
        sim.add_port((4., 2., 4.), component="ey", extent=3.)
    else:
        reference = (Box((0., 0., 1.), (4., 8., 1.)) if kind == "partial_reference"
                     else Box((0., 3., 1.), (8., 5., 1.)) if kind == "wide_signal"
                     else Box((0., 0., 1.), (8., 8., 1.)))
        signal = (Box((0., 1., 4.), (8., 7., 4.)) if kind == "wide_signal"
                  else Box((0., 3., 4.), (8., 5., 4.)))
        sim.add(reference, material="pec")
        sim.add(signal, material="pec")
        sim.add_port((2., 4., 3. if kind == "dangling_lumped" else 1.), component="ez",
                     **({} if kind == "dangling_lumped" else {"extent": 3.}))
    grid, arrays, poles, nodes = assembled_arrays(sim)
    rows = [(1, "x-lo"), (1, "x-hi")]
    if kind in ("equal", "wide_signal"):
        rows += [(0, "x-lo"), (0, "x-hi")]
    _assert_empty_sheet_faces(grid, arrays, rows)
    assert arrays["sheet_1"][2].any() and arrays["sheet_1"][-3].any()
    if kind in ("partial_reference", "dangling_lumped"):
        assert arrays["sheet_0"][0, :, 3].all()
    bad, _ = violations(sim, grid, arrays, poles, nodes, held_faces=rows)
    assert not bad, bad


@pytest.mark.parametrize("offset", (0., .5, 1.))
def test_pec_floor_post_stops_at_the_high_absorber(offset):
    from rfx import Box, Simulation
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2, pec_faces={"z_lo"})
    sim.add(Box((3., 3., 2.), (5., 5., 8.)), material="pec")
    sim.add_port((4., 4., offset), component="ez", extent=2.-offset)
    _, arrays, _, _ = assembled_arrays(sim, include_smoothed=False)
    assert arrays["pec_mask"][5:7, 5:7, 7].all()
    assert not arrays["pec_mask"][:, :, 8:10].any()
