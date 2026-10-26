"""#2047: thin elements in a modal section are refused by their realized operand."""
import re
import warnings

import numpy as np
import pytest

from rfx import Box, PolylineWire, Simulation


ELEMENTS = (
    ("thin_conductor", r"PEC edge \(volume, sheet, wire or pinned sheet\) \(pec_edge_masks\.[12]\)"),
    ("lumped_rlc", r"lumped RLC element \(lumped_rlc\.ez\)"),
    ("sheet_impedance", r"surface-impedance sheet \(sheet_impedance\.mask_e[yz]\)"),
    ("film", r"film conductivity \(materials\.(components\.cells_view\.)?sigma_film\.[12]\)"),
    ("pec_sheet", r"PEC edge \(volume, sheet, wire or pinned sheet\) \(pec_edge_masks\.[12]\)"),
    ("pec_wire", r"PEC edge \(volume, sheet, wire or pinned sheet\) \(pec_edge_masks\.2\)"),
    ("lumped_port", r"lumped conductivity \(materials\.(components\.(cells_view\.)?)?sigma_lumped\.2\)"),
)


def _guide(element, *, inside=True, graded=False, two_ports=False):
    sim = Simulation(
        freq_max=10e9, domain=(0.12, 0.042, 0.021), dx=0.003,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=10, snap="declared",
        **({"dx_profile": np.array([0.003] * 12 + [0.0027, 0.0033] * 4 + [0.003] * 20)} if graded else {}))
    sim.add_waveguide_port(0.090, direction="-x", name="right", mode=(1, 0),
                          mode_type="TE", freqs=np.array([5e9, 6.5e9]),
                          f0=6e9, bandwidth=0.5, reference_plane=0.075)
    if two_ports:
        sim.add_waveguide_port(0.012, direction="+x", name="left", mode=(1, 0),
                              mode_type="TE", freqs=np.array([5e9, 6.5e9]),
                              f0=6e9, bandwidth=0.5, reference_plane=0.021)
    x = 0.084 if inside else 0.048
    sheet = Box((x, 0.006, 0.003), (x, 0.015, 0.009))
    if element == "thin_conductor":
        sim.add_thin_conductor(sheet, sigma_bulk=5.8e7)
    elif element == "sheet_impedance":
        sim.add_thin_conductor(sheet, sigma_bulk=1e3, thickness=1e-4,
                               surface_impedance_f0=6e9)
    elif element == "film":
        sim.add_thin_conductor(sheet, sigma_bulk=1e3, thickness=1e-5)
    elif element == "pec_sheet":
        sim.add(Box((x, 0, 0), (x, 0.042, 0.021)), material="pec")
    elif element == "pec_wire":
        sim.add(PolylineWire(((x, 0.009, 0.003), (x, 0.009, 0.009)), radius=0.), material="pec")
    elif element == "lumped_rlc":
        sim.add_lumped_rlc((x, 0.009, 0.006), "ez", R=10., L=1e-9)
    elif element == "lumped_port":
        sim.add_port((x, 0.009, 0.006), "ez", excite=False)
    return sim


def _execute(sim, entry="run"):
    # run() already refuses every mixed lumped/modal model, wherever the
    # lumped port is drawn. forward() reaches the shared section authority.
    if entry == "run" and sim._ports:
        entry = "forward"
    kwargs = {"n_steps": 2, "skip_preflight": True}
    if entry == "run":
        kwargs["compute_s_params"] = False
    if entry == "compute_waveguide_s_matrix":
        kwargs["normalize"] = "flux" if sim._uses_nonuniform_mesh else False
    return getattr(sim, entry)(**kwargs)


@pytest.mark.parametrize("element,label", ELEMENTS, ids=[row[0] for row in ELEMENTS])
def test_inside_refused(element, label):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _guide(element)
        with pytest.raises(ValueError, match=(
                r"Waveguide port 'right': section cells .*: " + label
                + r" at cell \(\d+, \d+, \d+\); the port has no model for a thin element "
                r"in its section \(rfx#2047\); draw it beyond the reference plane\.")):
            _execute(sim)


@pytest.mark.parametrize("element,label", ELEMENTS, ids=[row[0] for row in ELEMENTS])
def test_beyond_reference_admitted(element, label):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _execute(_guide(element, inside=False))


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("entry", ["forward", "compute_waveguide_s_matrix"])
def test_other_entries_refuse(graded, entry):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _guide("thin_conductor", graded=graded, two_ports=entry == "compute_waveguide_s_matrix")
        with pytest.raises(ValueError, match=(
                r"Waveguide port 'right': section cells .*: PEC edge "
                r"\(volume, sheet, wire or pinned sheet\) \(pec_edge_masks\.[12]\) "
                r"at cell .*rfx#2047")):
            _execute(sim, entry)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_beyond_sheet_preflight_and_calculator_admitted(graded):
    """A sheet past the reference plane must survive both material collectors."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        sim = _guide("thin_conductor", inside=False, graded=graded, two_ports=True)
        report = sim.preflight()
        assert report.ok, str(report)
        collector_error = "pec_sheets/pec_wires collector"
        assert not [str(w.message) for w in recorded if collector_error in str(w.message)]
        assert not [str(issue) for issue in report if collector_error in str(issue)]
        if not graded:
            report = sim.preflight_sparameters(calculator="waveguide")
            assert report.ok, str(report)
            assert not [str(w.message) for w in recorded if collector_error in str(w.message)]
            assert not [str(issue) for issue in report if collector_error in str(issue)]
        _execute(sim, "compute_waveguide_s_matrix")
        assert not [str(w.message) for w in recorded if collector_error in str(w.message)]


@pytest.mark.parametrize("direction", ["+x", "-x"])
@pytest.mark.parametrize("scene", ["a", "b", "c", "d"])
def test_b4_section_boundary_nodes(direction, scene):
    """Reference faces refuse; faces one cell behind the source admit (B4)."""
    a, b = 0.042, 0.021
    positions = {
        "+x": {"a": (0.045, 0.045), "b": (0.045, 0.054),
               "c": (0.015, 0.027), "d": (0.027, 0.027)},
        "-x": {"a": (0.075, 0.075), "b": (0.066, 0.075),
               "c": (0.093, 0.105), "d": (0.093, 0.093)},
    }
    source, reference, name = ((0.030, 0.045, "left") if direction == "+x"
                               else (0.090, 0.075, "right"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(0.12, a, b), dx=0.003,
                         boundary={"x": "cpml", "y": "pec", "z": "pec"},
                         cpml_layers=10, snap="declared")
        sim.add_waveguide_port(source, direction=direction, name=name, mode=(1, 0),
                              mode_type="TE", freqs=np.array([5e9, 6.5e9]),
                              f0=6e9, bandwidth=0.5, reference_plane=reference)
        lo, hi = positions[direction][scene]
        sim.add(Box((lo, 0, 0), (hi, a, b)), material="pec")
        if scene in ("a", "b"):
            with pytest.raises(ValueError, match=re.escape(
                    "PEC edge (volume, sheet, wire or pinned sheet) (pec_edge_masks.1)")) as exc:
                sim.forward(n_steps=2, skip_preflight=True)
            assert re.search(re.escape(f"Waveguide port '{name}': section cells "), str(exc.value))
            assert re.search(re.escape(
                "the port has no model for a thin element in its section (rfx#2047); "
                "draw it beyond the reference plane."), str(exc.value))
        else:
            sim.forward(n_steps=2, skip_preflight=True)


def test_pec_box_aperture_walls_admitted():
    """Bounding-node tangential edges are guide walls, not interior sheets (m2)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(0.12, 0.054, 0.021), dx=0.003,
                         boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=10)
        sim.add(Box((0, 0, 0), (0.12, 0.006, 0.021)), material="pec")
        sim.add(Box((0, 0.048, 0), (0.12, 0.054, 0.021)), material="pec")
        sim.add_waveguide_port(0.090, direction="-x", name="right", y_range=(0.006, 0.048),
                              f0=6e9, freqs=np.array([5e9, 6.5e9]), reference_plane=0.075)
        _execute(sim)



def _buried_port_guide(graded):
    """A straight guide whose third port lies inside a solid PEC block.

    The compact T-junction's horizontal reference sim has this shape: its top
    port's section and reference node are interior to the block."""
    profile = np.array([0.002] * 20 + [0.0018, 0.0022] * 10 + [0.002] * 20)
    sim = Simulation(freq_max=10e9, domain=(0.12, 0.12, 0.02), boundary="cpml",
                     cpml_layers=10, dx=0.002, **({"dx_profile": profile} if graded else {}))
    sim.add(Box((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)), material="pec")
    sim.add(Box((0.0, 0.08, 0.0), (0.12, 0.12, 0.02)), material="pec")
    common = dict(mode=(1, 0), mode_type="TE", freqs=np.array([5e9, 6.5e9]), f0=6e9,
                  ref_offset=3, probe_offset=15, z_range=(0.0, 0.02))
    sim.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x", name="left", **common)
    sim.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x", name="right", **common)
    sim.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y", name="top", **common)
    return sim


@pytest.mark.parametrize("traced", [False, True], ids=["plain", "traced"])
def test_interior_conductor_edges_admitted_on_the_reference_path(traced):
    """An edge closed by PEC on every neighbouring cell is the conductor, not a
    sheet (m5, m6). The reference-sim path carries realized PEC edges and no
    cell mask (rfx/sparams/waveguide.py port_reference_sims); the section check
    reads the same edges plain or traced (rfx-core-desk L:0bbcf4) and admits a
    port section inside the block. A block face on the reference plane stays
    refused (B4 scene b). Direct entries refuse this scene earlier (#1994)."""
    import jax
    import jax.numpy as jnp
    from rfx.boundaries.features import resolve_waveguide_section_medium
    from rfx.boundaries.pec import realized_pec_edge_masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _buried_port_guide(False)
        grid = sim._build_grid()
        materials, _, _, pec, _, _, _ = sim._assemble_materials(grid)
        edges = tuple(jnp.asarray(m) for m in realized_pec_edge_masks(pec, periodic=sim._periodic_flags()))
        freqs = jnp.asarray([5e9, 6.5e9])
        cfgs = [sim._build_waveguide_port_config(entry, grid, freqs, 20)
                for entry in sim._waveguide_ports]

        def resolve(masks):
            resolve_waveguide_section_medium(cfgs, grid, materials, dict(pec_edge_masks=masks), ())
            return jnp.zeros(())

        if traced:
            jax.jit(resolve)(edges).block_until_ready()
        else:
            resolve(edges)
