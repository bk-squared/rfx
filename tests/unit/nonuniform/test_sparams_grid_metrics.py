"""S1 local-cell witnesses; scalar-restoration mutations keep callers intact."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from rfx.boundaries.spec import BoundarySpec
from rfx.nonuniform import make_nonuniform_grid
from rfx.preflight._common import local_cell
from rfx.probes.flux_region import resolve_flux_region
from rfx.sparams._common import (
    _resolve_msl_auto_offsets, _warn_junction_cpml_thickness, _warn_junction_probe_clearance,
    _warn_thin_absorber_vs_guide_wavelength,
)


def _grid():
    # x boundary scalar 1 mm, z-low pad cells 2 mm, interior cells 1 mm.
    return make_nonuniform_grid((.020, .020),
                                np.array([.002] * 4 + [.001] * 12 + [.002] * 4),
                                .001, cpml_layers=8)


def test_absorber_reads_propagation_face_cells():
    grid = _grid()
    assert grid.boundary_cell("z", "lo") == 2 * grid.dx
    assert local_cell(grid, "z", .012) == grid.dx
    cfg = SimpleNamespace(normal_axis="z", f_cutoff=1e9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_thin_absorber_vs_guide_wavelength(
            grid, [cfg], [10e9], 8, BoundarySpec(x="pec", y="pec", z="cpml"))
    # Half lambda_g = 15.065 mm. Actual 16 mm pads clear it; scalar 8 mm do not.
    assert not caught


def test_absorber_reports_only_thin_face_and_its_layer_count():
    grid = make_nonuniform_grid((.020, .020),
                               np.array([.002] * 4 + [.001] * 12 + [.004] * 4),
                               .001, cpml_layers=8)
    cfg = SimpleNamespace(normal_axis="z", f_cutoff=1e9)
    with pytest.warns(UserWarning) as caught:
        _warn_thin_absorber_vs_guide_wavelength(
            grid, [cfg], [5e9], 8, BoundarySpec(x="pec", y="pec", z="cpml"))
    message = str(caught[0].message)
    assert "z-lo 8 cells = 16.0 mm" in message
    assert "z-hi" not in message  # 32 mm clears the 30.6 mm requirement.
    assert "at least 16 (0.75 lambda_g needs 23)" in message


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
@pytest.mark.parametrize("fine_edge", ["both", "lo", "hi"])
def test_auto_offset_uses_both_strip_edges(direction, fine_edge):
    from tests.unit.sparams.test_msl_probe_offset_interval import (
        DX, H_SUB, W_TRACE, _N_W, _W_C, _nu_sim, _pw_box,
    )
    # Keep the width span, but split cells around either/both strip edges.
    profile = np.array([cell for i in range(_N_W) for cell in (
        [DX / 2, DX / 2] if 0 < i < _N_W - 1 and (
            fine_edge == "both" or (fine_edge == "lo" and i < _N_W // 2)
            or (fine_edge == "hi" and i >= _N_W // 2)) else [DX])])
    width_axis = "y" if direction[1] == "x" else "x"
    sim = _nu_sim(direction, **{f"d{width_axis}_profile": profile})
    # A widening of 0.75 boundary cells is a discontinuity at fine
    # resolution, but the scalar-width scan mistakes it for the own trace.
    width = W_TRACE + .75 * DX
    sim.add(_pw_box(direction, .001, .012466, _W_C - width / 2,
                    _W_C + width / 2, H_SUB, H_SUB), material="pec")
    grid = sim._build_nonuniform_grid()
    for side, edge in zip(("lo", "hi"), (_W_C - W_TRACE / 2, _W_C + W_TRACE / 2)):
        ratio = 2 if fine_edge in ("both", side) else 1
        assert local_cell(grid, width_axis, edge) * ratio == grid.boundary_cell(width_axis, "lo")
    with pytest.warns(UserWarning, match="mutually unsatisfiable"):
        resolved, = _resolve_msl_auto_offsets(sim, sim._msl_ports, grid)
    assert resolved.n_probe_offset == 20


def test_graded_flux_face_uses_nodes_not_scalar_indices():
    grid = _grid()
    # z=12 mm is node 8 (four 2 mm cells + four 1 mm cells), plus padding.
    entry = SimpleNamespace(axis="z", coordinate=.012, size=(.008, .008),
                            center=(.010, .010), name="graded")
    record = resolve_flux_region(grid, entry, (.020, .020, .028), warn=False)
    assert record["normal_index"] == grid.pad_z_lo + 8
    assert record["normal_index"] != grid.pad_z_lo + round(entry.coordinate / grid.dx)
    assert record["realized_coordinate_m"] == pytest.approx(.012, abs=1e-15)
    assert record["cell_slices"] == [[14, 22], [14, 22]]


def test_junction_distance_direct_helper_uses_nodes():
    # The public NU waveguide lane rejects port_reference_sims; this is
    # a direct-helper witness, not a claim that that lane is reachable.
    grid = _grid()
    dev = np.zeros(grid.shape)
    ref = dev.copy()
    ref[:, :, grid.pad_z_lo + 4] = 1
    cfg = SimpleNamespace(a=.04, normal_axis="z", probe_x=grid.pad_z_lo)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_junction_probe_clearance(grid, [cfg], dev, [ref], [4.5e9, 6.5e9])
    assert len(caught) == 1
    assert "8.0 mm from the junction" in str(caught[0].message)


def test_junction_cpml_direct_helper_uses_axis_pads():
    # Also unreachable through public NU port_reference_sims today.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_junction_cpml_thickness(
            _grid(), [SimpleNamespace(a=.04, normal_axis="z")], [12e9], 8)
    assert not caught


def test_junction_cpml_direct_helper_reads_realized_face_layers():
    grid = make_nonuniform_grid((.020, .020), np.full(12, .002), .001,
                               cpml_layers=8, face_layers={"z_lo": 4, "z_hi": 8})
    with pytest.warns(UserWarning, match="CPML stack is 8.0 mm"):
        _warn_junction_cpml_thickness(
            grid, [SimpleNamespace(a=.04, normal_axis="z")], [12e9], 8)


def test_junction_nearest_difference_is_measured_in_metres():
    grid = _grid()
    dev = np.zeros(grid.shape)
    ref = dev.copy()
    # At z=8 mm, the left difference is four coarse cells / 8 mm away;
    # the right is five fine cells / 5 mm away and is physically nearer.
    ref[:, :, grid.pad_z_lo] = 1
    ref[:, :, grid.pad_z_lo + 9] = 1
    cfg = SimpleNamespace(a=.04, normal_axis="z", probe_x=grid.pad_z_lo + 4)
    with pytest.warns(UserWarning) as caught:
        _warn_junction_probe_clearance(grid, [cfg], dev, [ref], [4.5e9, 6.5e9])
    assert "5.0 mm from the junction" in str(caught[0].message)


def test_traced_width_keeps_registered_offsets_and_reports_unavailable():
    import jax
    import jax.numpy as jnp
    from tests.unit.sparams.test_msl_probe_offset_interval import DX, _N_W, _nu_sim

    sim = _nu_sim(dy_profile=np.full(_N_W, DX))
    grid = sim._build_nonuniform_grid()

    def solve(width_cells):
        traced = grid._replace(dy_arr=width_cells)
        with pytest.warns(UserWarning, match="y-axis cell sizes are a traced"):
            entry, = _resolve_msl_auto_offsets(sim, sim._msl_ports, traced)
        assert entry.n_probe_offset == sim._msl_ports[0].n_probe_offset
        assert entry.n_probe_spacing == sim._msl_ports[0].n_probe_spacing
        return jnp.sum(width_cells)

    jax.jit(solve)(grid.dy_arr).block_until_ready()
