"""Physical attachment of MSL ports, before source setup can clear metal."""
from dataclasses import replace

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.msl_port import (
    msl_cross_section_span, msl_physical_point, msl_port_from_entry,
    validate_msl_port_geometry,
)

DX = 1e-3


def _model(*, direction="+x", nonuniform=False, ground=6., top=10.,
           ground_thickness=0., trace_thickness=0., ground_kind="pec",
           trace_kind="pec", domain_ground=None, trace_width=(10., 14.),
           trace_length=None, port_top=None, port_ground=None):
    profiles = (np.full(24, DX), np.full(24, 1.2*DX),
                np.array([DX]*4+[.5*DX]*8+[1.5*DX]*12))
    nodes = tuple(np.r_[0., np.cumsum(a)] for a in profiles) if nonuniform else None
    def coord(axis, index):
        return (float(np.interp(index, np.arange(25), nodes[axis]))
                if nonuniform else index*DX)
    prop = 0 if direction.endswith("x") else 1
    width = 1-prop
    domain = tuple(coord(a, 24) for a in range(3))
    bc = BoundarySpec(x="cpml", y="cpml", z="cpml")
    if domain_ground is not None:
        bc = BoundarySpec(x="cpml", y="cpml", z=("periodic" if domain_ground == "periodic"
                          else Boundary(lo=domain_ground, hi="cpml")))
    kwargs = dict(dx_profile=profiles[0], dy_profile=profiles[1], dz_profile=profiles[2]) if nonuniform else {}
    sim = Simulation(freq_max=5e9, domain=domain, dx=DX, cpml_layers=0, boundary=bc, **kwargs)
    lo, hi = coord(2, ground), coord(2, top)

    def conductor(shape, kind):
        if kind == "pec":
            sim.add(shape, material="pec")
        elif kind == "f0":
            sim.add_thin_conductor(shape, sigma_bulk=5.8e7, surface_impedance_f0=5e9)

    conductor(Box((0., 0., coord(2, ground-ground_thickness)),
                   (domain[0], domain[1], lo)), ground_kind)
    p_lo, p_hi = (0., 24.) if trace_length is None else trace_length
    conductor(Box(msl_physical_point(direction, coord(prop, p_lo), coord(width, trace_width[0]), hi),
                   msl_physical_point(direction, coord(prop, p_hi), coord(width, trace_width[1]),
                                      coord(2, top+trace_thickness))), trace_kind)
    sim.add_material("substrate", eps_r=3.)
    sim.add(Box((0., 0., lo), (domain[0], domain[1], hi)), material="substrate")
    sim.add_msl_port(position=msl_physical_point(direction, coord(prop, 8), coord(width, 12),
                                                lo if port_ground is None else coord(2, port_ground)),
                     width=coord(width, 14)-coord(width, 10),
                     height=(hi if port_top is None else coord(2, port_top))
                            -(lo if port_ground is None else coord(2, port_ground)),
                     direction=direction, name="checked")
    return sim, nonuniform


def _geometry(sim, nonuniform):
    grid = sim._build_nonuniform_grid() if nonuniform else sim._build_grid()
    sheets, wires, lossy = [], [], []
    assemble = sim._assemble_materials_nu if nonuniform else sim._assemble_materials
    result = assemble(grid, pec_sheets=sheets, pec_wires=wires, sheet_specs=lossy)
    hard = None
    if result[3] is not None or sheets or wires:
        hard = realized_pec_edge_masks(result[3], sheets, wires, periodic=sim._periodic_flags())
    return grid, msl_port_from_entry(sim._msl_ports[0]), hard, lossy


def _validate(model, **overrides):
    sim, nu = model
    grid, port, hard, lossy = _geometry(sim, nu)
    validate_msl_port_geometry(grid, replace(port, **overrides), pec_edge_masks=hard,
                               sheet_specs=lossy, pec_faces=sim._boundary_spec.pec_faces(),
                               periodic=sim._periodic_flags(), name="checked")
    return grid, port, hard, lossy


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
@pytest.mark.parametrize("nu", [False, True])
def test_offset_surface_attachment_uses_actual_axis_coordinates(direction, nu):
    _validate(_model(direction=direction, nonuniform=nu))


@pytest.mark.parametrize("ground_t,trace_t", [(2., 0.), (0., 2.), (2., 2.)])
def test_volume_surfaces_face_the_open_substrate(ground_t, trace_t):
    model = _model(ground_thickness=ground_t, trace_thickness=trace_t)
    _validate(model)
    if ground_t:
        with pytest.raises(ValueError, match="intersects PEC"):
            _validate(model, z_lo=5*DX)
    if trace_t:
        with pytest.raises(ValueError, match="intersects PEC|additional conductor"):
            _validate(model, z_hi=11*DX)


@pytest.mark.parametrize("nu", [False, True])
def test_half_cell_tie_uses_the_same_lower_plane_as_the_sheet(nu):
    grid, port, _, _ = _validate(_model(nonuniform=nu, ground=5.5, top=9.5))
    span = msl_cross_section_span(grid, port)
    assert (span["n_lo"], span["n_hi"]) == (5, 9)


def test_small_plane_rounding_is_allowed_but_a_different_node_is_not():
    model = _model()
    _validate(model, z_lo=6.2*DX, z_hi=9.8*DX)
    with pytest.raises(ValueError, match="declared trace"):
        _validate(model, z_hi=11*DX)


@pytest.mark.parametrize("kind,passes", [("pec", True), ("pmc", False),
                                         ("periodic", False), ("cpml", False)])
def test_zero_padding_does_not_imply_a_domain_pec_ground(kind, passes):
    model = _model(ground=0., top=4., ground_kind="none", domain_ground=kind)
    assert _geometry(*model)[0].pad_z_lo == 0
    if passes:
        _validate(model)
    else:
        with pytest.raises(ValueError, match="declared ground"):
            _validate(model)


@pytest.mark.parametrize("nu", [False, True])
@pytest.mark.parametrize("ground_kind,trace_kind", [("f0", "pec"), ("pec", "f0"), ("f0", "f0")])
def test_lossy_sheet_geometry_is_observed_without_becoming_pec(nu, ground_kind, trace_kind):
    model = _model(nonuniform=nu, ground_kind=ground_kind, trace_kind=trace_kind)
    grid, port, hard, specs = _geometry(*model)
    before = None if hard is None else tuple(np.asarray(m).copy() for m in hard)
    validate_msl_port_geometry(grid, port, pec_edge_masks=hard, sheet_specs=specs)
    if before is not None:
        for a, b in zip(before, hard):
            np.testing.assert_array_equal(a, b)
    else:
        assert hard is None


def test_a_transverse_crossbar_is_not_a_longitudinal_trace():
    with pytest.raises(ValueError, match="longitudinal conductor"):
        _validate(_model(trace_length=(7.8, 8.2)))


def test_partial_trace_coverage_is_not_hidden_by_the_center_column():
    with pytest.raises(ValueError, match="width node"):
        _validate(_model(trace_width=(11., 13.)))


def test_another_plane_cannot_be_selected_as_a_replacement_trace():
    model = _model(port_top=11.)
    sim, _ = model
    sim.add(Box((0, 0, 15*DX), (24*DX, 24*DX, 15*DX)), material="pec")
    with pytest.raises(ValueError, match="declared trace"):
        _validate(model)


def test_a_port_cannot_jump_over_an_intermediate_conductor():
    model = _model()
    sim, _ = model
    sim.add(Box((0, 0, 8*DX), (24*DX, 24*DX, 8*DX)), material="pec")
    with pytest.raises(ValueError, match="additional conductor"):
        _validate(model)


@pytest.mark.parametrize("feed_offset,accepted", [(0., False), (2., True)])
def test_vertical_lossy_sheet_cannot_load_the_port_source(feed_offset, accepted):
    model = _model()
    sim, _ = model
    x = (8.+feed_offset)*DX
    sim.add_thin_conductor(Box((x, 11.8*DX, 7*DX), (x, 12.2*DX, 9*DX)),
                           sigma_bulk=5.8e7, surface_impedance_f0=5e9)
    if accepted:
        _validate(model)
    else:
        with pytest.raises(ValueError, match="surface-impedance sheet edges"):
            _validate(model)


def test_normal_coordinates_are_not_silently_clamped_to_the_grid():
    with pytest.raises(ValueError, match="outside.*grid"):
        _validate(_model(), z_lo=-2*DX)


@pytest.mark.parametrize("nu,method", [(False, "run"), (False, "forward"), (True, "run")])
def test_skip_preflight_cannot_bypass_attachment_validation(nu, method):
    sim, _ = _model(nonuniform=nu, port_top=11.)
    with pytest.raises(ValueError, match="declared trace"):
        getattr(sim, method)(n_steps=1, skip_preflight=True)


def test_preflight_reports_a_blocking_plane_mismatch():
    sim, _ = _model(port_top=11.)
    findings = sim.preflight()
    assert any(getattr(x, "code", None) == "msl_port_conductor_planes"
               and x.severity == "error" for x in findings)
    assert not findings.ok


@pytest.mark.parametrize("trace_kind", ["sphere", "sheet", "sheet_with_sphere_away"])
def test_run_checks_kottke_surfaces_after_fractional_edges_are_released(trace_kind, monkeypatch):
    """Exercise actual assembly and runner setup, stopping before any FDTD."""
    import rfx.simulation as stepping
    from rfx import Sphere
    sim, _ = _model(trace_kind="none" if trace_kind == "sphere" else "pec")
    sim._msl_ports[0] = replace(sim._msl_ports[0], width=2*DX)
    if trace_kind != "sheet":
        x = 8. if trace_kind == "sphere" else 18.
        sim.add(Sphere(center=(x*DX, 12*DX, 12*DX), radius=2.1*DX), material="pec")
    # The staircase surface would certify the sphere at the declared trace.
    _validate((sim, False))

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        assert kwargs["aniso_inv_eps"] is not None
        raise Captured

    monkeypatch.setattr(stepping, "run", capture)
    if trace_kind == "sphere":
        with pytest.raises(ValueError, match="no longitudinal conductor edge"):
            sim.run(n_steps=1, subpixel_smoothing="kottke_pec", skip_preflight=True)
    else:
        with pytest.raises(Captured):
            sim.run(n_steps=1, subpixel_smoothing="kottke_pec", skip_preflight=True)


def test_direct_transition_cannot_silently_drop_lossy_sheets():
    sim, _ = _model(trace_kind="f0")
    with pytest.raises(ValueError, match="not supported on the coax-MSL transition lane") as caught:
        sim.compute_coax_msl_transition(junction_x=8*DX, n_steps=1)
    assert any(frame.name == "compute_coax_msl_transition" for frame in caught.traceback)


@pytest.mark.parametrize("direction", ["+x", "+y"])
@pytest.mark.parametrize("width_node", [10, 8], ids=["trace", "fringe"])
def test_forward_reserves_every_cell_that_owns_the_source_edge(direction, width_node, monkeypatch):
    import jax.numpy as jnp
    import rfx.simulation as stepping
    from rfx.sources.msl_port import msl_cell
    sim, _ = _model(direction=direction)
    grid = sim._build_grid()
    edge = msl_cell(direction, 8, width_node, 8)
    diagonal_owner = (edge[0]-1, edge[1]-1, edge[2])
    remote = (2, 2, 2)
    occupancy = jnp.zeros(grid.shape).at[diagonal_owner].set(1.).at[remote].set(.4)
    # Independent observable: the original density freezes the source Ez.
    assert realized_pec_edge_masks(occupancy > .5)[2][edge]

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        driven = [s for s in kwargs["sources"] if (s.i, s.j, s.k) == edge]
        assert len(driven) == 1 and np.max(np.abs(driven[0].waveform)) > 0.
        actual = kwargs["pec_occupancy"]
        assert actual[diagonal_owner] == 0.
        assert actual[remote] == pytest.approx(.4)
        assert not realized_pec_edge_masks(actual > .5)[2][edge]
        raise Captured

    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "0")
    monkeypatch.setattr(stepping, "run", capture)
    with pytest.raises(Captured):
        sim.forward(n_steps=1, pec_occupancy_override=occupancy, skip_preflight=True)


def test_forward_density_reservation_preserves_derivatives_outside_the_port(monkeypatch):
    import jax
    import jax.numpy as jnp
    import rfx.simulation as stepping
    sim, _ = _model()
    grid = sim._build_grid()
    diagonal, remote = (7, 7, 8), (2, 2, 2)  # fringe source's diagonal owner

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        density = kwargs["pec_occupancy"]
        raise Captured(density[diagonal]**2+3*density[remote])

    def observe(values):
        density = jnp.zeros(grid.shape).at[diagonal].set(values[0]).at[remote].set(values[1])
        try:
            sim.forward(n_steps=1, pec_occupancy_override=density, skip_preflight=True)
        except Captured as result:
            return result.args[0]
        raise AssertionError("forward never reached the stepper")

    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "0")
    monkeypatch.setattr(stepping, "run", capture)
    value, derivative = jax.value_and_grad(observe)(jnp.array([.7, .4]))
    assert value == pytest.approx(1.2)
    np.testing.assert_array_equal(derivative, [0., 3.])


def test_kottke_density_guard_covers_the_outer_fringe_neighbour(monkeypatch):
    import jax.numpy as jnp
    import rfx.geometry.smoothing as smoothing
    import rfx.simulation as stepping
    sim, _ = _model()
    grid = sim._build_grid()
    # Four cells of lateral Laplace padding: trace ends at w=14,
    # last fringe source at w=18. Its positive w=19 neighbour is NOT
    # among the four incident owners, but Kottke's dilation reads it.
    edge, neighbour, remote = (8, 18, 8), (8, 19, 8), (2, 2, 2)
    density = jnp.zeros(grid.shape).at[neighbour].set(1.).at[remote].set(.4)
    original = smoothing.kottke_inv_eps_from_occupancy
    seen = []

    def observe(grid_arg, actual, **kwargs):
        assert actual[neighbour] == 0.
        assert actual[remote] == pytest.approx(.4)
        seen.append(True)
        return original(grid_arg, actual, **kwargs)

    class Captured(Exception):
        pass

    def capture(grid_arg, materials, *args, **kwargs):
        assert kwargs["pec_occupancy"] is None
        assert kwargs["aniso_inv_eps"][2][edge] > .99/materials.eps_r[edge]
        driven = [s for s in kwargs["sources"] if (s.i, s.j, s.k) == edge]
        assert len(driven) == 1 and np.max(np.abs(driven[0].waveform)) > 0.
        raise Captured

    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "1")
    monkeypatch.setattr(smoothing, "kottke_inv_eps_from_occupancy", observe)
    monkeypatch.setattr(stepping, "run", capture)
    with pytest.raises(Captured):
        sim.forward(n_steps=1, pec_occupancy_override=density, skip_preflight=True)
    assert seen == [True]


def test_a_narrow_trace_uses_the_registered_center_not_the_rounded_endpoints():
    model = _model(trace_width=(10.4, 11.4))
    grid, port, _, _ = _validate(model, y_lo=10.4*DX, y_hi=11.4*DX)
    port = replace(port, y_lo=10.4*DX, y_hi=11.4*DX)
    # The only realized trace row is node 11. Averaging rounded endpoints
    # 10 and 11 chose node 10, outside the conductor, in the old reader.
    assert msl_cross_section_span(grid, port)["w_centre"] == 11
    from rfx.sources.msl_port import compute_msl_mode_profile
    profile = compute_msl_mode_profile(grid, port, 3.)
    row = 11-profile["j_grid_lo"]
    assert np.sum(profile["ez_profile"][row])*DX == pytest.approx(1., rel=1e-12)


def test_a_geometry_addition_invalidates_an_earlier_attachment_check():
    model = _model()
    _validate(model)
    sim, _ = model
    sim.add(Box((0, 0, 8*DX), (24*DX, 24*DX, 8*DX)), material="pec")
    report = sim.preflight()
    assert any(getattr(x, "code", None) == "msl_port_conductor_planes"
               and x.severity == "error" for x in report)
    with pytest.raises(ValueError, match="additional conductor"):
        sim.run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize("lane", ["mixed", "coax"])
def test_transition_readers_use_the_same_center_and_bounding_plane(lane, monkeypatch):
    """Capture the real trace lookup; no field solve or calibration claim."""
    import rfx.probes.msl_wave_decomp as decomposition

    sim = Simulation(freq_max=5e9, domain=(32*DX, 32*DX, 48*DX),
                     dx=DX, cpml_layers=4, boundary="cpml")
    sim.add_material("substrate", eps_r=3.)
    sim.add(Box((0, 0, 16*DX), (32*DX, 32*DX, 20*DX)), material="substrate")
    sim.add(Box((0, 0, 16*DX), (32*DX, 32*DX, 16*DX)), material="pec")
    sim.add(Box((10*DX, 12.4*DX, 20*DX), (32*DX, 13.4*DX, 20*DX)), material="pec")
    sim.add_msl_port(position=(26*DX, 12.9*DX, 16*DX), width=DX,
                     height=4*DX, direction="-x", eps_r_sub=3.,
                     n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    if lane == "coax":
        sim.add_coaxial_port(position=(10*DX, 12.9*DX, 16*DX), face="bottom",
                             pin_radius=DX, outer_radius=5*DX)
    else:
        sim.add_port(position=(5*DX, 12.9*DX, 18*DX), component="ez", impedance=50.)
    grid = sim._build_grid()
    original = decomposition.realized_trace_planes_on_column
    seen = []

    class Captured(Exception):
        pass

    def capture(edges, axis, ij, top, **kwargs):
        assert axis == 2
        assert tuple(ij) == (grid.pad_x_lo+26, grid.pad_y_lo+13)
        assert top == grid.pad_z_lo+20
        found = original(edges, axis, ij, top, **kwargs)
        assert found == (top, top)
        seen.append(found)
        raise Captured

    monkeypatch.setattr(decomposition, "realized_trace_planes_on_column", capture)
    with pytest.raises(Captured):
        if lane == "coax":
            sim.compute_coax_msl_transition(junction_x=10*DX, n_steps=1,
                                            n_freqs=2, probe_count=3,
                                            probe_start_cells=3, probe_spacing_cells=2,
                                            skip_preflight=True)
        else:
            sim.compute_mixed_s_matrix(n_steps=1, n_freqs=2)
    assert len(seen) == 1


@pytest.mark.parametrize("ground", [15., 15.5])
def test_coax_stub_cannot_overwrite_the_registered_junction_and_substrate(ground, monkeypatch):
    """Capture material assembly, without asserting transition calibration."""
    import rfx.sources.msl_port as sources
    from rfx.sources.coaxial_port import PEC_SIGMA, PTFE_EPS_R
    top = ground+4
    sim = Simulation(freq_max=5e9, domain=(32*DX, 24*DX, 40*DX),
                     dx=DX, cpml_layers=4, boundary="cpml")
    sim.add_material("substrate", eps_r=3., sigma=.02)
    sim.add(Box((0, 0, ground*DX), (32*DX, 24*DX, top*DX)), material="substrate")
    sim.add(Box((0, 0, ground*DX), (32*DX, 24*DX, ground*DX)), material="pec")
    sim.add(Box((0, 10*DX, top*DX), (32*DX, 14*DX, top*DX)), material="pec")
    sim.add_coaxial_port(position=(8*DX, 12*DX, ground*DX), face="bottom",
                         pin_radius=DX, outer_radius=4.5*DX)
    sim.add_msl_port(position=(8*DX, 12*DX, ground*DX), width=4*DX,
                     height=4*DX, direction="+x", eps_r_sub=3.)
    grid = sim._build_grid()
    registered = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    join = grid.pad_z_lo+15  # exact half-cell tie must also select node 15
    i, j = grid.pad_x_lo+8, grid.pad_y_lo+12

    class Captured(Exception):
        pass

    def capture(grid_arg, port, materials, **kwargs):
        for name in ("eps_r", "sigma"):
            np.testing.assert_array_equal(getattr(materials, name)[:, :, join:],
                                          getattr(registered, name)[:, :, join:])
        assert materials.sigma[i, j, join-1] == PEC_SIGMA
        assert materials.sigma[i, j+4, join-1] == PEC_SIGMA
        assert materials.eps_r[i, j+2, join-1] == pytest.approx(PTFE_EPS_R)
        assert materials.sigma[i, j+2, join-1] == 0.
        raise Captured

    monkeypatch.setattr(sources, "setup_msl_port", capture)
    with pytest.raises(Captured):
        sim.compute_coax_msl_transition(junction_x=8*DX, n_steps=1,
                                        n_freqs=2, probe_count=3,
                                        probe_start_cells=3, probe_spacing_cells=2,
                                        skip_preflight=True)


def test_instrument_junction_has_a_connected_post_and_an_open_clearance():
    from tests._coax_msl_instrument_fixture import build_instrument_junction
    sim = build_instrument_junction()
    grid, _, hard, _ = _validate((sim, False))
    i, j, k = grid.pad_x_lo+10, grid.pad_y_lo+17, grid.pad_z_lo+25
    # Three real normal edges connect ground level to the trace at z=28dx.
    assert np.asarray(hard[2])[i, j, k:k+3].all()
    assert hard[0][i, j, k+3]
    # A point between pin and ground hole has NO incident tangential metal.
    for a, b in ((i+3, j), (i, j+3)):
        assert not (hard[0][a, b, k] or hard[0][a-1, b, k]
                    or hard[1][a, b, k] or hard[1][a, b-1, k])
    assert hard[0][i+7, j, k]


def test_historical_attempt3_cannot_run_with_its_displaced_port():
    from tests.unit.sparams.test_coax_msl_transition import (
        _build_coax_msl_transition_sim_attempt3, _attempt2_kwargs,
    )
    sim = _build_coax_msl_transition_sim_attempt3()
    with pytest.raises(ValueError, match="declared ground"):
        sim.compute_coax_msl_transition(**_attempt2_kwargs(1))
