"""Full-declaration automatic probes and microstrip admission diagnostics."""
import warnings

import pytest

from rfx import Box, Simulation


def _repro(order, *, eps_r_sub=None):
    sim = Simulation(freq_max=5e9, domain=(.014, .00297, .001754),
                     dx=254e-6 / 3, boundary="cpml", cpml_layers=8)

    def structure():
        sim.add_material("sub", eps_r=3.66)
        sim.add(Box((0, 0, 0), (.014, .00297, 254e-6)), material="sub")
        sim.add(Box((0, 0, 0), (.014, .00297, 0)), material="pec")
        sim.add(Box((.002, .001185, 254e-6), (.012, .001785, 254e-6)),
                material="pec")

    def ports():
        for name, x, direction in (("p1", .002, "+x"), ("p2", .012, "-x")):
            sim.add_msl_port(position=(x, .001485, 0), width=600e-6,
                             height=254e-6, direction=direction, name=name,
                             eps_r_sub=eps_r_sub)

    for declare in ((structure, ports) if order == "structure-first"
                    else (ports, structure)):
        declare()
    return sim


def _counts(entries):
    return [(pe.n_probe_offset, pe.n_probe_spacing) for pe in entries]


def _automatic_rows(sim):
    return (dict(sim._msl_auto_offset_min), dict(sim._msl_auto_probe_spacing),
            dict(sim._msl_auto_probe_lengths))


def test_t1_full_declaration_sets_automatic_probe_planes():
    # Measured on cd180aa36 before implementation by building _repro in both
    # orders and reading raw entries and _resolve_msl_probe_entries(_build_grid()).
    # Structure-first raw: (29, 12); resolved: (29, 6), for BOTH ports.
    # Ports-first raw: (56, 22); resolved: (56, 2), for BOTH ports.
    # The absorber budget makes the resolved spacing 6, not the design's 12.
    first = _repro("structure-first")
    last = _repro("ports-first")
    explicit = _repro("ports-first", eps_r_sub=3.66)
    assert _counts(first._msl_ports) == [(29, 12), (29, 12)]
    assert _counts(last._msl_ports) == [(56, 22), (56, 22)]
    assert _counts(explicit._msl_ports) == [(29, 12), (29, 12)]
    # The three bookkeeping rows were printed in the same base measurement.
    first_rows = (
        {"p1": 29, "p2": 29},
        {"p1": 2.8693862252597855, "p2": 2.8693862252597855},
        {"p1": (0.0024940220675408664, 0.003917600702638606),
         "p2": (0.0024940220675408664, 0.003917600702638606)},
    )
    assert _automatic_rows(first) == first_rows
    for sim in (first, last, explicit):
        raw = list(sim._msl_ports)
        grid = sim._build_grid()
        resolved = sim._resolve_msl_probe_entries(grid)
        assert _counts(resolved) == [(29, 6), (29, 6)]
        assert _automatic_rows(sim) == first_rows
        assert _counts(sim._resolve_msl_probe_entries(grid)) == [(29, 6), (29, 6)]
        assert sim._msl_ports == raw
    # The forward routing guard reads automatic lengths without resolving
    # entries first, even though it ultimately refuses MSL S11 requests.
    direct = _repro("ports-first")
    with pytest.raises(ValueError, match="MSL ports use compute_msl_s_matrix"):
        direct._validate_forward_sparameter_request()
    assert _automatic_rows(direct) == first_rows
    # preflight() admits once before any check reads the automatic counts.
    checked = _repro("ports-first")
    real_check = Simulation._check_msl_port_geometry
    seen = []

    def spy(self, *args, **kwargs):
        seen.append(_automatic_rows(self))
        return real_check(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp.setattr(Simulation, "_check_msl_port_geometry", spy)
        checked.preflight()
    assert seen == [first_rows]


def _box_board(*, sideways=0.0, normal=0.0, second_trace=False):
    sim = Simulation(freq_max=20e9, domain=(.003, .003, .001),
                     dx=50e-6, cpml_layers=0)
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, .0002), (.003, .003, .0004)), material="sub")
    sim.add(Box((0, 0, .0002), (.003, .003, .0002)), material="pec")
    sim.add(Box((0, .000987, .0004), (.003, .001587, .0004)), material="pec")
    if second_trace:  # another line on the same plane, not under the port
        sim.add(Box((0, .0021, .0004), (.003, .0025, .0004)), material="pec")
    sim.add_msl_port(position=(.0005, .001287 + sideways, .0002 + normal),
                     width=.0006, height=.0002, direction="+x", name="checked")
    return sim


def test_t2_sideways_overhang_names_width_and_preserves_z_mismatch():
    aligned = _box_board()
    aligned._msl_conductor_gap(aligned._msl_ports[0], aligned._msl_assemble_once())
    sideways = _box_board(sideways=13e-6)
    with pytest.raises(ValueError) as caught:
        sideways._msl_conductor_gap(sideways._msl_ports[0], sideways._msl_assemble_once())
    message = str(caught.value)
    assert "past" in message and "along y" in message
    assert "maps to node" not in message
    assert "trace" in message and "checked" in message
    assert "[0.001, 0.0016] m" in message
    assert "[0.001, 0.00155] m" in message
    assert "50 um past" in message
    neighbour = _box_board(sideways=13e-6, second_trace=True)
    with pytest.raises(ValueError) as caught:
        neighbour._msl_conductor_gap(neighbour._msl_ports[0],
                                     neighbour._msl_assemble_once())
    assert str(caught.value) == message
    normal = _box_board(normal=50e-6)
    with pytest.raises(ValueError, match="maps to node"):
        normal._msl_conductor_gap(normal._msl_ports[0], normal._msl_assemble_once())


def test_t3_memory_error_propagates_from_preflight(monkeypatch):
    sim = _box_board()
    # General assembly validation runs before the MSL check. Populate its
    # cache so this injection reaches the MSL check's own assembly call.
    sim._campaign_ctx().realized()

    def fail(*args, **kwargs):
        raise MemoryError("injected MSL assembly allocation failure")

    monkeypatch.setattr(sim, "_assemble_realized", fail)
    with pytest.raises(MemoryError, match="injected MSL assembly allocation failure"):
        sim.preflight()


def test_t3_value_error_remains_an_assembly_diagnostic(monkeypatch):
    sim = _box_board()
    sim._campaign_ctx().realized()

    def fail(*args, **kwargs):
        raise ValueError("injected MSL geometry refusal")

    monkeypatch.setattr(sim, "_assemble_realized", fail)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    assert any(issue.diagnostic is not None
               and issue.diagnostic.code == "msl.conductor_assembly_unavailable"
               for issue in report)


def _port_first(sim):
    """Re-declare ``sim``'s single port before all of its geometry."""
    pe, = sim._msl_ports
    geometry = list(sim._geometry)
    sim._geometry.clear()
    sim._msl_ports.clear()
    for rows in (sim._msl_auto_offset_min, sim._msl_auto_probe_spacing,
                 sim._msl_auto_probe_lengths):
        rows.clear()
    sim.add_msl_port(position=pe.position, width=pe.width, height=pe.height,
                     direction=pe.direction, impedance=pe.impedance, name=pe.name)
    sim._geometry.extend(geometry)
    return sim


@pytest.mark.parametrize("direction", ("+x", "-y"))
def test_a_ladder_crossing_a_ramp_keeps_the_full_declaration_counts(direction):
    """Graded runway, ladder crossing the trailing ramp: the resolver keeps
    scalar-cell counts instead of recounting. Structure first they are
    (9, 4) (tests/unit/sparams/test_msl_probe_offset_interval.py); declared
    port first, cd180aa36 kept the add-time eps=1 counts (17, 7)."""
    from tests.unit.sparams.test_msl_probe_offset_interval import (
        _resolve_on_its_grid, _runway_board)
    sim = _port_first(_runway_board(direction, 5.5e9, runway_node=50.3))
    assert (sim._msl_ports[0].n_probe_offset,
            sim._msl_ports[0].n_probe_spacing) == (17, 7)
    r, _, skips = _resolve_on_its_grid(sim)
    assert (r.n_probe_offset, r.n_probe_spacing) == (9, 4)
    (msg,) = skips
    assert "would cross a grading ramp" in msg
