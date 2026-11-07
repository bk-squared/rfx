"""A soft point source from ``add_source`` is a source, never a lumped port.

``Simulation._ports`` holds both ``add_source`` soft sources (impedance 0)
and ``add_port`` ports (impedance > 0). The realized-geometry record must
list a soft source once, as a ``soft_point`` source row, and list only the
``add_port`` entries under ``ports``. Expected labels, kinds and edge
indices are written by hand (nearest declared node plus the absorber
padding), not read from the helpers under test.
"""
import warnings

import numpy as np
import pytest

from rfx import Simulation

DZ = np.array([1e-3] * 3 + [0.75e-3] * 6 + [1e-3] * 3)
SOURCE = (0.004, 0.005, 0.0060)    # graded z = 6.0 mm is a node (3 mm + 4 x 0.75 mm)
PORT = (0.008, 0.007, 0.0030)      # z = 3.0 mm is a node on both meshes
PROBE = (0.006, 0.006, 0.0075)
LANES = ("uniform_record", "graded_record", "uniform_run", "graded_run")


def _scene(graded, with_port):
    kw = dict(dz_profile=DZ) if graded else {}
    sim = Simulation(freq_max=10e9, domain=(0.012, 0.012, 0.0 if graded else 0.012),
                     dx=1e-3, boundary="cpml", cpml_layers=4, **kw)
    sim.add_source(SOURCE, "ez", amplitude_kind="current")
    if with_port:
        sim.add_port(PORT, "ez", impedance=50.0)
    sim.add_probe(PROBE, "ez")
    return sim


def _record(lane, with_port):
    graded = lane.startswith("graded")
    sim = _scene(graded, with_port)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if lane.endswith("_record"):
            return sim.realized_geometry()
        return sim.run(n_steps=1, skip_preflight=True, compute_s_params=False).realized_geometry


def _hand_index(position, graded, pad=4):
    x_nodes = np.arange(13) * 1e-3
    z_nodes = np.concatenate([[0.0], np.cumsum(DZ)]) if graded else x_nodes
    return tuple(int(np.argmin(np.abs(nodes - p))) + pad
                 for nodes, p in zip((x_nodes, x_nodes, z_nodes), position))


def _labels(record):
    return [row.label for row in record.realization_table.rows]


def test_entry_rule_for_every_stored_kind():
    sim = Simulation(freq_max=10e9, domain=(0.012,) * 3, dx=1e-3, boundary="cpml", cpml_layers=4)
    sim.add_source(SOURCE, "ez")
    sim.add_port(PORT, "ez", impedance=50.0)
    sim.add_port((0.006, 0.006, 0.003), "ez", impedance=1e-6, extent=0.003)
    sim.add_port((0.003, 0.003, 0.003), "ez", impedance=50.0, excite=False)
    assert [(pe.is_soft_source, pe.is_port) for pe in sim._ports] == [
        (True, False), (False, True), (False, True), (False, True)]


@pytest.mark.parametrize("lane", LANES)
def test_soft_source_alone_lists_no_port(lane):
    graded = lane.startswith("graded")
    record = _record(lane, with_port=False)
    assert record.ports == ()
    (source,) = record.sources
    assert (source.label, source.kind, source.component) == ("port[0]", "soft_point", "ez")
    assert source.edges == (_hand_index(SOURCE, graded),)
    assert _labels(record).count("port[0]") == 1
    (row,) = [r for r in record.realization_table.rows if r.label == "port[0]"]
    assert row.family == "soft_point"


@pytest.mark.parametrize("lane", LANES)
def test_soft_source_and_port_list_the_port_once(lane):
    graded = lane.startswith("graded")
    record = _record(lane, with_port=True)
    (port,) = record.ports
    assert (port.label, port.kind, port.component) == ("port[1]", "lumped", "ez")
    assert port.edges == (_hand_index(PORT, graded),)
    (source,) = record.sources
    assert (source.label, source.kind) == ("port[0]", "soft_point")
    assert source.edges == (_hand_index(SOURCE, graded),)
    labels = _labels(record)
    assert labels.count("port[0]") == 1 and labels.count("port[1]") == 1
    families = {r.label: r.family for r in record.realization_table.rows}
    assert (families["port[0]"], families["port[1]"]) == ("soft_point", "lumped")
