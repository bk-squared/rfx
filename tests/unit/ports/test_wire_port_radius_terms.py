"""A declared lattice radius adds only the four contour E storage terms."""

import jax
import numpy as np

from rfx import Box, GaussianPulse, Simulation
from rfx._internal import _realized
from rfx.sources.wire_radius import LATTICE_RADIUS


D = 1e-3
PORT_CELL = (4, 4, 4)
COMPONENT = "ey"


def _scene(radius):
    sim = Simulation(freq_max=15e9, domain=(8 * D,) * 3, dx=D,
                     boundary="pec", cpml_layers=0)
    sim.add_material("background", eps_r=4.0, mu_r=2.0)
    sim.add(Box((0.,) * 3, (8 * D,) * 3), material="background")
    sim.add_port(tuple(i * D for i in PORT_CELL), COMPONENT, extent=D,
                 radius=radius, waveform=GaussianPulse(f0=5e9))
    return sim


def _realize(sim):
    with _realized.capture() as captured:
        sim.run(n_steps=4, compute_s_params=False, skip_preflight=True)
    electric = [r for r in captured.records if "eps_e" in r and "materials" in r]
    magnetic = [r for r in captured.records if "mu_h" in r and "materials" in r]
    assert electric and magnetic
    return electric[0], magnetic[0]


def _parts(record, shape):
    return tuple(np.zeros(shape) if part is None else np.asarray(part)
                 for part in ((None,) * 3 if record is None else record))


def _assert_same_tree(actual, expected):
    have, have_tree = jax.tree.flatten(actual)
    want, want_tree = jax.tree.flatten(expected)
    assert have_tree == want_tree
    for a, b in zip(have, want, strict=True):
        np.testing.assert_array_equal(a, b)


def test_lattice_radius_adds_only_four_contour_storage_terms(capsys):
    bare = _scene(None)
    declared = _scene(LATTICE_RADIUS * D)
    # The uniform scene declaration supplies the oracle, independently of stamping.
    eps_c = declared._materials[declared._geometry[0].material_name].eps_r
    with capsys.disabled():
        for issue in bare.preflight():
            if issue.code == "wire_port_no_radius":
                print(str(issue))
    bare_e, bare_h = _realize(bare)
    declared_e, declared_h = _realize(declared)
    before, after = bare_e["materials"], declared_e["materials"]
    shape = after.eps_r.shape

    for part in _parts(after.mu_r_wire, shape):
        np.testing.assert_array_equal(part, np.zeros(shape))
    for part in _parts(before.mu_r_wire, shape):
        np.testing.assert_array_equal(part, np.zeros(shape))
    _assert_same_tree(declared_h["mu_h"], bare_h["mu_h"])

    expected = np.zeros((3, *shape))
    neighbours = ((3, 4, 4), (5, 4, 4), (4, 4, 3), (4, 4, 5))
    for cell in neighbours:
        expected[(1, *cell)] = eps_c / 4
    difference = (np.stack(_parts(after.eps_r_lumped, shape))
                  - np.stack(_parts(before.eps_r_lumped, shape)))
    with capsys.disabled():
        for axis, i, j, k in np.argwhere(difference != 0):
            print(f"cell={(int(i), int(j), int(k))}, component={'ex ey ez'.split()[axis]}, "
                  f"delta_eps_r={difference[axis, i, j, k]:g}")
    np.testing.assert_array_equal(difference, expected)
    np.testing.assert_array_equal(after.eps_r - before.eps_r, expected.sum(axis=0))
    np.testing.assert_array_equal(
        np.stack(declared_e["eps_e"]) - np.stack(bare_e["eps_e"]), expected)
    _assert_same_tree(declared_e["sigma_e"], bare_e["sigma_e"])
    for field in before._fields:
        if field not in ("eps_r", "eps_r_lumped", "mu_r_wire"):
            _assert_same_tree(getattr(after, field), getattr(before, field))
