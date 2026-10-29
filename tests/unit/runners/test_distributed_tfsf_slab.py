"""Predeclared TF/SF slab witnesses; the physical anchor is checked first."""

import importlib.util
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

from rfx import Simulation, Sphere
from rfx.measurement.farfield import compute_far_field
from rfx.grid import C0
from rfx.measurement.incident import measure_normal_incident_spectrum
from rfx.sources.tfsf import tfsf_from_declaration


def test_f4_one_device_mie_anchor():
    """Simulation's declared PEC sphere must independently meet the 1 dB bar."""
    fixture_dir = Path(__file__).resolve().parents[2] / "fixtures" / "rcs_sphere_mie"
    geo = json.loads((fixture_dir / "fixture.json").read_text())["geometry"]
    spec = importlib.util.spec_from_file_location(
        "tfsf_slab_mie_oracle", fixture_dir / "mie_oracle.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    oracle.validate_oracle()

    f0, radius = geo["f0_hz"], geo["radius_m"]
    dx = C0 / f0 / geo["resolution_cells_per_lambda"]
    sim = Simulation(freq_max=f0 * 1.5, domain=tuple(geo["domain_m"]),
                     dx=dx, boundary="cpml", cpml_layers=geo["cpml_layers"])
    sim.add_material("PEC", eps_r=1.0, sigma=geo["pec_sigma_s_per_m"])
    sim.add(Sphere(center=tuple(d / 2 for d in geo["domain_m"]), radius=radius),
            material="PEC")
    sim.add_tfsf_source(f0=f0, bandwidth=geo["bandwidth"],
                        polarization=geo["polarization"], direction="+x",
                        closed_box=True, margin=3)
    grid = sim._build_grid()
    assert tuple(grid.shape) == tuple(geo["grid_shape"])
    # TF box nodes are 27..62; these NTFF nodes (26, 64) enclose all
    # correction planes, with the face-centred H interpolation outside TF.
    sim.add_ntff_box((2 * dx,) * 3, (40 * dx,) * 3, freqs=[f0])
    cfg, state = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max,
                                       n_steps=geo["n_steps"])
    assert (cfg.x_lo, cfg.x_hi) == (27, 62)
    with redirect_stdout(sys.stderr):
        result = sim.run(n_steps=geo["n_steps"])
    box = result.ntff_box
    assert (box.i_lo, box.i_hi, box.j_lo, box.j_hi, box.k_lo, box.k_hi) == (
        26, 64, 26, 64, 26, 64)
    ff = compute_far_field(result.ntff_data, box, result.grid, [np.pi / 2], [np.pi])
    incident = measure_normal_incident_spectrum(
        cfg, state, geo["n_steps"], [f0], grid.dt)
    power = np.abs(np.asarray(ff.E_theta)) ** 2 + np.abs(np.asarray(ff.E_phi)) ** 2
    sigma = float((4 * np.pi * power / np.abs(incident[:, None, None]) ** 2).item())
    dbsm = 10 * np.log10(sigma)
    ka = 2 * np.pi * radius * f0 / C0
    s1, _ = oracle.mie_S1_S2(ka, np.pi, n_max=20)
    mie_sigma = 4 * np.pi * radius ** 2 * abs(s1) ** 2 / ka ** 2
    mie_dbsm = 10 * np.log10(mie_sigma)
    error = abs(dbsm - mie_dbsm)
    print(f"F4: one-device={dbsm:.12f} dBsm; Mie={mie_dbsm:.12f} dBsm; "
          f"one-device error={error:.12f} dB; two-device=not run", file=sys.stderr)
    assert error <= 1.0, (
        f"F4 STOP: one-device Simulation {dbsm:.12f} dBsm vs Mie "
        f"{mie_dbsm:.12f} dBsm; error {error:.12f} dB > 1.0 dB")

    if len(jax.devices("cpu")) < 2:
        pytest.skip("F4 two-device anchor requires two host devices")
    with redirect_stdout(sys.stderr):
        two = sim.run(n_steps=geo["n_steps"], devices=jax.devices("cpu")[:2])
    ff_two = compute_far_field(two.ntff_data, two.ntff_box, two.grid, [np.pi / 2], [np.pi])
    power_two = np.abs(np.asarray(ff_two.E_theta)) ** 2 + np.abs(np.asarray(ff_two.E_phi)) ** 2
    two_dbsm = 10 * np.log10(float((4 * np.pi * power_two / np.abs(incident[:, None, None]) ** 2).item()))
    two_error = abs(two_dbsm - mie_dbsm)
    a = np.stack([np.asarray(ff.E_theta), np.asarray(ff.E_phi)])
    b = np.stack([np.asarray(ff_two.E_theta), np.asarray(ff_two.E_phi)])
    delta = np.max(np.abs(a - b))
    peak = np.max(np.abs(a))
    print(f"F4: one error={error:.12f} dB; two error={two_error:.12f} dB; "
          f"one={dbsm:.12f}, two={two_dbsm:.12f}, Mie={mie_dbsm:.12f} dBsm; "
          f"far-field max delta={delta:.12g}, peak={peak:.12g}, relative={delta / peak:.12g}", file=sys.stderr)
    assert two_error <= 1.0
    assert delta <= 1e-4 * peak


@pytest.mark.parametrize("count", [2, 3])
def test_splitter_owned_rows_and_padding(count):
    """Expected indices are plain NumPy arithmetic, independent of the splitter."""
    from jax.sharding import Mesh
    from rfx.sources.tfsf_discrete_plane_wave import CorrectionTable
    from rfx.stepping.slab import Slab
    from rfx.stepping.slab_tfsf import split_tables
    if len(jax.devices("cpu")) < count:
        pytest.skip(f"requires {count} host devices")
    shape = (11, 5, 4)
    layout = Slab(shape[0], count)
    mesh = Mesh(np.array(jax.devices("cpu")[:count]), ("x",))
    # Deliberately unordered targets and unequal rank counts, including both seam neighbours.
    nodes = np.array([(10, 4, 3), (0, 1, 2), (4, 3, 1), (6, 0, 2), (5, 2, 0), (8, 1, 1)])
    target = np.ravel_multi_index(nodes.T, shape).astype(np.int32)
    source = np.arange(len(target), dtype=np.int32) + 7
    coefficient = np.arange(len(target), dtype=np.float32) + .125
    table = CorrectionTable(2, target, 4, source, coefficient)
    actual, = split_tables((table,), shape, layout, mesh)
    cut_target, cut_source, cut_coefficient = map(np.asarray, actual)
    width = (shape[0] + count - 1) // count
    for rank in range(count):
        selected = np.flatnonzero(nodes[:, 0] // width == rank)
        local = nodes[selected].copy()
        local[:, 0] -= rank * width - 1
        expected = np.ravel_multi_index(local.T, (width + 2, *shape[1:]))
        np.testing.assert_array_equal(cut_target[rank, :len(selected)], expected)
        np.testing.assert_array_equal(cut_source[rank, :len(selected)], source[selected])
        np.testing.assert_array_equal(cut_coefficient[rank, :len(selected)], coefficient[selected])
        assert np.all(cut_target[rank, len(selected):] == (width + 2) * shape[1] * shape[2])
        assert np.all((local[:, 0] > 0) & (local[:, 0] < width + 1))
    assert all(tuple(a.sharding.spec) == ("x",) for a in actual)


@pytest.mark.parametrize("count", [2, 3])
def test_carry_shardings(count):
    from jax.sharding import Mesh
    from rfx.runners._distributed_records import SlabRecords
    from rfx.stepping.slab import Slab
    from rfx.stepping.slab_tfsf import prepare
    if len(jax.devices("cpu")) < count:
        pytest.skip(f"requires {count} host devices")
    sim = Simulation(freq_max=15e9, domain=(.024, .018, .016), dx=.001, cpml_layers=4)
    sim.add_tfsf_source(f0=7.5e9, closed_box=True)
    sim.add_ntff_box((.002,) * 3, (.022, .016, .014), freqs=[7.5e9])
    sim.add_dft_plane_probe(axis="y", coordinate=.01, component="ez", freqs=[7.5e9])
    grid = sim._build_grid()
    layout = Slab(grid.nx, count)
    mesh = Mesh(np.array(jax.devices("cpu")[:count]), ("x",))
    _, line, tables = prepare(sim, grid, 60, layout, mesh)
    for field, leaf in zip(line._fields, line):
        assert tuple(leaf.sharding.spec) == ()
        print(f"{count} devices carry/tfsf/{field}: shape={leaf.shape}, {leaf.sharding}", file=sys.stderr)
    records = SlabRecords(sim, grid, layout.nx_per_rank, mesh, np.float32, 60).initial()
    for name in ("ntff", "planes"):
        leaf = records[name]
        assert tuple(leaf.sharding.spec) == ("x",)
        print(f"{count} devices carry/records/{name}: shape={leaf.shape}, {leaf.sharding}", file=sys.stderr)
    assert all(tuple(a.sharding.spec) == ("x",) for a in jax.tree.leaves(tables))
