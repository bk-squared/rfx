"""E3 frozen contracts and replay; no FDTD ladder is run by these tests."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, Box
from rfx.runners import nonuniform as nu
from validation.research.multiband_nu import w7_accuracy_ad as w7

RESULT = Path(w7.RESULTS_DIR) / "e3_interface_eps_rule.json"


def fixture(arm="mb", scale=2, **kwargs):
    prof = w7.a1_profile(arm, scale)
    sim = Simulation(freq_max=2*w7.a1_f_true(),
                     domain=(w7.A1_A_X, w7.A1_B_Y, w7.L_Z), boundary="pec",
                     dx=w7.DXY0*scale, dz_profile=prof, **kwargs)
    for i, eps in enumerate(w7.A1_EPS[:-1]):
        sim.add_material(f"layer{i}", eps_r=eps)
        sim.add(Box((0, 0, w7.A1_EDGES[i]),
                    (w7.A1_A_X, w7.A1_B_Y, w7.A1_EDGES[i+1])), material=f"layer{i}")
    return sim, prof


def test_default_bit_identity():
    omitted, _ = fixture()
    explicit, _ = fixture(interface_eps="sampled")
    arrays = []
    traces = []
    for sim in (omitted, explicit):
        grid = sim._build_nonuniform_grid()
        arrays.append(sim._assemble_materials_nu(grid))
        sim.add_source((w7.A1_A_X/3, w7.A1_B_Y/2, w7.Z_SRC), "ey")
        sim.add_probe((w7.A1_A_X/3, w7.A1_B_Y/2, w7.Z_PRB), "ey")
        traces.append(np.asarray(sim.run(n_steps=200).time_series))
    for a, b in zip(jax.tree_util.tree_leaves(arrays[0]), jax.tree_util.tree_leaves(arrays[1])):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert np.any(traces[0] != 0), "identity comparison must exercise a live signal"
    assert traces[0].tobytes() == traces[1].tobytes()
    print("E3-B: material bytes identical; 200-step max |diff| =", np.max(abs(traces[0]-traces[1])))


@pytest.mark.parametrize("arm", ["uc", "mb", "az"])
@pytest.mark.parametrize("scale", [0.5, 1, 2])
def test_columns_and_default_table(arm, scale):
    sim, prof = fixture(arm, scale, interface_eps="dual_average")
    grid = sim._build_nonuniform_grid()
    mats = sim._assemble_materials_nu(grid)[0]
    ex, ey, ez = nu.assemble_interface_eps_nu(sim, grid, mats)
    idx = (grid.nx//2, grid.ny//2)
    ref = np.asarray(w7.dual_eps_nodes(prof), np.float32)
    rel = np.max(abs(np.asarray(ey[idx])-ref)/ref)
    assert rel <= 5e-7
    np.testing.assert_array_equal(ex[idx], ey[idx])
    ce = w7.cell_eps(prof, w7.A1_EDGES, w7.A1_EPS)
    np.testing.assert_array_equal(ez[idx], np.asarray(np.r_[ce, ce[-1]], np.float32))
    default = w7.a1_production_column(prof, scale)
    assert tuple(v["eps"] for v in default["interface_table"].values()) == tuple(
        float(np.float32(v)) for v in w7.A1_INTERFACE_TABLE[arm, scale])
    np.testing.assert_array_equal(w7.a1_production_column(prof, scale, interface_eps="dual_average")["column"], ey[idx])
    print(f"E3-C {arm}|{scale:g}: Ey relative={rel}; Ez max difference=0; default table unchanged")


def test_four_cell_weights_and_pec():
    # Ex shares four different y/z cells; unequal areas make equal weights wrong.
    sim = Simulation(freq_max=1e9, domain=(.004, .008, .008), boundary="pec",
                     dx=.002, dy_profile=np.array([.002, .004, .002]),
                     dz_profile=np.array([.003, .005]), interface_eps="dual_average")
    for y, z, eps in [(0, 0, 2), (0, 1, 4), (1, 0, 8), (1, 1, 16)]:
        sim.add_material(f"m{y}{z}", eps_r=eps)
        sim.add(Box((0, [0,.002][y], [0,.003][z]),
                    (.004, [.002,.006][y], [.003,.008][z])), material=f"m{y}{z}")
    grid = sim._build_nonuniform_grid()
    def ex():
        return np.asarray(nu.assemble_interface_eps_nu(sim, grid, sim._assemble_materials_nu(grid)[0])[0])
    expected = (2*2*3 + 4*2*5 + 8*4*3 + 16*4*5)/(6*8)
    assert ex()[0,1,1] == np.float32(expected)
    # PEC. Under the lattice ownership contract (#931) a Box is a VOLUME that
    # owns every node in its drawn range, so the shared corner node (1, 1) is
    # a PEC node whichever of the four cells the Box fills, and the rule's
    # all-PEC-edge branch applies: the node keeps its SAMPLED eps as a finite
    # fallback (the PEC mask enforces its fields). The Box goes in the large
    # (y1, z1) cell because §1.5 measures "sub-cell" against the mean of the
    # two cells at the nearest node, which reads the exact small (y0, z0)
    # cell (0.002 beside 0.004; 0.003 beside 0.005) as thinner than one cell.
    sim.add(Box((0, .002, .003), (.004, .006, .008)), material="pec")
    assert ex()[0,1,1] == np.float32(16)   # sampled fallback: the (y1, z1) cell's eps
    assert np.all(np.isfinite(ex()))


def test_opt_in_reaches_stepper(monkeypatch):
    sim, _ = fixture(interface_eps="dual_average")
    original = nu.run_nonuniform
    seen = []
    def capture(grid, materials, n_steps, **kwargs):
        expected = nu.assemble_interface_eps_nu(sim, grid, materials)
        for actual, ref in zip(kwargs["aniso_eps"], expected):
            np.testing.assert_array_equal(actual, ref)
        seen.append(True)
        return original(grid, materials, n_steps, **kwargs)
    monkeypatch.setattr(nu, "run_nonuniform", capture)
    sim.run(n_steps=2, skip_preflight=True)
    assert seen == [True]


@pytest.mark.parametrize("kind", ["debye", "lorentz", "smoothing", "kottke", "thin", "sheet", "rlc", "override", "traced", "distributed", "sparams"])
def test_refusals_before_step(kind, monkeypatch):
    sim, _ = fixture(interface_eps="dual_average")
    def forbidden(*args, **kwargs):
        pytest.fail("a refused combination reached the time stepper")
    monkeypatch.setattr(nu, "run_nonuniform", forbidden)
    kwargs = {}
    if kind in ("debye", "lorentz"):
        from rfx.materials.debye import DebyePole
        from rfx.materials.lorentz import lorentz_pole
        pole_kw = ({"debye_poles": [DebyePole(delta_eps=1, tau=1e-11)]} if kind == "debye"
                   else {"lorentz_poles": [lorentz_pole(delta_eps=1, omega_0=2*np.pi*3e9, delta=1e9)]})
        sim.add_material("pole", eps_r=2, **pole_kw)
        sim.add(Box((0,0,0), (w7.A1_A_X,w7.A1_B_Y,.01)), material="pole")
    elif kind in ("smoothing", "kottke"):
        kwargs["subpixel_smoothing"] = True if kind == "smoothing" else "kottke_pec"
    elif kind in ("thin", "sheet"):
        sim.add_thin_conductor(Box((0,0,.01),(w7.A1_A_X,w7.A1_B_Y,.01)),
                               **({"surface_impedance_f0": 1e10} if kind == "sheet" else {}))
    elif kind == "rlc":
        sim.add_lumped_rlc((.01,.001,.01), component="ey", R=50)
    elif kind == "override":
        kwargs["eps_override"] = jnp.ones(sim._build_nonuniform_grid().shape)
    with pytest.raises((ValueError, NotImplementedError), match="interface_eps"):
        if kind == "traced":
            def traced(profile):
                sim._dz_profile = profile
                return nu.run_nonuniform_path(sim, n_steps=1)
            jax.make_jaxpr(traced)(jnp.asarray(sim._dz_profile))
        elif kind == "distributed":
            sim._forward_distributed_nonuniform_from_materials(n_steps=1)
        elif kind == "sparams":
            sim._compute_waveguide_s_matrix_nu(n_steps=1, num_periods=1, normalize=True)
        else:
            nu.run_nonuniform_path(sim, n_steps=1, **kwargs)


@pytest.mark.parametrize("rule", ["sampled", "dual_average"])
def test_report_rule(rule, capsys):
    sim, _ = fixture(interface_eps=rule)
    report = sim.fidelity_report()
    assert f"interface eps rule (NU lane): {rule}" in capsys.readouterr().out
    domain = next(row for row in report if row["entity"] == "domain (the solved box)")
    assert domain.get("interface_eps_rule") == (rule if rule == "dual_average" else None)
    assert ("interface_eps_rule" in domain) == (rule == "dual_average")


def test_invalid_rule():
    with pytest.raises(ValueError, match="interface_eps"):
        fixture(interface_eps="harmonic")


def test_e3_replay():
    if not RESULT.exists():
        pytest.skip("E3 ladder not measured yet")
    data = json.loads(RESULT.read_text())
    if "a1" not in data:
        pytest.skip("E3 selfcheck only; ladder not measured yet")
    assert data["selfcheck"]["all_pass"]
    units = data["a1"]["units"]
    reference = json.loads(Path(w7.DEFAULT_OUT).read_text())["a1"]["units"]
    for arm in ("uc", "mb"):
        points = []
        for scale in (0.5, 1, 2):
            row = units[w7.a1_key(arm, scale, "production")]
            assert row["rfx_file"] == "/Users/byungkwankim/Documents/rfx-nu-exp3/rfx/__init__.py"
            assert row["git_sha"] and row["interface_eps_rule"] == "dual_average"
            assert row["valid"] and row["valid_10ns"] and not row["smoke"]
            assert abs(row["f_meas"]-row["f_model"]) <= .15e6 and row["g3_pass"]
            assert abs(row["f_meas"]-row["f_meas_10ns"]) <= .1e6 and row["invariance_pass"]
            ref = reference[w7.a1_key(arm, scale, "dual")]
            delta = abs(row["err_hz"]-ref["err_hz"])
            assert delta <= .5e6
            points.append((w7.DF0*scale, abs(row["err_hz"])))
            print(f"E3-F2 {arm}|{scale:g}: difference={delta/1e6} MHz")
        order = w7.fit_line(points)[0]
        assert order >= 1.8
        if arm == "uc":
            assert 1.8 <= order <= 2.2
        assert order == pytest.approx(data["a1"]["judge"]["reported"]["production_orders"][arm])
        print(f"E3-F1 {arm}: order={order}")
