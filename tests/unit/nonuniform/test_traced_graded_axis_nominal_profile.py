"""A port and a probe past the fine cells of a graded mesh that is a design variable.

The board is a grounded slab under a 50 ohm wire port. Its x cells are a JAX
tracer, so ``jax.jvp`` differentiates through them, and the nominal x profile is
graded: five 0.5 mm cells, eight 0.25 mm cells, then fifteen 0.5 mm cells. The
port is declared at x = 8.0 mm, which is node 20 of that profile; a probe sits at
9.0 mm (node 22) and another at 3.25 mm (node 8, inside the fine cells).

A tracer has no host node line, so before ``nominal_profiles=`` a metre position
on a traced axis was resolved on a uniform 0.5 mm mesh: 8.0 mm named node 16,
the run built node 20, and the run was refused. Nobody could place a port or a
probe past the fine cells. The caller now declares the concrete profile at the
primal point; positions resolve on its node line and the run-time check that
the built mesh names the same node still stands.

Rows: (a) the resolved nodes are the concrete run's; (b) the traced record at
the primal is the concrete ``forward()``; (c) a nominal that is not the traced
primal is refused at run time, naming both nodes; (d) without the declaration
the old refusal fires and names ``nominal_profiles=``; (e) declaration refusals.
"""

from __future__ import annotations

import warnings

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.core.jax_utils import is_tracer
from rfx.geometry import Box
from rfx.nonuniform import make_nonuniform_grid, position_to_index
from rfx.runners.nonuniform import pos_to_nu_index
from rfx.sources import GaussianPulse

C0 = 299792458.0

DX = 0.5e-3
PROF = np.concatenate([np.full(5, DX), np.full(8, DX / 2), np.full(15, DX)])
N_X, N_Y, N_Z = PROF.size, 20, 20
LX = float(PROF.sum())                       # 12 mm
CPML = 4
K_GND, K_PATCH = 6, 12
Z_G, Z_P = K_GND * DX, K_PATCH * DX
Y_C = 10 * DX
X_PORT, X_PROBE, X_FINE = 8.0e-3, 9.0e-3, 3.25e-3
#: Interior x nodes of PROF, counted by hand: node 5 at 2.5 mm, nodes 5..13 a
#: quarter millimetre apart (node 13 at 4.5 mm), then half a millimetre apart.
NODE_PORT, NODE_PROBE, NODE_FINE = 20, 22, 8
#: What the uniform 0.5 mm stand-in names for the same coordinates.
UNIFORM_PORT = 16

D_MIN = DX / 2
DT_PIN = 0.9 / (C0 * np.sqrt(1.0 / D_MIN ** 2 + 2.0 / DX ** 2))
FREQS = np.arange(10e9, 16e9 + 1e6, 1e9)
N_STEPS = 200


def _prof(delta, base=PROF):
    """Two cells grow by ``delta/2`` and the next two shrink by it, between
    nodes 16 and 20: every node the board names stays where it is."""
    if is_tracer(delta):
        d = jnp.asarray(base, dtype=jnp.float32)
        return d.at[16:18].add(delta / 2).at[18:20].add(-delta / 2)
    d = np.array(base, dtype=np.float64)
    d[16:18] += float(delta) / 2
    d[18:20] -= float(delta) / 2
    return d


def _board(delta, *, nominal=PROF, base=PROF):
    kw = {}
    if nominal is not None and is_tracer(delta):
        kw["nominal_profiles"] = {"x": nominal}
    sim = Simulation(freq_max=40e9, domain=(LX, N_Y * DX, N_Z * DX), dx=DX,
                     cpml_layers=CPML, boundary="cpml",
                     dx_profile=_prof(delta, base), dt=float(DT_PIN),
                     dt_min_cell=float(D_MIN), **kw)
    sim.add_material("sub", eps_r=3.38)
    sim.add(Box((0.0, 2 * DX, Z_G), (LX, 18 * DX, Z_P)), material="sub")
    sim.add_port(position=(X_PORT, Y_C, Z_G), component="ez", impedance=50.0,
                 extent=Z_P - Z_G,
                 waveform=GaussianPulse(f0=13e9, bandwidth=0.9))
    sim.add_probe((X_PROBE, Y_C, 0.5 * (Z_G + Z_P)), "ez")
    sim.add_probe((X_FINE, Y_C, 0.5 * (Z_G + Z_P)), "ez")
    return sim


def _forward(delta, n_steps=N_STEPS, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _board(delta, **kw).forward(n_steps=n_steps, port_s11_freqs=FREQS)
    return r.time_series, r.s_params


POINTS = ((X_PORT, Y_C, Z_G), (X_PROBE, Y_C, Z_G), (X_FINE, Y_C, Z_G))


# --------------------------------------------------------------------------
# (a) the nodes the traced run resolves are the concrete run's
# --------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [0.0, 0.25 * DX])
def test_a_ports_and_probes_past_the_fine_cells_resolve_to_the_concrete_nodes(delta):
    """Expected: the hand count of PROF, and the concrete grid of the same
    (deformed) profile, which resolves on its own float64 node line."""
    concrete_grid = _board(delta)._build_nonuniform_grid()
    concrete = [position_to_index(concrete_grid, p)[0] for p in POINTS]
    assert concrete == [n + CPML for n in (NODE_PORT, NODE_PROBE, NODE_FINE)]

    traced = []

    def _go(d):
        sim = _board(d)
        grid = sim._build_nonuniform_grid()
        assert is_tracer(grid.dx_arr)
        traced.extend(int(pos_to_nu_index(grid, p)[0]) for p in POINTS)
        return jnp.sum(grid.dx_arr)

    jax.jvp(_go, (jnp.float32(delta),), (jnp.float32(1.0),))
    assert traced == concrete, (traced, concrete)
    assert traced[0] != UNIFORM_PORT + CPML


# --------------------------------------------------------------------------
# (b) at the primal the traced run is the concrete forward()
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "#1373: the traced and the concrete graded paths form some coefficients "
    "by different expressions (host float64 against traced float32; candidate "
    "sites: port metric widths, the drive coefficient Cb), so the jvp primal "
    "is not the concrete forward(). Declared bar 1 float32 ULP of the record "
    "peak; measured here (100/200/400 steps) probe record 3.0/9.0/9.6 ULP, S "
    "1.1/2.0/3.2 ULP of max|S|; the same board on a uniform traced x without "
    "nominal_profiles 2.0/5.0/5.0 and 0.7/0.5/2.1 (pre-existing). The fix of "
    "#1373's graded coefficient forms removes this mark."))
def test_b_the_traced_record_at_the_primal_is_the_concrete_forward():
    ts_c, s_c = _forward(0.0)
    (ts_t, s_t), _ = jax.jvp(lambda d: _forward(d), (jnp.float32(0.0),),
                             (jnp.float32(1.0),))
    ts_c, ts_t = np.asarray(ts_c), np.asarray(ts_t)
    s_c, s_t = np.asarray(s_c), np.asarray(s_t)
    ulp_ts = np.spacing(np.float32(np.max(np.abs(ts_c))))
    ulp_s = np.spacing(np.float32(np.max(np.abs(s_c))))
    assert np.max(np.abs(ts_t - ts_c)) <= ulp_ts, (
        f"record: {np.max(np.abs(ts_t - ts_c)) / ulp_ts:.1f} ULP of its peak")
    assert np.max(np.abs(s_t - s_c)) <= ulp_s, (
        f"S: {np.max(np.abs(s_t - s_c)) / ulp_s:.1f} ULP of max|S|")


# --------------------------------------------------------------------------
# (c) a nominal that is not the traced primal is refused at run time
# --------------------------------------------------------------------------

#: The fine cells start two cells later. Same length, same sum; 3.0 mm is node
#: 7 of PROF but node 6 of this profile.
SHIFTED = np.concatenate([np.full(7, DX), np.full(8, DX / 2), np.full(13, DX)])


def test_c_a_declared_profile_the_run_does_not_build_is_refused():
    def _go(d):
        traced = jnp.asarray(SHIFTED, dtype=jnp.float32) + 0.0 * d
        grid = make_nonuniform_grid(
            (LX, N_Y * DX), np.full(N_Z, DX), DX, CPML, dx_profile=traced,
            nominal_profiles={"x": PROF})
        position_to_index(grid, (3.0e-3, Y_C, Z_G))
        return jnp.sum(grid.dx_arr)

    with pytest.raises(ValueError, match=(
            r"resolves to interior node 7 on the declared nominal_profiles= "
            r"profile, but to node 6 on the mesh the run built")):
        jax.jvp(_go, (jnp.float32(0.0),), (jnp.float32(1.0),))


# --------------------------------------------------------------------------
# (d) without the declaration the old refusal fires and names the keyword
# --------------------------------------------------------------------------

def test_d_without_a_declared_profile_the_graded_scene_is_refused():
    with pytest.raises(ValueError, match=(
            r"(?s)resolves to interior node 16 on the nominal uniform mesh"
            r".*but to node 20 on the mesh the run built.*nominal_profiles=")):
        jax.jvp(lambda d: _forward(d, n_steps=8, nominal=None),
                (jnp.float32(0.0),), (jnp.float32(1.0),))


# --------------------------------------------------------------------------
# (e) what the declaration refuses, before any grid exists
# --------------------------------------------------------------------------

def _declare(d, nominal, *, traced_x=True):
    return Simulation(freq_max=40e9, domain=(LX, N_Y * DX, N_Z * DX), dx=DX,
                      cpml_layers=CPML, boundary="cpml",
                      dx_profile=_prof(d) if traced_x else PROF,
                      dt=float(DT_PIN), dt_min_cell=float(D_MIN),
                      nominal_profiles=nominal)


def _refused(nominal, match, *, traced_x=True):
    def _go(d):
        _declare(d, nominal, traced_x=traced_x)
        return d

    with pytest.raises(ValueError, match=match):
        jax.jvp(_go, (jnp.float32(0.0),), (jnp.float32(1.0),))


def test_e_a_key_for_an_axis_that_is_not_traced_is_refused():
    _refused({"y": np.full(N_Y, DX)}, r"nominal_profiles\['y'\].*not traced")
    _refused({"x": PROF}, r"nominal_profiles\['x'\].*not traced",
             traced_x=False)
    with pytest.raises(ValueError, match=r"nominal_profiles\['x'\].*not traced"):
        _declare(0.0, {"x": PROF})


def test_e_a_wrong_length_is_refused():
    _refused({"x": PROF[:-1]}, r"nominal_profiles\['x'\] has shape \(27,\).*28 cells")


def test_e_a_sum_that_is_not_the_domain_is_refused():
    wrong = PROF.copy()
    wrong[-1] += 1e-6
    _refused({"x": wrong}, r"nominal_profiles\['x'\] sums to .*domain x extent")


def test_e_non_finite_non_positive_traced_or_unknown_values_are_refused():
    bad = PROF.copy()
    bad[3] = -bad[3]
    _refused({"x": bad}, "finite positive")
    bad = PROF.copy()
    bad[3] = np.nan
    _refused({"x": bad}, "finite positive")
    _refused({"w": PROF}, "is not an axis")

    def _go(d):
        _declare(d, {"x": _prof(d)})
        return d

    with pytest.raises(ValueError, match=r"nominal_profiles\['x'\] is a JAX tracer"):
        jax.jvp(_go, (jnp.float32(0.0),), (jnp.float32(1.0),))


def test_f_the_padded_nominal_widths_index_like_the_grids_own_widths():
    """``nominal_cells(grid, axis, padded=True)`` is read by callers that
    index cells by full-grid index (the wire-port radius stamp). It must have
    the length and indexing of the grid's own width array: compare it with the
    float64 widths the CONCRETE build of the same profile carries."""
    from rfx.model.graded_admission import nominal_cells
    concrete = make_nonuniform_grid((LX, N_Y * DX), np.full(N_Z, DX), DX, CPML,
                                    dx_profile=PROF)
    seen = {}

    def traced(delta):
        grid = make_nonuniform_grid((LX, N_Y * DX), np.full(N_Z, DX), DX, CPML,
                                    dx_profile=_prof(delta),
                                    nominal_profiles={"x": PROF})
        seen["x"] = nominal_cells(grid, "x", padded=True)
        seen["interior"] = nominal_cells(grid, "x")
        seen["y"] = nominal_cells(grid, "y", padded=True)
        seen["n"] = grid.shape[0]
        return jnp.sum(grid.dx_arr)

    jax.jvp(traced, (jnp.float32(0.0),), (jnp.float32(1.0),))
    assert seen["x"].shape == (seen["n"],) == (concrete.shape[0],)
    np.testing.assert_array_equal(seen["x"], np.asarray(concrete.dx_arr_f64))
    np.testing.assert_array_equal(seen["interior"], PROF)
    assert seen["y"] is None                      # y is not traced
    assert nominal_cells(concrete, "x", padded=True) is None
