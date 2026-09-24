"""A port's drive on the graded-mesh lane sees ``eps_override`` (#1267).

A 50 ohm wire port feeds a patch through a 1.5 mm substrate of two layers --
1.0 mm of eps_r 3.38 under 0.5 mm of eps_r 10.2 -- on a 0.5 mm mesh whose air
cells above the patch grow to 1 mm. The port is a current source in parallel
with its load on the three Ez edges it spans, and a current source enters the
field through the update coefficient of its edge, Cb = dt/(eps + sigma*dt/2),
which carries the permittivity there. The same board can be declared two
ways: the layers as materials, or a plain 3.38 slab whose permittivity is
replaced by the layered array through ``eps_override`` -- how
``Simulation.forward()`` hands a design variable to the solver. The E update
reads the overridden permittivity; the graded lane built every port drive
from the permittivity as DRAWN. So the override board drove the top edge
with Cb(3.38) while stepping it with Cb(10.2), and the two declarations of
one board gave different answers: S11 0.027 apart at 8 GHz, and at 6 GHz
the override board reflected more than it received (+0.065 dB), which a
passive board cannot. On a one-material 3.38 slab, raising eps_r by e^0.1
through the override scaled every field by kappa = Cb_drawn/Cb_override at
the port edges -- 1.0361 here, the closed form below, measured 1.036126 --
on top of what the substrate change itself does. ``jax.grad`` through the
override differentiated that wrong board. The derivative of the probe's
ln|E|^2 at 9 GHz with respect to the slab's ln eps_r came out 1.521 against
0.834 from central differences of boards built with that eps_r; the excess,
0.687, is 2 d(ln kappa)/d(ln eps_r) = 2 eps/(eps + sigma_port*dt/2) = 0.687
by the closed form. Where kappa differs between the port's edges it moves
S11 too: d|S11|^2/d(ln eps_r of the 10.2 layer) at 9 GHz read -0.6939
against -0.6869 (1.0 %). The uniform ``forward()`` builds its drives from
the overridden, traced materials; the graded lane now does too.

What is checked:

1. override == materials: the layered board declared by override and by
   materials, S11 at 4/6/8 GHz and the probe record -- identical;
2. the drive-strength ratio is gone: a one-material slab's eps_r x e^0.1
   by override vs built with it, probe field ratio 1 within 1e-5;
3. the derivative: ``jax.grad`` through the override against a Richardson
   central difference (h = 0.02, 0.01; 0.04 as the ladder check) of boards
   BUILT with the perturbed eps_r -- a route that never touches the
   override -- within 1e-3, for |S11|^2 (the 10.2 layer of the layered
   board) and for the probe's ln|E|^2 (a one-material slab);
4. without an override nothing moved: the run is byte-identical to the
   same run with the pre-#1267 drive rebuilt from the drawn arrays. Checked
   the same way against ``origin/main`` itself (4975f2ee) outside the suite:
   the wire port (with and without a 0.3 pF capacitor across its middle
   edge), the single-cell lumped port (with and without it), the MSL feed
   and a soft source, no override, and the wire and MSL feed under an
   override equal to the drawn array -- time series and S, 0 differing
   bytes;
5. mutation (b): the drive handed the pre-override arrays, every builder
   call kept (the lumped stamps put back on top, which is exactly the copy
   the runner kept before the fix), sends 1-3 red by the factors recorded
   in each test;
6. at build time, every port kind the graded lane drives (wire, single-cell
   lumped, MSL feed, and a wire or lumped port with a capacitor across its
   edge) under an override: the Cb each drive is built from equals the Cb
   the stepper applies on that edge; and a traced override reaches the MSL
   feed without touching its static launch fixture.

The drive is traced when the override is (as on the uniform lane); the MSL
launch fixture's substrate eps_r -- a static mode shape, the uniform
``forward()``'s #483 rule -- is still read from the drawn array, so a traced
override never reaches the host-side Laplace solve.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager, nullcontext
from unittest import mock

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.core.yee import cell_component_e_coeffs, cell_component_e_materials
from rfx.geometry import Box
from rfx.runners import nonuniform as _nu_runner
from rfx.runners.nonuniform import assemble_materials_nu, run_nonuniform_path
from rfx.sources import GaussianPulse
from tests._x64_compat import enable_x64

DX = 0.5e-3
NX, NY = 24, 20
#: 0.5 mm up to one cell above the patch, then air cells growing to 1 mm.
DZ = np.array([DX] * 8 + [0.6e-3, 0.7e-3, 0.85e-3, 1.0e-3, 1.0e-3])
CPML = 6
K_G, K_P = 5, 8                     # ground and patch node planes on z
Z_G, Z_P = K_G * DX, K_P * DX
Z_L = (K_G + 2) * DX                # the layer interface: 2 cells + 1 cell
EPS_LO, EPS_HI = 3.38, 10.2
Z0 = 50.0
I_PORT, J_PORT = 9, 10
PROBE = (20 * DX, J_PORT * DX, 0.5 * (Z_G + Z_P))
FREQS = np.array([4e9, 6e9, 8e9])
F_BIN = 8e9                         # the probe field bin of test 2
T_RECORD = 2.5e-9
N_AD = 1200                         # the gradient record (1.14 ns)
F_AD = np.array([9e9])              # on the slope of the patch's dip
LN_STEP = 0.1                       # test 2: eps_r x e^0.1
RAISED = (EPS_LO * np.exp(LN_STEP),) * 2

# Pre-declared gates.
S11_ATOL = 1e-6
FIELD_RTOL = 1e-6
KAPPA_RTOL = 1e-5
GRAD_RTOL = 1e-3
FD_LADDER = (0.04, 0.02, 0.01)

EPS_0 = 8.8541878128e-12


def _board(eps_lo, eps_hi, *, port="wire", cap=None):
    sim = Simulation(freq_max=30e9, domain=(NX * DX, NY * DX, float(DZ.sum())),
                     dx=DX, cpml_layers=CPML, boundary="cpml", dz_profile=DZ)
    sim.add_material("lower", eps_r=float(eps_lo))
    sim.add_material("upper", eps_r=float(eps_hi))
    sim.add(Box((2 * DX, 2 * DX, Z_G), ((NX - 2) * DX, (NY - 2) * DX, Z_L)),
            material="lower")
    sim.add(Box((2 * DX, 2 * DX, Z_L), ((NX - 2) * DX, (NY - 2) * DX, Z_P)),
            material="upper")
    sim.add_pinned_sheet(plane_index=K_G, i_range=(2, NX - 2),
                         j_range=(2, NY - 2), name="ground")
    pulse = GaussianPulse(f0=10e9, bandwidth=0.9)
    if port == "wire":
        sim.add_pinned_sheet(plane_index=K_P, i_range=(6, 18),
                             j_range=(5, 15), name="patch")
        sim.add_port(position=(I_PORT * DX, J_PORT * DX, Z_G),
                     component="ez", impedance=Z0, extent=Z_P - Z_G,
                     excite=True, waveform=pulse)
    elif port == "lumped":
        # One Ez edge, in the top (eps_r 10.2) layer.
        sim.add_port(position=(I_PORT * DX, J_PORT * DX, Z_G + 2 * DX),
                     component="ez", impedance=Z0, excite=True,
                     waveform=pulse)
    elif port == "msl":
        # A 2 mm trace on the laminate face, fed from the -x end.
        sim.add_pinned_sheet(plane_index=K_P, i_range=(2, NX - 2),
                             j_range=(8, 12), name="trace")
        sim.add_msl_port(position=(4 * DX, 10 * DX, Z_G), width=4 * DX,
                         height=Z_P - Z_G, direction="+x", impedance=Z0,
                         waveform=pulse)
    else:
        raise ValueError(port)
    if cap is not None:
        # across the port's top edge (the wire's third, the lumped port's own)
        sim.add_lumped_rlc((I_PORT * DX, J_PORT * DX, Z_G + 2 * DX), "ez",
                           C=float(cap), topology="parallel")
    sim.add_probe(PROBE, "ez")
    return sim


def _drawn(sim):
    """The materials as the geometry draws them -- what the runner assembles
    before any override."""
    grid = sim._build_nonuniform_grid()
    materials, *_ = assemble_materials_nu(sim, grid, sheet_specs=[],
                                          pec_sheets=[], pec_wires=[])
    return materials


@contextmanager
def _pre_override_drive(drawn):
    """Mutation (b): the pre-#1267 drive. Every builder call is kept; the
    materials a builder receives get eps_r and sigma back from the arrays as
    DRAWN, with the lumped stamps the runner has added since (port loads, a
    capacitor's fold) put back on top -- the copy the runner kept before the
    fix. With no override it is the materials the builder already gets.
    Wraps whatever builders are installed when it is entered."""
    import rfx.sources.msl_port as _msl
    inner_cs = _nu_runner.make_current_source
    inner_msl = _msl.make_msl_port_sources

    def _back(m):
        eps_l = getattr(m, "eps_r_lumped", None)
        sig_l = getattr(m, "sigma_lumped", None)
        return m._replace(
            eps_r=drawn.eps_r if eps_l is None else drawn.eps_r + eps_l,
            sigma=drawn.sigma if sig_l is None else drawn.sigma + sig_l)

    def _cs(grid, ijk, comp, wf, n, materials, *a, **kw):
        return inner_cs(grid, ijk, comp, wf, n, _back(materials), *a, **kw)

    def _msl_src(grid, port, materials, *a, **kw):
        return inner_msl(grid, port, _back(materials), *a, **kw)

    with mock.patch.object(_nu_runner, "make_current_source", _cs), \
            mock.patch.object(_msl, "make_msl_port_sources", _msl_src):
        yield


def _dft(ts, dt, f):
    """Continuous-time Fourier integral of the probe record at ``f``."""
    x = jnp.asarray(ts).reshape(ts.shape[0], -1)[:, 0]
    t = jnp.arange(x.shape[0]) * dt
    return jnp.sum(x * jnp.exp(-2j * jnp.pi * f * t)) * dt


def _n_record():
    return int(round(T_RECORD / float(
        _board(EPS_LO, EPS_HI)._build_nonuniform_grid().dt)))


@functools.lru_cache(maxsize=None)
def _solve(case, mutated=False):
    """``(S11 at FREQS, probe record, dt)`` for one full-record run.

    ``case`` names a (board, override) pair:
      "layered"      the two layers as materials
      "layered_ovr"  a 3.38 slab, the layered array by override
      "raised"       a one-material substrate built with 3.38 x e^0.1
      "raised_ovr"   a 3.38 slab, that array by override
    """
    ovr_from = {"layered_ovr": (EPS_LO, EPS_HI), "raised_ovr": RAISED}
    drawn_eps = {"layered": (EPS_LO, EPS_HI), "layered_ovr": (EPS_LO, EPS_LO),
                 "raised": RAISED, "raised_ovr": (EPS_LO, EPS_LO)}[case]
    sim = _board(*drawn_eps)
    kw = dict(n_steps=_n_record(), compute_s_params=True, s_param_freqs=FREQS)
    if case in ovr_from:
        kw["eps_override"] = _drawn(_board(*ovr_from[case])).eps_r
    ctx = _pre_override_drive(_drawn(sim)) if mutated else nullcontext()
    with ctx:
        r = run_nonuniform_path(sim, **kw)
    return (np.asarray(r.s_params).reshape(-1), np.asarray(r.time_series),
            float(r.dt))


# --------------------------------------------------------------------------
# 1. the same board, declared by override and by materials
# --------------------------------------------------------------------------

def _override_vs_materials(mutated):
    s_m, ts_m, _ = _solve("layered")
    s_o, ts_o, _ = _solve("layered_ovr", mutated)
    ds = np.abs(s_o - s_m)
    dts = float(np.max(np.abs(ts_o - ts_m)) / np.max(np.abs(ts_m)))
    return ds, dts, s_m, s_o


def test_the_layers_declared_by_override_and_by_materials_are_one_board():
    # realized, not declared: the column under the wire is 3.38/3.38/10.2
    col = np.asarray(_drawn(_board(EPS_LO, EPS_HI)).eps_r)[
        I_PORT + CPML, J_PORT + CPML, K_G + CPML:K_P + CPML]
    np.testing.assert_allclose(col, [EPS_LO, EPS_LO, EPS_HI], rtol=1e-6)
    ds, dts, s_m, s_o = _override_vs_materials(mutated=False)
    print(f"[override==materials] |dS11| at {FREQS/1e9} GHz = {ds}; "
          f"probe max|dE|/max|E| = {dts:.2e}; S11 {20*np.log10(abs(s_o))} dB")
    assert np.all(ds <= S11_ATOL), ds
    assert dts <= FIELD_RTOL, dts


# --------------------------------------------------------------------------
# 2. whole-substrate eps_r x e^0.1: override vs built
# --------------------------------------------------------------------------

class _Built(Exception):
    pass


def _built_up_to_the_scan(sim, eps_override=None, mutate_from=None):
    """Run the graded build up to the scan; return
    ``[(cell, component, drive materials)], stepper materials, dt``."""
    import rfx.sources.msl_port as _msl
    handed = []
    real_cs = _nu_runner.make_current_source
    real_msl = _msl.make_msl_port_sources
    stepped = {}

    def _cs(grid, ijk, comp, wf, n, materials, *a, **kw):
        out = real_cs(grid, ijk, comp, wf, n, materials, *a, **kw)
        handed.append(((out[0], out[1], out[2]), out[3], materials))
        return out

    def _msl_src(grid, port, materials, *a, **kw):
        out = real_msl(grid, port, materials, *a, **kw)
        handed.extend(((s[0], s[1], s[2]), s[3], materials) for s in out)
        return out

    def _scan(grid, materials, *a, **kw):
        stepped["materials"] = materials
        stepped["dt"] = float(grid.dt)
        raise _Built

    # the mutation is entered LAST, so it wraps the recorder and the
    # recorder sees what the real builder receives
    with mock.patch.object(_nu_runner, "make_current_source", _cs), \
            mock.patch.object(_msl, "make_msl_port_sources", _msl_src), \
            mock.patch.object(_nu_runner, "run_nonuniform", _scan):
        ctx = (_pre_override_drive(mutate_from) if mutate_from is not None
               else nullcontext())
        with ctx, pytest.raises(_Built):
            run_nonuniform_path(sim, n_steps=8, compute_s_params=False,
                                eps_override=eps_override)
    return handed, stepped["materials"], stepped["dt"]


_WIRE_EDGES = [(I_PORT + CPML, J_PORT + CPML, k + CPML)
               for k in range(K_G, K_P)]


def _kappa_closed_form():
    """Cb_drawn / Cb_override on the wire's three edges, each carrying its
    50 ohm load: what a drive built from the drawn 3.38 slab over-drives
    the raised board by. One number on all three edges (one material)."""
    _, m_drawn, dt = _built_up_to_the_scan(_board(EPS_LO, EPS_LO))
    _, m_raised, dt2 = _built_up_to_the_scan(_board(*RAISED))
    assert dt == dt2
    return np.array([
        float(cell_component_e_coeffs(m_drawn, c, "ez", dt)[1])
        / float(cell_component_e_coeffs(m_raised, c, "ez", dt)[1])
        for c in _WIRE_EDGES])


def _dlnkappa_closed_form():
    """d ln(Cb_drawn/Cb_override) / d ln eps_r at the drawn board, on the
    wire's edges: eps / (eps + sigma*dt/2) with the port's own load in
    sigma. A drive from the drawn arrays adds twice this to d ln|E|^2."""
    _, m, dt = _built_up_to_the_scan(_board(EPS_LO, EPS_LO))
    out = []
    for c in _WIRE_EDGES:
        eps_r, sigma = cell_component_e_materials(m, c, "ez")
        eps = float(eps_r) * EPS_0
        out.append(eps / (eps + float(sigma) * dt / 2.0))
    return np.array(out)


def _raised_field_ratio(mutated):
    _, ts_b, dt = _solve("raised")
    _, ts_o, _ = _solve("raised_ovr", mutated)
    return complex(_dft(ts_o, dt, F_BIN) / _dft(ts_b, dt, F_BIN))


def test_raising_eps_r_by_override_changes_the_board_not_the_drive():
    ratio = _raised_field_ratio(mutated=False)
    kappa = _kappa_closed_form()
    print(f"[kappa] probe field at {F_BIN/1e9:g} GHz, override/built = "
          f"{ratio:.8f}; a drive from the drawn arrays predicts "
          f"{kappa} (per wire edge)")
    assert abs(ratio - 1.0) <= KAPPA_RTOL, ratio


# --------------------------------------------------------------------------
# 3. the derivative through the override
# --------------------------------------------------------------------------

#: kind -> the drawn board's two layers. ``upper`` scales the eps_r 10.2
#: layer of the layered board; ``whole`` scales a one-material 3.38 slab.
_AD_BOARD = {"upper": (EPS_LO, EPS_HI), "whole": (EPS_LO, EPS_LO)}


def _mask(kind):
    eps0 = np.asarray(_drawn(_board(*_AD_BOARD[kind])).eps_r)
    if kind == "upper":
        m = np.zeros(eps0.shape)
        m[:, :, K_P - 1 + CPML] = 1.0
        return m * (eps0 > 5.0)          # the eps_r 10.2 cells
    return (eps0 > 2.0).astype(float)    # every substrate cell


def _observable(r, kind):
    if kind == "upper":                  # |S11|^2
        s = jnp.asarray(r.s_params).reshape(-1)[0]
        return jnp.abs(s) ** 2
    e = _dft(jnp.asarray(r.time_series), float(r.dt), float(F_AD[0]))
    return jnp.log(jnp.abs(e) ** 2)      # the probe's ln|E|^2


def _run_obs(sim, kind, eps_override=None):
    r = run_nonuniform_path(sim, n_steps=N_AD, compute_s_params=True,
                            s_param_freqs=F_AD, eps_override=eps_override,
                            checkpoint=True)
    return _observable(r, kind)


def _built_obs(kind, a):
    lo, hi = _AD_BOARD[kind]
    if kind == "upper":
        return float(_run_obs(_board(lo, hi * np.exp(a)), kind))
    return float(_run_obs(_board(lo * np.exp(a), hi * np.exp(a)), kind))


@functools.lru_cache(maxsize=None)
def _fd_built(kind):
    """Central differences of boards BUILT with eps_r x e^(+-h): the ladder
    and the Richardson values ``(4 D(h/2) - D(h)) / 3`` from its two pairs."""
    with enable_x64():
        d = {h: (_built_obs(kind, h) - _built_obs(kind, -h)) / (2 * h)
             for h in FD_LADDER}
    h0, h1, h2 = FD_LADDER
    rich_fine = (4 * d[h2] - d[h1]) / 3
    rich_coarse = (4 * d[h1] - d[h0]) / 3
    return d, rich_fine, rich_coarse


@functools.lru_cache(maxsize=None)
def _ad(kind, mutated=False):
    """``jax.grad`` at ln-scale 0 through ``eps_override``."""
    with enable_x64():
        sim = _board(*_AD_BOARD[kind])
        drawn = _drawn(sim)
        eps0 = jnp.asarray(drawn.eps_r)
        mask = jnp.asarray(_mask(kind))

        def f(a):
            return _run_obs(sim, kind, eps0 * jnp.exp(a * mask))

        ctx = _pre_override_drive(drawn) if mutated else nullcontext()
        with ctx:
            return float(jax.grad(f)(jnp.asarray(0.0)))


@pytest.mark.parametrize("kind", ["upper", "whole"])
def test_the_gradient_through_the_override_is_the_boards_gradient(kind):
    """``upper``: d|S11|^2/d(ln eps_r of the 10.2 layer) at 9 GHz -- the
    drive ratio differs between the wire's cells, so S11 itself carries it.
    ``whole``: d ln|E_probe|^2/d(ln eps_r of a one-material slab) -- one
    ratio on every port edge, which S11 cancels and the absolute field
    keeps. Measured: |AD/FD - 1| = 7.1e-5 and 2.9e-4 (before the fix
    1.0e-2 and 0.82, test 5)."""
    g = _ad(kind)
    d, rich, rich_coarse = _fd_built(kind)
    rel = abs(g - rich) / abs(rich)
    print(f"[grad {kind}] AD {g:.6f}; FD built {d}; Richardson {rich:.6f} "
          f"(coarse pair {rich_coarse:.6f}); |AD/FD - 1| = {rel:.2e}")
    # the ladder has converged well inside the gate
    assert abs(rich_coarse - rich) / abs(rich) < 5 * GRAD_RTOL
    assert rel <= GRAD_RTOL, (g, rich)


# --------------------------------------------------------------------------
# 4. no override: the pre-#1267 spelling and the shipped one, byte for byte
# --------------------------------------------------------------------------

def test_without_an_override_the_run_is_byte_identical():
    sim = _board(EPS_LO, EPS_HI)
    s_ship, ts_ship, _ = _solve("layered")
    with _pre_override_drive(_drawn(sim)):
        r = run_nonuniform_path(sim, n_steps=_n_record(),
                                compute_s_params=True, s_param_freqs=FREQS)
    s_pre = np.asarray(r.s_params).reshape(-1)
    ts_pre = np.asarray(r.time_series)
    assert s_ship.tobytes() == s_pre.tobytes()
    assert ts_ship.tobytes() == ts_pre.tobytes()


# --------------------------------------------------------------------------
# 5. mutation (b): the drive handed the pre-override arrays
# --------------------------------------------------------------------------

def test_mutation_the_drive_from_the_drawn_arrays_sends_1_to_3_red():
    ds, dts, s_m, s_o = _override_vs_materials(mutated=True)
    ratio = _raised_field_ratio(mutated=True)
    kappa = _kappa_closed_form()
    dlnk = _dlnkappa_closed_form()
    g_up, g_whole = _ad("upper", True), _ad("whole", True)
    fd_up, fd_whole = _fd_built("upper")[1], _fd_built("whole")[1]
    print(f"[mutation] 1: |dS11| {ds}, S11(override) "
          f"{20*np.log10(abs(s_o))} dB, probe {dts:.3f}; "
          f"2: ratio {ratio:.6f} vs closed form {kappa}; "
          f"3: AD upper {g_up:.6f} vs FD {fd_up:.6f}, "
          f"AD whole {g_whole:.6f} vs FD {fd_whole:.6f} "
          f"(excess {g_whole - fd_whole:.6f}, closed form 2*dln(kappa) "
          f"{2 * dlnk})")
    # 1
    assert np.max(ds) > 1e3 * S11_ATOL
    assert dts > 1e3 * FIELD_RTOL
    # 2, by the closed form: one ratio on every wire edge, and the system is
    # linear with the drive entering only through its per-step increment
    assert np.ptp(kappa) < 1e-6 * kappa[0]
    assert abs(ratio - 1.0) > 1e3 * KAPPA_RTOL
    assert abs(ratio / kappa[0] - 1.0) < 1e-5, (ratio, kappa)
    # 3, and the whole-slab excess is the closed form's 2 d ln(kappa)
    assert abs(g_up - fd_up) / abs(fd_up) > 3 * GRAD_RTOL     # 1.0e-2
    assert abs(g_whole - fd_whole) / abs(fd_whole) > 10 * GRAD_RTOL  # 0.82
    assert np.ptp(dlnk) < 1e-6 * dlnk[0]
    assert abs((g_whole - fd_whole) / (2 * dlnk[0]) - 1.0) < 1e-2


# --------------------------------------------------------------------------
# 6. build time, every port kind, under an override
# --------------------------------------------------------------------------

#: port kind -> (port, capacitor across its top edge). The drawn board is a
#: plain 3.38 slab; the override supplies the layered array, so the port's
#: top edge sits in eps_r 10.2 only through the override.
_BUILD_CASES = {
    "wire": ("wire", None),
    "lumped": ("lumped", None),
    "msl": ("msl", None),
    "wire+C": ("wire", 1e-12),
    "lumped+C": ("lumped", 1e-12),
}


def _worst_cb_mismatch(case, *, mutated=False):
    port, cap = _BUILD_CASES[case]
    sim = _board(EPS_LO, EPS_LO, port=port, cap=cap)
    ovr = _drawn(_board(EPS_LO, EPS_HI, port=port, cap=cap)).eps_r
    handed, stepper, dt = _built_up_to_the_scan(
        sim, eps_override=ovr, mutate_from=_drawn(sim) if mutated else None)
    assert handed, f"no drive was built for the {case} port"
    if cap is not None:
        lumped = np.asarray(stepper.eps_r_lumped)
        on = {tuple(int(v) for v in c) for c in np.argwhere(lumped != 0)}
        driven = {tuple(int(v) for v in cell) for cell, _, _ in handed}
        assert len(on) == 1 and on <= driven, (on, driven)
    # realized: some driven cell is eps_r 10.2 by the override only
    drawn_eps, ovr_eps = np.asarray(_drawn(sim).eps_r), np.asarray(ovr)
    assert any(ovr_eps[c] == np.float32(EPS_HI) and drawn_eps[c] ==
               np.float32(EPS_LO) for c, _, _ in handed)
    worst = 0.0
    for cell, comp, drive in handed:
        cb_drive = float(cell_component_e_coeffs(drive, cell, comp, dt)[1])
        cb_step = float(cell_component_e_coeffs(stepper, cell, comp, dt)[1])
        worst = max(worst, abs(cb_drive / cb_step - 1.0))
    return worst, len(handed)


@pytest.mark.parametrize("case", list(_BUILD_CASES))
def test_every_graded_port_drive_under_an_override_uses_the_steppers_cb(case):
    worst, n = _worst_cb_mismatch(case)
    print(f"[build] {case}: {n} driven edges, "
          f"max |Cb_drive/Cb_step - 1| = {worst:.3e}")
    assert worst <= 1e-6, worst


@pytest.mark.parametrize("case", list(_BUILD_CASES))
def test_the_build_check_sees_a_drive_from_the_drawn_arrays(case):
    worst, _ = _worst_cb_mismatch(case, mutated=True)
    print(f"[build-mutation] {case}: max |Cb_drive/Cb_step - 1| = {worst:.3e}")
    assert worst > 1e-2


def test_a_traced_override_drives_the_msl_feed_and_keeps_its_fixture():
    """The MSL feed's Cb follows a traced override (the drive table is
    traced), while the launch fixture -- the static-Laplace mode shape,
    built from the substrate eps_r at the feed's centre cell -- is still
    computed from the drawn array, on the host."""
    import rfx.sources.msl_port as _msl
    sim = _board(EPS_LO, EPS_LO, port="msl")
    eps0 = jnp.asarray(_drawn(sim).eps_r)
    mask = jnp.asarray(_mask("whole"))
    seen = []
    real_profile = _msl.compute_msl_mode_profile

    def _profile(grid, port, eps_r_sub, *a, **kw):
        seen.append(eps_r_sub)
        return real_profile(grid, port, eps_r_sub, *a, **kw)

    def f(a):
        r = run_nonuniform_path(sim, n_steps=300, compute_s_params=False,
                                eps_override=eps0 * jnp.exp(a * mask))
        return jnp.sum(jnp.asarray(r.time_series) ** 2)

    with mock.patch.object(_msl, "compute_msl_mode_profile", _profile):
        g = float(jax.grad(f)(jnp.asarray(0.0)))
    assert np.isfinite(g) and g != 0.0
    assert seen and all(type(v) is float and v == pytest.approx(EPS_LO)
                        for v in seen), seen
