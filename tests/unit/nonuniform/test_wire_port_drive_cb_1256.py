"""A port's drive on the graded-mesh lane sees the port's own load (#1256).

A 50 ohm wire port feeds a 6 x 5 mm patch through a 1.5 mm, eps_r 3.38
substrate on a 0.5 mm mesh. The port is a current source in parallel with its
50 ohm load, and both sit on the same three Ez edges: the load is a
conductance ``sigma_port = n * d / (Z0 * d_perp1 * d_perp2)`` = 120 S/m
stamped into the edges' update coefficient ``Cb = dt / (eps + sigma*dt/2)``,
and the drive is the declared current pushed through that same Cb. The graded
lane built the drive from a copy of the materials taken BEFORE the load was
stamped, so its Cb lacked sigma_port and the edges received
``1 + sigma_port*dt/(2*eps)`` times the declared current -- 2.911 at this
board's Courant step. The factor carries dt, so the absolute field per unit
drive changed with the time step (x0.850 for a step 0.772 times shorter), and
it carries the dual cell sizes at the feed through sigma_port. On this board
all three port cells see the same factor, so S11 -- a ratio of two responses
to the same drive -- did not move (4 GHz S11 changed by 1e-4 between the two
steps, before and after the fix). When the port's cells see DIFFERENT
factors the drive is also mis-shaped along the wire and S-parameters move:
a layered substrate or graded cells under the port, or a lumped R/C across
one of the port's edges, whose fold into Cb the drive also lacked (0.3 pF on
the middle edge here: |S11| read +0.143 dB at 2 GHz on the graded lane, a
passive board reflecting more than it receives, against -0.005 dB on the
uniform lane).

What is checked, at 4 GHz, below the patch resonance:

1. lane parity -- the graded lane on a uniform-valued mesh and the uniform
   lane, same board, same step, give the same probe field per unit drive
   (measured 1 - 2e-7 after the fix; 2.911 before);
2. dt invariance -- two pinned steps inside the Courant limit, the record
   held at the same length in ns: the probe field per unit drive agrees to
   1 % (measured |ratio - 1| = 2.5e-3; 0.850 before) and S11, the control, to
   0.1 % (measured 1.0e-4, before and after);
3. the defect restored with every helper call kept -- the drive handed the
   materials with the port's own stamp removed -- sends both red, by the
   factor the closed form predicts;
4. a build-time check on every port kind the graded lane drives (wire,
   single-cell lumped, MSL feed, and a wire or lumped port with a lumped
   capacitor across its edge): the Cb each drive is built from equals the
   Cb the stepper applies on that edge. It needs no time stepping;
5. end to end, a 0.3 pF capacitor across the middle edge of the wire port:
   the two lanes agree on |S11| at 2 and 4 GHz within 1e-4.

The two lanes read a port waveform in different units, and (1) converts
between them. The uniform lane adds ``Cb * w / (n * d_par)`` per step per live
cell (``rfx.simulation.make_wire_port_sources``); the graded lane adds
``Cb * w / (n * dV)`` with ``dV = d_par * d_perp1 * d_perp2``
(``rfx.nonuniform.make_current_source``, the current convention of
``rfx.api._source_semantics``). On the same board and step the graded field
is therefore the uniform one divided by ``d_perp1 * d_perp2`` = dx**2 here --
a geometric constant with no material and no dt in it, so it cannot absorb
the defect. That the same ``add_port(waveform=...)`` gives fields dx**2
apart on the two lanes is a pre-existing difference this test records, not
one it fixes.
"""

from __future__ import annotations

import functools
from unittest import mock

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.core.yee import cell_component_e_coeffs
from rfx.geometry import Box
from rfx.runners import nonuniform as _nu_runner
from rfx.runners.nonuniform import run_nonuniform_path
from rfx.sources import GaussianPulse

EPS_0 = 8.8541878128e-12

DX = 0.5e-3
NX, NY, NZ = 24, 20, 16
K_G, K_P = 5, 8                     # ground and patch node planes on z
N_SUB = K_P - K_G                   # 3 substrate cells = 1.5 mm
Z_G, Z_P = K_G * DX, K_P * DX
EPS_R = 3.38
Z0 = 50.0
I_PORT, J_PORT = 9, 10
PROBE = (20 * DX, J_PORT * DX, 0.5 * (Z_G + Z_P))
T_RECORD = 2.5e-9                   # record length, held in ns across steps
F_BIN = 4e9                         # below the patch resonance
DT_RATIO = 0.772                    # the issue's 0.59613 / 0.77228 ps

#: Closed form of the port's own conductance on this board: n * d_par /
#: (Z0 * d_perp1 * d_perp2), all three 0.5 mm.
SIGMA_PORT = N_SUB * DX / (Z0 * DX * DX)

# Pre-declared gates.
LANE_PARITY_RTOL = 1e-3
DT_FIELD_RTOL = 1e-2
DT_S11_RTOL = 1e-3


def _drive_factor(dt):
    """``1 + sigma_port*dt/(2*eps)``: how much too strong a drive built
    without the port's own load is, at the port edge of this board."""
    return 1.0 + SIGMA_PORT * dt / (2.0 * EPS_R * EPS_0)


#: The port edge a lumped capacitor is put across: the wire port's middle
#: Ez edge, and the single-cell lumped port's own edge.
RLC_POS = (I_PORT * DX, J_PORT * DX, Z_G + DX)


def _board(*, graded, dt=None, port="wire", cap=None):
    kw = {}
    if graded:
        # A uniform-VALUED profile: the graded lane on the same cells.
        kw["dz_profile"] = np.full(NZ, DX)
        if dt is not None:
            kw.update(dt=float(dt), dt_min_cell=DX)
    sim = Simulation(freq_max=30e9, domain=(NX * DX, NY * DX, NZ * DX),
                     dx=DX, cpml_layers=6, boundary="cpml", **kw)
    sim.add_material("sub", eps_r=EPS_R)
    sim.add(Box((2 * DX, 2 * DX, Z_G), ((NX - 2) * DX, (NY - 2) * DX, Z_P)),
            material="sub")
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
        # One Ez edge in the middle of the substrate, no extent.
        sim.add_port(position=(I_PORT * DX, J_PORT * DX, Z_G + DX),
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
        sim.add_lumped_rlc(RLC_POS, "ez", C=float(cap), topology="parallel")
    sim.add_probe(PROBE, "ez")
    return sim


def _lumped_total(record):
    """The cell-level sum of a lumped record: since #1236 a record is None or
    a 3-tuple of per-E-component arrays / None; the cell total is their sum
    (what ``materials.sigma`` / ``eps_r`` carry on top of the volume)."""
    if record is None:
        return None
    if not isinstance(record, (tuple, list)):
        return record
    parts = [r for r in record if r is not None]
    if not parts:
        return None
    total = parts[0]
    for r in parts[1:]:
        total = total + r
    return total


def _strip_lumped_sigma(materials):
    """The materials without any lumped conductance stamp -- what the drive
    was built from before the fix, on a board whose only lumped stamps are
    its ports."""
    lumped = _lumped_total(getattr(materials, "sigma_lumped", None))
    if lumped is None:
        return materials
    return materials._replace(sigma=materials.sigma - lumped,
                              sigma_lumped=None)


def _strip_lumped_eps(materials):
    """The materials without any lumped permittivity stamp -- a capacitor's
    fold. On the boards here the only one is the capacitor across the port
    edge, so this is the drive the runner built before the RLC fold reached
    it (port conductance kept)."""
    lumped = _lumped_total(getattr(materials, "eps_r_lumped", None))
    if lumped is None:
        return materials
    return materials._replace(eps_r=materials.eps_r - lumped,
                              eps_r_lumped=None)


_STRIP = {"sigma": _strip_lumped_sigma, "eps": _strip_lumped_eps}


def _unstamped_drive(strip="sigma"):
    """Mutation (b): the runner still calls ``make_current_source`` exactly
    as it does; only the materials it hands over lose the port's own load
    (``strip="sigma"``) or the capacitor's fold (``strip="eps"``)."""
    real = _nu_runner.make_current_source
    _cut = _STRIP[strip]

    def _pre_fix(grid, ijk, comp, wf, n, materials, *a, **kw):
        return real(grid, ijk, comp, wf, n, _cut(materials), *a, **kw)

    return mock.patch.object(_nu_runner, "make_current_source", _pre_fix)


def _dft_at_bin(ts, dt):
    """Continuous-time Fourier integral of the probe record at F_BIN,
    sampled at the run's own step, so records at two steps compare."""
    x = np.asarray(ts, dtype=np.float64).reshape(len(ts), -1)[:, 0]
    t = np.arange(len(x)) * dt
    return complex(np.sum(x * np.exp(-2j * np.pi * F_BIN * t)) * dt)


@functools.lru_cache(maxsize=None)
def _solve(lane, dt_scale=1.0, mutated=False):
    """``(dt, probe field at F_BIN, S11 at F_BIN)`` for one run.

    ``lane`` is "uniform" or "graded"; a graded run is pinned to
    ``dt_scale`` times the uniform lane's step, read off the uniform grid.
    """
    dt_uniform = float(_board(graded=False)._build_grid().dt)
    freqs = np.array([F_BIN])
    if lane == "uniform":
        assert dt_scale == 1.0 and not mutated
        sim = _board(graded=False)
        n = int(round(T_RECORD / dt_uniform))
        r = sim.run(n_steps=n, compute_s_params=True,
                    s_param_freqs=jnp.asarray(freqs), skip_preflight=True)
    else:
        sim = _board(graded=True, dt=dt_uniform * dt_scale)
        dt_grid = float(sim._build_nonuniform_grid().dt)
        n = int(round(T_RECORD / dt_grid))
        if mutated:
            with _unstamped_drive():
                r = run_nonuniform_path(sim, n_steps=n, compute_s_params=True,
                                        s_param_freqs=freqs)
        else:
            r = run_nonuniform_path(sim, n_steps=n, compute_s_params=True,
                                    s_param_freqs=freqs)
    dt = float(r.dt)                        # realized, not declared
    field = _dft_at_bin(r.time_series, dt)
    s11 = complex(np.asarray(r.s_params).reshape(-1)[0])
    return dt, field, s11


def _lane_ratio(mutated):
    dt_u, e_u, _ = _solve("uniform")
    dt_g, e_g, _ = _solve("graded", 1.0, mutated)
    assert dt_g == dt_u, f"the two lanes ran at {dt_g} and {dt_u} s"
    # graded field * d_perp1 * d_perp2 is in the uniform lane's units
    # (module docstring).
    return e_g * DX * DX / e_u, dt_u


def _dt_ratios(mutated):
    dt1, e1, s1 = _solve("graded", 1.0, mutated)
    dt2, e2, s2 = _solve("graded", DT_RATIO, mutated)
    assert abs(dt2 / dt1 - DT_RATIO) < 1e-6, (dt1, dt2)
    return e2 / e1, s2 / s1, dt1, dt2


# --------------------------------------------------------------------------
# 1. lane parity
# --------------------------------------------------------------------------

def test_graded_and_uniform_lanes_give_the_same_field_per_unit_drive():
    ratio, dt = _lane_ratio(mutated=False)
    print(f"[lane] dt = {dt:.6e} s  graded*dx^2/uniform at {F_BIN/1e9:g} GHz"
          f" = {ratio:.7f}")
    assert abs(ratio - 1.0) <= LANE_PARITY_RTOL, (
        f"graded field per unit drive is {ratio:.6f} x the uniform lane's; "
        f"a drive built without the port's own load reads "
        f"{_drive_factor(dt):.4f}")


# --------------------------------------------------------------------------
# 2. dt invariance, S11 as the control
# --------------------------------------------------------------------------

def test_field_per_unit_drive_does_not_depend_on_the_time_step():
    e_ratio, s_ratio, dt1, dt2 = _dt_ratios(mutated=False)
    print(f"[dt] {dt1:.6e} -> {dt2:.6e} s: field ratio {e_ratio:.6f}, "
          f"S11 ratio {s_ratio:.6f}")
    assert abs(s_ratio - 1.0) <= DT_S11_RTOL, (
        f"the control moved: S11({dt2:.3e})/S11({dt1:.3e}) = {s_ratio:.6f}")
    assert abs(e_ratio - 1.0) <= DT_FIELD_RTOL, (
        f"field per unit drive changed by {e_ratio:.6f} with the step; a "
        f"drive without the port's load predicts "
        f"{_drive_factor(dt2) / _drive_factor(dt1):.4f}")


# --------------------------------------------------------------------------
# 3. the defect restored, helper calls kept
# --------------------------------------------------------------------------

def test_mutation_the_drive_without_the_ports_load_sends_both_red():
    """Both checks above go red, by the closed-form factor: the lane ratio
    by ``1 + sigma_port*dt/(2 eps)`` exactly (the system is linear and the
    drive enters only through its per-step increment), the step ratio by the
    ratio of that factor at the two steps, within the 1 % the unmutated check
    allows."""
    lane, dt = _lane_ratio(mutated=True)
    want_lane = _drive_factor(dt)
    e_ratio, s_ratio, dt1, dt2 = _dt_ratios(mutated=True)
    want_dt = _drive_factor(dt2) / _drive_factor(dt1)
    print(f"[mutation] lane ratio {lane:.6f} (closed form {want_lane:.6f}); "
          f"step ratio {e_ratio:.6f} (closed form {want_dt:.6f}); "
          f"S11 ratio {s_ratio:.6f}")
    assert abs(lane - 1.0) > LANE_PARITY_RTOL
    assert abs(e_ratio - 1.0) > DT_FIELD_RTOL
    assert abs(lane / want_lane - 1.0) < 1e-4, (lane, want_lane)
    assert abs(e_ratio / want_dt - 1.0) < DT_FIELD_RTOL, (e_ratio, want_dt)
    # and the S-parameter hides it, which is how it went unseen
    assert abs(s_ratio - 1.0) <= DT_S11_RTOL


# --------------------------------------------------------------------------
# 4. build time, every port kind: drive Cb == stepper Cb on the driven edge
# --------------------------------------------------------------------------

class _Built(Exception):
    pass


def _drive_and_stepper_materials(sim, *, strip=None):
    """Build the graded run up to the scan and return
    ``[(cell, component, drive materials)], stepper materials, dt``.

    The recorder sits between the runner and each real drive builder and
    records what the builder RECEIVES. ``strip=True`` is mutation (b): the
    materials lose their lumped stamps on the way in -- the pre-fix drive --
    and every call is kept. ``strip`` names which stamps: "sigma" (the
    port's own load) or "eps" (a capacitor's fold).
    """
    handed = []
    import rfx.sources.msl_port as _msl
    real_cs = _nu_runner.make_current_source
    real_msl = _msl.make_msl_port_sources

    def _in(materials):
        return _STRIP[strip](materials) if strip else materials

    def _cs(grid, ijk, comp, wf, n, materials, *a, **kw):
        m = _in(materials)
        out = real_cs(grid, ijk, comp, wf, n, m, *a, **kw)
        handed.append(((out[0], out[1], out[2]), out[3], m))
        return out

    def _msl_src(grid, port, materials, *a, **kw):
        m = _in(materials)
        out = real_msl(grid, port, m, *a, **kw)
        handed.extend(((s[0], s[1], s[2]), s[3], m) for s in out)
        return out

    stepped = {}

    def _scan(grid, materials, *a, **kw):
        stepped["materials"] = materials
        stepped["dt"] = float(grid.dt)
        raise _Built

    with mock.patch.object(_nu_runner, "make_current_source", _cs), \
            mock.patch.object(_msl, "make_msl_port_sources", _msl_src), \
            mock.patch.object(_nu_runner, "run_nonuniform", _scan):
        with pytest.raises(_Built):
            run_nonuniform_path(sim, n_steps=8, compute_s_params=False)
    return handed, stepped["materials"], stepped["dt"]


#: port kind -> (board port, capacitor across the port edge, what mutation
#: (b) strips from the drive). 1 pF is large enough that a drive without the
#: fold is off by an order of magnitude.
_BUILD_CASES = {
    "wire": ("wire", None, "sigma"),
    "lumped": ("lumped", None, "sigma"),
    "msl": ("msl", None, "sigma"),
    "wire+C": ("wire", 1e-12, "eps"),
    "lumped+C": ("lumped", 1e-12, "eps"),
}


def _worst_cb_mismatch(case, *, mutated=False):
    port, cap, strip = _BUILD_CASES[case]
    handed, stepper, dt = _drive_and_stepper_materials(
        _board(graded=True, port=port, cap=cap),
        strip=strip if mutated else None)
    assert handed, f"no drive was built for the {case} port"
    if cap is not None:
        # realized, not declared: the capacitor sits on a driven edge
        lumped = np.asarray(_lumped_total(stepper.eps_r_lumped))
        on = {tuple(int(v) for v in c) for c in np.argwhere(lumped != 0)}
        driven = {tuple(int(v) for v in cell) for cell, _, _ in handed}
        assert len(on) == 1 and on <= driven, (on, driven)
    worst = 0.0
    for cell, comp, drive in handed:
        cb_drive = float(cell_component_e_coeffs(drive, cell, comp, dt)[1])
        cb_step = float(cell_component_e_coeffs(stepper, cell, comp, dt)[1])
        worst = max(worst, abs(cb_drive / cb_step - 1.0))
    return worst, len(handed)


@pytest.mark.parametrize("case", list(_BUILD_CASES))
def test_every_graded_port_drive_uses_the_steppers_cb(case):
    worst, n = _worst_cb_mismatch(case)
    print(f"[build] {case}: {n} driven edges, "
          f"max |Cb_drive/Cb_step - 1| = {worst:.3e}")
    assert worst <= 1e-6, (
        f"{case} port: a drive is built on a Cb {worst:.4f} off the "
        f"stepper's -- a lumped load on the driven edge is missing from it")


@pytest.mark.parametrize("case", list(_BUILD_CASES))
def test_the_build_check_sees_a_drive_without_the_load(case):
    """Mutation (b) of the build check: every drive builder is handed the
    materials without the port's load (or without the capacitor's fold),
    every call is kept, and the check above -- unchanged -- goes red."""
    worst, _ = _worst_cb_mismatch(case, mutated=True)
    print(f"[build-mutation] {case}: max |Cb_drive/Cb_step - 1| = {worst:.3e}")
    assert worst > 1e-2


# --------------------------------------------------------------------------
# 5. end to end: a capacitor across the port edge
# --------------------------------------------------------------------------

CAP_E2E = 0.3e-12
F_RLC = (2e9, 4e9)
RLC_LANE_ATOL = 1e-4


@functools.lru_cache(maxsize=None)
def _s11_with_cap(lane, mutated=False):
    """|S11| at F_RLC through the public ``run()``, 0.3 pF across the wire
    port's middle edge. The graded run is pinned to the uniform step."""
    dt_uniform = float(_board(graded=False)._build_grid().dt)
    n = int(round(T_RECORD / dt_uniform))
    freqs = jnp.asarray(F_RLC)
    if lane == "uniform":
        assert not mutated
        sim = _board(graded=False, cap=CAP_E2E)
    else:
        sim = _board(graded=True, dt=dt_uniform, cap=CAP_E2E)
    kw = dict(n_steps=n, compute_s_params=True, s_param_freqs=freqs,
              skip_preflight=True)
    if mutated:
        with _unstamped_drive("eps"):
            r = sim.run(**kw)
    else:
        r = sim.run(**kw)
    assert float(r.dt) == dt_uniform
    return np.abs(np.asarray(r.s_params).reshape(-1))


def test_lanes_agree_on_s11_with_a_capacitor_across_the_port_edge():
    """A lumped capacitor in parallel with the port's middle edge makes the
    three port cells unequal, so a drive that leaves the capacitor out of
    one cell's Cb is mis-shaped along the wire and S11 itself moves."""
    uni = _s11_with_cap("uniform")
    grd = _s11_with_cap("graded")
    diff = np.max(np.abs(grd - uni))
    print(f"[rlc] |S11| uniform {20*np.log10(uni)} dB, graded "
          f"{20*np.log10(grd)} dB, max |d|S11|| = {diff:.2e}")
    assert diff <= RLC_LANE_ATOL, (uni, grd)


def test_mutation_a_drive_without_the_capacitor_fold_breaks_s11_parity():
    uni = _s11_with_cap("uniform")
    grd = _s11_with_cap("graded", mutated=True)
    diff = np.max(np.abs(grd - uni))
    print(f"[rlc-mutation] |S11| uniform {20*np.log10(uni)} dB, graded "
          f"{20*np.log10(grd)} dB, max |d|S11|| = {diff:.2e}")
    assert diff > 10 * RLC_LANE_ATOL
