"""``forward()`` under ``jax.jit``: set-up is concrete, time stepping is recorded (#1367, #1354, #1364).

The rule, whatever the port or monitor: set-up that reads only the
declaration is evaluated while the caller's function is traced, and the
time-stepping loop is compiled into the caller's program. Before it, #1229
evaluated set-up at trace time for a whitelist of ports (wire ports, loaded
ports on the graded lane) and recorded it for every other model, so a host
read of declaration-only data failed under ``jax.jit``:

* an MSL port's geometry check reads the realized PEC edge mask
  (``TracerArrayConversionError`` at ``rfx/sources/msl_port.py``, #1367);
* ``compute_far_field`` reads the NTFF box's frequencies (#1364; the
  transform on its own is pinned in ``tests/unit/farfield/test_farfield_jit_dispatch.py``).

And for a whitelisted model with no traced input, the time stepping itself ran
while tracing: JAX 0.6.2 (the CI build) raised ``Evaluation rule for 'empty'
not implemented`` (#1354), and 0.10.2 ran the solve uncompiled.

One tiny model per host-read class (wire port, MSL port, graded lumped port,
NTFF box with a far-field loss), on the uniform and graded lanes, under
``jax.jit(jax.value_and_grad)`` with a design input and ``jax.jit`` with no
traced input. Each must trace and agree with the plain call under the PI's
cross-trace rule: the per-step probe record within 9 float32 ULP at its peak,
quantities summed over the record (objective, gradient, far field, S) within
1e-4 of their peak. The time-stepping scan must appear in the jaxpr of a
function with no traced input, for every model. There is no whitelist, so a
port type added later is covered by construction.

Two of the boards are open (CPML, a lumped port per lane) and run 120 steps.
When the loop's operands reached it as constants of the caller's program,
XLA rewrote the loop around their values: the PEC boards at 40 steps still
agreed with the plain call to 3 ULP, while the open boards read 26-33 ULP at
120 steps (up to 128 at 200). ``recorded_scan`` hands the loop operands the
compiler cannot read, as a plain call does, and the jitted record equals
the plain call's.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation

MM = 1e-3
N_STEPS = 40
N_STEPS_OPEN = 120           # the open (CPML) boards: long enough to show drift
MAX_ULP_AT_PEAK = 9          # per-step quantities (PI 2026-09-23)
MAX_REL_SUMMED = 1.0e-4      # quantities summed over the record (PI 2026-09-25)
# Graded z: 0.5 mm cells through the ground and the substrate (z 0-3 mm),
# then a 1.25x growth into the air.
DZ = np.concatenate([np.full(6, 0.5), 0.5 * 1.25 ** np.arange(1, 6)]) * MM
DESIGN_BOX = ((9 * MM, 3 * MM, 1.5 * MM), (11 * MM, 5 * MM, 2.5 * MM))
THETA = np.linspace(0.2, 2.9, 4)
PHI = np.array([0.0, 1.5])


def _ulp_at_peak(plain, other):
    """``max|plain - other|`` in float32 spacings at ``max|plain|``."""
    plain = np.asarray(plain, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    return float(np.max(np.abs(plain - other))
                 / np.spacing(np.float32(np.max(np.abs(plain)))))


def _rel_at_peak(plain, other):
    """``max|plain - other| / max|plain|``."""
    plain = np.asarray(plain).astype(np.complex128)
    other = np.asarray(other).astype(np.complex128)
    return float(np.max(np.abs(plain - other)) / np.max(np.abs(plain)))


def _pulse():
    return GaussianPulse(f0=8e9, bandwidth=0.8)


def _board(graded, domain_xy=(16, 12), height=8, boundary="pec"):
    """eps_r 3 substrate (z 1-3 mm) on a PEC ground (z 0-1 mm).

    Closed PEC walls by default: the subject is tracing, and absorbers would
    double the compile time of every case. ``boundary="cpml"`` (4 layers)
    for the open boards.
    """
    extra = {"dz_profile": DZ} if graded else {}
    if boundary == "cpml":
        extra["cpml_layers"] = 4
    height_m = float(DZ.sum()) if graded else height * MM
    sim = Simulation(freq_max=16e9,
                     domain=(domain_xy[0] * MM, domain_xy[1] * MM, height_m),
                     dx=1.0 * MM, boundary=boundary, **extra)
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box((2 * MM, 2 * MM, 1 * MM), (14 * MM, 10 * MM, 3 * MM)),
            material="sub")
    sim.add(Box((2 * MM, 2 * MM, 0), (14 * MM, 10 * MM, 1 * MM)),
            material="pec")
    return sim


def _wire(graded):
    sim = _board(graded)
    sim.add_port(position=(6 * MM, 6 * MM, 1 * MM), component="ez",
                 extent=2 * MM, impedance=50.0, direction="-x",
                 waveform=_pulse())
    sim.add_probe((10 * MM, 6 * MM, 2 * MM), "ez")
    return sim


def _msl(graded):
    sim = _board(graded, domain_xy=(24, 12))
    sim.add(Box((2 * MM, 5 * MM, 3 * MM), (22 * MM, 7 * MM, 4 * MM)),
            material="pec")
    sim.add_msl_port(position=(5 * MM, 6 * MM, 1 * MM), width=2 * MM,
                     height=2 * MM, direction="+x", impedance=50.0,
                     waveform=_pulse())
    sim.add_probe((15 * MM, 6 * MM, 2 * MM), "ez")
    return sim


def _lumped(graded, boundary="pec"):
    sim = _board(graded, boundary=boundary)
    sim.add_port(position=(6 * MM, 6 * MM, 2 * MM), component="ez",
                 impedance=50.0, waveform=_pulse())
    sim.add_probe((10 * MM, 6 * MM, 2 * MM), "ez")
    return sim


def _ntff(graded):
    sim = _board(graded, domain_xy=(16, 16), height=12)
    sim.add_source((8 * MM, 8 * MM, 2 * MM), "ez", waveform=_pulse(),
                   amplitude_kind="field")
    sim.add_probe((10 * MM, 6 * MM, 2 * MM), "ez")
    # numpy float64 frequencies: the natural input, and the one that failed.
    sim.add_ntff_box((1 * MM, 1 * MM, 1 * MM), (15 * MM, 15 * MM, 6 * MM),
                     freqs=np.array([7e9, 8e9]))
    return sim


def _lumped_open(graded):
    return _lumped(graded, boundary="cpml")


# One model per host-read class, on the lanes that carry it, and a lumped
# port on an open board per lane:
# name -> (builder, graded, summed observable, design input, steps).
# "s": the port's S (uniform: port_s11_freqs); "far field": compute_far_field
# of the NTFF box; None: the probe record only. The design input is a
# design-box permittivity, or a whole-grid eps_override where the lane
# refuses a design box (an MSL port on the graded lane).
MODELS = {
    "wire port/uniform": (_wire, False, "s", "box", N_STEPS),
    "wire port/graded": (_wire, True, "s", "box", N_STEPS),
    "msl port/uniform": (_msl, False, None, "box", N_STEPS),
    "msl port/graded": (_msl, True, None, "eps", N_STEPS),
    "lumped port/graded": (_lumped, True, None, "box", N_STEPS),
    "ntff + compute_far_field/uniform": (_ntff, False, "far field", "box", N_STEPS),
    "ntff + compute_far_field/graded": (_ntff, True, "far field", "box", N_STEPS),
    "lumped port, cpml/uniform": (_lumped_open, False, "s", "box", N_STEPS_OPEN),
    "lumped port, cpml/graded": (_lumped_open, True, None, "box", N_STEPS_OPEN),
}


def _grid(sim, graded):
    return sim._build_nonuniform_grid() if graded else sim._build_grid()


def _design_shape(sim, graded, design):
    grid = _grid(sim, graded)
    if design == "eps":
        return tuple(grid.shape)
    if graded:
        from rfx.nonuniform import position_to_index
        lo = position_to_index(grid, DESIGN_BOX[0])
        hi = position_to_index(grid, DESIGN_BOX[1])
    else:
        lo = grid.position_to_index(DESIGN_BOX[0])
        hi = grid.position_to_index(DESIGN_BOX[1])
    return tuple(int(hi[d]) - int(lo[d]) + 1 for d in range(3))


def _run(sim, graded, observable, design, n_steps, eps):
    """``forward()`` -> (probe record, the summed observable or None)."""
    kw = {}
    if eps is not None and design == "eps":
        kw["eps_override"] = eps
    elif eps is not None:
        kw = {"design_box": DESIGN_BOX, "design_eps_override": eps}
    if observable == "s" and not graded:
        # numpy, so a function with no traced input really has none
        kw["port_s11_freqs"] = np.array([6e9, 8e9], np.float32)
    r = sim.forward(n_steps=n_steps, skip_preflight=True, checkpoint=False, **kw)
    summed = None
    if observable == "far field":
        from rfx.farfield import compute_far_field
        ff = compute_far_field(r.ntff_data, r.ntff_box, _grid(sim, graded),
                               THETA, PHI)
        summed = jnp.stack([ff.E_theta, ff.E_phi])
    elif observable == "s":
        summed = r.s_params
    return r.time_series, summed


def _loss(sim, graded, observable, design, n_steps):
    """A smooth O(1) objective of the record and the summed observable.

    Each part is divided by its own peak (held constant for the gradient):
    the lanes' records span 1e-4 to 1e5 V/m at 40 steps, and the far field
    of a 40-step record is far below 1 V/m.
    """
    def _normalized_power(a):
        a = jnp.asarray(a)
        peak = jax.lax.stop_gradient(jnp.max(jnp.abs(a)))
        return jnp.mean(jnp.abs(a / peak) ** 2)

    def loss(eps):
        ts, summed = _run(sim, graded, observable, design, n_steps, eps)
        out = _normalized_power(ts)
        if summed is not None:
            out = out + _normalized_power(summed)
        return out
    return loss


@functools.lru_cache(maxsize=None)
def _model(name):
    build, graded, observable, design, n_steps = MODELS[name]
    return build(graded), graded, observable, design, n_steps


# The open boards are left out here: their subject is the per-step record,
# and their jitted gradient costs 20 s each.
GRAD_MODELS = sorted(name for name, m in MODELS.items() if m[4] == N_STEPS)


@pytest.mark.parametrize("name", GRAD_MODELS)
def test_jit_of_the_gradient_equals_the_plain_call(name):
    """``jax.jit(jax.value_and_grad(loss))`` with a design input traces and
    agrees with the plain call (the objective and its gradient are sums over
    the record: 1e-4 of the peak)."""
    sim, graded, observable, design, n_steps = _model(name)
    loss = _loss(sim, graded, observable, design, n_steps)
    eps = jnp.asarray(2.0 + np.random.default_rng(3).random(
        _design_shape(sim, graded, design)), jnp.float32)

    v, g = jax.value_and_grad(loss)(eps)
    vj, gj = jax.jit(jax.value_and_grad(loss))(eps)
    assert np.isfinite(float(v)) and float(jnp.max(jnp.abs(g))) > 0.0
    assert np.all(np.isfinite(np.asarray(g)))
    assert _rel_at_peak(v, vj) <= MAX_REL_SUMMED, (float(v), float(vj))
    assert _rel_at_peak(g, gj) <= MAX_REL_SUMMED, _rel_at_peak(g, gj)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_jit_with_no_traced_input_equals_the_plain_call(name):
    """``jax.jit`` of a function with no traced input compiles the solve (the
    #1354 case) and agrees with the plain call: the probe record within 9 ULP
    at its peak, the far field or S within 1e-4 of its peak."""
    sim, graded, observable, design, n_steps = _model(name)

    def solve():
        return _run(sim, graded, observable, design, n_steps, None)

    ts, summed = solve()
    ts_j, summed_j = jax.jit(solve)()
    assert float(np.max(np.abs(np.asarray(ts)))) > 0.0
    assert _ulp_at_peak(ts, ts_j) <= MAX_ULP_AT_PEAK, _ulp_at_peak(ts, ts_j)
    if summed is not None:
        assert _rel_at_peak(summed, summed_j) <= MAX_REL_SUMMED


def _scan_lengths(jaxpr):
    """Lengths of every ``scan`` in a (closed) jaxpr, sub-jaxprs included."""
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "scan":
            found.append(int(eqn.params["length"]))
        for value in eqn.params.values():
            for sub in (value if isinstance(value, (tuple, list)) else (value,)):
                if hasattr(sub, "eqns") or hasattr(getattr(sub, "jaxpr", None), "eqns"):
                    found.extend(_scan_lengths(sub))
    return found


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_time_stepping_is_recorded_not_run_while_tracing(name):
    """With no traced input, the jaxpr holds the record-long scan, for every model.

    Before the fix a whitelisted model (a wire port) evaluated the whole
    solve inside the trace-time region: JAX 0.10.2 returned its result as a
    constant, with no scan in the program, and a record that agreed with
    the plain call to 0 ULP because it WAS the plain call. So the value
    tests above cannot see this; a scan of the record's length in the jaxpr
    is what "the time stepping is recorded" means.
    """
    sim, graded, observable, design, n_steps = _model(name)
    jaxpr = jax.make_jaxpr(
        lambda: _run(sim, graded, observable, design, n_steps, None))()
    assert n_steps in _scan_lengths(jaxpr), _scan_lengths(jaxpr)


def test_the_adi_time_stepping_is_recorded_not_run_while_tracing():
    """The ADI lane's scan entry obeys the same rule (a PEC box, point source):
    the scan is in the jaxpr, and the jitted record agrees with the plain
    call (it differed by 9-10 ULP at 40 steps before its loop's operands
    were hidden from the compiler)."""
    sim = Simulation(freq_max=10e9, domain=(12 * MM, 10 * MM, 6 * MM),
                     dx=1.0 * MM, boundary="pec", solver="adi")
    sim.add_source((6 * MM, 5 * MM, 3 * MM), "ez", waveform=_pulse(),
                   amplitude_kind="field")
    sim.add_probe((8 * MM, 5 * MM, 3 * MM), "ez")

    def solve():
        return sim.forward(n_steps=N_STEPS, skip_preflight=True).time_series

    assert N_STEPS in _scan_lengths(jax.make_jaxpr(solve)())
    ts = np.asarray(solve())
    ts_j = np.asarray(jax.jit(solve)())
    assert ts_j.shape == (N_STEPS, 1) and float(np.max(np.abs(ts))) > 0.0
    assert _ulp_at_peak(ts, ts_j) <= MAX_ULP_AT_PEAK, _ulp_at_peak(ts, ts_j)


def _msl_direct_call(graded):
    """The MSL board's solve through the shared entry ``forward()`` calls,
    called directly, as ``compute_mixed_s_matrix`` and the S-matrix driver
    do (they call it without an outer trace)."""
    sim, *_ = _model(f"msl port/{'graded' if graded else 'uniform'}")
    if graded:
        def solve():
            return sim._forward_nonuniform_from_materials(
                n_steps=N_STEPS, checkpoint=False).time_series
        return solve
    grid = sim._build_grid()
    sheets, wires = [], []
    materials, debye, lorentz, pec_mask, *_ = sim._assemble_materials(
        grid, pec_sheets=sheets, pec_wires=wires)

    def solve():
        return sim._forward_from_materials(
            grid, materials, debye, lorentz, n_steps=N_STEPS,
            checkpoint=False, pec_mask=pec_mask, pec_sheets=tuple(sheets),
            pec_wires=tuple(wires)).time_series
    return solve


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_a_staged_direct_call_of_the_shared_entry_obeys_the_rule(graded):
    """The rule sits on the entries ``forward()`` shares with the calculators
    (``_forward_from_materials``, ``_forward_nonuniform_from_materials``), not
    only on ``forward()``: staged directly under ``jax.jit``, the MSL board's
    host read of its PEC edge mask is concrete, the scan is recorded, and
    the record equals the plain call's. (The calculators themselves run it
    without an outer trace, so for them the rule is inactive.)"""
    solve = _msl_direct_call(graded)
    assert N_STEPS in _scan_lengths(jax.make_jaxpr(solve)())
    ts = solve()
    ts_j = jax.jit(solve)()
    assert _ulp_at_peak(ts, ts_j) <= MAX_ULP_AT_PEAK, _ulp_at_peak(ts, ts_j)


# ---------------------------------------------------------------------------
# The helper's mechanisms (rfx.core.jax_utils)
# ---------------------------------------------------------------------------

def test_the_compile_time_switch_is_found_on_the_ci_jax_versions():
    """The probe finds JAX's own switch on the versions CI runs (0.6.2 and
    0.10.2), so the gate above exercises it; a JAX that drops it falls back
    to the anchor, which the next test exercises (and which the 0.4.20
    floor runs for real)."""
    from rfx.core.jax_utils import compile_time_switch

    major_minor = tuple(int(p) for p in jax.__version__.split(".")[:2])
    if major_minor < (0, 6):
        pytest.skip(f"JAX {jax.__version__} predates the CI builds")
    assert compile_time_switch() is not None


def test_the_fallback_records_the_time_stepping_over_opaque_operands(monkeypatch):
    """Without JAX's switch (the 0.4.20 floor), ``recorded_scan`` lifts the
    loop's concrete operands into the outer trace through the barrier and a
    tracer made before the region: the open board's solve with no traced
    input is still compiled (scan in the jaxpr; on 0.6.2 no 'empty' error)
    and agrees with the plain call."""
    from rfx.core import jax_utils

    monkeypatch.setattr(jax_utils, "compile_time_switch", lambda: None)
    sim, graded, observable, design, n_steps = _model("lumped port, cpml/uniform")

    def solve():
        return _run(sim, graded, observable, design, n_steps, None)

    assert n_steps in _scan_lengths(jax.make_jaxpr(solve)())
    ts, summed = solve()
    ts_j, summed_j = jax.jit(solve)()
    assert _ulp_at_peak(ts, ts_j) <= MAX_ULP_AT_PEAK, _ulp_at_peak(ts, ts_j)
    assert _rel_at_peak(summed, summed_j) <= MAX_REL_SUMMED


def test_recorded_scan_outside_a_compile_time_region_is_the_plain_scan():
    """No region, no anchor: the helper is ``jax.lax.scan``, eager and traced."""
    from rfx.core.jax_utils import recorded_scan

    def body(c, x):
        return c * 0.5 + x, c

    xs = jnp.arange(5.0, dtype=jnp.float32)
    plain = jax.lax.scan(body, jnp.float32(1.0), xs)
    mine = recorded_scan(body, jnp.float32(1.0), xs)
    assert not isinstance(mine[0], jax.core.Tracer)
    for a, b in zip(jax.tree_util.tree_leaves(plain), jax.tree_util.tree_leaves(mine)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert jax.make_jaxpr(lambda: recorded_scan(body, jnp.float32(1.0), xs))(
    ).jaxpr.eqns[-1].primitive.name == "scan"


def _toy_region_scan(xs):
    """A scan with a closed-over array, run through ``recorded_scan`` inside
    the region, as a forward entry does under an outer trace."""
    from rfx.core.jax_utils import declaration_setup, recorded_scan

    weights = jnp.linspace(0.5, 1.5, 7, dtype=jnp.float32)

    def body(c, x):
        return c * weights + x, jnp.sum(c)

    def solve():
        with declaration_setup():
            return recorded_scan(body, jnp.zeros(7, jnp.float32), xs)
    return body, solve


@pytest.mark.parametrize("fallback", [False, True], ids=["switch", "fallback"])
def test_recorded_scan_in_the_region_hides_every_concrete_operand(
        fallback, monkeypatch):
    """In the region every operand of the recorded loop (the arrays its body
    closes over, the carry, ``xs``) is an output of an
    ``optimization_barrier``, so the compiler cannot rewrite the loop around
    known values, and the result is the plain scan's."""
    from rfx.core import jax_utils

    if fallback:
        monkeypatch.setattr(jax_utils, "compile_time_switch", lambda: None)
    xs = jnp.ones((5, 7), jnp.float32)
    body, solve = _toy_region_scan(xs)
    jaxpr = jax.make_jaxpr(solve)().jaxpr
    scans = [e for e in jaxpr.eqns if e.primitive.name == "scan"]
    hidden = {v for e in jaxpr.eqns
              if e.primitive.name == "optimization_barrier" for v in e.outvars}
    assert len(scans) == 1 and len(scans[0].invars) == 3  # weights, carry, xs
    assert all(v in hidden for v in scans[0].invars), jaxpr
    plain = jax.lax.scan(body, jnp.zeros(7, jnp.float32), xs)
    for a, b in zip(jax.tree_util.tree_leaves(plain),
                    jax.tree_util.tree_leaves(jax.jit(solve)())):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_recorded_scan_in_the_region_of_zero_length():
    """A zero-length record (nothing to step) still returns the plain scan's
    empty outputs and the carry unchanged."""
    body, solve = _toy_region_scan(jnp.ones((0, 7), jnp.float32))
    carry, ys = jax.jit(solve)()
    assert ys.shape == (0,) and np.all(np.asarray(carry) == 0.0)
