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
function with no traced input. There is no whitelist, so a port type added
later is covered by construction.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation

MM = 1e-3
BOUNDARY = "pec"
N_STEPS = 40
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


def _board(graded, domain_xy=(16, 12), height=8):
    """eps_r 3 substrate (z 1-3 mm) on a PEC ground (z 0-1 mm).

    Closed PEC walls: the subject is tracing, and absorbers would double the
    compile time of every case.
    """
    extra = {"dz_profile": DZ} if graded else {}
    height_m = float(DZ.sum()) if graded else height * MM
    sim = Simulation(freq_max=16e9,
                     domain=(domain_xy[0] * MM, domain_xy[1] * MM, height_m),
                     dx=1.0 * MM, boundary=BOUNDARY, **extra)
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


def _lumped(graded):
    sim = _board(graded)
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


# One model per host-read class, on the lanes that carry it:
# name -> (builder, graded, summed observable, design input).
# "s": the port's S (uniform: port_s11_freqs); "far field": compute_far_field
# of the NTFF box; None: the probe record only. The design input is a
# design-box permittivity, or a whole-grid eps_override where the lane
# refuses a design box (an MSL port on the graded lane).
MODELS = {
    "wire port/uniform": (_wire, False, "s", "box"),
    "wire port/graded": (_wire, True, "s", "box"),
    "msl port/uniform": (_msl, False, None, "box"),
    "msl port/graded": (_msl, True, None, "eps"),
    "lumped port/graded": (_lumped, True, None, "box"),
    "ntff + compute_far_field/uniform": (_ntff, False, "far field", "box"),
    "ntff + compute_far_field/graded": (_ntff, True, "far field", "box"),
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


def _run(sim, graded, observable, design, eps):
    """``forward()`` -> (probe record, the summed observable or None)."""
    kw = {}
    if eps is not None and design == "eps":
        kw["eps_override"] = eps
    elif eps is not None:
        kw = {"design_box": DESIGN_BOX, "design_eps_override": eps}
    if observable == "s" and not graded:
        # numpy, so a function with no traced input really has none
        kw["port_s11_freqs"] = np.array([6e9, 8e9], np.float32)
    r = sim.forward(n_steps=N_STEPS, skip_preflight=True, checkpoint=False, **kw)
    summed = None
    if observable == "far field":
        from rfx.farfield import compute_far_field
        ff = compute_far_field(r.ntff_data, r.ntff_box, _grid(sim, graded),
                               THETA, PHI)
        summed = jnp.stack([ff.E_theta, ff.E_phi])
    elif observable == "s":
        summed = r.s_params
    return r.time_series, summed


def _loss(sim, graded, observable, design):
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
        ts, summed = _run(sim, graded, observable, design, eps)
        out = _normalized_power(ts)
        if summed is not None:
            out = out + _normalized_power(summed)
        return out
    return loss


@functools.lru_cache(maxsize=None)
def _model(name):
    build, graded, observable, design = MODELS[name]
    return build(graded), graded, observable, design


@pytest.mark.parametrize("name", sorted(MODELS))
def test_jit_of_the_gradient_equals_the_plain_call(name):
    """``jax.jit(jax.value_and_grad(loss))`` with a design input traces and
    agrees with the plain call (the objective and its gradient are sums over
    the record: 1e-4 of the peak)."""
    sim, graded, observable, design = _model(name)
    loss = _loss(sim, graded, observable, design)
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
    sim, graded, observable, design = _model(name)

    def solve():
        return _run(sim, graded, observable, design, None)

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


@pytest.mark.parametrize("name", ["wire port/uniform", "wire port/graded"])
def test_the_time_stepping_is_recorded_not_run_while_tracing(name):
    """With no traced input, the jaxpr holds the ``N_STEPS``-long scan.

    Before the fix a whitelisted model (a wire port) evaluated the whole
    solve inside the trace-time region: JAX 0.10.2 returned its result as a
    constant, with no scan in the program. A scan of the record's length in
    the jaxpr is what "the time stepping is recorded" means.
    """
    sim, graded, observable, design = _model(name)
    jaxpr = jax.make_jaxpr(lambda: _run(sim, graded, observable, design, None))()
    assert N_STEPS in _scan_lengths(jaxpr), _scan_lengths(jaxpr)


def test_the_adi_time_stepping_is_recorded_not_run_while_tracing():
    """The ADI lane's scan entry obeys the same rule (a PEC box, point source).

    Only the rule is gated here. The jitted ADI probe record differs from the
    plain call by 9-10 float32 ULP at its peak at 40 steps, on main as with
    this change (the implicit solve compiled as one program), which is not
    this entry's to fix.
    """
    sim = Simulation(freq_max=10e9, domain=(12 * MM, 10 * MM, 6 * MM),
                     dx=1.0 * MM, boundary="pec", solver="adi")
    sim.add_source((6 * MM, 5 * MM, 3 * MM), "ez", waveform=_pulse(),
                   amplitude_kind="field")
    sim.add_probe((8 * MM, 5 * MM, 3 * MM), "ez")

    def solve():
        return sim.forward(n_steps=N_STEPS, skip_preflight=True).time_series

    assert N_STEPS in _scan_lengths(jax.make_jaxpr(solve)())
    ts_j = np.asarray(jax.jit(solve)())
    assert ts_j.shape == (N_STEPS, 1) and np.all(np.isfinite(ts_j))
    assert float(np.max(np.abs(ts_j))) > 0.0


# ---------------------------------------------------------------------------
# The helper's two mechanisms (rfx.core.jax_utils)
# ---------------------------------------------------------------------------

def test_the_compile_time_switch_is_found_on_the_ci_jax_versions():
    """The probe finds JAX's own switch on the versions CI runs (0.6.2 and
    0.10.2), so the gate above exercises it; a JAX that drops it falls back
    to the anchor, which the next test exercises."""
    from rfx.core.jax_utils import compile_time_switch

    major_minor = tuple(int(p) for p in jax.__version__.split(".")[:2])
    if major_minor < (0, 6):
        pytest.skip(f"JAX {jax.__version__} predates the CI builds")
    assert compile_time_switch() is not None


def test_the_fallback_anchor_records_the_time_stepping(monkeypatch):
    """Without JAX's switch, ``recorded_scan`` anchors an all-constant carry
    to the outer trace: the wire board's solve with no traced input is still
    compiled (scan in the jaxpr; on 0.6.2 no 'empty' error) and agrees with
    the plain call."""
    from rfx.core import jax_utils

    monkeypatch.setattr(jax_utils, "compile_time_switch", lambda: None)
    sim, graded, observable, design = _model("wire port/uniform")

    def solve():
        return _run(sim, graded, observable, design, None)

    assert N_STEPS in _scan_lengths(jax.make_jaxpr(solve)())
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
