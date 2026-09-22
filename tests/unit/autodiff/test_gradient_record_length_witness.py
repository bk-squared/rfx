"""A gradient converges later in record length than the value it belongs to.

The physics. A resonant structure hit with a pulse rings. Every observable this
simulator differentiates is a DFT of the record, so if the structure is still
ringing when the record ends, each bin keeps a leftover term whose phase is
``(w - w_r) * T``. A parameter that moves the resonance spins that phase, and
because the phase is proportional to ``T`` the spin lands in the DERIVATIVE
magnified by the record length while staying small in the VALUE. Measured on
the cavity below: at 1600 steps the record has decayed 46.7 dB -- a pass on the
repo's -40 dB settling rule -- the value is converged to 0.013 %, and the
gradient with respect to the fill permittivity is still 16.8 % away from its
converged value. Doubling the record fixes the gradient, not the value.

Two existing checks do not see this. ``settling_verdict`` scores the end of the
record against its peak, which is a statement about the value. AD against a
finite difference agrees at every record length, because both sides
differentiate the same truncated record.

Mutation evidence for this gate (repo rule: a new gate ships with both arms).

(a) Check disabled -- ``gradient_record_length_witness`` returns
    ``passed=True`` unconditionally: the fixture test below fails on its RED
    assertion, since a red arm is what the short record is for.

(b) Helper call kept, defect reintroduced -- the witness still runs both
    records and still reports, but the verdict is taken from
    ``worst_value_rel_change`` (the VALUE's relative change) instead of the
    gradient's. This is the tautology the issue names: the value converges
    first, so the short 1600-step record reads 0.013 % and passes any
    sensible tolerance while its gradient is 16.8 % out. Mutation (b) is the
    one that matters, because it is the check a reader would believe: the
    helper is called, both records are run, a number is compared to a
    tolerance, and the answer is wrong. The fixture test's RED assertion is
    what refuses it.

Neither mutation is applied by a test here. They are run by hand against the
source and the literal pytest output is recorded in the pull request.
"""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api._gradient_witness import gradient_record_length_witness


# ---------------------------------------------------------------------------
# A closed-form truncated record: one damped pole, cut at T.
#
#   x(t) = exp(-(alpha + i w_r) t),  0 <= t < T
#   X(w) = (1 - exp(-s T)) / s,      s = alpha + i (w - w_r)
#
# The converged (T -> inf) spectrum is 1/s; the whole truncation effect is the
# exp(-sT) term. Differentiating it with respect to a parameter that moves w_r
# produces a term proportional to T -- the spin this witness exists to catch.
# ---------------------------------------------------------------------------

_ALPHA = 1.21e9          # Np/s -> Q = w_r / (2 alpha) ~ 18 at 7 GHz
_OMEGA_R0 = 2.0 * math.pi * 7.0e9
_DT = 3.8e-12            # s, the cavity fixture's own timestep to 2 digits
_OMEGAS = 2.0 * math.pi * np.array([5.5e9, 7.0e9])


def _spectrum_objective(p, n_steps):
    """|X(w)|^2 of the truncated pole, as a jnp expression AD walks through.

    ``p`` is a log-parameter on the resonant frequency: ``w_r = w_r0 exp(p)``.
    Written with ``real**2 + imag**2`` rather than ``abs(...)**2`` so nothing
    depends on how ``jnp.abs`` differentiates a complex intermediate.
    """
    T = n_steps * _DT
    omega_r = _OMEGA_R0 * jnp.exp(p)
    s = _ALPHA + 1j * (jnp.asarray(_OMEGAS) - omega_r)
    X = (1.0 - jnp.exp(-s * T)) / s
    return X.real ** 2 + X.imag ** 2


def _analytic_gradient(n_steps):
    """d|X|^2/dp at p = 0, from the hand-derived derivative, in float64.

    Independent of the expression above: the product rule is written out here
    (``dX/ds`` times ``ds/dp``) instead of being obtained by differentiating
    ``X``. That is what keeps this an oracle rather than a second copy.
    """
    T = n_steps * _DT
    s = _ALPHA + 1j * (_OMEGAS - _OMEGA_R0)
    e = np.exp(-s * T)
    X = (1.0 - e) / s
    dX_ds = T * e / s - (1.0 - e) / s ** 2
    ds_dp = -1j * _OMEGA_R0          # w_r = w_r0 exp(p) -> dw_r/dp = w_r0 at p=0
    dX_dp = dX_ds * ds_dp
    return 2.0 * (X.real * dX_dp.real + X.imag * dX_dp.imag)


def test_analytic_truncated_record_reproduces_the_closed_form_change():
    """The witness's per-bin number IS the closed-form truncation spin."""
    from tests._x64_compat import enable_x64

    n_short, factor = 1000, 2.0
    n_long = int(math.ceil(factor * n_short))

    g_short = _analytic_gradient(n_short)
    g_long = _analytic_gradient(n_long)
    floor_frac = 1e-3
    scale = np.maximum(np.abs(g_long), np.abs(g_short)).max()
    den = np.maximum(np.abs(g_long), floor_frac * scale)
    expected = np.abs(g_long - g_short) / den

    with enable_x64():
        witness = gradient_record_length_witness(
            _spectrum_objective, jnp.float64(0.0), n_short,
            tol=1e9, factor=factor, floor_frac=floor_frac,
        )

    rel = witness.grad_rel_change["<root>"]
    assert rel.shape == (len(_OMEGAS), )
    np.testing.assert_allclose(rel, expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(witness.worst, expected.max(), rtol=1e-6)
    np.testing.assert_allclose(
        witness.grad["<root>"], g_short, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(
        witness.grad_long["<root>"], g_long, rtol=1e-6, atol=0.0)

    # Which bin moves, and which does not. The second bin sits exactly on the
    # pole: there the leftover term is in phase with the converged one, so
    # moving the resonance does not spin it and the gradient is already right.
    # The bin 1.5 GHz below it is out by a third of itself.
    assert rel[1] == pytest.approx(0.0, abs=1e-9)
    assert rel[0] > 0.3
    assert witness.worst_bin == 0
    assert not witness.observable_is_scalar

    # At this damping the record is long enough for the VALUE and not for the
    # GRADIENT -- the whole reason this witness is not the settling witness.
    assert witness.worst_value_rel_change < 0.1 * witness.worst


def test_a_long_enough_record_passes_the_same_objective():
    """Same pole, a record 8x longer: the spin has decayed out of both arms."""
    from tests._x64_compat import enable_x64

    with enable_x64():
        witness = gradient_record_length_witness(
            _spectrum_objective, jnp.float64(0.0), 8000, tol=0.05, factor=2.0,
        )
    assert witness.passed, witness.summary()
    assert witness.worst < 0.05


def test_scalar_observable_and_pytree_params_report_per_leaf():
    """A scalar objective over a dict of parameters: one bin, named leaves."""

    def objective(params, n_steps):
        T = n_steps * _DT
        omega_r = _OMEGA_R0 * jnp.exp(params["global"])
        s = _ALPHA * jnp.exp(params["damping"]) + 1j * (_OMEGAS[1] - omega_r)
        X = (1.0 - jnp.exp(-s * T)) / s
        return X.real ** 2 + X.imag ** 2

    params = {"damping": jnp.float32(0.0), "global": jnp.float32(0.0)}
    witness = gradient_record_length_witness(objective, params, 1000, tol=0.05)

    assert witness.observable_is_scalar
    assert witness.value.shape == (1,)
    assert sorted(witness.grad_rel_change) == ["['damping']", "['global']"]
    for arr in witness.grad_rel_change.values():
        assert arr.shape == (1,)
    assert witness.worst_leaf in witness.grad_rel_change
    assert witness.worst_bin == 0
    assert not witness.passed          # 1000 steps is short for this pole
    assert "gradient record-length witness FAIL" in witness.summary()


def test_array_leaf_is_reported_elementwise():
    """A leaf that is an array keeps its shape in the report."""

    weights = jnp.asarray([1.0, 0.5, 0.25], dtype=jnp.float32)

    def objective(params, n_steps):
        T = n_steps * _DT
        omega_r = _OMEGA_R0 * jnp.exp(jnp.sum(params[0] * weights))
        s = _ALPHA + 1j * (jnp.asarray(_OMEGAS) - omega_r)
        X = (1.0 - jnp.exp(-s * T)) / s
        return X.real ** 2 + X.imag ** 2

    params = (jnp.zeros((3,), dtype=jnp.float32),)
    witness = gradient_record_length_witness(objective, params, 1000, tol=0.05)
    (path,) = witness.grad_rel_change
    assert witness.grad_rel_change[path].shape == (len(_OMEGAS), 3)
    assert witness.grad[path].shape == (len(_OMEGAS), 3)


def test_aux_protocol_is_optional_and_reads_only_settling_db():
    """``aux`` may carry the record's ring-down level; anything else is fine."""

    def with_settling(p, n_steps):
        value = _spectrum_objective(p, n_steps)
        return value, {"settling_db": -20.0 * math.log10(float(n_steps))}

    witness = gradient_record_length_witness(
        with_settling, jnp.float32(0.0), 1000, tol=0.05, has_aux=True)
    assert witness.settling_db == pytest.approx(-60.0)
    assert witness.settling_db_long == pytest.approx(-66.0206, abs=1e-3)
    assert "record settled to -60.0 dB" in witness.summary()

    def other_aux(p, n_steps):
        return _spectrum_objective(p, n_steps), ("not a mapping",)

    other = gradient_record_length_witness(
        other_aux, jnp.float32(0.0), 1000, tol=0.05, has_aux=True)
    assert other.settling_db is None and other.settling_db_long is None
    assert other.worst == pytest.approx(witness.worst, rel=1e-6)


def test_tolerance_has_no_default_and_the_arms_must_differ():
    """``tol`` is the caller's declaration, not a number this module picked."""
    with pytest.raises(TypeError):
        gradient_record_length_witness(
            _spectrum_objective, jnp.float32(0.0), 1000)

    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="tol"):
            gradient_record_length_witness(
                _spectrum_objective, jnp.float32(0.0), 1000, tol=bad)

    for bad in (1.0, 0.5, float("inf")):
        with pytest.raises(ValueError, match="factor"):
            gradient_record_length_witness(
                _spectrum_objective, jnp.float32(0.0), 1000, tol=0.05,
                factor=bad)

    with pytest.raises(ValueError, match="n_steps must be positive"):
        gradient_record_length_witness(
            _spectrum_objective, jnp.float32(0.0), 0, tol=0.05)


def test_a_two_dimensional_observable_is_refused():
    """Reduce or flatten it yourself, so the reported bins are the meant ones."""

    def matrix_objective(p, n_steps):
        return jnp.outer(_spectrum_objective(p, n_steps), jnp.ones((2,)))

    with pytest.raises(ValueError, match="scalar or a 1-D array"):
        gradient_record_length_witness(
            matrix_objective, jnp.float32(0.0), 1000, tol=0.05)


def test_a_record_length_independent_objective_passes_exactly():
    """No record, no spin: the two arms agree to the bit."""

    def no_record(p, n_steps):
        del n_steps
        return jnp.sum(p ** 2)

    witness = gradient_record_length_witness(
        no_record, jnp.asarray([1.0, 2.0], dtype=jnp.float32), 500, tol=1e-12)
    assert witness.passed
    assert witness.worst == 0.0
    assert witness.n_steps_long == 1000


# ---------------------------------------------------------------------------
# The FDTD fixture: a PEC box filled with a lossy dielectric, driven by a
# pulse, with the permittivity as the parameter. A cavity is the cheapest
# structure that rings, and the permittivity is the cheapest parameter that
# moves the resonance -- the two ingredients the mechanism needs.
# ---------------------------------------------------------------------------

_CAVITY = dict(
    a=30e-3, b=20e-3, d=20e-3, dx=2e-3,
    eps_r=2.0, sigma=0.039,        # tan d ~ 0.05 at 7 GHz -> Q ~ 20
    f0=7.0e9, bandwidth=0.9,
    freqs_hz=(5.5e9, 6.5e9, 7.5e9),
    n_short=1600, factor=2.0,
)


def _build_cavity():
    from rfx import Simulation, Box, GaussianPulse

    c = _CAVITY
    a, b, d = c["a"], c["b"], c["d"]
    sim = Simulation(freq_max=2 * c["f0"], domain=(a, b, d),
                     boundary="pec", dx=c["dx"])
    sim.add_material("fill", eps_r=c["eps_r"], sigma=c["sigma"])
    sim.add(Box((0, 0, 0), (a, b, d)), material="fill")
    sim.add_source((a / 4, b / 3, d / 2), "ez",
                   waveform=GaussianPulse(f0=c["f0"], bandwidth=c["bandwidth"]),
                   amplitude_kind="current")
    sim.add_probe((3 * a / 4, 2 * b / 3, d / 2), "ez")
    return sim


def test_cavity_gradient_needs_a_longer_record_than_its_value():
    """RED at a record the -40 dB rule passes; GREEN once it is doubled."""
    from rfx.api._sparams import settling_verdict

    c = _CAVITY
    sim = _build_cavity()
    grid = sim._build_grid()
    base_materials, *_ = sim._assemble_materials(grid)
    base_eps = jnp.asarray(base_materials.eps_r)

    # Realized, not declared.
    nx, ny, nz = grid.shape
    assert (nx, ny, nz) == (16, 11, 11), grid.shape
    assert float(base_eps.max()) == pytest.approx(c["eps_r"])

    # ``eps_local`` scales only the half of the box the probe sits in, so the
    # two leaves are genuinely different directions in parameter space.
    local = np.zeros(grid.shape, dtype=np.float32)
    local[nx // 2:, :, :] = 1.0
    local_j = jnp.asarray(local)

    freqs = jnp.asarray(c["freqs_hz"])
    dt = float(grid.dt)
    n_short = c["n_short"]
    n_long = int(math.ceil(c["factor"] * n_short))
    n_longer = int(math.ceil(c["factor"] * n_long))

    def permittivity(params):
        return (base_eps * jnp.exp(params["eps_global"])
                * jnp.exp(params["eps_local"] * local_j))

    # Ring-down of the unperturbed record at each length, scored by the repo's
    # own -40 dB witness. Measured on the concrete run: the settling property
    # is a host diagnostic and reports absence while the record is traced.
    settling = {}
    for n in (n_short, n_long, n_longer):
        result = sim.forward(eps_override=base_eps, n_steps=n,
                             skip_preflight=True)
        settling[n] = result.settling_db

    def objective(params, n_steps):
        result = sim.forward(eps_override=permittivity(params),
                             n_steps=n_steps, skip_preflight=True)
        ts = result.time_series
        if ts.ndim == 2:
            ts = ts[:, 0]
        k = jnp.arange(n_steps, dtype=ts.dtype)
        spectrum = jnp.sum(
            ts[None, :] * jnp.exp(-2j * jnp.pi * freqs[:, None] * k[None, :] * dt),
            axis=1,
        )
        power = spectrum.real ** 2 + spectrum.imag ** 2
        return jnp.log(power), {"settling_db": settling[n_steps]}

    params = {"eps_global": jnp.float32(0.0), "eps_local": jnp.float32(0.0)}
    tol = 0.05

    red = gradient_record_length_witness(
        objective, params, n_short, tol=tol, factor=c["factor"], has_aux=True)
    green = gradient_record_length_witness(
        objective, params, n_long, tol=tol, factor=c["factor"], has_aux=True)

    print("\nshort record:", red.summary())
    print("long record: ", green.summary())
    for name, w in (("short", red), ("long", green)):
        for path, arr in sorted(w.grad_rel_change.items()):
            for i, f in enumerate(c["freqs_hz"]):
                print(f"  {name} {path:14s} {f/1e9:5.2f} GHz  "
                      f"g={w.grad[path][i]: .5f} -> "
                      f"g_long={w.grad_long[path][i]: .5f}  "
                      f"rel {arr[i] * 100:7.3f}%")

    # The short record passes the settling rule -- that is what makes this a
    # defect and not a too-short run.
    assert settling_verdict(settling[n_short]) == "pass", settling[n_short]
    assert settling_verdict(settling[n_long]) == "pass", settling[n_long]
    assert red.settling_db == settling[n_short]
    assert red.settling_db_long == settling[n_long]

    # ... and its VALUE is converged, by a wide margin over the tolerance.
    assert red.worst_value_rel_change < 0.01 * tol, red.value_rel_change

    # ... while its GRADIENT is not. This assertion is the one mutation (b)
    # breaks: a verdict read off the value above would call this record good.
    assert not red.passed, red.summary()
    assert red.worst > 0.10, red.summary()

    # Doubling the record closes it.
    assert green.passed, green.summary()
    assert green.worst < 0.5 * tol, green.summary()

    assert np.all(np.isfinite(red.value)) and np.all(np.isfinite(green.value))
    assert red.n_steps_long == n_long and green.n_steps_long == n_longer


def test_the_witness_is_reachable_from_the_public_package():
    import rfx

    assert rfx.gradient_record_length_witness is gradient_record_length_witness
    assert rfx.GradientRecordLengthWitness.__name__ == "GradientRecordLengthWitness"
