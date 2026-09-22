"""A gradient converges later in record length than the value it belongs to.

The physics. A resonant structure hit with a pulse rings. Every observable this
simulator differentiates is a DFT of the record, so if the structure is still
ringing when the record ends, each bin keeps a leftover term whose phase is
``(w - w_r) * T``. A parameter that moves the resonance spins that phase, and
because the phase is proportional to ``T`` the spin lands in the DERIVATIVE
magnified by the record length while staying small in the VALUE. Measured on
the cavity below: at 1600 steps the record has decayed 46.7 dB -- a pass on the
repo's -40 dB settling rule -- the POWER a user reads is converged to 0.399 %,
and the gradient vector with respect to the fill permittivity is still 11.0 %
away from its converged value, with its direction turned by cos 0.9989 (the
worst single element is 16.8 % out). Doubling the record brings the gradient to
0.12 % and leaves the power where it already was.

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
    first, so the short 1600-step record reads 0.013 % on the log observable
    (0.399 % on the power) and passes any sensible tolerance while its
    gradient is 11.0 % out. Mutation (b) is the one that matters, because it
    is the check a reader would believe: the helper is called, both records
    are run, a number is compared to a tolerance, and the answer is wrong.
    The fixture test's RED assertion is what refuses it.

(c) Only the real part of a complex sensitivity compared -- the state before
    this review. ``test_complex_observable_compares_the_whole_complex_sensitivity``
    goes red: the on-pole bin's sensitivity is purely imaginary, so the real
    part alone reports 0 % where the phasor's sensitivity moved 5.5 %.

(d) The report table's floor taken from the SHORT record only, and (e) taken
    globally instead of per bin. ``test_the_report_floor_is_per_bin_and_spans_both_records``
    goes red on (d) at bin 2 (a floor eight decades too small turns a 1e-9
    element into a 100 % change) and on (e) at bin 1 (a weak bin that really
    doubled gets damped to 0.1 %).

No mutation is applied by a test here. They are run by hand against the source
and the literal pytest output is recorded in the pull request.
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


def _phasor_objective(p, n_steps):
    """The truncated pole's COMPLEX spectrum -- the shape of an S-parameter."""
    T = n_steps * _DT
    omega_r = _OMEGA_R0 * jnp.exp(p)
    s = _ALPHA + 1j * (jnp.asarray(_OMEGAS) - omega_r)
    return (1.0 - jnp.exp(-s * T)) / s


def _analytic_complex_sensitivity(n_steps):
    """dX/dp at p = 0 -- the FULL complex sensitivity, hand-derived.

    The product rule is written out here (``dX/ds`` times ``ds/dp``) instead of
    being obtained by differentiating ``X``, which is what keeps this an oracle
    rather than a second copy of the expression under test.
    """
    T = n_steps * _DT
    s = _ALPHA + 1j * (_OMEGAS - _OMEGA_R0)
    e = np.exp(-s * T)
    dX_ds = T * e / s - (1.0 - e) / s ** 2
    ds_dp = -1j * _OMEGA_R0          # w_r = w_r0 exp(p) -> dw_r/dp = w_r0 at p=0
    return dX_ds * ds_dp


def _analytic_spectrum(n_steps):
    """X(w) at p = 0, the converged-minus-leftover closed form."""
    T = n_steps * _DT
    s = _ALPHA + 1j * (_OMEGAS - _OMEGA_R0)
    return (1.0 - np.exp(-s * T)) / s


def _analytic_gradient(n_steps):
    """d|X|^2/dp at p = 0, one chain-rule step past the complex sensitivity."""
    X = _analytic_spectrum(n_steps)
    dX_dp = _analytic_complex_sensitivity(n_steps)
    return 2.0 * (X.real * dX_dp.real + X.imag * dX_dp.imag)


def _expected_rel(g_short, g_long):
    """The verdict's own arithmetic on one parameter element per bin.

    Floor-free by construction: with a single element the norm IS the absolute
    value, so this is the closed form and nothing else.
    """
    delta = np.abs(g_long - g_short)
    return np.where(delta == 0.0, 0.0,
                    delta / np.where(g_long == 0.0, np.inf, np.abs(g_long)))


def test_analytic_truncated_record_reproduces_the_closed_form_change():
    """The witness's per-bin verdict IS the closed-form truncation spin."""
    from tests._x64_compat import enable_x64

    n_short, factor = 1000, 2.0
    n_long = int(math.ceil(factor * n_short))

    g_short = _analytic_gradient(n_short)
    g_long = _analytic_gradient(n_long)
    expected = _expected_rel(g_short, g_long)

    with enable_x64():
        witness = gradient_record_length_witness(
            _spectrum_objective, jnp.float64(0.0), n_short,
            tol=1e9, factor=factor,
        )

    assert witness.rel_by_bin.shape == (len(_OMEGAS),)
    np.testing.assert_allclose(witness.rel_by_bin, expected, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(witness.worst, expected.max(), rtol=1e-6)
    np.testing.assert_allclose(
        witness.grad["<root>"], g_short, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(
        witness.grad_long["<root>"], g_long, rtol=1e-6, atol=0.0)
    assert witness.worst_bin == 0
    assert witness.rel_by_bin[0] == pytest.approx(0.3607, abs=1e-3)
    assert not witness.observable_is_scalar
    assert not witness.observable_is_complex

    # What the second bin does and does NOT say. It sits exactly on the pole,
    # where |X|^2 is STATIONARY in w_r, so d|X|^2/dp is identically zero at
    # every record length -- both arms below are exact zeros. That is a fact
    # about this observable, not a measurement that the record is long enough:
    # the witness has nothing to compare there. The same bin's COMPLEX
    # sensitivity does move with the record
    # (test_complex_observable_compares_the_whole_complex_sensitivity), which
    # is why a zero here must not be read as "this bin is converged".
    assert witness.grad["<root>"][1] == 0.0
    assert witness.grad_long["<root>"][1] == 0.0

    # At this damping the record is long enough for the VALUE and not for the
    # GRADIENT -- the whole reason this witness is not the settling witness.
    assert witness.worst_value_rel_change == pytest.approx(0.0198, abs=5e-4)


def test_complex_observable_compares_the_whole_complex_sensitivity():
    """A phasor's sensitivity is two numbers, and half of one is not a check.

    The same pole, differentiated as the COMPLEX spectrum instead of its
    squared magnitude. At the bin sitting on the pole the parameter rotates the
    phasor at constant magnitude: the real part of the sensitivity is exactly
    zero and the imaginary part is not, and it is the imaginary part that moves
    with the record length. Comparing only the real part reports that bin as
    perfectly converged when its sensitivity has moved 5.5 %.
    """
    from tests._x64_compat import enable_x64

    n_short, factor = 1000, 2.0
    n_long = int(math.ceil(factor * n_short))

    c_short = _analytic_complex_sensitivity(n_short)
    c_long = _analytic_complex_sensitivity(n_long)
    expected = _expected_rel(c_short, c_long)
    x_short = _analytic_spectrum(n_short)
    x_long = _analytic_spectrum(n_long)

    with enable_x64():
        witness = gradient_record_length_witness(
            _phasor_objective, jnp.float64(0.0), n_short,
            tol=1e9, factor=factor,
        )

    assert witness.observable_is_complex
    np.testing.assert_allclose(
        witness.grad["<root>"], c_short, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(
        witness.grad_long["<root>"], c_long, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(witness.rel_by_bin, expected, rtol=1e-6, atol=0.0)
    assert witness.rel_by_bin[0] == pytest.approx(0.36879, abs=1e-4)

    # The on-pole bin: real part exactly zero, imaginary part carrying
    # everything, and a 5.5 % record-length change that only the complex
    # comparison can see. The |X|^2 objective reports exactly 0 % here.
    assert c_short[1].real == 0.0 and abs(c_short[1].imag) > 0.0
    assert witness.grad["<root>"][1].real == 0.0
    assert witness.rel_by_bin[1] == pytest.approx(0.05541, abs=1e-4)
    assert _expected_rel(
        _analytic_gradient(n_short), _analytic_gradient(n_long))[1] == 0.0

    # The value's relative change uses the complex magnitude, not the real part.
    np.testing.assert_allclose(
        witness.value_rel_change,
        np.abs(x_long - x_short) / np.abs(x_long), rtol=1e-6, atol=0.0)
    assert witness.worst_value_rel_change == pytest.approx(0.01010, abs=1e-4)
    assert np.iscomplexobj(witness.value)

    # The direction cosine is the HERMITIAN product. On the pole the two arms
    # are the same purely imaginary vector: the plain product would call that
    # a reversed direction (-1); the Hermitian one says +1.
    hand = np.array([
        float(np.real(np.vdot(np.atleast_1d(c_short[b]), np.atleast_1d(c_long[b])))
              / (abs(c_short[b]) * abs(c_long[b])))
        for b in range(len(c_short))
    ])
    np.testing.assert_allclose(witness.cosine_by_bin, hand, rtol=0.0, atol=1e-12)
    assert witness.cosine_by_bin[1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(witness.cosine_by_bin > 0.99)


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


def _prescribed_gradients(short_matrix, long_matrix, n_short):
    """An objective whose gradients at the two arms are exactly given.

    ``y_b = sum_e A[b, e] p_e`` makes ``dy_b/dp_e`` exactly ``A[b, e]``, so the
    two arms' gradient vectors are whatever the two matrices say. These tests
    are about the arithmetic of the verdict and of the report table, so they
    use no simulation and no fixture.
    """
    short = np.asarray(short_matrix, dtype=np.float64)
    long = np.asarray(long_matrix, dtype=np.float64)

    def objective(params, n_steps):
        a = short if n_steps == n_short else long
        return jnp.asarray(a) @ params

    return objective


def test_the_report_floor_is_per_bin_and_spans_both_records():
    """The per-element table is floored, and against the right thing.

    Three bins, one two-element parameter, gradients set by hand.

    * Bin 0 -- the dominant element does not move and a second element goes
      from exactly 0 to 1e-9. Unfloored that element reads a 100 % change off
      pure rounding; the verdict says the gradient VECTOR moved by 1e-9.
    * Bin 1 -- a bin whose whole gradient is a millionth of bin 0's, and which
      really did double. A floor taken globally instead of per bin would damp
      that to 0.1 % and hide a bin that genuinely moved.
    * Bin 2 -- the short record's gradient is 1e-8 and the long record's is 1,
      so a floor taken from the SHORT arm alone would be eight decades too
      small and report the trailing element as another 100 % change.
    """
    from tests._x64_compat import enable_x64

    n_short = 100
    short = [[1.0, 0.0], [1.0e-6, 0.0], [1.0e-8, 0.0]]
    long = [[1.0, 1.0e-9], [2.0e-6, 1.0e-9], [1.0, 1.0e-9]]
    objective = _prescribed_gradients(short, long, n_short)

    with enable_x64():
        witness = gradient_record_length_witness(
            objective, jnp.zeros((2,), dtype=jnp.float64), n_short,
            tol=1e9, factor=2.0, floor_frac=1e-3,
        )

    table = witness.grad_rel_change["<root>"]
    assert table.shape == (3, 2)

    # Bin 0: the floor keeps rounding out of the table, and the verdict is
    # floor-free and says the vector did not move.
    assert table[0, 1] == pytest.approx(1.0e-6, rel=1e-6)
    assert witness.rel_by_bin[0] == pytest.approx(1.0e-9, rel=1e-6)

    # Bin 1: per-bin floor. A global floor would read 1e-3 here.
    assert table[1, 0] == pytest.approx(0.5, rel=1e-9)
    assert table[1, 1] == pytest.approx(0.5, rel=1e-9)
    assert witness.rel_by_bin[1] == pytest.approx(0.5, rel=1e-6)

    # Bin 2: floor spans both records. From the short arm alone it would read 1.
    assert table[2, 1] == pytest.approx(1.0e-6, rel=1e-6)
    assert witness.rel_by_bin[2] == pytest.approx(1.0, rel=1e-6)


def test_the_verdict_is_norm_level_not_the_worst_element():
    """A per-element maximum is not a statement about the gradient.

    One dominant element that does not move, and three elements at 1 % of it
    that triple. Every one of those three reads a 67 % change on its own scale
    -- this is the shape of a per-cell permittivity leaf -- while the gradient
    vector moved 3.5 % and its direction turned by less than a twentieth of a
    degree. The verdict is the vector; the element table locates where the
    change sits once the verdict has already failed.
    """
    from tests._x64_compat import enable_x64

    n_short = 100
    short = [[1.0, 1.0e-2, 1.0e-2, 1.0e-2]]
    long = [[1.0, 3.0e-2, 3.0e-2, 3.0e-2]]
    objective = _prescribed_gradients(short, long, n_short)

    with enable_x64():
        witness = gradient_record_length_witness(
            objective, jnp.zeros((4,), dtype=jnp.float64), n_short,
            tol=0.05, factor=2.0,
        )

    delta = np.asarray(long) - np.asarray(short)
    expected = np.linalg.norm(delta) / np.linalg.norm(np.asarray(long))
    assert witness.worst == pytest.approx(expected, rel=1e-9)
    assert witness.worst == pytest.approx(0.03459, abs=1e-4)
    assert witness.passed, witness.summary()

    assert witness.worst_elementwise == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert witness.worst_elementwise > 10.0 * witness.worst
    assert witness.cosine_by_bin[0] > 0.999


def test_the_verdict_concatenates_every_leaf_not_the_first():
    """Two leaves; the first alone would pass, the pair does not.

    Leaf ``a`` (one element) moves by 1 %; leaf ``b`` (three elements, each
    as large as ``a``) moves by 40 %. Over the concatenated vector the change
    is 26 %; over the first leaf alone it would be 1 %. A verdict that reads
    one leaf reports the wrong number and a cosine of exactly 1.
    """
    from tests._x64_compat import enable_x64

    n_short = 100
    short_a, long_a = [[1.00]], [[1.01]]
    short_b, long_b = [[1.0, 1.0, 1.0]], [[1.4, 1.4, 1.4]]

    def objective(params, n_steps):
        a = (short_a, short_b) if n_steps == n_short else (long_a, long_b)
        return (jnp.asarray(np.asarray(a[0], dtype=np.float64)) @ params["a"]
                + jnp.asarray(np.asarray(a[1], dtype=np.float64)) @ params["b"])

    with enable_x64():
        # Built inside the x64 scope: outside it a float64 request is
        # silently float32 and the arithmetic below loses 8 digits.
        params = {"a": jnp.zeros((1,), dtype=jnp.float64),
                  "b": jnp.zeros((3,), dtype=jnp.float64)}
        witness = gradient_record_length_witness(
            objective, params, n_short, tol=0.05, factor=2.0)

    g_s = np.concatenate([np.asarray(short_a)[0], np.asarray(short_b)[0]])
    g_l = np.concatenate([np.asarray(long_a)[0], np.asarray(long_b)[0]])
    expected = np.linalg.norm(g_l - g_s) / np.linalg.norm(g_l)
    assert witness.worst == pytest.approx(expected, rel=1e-9)
    assert witness.worst == pytest.approx(0.2638, abs=1e-3)
    assert not witness.passed, witness.summary()
    assert witness.cosine_by_bin[0] < 0.9999
    assert witness.cosine_by_bin[0] == pytest.approx(
        float(g_s @ g_l / (np.linalg.norm(g_s) * np.linalg.norm(g_l))), rel=1e-9)


def test_a_gradient_that_vanishes_on_the_long_record_is_not_a_small_change():
    """No floor to divide by: it reads infinite, and fails."""
    from tests._x64_compat import enable_x64

    n_short = 100
    objective = _prescribed_gradients([[1.0]], [[0.0]], n_short)
    with enable_x64():
        witness = gradient_record_length_witness(
            objective, jnp.zeros((1,), dtype=jnp.float64), n_short, tol=0.05)
    assert witness.rel_by_bin[0] == math.inf
    assert not witness.passed
    assert math.isnan(witness.cosine_by_bin[0])


def test_a_tuple_returning_objective_without_has_aux_says_so():
    def objective(params, n_steps):
        del n_steps
        return jnp.sum(params ** 2), {"settling_db": -50.0}

    with pytest.raises(ValueError, match="has_aux=True"):
        gradient_record_length_witness(
            objective, jnp.zeros((2,), dtype=jnp.float32), 100, tol=0.05)


def test_an_integer_parameter_leaf_is_refused():
    def objective(params, n_steps):
        del n_steps
        return jnp.sum(params.astype(jnp.float32) ** 2)

    with pytest.raises(ValueError, match="carries no gradient"):
        gradient_record_length_witness(
            objective, jnp.zeros((2,), dtype=jnp.int32), 100, tol=0.05)


def test_a_complex_parameter_leaf_is_refused():
    def objective(params, n_steps):
        del n_steps
        return jnp.abs(jnp.sum(params)) ** 2

    with pytest.raises(ValueError, match="is complex"):
        gradient_record_length_witness(
            objective, jnp.zeros((2,), dtype=jnp.complex64), 100, tol=0.05)


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

    # The objective returns ln(power), so its own relative change is a change
    # of a logarithm. What a user reads is the POWER, so that is what gets
    # compared against the gradient.
    power_rel_change = np.abs(
        np.expm1(np.asarray(red.value_long) - np.asarray(red.value)))

    print("\nshort record:", red.summary())
    print("long record: ", green.summary())
    print(f"  power |dP/P| per bin: {power_rel_change * 100} %")
    print(f"  rel_by_bin short {red.rel_by_bin * 100} %  "
          f"cos {red.cosine_by_bin}")
    print(f"  rel_by_bin long  {green.rel_by_bin * 100} %  "
          f"cos {green.cosine_by_bin}")
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

    # ... and the POWER a user reads is converged to well under half a percent.
    assert power_rel_change.max() == pytest.approx(0.00399, abs=5e-4), (
        power_rel_change)

    # ... while its GRADIENT is not. This assertion is the one mutation (b)
    # breaks: a verdict read off the value above would call this record good.
    assert not red.passed, red.summary()
    assert red.worst > 0.10, red.summary()
    assert red.worst > 20.0 * power_rel_change.max(), red.summary()

    # Doubling the record closes it.
    assert green.passed, green.summary()
    assert green.worst < 0.5 * tol, green.summary()

    assert np.all(np.isfinite(red.value)) and np.all(np.isfinite(green.value))
    assert red.n_steps_long == n_long and green.n_steps_long == n_longer


def test_the_witness_is_reachable_from_the_public_package():
    import rfx

    assert rfx.gradient_record_length_witness is gradient_record_length_witness
    assert rfx.GradientRecordLengthWitness.__name__ == "GradientRecordLengthWitness"
