"""tests/_gate_policy.py

Single shared definition of the envelope -> gate derivation used by every
frozen-fixture crossval case that gates a measured envelope (issue #528).

Every gated crossval case derives its enforced numeric gate as

    gate = round-up(measured envelope x ENVELOPE_GATE_MULTIPLIER)

quantized to a case-specific number of decimal places (100 -> 0.01 for
absolute |S11| gates, 10 -> 0.1 for dB gates). Before this module existed,
each case hardcoded the ``1.5`` multiplier independently in its own test
file's gate-derivation assertion AND, for the crossval scripts that
regenerate their fixtures, a second time in a script-side self-check --
nothing cross-checked the two, so relaxing the multiplier in one case (a
one-line find-replace touching only that case's file) read as compliance
with a repo-wide convention while silently widening only that case's gate.
An adversarial review of PR #499 demonstrated exactly that: 1.5 -> 3.0 in
one case doubled its gate with every existing guard still green.

Consumers (as of this writing):
  * tests/test_wr90_iris_modematch_gates.py        (quantum=100, abs |S11|)
  * tests/test_rcs_mie_ka_sweep_gates.py            (quantum=10,  dB)
  * tests/test_rcs_dielectric_sphere_mie_gates.py   (quantum=10,  dB)
  * validation/crossval/18_wr90_iris_modematch.py       (--write-fixture self-check)
  * validation/crossval/16_pec_sphere_mie_ka_sweep.py   (--write-fixture self-check)
  * validation/crossval/17_dielectric_sphere_mie.py     (--write-fixture self-check)

The bounded-margin lanes (``test_waveguide_broad_e5_tolerance_envelope.py``
and its phase / group-delay siblings) check a structurally different shape
-- a PINNED module constant bounded by ``[worst_measured, worst_measured x
MULTIPLIER]`` rather than a quantized derived value, so they do not call
``gate_from_envelope`` -- but they share the SAME multiplier (their own
comments already say so: "Same governance margin ceiling as the magnitude
lane's MARGIN_CEIL=1.5"), so they import ``ENVELOPE_GATE_MULTIPLIER``
directly instead of restating ``1.5`` as a fresh local literal.

A change to ``ENVELOPE_GATE_MULTIPLIER`` here moves every one of the above
at once -- that is the visibility guarantee issue #528 asks for: a per-case
relaxation now requires editing a shared, reviewer-visible object instead of
a local literal. ``tests/test_gate_policy_is_shared.py`` is the cross-check
(no consumer may carry a local literal instead of this import) and the
falsifier (mutating the constant moves every quantized-gate consumer's
derived value in lockstep).
"""

from __future__ import annotations

import math

ENVELOPE_GATE_MULTIPLIER: float = 1.5
"""The repo-wide margin multiplier every measured-envelope gate is derived
from. This is a GOVERNANCE choice (how much slack a reviewer tolerates
above the measured worst case), not a physical bound. Widening it widens
every consumer listed in this module's docstring at once -- do not change
it without a written root-cause per case (no-silent-gate-loosening)."""


def gate_from_envelope(measured_envelope: float, *, quantum: float) -> float:
    """Round ``measured_envelope * ENVELOPE_GATE_MULTIPLIER`` UP to the
    nearest ``1 / quantum``.

    ``quantum=100`` -> 2 decimal places (e.g. absolute |S11| gates).
    ``quantum=10``  -> 1 decimal place (e.g. dB gates).

    Reads ``ENVELOPE_GATE_MULTIPLIER`` from this module's namespace at call
    time rather than capturing it as a bound default, so monkeypatching the
    module attribute changes every subsequent call made through this same
    function object -- including calls reached via
    ``from tests._gate_policy import gate_from_envelope`` in a consumer
    module, since Python functions resolve globals against their *defining*
    module, not the importer's. This is the mechanism the falsifier test in
    ``test_gate_policy_is_shared.py`` relies on.
    """
    return math.ceil(measured_envelope * ENVELOPE_GATE_MULTIPLIER * quantum) / quantum
