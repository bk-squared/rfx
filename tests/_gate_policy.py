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
  * tests/unit/sparams/test_msl_port_integration.py              (quantum=1000, |Z0| length-spread, #518)
  * tests/crossval/test_wr90_iris_modematch_gates.py        (quantum=100, abs |S11|)
  * tests/crossval/test_rcs_mie_ka_sweep_gates.py            (quantum=10,  dB)
  * tests/crossval/test_rcs_dielectric_sphere_mie_gates.py   (quantum=10,  dB)
  * tests/crossval/test_wr90_iris_filter_gates.py            (quantum=1,   MHz)
  * validation/crossval/18_wr90_iris_modematch.py       (--write-fixture self-check)
  * validation/crossval/16_pec_sphere_mie_ka_sweep.py   (--write-fixture self-check)
  * validation/crossval/17_dielectric_sphere_mie.py     (--write-fixture self-check)
  * validation/crossval/19_wr90_iris_filter_aghanim.py  (--write-fixture self-check)
  * tests/unit/farfield/test_rcs280_reference_subtraction.py    (quantum=100,  dB pattern mean, #888/#280)
  * tests/unit/sources/test_tfsf_aux_absorber_reflection.py     (quantum=1e5..1e7, |B/A| reflection amplitude, #888)

This list is a SNAPSHOT of the quantized-gate lanes, not a discovered set, and
it has drifted: ``grep -rl gate_from_envelope tests/ validation/ scripts/``
finds considerably more files than appear above. What actually binds the lanes
is ``tests/contracts/test_gate_policy_is_shared.py`` -- its fixture-glob
discovery for the ``gates``-dict cases, and its hand-listed
``_QUANTIZED_GATE_FILES`` tripwire for the flat-JSON ones. Reconciling this
docstring against the full grep is not in PR #1005's scope; the two lanes added
above are the ones that PR introduces.

#931 (lattice ownership contract), and this is a CHECKLIST, not a change:
``ENVELOPE_GATE_MULTIPLIER`` does not move. What moves is the measured
envelope of every case whose realized geometry changed, and each of those
gates must be RE-DERIVED through :func:`gate_from_envelope` and re-pinned in
the SAME commit as the regenerated fixture, with a written root cause. The
cases that needed that (the figures here are the pre-#931 baselines, not
the regenerated gates):

  * cv18 (``fine_gate_abs`` 0.04, ``richardson_gate_abs`` 0.01) — the fin
    aperture loses its ``- 1``;
  * cv19 (``f0_gate_mhz`` 19.0 from a 12.1230 MHz envelope) — the iris and
    cavity lose their ``+1`` / ``-1`` pair;
  * cv05 / cv06b / cv07 / cv15 — foil becomes a sheet, so every measured
    envelope on those boards is measured on a different board.

NOT on the list, measured rather than assumed: cv16 and cv17 build their
scatterers with the low-level ``rasterize(..., sigma=1e7)`` cell fill, which
design note §1.8 fences out of the contract, so their 3.3 / 4.0 dB and dB
gates do not move and no re-run is scheduled for them.

The no-silent-loosening rule binds here more than anywhere else in the repo:
a gate that grows because its case was re-solved is a measurement; a gate
that grows because the re-derivation was skipped is the failure this module
exists to prevent. ``tests/contracts/test_gate_policy_is_shared.py``
re-derives every discovered case from OUTSIDE its own file, so a half-finished
re-pin fails there rather than locally — which is the intended order.

The bounded-margin lanes (``test_waveguide_broad_e5_tolerance_envelope.py``
and its phase / group-delay siblings) check a structurally different shape
-- a PINNED module constant bounded by ``[worst_measured, worst_measured x
MULTIPLIER]`` rather than a quantized derived value, so they do not call
``gate_from_envelope`` -- they import ``ENVELOPE_GATE_MULTIPLIER`` directly
instead of restating ``1.5`` as a fresh local literal. Sharing the name is
not what makes that binding real, though: a local ``MARGIN_CEIL = 3.0``
plant still passes the source-grep in ``test_gate_policy_is_shared.py`` (the
grep only rejects restating ``1.5``, not any other value, an alias, or a
formatting variant like ``1.50``). What actually binds those three lanes to
this constant is
``test_margin_ceiling_lanes_are_bound_by_the_shared_multiplier_from_outside``
in ``test_gate_policy_is_shared.py``: it independently re-derives each
lane's ``worst`` from the same committed artifacts the lane itself reads,
imports each lane's PINNED constant, and asserts
``worst <= PINNED <= worst * ENVELOPE_GATE_MULTIPLIER`` from OUTSIDE the
lane's own file -- so a coherent in-file plant (widen the local ceiling AND
re-pin the constant to fit under it) is checked against the ONE shared
number regardless of what the plant's own file claims.

A change to ``ENVELOPE_GATE_MULTIPLIER`` here moves every one of the above
at once -- that is the visibility guarantee issue #528 asks for: a per-case
relaxation now requires editing a shared, reviewer-visible object instead of
a local literal. ``tests/contracts/test_gate_policy_is_shared.py`` carries three kinds
of guard, in increasing order of how load-bearing they are: (1) a source
grep that only catches the literal string ``1.5`` (cosmetic -- evadable, see
above); (2) two falsifiers for the quantized-gate lanes -- one execs a
source-mutated copy of this file (the exact "find-replace 1.5 -> 3.0" class
of edit from the #499 review) and one monkeypatches the already-imported
``ENVELOPE_GATE_MULTIPLIER`` on the live module -- both show every real
gated case's derived value moves together and reverting reproduces every
case's frozen CI-pinned gate bit-for-bit; (3) the from-outside numeric
cross-check for the bounded-margin lanes described above. (1) is a tripwire,
not a guarantee; (2) and (3) are the load-bearing checks.

Coordination note: if ``ENVELOPE_GATE_MULTIPLIER`` is ever deliberately
changed, the PROSE restatements of "x 1.5" scattered through docstrings,
comments, and printed diagnostics in the six consumer files above (roughly
30 occurrences, none of them load-bearing -- they are not touched by this
module and nothing here checks them) will read stale, including the
runtime mismatch message in ``validation/crossval/18_wr90_iris_modematch.py``
(``f"must equal round-up(env x 1.5) = {required}"``). Re-derive and update
those by hand in the same change; this module does not do it for you.
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
    time rather than capturing it as a bound default, so it is
    monkeypatchable: patching the module attribute on an ALREADY-IMPORTED
    copy of this module changes every subsequent call made through this
    same function object -- including calls reached via
    ``from tests._gate_policy import gate_from_envelope`` in a consumer
    module, since Python functions resolve globals against their *defining*
    module, not the importer's.
    ``test_gate_policy_is_shared.py`` exercises BOTH falsifier mechanisms
    against this property: ``test_mutating_the_shared_multiplier_...``
    execs a source-mutated COPY of this file (closer to the real "edit the
    file" attack shape, but does not touch any already-imported module) and
    ``test_monkeypatching_the_live_shared_multiplier_...`` patches this
    already-imported module's attribute directly (closer to proving
    consumers read the constant at call time rather than at import time).
    """
    return math.ceil(measured_envelope * ENVELOPE_GATE_MULTIPLIER * quantum) / quantum
