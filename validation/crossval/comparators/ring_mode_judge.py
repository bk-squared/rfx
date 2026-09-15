"""Ring-resonator mode judge for cv02 — assignment decoupled from tolerance.

Plain numpy/scipy. No rfx import, no Simulation, no solve: this module only
compares two lists of extracted modes, so both the crossval script and
``tests/crossval/test_cv02_ring_mode_judge.py`` can drive the same code.

Why this module exists (issue #812)
-----------------------------------
The judge that shipped inside ``02_ring_resonator.py`` matched modes with a
``best_diff < 0.05`` window and then gated ``mean(|df|/f) < 5%`` over exactly
the pairs that window admitted. Every admitted pair therefore had an error
below 5% *by construction*, so the headline gate was entailed by the matcher
and could not fail for its own stated reason. The audit measured it: 200,000
random trials through the verbatim judge, maximum ``mean_err`` ever observed
4.9997%, zero failures of the mean gate.

That judge is kept here verbatim as :func:`legacy_shipped_judge` so the
tautology stays executable and every falsifier can be shown against it.

The replacement, :func:`judge`, separates the two questions:

* *which rfx mode corresponds to this reference mode* — answered by a
  one-to-one assignment minimising total relative frequency distance, with
  **no tolerance anywhere in it**;
* *how far apart are they* — answered afterwards, by gates that no longer
  select their own input.

Gates (all evaluated only when the external reference is present):

============  ==========================================================
``unmatched`` every admitted reference mode receives a distinct rfx
              partner (a reference mode rfx never found is a FAIL, not a
              silently dropped row)
``count``     at least ``min_matched`` (2) reference modes assigned
``mean_err``  mean relative frequency error over ALL assigned pairs < 5%
``max_err``   max relative frequency error over ALL assigned pairs < 5%
``q``         for every mode whose decay the record actually observed,
              the Q interval obtained by transforming the decay-rate
              interval **at that pair's own two frequencies**; it is
              asymmetric when ``tau_ref/T > 0``
============  ==========================================================

    The ``q`` gate is built from THREE separable ingredients, and only one
    of them is derived. They are named individually in
    :data:`Q_GATE_INGREDIENTS`, printed by
    :func:`format_q_window_provenance`, and persisted by the crossval script
    under ``gate_limits.q_gate_ingredients`` — so no reader has to infer
    which part is algebra and which part is a board decision:

    1. ``estimator_uncertainty`` — **declared policy**. The fractional
       decay-rate scale ``s = tau_ref/T`` (:func:`q_window`). Its functional
       form is NOT this estimator's measured error law (#907).
    2. ``rate_to_q_transform`` — **derived**. The exact image of the rate
       interval in log Q (:func:`rate_interval_to_log_q_bounds`), #945.
    3. ``discretization_budget`` — **absent**. No permitted rfx-vs-Meep
       discretization disagreement has ever been declared (#907).

    Because ingredient 3 does not exist, this gate is a two-solver
    **consistency heuristic**, not a Q-accuracy guarantee; do not cite a cv02
    ``q`` PASS as a bound on rfx's Q accuracy.

    **PERMANENT DECLARATION (#907, 2026-09-15).** That reading is the gate's
    standing character, not a placeholder awaiting a number. #907 offered two
    dispositions: derive ingredient 1 and declare ingredient 3, or state
    permanently that the cv02 ``q`` gate is a two-solver **consistency
    envelope**. This module takes the second. Ingredient 3's ``kind`` stays
    ``"absent"``, no floor is introduced, and no judge behaviour changes.

    **What would have to exist before that flips** -- the standard this
    module already recorded, re-measured on 2026-09-15 and unmet on all three
    counts:

    * the rfx matrix-pencil estimator's real SNR / model-order uncertainty
      law on the REAL multi-mode record, not on a clean-exponential proxy;
    * a source-free Meep reference record regenerated under the same
      conditions (today's reference runs Harminv while the source is still
      on, so it violates the free-decay model any uncertainty law assumes);
    * a CONVERGED spatial/timestep ladder against the exact annulus. The
      third is worse than missing: the first ladder anybody ran has rfx's
      ``Q`` on this annulus moving AWAY from the continuum as the mesh
      refines, which is its own open finding.

    Two measurements make this the honest disposition rather than a deferral.
    Both are reproducible from
    ``scripts/diagnostics/cv02_exact_annulus_qnm.py`` and its frozen fixture
    ``tests/fixtures/cv02_ring_judge/exact_annulus_qnm.json``:

    1. **Estimator uncertainty cannot license the observed gap.** The exact
       finite-record Cramer-Rao bound for a damped exponential, evaluated at
       cv02's own ``T`` and sampling rather than in the long-record limit, is
       ``sigma_lnQ ~ tau * sqrt(6 dt / rho) * T**-1.5`` in this regime: about
       ``5e-3`` for the slowest mode at a per-sample SNR ``rho = 1e4``, which
       is ~15x SMALLER than that mode's observed ``|lnQ| = 0.076``. Admitting
       the gap on estimator noise alone would need ``rho`` of order 1 to 45 --
       and ``rho`` is not measured on the real record, which is the first
       missing artifact above. Note what the correct law implies: it shrinks
       with ``T`` FASTER than the ``tau/T`` window does (``T**-1.5`` against
       ``T**-1``), so an honestly derived ingredient 1 would make this gate
       TIGHTER with record length, not looser. That is the sharper form of
       the complaint #907 opened with, and it is why "derive ingredient 1"
       is not a route to a wider gate.
    2. **A frequency-error budget cannot license it either.** The exact
       annulus gives ``|dlnQ/dlnf| = 4.97 / 6.91 / 8.85`` in the INDEX
       channel and ``0`` exactly in the uniform-radius channel
       (``dlnf/dlnR = -1`` and ``dlnQ/dlnR = 0``, Maxwell scale invariance).
       So the leverage is an assumption about which channel a solver's error
       lives in -- it ranges over the whole interval ``[0, 8.85]`` -- not an
       analytic property of the annulus. Even at the largest leverage and the
       most generous frequency error (BOTH solvers' full disagreement with
       the continuum, added) the transport gives ``0.013 / 0.023 / 0.051``
       against observed ``|lnQ|`` of ``0.058 / 0.048 / 0.076``. Transporting
       the rfx-vs-Meep frequency DIFFERENCE instead gives
       ``0.0026 / 0.0021 / 0.0032``, short by 23x.

    Both branches say the same thing: **the observed Q gap is not explained
    by either candidate ingredient.** A floor wide enough to cover it would
    therefore have to be read off the gap, and then the gate would certify
    the agreement it exists to test. Declaring the envelope is what is left,
    and it is a statement about the gate, not a claim about rfx.

Frequencies and the record length must be in reciprocal units (the script
passes both in Meep normalised units: ``f`` in ``c/a``, ``T`` in ``a/c``).
Pre-declaration: ``docs/design_notes/20260831_cv02_ring_judge_predeclaration.md``
— read its **Corrections 4 and 5** with it: the gate form declared there was
superseded twice, both times under #945 (Correction 4(a), the interval
inverts end for end; Correction 5, the inversion is evaluated at each pair's
own two frequencies), and its "the Q window carries no chosen value at all"
claim is withdrawn (#907).
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

# --- pre-declared constants (see the design note; none is fitted here) -------

#: Published gate value (docs/public/guide/benchmarks.mdx), unchanged. Applied
#: to the mean AND, after #812, to each individual mode.
FREQ_TOL_PCT = 5.0

#: Published mode-count gate, unchanged.
MIN_MATCHED = 2

#: A mode's Q is gated only if the record spans at least this many amplitude
#: e-foldings of the REFERENCE mode. Prior-provenance: #812 published
#: ``T/tau = 0.376`` (resolved) and ``0.086`` (not resolved, "must be
#: excluded"); any cut inside that interval implements the published finding,
#: and 1/4 is the round geometric fraction in it (a quarter e-folding = 22%
#: of observed amplitude decay). This cut controls whether the record is long
#: enough to judge Q; it does not impose a finite upper Q bound when the
#: transformed rate interval reaches zero (s >= 1).
Q_RECORD_MIN_EFOLDS = 0.25

#: Mode-admission floor, applied symmetrically to both solvers' harminv output.
MIN_Q = 1.0


@dataclass(frozen=True)
class GateIngredient:
    """One named input to the ``q`` gate, with its epistemic status.

    ``kind`` is the whole point of this dataclass and takes exactly three
    values:

    ``derived``
        follows from stated premises by algebra that can be re-done on paper;
        a reader can check it without trusting a board decision.
    ``declared-policy``
        a chosen envelope. It may have provenance (a published measurement
        that motivated it) without being entailed by one. A policy is not
        wrong; presenting it as derived is.
    ``absent``
        the quantity the gate would need in order to mean what its name
        suggests, which nobody has declared. Naming it keeps the hole
        visible instead of letting the other two ingredients imply it.
    """

    name: str
    kind: str
    quantity: str
    basis: str
    source: str


#: The ``q`` gate's inputs, separated (#907). Read this before quoting a cv02
#: ``q`` verdict: with ``discretization_budget`` absent the gate compares two
#: finite-grid solvers to each other, which is a consistency statement, not an
#: accuracy statement about either one.
Q_GATE_INGREDIENTS: tuple[GateIngredient, ...] = (
    GateIngredient(
        name="estimator_uncertainty",
        kind="declared-policy",
        quantity="s = tau_ref / T   (per reference mode; see q_window)",
        basis=(
            "A record-length-scaled envelope whose whole provenance is #812's "
            "published bracket (T/tau = 0.376 resolved vs 0.086 'must be "
            "excluded'). A second prop - 'the measured degradation of the "
            "decimated path cv02 actually runs' - was cited here until "
            "2026-09-13 and is WITHDRAWN: re-measured on the configuration it "
            "named it does not reproduce, and at the shorter rung no "
            "decimation stage fires at all, so there is no decimated path "
            "there to degrade (ladder + per-rung decimation plans: "
            "tests/fixtures/cv02_ring_judge/harminv_decimation_ladder.json, "
            "pinned by test_the_decimation_penalty_claim_does_not_reproduce). "
            "Whether the REAL multi-mode cv02 record degrades under "
            "decimation is UNMEASURED. The envelope is NOT the estimator's "
            "error law either: on a clean "
            "damped exponential at cv02's sampling density, rfx's "
            "matrix-pencil harminv with decimate=False recovers Q to ~3e-12 "
            "relative at T/tau = 0.0822, so there is no 1/T information "
            "barrier to import. The reference side (Meep filter "
            "diagonalisation) obeys a different law again and is not "
            "modelled separately; Meep's err field is a fit residual, not an "
            "accuracy bound. Building the uncertainty from rfx's own residual "
            "instead is not available either: HarminvMode.error is "
            "1 - exp(-decay * dt_eff) exactly (rfx/harminv.py: "
            "err = 1 - min(|lam|, 1/|lam|) on a decaying pole), i.e. a "
            "monotone function of the reported decay itself, so a tolerance "
            "built from it would scale with the quantity it gates - the "
            "self-referential class #812 catalogued. Pinned by "
            "test_harminv_error_field_is_the_decay_restated."
        ),
        source="#907 (2026-09-07, 2026-09-10 comments); #812 published bracket",
    ),
    GateIngredient(
        name="rate_to_q_transform",
        kind="derived",
        quantity=(
            "ln(Q_rfx/Q_ref) - ln(f_rfx/f_ref) in [-log1p(s), -log1p(-s)], "
            "upper = +inf for s >= 1"
        ),
        basis=(
            "Q = pi f / alpha, so alpha_rfx/alpha_ref = "
            "(f_rfx/f_ref) * (Q_ref/Q_rfx) and the declared rate interval "
            "[1-s, 1+s] maps to Q_rfx/Q_ref in "
            "[(f_rfx/f_ref)/(1+s), (f_rfx/f_ref)/(1-s)]. Exact algebra, no "
            "choice in it. TWO corrections got it here and both are in that "
            "one line. (i) Q is monotone DECREASING in alpha, so the interval "
            "inverts end for end; the symmetric +-log1p(s) window this "
            "replaced rejected a mode whose rate differed by exactly the "
            "tolerance the gate claimed to allow, and stayed finite where the "
            "transformed interval is unbounded. (ii) The inversion holds at "
            "FIXED frequency, so the transform is applied at each mode's own "
            "realized frequency pair: without the ln(f_rfx/f_ref) term a "
            "frequency error admitted by the 5% freq gate was charged to the "
            "Q gate, and a mode whose decay rate was exactly the reference's "
            "failed it (4.9% low in f, alpha ratio 1.000000, s = 0.02 -> "
            "ln(Q_rfx/Q_ref) = -0.0502 against bounds [-0.0198, +0.0202]). "
            "The tolerance s itself is still reference-only (q_window); what "
            "the frequency term enters is the COMPARAND -- the decay-rate "
            "ratio, which is a function of both measured pairs because a rate "
            "is. Pinned by test_issue945_a_frequency_error_is_not_charged_to_"
            "the_q_gate."
        ),
        source="#945 (PR #999 for (i); reopened 2026-09-13 for (ii)); "
               "rate_interval_to_log_q_bounds + log_frequency_term",
    ),
    GateIngredient(
        name="discretization_budget",
        kind="absent",
        quantity="(none declared)",
        basis=(
            "Nothing here states how far two finite-grid solvers with "
            "different boundary rasterisations are ALLOWED to disagree on a "
            "radiation Q. A budget cannot be back-filled from the observed "
            "gap without the gate certifying the agreement it is supposed to "
            "test. Deriving one needs the rfx estimator's real SNR / "
            "model-order uncertainty, a source-free Meep reference record "
            "regenerated under the same conditions, and a CONVERGED "
            "spatial/timestep ladder against the exact annulus - a "
            "pre-declared campaign, not a number chosen here. "
            "DECLARED PERMANENT 2026-09-15 (#907): this kind stays 'absent' "
            "and the q gate is a two-solver consistency envelope, because "
            "neither candidate ingredient accounts for the gap it would have "
            "to license. The exact finite-record Cramer-Rao bound at cv02's "
            "own T and sampling is ~5e-3 in |lnQ| for the slowest mode at "
            "per-sample SNR 1e4, ~15x below that mode's observed 0.076; and "
            "it falls as T**-1.5, i.e. FASTER than this gate's tau/T scale, "
            "so a derived ingredient 1 tightens the gate with record length "
            "rather than widening it. Transporting the measured frequency "
            "disagreement through the exact annulus is short too: the "
            "rfx-vs-Meep frequency difference carries 0.0026/0.0021/0.0032 "
            "in |lnQ| (23x short of 0.058/0.048/0.076) and even both "
            "solvers' full error against the continuum, added and taken at "
            "the largest of the two leverage channels, carries only "
            "0.013/0.023/0.051. The leverage itself is a CHOICE: the annulus "
            "gives |dlnQ/dlnf| = 4.97/6.91/8.85 for an index error and "
            "exactly 0 for a uniform radius error (dlnf/dlnR = -1, "
            "dlnQ/dlnR = 0), so it spans [0, 8.85] and a floor built on it "
            "declares an error channel rather than deriving one. Oracle and "
            "frozen numbers: scripts/diagnostics/cv02_exact_annulus_qnm.py, "
            "tests/fixtures/cv02_ring_judge/exact_annulus_qnm.json."
        ),
        source="#907 closing decision 2026-09-13; permanent declaration "
               "2026-09-15 (option (b) of the issue's own two)",
    ),
)

#: One-line reading of the table above, quoted by the report and the artifact.
Q_GATE_CHARACTER = (
    "PERMANENT two-solver consistency envelope - a limited consistency "
    "heuristic, NOT a Q-accuracy guarantee. No discretization budget is "
    "declared, none is being back-filled from the observed gap, and this is "
    "the gate's standing character, not a placeholder: #907, declared "
    "2026-09-15. Q_GATE_INGREDIENTS[2] records the three artifacts that "
    "would have to exist before it changes"
)


@dataclass(frozen=True)
class ReferenceMode:
    """One external-solver (Meep harminv) mode."""

    freq: float
    Q: float


@dataclass(frozen=True)
class SolverMode:
    """One rfx harminv mode."""

    freq: float
    Q: float
    amplitude: float = 1.0


@dataclass
class PairRow:
    """One reference mode and the rfx mode assigned to it (or none)."""

    ref_freq: float
    ref_Q: float
    rfx_freq: float | None = None
    rfx_Q: float | None = None
    freq_err_pct: float | None = None
    t_over_tau: float = 0.0
    q_window: float = float("inf")
    q_log_ratio: float | None = None
    #: SIGNED ``ln(Q_rfx/Q_ref)``. The gate's interval is asymmetric (#945),
    #: so the side matters: ``q_log_ratio`` (absolute) cannot say which bound
    #: a row is near, and a row can sit 2.76x closer to one end than the
    #: other. Kept beside the bounds that judged it so the verdict is
    #: re-derivable from the retained record without re-running the judge.
    q_log_ratio_signed: float | None = None
    #: ``ln(f_rfx/f_ref)`` -- the term the fixed-frequency inversion assumed
    #: away (#945, second correction). The declared interval bounds the
    #: DECAY-RATE ratio, and ``alpha = pi f / Q`` carries both frequencies, so
    #: the Q bounds below are this row's transform shifted by this number.
    #: Reported so a reader can see what was subtracted rather than having to
    #: re-derive it from the two frequency columns.
    q_log_freq_term: float | None = None
    #: ``ln(alpha_rfx/alpha_ref) = q_log_freq_term - q_log_ratio_signed`` --
    #: the quantity the declared interval ``[1-s, 1+s]`` actually bounds,
    #: stated directly so the gate can be checked against ``q_window`` without
    #: re-doing the algebra. The verdict is evaluated in log-Q space (the
    #: bounds below), which is the same inequality; only the last bit can
    #: differ between the two routes, exactly on a boundary.
    q_log_rate_ratio_signed: float | None = None
    #: The bounds that judged this row: the transform of ``q_window``
    #: SHIFTED by ``q_log_freq_term``. Not ``rate_interval_to_log_q_bounds``
    #: of ``q_window`` alone unless the two frequencies happen to coincide.
    q_log_lower: float | None = None
    q_log_upper: float | None = None
    q_gated: bool = False
    q_pass: bool | None = None

    @property
    def matched(self) -> bool:
        return self.rfx_freq is not None


@dataclass
class Verdict:
    """Full outcome: per-mode rows, the gate booleans, and the numbers."""

    rows: list[PairRow] = field(default_factory=list)
    surplus: list[SolverMode] = field(default_factory=list)
    record_length: float = 0.0
    n_matched: int = 0
    n_unmatched: int = 0
    mean_err_pct: float | None = None
    max_err_pct: float | None = None
    gates: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(self.gates.values())

    @property
    def q_gated_rows(self) -> list[PairRow]:
        return [row for row in self.rows if row.q_gated]


# --- pieces, each independently testable ------------------------------------


def admit(modes, f_min: float, f_max: float, min_Q: float = MIN_Q) -> list:
    """Keep modes inside the harminv search band with ``Q > min_Q``.

    Applied to BOTH sides. The shipped script filtered the rfx side only, so a
    reference-side harminv artefact with ``Q < 1`` used to enter the comparison
    as a full-weight mode.
    """
    return [
        mode
        for mode in modes
        if mode.Q > min_Q and f_min <= mode.freq <= f_max
    ]


def assign(ref_freqs, rfx_freqs) -> list[int | None]:
    """One-to-one nearest assignment of rfx modes to reference modes.

    Minimises the total relative frequency distance
    ``|f_rfx - f_ref| / f_ref`` with :func:`scipy.optimize.linear_sum_assignment`.

    **No tolerance enters here.** That is the whole point: the shipped matcher
    used the same 5% number it later gated, so the gate was its own filter's
    consequence. When rfx has fewer modes than the reference, the assignment
    drops the reference mode it can serve worst, and that mode comes back as
    ``None`` — an explicit unmatched-mode failure rather than a shorter list.

    Returns one entry per reference mode: the index into ``rfx_freqs``, or
    ``None`` if that reference mode got no partner.
    """
    n_ref = len(ref_freqs)
    out: list[int | None] = [None] * n_ref
    if n_ref == 0 or len(rfx_freqs) == 0:
        return out
    ref = np.asarray(ref_freqs, dtype=float)[:, None]
    rfx = np.asarray(rfx_freqs, dtype=float)[None, :]
    cost = np.abs(rfx - ref) / np.abs(ref)
    rows, cols = linear_sum_assignment(cost)
    for row, col in zip(rows, cols):
        out[int(row)] = int(col)
    return out


def q_window(ref_freq: float, ref_Q: float, record_length: float
             ) -> tuple[float, float]:
    """DECLARED-POLICY decay-rate scale ``s`` for one REFERENCE mode.

    **This is ingredient 1 of three (:data:`Q_GATE_INGREDIENTS`), and it is
    the declared one.** It returns the fractional decay-rate scale the gate
    is willing to spend, not a measured uncertainty. With amplitude decay
    rate ``alpha = pi f / Q`` (e-folding time ``tau = Q / (pi f)``), a
    fractional rate scale of ``1/T`` per unit record reads::

        s = delta_alpha / alpha = (1/T) / (pi f / Q) = tau / T

    Both inputs are the reference's; no measured rfx quantity appears, so the
    scale is not fitted to the agreement it judges. That is its one real
    virtue, and it is separate from the question of whether ``1/T`` is the
    right law.

    **It is not.** The argument this docstring used to make -- that a record
    of length ``T`` cannot resolve decay rates finer than ``1/T``, by analogy
    with Fourier frequency resolution -- was measured and refuted (#907,
    2026-09-10). A damped exponential fixes its exponent from adjacent-sample
    ratios; there is no Fourier separation limit to import. On a clean
    synthetic exponential at cv02's own sampling density, rfx's matrix-pencil
    harminv with ``decimate=False`` returns Q to ~3e-12 relative error at
    ``T/tau = 0.0822`` -- a third of :data:`Q_RECORD_MIN_EFOLDS`.

    **A second empirical prop is WITHDRAWN (2026-09-13).** This docstring
    briefly read "what DOES degrade at short records is the decimated path
    cv02 actually runs (3.49% there, 0.24% at the 0.25 cut)". Re-measured on
    the configuration that sentence names -- #907's synthetic single damped
    exponential at cv02's own ``dt`` -- it does not reproduce, and it cannot:
    at ``T/tau = 0.0822`` the record is too short for any decimation stage to
    leave the requested pencil capacity, so ``decimate='auto'`` decimates by
    nothing and there is no decimated path to measure there. On the rungs
    where a stage does fire, the decimated path is not worse than
    ``decimate=False``. The ladder, its per-rung decimation plans and every
    number are in
    ``tests/fixtures/cv02_ring_judge/harminv_decimation_ladder.json``, pinned
    by ``test_the_decimation_penalty_claim_does_not_reproduce``; the
    withdrawn figures are recorded there under ``withdrawn_claim`` so the
    retraction is checkable and not just an absence.

    That leaves #812's published bracket as this envelope's whole provenance.
    Whether the REAL cv02 record -- multi-mode, with source contamination
    still inside the analysed window -- degrades under decimation is
    UNMEASURED; the synthetic ladder is a clean single exponential and does
    not settle it. So ``tau/T`` is a policy envelope with one published
    bracket behind it, not a derived bound, and this function's name is
    historical.

    **Known limitation -- this scale behaves like a RESOLUTION bound, not an
    accuracy bound, and it therefore shrinks with run length while the physics
    does not.** ``tau/T`` scales with how finely a record of length ``T``
    separates two decay rates; it says nothing about how far apart two
    *solvers* should be.
    The rfx-vs-Meep Q gap on cv02 does not shrink with ``T``, and WHAT
    produces it is UNRESOLVED -- including whether any of it is the two
    estimators rather than the two discretizations, since rfx's matrix pencil
    and Meep's filter diagonalisation have different and unmodelled error laws.
    This paragraph used to assert a staircased ring boundary and subpixel
    treatment as the cause; #907 (2026-09-10) retracted
    that as an overclaim, on a counterexample that moves Q while keeping all
    three frequencies inside the two solvers' observed mutual agreement, so
    the frequency agreement cannot pin the geometry and the log decomposition
    cannot settle the attribution. rfx's own Q for modes 2 and 3 is
    stable over every RESOLVED span that was measured (mode 1's recorded
    readings spread ~7% across T=291/561/1101 and are NOT cited as invariance evidence). Measured ``|ln(Q_rfx/Q_ref)| = 0.070`` (mode 1)
    and ``0.123`` (mode 2); rfx mode 2 reads ``Q = 357.61 -> 356.83`` (0.22%)
    between ``T = 291`` and ``T = 1101``
    (``docs/research_notes/audit-2026-09-02/verify/G2_cv02.md``), and the
    slowest in-band mode (``f = 0.1753``) reads ``Q = 1787.6 @ T = 1575 ->
    1757.3 @ T = 3281`` (1.7%) -- both RESOLVED readings, rungs 1-2 of the
    recorded Meep-absent run
    ``docs/research_notes/audit-2026-09-02/fix2/i4_PR896_cv02_meep_absent.log``.
    (That run's bootstrap reading ``Q = 1686.9 @ T = 385`` is deliberately not
    quoted as invariance evidence: at ``T/tau = 0.126`` this module's own floor
    calls it UNRESOLVED, i.e. not a measurement.) Consequently, on cv02's
    committed reference/rfx pair this gate PASSES at ``T=291`` (mode-1 window
    0.747) and FAILS at ``T=3385`` (window 0.064) purely because the record got
    longer and better settled. A longer record reds a physically stable case.
    Measured across ``T in {260.98, 291, 3385, 15600, 1e6}`` on both boards
    the ``q`` gate is False in 5 of those 10 cells, and it is the ONLY gate
    that moves: ``unmatched``, ``count``, ``mean_err``, ``max_err`` and every
    row's ``|lnQ|`` are identical at every ``T``
    (``test_the_q_gate_is_the_only_gate_that_moves_with_record_length``).

    **This is not being repaired, and that is a decision, not a deferral**
    (#907, 2026-09-15). The gate is declared a two-solver consistency
    envelope; see the module docstring for the two measurements behind it and
    :data:`Q_GATE_INGREDIENTS` for the three artifacts that would have to
    exist before ingredient 3 could be declared. In short: the exact
    finite-record Cramer-Rao bound at cv02's own ``T`` and sampling puts the
    estimator floor ~15x BELOW the observed ``|lnQ|`` gap, and it falls as
    ``T**-1.5``, i.e. faster than this ``tau/T`` scale -- so deriving
    ingredient 1 honestly would TIGHTEN this gate with record length, not
    widen it. The frequency-error route is short by 1.5x to 23x depending on
    which error the budget is built from, and the leverage it would be
    transported through spans ``[0, 8.85]`` depending on an assumed error
    channel. A floor big enough to cover the gap would have to be read off
    the gap, which would make the gate certify the agreement it exists to
    test; that was refused in 2026-09 and is refused again here. The gate's
    ``T``-contingency therefore stands as declared behaviour, and it remains
    the reason cv02's Meep (verdict) lane keeps its calibrated record length
    instead of the tau-scaled one.

    ``window`` is retained as the scalar rate scale ``s`` for callers and
    reports.  The Q gate itself uses the exact, asymmetric transformed
    interval from :func:`q_log_bounds` -- that transform (ingredient 2) IS
    derived, and its exactness says nothing about ingredient 1's provenance.
    Ingredient 3, the permitted discretization disagreement, does not exist:
    see :data:`Q_GATE_CHARACTER`.

    Returns ``(T/tau, window)``. ``T/tau`` is the number of amplitude
    e-foldings the record observed; a mode is Q-gated only when it reaches
    :data:`Q_RECORD_MIN_EFOLDS`.
    """
    tau = ref_Q / (math.pi * ref_freq)
    if tau <= 0 or record_length <= 0:
        return 0.0, float("inf")
    t_over_tau = record_length / tau
    return t_over_tau, tau / record_length


def log_frequency_term(ref_freq: float, rfx_freq: float | None) -> float:
    """``ln(f_rfx/f_ref)`` for one assigned pair -- what the fixed-frequency
    inversion assumed away (#945, second correction).

    The Q gate's declared interval bounds the DECAY-RATE ratio, and a rate
    carries a frequency: ``alpha = pi f / Q`` gives

        alpha_rfx / alpha_ref  =  (f_rfx / f_ref) * (Q_ref / Q_rfx)

    so ``ln(alpha_rfx/alpha_ref) = ln(f_rfx/f_ref) - ln(Q_rfx/Q_ref)``. Reading
    the Q ratio alone against a fixed-frequency interval charges the frequency
    disagreement -- which cv02 gates separately, at 5% -- to the Q gate.

    Returns ``0.0`` when there is no partner (``rfx_freq is None``), i.e. the
    unshifted fixed-frequency transform; an unmatched row is not Q-gated
    anyway. Raises ``ValueError`` on a non-positive or non-finite frequency:
    a mode at ``f <= 0`` has no rate ratio, and taking ``abs`` or clamping
    would invent one.
    """
    if rfx_freq is None:
        return 0.0
    for name, value in (("ref_freq", ref_freq), ("rfx_freq", rfx_freq)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be finite and > 0 to form ln(f_rfx/f_ref), "
                f"got {value!r}")
    return math.log(rfx_freq / ref_freq)


def rate_interval_to_log_q_bounds(s: float, log_freq_ratio: float = 0.0
                                 ) -> tuple[float, float]:
    """The transform (ingredient 2). Pure, exact, DERIVED -- the whole of #945.

    This is the one ingredient of the ``q`` gate that is derived rather than
    declared: everything below follows from ``Q = pi f / alpha`` by algebra a
    reader can redo on paper. Its exactness says nothing about ingredient 1's
    provenance (see :func:`q_window`) or about ingredient 3's absence.

    Takes the fractional decay-rate scale ``s >= 0`` -- i.e. the declared
    admissible rate interval ``alpha_rfx/alpha_ref in [1-s, 1+s]`` -- together
    with this pair's ``log_freq_ratio = ln(f_rfx/f_ref)``, and returns the
    exact image of that interval in ``ln(Q_rfx/Q_ref)``.

    ``Q = pi f / alpha``, so ``Q`` is monotone **decreasing** in ``alpha`` and
    the interval inverts end for end::

        alpha/alpha_ref in [1-s, 1+s]
            =>  Q_rfx/Q_ref in [ (f_rfx/f_ref)/(1+s), (f_rfx/f_ref)/(1-s) ]

    In logs that is ``[-log1p(s), -log1p(-s)]`` shifted by
    ``log_freq_ratio``: asymmetric for every ``s > 0``, because
    ``(1+s)(1-s) = 1-s^2 < 1``.  The lower (rfx too lossy) side is the one the
    old symmetric ``+-log1p(s)`` window got right.

    **The frequency term is not optional algebra** (#945, reopened
    2026-09-13). The inversion above holds at one frequency; the two solvers
    report two. cv02 admits a 5% frequency disagreement on its own gate, and
    with ``log_freq_ratio`` dropped every bit of that disagreement lands on
    the Q gate instead: at ``s = 0.02``, a mode 4.9% low in frequency whose
    decay rate is EXACTLY the reference's reads
    ``ln(Q_rfx/Q_ref) = -0.0502`` against bounds ``[-0.0198, +0.0202]`` and
    fails. The default ``0.0`` keeps the pure fixed-frequency image available
    for tests and for callers who have no partner frequency; the judge always
    passes the measured one.

    The scale ``s`` is still built from the reference alone
    (:func:`q_window`) -- the frequency term enters the COMPARAND, not the
    tolerance, so the envelope is still not fitted to the agreement it judges.

    ``s >= 1`` is not an edge case here -- cv02's committed mode 2 runs at
    ``s = 2.83``.  The admissible rate interval then reaches zero, an
    arbitrarily small rate is admitted, and the high-Q side has **no finite
    bound**: the function returns ``+inf``, and the low-Q side stays bounded
    by the positive-rate edge ``1+s``.  Capping it at ``1+s`` (what the code
    did before #945) rejected modes the stated policy admits -- at
    ``s = 0.798`` the cap was 2.76x too small, and above ``s = 1`` it
    manufactured a bound the premise does not contain.

    ``s = inf`` (no usable record) falls out as ``(-inf, +inf)``, i.e. no
    restriction, without a special case.

    Raises ``ValueError`` for ``s < 0`` or NaN: a negative rate scale is not
    an interval, and silently taking ``abs`` would hide a caller's sign bug.
    """
    if math.isnan(s) or s < 0.0:
        raise ValueError(f"rate scale s must be >= 0 and not NaN, got {s!r}")
    if not math.isfinite(log_freq_ratio):
        raise ValueError(
            "log_freq_ratio must be finite (it is ln(f_rfx/f_ref) for one "
            f"assigned pair), got {log_freq_ratio!r}")
    lower = -math.log1p(s) + log_freq_ratio
    upper = (float("inf") if s >= 1.0
             else -math.log1p(-s) + log_freq_ratio)
    return lower, upper


def q_log_bounds(ref_freq: float, ref_Q: float, record_length: float,
                 *, rfx_freq: float | None = None) -> tuple[float, float]:
    """Log-Q bounds for one assigned pair: policy scale, then transform.

    Composes the two ingredients that exist -- :func:`q_window` (declared
    policy, ingredient 1) and :func:`rate_interval_to_log_q_bounds` (derived,
    ingredient 2), the latter evaluated at this pair's own frequencies via
    :func:`log_frequency_term`.  Ingredient 3 (a discretization budget) is
    absent, so these bounds do not encode any permitted solver-vs-solver
    disagreement; see :data:`Q_GATE_CHARACTER`.

    ``rfx_freq`` is the assigned rfx mode's frequency. Omitting it returns
    the fixed-frequency image, which is the right answer only when the two
    frequencies agree -- the judge always passes the measured one (#945).

    A non-positive record gives ``s = inf`` and hence an unrestrictive pair
    (the frequency shift leaves ``(-inf, +inf)`` unchanged). Such a row is not
    Q-gated by :func:`judge` anyway, because its ``T/tau`` is below
    :data:`Q_RECORD_MIN_EFOLDS`.
    """
    _t_over_tau, window = q_window(ref_freq, ref_Q, record_length)
    return rate_interval_to_log_q_bounds(
        window, log_frequency_term(ref_freq, rfx_freq))


def format_q_window_provenance() -> str:
    """Print which ingredients of the ``q`` gate are derived and which are
    declared policy (#907).

    The gate's number used to arrive with a derivation attached to all of it.
    This block is the correction: one line per ingredient, its epistemic
    ``kind`` first, so a reader of the crossval log or of the retained
    artifact sees the split without reading the module.
    """
    lines = ["  Q gate ingredients (#907) — what is derived, what is policy:"]
    for item in Q_GATE_INGREDIENTS:
        lines.append(f"    [{item.kind:>15}] {item.name}: {item.quantity}")
        lines.extend(textwrap.wrap(f"basis: {item.basis}", width=76,
                                   initial_indent=" " * 8,
                                   subsequent_indent=" " * 15))
        lines.append(f"        source: {item.source}")
    lines.extend(textwrap.wrap(f"=> the q gate is a {Q_GATE_CHARACTER}",
                               width=76, initial_indent=" " * 4,
                               subsequent_indent=" " * 7))
    return "\n".join(lines)


def judge(
    reference: list[ReferenceMode],
    rfx_modes: list[SolverMode],
    record_length: float,
    *,
    f_min: float,
    f_max: float,
    freq_tol_pct: float = FREQ_TOL_PCT,
    min_matched: int = MIN_MATCHED,
    q_record_min_efolds: float = Q_RECORD_MIN_EFOLDS,
) -> Verdict:
    """Judge an rfx mode list against an external-solver mode list."""
    ref = admit(reference, f_min, f_max)
    rfx = admit(rfx_modes, f_min, f_max)

    pairing = assign([m.freq for m in ref], [m.freq for m in rfx])
    used = {i for i in pairing if i is not None}

    verdict = Verdict(record_length=record_length)
    verdict.surplus = [m for i, m in enumerate(rfx) if i not in used]

    errs: list[float] = []
    for ref_mode, idx in zip(ref, pairing):
        t_over_tau, window = q_window(ref_mode.freq, ref_mode.Q, record_length)
        row = PairRow(
            ref_freq=ref_mode.freq,
            ref_Q=ref_mode.Q,
            t_over_tau=t_over_tau,
            q_window=window,
            q_gated=t_over_tau >= q_record_min_efolds,
        )
        if idx is not None:
            partner = rfx[idx]
            row.rfx_freq = partner.freq
            row.rfx_Q = partner.Q
            row.freq_err_pct = (
                abs(partner.freq - ref_mode.freq) / abs(ref_mode.freq) * 100.0
            )
            errs.append(row.freq_err_pct)
            if (row.q_gated and partner.Q > 0 and ref_mode.Q > 0
                    and partner.freq > 0 and ref_mode.freq > 0):
                signed_log_ratio = math.log(partner.Q / ref_mode.Q)
                row.q_log_ratio = abs(signed_log_ratio)
                row.q_log_ratio_signed = signed_log_ratio
                # The declared interval bounds the decay-RATE ratio, and
                # alpha = pi f / Q carries both frequencies (#945). Gate in
                # log-Q space against the transform shifted by this pair's own
                # ln(f_rfx/f_ref), so the row's stored bounds reproduce its own
                # verdict; the rate ratio is reported beside it.
                freq_term = log_frequency_term(ref_mode.freq, partner.freq)
                row.q_log_freq_term = freq_term
                row.q_log_rate_ratio_signed = freq_term - signed_log_ratio
                q_lower, q_upper = q_log_bounds(
                    ref_mode.freq, ref_mode.Q, record_length,
                    rfx_freq=partner.freq,
                )
                row.q_log_lower = q_lower
                row.q_log_upper = q_upper
                row.q_pass = q_lower <= signed_log_ratio <= q_upper
            elif row.q_gated:
                row.q_pass = False
        verdict.rows.append(row)

    verdict.n_matched = len(errs)
    verdict.n_unmatched = len(ref) - len(errs)
    if errs:
        verdict.mean_err_pct = float(np.mean(errs))
        verdict.max_err_pct = float(np.max(errs))

    verdict.gates = {
        "unmatched": verdict.n_unmatched == 0 and len(ref) > 0,
        "count": verdict.n_matched >= min_matched,
        "mean_err": (
            verdict.mean_err_pct is not None
            and verdict.mean_err_pct < freq_tol_pct
        ),
        "max_err": (
            verdict.max_err_pct is not None
            and verdict.max_err_pct < freq_tol_pct
        ),
        "q": all(
            row.q_pass is True for row in verdict.rows if row.q_gated
        ),
    }
    return verdict


def format_report(verdict: Verdict, freq_tol_pct: float = FREQ_TOL_PCT) -> str:
    """Human-readable table + gate lines, for the crossval script's stdout."""
    lines: list[str] = []
    lines.append(
        f"  harminv record length T = {verdict.record_length:.1f} "
        f"(Meep units); the 'Q window' column is the DECLARED rate scale "
        f"s = tau_ref/T"
    )
    lines.append(
        "  (the gate is the transformed interval [-log1p(s), -log1p(-s)] "
        "shifted by that pair's ln(f_rfx/f_ref), printed per row as "
        "'Q bounds'; the interval bounds ln(alpha_rfx/alpha_ref), #945)"
    )
    lines.append("")
    lines.append(
        f"  {'ref freq':>10} {'ref Q':>9} {'rfx freq':>10} {'rfx Q':>9} "
        f"{'df/f (%)':>9} {'T/tau':>7} {'Q window':>9} {'Q':>10}"
    )
    for row in verdict.rows:
        if not row.matched:
            lines.append(
                f"  {row.ref_freq:>10.6f} {row.ref_Q:>9.1f} "
                f"{'--':>10} {'--':>9} {'UNMATCHED':>9} "
                f"{row.t_over_tau:>7.3f} {'--':>9} {'--':>10}"
            )
            continue
        if not row.q_gated:
            q_note = "not gated"
            window = "--"
        else:
            q_note = "PASS" if row.q_pass else "FAIL"
            window = f"{row.q_window:>9.3f}"
        lines.append(
            f"  {row.ref_freq:>10.6f} {row.ref_Q:>9.1f} "
            f"{row.rfx_freq:>10.6f} {row.rfx_Q:>9.1f} "
            f"{row.freq_err_pct:>9.3f} {row.t_over_tau:>7.3f} "
            f"{window:>9} {q_note:>10}"
        )
        if row.q_gated and row.q_log_ratio_signed is not None:
            upper = ("+inf" if row.q_log_upper == float("inf")
                     else f"{row.q_log_upper:+.4f}")
            lines.append(
                f"  {'':>10} {'':>9} {'':>10} {'':>9} "
                f"{'Q bounds':>9} signed ln(Q_rfx/Q_ref) = "
                f"{row.q_log_ratio_signed:+.4f} in "
                f"[{row.q_log_lower:+.4f}, {upper}]"
            )
            lines.append(
                f"  {'':>10} {'':>9} {'':>10} {'':>9} "
                f"{'(bounds':>9} shifted by ln(f_rfx/f_ref) = "
                f"{row.q_log_freq_term:+.6f}; declared quantity "
                f"ln(alpha_rfx/alpha_ref) = "
                f"{row.q_log_rate_ratio_signed:+.4f})"
            )
    for mode in verdict.surplus:
        lines.append(
            f"  {'--':>10} {'--':>9} {mode.freq:>10.6f} {mode.Q:>9.1f} "
            f"{'SURPLUS':>9} {'--':>7} {'--':>9} {'reported':>10}"
        )
    lines.append("")
    for name, ok in verdict.gates.items():
        lines.append(f"  {'PASS' if ok else 'FAIL'}: gate {name}")
    if verdict.mean_err_pct is not None:
        lines.append(
            f"  mean df/f = {verdict.mean_err_pct:.3f}% , "
            f"max df/f = {verdict.max_err_pct:.3f}% "
            f"(gate {freq_tol_pct:.1f}% on both, over ALL "
            f"{verdict.n_matched} assigned pairs)"
        )
    if verdict.n_unmatched:
        lines.append(
            f"  {verdict.n_unmatched} reference mode(s) UNMATCHED — rfx "
            f"produced no counterpart"
        )
    ungated = [row for row in verdict.rows if not row.q_gated]
    if ungated:
        lines.append(
            "  Q not gated for "
            + ", ".join(f"f={row.ref_freq:.6f} (T/tau={row.t_over_tau:.3f})"
                        for row in ungated)
            + f" — record spans < {Q_RECORD_MIN_EFOLDS} e-folding; gating "
              "these would measure run length, not physics (#812)"
        )
    lines.append("")
    lines.append(format_q_window_provenance())
    return "\n".join(lines)


# --- per-mode ring-down settling witness (cv02 audit G2) --------------------
#
# Why this lives here and not inline in the script: it is pure array math on
# an extracted mode list and a recorded signal (no rfx import, no solve), so
# the crossval script and ``tests/crossval/test_cv02_ring_mode_judge.py`` drive
# exactly the same witness code, the same way they share the judge above.
#
# The repo rule (``rfx/CLAUDE.md`` "Ring-down settling witness"): a
# claims-bearing Harminv/DFT number taken in an open (CPML) domain must be
# quoted together with how far below the post-source peak the record's end
# energy sits, because a fixed run length can truncate a high-Q ring-down and
# fake a clean spectrum. cv02 is such a structure (open UPML, Harminv modes),
# and it recorded no witness. It is also multi-Q, so ONE global end/peak dB is
# set entirely by the slowest-decaying mode and says nothing about the faster
# ones -- hence a PER-MODE witness derived from each mode's own decay.

#: Energy (amplitude^2) ring-down in dB per amplitude e-folding time ``tau``.
#: A mode whose amplitude envelope is ``A0 * exp(-t/tau)`` carries energy
#: proportional to ``exp(-2 t/tau)``; after a free-decay span ``T`` its energy
#: relative to its own peak is ``exp(-2 T/tau)``, i.e.
#: ``10*log10(exp(-2 T/tau)) = (T/tau) * 10*log10(e**-2)`` dB. The coefficient
#: ``10*log10(e**-2) = -8.6859 dB`` per e-folding is a property of exponential
#: decay, geometry-independent; the only per-board inputs are the mode's own
#: extracted ``(f, Q)`` and the run's own free-decay record length.
ENERGY_DB_PER_EFOLD = 10.0 * math.log10(math.e ** -2)


def amplitude_tau(freq: float, Q: float) -> float:
    """Amplitude e-folding time ``tau = Q / (pi f)`` of one mode.

    The same definition the Q window uses (see :func:`q_window`): amplitude
    decay rate ``alpha = pi f / Q`` gives e-folding time ``tau = 1/alpha``.
    Units follow the inputs (Hz -> s, ``c/a`` -> ``a/c``). Returns ``inf`` for
    a non-decaying (``Q<=0``) or non-physical (``f<=0``) mode.
    """
    if freq <= 0 or Q <= 0:
        return float("inf")
    return Q / (math.pi * freq)


def slowest_amplitude_tau(modes) -> float | None:
    """Largest amplitude e-folding time over ``modes`` -- the slowest-decaying
    (highest-Q) mode.

    That mode alone sets the record length a run needs to observe a target
    number of e-foldings of *every* mode, because a record that gives the
    slowest mode ``k`` e-foldings gives every faster mode more. ``modes`` is
    any iterable of objects with ``.freq`` and ``.Q``. Returns ``None`` when no
    mode carries a finite positive tau, so the caller can fall back instead of
    scaling off nothing.
    """
    taus = [amplitude_tau(m.freq, m.Q) for m in modes]
    taus = [t for t in taus if math.isfinite(t) and t > 0]
    return max(taus) if taus else None


def record_length_for_efolds(modes, target_efolds: float) -> float | None:
    """Free-decay record length that observes ``target_efolds`` amplitude
    e-foldings of the SLOWEST mode (and at least that many of every faster
    mode).

    Returns ``target_efolds * slowest_amplitude_tau(modes)`` in the modes' own
    time units, or ``None`` when no mode sets a tau.

    **Unbounded primitive -- do not drive a run length with it directly.**
    ``max(tau)`` over a raw harminv mode list is exactly the quantity a
    band-edge artefact corrupts: harminv searches a 10%-widened band, and a
    mode sitting at the edge of it can report a Q three orders of magnitude
    away from its value on a different window (measured on cv02: f=0.2027 read
    Q=1.0e3 on one record and Q=1.0e6 on another), which would ask for a
    record ~4500x the committed one. Use :func:`plan_record`, which feeds this
    primitive only modes inside the judge's own :func:`admit` band whose decay
    the present record actually resolved, and clamps the answer to what that
    record can justify (:func:`resolvable_tau_bound`).
    """
    tau = slowest_amplitude_tau(modes)
    if tau is None:
        return None
    return float(target_efolds) * tau


def resolvable_tau_bound(record_after_source: float,
                         min_efolds: float = Q_RECORD_MIN_EFOLDS) -> float:
    """Largest amplitude e-folding time a record of this length can be said to
    have MEASURED (same units as the record).

    No new number enters. #812 published, and this module gates on,
    :data:`Q_RECORD_MIN_EFOLDS`: a Q read off a record spanning fewer than that
    many amplitude e-foldings of the mode has not observed the decay and must
    not be trusted. Inverting the same inequality ``T/tau >= min_efolds``
    gives ``tau <= T / min_efolds``. A tau above that bound is a lower bound,
    not a measurement, so (a) it must not set a run length, and (b) the bound
    itself is the longest record the present record can justify asking for.
    """
    if record_after_source <= 0 or min_efolds <= 0:
        return 0.0
    return float(record_after_source) / float(min_efolds)


@dataclass(frozen=True)
class RecordPlan:
    """One rung of the free-decay record-length ladder (see :func:`plan_record`)."""

    length: float                  # free-decay record to run next
    cap: float                     # resolvable_tau_bound of the present record
    present: float                 # the present record's free-decay length
    slowest_tau: float | None      # slowest tau this record actually resolved
    kept: tuple                    # in-band modes whose decay this record saw
    out_of_band: tuple             # modes outside [f_min, f_max]
    below_min_q: tuple             # IN-band modes rejected by the MIN_Q floor
    unresolved: tuple              # in-band modes with tau above the cap
    reason: str

    @property
    def extend(self) -> bool:
        """True when the next record is longer than the present one."""
        return self.length > self.present


def plan_record(modes, *, f_min: float, f_max: float,
                record_after_source: float, target_efolds: float,
                min_Q: float = MIN_Q,
                min_efolds: float = Q_RECORD_MIN_EFOLDS) -> RecordPlan:
    """Next free-decay record length, derived from THIS record's own modes.

    Two filters stand between raw harminv output and the run length, both
    derived from values this module already publishes, neither pinned to a
    geometry:

    * **band** -- the pool is :func:`admit`'s band, i.e. exactly the band the
      judge scores. ``rfx.harminv`` deliberately searches a 10%-widened band
      so the requested band is interior to the search; modes it returns
      outside ``[f_min, f_max]`` are band-edge content no gate ever reads, and
      their Q is the least reproducible thing harminv reports. They are
      returned in ``out_of_band`` (report them, never scale off them). An
      in-band mode that :func:`admit` drops on the ``Q > min_Q`` floor instead
      is NOT out of band and is not labelled as such: it goes to its own
      ``below_min_q`` bucket, so a printed rung never calls an in-band mode
      OUT-OF-BAND. Neither bucket can set a record length.
    * **resolvability** -- a mode enters the tau pool only if the present
      record observed its decay to the published floor, ``tau <=
      resolvable_tau_bound(T)`` (:data:`Q_RECORD_MIN_EFOLDS`). Everything
      above that lands in ``unresolved``.

    The length itself::

        target = target_efolds * max(tau over kept)      # the tau-scaling
        if unresolved:  target = max(target, cap)        # see below
        length  = min(max(target, T), cap)

    The ``unresolved`` clause is what lets the ladder climb: if an in-band mode
    exists whose tau this record could not resolve, the present record does not
    know the slowest tau, so the run is extended as far as the present record
    justifies -- the cap -- and the next rung re-measures. The final ``min``
    is the guarantee that matters: **one rung can never ask for more than
    ``1/min_efolds`` times the record in hand** (4x at the published floor),
    whatever a mode's Q happens to read. Termination is the caller's: it runs
    rungs while ``plan.extend`` and its own step budget both hold.
    """
    cap = resolvable_tau_bound(record_after_source, min_efolds)
    in_band = admit(modes, f_min, f_max, min_Q=min_Q)
    in_band_ids = {id(m) for m in in_band}
    out_of_band = tuple(m for m in modes if id(m) not in in_band_ids
                        and not (f_min <= m.freq <= f_max))
    below_min_q = tuple(m for m in modes if id(m) not in in_band_ids
                        and f_min <= m.freq <= f_max)

    kept, unresolved = [], []
    for mode in in_band:
        tau = amplitude_tau(mode.freq, mode.Q)
        (kept if math.isfinite(tau) and 0 < tau <= cap else unresolved
         ).append(mode)

    slowest = slowest_amplitude_tau(kept)
    if slowest is None:
        target = float(record_after_source)
        reason = ("no in-band mode's decay resolved by this record — "
                  "nothing to scale off")
    else:
        target = float(target_efolds) * slowest
        reason = (f"{target_efolds:g} e-folding(s) of the slowest RESOLVED "
                  f"in-band tau")
    if unresolved:
        target = max(target, cap)
        reason = (f"{len(unresolved)} in-band mode(s) with tau above this "
                  f"record's resolvable bound — extending to the bound "
                  f"(T/{min_efolds:g}) and re-measuring")
    length = min(max(target, float(record_after_source)), cap)
    if length >= cap and target >= cap:
        reason += "; clamped at the resolvable bound"
    return RecordPlan(
        length=length, cap=cap, present=float(record_after_source),
        slowest_tau=slowest, kept=tuple(kept), out_of_band=out_of_band,
        below_min_q=below_min_q, unresolved=tuple(unresolved), reason=reason,
    )


def format_record_plan(plan: RecordPlan, scale: float = 1.0,
                       unit: str = "") -> str:
    """The ladder rung, printed: every mode harminv returned, which filter it
    fell to, and the length that came out.

    Tags: ``pool`` (in band, decay resolved -- the only modes that can set the
    length), ``UNRESOLVED`` (in band, tau above this record's resolvable
    bound), ``LOW-Q`` (in band but under the ``MIN_Q`` floor), ``OUT-OF-BAND``
    (outside ``[f_min, f_max]``). ``LOW-Q`` is printed separately precisely so
    that an in-band mode is never labelled OUT-OF-BAND.
    """
    suffix = f" {unit}" if unit else ""
    # ``scale`` converts the modes' TIME unit (1/freq) into the printed one, so
    # a frequency converts by its reciprocal -- getting this backwards printed
    # 1.5e28 for a 0.166 c/a mode.
    lines = [
        f"  record ladder: present free-decay T = {plan.present * scale:.1f}"
        f"{suffix}; resolvable-tau bound (T / {Q_RECORD_MIN_EFOLDS:g}) = "
        f"{plan.cap * scale:.1f}{suffix}",
    ]
    for tag, group in (("pool", plan.kept), ("UNRESOLVED", plan.unresolved),
                       ("LOW-Q", plan.below_min_q),
                       ("OUT-OF-BAND", plan.out_of_band)):
        for mode in group:
            tau = amplitude_tau(mode.freq, mode.Q)
            t_over_tau = (plan.present / tau
                          if math.isfinite(tau) and tau > 0 else float("inf"))
            lines.append(
                f"    {tag:>11}  f={mode.freq / scale:>12.6g}  "
                f"Q={mode.Q:>10.1f}  tau={tau * scale:>10.4g}{suffix}  "
                f"T/tau={t_over_tau:>7.3f}"
            )
    lines.append(f"    -> next free-decay record {plan.length * scale:.1f}"
                 f"{suffix}: {plan.reason}")
    return "\n".join(lines)


@dataclass(frozen=True)
class ModeSettling:
    """Per-mode ring-down witness on one run's free-decay record."""

    freq: float
    Q: float
    tau: float
    t_over_tau: float     # free-decay amplitude e-foldings the record observed
    energy_db: float      # energy end/peak this mode's own decay implies, dB
    observed: bool        # record spans >= the judge's Q-gating e-folding floor


def mode_settling(freq: float, Q: float, record_after_source: float,
                  observe_efolds: float = Q_RECORD_MIN_EFOLDS) -> ModeSettling:
    """Per-mode settling witness for one extracted mode.

    ``record_after_source`` is the record length AFTER the source is off and
    retained for the pole fit, in the same time units as ``1/freq``. Exclude
    time discarded by preprocessing. The witness is ``T/tau`` amplitude
    e-foldings and the energy end/peak dB they imply
    (``T/tau * ENERGY_DB_PER_EFOLD``). ``observed`` reuses the judge's own
    Q-gating floor (:data:`Q_RECORD_MIN_EFOLDS`) as the line below which the
    record has not seen enough decay to trust the number -- the same
    truncation cut, applied here as a report flag, not a hard gate. Every value
    is computed from the mode's own ``(f, Q)`` and the run's own record length.
    """
    tau = amplitude_tau(freq, Q)
    t_over_tau = (record_after_source / tau
                  if math.isfinite(tau) and tau > 0 else 0.0)
    return ModeSettling(
        freq=freq, Q=Q, tau=tau, t_over_tau=t_over_tau,
        energy_db=t_over_tau * ENERGY_DB_PER_EFOLD,
        observed=t_over_tau >= observe_efolds,
    )


def signal_settling_db(signal, tail_fraction: float = 0.1) -> float:
    """Measured energy ring-down of ONE recorded time series, in dB.

    The same end/peak ARITHMETIC as the S-parameter settling witness
    (:func:`rfx.sources.waveguide_port.settling_db_from_port_records`):
    ``10*log10(mean(P[last tail_fraction]) / max(P))`` with ``P = |signal|**2``.
    The arithmetic is all it shares. It does NOT carry that witness's #869
    underflow floor (``_settling_record_floor_amplitude``: a record whose peak
    amplitude is below ``tiny_normal * 10**(40/20)`` -- 1.1754944e-36 for
    float32 -- is dropped and named instead of scored, because the tail the
    decision reads would be subnormal). This function guards only ``peak > 0``.
    The floor is deliberately not ported: this module declares "no rfx import"
    in its header, which is what lets the judge tests drive it without a
    solver, and hand-copying the threshold would leave a second copy of a
    number derived from ``rfx.api._sparams._SETTLING_WITNESS_DB`` in a file
    that cannot see that constant change. On cv02 the guard is unreachable in
    any case -- the probe sits on the ring and the record peaks at the driven
    field (largest extracted mode amplitude 1.08e-07 on the Meep-absent lane),
    some 29 decades above the float32 floor. A caller that feeds this function
    near-underflow records must apply the floor itself.

    Two things this number is NOT, both of which matter when it is printed
    beside the per-mode witness:

    * **``max(P)`` is the peak of whatever span the caller passes, not
      necessarily the post-source peak.** It is the post-source peak only when
      the caller starts the span AT source-off. cv02 does that on its
      Meep-absent lane (span starts at the waveform's own ``2*t0``), but its
      Meep (verdict) lane passes ``ts[int(0.4*len(ts)):]``, which on the
      committed record begins ~94 Meep units AFTER source-off -- by then
      mode 2 is already ~1.1 dB down, so the ratio reported there is
      optimistic by that much. :func:`format_settling_report` takes the
      offset and prints it; pass it.
    * **It is 3 dB below the per-mode ``energy_db`` by construction, even for
      a signal that never settles.** ``max(P)`` is a single-sample maximum of
      ``A**2 sin**2`` (so ``~A**2``) while the tail is a *mean* of
      ``A**2 sin**2`` (so ``~A**2/2``): an undecayed pure tone reads
      ``10*log10(1/2) = -3.01 dB``, not 0 dB. The per-mode
      :class:`ModeSettling` ``energy_db`` is an envelope quantity and reads
      0 dB for the same tone. Do not compare the two directly without
      subtracting the 3 dB.

    On cv02 this single number is also dominated by the largest-amplitude
    mode's decay rather than by the slowest mode -- on the measured run the
    whole-signal figure (-26.9 dB) sits 18 dB below the slowest in-band mode's
    (-8.9 dB), which is the mode-2/mode-3 amplitude ratio (13 dB) plus the
    3 dB offset plus decay. It is reported next to, never instead of, the
    per-mode witness, which resolves each mode separately.
    """
    p = np.abs(np.asarray(signal, dtype=float)) ** 2
    if p.size == 0:
        return float("nan")
    peak = float(p.max())
    if not (peak > 0.0):
        return float("nan")
    tail = max(1, int(p.size * tail_fraction))
    end = float(p[-tail:].mean())
    tiny = float(np.finfo(float).tiny)
    return float(10.0 * np.log10((end + tiny) / (peak + tiny)))


def format_settling_report(rows, signal_db: float,
                           record_after_source: float,
                           observe_efolds: float = Q_RECORD_MIN_EFOLDS,
                           peak_offset_after_source: float = 0.0, *,
                           analysis_duration: float | None = None) -> str:
    """Human-readable per-mode settling table for the crossval script's stdout.

    ``rows`` is a list of :class:`ModeSettling`. Prints, per mode, ``tau``, the
    e-foldings ``T/tau`` the record observed, and the energy end/peak dB that
    decay implies -- plus the measured whole-signal end/peak dB and an explicit
    statement of the physical limitation: the slowest (radiation-limited) mode
    cannot be run down to the -40 dB rule in feasible time, so its shortfall is
    reported, not gated.

    ``record_after_source`` and ``peak_offset_after_source`` describe the raw
    signal used for the measured whole-signal end/peak. ``analysis_duration``
    is the retained pole-fit span used to construct ``rows``; it defaults to
    the raw duration when preprocessing discards no time.

    ``peak_offset_after_source`` is how long AFTER source-off the raw span
    begins, in the same units as ``record_after_source``. It is 0 when the
    caller starts the span at source-off; when it is not, the whole-signal
    peak is an already-decayed one and the caption says so (see
    :func:`signal_settling_db`).
    """
    if analysis_duration is None:
        analysis_duration = record_after_source
    lines: list[str] = []
    lines.append(
        f"  pole-fit span T = {analysis_duration:.3e} (1/freq units); "
        f"per-mode T/tau, energy and Q observability use this retained span"
    )
    lines.append(
        f"  raw whole-signal span = {record_after_source:.3e} (1/freq units), "
        f"starting {peak_offset_after_source:.3e} after source-off"
    )
    lines.append("  end/peak below are computed from each mode's own extracted "
                 "(f, Q) -- no pinned value")
    lines.append("")
    lines.append(
        f"  {'freq':>16} {'Q':>10} {'tau':>12} {'T/tau':>8} "
        f"{'E end/peak':>12} {'decay':>14}"
    )
    for row in rows:
        note = "observed" if row.observed else "truncation-susp"
        lines.append(
            f"  {row.freq:>16.6e} {row.Q:>10.1f} {row.tau:>12.4e} "
            f"{row.t_over_tau:>8.3f} {row.energy_db:>10.1f} dB {note:>14}"
        )
    lines.append("")
    peak_frame = (
        "peak = post-source peak (span starts at source-off)"
        if peak_offset_after_source <= 0.0 else
        f"peak = the ALREADY-DECAYED peak {peak_offset_after_source:.3e} "
        f"after source-off, so this figure is optimistic by that decay"
    )
    lines.append(
        f"  measured whole-signal end/peak energy = {signal_db:.1f} dB "
        f"({peak_frame})"
    )
    lines.append(
        "  NOTE the two columns are not the same quantity: the whole-signal "
        "figure is a single-sample"
    )
    lines.append(
        "  max over a mean, so an UNDECAYED pure tone reads -3.01 dB there and "
        "0 dB in the per-mode"
    )
    lines.append(
        "  envelope column; and it tracks the largest-amplitude mode, not the "
        "slowest one."
    )
    lines.append(
        f"  PHYSICAL LIMITATION (not a gate): the -40 dB settling rule needs "
        f"{-40.0 / ENERGY_DB_PER_EFOLD:.2f} e-foldings; a radiation-limited"
    )
    lines.append(
        "  high-Q ring mode's tau can be arbitrarily large, so driving the "
        "slowest mode that deep is not"
    )
    lines.append(
        "  generally feasible. On the no-verdict lane the record is scaled at "
        "runtime to the slowest"
    )
    lines.append(
        "  in-band mode whose decay the previous record RESOLVED (see "
        "plan_record); 'truncation-susp' marks"
    )
    lines.append(
        f"  a mode below the judge's {observe_efolds:g}-e-folding Q-gating "
        f"floor. Faster modes settle deeper. Such modes are"
    )
    lines.append(
        "  reported, and the judge Q-gates only modes whose decay the record "
        "observed (same floor)."
    )
    return "\n".join(lines)


# --- the judge that shipped, kept executable --------------------------------


def legacy_shipped_judge(meep_freqs, meep_Qs, rfx_freqs, rfx_Qs=None):
    """The pre-#812 judge, transcribed verbatim from ``02_ring_resonator.py``.

    Retained so the tautology it embodies stays measurable: the matcher window
    ``best_diff < 0.05`` and the verdict ``mean_err < 5.0`` are the same
    number, so ``mean_err`` is bounded below 5% for every possible input and
    the headline gate can only fail through ``len(matched) >= 2``.

    Returns ``(passed, mean_err_pct_or_None, n_matched)``. ``rfx_Qs`` is
    accepted and ignored — the shipped judge gated no Q at all.
    """
    matched = []
    for mf, mQ in zip(meep_freqs, meep_Qs):
        best_idx = None
        best_diff = 1.0
        for i, rf in enumerate(rfx_freqs):
            diff = abs(rf - mf) / mf
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        if best_idx is not None and best_diff < 0.05:
            matched.append((mf, mQ, rfx_freqs[best_idx], None))

    passed = True
    mean_err = None
    if matched:
        errs = [abs(rf - mf) / mf * 100 for mf, _, rf, _ in matched]
        mean_err = float(np.mean(errs))
        if mean_err >= 5.0:
            passed = False
        if len(matched) < 2:
            passed = False
    else:
        passed = False
    return passed, mean_err, len(matched)
