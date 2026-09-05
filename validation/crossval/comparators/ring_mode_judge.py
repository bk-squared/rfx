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
              ``|ln(Q_rfx / Q_ref)| <= ln(1 + tau_ref / T)``
============  ==========================================================

The Q window is derived, not chosen — see :func:`q_window`.

Frequencies and the record length must be in reciprocal units (the script
passes both in Meep normalised units: ``f`` in ``c/a``, ``T`` in ``a/c``).
Pre-declaration: ``docs/design_notes/20260831_cv02_ring_judge_predeclaration.md``.
"""

from __future__ import annotations

import math
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
#: of observed amplitude decay). Consequence: the loosest admissible Q window
#: is tau/T <= 4, so every gated mode still rejects a factor-3.35 Q error.
Q_RECORD_MIN_EFOLDS = 0.25

#: Mode-admission floor, applied symmetrically to both solvers' harminv output.
MIN_Q = 1.0


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
    """Record-length-derived Q tolerance for one REFERENCE mode.

    A record of length ``T`` cannot resolve exponential decay rates finer than
    ``1/T`` — the same record-length limit that sets ``1/T`` Fourier frequency
    resolution; two envelopes whose rates differ by less than ``1/T`` differ by
    less than one factor of ``e`` over the whole record and are not separable.
    With amplitude decay rate ``alpha = pi f / Q`` (e-folding time
    ``tau = Q / (pi f)``)::

        delta_Q / Q = delta_alpha / alpha = (1/T) / (pi f / Q) = tau / T

    Both inputs are the reference's; no measured rfx quantity appears, so this
    window is not fitted to the agreement it judges.

    **Known limitation -- this window is a RESOLUTION bound, not an accuracy
    bound, and it therefore shrinks with run length while the physics does
    not.** ``tau/T`` says how finely a record of length ``T`` can separate two
    decay rates; it says nothing about how far apart two *solvers* should be.
    The rfx-vs-Meep Q gap on cv02 is a discretization offset (staircased ring
    boundary, subpixel treatment, hence a slightly different radiation Q), so
    it is roughly constant in ``T``, while rfx's own Q for the same modes is
    stable over every span that was actually measured. Measured ``|ln(Q_rfx/Q_ref)| = 0.070`` (mode 1)
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
    Fixing it needs a floor on the window encoding the expected
    discretization Q gap (or a pre-declared |ln Q| envelope); that is a change
    to a claims-bearing gate and is NOT done here -- it is tracked as issue
    #907 (the ``tau_ref/T`` window shrinks with ``T`` faster than the physics
    does, so a longer record fails a stable Q), and it is the reason cv02's
    Meep (verdict) lane keeps its calibrated record length instead of the
    tau-scaled one.

    Returns ``(T/tau, window)``. ``T/tau`` is the number of amplitude
    e-foldings the record observed; a mode is Q-gated only when it reaches
    :data:`Q_RECORD_MIN_EFOLDS`.
    """
    tau = ref_Q / (math.pi * ref_freq)
    if tau <= 0 or record_length <= 0:
        return 0.0, float("inf")
    t_over_tau = record_length / tau
    return t_over_tau, tau / record_length


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
            if row.q_gated and partner.Q > 0 and ref_mode.Q > 0:
                row.q_log_ratio = abs(math.log(partner.Q / ref_mode.Q))
                row.q_pass = row.q_log_ratio <= math.log(1.0 + window)
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
        f"(Meep units); Q windows below are tau_ref/T, not chosen values"
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

    ``record_after_source`` is the record length AFTER the source is off, in
    the same time units as ``1/freq``. The witness is ``T/tau`` amplitude
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

    Same end/peak arithmetic as the S-parameter settling witness
    (:func:`rfx.sources.waveguide_port.settling_db_from_port_records`):
    ``10*log10(mean(P[last tail_fraction]) / max(P))`` with ``P = |signal|**2``.

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
    whole-signal figure (-26.3 dB) sits 18 dB below the slowest mode's
    (-8.4 dB), which is the mode-2/mode-3 amplitude ratio (13 dB) plus the
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
                           peak_offset_after_source: float = 0.0) -> str:
    """Human-readable per-mode settling table for the crossval script's stdout.

    ``rows`` is a list of :class:`ModeSettling`. Prints, per mode, ``tau``, the
    e-foldings ``T/tau`` the record observed, and the energy end/peak dB that
    decay implies -- plus the measured whole-signal end/peak dB and an explicit
    statement of the physical limitation: the slowest (radiation-limited) mode
    cannot be run down to the -40 dB rule in feasible time, so its shortfall is
    reported, not gated.

    ``peak_offset_after_source`` is how long AFTER source-off the analysed span
    begins, in the same units as ``record_after_source``. It is 0 when the
    caller starts the span at source-off; when it is not, the whole-signal
    peak is an already-decayed one and the caption says so (see
    :func:`signal_settling_db`).
    """
    lines: list[str] = []
    lines.append(
        f"  analysed span T = {record_after_source:.3e} (1/freq units), "
        f"starting {peak_offset_after_source:.3e} after source-off; per-mode "
        f"T/tau and energy"
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
