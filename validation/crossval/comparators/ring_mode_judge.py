"""Ring-resonator mode judge for cv02 — assignment decoupled from tolerance.

Plain numpy/scipy. No rfx import, no Simulation, no solve: this module only
compares two lists of extracted modes, so both the crossval script and
``tests/test_cv02_ring_mode_judge.py`` can drive the same code.

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
