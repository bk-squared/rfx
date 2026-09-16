"""Per-arm, per-bin continuum windows for the slab family's E2 gates.

Pre-declared in
``docs/design_notes/slab_family_per_arm_lattice_window_predeclaration.md``;
change the note (append-only) before changing anything numeric here.

WHAT THIS REPLACES. cv22 and cv23 sized the per-bin window their ``G1`` gates
enforce -- ``|X_rfx(f) - X_TMM(f)| <= W_BIN + w_ade,X(f)`` -- from cv04's
per-bin maximum ``|R+T-1|``, scaled by the repo's gate multiplier. ``|R+T-1|``
is an ENERGY-LEAK quantity: it grows when a record is cut short or an absorber
reflects. The residual the window has to cover is LATTICE DISPERSION -- the Yee
grid makes the slab look slightly electrically thick, so the Fabry-Perot
fringes move -- and that conserves energy, so it never appears in ``|R+T-1|``
at all. The window was sized by one quantity and spent on another, on a
different board besides (cv04's lossless eps' = 4 slab, judging three
dispersive and three lossy ones). It passed because cv04's pre-settling
record leaked enough to make the window wide by accident; cv04's settled
record on the shipped absorber leaks two orders of magnitude less, and the
same recipe then fails every declared arm (issue #928; the two readings and
their ratio are in ``_04_fresnel_results/RECOMPUTE.md``, which is where cv04's
own measurements belong).

THE DERIVATION. For an arm with DECLARED model ``m`` and DECLARED parameters
``p``, run at ``(dx, dt)``, per bin and per observable X in {R, T, A}:

    W_X(f) = MULTIPLIER * [ W_lat,X(f) + W_wit,X(f) ]

    W_lat,X(f) = | X_lattice(f; m, p, d, dx, dt) - X_TMM(f; m, p, d) |
    W_wit,X(f) = the lattice-witness budget window of this record

which is an exact decomposition, not a pair of allowances::

    | X_rfx - X_TMM |  <=  | X_rfx - X_lattice |  +  | X_lattice - X_TMM |
                       <=         W_wit          +         W_lat

The first inequality is the triangle inequality; the second is ``GL1``, the
gate ``comparators/lattice_witness.py`` already enforces at every rung
(``docs/design_notes/20260903_lattice_witness_standard.md``). So the window is
the SUM of a model term and a record term, and ``W_wit`` is the FLOOR the task
asks for -- derived, not chosen: a per-bin window cannot sit below the distance
the record's own finiteness puts between the measurement and the lattice it
stepped.

Every input is the arm's own: its ``eps(f)``, the family thickness, its own
``dx`` rung, its own ``dt``, its own settling tails and step count. Nothing is
read from cv04 or from a sibling arm.

``w_ade`` IS NOT ADDED. ``X_lattice`` is built on the DISCRETE-TIME
permittivity ``eps_numerical_ade(f, m, p, dt)`` -- it is the exact steady state
of the update that ran, ADE included -- so the material model's time-
discretization error is already inside ``W_lat`` and adding ``w_ade`` on top
double-counts it. The two are not nested (in 14 of the 29 committed records
there are gated bins where ``W_lat < w_ade``, worst ratio 7.9 on cv22's Drude
R): those are bins where the spatial and temporal lattice terms partially
CANCEL, which the lattice model represents and a sum of magnitudes cannot.

NOT QUANTIZED. ``tests/_gate_policy.gate_from_envelope`` rounds up to
``1/quantum``, and at the cases' declared ``quantum = 1000`` that would raise
every bin whose derived window is under 1e-3 to 1e-3 -- most bins on the fine
rungs -- widening the gate by up to two orders of magnitude exactly where the
derivation is tightest. Quantization makes a HAND-PINNED SCALAR readable; these
windows are recomputed from the record at evaluation time and written down as a
literal nowhere. The MULTIPLIER is still the shared one, imported rather than
restated, so widening anything here is still one reviewer-visible edit.

WHAT THIS MODULE DOES NOT TOUCH. The Meep legs (``G4``, ``G5``) keep their
cv04-adopted scalar windows: the term a ``G4`` window must cover is MEEP's
discretization, which is neither cv04's nor the rfx arm's -- section 7 of the
pre-declaration measures a per-arm Meep derivation and says why adopting it
here would be the self-certification #928 was filed about. ``GL1`` / ``GL2``,
the settling witness, the passivity gate and the record-length recipe are
unchanged.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from typing import NamedTuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dispersive_eps as de  # noqa: E402
import lattice_witness as LW  # noqa: E402
import slab_family as SF  # noqa: E402
from tests import _gate_policy  # noqa: E402

SCHEMA = "slab-arm-window/v1"


class ArmWindows(NamedTuple):
    """The windows ONE arm's E2 gates enforce, per bin and in the band mean.

    ``R``/``T``/``A`` are per-bin arrays on the arm's own bin grid; the three
    ``mean_*`` are the gated-band means of those same arrays. ``parts`` carries
    both components and the multiplier, so the artifact can state what the
    window was made of and a contract test can re-derive it from outside
    without importing either consumer.

    Handed to ``cv22_dispersive_gates.evaluate_e2`` in place of the scalar
    ``slab_family.Windows``; the evaluator adds no ``w_ade`` on top of these
    (see the module docstring).
    """

    R: np.ndarray
    T: np.ndarray
    A: np.ndarray
    mean_R: float
    mean_T: float
    mean_A: float
    parts: dict


def lattice_term(freqs_hz, model: str, params: dict, dx: float, dt: float,
                 *, d_slab_m: float = SF.D_SLAB_M):
    """``W_lat,{R,T,A}(f) = |lattice(dx, dt) - continuum|`` for ONE arm.

    ``params`` must be the DECLARED parameters, never a falsifier's run
    parameters: a wrong model that sized its own window would pass the gate
    meant to catch it. The callers take them from the arm record's ``params``
    key, which is the declared set (``params_run`` holds the defect).
    """
    f = np.asarray(freqs_hz, dtype=float)
    R_an, T_an = de.tmm_slab_rt(f, de.eps_analytic(f, model, params), d_slab_m)
    R_lat, T_lat = de.yee_lattice_slab_rt_model(f, model, params, d_slab_m,
                                                float(dx), float(dt))
    A_an = 1.0 - R_an - T_an
    A_lat = 1.0 - R_lat - T_lat
    return np.abs(R_lat - R_an), np.abs(T_lat - T_an), np.abs(A_lat - A_an)


def ringdown_rate(record, tail: dict):
    """The truncation bound's decay rate, and where it came from.

    Normally the standard's own ``lattice_witness.ringdown_rate``: the slower
    of the case's DERIVED ring-down rate and any reliable fitted tail rate.

    A run made under ``--recipe cv04`` derives no record at all, and the only
    ring-down estimate it carries is the fit to its own stored tail envelope.
    Falling back to the slowest of those two fits keeps the window's floor
    defined -- with no floor there is no gate, and a crash where a red gate
    belongs hides the same thing a silent pass would. The fallback is named in
    the artifact (``rate_ringdown_source``), and ``--recipe cv04`` cannot write
    a committed baseline in either case
    (``test_recipe_cv04_without_tag_is_refused``).
    """
    rec = record or {}
    if rec.get("rate_ring_1_s") is not None:
        return LW.ringdown_rate(rec, tail)
    fits = {}
    for key in ("fitted_rate_scat_refl_1_s", "fitted_rate_total_trans_1_s"):
        value = (tail or {}).get(key)
        if value is not None and np.isfinite(value) and float(value) > 0.0:
            fits[key] = float(value)
    if not fits:
        raise ValueError(
            "this run derives no record length AND carries no usable fitted tail "
            "rate, so the window's truncation term has no decay rate and the "
            "per-bin gate would be undefined")
    src = min(fits, key=fits.get)
    return fits[src], src + " (no derived record)"


def witness_term(freqs_hz, model: str, params: dict, *, dx: float, dt: float,
                 n_steps: int, inc_amp_rel, tail: dict, record: dict,
                 d_slab_m: float = SF.D_SLAB_M):
    """``W_wit,{R,T,A}(f)``: the lattice-witness budget window of this record.

    Computed by the standard's own functions
    (``lattice_witness.budget_terms`` + ``windows_from_terms``), on this
    record's settling tails, incident purity, ring-down rate and step count --
    the same numbers ``GL1`` is judged against, so the two gates cannot drift
    apart. No second copy of the budget lives here.
    """
    R_lat, T_lat = de.yee_lattice_slab_rt_model(freqs_hz, model, params, d_slab_m,
                                                float(dx), float(dt))
    rate, rate_src = ringdown_rate(record, tail)
    terms = LW.budget_terms(freqs_hz, inc_amp_rel, dt=float(dt), n_steps=int(n_steps),
                            scat_tail_rel=tail["scat_refl_rel"],
                            trans_tail_rel=tail["total_trans_rel"],
                            purity_rel=tail["purity_inc_rel"], rate_1_s=rate)
    wR, wT, wA = LW.windows_from_terms(R_lat, T_lat, terms)
    return wR, wT, wA, rate, rate_src


def derive(freqs_hz, model: str, params: dict, *, dx: float, dt: float, n_steps: int,
           inc_amp_rel, tail: dict, record: dict,
           d_slab_m: float = SF.D_SLAB_M) -> ArmWindows:
    """``MULTIPLIER * (W_lat + W_wit)`` per bin, and its gated-band mean.

    Raises rather than returning a window that cannot be enforced: a
    non-finite or non-positive bin means the decomposition has no floor there
    and the gate would be undefined, which must be a red gate and not a
    silently permissive one.
    """
    f = np.asarray(freqs_hz, dtype=float)
    g = SF.gated_mask(f)
    if not g.any():
        raise ValueError("no gated bin on this frequency grid; the band-mean window "
                         "would be undefined")
    lat_R, lat_T, lat_A = lattice_term(f, model, params, dx, dt, d_slab_m=d_slab_m)
    wit_R, wit_T, wit_A, rate, rate_src = witness_term(
        f, model, params, dx=dx, dt=dt, n_steps=n_steps, inc_amp_rel=inc_amp_rel,
        tail=tail, record=record, d_slab_m=d_slab_m)
    # Read at CALL time, not captured at import: `from ... import
    # ENVELOPE_GATE_MULTIPLIER` binds a value, and a change to the shared
    # constant would then move every other gated case and silently miss this
    # one. `gate_from_envelope` is built the same way and for the same reason.
    m = float(_gate_policy.ENVELOPE_GATE_MULTIPLIER)
    win = {}
    for key, lat, wit in (("R", lat_R, wit_R), ("T", lat_T, wit_T), ("A", lat_A, wit_A)):
        w = m * (np.asarray(lat, dtype=float) + np.asarray(wit, dtype=float))
        if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
            bad = int(np.argmin(np.where(np.isfinite(w), w, -np.inf)))
            raise ValueError(
                f"the derived {key} window is not finite and positive at bin {bad} "
                f"({f[bad]:.4e} Hz, value {w[bad]!r}): the lattice-continuum term and "
                f"the record budget must both be defined at every bin, or the gate is "
                f"not a gate there")
        win[key] = w
    parts = {
        "schema": SCHEMA,
        "multiplier": m,
        "multiplier_source": "tests/_gate_policy.py ENVELOPE_GATE_MULTIPLIER",
        "quantized": False,
        "ade_additive": False,
        "predeclaration": ("docs/design_notes/"
                           "slab_family_per_arm_lattice_window_predeclaration.md"),
        "dx_m": float(dx), "dt_s": float(dt), "n_steps": int(n_steps),
        "d_slab_m": float(d_slab_m),
        "model": model, "params": {k: float(v) for k, v in params.items()},
        "rate_ringdown_1_s": float(rate), "rate_ringdown_source": rate_src,
        "W_lat_R": lat_R.tolist(), "W_lat_T": lat_T.tolist(), "W_lat_A": lat_A.tolist(),
        "W_wit_R": wit_R.tolist(), "W_wit_T": wit_T.tolist(), "W_wit_A": wit_A.tolist(),
        "mean_W_lat_R_gated": float(lat_R[g].mean()),
        "mean_W_lat_T_gated": float(lat_T[g].mean()),
        "mean_W_lat_A_gated": float(lat_A[g].mean()),
        "mean_W_wit_R_gated": float(wit_R[g].mean()),
        "mean_W_wit_T_gated": float(wit_T[g].mean()),
        "mean_W_wit_A_gated": float(wit_A[g].mean()),
        "max_W_lat_R_gated": float(lat_R[g].max()),
        "max_W_lat_T_gated": float(lat_T[g].max()),
        "max_W_lat_A_gated": float(lat_A[g].max()),
    }
    return ArmWindows(R=win["R"], T=win["T"], A=win["A"],
                      mean_R=float(win["R"][g].mean()),
                      mean_T=float(win["T"][g].mean()),
                      mean_A=float(win["A"][g].mean()),
                      parts=parts)


def from_arm_doc(arm_doc: dict, *, model: str | None = None, params: dict | None = None,
                 d_slab_m: float = SF.D_SLAB_M) -> ArmWindows:
    """``derive`` on the per-arm block of a committed cv22 / cv23 ``rfx*.json``.

    The entry point a re-judgement or a from-outside contract test uses: it
    reads the DECLARED ``params`` (not ``params_run``) and the record's own
    ``run.dx_m`` / ``run.n_steps`` / ``run.record`` / ``tail`` /
    ``inc_amp_rel``, so nothing about the window depends on which module is
    asking.
    """
    run = arm_doc["run"]
    return derive(arm_doc["freqs_hz"],
                  model or arm_doc["model"],
                  params if params is not None else arm_doc["params"],
                  dx=float(run["dx_m"]), dt=float(arm_doc["dt_s"]),
                  n_steps=int(run["n_steps"]), inc_amp_rel=arm_doc["inc_amp_rel"],
                  tail=arm_doc["tail"], record=run["record"], d_slab_m=d_slab_m)


def from_run(run: dict, model: str, params: dict, *,
             d_slab_m: float = SF.D_SLAB_M) -> ArmWindows:
    """``derive`` on the FLAT dict a case script has in hand after a run.

    ``params`` is the DECLARED set the script gates against, not the
    ``params_run`` it built the FDTD with. A run with no derived record
    (``--recipe cv04``) still gets a window -- see ``ringdown_rate`` -- because
    that path has its own red gate (the missing ``GL_witness`` makes the
    verdict incomplete) and an exception there would replace one red signal
    with a worse one.
    """
    return derive(run["freqs_hz"], model, params, dx=float(run["dx_m"]),
                  dt=float(run["dt_s"]), n_steps=int(run["n_steps"]),
                  inc_amp_rel=run["inc_amp_rel"], tail=run["tail"],
                  record=run["record"], d_slab_m=d_slab_m)


def resolve(windows, w_ade_R, w_ade_T, gated):
    """(win_R, win_T, mean_win_R, mean_win_T, provenance) for either window shape.

    ``ArmWindows`` -- the derivation above -- supplies per-bin R and T windows
    that already carry the ADE term, so nothing is added. A legacy scalar
    ``slab_family.Windows`` keeps the arithmetic it always had
    (``w_bin + w_ade``), bit for bit: cv26 builds one, and so does the Meep
    leg. Keeping both shapes in ONE resolver is what stops a third shape from
    being introduced quietly in a consumer.
    """
    g = np.asarray(gated, dtype=bool)
    if isinstance(windows, ArmWindows):
        return (windows.R, windows.T, windows.mean_R, windows.mean_T,
                dict(windows.parts))
    w_ade_R = np.asarray(w_ade_R, dtype=float)
    w_ade_T = np.asarray(w_ade_T, dtype=float)
    return (windows.w_bin + w_ade_R, windows.w_bin + w_ade_T,
            windows.w_mean_R + float(np.mean(w_ade_R[g])),
            windows.w_mean_T + float(np.mean(w_ade_T[g])),
            {"schema": "cv04-adopted-scalar/v1", "w_bin": float(windows.w_bin),
             "w_mean_R": float(windows.w_mean_R), "w_mean_T": float(windows.w_mean_T),
             "ade_additive": True, "quantized": True})
