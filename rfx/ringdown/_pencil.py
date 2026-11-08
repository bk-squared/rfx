"""Identification numerics used by :mod:`rfx.ringdown`."""

from __future__ import annotations

from rfx.errors import RfxRefusal

from importlib import import_module
import math
from numbers import Number

import numpy as np

from rfx.measurement.matrix_pencil import (
    _decimate as _core_decimate,
    _pencil as _core_pencil,
    _to_original_rate as _core_to_original_rate,
    _fit_residues,
)

# Import the module rather than rfx's re-exported harminv function.
_harminv = import_module("rfx.harminv")

# Mac per-stage peaks / formula, maxima 1.780 and 1.626, plus 5 %:
# records 20261012-pencil-stage-peaks, decision g.
_STAGE1_FACTOR = 1.87
_STAGE2_FACTOR = 1.71


class IdentificationMemoryRefused(RfxRefusal):
    """An identification stage would exceed the configured memory bound."""

    __module__ = "rfx.ringdown"

    def __init__(self, stage, needed_bytes, bound_bytes, *, n_win, Nd, C, L, M, k, r):
        self.stage = stage
        self.needed_bytes = needed_bytes
        self.bound_bytes = bound_bytes
        self.n_win, self.Nd, self.C = n_win, Nd, C
        self.L, self.M, self.k, self.r = L, M, k, r
        name = "pencil" if stage == 1 else "residue fit"
        order = f"k={k}" if stage == 1 else f"rank={r}"
        super().__init__(
            f"identification would need {needed_bytes / 2**30:.2f} GiB "
            f"at stage {stage} ({name}; n_win={n_win}, Nd={Nd}, C={C}, L={L}, "
            f"{order}), above the {bound_bytes / 2**30:.2f} GiB bound; "
            "shorten the identification window, identify on fewer channels, "
            "or raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES")

    def __reduce__(self):
        dimensions = {name: getattr(self, name) for name in ("n_win", "Nd", "C", "L", "M", "k", "r")}
        return (_restore_memory_refusal, (self.stage, self.needed_bytes, self.bound_bytes, dimensions))


def _restore_memory_refusal(stage, needed_bytes, bound_bytes, dimensions):
    return IdentificationMemoryRefused(stage, needed_bytes, bound_bytes, **dimensions)


def _validate_memory_bound(bound):
    """Keep invalid public settings on the ValueError verdict path."""
    try:
        valid = (isinstance(bound, Number)
                 and not isinstance(bound, (bool, np.bool_, complex, np.complexfloating))
                 and 0 < bound <= math.inf)
    except (TypeError, ArithmeticError):
        valid = False
    if not valid:
        raise ValueError("rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES must be a positive finite number or math.inf")


def _check_identification_memory(n_win, shape, *, r=None, bound, pencil_parameter):
    """Refuse before the pencil or original-rate residue fits allocate."""
    _validate_memory_bound(bound)
    Nd, C = shape
    L = int(_harminv._pencil_columns(Nd, pencil_parameter))
    M = Nd - L
    k = min(M, L * C)
    stage = 1 if r is None else 2
    needed = (_STAGE1_FACTOR * 16 * (3*M*L*C + M*k + 2*k*L*C + 4*k*k)
              if r is None else _STAGE2_FACTOR * 16 * 5 * n_win * r)
    if needed > bound:
        raise IdentificationMemoryRefused(
            stage, needed, bound,
            n_win=n_win, Nd=Nd, C=C, L=L, M=M, k=k, r=r)


def _check_run_identification_memory(plan):
    """The fixed run's main pencil, before the lane takes any step."""
    from rfx import ringdown as rd

    bound = rd.IDENTIFY_MEMORY_BOUND_BYTES
    _validate_memory_bound(bound)
    bins = plan.bins
    # Default and graded-lane result bins are the accumulator's float32 bins.
    peak = (float(np.float32(plan.sim._freq_max)) if bins is None else
            float(np.max(np.asarray(bins, dtype=np.float32 if plan.lane == "graded" else np.float64))))
    n_win = plan.n_steps - plan.n_start
    _, Nd = rd.decimation_plan(n_win, float(plan.dt), max(plan.freq_max, peak))
    C = 2 * len(plan.wire_entries) + len(plan.identification_keys)
    rd._check_identification_memory(n_win, (Nd, C), bound=bound, pencil_parameter=rd.PENCIL_PARAMETER)


def _witness_identification(identify_window):
    """A long-window refusal is a failed witness, preserving the main result."""
    def identify_long_window(window):
        try:
            return identify_window(window)
        except IdentificationMemoryRefused as exc:
            raise ValueError(str(exc)) from exc
    return identify_long_window


def decimation_plan(n_samples: int, dt: float, freq_max: float) -> tuple:
    """harminv's plan for a window of ``n_samples``: ``(factors, kept samples)``."""
    from rfx.ringdown import HARMINV_MAX_MODES, PENCIL_PARAMETER

    factors, kept = _harminv._decimation_plan(
        int(n_samples), float(dt), float(freq_max), "auto",
        pencil_parameter=PENCIL_PARAMETER, max_modes=HARMINV_MAX_MODES)
    return tuple(int(q) for q in factors), int(kept)


def _decimate(Y: np.ndarray, factors) -> np.ndarray:
    """Keep the original patchable FIR entry point."""
    return _core_decimate(Y, factors, decimate_fn=_harminv.scipy_decimate)


def _pencil(Yd: np.ndarray, sv_rel: float, unit_tol: float):
    """Keep ringdown's configured pencil fraction and uncapped rank."""
    from rfx.ringdown import PENCIL_PARAMETER

    return _core_pencil(Yd, sv_rel, unit_tol, pencil_parameter=PENCIL_PARAMETER)


def _to_original_rate(lam_dec: np.ndarray, D: int, dt: float, guard):
    """Preserve ringdown's two-value private interface."""
    s, n_guard, _ = _core_to_original_rate(lam_dec, D, dt, guard)
    return s, n_guard


def _on_nearest_bin(s, f_bins, record_s: float) -> np.ndarray:
    """Each pole moved onto the nearest requested bin within ``1 / record_s``.

    A record of length ``T`` places a pole's frequency only to about ``1/T``:
    an unresolved pair is identified as one pole between its members, which
    can sit far from every bin while the members sit on bins. For the floor a
    pole is weighed as if it sat on the nearest bin it could be on; its decay
    and residue are kept. A pole with no bin within ``1/T`` is left in place.
    """
    s = np.asarray(s, dtype=np.complex128)
    fb = np.asarray(f_bins, dtype=np.float64)
    if s.size == 0 or fb.size == 0 or not record_s > 0.0:
        return s
    f = s.imag / (2.0 * np.pi)
    j = np.argmin(np.abs(fb[None, :] - f[:, None]), axis=1)
    # a pole the record cannot tell from DC stays where it is: moved onto a
    # low bin, a slowly relaxing static field would weigh like a mode there
    near = ((np.abs(fb[j] - f) <= 1.0 / float(record_s))
            & (np.abs(f) >= 1.0 / float(record_s)))
    return np.where(near, s.real + 1j * 2.0 * np.pi * fb[j], s)
