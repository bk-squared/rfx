"""Bookkeeping shared by the two design-loop drivers (issue 1359).

``design_taper.py`` and ``design_beam.py`` record a gradient-descent design
loop iterate by iterate and check its gradient before the loop starts.  The
pieces they share live here: the central-difference ladder and its verdict,
the per-iterate store that is rewritten after every iterate (a job killed at
iterate 100 leaves 0 ... 99 on disk), the Adam state beside it, and device
memory.  Only numpy and the standard library are imported at module level, so
the unit tests load it without jax.

The judging rules are the pre-declaration's
(``docs/design_notes/20260928_design_films_predeclaration.md``); this module
applies them and defines none of its own.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _record  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_record._jsonable) + "\n")
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def save_npz(path: Path, **arrays) -> None:
    """``np.savez`` through a temporary name, so a kill mid-write leaves the
    previous complete file rather than a truncated one."""
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def device_memory() -> dict:
    import jax
    ms = jax.devices()[0].memory_stats() or {}
    return {"peak_bytes_in_use": int(ms.get("peak_bytes_in_use", -1)),
            "bytes_in_use": int(ms.get("bytes_in_use", -1))}


# --------------------------------------------------------------- FD ladder
def fd_ladder(objective, x0: np.ndarray, variables, steps) -> dict:
    """Central differences of ``objective`` about ``x0`` for each variable.

    ``variables`` maps a name to an index into ``x0``; ``steps`` are the
    perturbations in the units of ``x0``.  Returns
    ``{name: {str(h): {"f_plus", "f_minus", "fd"}}}``.
    """
    out = {}
    for name, idx in variables.items():
        out[name] = {}
        for h in steps:
            e = np.zeros_like(x0)
            e[idx] = h
            fp, fm = float(objective(x0 + e)), float(objective(x0 - e))
            out[name][str(h)] = {"f_plus": fp, "f_minus": fm, "fd": (fp - fm) / (2.0 * h)}
    return out


def judge_fd(ad: dict, ladder: dict, steps, judged_step: float, rel_bar: float,
             judge_frac: float) -> dict:
    """The pre-declared verdict on AD against central differences.

    A variable is judged when ``|FD(judged_step)|`` is at least ``judge_frac``
    of the largest ``|FD(judged_step)|`` among the variables; the others are
    reported.  A judged variable passes when ``|AD - FD| / |FD| <= rel_bar``
    at ``judged_step``.  Its ladder is round-off-dominated when FD moves more
    between the judged step and the finest step than between the coarsest
    step and the judged step (truncation error falls as the step shrinks,
    round-off grows); the pre-declaration then calls for the float64 repeat.
    """
    coarse, mid, fine = (str(h) for h in steps)
    key = str(judged_step)
    fmax = max(abs(ladder[n][key]["fd"]) for n in ladder)
    rows, judged, roundoff = {}, [], []
    for n in ladder:
        f = ladder[n][key]["fd"]
        rel = abs(ad[n] - f) / abs(f) if f else float("inf")
        is_judged = fmax > 0 and abs(f) >= judge_frac * fmax
        ro = abs(ladder[n][fine]["fd"] - ladder[n][mid]["fd"]) > \
            abs(ladder[n][mid]["fd"] - ladder[n][coarse]["fd"])
        rows[n] = {"ad": ad[n], "fd": f, "rel": rel, "judged": is_judged,
                   "passed": (rel <= rel_bar) if is_judged else None,
                   "fd_by_step": {s: ladder[n][s]["fd"] for s in ladder[n]},
                   "rel_by_step": {s: (abs(ad[n] - ladder[n][s]["fd"]) / abs(ladder[n][s]["fd"])
                                       if ladder[n][s]["fd"] else None) for s in ladder[n]},
                   "ladder_roundoff": ro}
        if is_judged:
            judged.append(n)
            if ro:
                roundoff.append(n)
    return {"judged_step": judged_step, "rel_bar": rel_bar, "judge_frac": judge_frac,
            "fd_max_abs": fmax, "judged": judged, "roundoff": roundoff, "rows": rows,
            "all_judged_passed": all(rows[n]["passed"] for n in judged)}


# ------------------------------------------------------------ iterate store
class IterateStore:
    """Every iterate of a design loop, rewritten to ``iterations.npz`` after
    each one.  Keys are fixed at the first ``append``; an iterate that lacks
    a key (the final design has no gradient) is filled with NaN."""

    def __init__(self, path: Path, static: dict | None = None):
        self.path = Path(path)
        self.static = dict(static or {})
        self.rows: dict[str, list] = {}

    def append(self, **values) -> None:
        if not self.rows:
            self.rows = {k: [] for k in values}
        n = len(next(iter(self.rows.values())))
        for k in values:
            if k not in self.rows:
                raise KeyError(f"iterate key {k!r} was not present at iterate 0")
        for k, col in self.rows.items():
            if k in values:
                col.append(np.asarray(values[k]))
            else:
                ref = np.asarray(col[0])
                col.append(np.full(ref.shape, np.nan, dtype=float))
        assert all(len(c) == n + 1 for c in self.rows.values())

    def __len__(self) -> int:
        return len(next(iter(self.rows.values()))) if self.rows else 0

    def persist(self) -> None:
        save_npz(self.path, **self.static, **{k: np.stack(v) for k, v in self.rows.items()})


def adam_state_arrays(state) -> dict:
    """optax.adam's state as plain arrays (count, mu, nu), for the record."""
    s = state[0]
    return {"adam_count": np.asarray(s.count), "adam_mu": np.asarray(s.mu),
            "adam_nu": np.asarray(s.nu)}
