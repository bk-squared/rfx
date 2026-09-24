"""What the three chain-battery drift guards in ``tests/locks/`` share.

Each port family's chain battery stores the complex S it measured
(``tests/fixtures/<family>_chain_battery/fixture.json``) and its replay test
re-derives every verdict from that stored S. The replay solves nothing, so a
change that moves the solved line leaves it green on records that no longer
describe the code: when a Yee E component started taking the mean of its four
incident cells' permittivity (#1213), the microstrip battery's stub notch moved
from 2.62 % above the quarter-wave closed form to 0.81 % below it on the
100 um mesh, and nothing noticed.

The guards solve the CHEAPEST rung of each battery again, with the battery's
own driver so that the live case is the recorded case, check that the grid it
builds is the recorded grid, and hold the live S to the stored S with the bar
the battery itself is judged by (``docs/design_notes/chain_closure_contract.md``,
"The v2.0 battery for lumped/wire, MSL and coax", and the PI's deep-null
ruling of 2026-09-21): magnitude within 2 dB outside the core of a deep null,
a quantity that is near zero by construction held to -20 dB, and a frequency
feature (a notch, a reflection zero, a phase crossing) within 1 %.

This module holds the arithmetic the three guards share. The rules each family
applies are in its own lock module.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# The v2 accuracy bar. Written here, not read from a fixture, so a fixture
# cannot move the bar it is judged by.
MAGNITUDE_DB = 2.0
FREQUENCY_FRAC = 0.01
DEEP_NULL_DB = -20.0
COLUMN_POWER_MAX = 1.02
RECIPROCITY_MAX = 0.02
SETTLING_DB = -40.0

DB_FLOOR = 1e-300

# A realized-geometry block is cell counts, node indices and positions in
# metres. A cell is at least 1e-4 of any length in these blocks, so this
# tolerance only absorbs the last bits of platform arithmetic.
REALIZED_REL_TOL = 1e-9
REALIZED_ABS_TOL = 1e-15


def load_driver(relpath: str):
    """Import a battery's measurement driver once per process, under a private
    module name, so the guard builds its case with the driver's own code."""
    path = REPO / relpath
    name = f"_chain_battery_driver_{path.stem}"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def complex_array(block) -> np.ndarray:
    return (np.asarray(block["real"], dtype=float)
            + 1j * np.asarray(block["imag"], dtype=float))


def db(x) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(x)), DB_FLOOR))


def realized_differences(stored, live, where: str = "") -> list[str]:
    """Every place the live realized block differs from the recorded one.

    Each recorded key has to be present live with the same value. Keys the
    live block carries and the record does not are a newer driver recording
    more about the same build, and are not compared.
    """
    here = where or "the block"
    if isinstance(stored, dict):
        if not isinstance(live, dict):
            return [f"{here}: recorded a mapping, built {live!r}"]
        out: list[str] = []
        for key, value in stored.items():
            sub = f"{where}.{key}" if where else str(key)
            if key not in live:
                out.append(f"{sub}: recorded {value!r}, absent from the build")
            else:
                out.extend(realized_differences(value, live[key], sub))
        return out
    if isinstance(stored, (list, tuple)):
        if not isinstance(live, (list, tuple)) or len(live) != len(stored):
            return [f"{here}: recorded {stored!r}, built {live!r}"]
        out = []
        for i, (a, b) in enumerate(zip(stored, live)):
            out.extend(realized_differences(a, b, f"{where}[{i}]"))
        return out
    if isinstance(stored, bool) or stored is None or isinstance(stored, str):
        return [] if stored == live else [f"{here}: recorded {stored!r}, built {live!r}"]
    if isinstance(stored, (int, float)):
        if isinstance(live, bool) or not isinstance(live, (int, float, np.integer,
                                                           np.floating)):
            return [f"{here}: recorded {stored!r}, built {live!r}"]
        if math.isclose(float(stored), float(live), rel_tol=REALIZED_REL_TOL,
                        abs_tol=REALIZED_ABS_TOL):
            return []
        return [f"{here}: recorded {stored!r}, built {live!r}"]
    return [] if stored == live else [f"{here}: recorded {stored!r}, built {live!r}"]


def magnitude_findings(label: str, freqs, stored, live, *, deep=None,
                       report: list | None = None) -> list[str]:
    """``|live|`` against ``|stored|`` in dB at every bin outside ``deep``.

    ``deep`` marks the core of a null — bins where the reference is at or below
    the deep-null level — in which a dB difference is the difference of two
    near-zeros and says nothing (the PI's 2026-09-21 ruling). By default the
    core is where the STORED curve is at or below -20 dB. The worst distance is
    appended to ``report`` whether or not it is inside the bar.
    """
    freqs = np.asarray(freqs, dtype=float)
    d_stored, d_live = db(stored), db(live)
    if deep is None:
        deep = d_stored <= DEEP_NULL_DB
    keep = ~np.asarray(deep, dtype=bool)
    if not keep.any():
        return [f"{label}: every bin is inside a deep-null core, so nothing was "
                "compared"]
    diff = np.where(keep, np.abs(d_live - d_stored), -np.inf)
    k = int(np.argmax(diff))
    n_core = int((~keep).sum())
    outside = f" outside the {n_core}-bin deep-null core" if n_core else ""
    if report is not None:
        report.append(f"{label} {diff[k]:.3f} dB at {freqs[k] / 1e9:.3f} GHz{outside}")
    if diff[k] <= MAGNITUDE_DB:
        return []
    return [f"{label} moved {diff[k]:.3f} dB at {freqs[k] / 1e9:.4f} GHz "
            f"(stored {d_stored[k]:.3f} dB, now {d_live[k]:.3f} dB); the bar is "
            f"{MAGNITUDE_DB} dB{outside}"]


def bound_findings(label: str, freqs, live, *,
                   report: list | None = None) -> list[str]:
    """A quantity near zero by construction, held to the deep-null level at
    every bin and compared with nothing."""
    d_live = db(live)
    k = int(np.argmax(d_live))
    at = np.asarray(freqs, dtype=float)[k] / 1e9
    if report is not None:
        report.append(f"{label} at most {d_live[k]:.2f} dB ({at:.3f} GHz)")
    if d_live[k] <= DEEP_NULL_DB:
        return []
    return [f"{label} reaches {d_live[k]:.3f} dB at {at:.4f} GHz, above its "
            f"{DEEP_NULL_DB} dB bound"]


def frequency_findings(label: str, stored_hz: float, live_hz: float, *,
                       report: list | None = None) -> list[str]:
    frac = abs(live_hz - stored_hz) / stored_hz
    if report is not None:
        report.append(f"{label} {stored_hz / 1e9:.5f} -> {live_hz / 1e9:.5f} GHz "
                      f"({frac * 100:.3f} %)")
    if frac <= FREQUENCY_FRAC:
        return []
    return [f"{label} moved from {stored_hz / 1e9:.5f} GHz to {live_hz / 1e9:.5f} GHz, "
            f"{frac * 100:.3f} % against a {FREQUENCY_FRAC * 100:.0f} % bar"]


def crossing_findings(label: str, stored: list[dict], live: list[dict],
                      f_lo: float, f_hi: float, *,
                      report: list | None = None) -> list[str]:
    """Every stored phase crossing against the live one of the same kind, and
    every live crossing against the stored ones.

    A crossing is a frequency at which the unwrapped angle passes a multiple
    of pi (``{"hz": ..., "multiple_of_pi": ...}``, the driver's own
    ``phase_crossings``). Its kind is the parity of that multiple — S11 real
    and positive or real and negative — which does not depend on the branch
    ``numpy.unwrap`` started from. Each crossing needs a partner of the same
    kind within 1 %.

    The one exemption is at the ends of the sweep: a crossing whose 1 % window
    reaches past a band edge can leave the band while moving less than 1 %, so
    a missing partner there is not evidence of drift. A partner that moved
    INTO the band by more than 1 % is still caught, from the other side.
    """
    def kind(c):
        return int(c["multiple_of_pi"]) % 2

    def window_inside(f):
        return f * (1.0 - FREQUENCY_FRAC) >= f_lo and f * (1.0 + FREQUENCY_FRAC) <= f_hi

    findings: list[str] = []
    matched: list[float] = []
    for mine, theirs, what in ((stored, live, "stored"), (live, stored, "live")):
        for c in mine:
            same = [o["hz"] for o in theirs if kind(o) == kind(c)]
            near = min(same, key=lambda f: abs(f - c["hz"]), default=None)
            frac = None if near is None else abs(near - c["hz"]) / c["hz"]
            if frac is not None and frac <= FREQUENCY_FRAC:
                if what == "stored":
                    matched.append(frac)
                continue
            if not window_inside(c["hz"]):
                continue
            partner = ("none of that kind" if near is None
                       else f"the nearest at {near / 1e9:.5f} GHz, {frac * 100:.3f} % away")
            other = "live" if what == "stored" else "stored"
            findings.append(
                f"{label}: the {what} crossing at {c['hz'] / 1e9:.5f} GHz "
                f"(S11 real, {'negative' if kind(c) else 'positive'}) has no {other} "
                f"partner within {FREQUENCY_FRAC * 100:.0f} % — {partner}")
    if report is not None and matched:
        report.append(f"{label} {len(matched)} crossings, worst "
                      f"{max(matched) * 100:.3f} %")
    if not matched and not findings:
        findings.append(f"{label}: no stored crossing was compared, so the phase "
                        "was not checked at all")
    return findings


def verdict_findings(stored_entry: dict, live_entry: dict, paths) -> list[str]:
    """The record's own boolean verdicts against the same verdicts computed
    from the live S. A verdict that flips means the record no longer says what
    this tree does, in whichever direction it flips."""
    findings = []
    for path in paths:
        s, v = stored_entry, live_entry
        for part in path:
            s = None if s is None else s.get(part)
            v = None if v is None else v.get(part)
        if s is None:
            continue
        if bool(v) != bool(s):
            findings.append(f"the record's verdict {'.'.join(path)} was {s}, the live "
                            f"S gives {v}")
    return findings


def stale_record(family: str, key: str, findings: list[str], remeasure: str) -> str:
    """The failure message: the record no longer describes this tree, what
    moved, and how to measure it again."""
    lines = [f"The {family} chain battery's stored record {key!r} no longer "
             "describes the code under test (its replay test reads only the stored "
             "S and cannot see this):"]
    lines += [f"  - {f}" for f in findings]
    lines.append(f"Re-measure the battery — {remeasure} — re-assemble its fixture, "
                 "and move this guard's LOCK_PROVENANCE to the new records.")
    return "\n".join(lines)
