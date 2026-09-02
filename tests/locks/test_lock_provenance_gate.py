"""Provenance gate for the tier-1 regression locks in ``tests/locks/``.

Policy (approved 2026-09-02). A lock pins a committed number, a committed
fixture, or bit-identity against an artifact, and the same FDTD code has been
measured to differ by up to 1.3e-3 relative between macOS and the Linux
cluster -- so a pinned number without its origin is unreproducible, and a red
lock cannot be diagnosed as platform-vs-code. Every ``tests/locks/test_*.py``
module therefore declares a module-level ``LOCK_PROVENANCE`` dict naming the
fixture it reads (or ``"none"``), the generator that produced the pinned
numbers (a script path, ``"hand-derived"`` or ``"unknown"``), the short commit
and date the numbers came from, the run id (a VESSL id, ``"local"`` or
``"unknown"``), the host (e.g. ``"macOS jax 0.8 cpu"`` /
``"remilab-c0 rtx4090 jax 0.4.33"`` / ``"unknown"``) and a ``pinned_until``
expiry. One lock per artifact; a retired number is deleted, not quoted; unknown
provenance gets a short expiry. When ``pinned_until`` passes, the lock must be
re-blessed: regenerate the number, update ``commit``/``date``, extend
``pinned_until`` -- the gate fails an expired lock so it cannot be forgotten.
The dict is read with ``ast`` (the lock modules are never imported here), so
this gate is independent of JAX and of the physics it guards.
"""

from __future__ import annotations

import ast
import datetime as _dt
import glob
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_LOCKS_DIR = _HERE.parent
_REPO = _HERE.parents[2]

REQUIRED_KEYS = (
    "fixture", "generator", "commit", "date", "run_id", "host", "pinned_until",
)

#: Deleting every lock module would make this gate vacuously green, so require
#: the population it was written for (the 2026-09-02 tier-1 move) to be there.
MIN_LOCK_MODULES = 12

_NO_FIXTURE = "none"


def _lock_modules() -> list[Path]:
    return sorted(
        p for p in _LOCKS_DIR.glob("test_*.py") if p.name != _HERE.name
    )


def _rel(path: Path) -> str:
    return str(path.relative_to(_REPO))


def _read_provenance(path: Path) -> dict[str, str]:
    """Return the module-level ``LOCK_PROVENANCE`` literal, or fail."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = None
    for node in tree.body:
        targets: tuple[str, ...] = ()
        if isinstance(node, ast.Assign):
            targets = tuple(
                t.id for t in node.targets if isinstance(t, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = (node.target.id,)
        if "LOCK_PROVENANCE" in targets:
            found = node.value
            break
    assert found is not None, (
        f"{_rel(path)} declares no module-level LOCK_PROVENANCE dict "
        f"(required keys: {', '.join(REQUIRED_KEYS)})"
    )
    assert isinstance(found, ast.Dict), (
        f"{_rel(path)}: LOCK_PROVENANCE must be a dict literal of string keys "
        "and string values (it is parsed with ast, not imported)"
    )
    out: dict[str, str] = {}
    for k, v in zip(found.keys, found.values):
        assert isinstance(k, ast.Constant) and isinstance(k.value, str), (
            f"{_rel(path)}: LOCK_PROVENANCE keys must be string literals"
        )
        assert isinstance(v, ast.Constant) and isinstance(v.value, str), (
            f"{_rel(path)}: LOCK_PROVENANCE[{k.value!r}] must be a string literal"
        )
        out[k.value] = v.value
    return out


def _parse_date(value: str, label: str, path: Path) -> _dt.date:
    try:
        return _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise AssertionError(
            f"{_rel(path)}: LOCK_PROVENANCE[{label!r}] = {value!r} "
            "is not a YYYY-MM-DD date"
        ) from exc


def _is_short_sha(value: str) -> bool:
    return 7 <= len(value) <= 40 and all(c in "0123456789abcdef" for c in value)


_MODULES = _lock_modules()
_IDS = [p.stem for p in _MODULES]


def test_the_lock_population_is_still_present():
    assert len(_MODULES) >= MIN_LOCK_MODULES, (
        f"only {len(_MODULES)} lock modules under tests/locks/; expected at least "
        f"{MIN_LOCK_MODULES}. A lock that is retired is deleted deliberately -- "
        "lower MIN_LOCK_MODULES in the same change and say why."
    )


@pytest.mark.parametrize("path", _MODULES, ids=_IDS)
def test_lock_module_declares_complete_provenance(path: Path):
    prov = _read_provenance(path)
    missing = [k for k in REQUIRED_KEYS if k not in prov]
    assert not missing, f"{_rel(path)}: LOCK_PROVENANCE is missing {missing}"
    empty = [k for k in REQUIRED_KEYS if not prov[k].strip()]
    assert not empty, f"{_rel(path)}: LOCK_PROVENANCE has empty values for {empty}"
    assert _is_short_sha(prov["commit"]) or prov["commit"] == "unknown", (
        f"{_rel(path)}: LOCK_PROVENANCE['commit'] = {prov['commit']!r} is not a "
        "short sha (or 'unknown')"
    )
    _parse_date(prov["date"], "date", path)
    _parse_date(prov["pinned_until"], "pinned_until", path)


@pytest.mark.parametrize("path", _MODULES, ids=_IDS)
def test_lock_fixture_paths_exist(path: Path):
    prov = _read_provenance(path)
    fixture = prov.get("fixture", _NO_FIXTURE).strip()
    if fixture == _NO_FIXTURE:
        return
    for entry in (e.strip() for e in fixture.split(",")):
        if any(ch in entry for ch in "*?["):
            assert glob.glob(str(_REPO / entry)), (
                f"{_rel(path)}: LOCK_PROVENANCE['fixture'] pattern {entry!r} "
                "matches nothing"
            )
        else:
            assert (_REPO / entry).exists(), (
                f"{_rel(path)}: LOCK_PROVENANCE['fixture'] {entry!r} does not exist"
            )


@pytest.mark.parametrize("path", _MODULES, ids=_IDS)
def test_lock_is_not_expired(path: Path):
    prov = _read_provenance(path)
    pinned_until = _parse_date(prov["pinned_until"], "pinned_until", path)
    date = _parse_date(prov["date"], "date", path)
    today = _dt.date.today()
    assert date <= today, (
        f"{_rel(path)}: LOCK_PROVENANCE['date'] {date} is in the future"
    )
    assert pinned_until >= date, (
        f"{_rel(path)}: pinned_until {pinned_until} precedes date {date}"
    )
    assert pinned_until >= today, (
        f"{_rel(path)}: lock expired on {pinned_until} (pinned at "
        f"{prov['commit']} on {date}, run_id={prov['run_id']!r}, "
        f"host={prov['host']!r}). Re-bless it: regenerate the number with "
        f"generator={prov['generator']!r}, update commit/date/run_id/host, "
        "and extend pinned_until -- do not just move the date."
    )
