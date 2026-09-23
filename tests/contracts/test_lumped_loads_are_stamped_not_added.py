"""A port's load or an RLC element is a device across ONE Yee edge, not a cell
material (#1210). It is folded into that cell's σ / ε_r as an equivalent, and
since the E coefficients average the four cells incident to each edge, a load
written with a bare ``sigma.at[...].add(...)`` is averaged with its
neighbours and quartered: a 50 Ω termination presents 200 Ω, a passive wire
port reads S11 = +1/3 instead of −1/3. The graded-mesh and subgridded runners
did exactly that after the uniform runner had been converted, and no fixture
on the subgridded lane could see it.

The rule: every additive write of a load into ``sigma`` or ``eps_r`` under the
lanes below goes through ``stamp_lumped_sigma`` / ``stamp_lumped_eps``
(``rfx/sources/sources.py``), which record the stamp so the average removes it
and adds it back at its own cell. This scan reads the source: a bare
``.at[...].add(`` on ``sigma`` / ``eps_r`` anywhere outside those two helpers
is red. Slice ``.set(`` writes (geometry, e.g. the coax junction) are not loads
and are not matched.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LANES = ("rfx/runners", "rfx/sources", "rfx/sparams", "rfx/simulation.py", "rfx/nonuniform.py")
ALLOWED = {("rfx/sources/sources.py", "stamp_lumped_sigma"),
           ("rfx/sources/sources.py", "stamp_lumped_eps")}
_BARE_ADD = re.compile(r"\b(sigma|eps_r)\.at\[[^\]]*\]\.add\(")
_DEF = re.compile(r"^\s*def\s+(\w+)\s*\(")


def bare_load_adds(root: Path) -> list[str]:
    """Every ``sigma/eps_r .at[...].add(`` outside the two stamp helpers, as
    ``path:line`` strings, under the lanes that stamp loads."""
    hits = []
    files = []
    for lane in LANES:
        p = root / lane
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    for f in files:
        rel = f.relative_to(root).as_posix()
        enclosing = None
        for n, line in enumerate(f.read_text().splitlines(), 1):
            m = _DEF.match(line)
            if m:
                enclosing = m.group(1)
            if _BARE_ADD.search(line) and (rel, enclosing) not in ALLOWED:
                hits.append(f"{rel}:{n}")
    return hits


def test_every_load_goes_through_the_stamp_helpers():
    assert bare_load_adds(REPO) == []


def test_the_scan_sees_a_bare_add_when_one_is_put_back(tmp_path):
    """Mutation (b): the helpers stay, one runner's stamp is reverted to the
    bare form on a copy of the tree; the scan must name that line."""
    for lane in LANES:
        src = REPO / lane
        dst = tmp_path / lane
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    victim = tmp_path / "rfx/runners/subgridded.py"
    text = victim.read_text()
    assert "stamp_lumped_sigma(" in text
    victim.write_text(text.replace(
        "stamp_lumped_sigma(",
        "(lambda m, c, v: m._replace(sigma=m.sigma.at[c].add(v)))(", 1))
    hits = bare_load_adds(tmp_path)
    assert any(h.startswith("rfx/runners/subgridded.py:") for h in hits), hits
