"""Every re-capture pointer in a committed waveguide fixture resolves in the tree.

The chain-closure contract (``docs/design_notes/chain_closure_contract.md``,
criterion 4) requires "re-capture commands that either run from a clean
checkout or name a tracked VESSL YAML". Before v1.8 every uniform broad-E5
envelope set ``rfx_manifest_path`` into the gitignored ``.omx/physics-gate/``,
the graded-dy envelope set it to an absolute path inside a secondary checkout
that no longer exists, and two absorber notes cited a job YAML that was never
committed. A pointer nobody can open is a claim nobody can re-check.

Now the pointer names the lane's tracked entry point and the sibling
``recapture_note`` keeps the original manifest location as provenance
(``scripts/diagnostics/_fixture_recapture.py``). This module is the persistent
form of the contract's shell falsifier::

    for p in $(jq -r '..|strings|select(startswith("scripts/"))' \\
             tests/fixtures/waveguide_*/*.json); do
      git ls-files --error-unmatch $p; done

plus two things the shell loop cannot say: that no fixture string carries a
private absolute path, and that the note in each fixture is exactly what its
builder would emit for the same inputs -- so a regeneration cannot quietly
bring the dangling pointer back.

Pure-Python, no FDTD. The #812 artifact lane's fixture
(``wr90_rectangular_broad_e4_comparison.json``) is read like every other file
under the glob but is not edited by this lane.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from _fixture_recapture import (  # type: ignore  # noqa: E402
    NU_WR90_E5_SETTLING_WITNESS_PRODUCER,
    RECAPTURE_NOTE_KEY,
    RECAPTURE_POINTER_KEY,
    nu_wr90_note,
    uniform_band_magnitude_note,
    uniform_band_phase_note,
)

FIXTURES = sorted(REPO.glob("tests/fixtures/waveguide_*/*.json"))
BAND_LABELS = {"wr28_kaband": "Ka", "wr62_kuband": "Ku", "wr15_vband": "V",
               "wr340_sband": "S", "wr10_wband": "W"}
PRIVATE_PREFIXES = ("/root/", "/home/", "~/", "/Users/")


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _tracked(relpath: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", relpath],
                       cwd=REPO, capture_output=True, text=True)
    return r.returncode == 0


def _with_note(paths):
    return [p for p in paths if RECAPTURE_NOTE_KEY in json.loads(p.read_text())]


def test_the_glob_finds_the_lanes() -> None:
    assert FIXTURES, "no waveguide fixtures found -- the glob is broken"
    noted = {p.name for p in _with_note(FIXTURES)}
    # five uniform magnitude + five uniform phase + the graded-dy lane
    assert len(noted) == 11, sorted(noted)
    assert "waveguide_wr90_nu_flux_broad_e5_envelope.json" in noted


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_every_scripts_pointer_is_a_bare_tracked_path(path: Path) -> None:
    """The contract's shell falsifier, in-process.

    A string that starts with ``scripts/`` is a pointer; it must be a bare
    path (no prose after it -- the shell loop word-splits) and ``git ls-files``
    must return it.
    """
    for s in _strings(json.loads(path.read_text())):
        if s.startswith("scripts/"):
            assert " " not in s, f"{path.name}: pointer carries prose: {s[:80]!r}"
            assert _tracked(s), f"{path.name}: {s} is not tracked"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_no_private_absolute_path_anywhere(path: Path) -> None:
    for s in _strings(json.loads(path.read_text())):
        for prefix in PRIVATE_PREFIXES:
            assert not s.startswith(prefix), f"{path.name}: {s[:80]!r}"
            assert f" {prefix}" not in s, f"{path.name}: {s[:120]!r}"


@pytest.mark.parametrize("path", _with_note(FIXTURES), ids=lambda p: p.name)
def test_recapture_pointer_is_tracked_and_note_matches_its_builder(path: Path) -> None:
    d = json.loads(path.read_text())
    pointer = d[RECAPTURE_POINTER_KEY]
    assert pointer.startswith("scripts/") and _tracked(pointer), pointer
    note = d[RECAPTURE_NOTE_KEY]
    original = note["original_manifest_path"]
    assert original.startswith(".omx/"), (
        "the original manifest location is provenance and must be kept, "
        "checkout-relative")

    name = path.name
    status = note["original_manifest_status"]
    if name == "waveguide_wr90_nu_flux_broad_e5_envelope.json":
        expected = nu_wr90_note(manifest_relpath=original,
                                original_manifest_status=status)
        witness = d["settling_witness"]
        assert witness["artifact_producer"] == NU_WR90_E5_SETTLING_WITNESS_PRODUCER
        assert _tracked(witness["artifact_producer"])
    else:
        token = name[len("waveguide_"):].split("_broad_e5_")[0]
        label = BAND_LABELS[token]
        maker = (uniform_band_phase_note if name.endswith("_phase_envelope.json")
                 else uniform_band_magnitude_note)
        expected = maker(band_token=token, band_label=label,
                         manifest_relpath=original, original_manifest_status=status)
    assert note == expected, (
        f"{name}: recapture_note differs from what its builder emits -- "
        "regenerate through the builder, do not hand-edit the note")


@pytest.mark.parametrize(
    "name", ["waveguide_wr28_kaband_broad_e5_envelope.json",
             "waveguide_wr15_vband_broad_e5_envelope.json"])
def test_absorber_probe_names_its_run_not_an_untracked_yaml(name: str) -> None:
    """The two probed bands used to cite a job YAML that was never committed."""
    d = json.loads((REPO / "tests/fixtures/waveguide_broad_e5" / name).read_text())
    ad = d["absorber_discipline"]
    assert ad["status"] == "below_floor_accepted"   # never removed by this lane
    probe = ad["probe"]
    assert probe["vessl_run_id"] == "369367252292"
    assert probe["vessl_run_id"] in ad["note"]
    assert _tracked(probe["sweep_script"])
    assert "--cpml-fraction 0.75" in probe["sweep_args"]
    assert "vessl_i496_band_absorber_probe.yaml" not in ad["note"]
