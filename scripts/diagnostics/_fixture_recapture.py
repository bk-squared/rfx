"""Re-capture pointer block shared by the committed waveguide envelope builders.

A committed envelope fixture has to be re-creatable from the tree alone. The
sweep manifest that actually built each one lives under the gitignored
``.omx/`` (one lane even recorded it as an absolute path inside a secondary
checkout that no longer exists), so a fixture whose ``rfx_manifest_path``
points there names something no reader can open. The v1.8 chain-closure
contract (``docs/design_notes/chain_closure_contract.md``, criterion 4) asks
for "re-capture commands that either run from a clean checkout or name a
tracked VESSL YAML", so:

* ``rfx_manifest_path`` names the TRACKED entry point of the lane's re-capture
  chain -- the VESSL job YAML where one exists, else the sweep script that runs
  from a clean checkout; and
* the sibling ``recapture_note`` keeps the manifest's original location and
  status as provenance, plus the exact sweep/builder steps, so nothing is
  erased and nothing dangles.

Every value that starts with ``scripts/`` is a bare path: the contract's
falsifier resolves each such string with ``git ls-files --error-unmatch``
(``tests/test_waveguide_fixture_recapture_pointers.py`` runs the same check).
The builders here emit the block and the fixtures carry it; the test proves the
two agree, so a regeneration cannot quietly bring the dangling pointer back.
"""
from __future__ import annotations

from pathlib import Path

RECAPTURE_POINTER_KEY = "rfx_manifest_path"
RECAPTURE_NOTE_KEY = "recapture_note"

# Uniform five-band lane (WR-340 / WR-62 / WR-28 / WR-15 / WR-10): a tracked
# VESSL job YAML runs the sweep and the magnitude builder for all five bands.
UNIFORM_BAND_E5_JOB_YAML = "scripts/vessl_waveguide_broad_e5.yaml"
UNIFORM_BAND_E5_SWEEP_SCRIPT = (
    "scripts/diagnostics/run_waveguide_band_broad_e5_flux_sweep.py")
UNIFORM_BAND_E5_MAGNITUDE_BUILDER = (
    "scripts/diagnostics/build_waveguide_band_broad_e5_envelope.py")
UNIFORM_BAND_E5_PHASE_BUILDER = (
    "scripts/diagnostics/build_waveguide_band_broad_e5_phase_envelope.py")

# WR-90 graded-dy (nonuniform) lane: no committed job YAML. The sweep script
# runs from a clean checkout on a GPU host and writes its manifest exactly
# where the builder reads it; the settling-witness artifacts the builder
# insists on come from the record-window falsifier.
NU_WR90_E5_SWEEP_SCRIPT = (
    "scripts/diagnostics/run_waveguide_wr90_nu_flux_broad_e5_sweep.py")
NU_WR90_E5_BUILDER = (
    "scripts/diagnostics/build_waveguide_wr90_nu_flux_broad_e5_envelope.py")
NU_WR90_E5_SETTLING_WITNESS_PRODUCER = (
    "scripts/diagnostics/i574_e5_absorber_window_falsifier.py")

NU_SETTLING_WITNESS_ARTIFACT_STATUS = (
    "gitignored (.omx/), not in the tree; the cells below are copied from it, "
    "so the gate reads this fixture, not the artifact")

GITIGNORED_MANIFEST_STATUS = (
    "gitignored (.omx/), not in the tree; commit_hash, generated_at and "
    "runtime_env above identify the capture that produced it")

CASES_NPZ_STATUS = (
    "per-case raw-data records under the same gitignored run directory: "
    "provenance of THIS capture, regenerated (not resolved) by the "
    "re-capture command")


def repo_relative(path: Path | str, repo: Path | str) -> str:
    """``path`` relative to ``repo`` when it lies inside it, else unchanged.

    Builders receive the manifest path from the command line (relative or
    absolute) or from a module constant built on ``REPO``; the note must not
    depend on which, and must never carry a private absolute path.
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path(repo).resolve()))
    except ValueError:
        return str(p)


def _note(*, entry_point_is: str, recapture_command: str, sweep_script: str,
          sweep_args: str, builder_script: str, builder_args: str,
          original_manifest_path: str, original_manifest_status: str,
          **extra: str) -> dict:
    note = {
        "rfx_manifest_path_is": entry_point_is,
        "recapture_command": recapture_command,
        "sweep_script": sweep_script,
        "sweep_args": sweep_args,
        "builder_script": builder_script,
        "builder_args": builder_args,
    }
    note.update(extra)
    note.update({
        "original_manifest_path": original_manifest_path,
        "original_manifest_status": original_manifest_status,
        "cases_rfx_npz_status": CASES_NPZ_STATUS,
    })
    return note


def _band_and_date(band_token: str, manifest_relpath: str) -> tuple[str, str]:
    # "wr28_kaband" -> "WR28" (the sweep's --band spelling); the run directory
    # is "<date-tag>-waveguide-broad-e5-<band_token>-flux".
    band = band_token.split("_")[0].upper()
    run_dir = Path(manifest_relpath).parent.parent.name
    date_tag = run_dir.split("-waveguide-")[0] if "-waveguide-" in run_dir else "<date-tag>"
    return band, date_tag


def uniform_band_magnitude_note(*, band_token: str, band_label: str,
                                manifest_relpath: str,
                                original_manifest_status: str = GITIGNORED_MANIFEST_STATUS) -> dict:
    band, date_tag = _band_and_date(band_token, manifest_relpath)
    return _note(
        entry_point_is=(
            "the tracked re-capture entry point for this lane (a VESSL job YAML "
            "that runs sweep_script then builder_script for all five bands), not "
            "the sweep manifest this envelope was built from; that manifest lives "
            "under the gitignored .omx/ and is not in the tree "
            "(original_manifest_path below)"),
        recapture_command=f"vessl run create -f {UNIFORM_BAND_E5_JOB_YAML}",
        sweep_script=UNIFORM_BAND_E5_SWEEP_SCRIPT,
        sweep_args=f"--band {band} --date-tag {date_tag}",
        builder_script=UNIFORM_BAND_E5_MAGNITUDE_BUILDER,
        builder_args=(f"--manifest {manifest_relpath} --band-token {band_token} "
                      f"--band-label {band_label}"),
        original_manifest_path=manifest_relpath,
        original_manifest_status=original_manifest_status,
    )


def uniform_band_phase_note(*, band_token: str, band_label: str,
                            manifest_relpath: str,
                            original_manifest_status: str = GITIGNORED_MANIFEST_STATUS) -> dict:
    band, date_tag = _band_and_date(band_token, manifest_relpath)
    return _note(
        entry_point_is=(
            "the tracked re-capture entry point for this lane (the VESSL job YAML "
            "that produces the npz/manifest pair this phase envelope re-analyses, "
            "see source_data_provenance), not the sweep manifest itself, which "
            "lives under the gitignored .omx/ and is not in the tree "
            "(original_manifest_path below). The YAML does not run the phase "
            "builder; builder_script is run afterwards from a checkout holding "
            "that pair"),
        recapture_command=(f"vessl run create -f {UNIFORM_BAND_E5_JOB_YAML}; then "
                           f"python {UNIFORM_BAND_E5_PHASE_BUILDER}"),
        sweep_script=UNIFORM_BAND_E5_SWEEP_SCRIPT,
        sweep_args=f"--band {band} --date-tag {date_tag}",
        builder_script=UNIFORM_BAND_E5_PHASE_BUILDER,
        builder_args=("(none: the builder resolves each band's manifest from its "
                      "BANDS table and writes this file directly)"),
        original_manifest_path=manifest_relpath,
        original_manifest_status=original_manifest_status,
    )


def nu_wr90_note(*, manifest_relpath: str,
                 original_manifest_status: str = GITIGNORED_MANIFEST_STATUS) -> dict:
    return _note(
        entry_point_is=(
            "the tracked re-capture entry point for this lane (the sweep script, "
            "run from a clean checkout on a GPU host: this lane has no committed "
            "VESSL job YAML), not the sweep manifest this envelope was built "
            "from; that manifest lives under the gitignored .omx/ and is not in "
            "the tree (original_manifest_path below)"),
        recapture_command=(f"python {NU_WR90_E5_SETTLING_WITNESS_PRODUCER} --all && "
                           f"python {NU_WR90_E5_SWEEP_SCRIPT} && "
                           f"python {NU_WR90_E5_BUILDER}"),
        sweep_script=NU_WR90_E5_SWEEP_SCRIPT,
        sweep_args="(none: the sweep writes its manifest to the path the builder reads)",
        builder_script=NU_WR90_E5_BUILDER,
        builder_args=("(none: the builder reads MANIFEST and writes OUT from its "
                      "module constants)"),
        settling_witness_producer=NU_WR90_E5_SETTLING_WITNESS_PRODUCER,
        settling_witness_step=(
            "run settling_witness_producer at the promoted absorber and at both "
            "record windows before the builder: it refuses to build an envelope "
            "whose witness artifacts are missing"),
        original_manifest_path=manifest_relpath,
        original_manifest_status=original_manifest_status,
    )
