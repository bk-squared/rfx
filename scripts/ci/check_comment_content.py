"""Check the content of Python comments on added working-tree lines.

Under rfx/, tests/, examples/ and scripts/ an added comment may only be a tool
directive: noqa, flake8: noqa, type: ignore, pragma: no cover|branch, fmt:,
isort:, ruff:, mypy:, pyright: or pylint:, several of them chained with ``#``,
plus a shebang on line 1 and a PEP 263 coding line on line 1 or 2. Anything else,
including text after a directive, is refused there. Every other Python path is
checked against the narrow rules: no date, issue number, predeclaration, move
note, run id or quoted Korean text.

Every tracked Python file that an entry of SWEPT_PREFIXES matches is also read
whole on every run, whatever the base, so that a file moved unchanged into a
swept package cannot bring prose comments with it. An entry is a
repository-relative package path ending with ``/`` or one file ending with
``.py``, under one of the four directories, and it must match a tracked file.
``--swept`` replaces the entries for one run; ``--swept`` with no value runs
with none.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import tokenize
from typing import Iterator, NamedTuple


RULES = (
    ("date", re.compile(
        r"\b(?:19|20)\d\d[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])\b"
        r"|\b20\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\b"
    ), "a dated record, or git history"),
    ("issue number", re.compile(r"(?<!&)#\d{3,}\b|(?i:\bPR) ?#\d+|/issues/\d+|/pull/\d+"),
     "the PR body or the issue; a strict-xfail reason (a string) may carry it"),
    ("predeclaration", re.compile(r"predeclar", re.I), "the predeclaration file"),
    ("move note", re.compile(r"\bmoved from\b", re.I), "git history (git log --follow)"),
    ("run id", re.compile(r"(?<![\d.])\d{10,}(?!\d)(?!\.\d)"), "the measurement record"),
    ("quoted Korean", re.compile(
        r"[\"'“”‘’「」『』][^\"'“”‘’「」『』]*[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]"
        r"[^\"'“”‘’「」『』]*[\"'“”‘’「」『』]"
    ), "a named test or assertion states the rule; the rationale goes to the record"),
)
GUIDANCE = "a comment says what the code does, not where it came from"
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
FULL_BAN_PREFIXES = ("rfx/", "tests/", "examples/", "scripts/")
FULL_BAN_DESTINATION = (
    "a name, an assertion, a test, the docstring, or a dated record; under rfx/, tests/, examples/ and scripts/"
    " a comment may only be a tool directive (noqa, type:, pragma:, fmt:, isort:, ruff:, mypy:, pyright:, pylint:)"
)
_CODE = r"[A-Z]+[0-9]+"
_CODES = rf"{_CODE}(?:[,\s]+{_CODE})*"
_NAME = r"[A-Za-z][\w-]*"
_BRACKET = rf"\[\s*{_NAME}(?:\s*,\s*{_NAME})*\s*\]"
_NOQA = r"(?i:noqa)"
_QUOTED_NAMES = rf'"\s*{_NAME}(?:\s*,\s*{_NAME})*\s*"'
_MYPY_OPTION = rf"{_NAME}(?:\s*=\s*(?:{_QUOTED_NAMES}|[\w,-]+))?"
_PYRIGHT_SETTING = rf"{_NAME}\s*=\s*(?:true|false|none|error|warning|information)"
DIRECTIVES = (
    rf"{_NOQA}(?:\s*:\s*{_CODES})?",
    rf"flake8\s*:\s*{_NOQA}",
    rf"type\s*:\s*ignore(?:{_BRACKET})?",
    r"pragma\s*:\s*no\s+(?:cover|branch)",
    r"fmt\s*:\s*(?:off|on|skip)",
    r"isort\s*:\s*(?:skip_file|skip|off|on|split)",
    rf"ruff\s*:\s*{_NOQA}(?:\s*:\s*{_CODES})?",
    r"ruff\s*:\s*isort\s*:\s*skip_file",
    rf"mypy\s*:\s*{_MYPY_OPTION}(?:\s*,\s*{_MYPY_OPTION})*",
    rf"pyright\s*:\s*(?:basic|standard|strict|ignore(?:{_BRACKET})?"
    rf"|{_PYRIGHT_SETTING}(?:\s*,\s*{_PYRIGHT_SETTING})*)",
    rf"pylint\s*:\s*(?:skip-file|(?:disable-next|disable|enable)\s*=\s*{_NAME}(?:\s*,\s*{_NAME})*)",
)
_DIRECTIVE = "(?:" + "|".join(DIRECTIVES) + ")"
DIRECTIVE_TOKEN = re.compile(rf"#\s*{_DIRECTIVE}(?:\s+#\s*{_DIRECTIVE})*\s*")
CODING_LINE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")
SWEPT_PREFIXES: tuple[str, ...] = (
    "rfx/_declarations.py",
    "rfx/_diagnostic_transport.py",
    "rfx/_gradient_witness_registry.py",
    "rfx/_host_limits.py",
    "rfx/_internal/__init__.py",
    "rfx/_internal/_grid_metric.py",
    "rfx/_internal/_precision.py",
    "rfx/_internal/_realized.py",
    "rfx/_internal/_uniform_specs.py",
    "rfx/_minimize.py",
    "rfx/_moved_names.py",
    "rfx/_plotting.py",
    "rfx/_radiated_power.py",
    "rfx/_ringdown_jax.py",
    "rfx/_runtime_floor.py",
    "rfx/_traced_input_table.py",
    "rfx/_traced_inputs.py",
    "rfx/_traced_refusal.py",
    "rfx/_warning_origin.py",
    "rfx/ad_diagnostics.py",
    "rfx/adjoint.py",
    "rfx/animation.py",
    "rfx/antenna.py",
    "rfx/api/__init__.py",
    "rfx/api/_ad_carry.py",
    "rfx/api/_ad_memory.py",
    "rfx/api/_ad_memory_compiled.py",
    "rfx/api/_ad_memory_model.py",
    "rfx/api/_ad_memory_types.py",
    "rfx/api/_artifact_serialization.py",
    "rfx/api/_artifacts.py",
    "rfx/api/_compile.py",
    "rfx/api/_compiled_memory.py",
    "rfx/api/_forward_result.py",
    "rfx/api/_gradient_witness.py",
    "rfx/api/_mesh.py",
    "rfx/api/_port_kind.py",
    "rfx/api/_preflight.py",
    "rfx/api/_refplane_checks.py",
    "rfx/api/_removed_kwargs.py",
    "rfx/api/_result_health.py",
    "rfx/api/_settling.py",
    "rfx/api/_source_semantics.py",
    "rfx/api/_sparams.py",
    "rfx/api/_spec.py",
    "rfx/api/_validators.py",
    "rfx/api/_wall_loss.py",
    "rfx/api/_waveguide_result.py",
    "rfx/artifacts.py",
    "rfx/auto_config.py",
    "rfx/backend.py",
    "rfx/batch.py",
    "rfx/boundaries/__init__.py",
    "rfx/boundaries/_cpml_slab.py",
    "rfx/boundaries/_face_frame.py",
    "rfx/boundaries/axes.py",
    "rfx/boundaries/cpml.py",
    "rfx/boundaries/depths.py",
    "rfx/boundaries/electric_coefficient.py",
    "rfx/boundaries/graded_periodic.py",
    "rfx/boundaries/model.py",
    "rfx/boundaries/pec.py",
    "rfx/boundaries/pmc.py",
    "rfx/boundaries/serialization.py",
    "rfx/boundaries/setup.py",
    "rfx/boundaries/spec.py",
    "rfx/boundaries/tfsf.py",
    "rfx/boundaries/upml.py",
    "rfx/boundaries/waveguide_edges.py",
    "rfx/checkpoint.py",
    "rfx/cli.py",
    "rfx/config/",
    "rfx/convergence.py",
    "rfx/core/",
    "rfx/current_moments.py",
    "rfx/dashboard/",
    "rfx/deembed.py",
    "rfx/diagnostic_records.py",
    "rfx/diagnostics.py",
    "rfx/differentiable_material_fit.py",
    "rfx/eigenmode.py",
    "rfx/errors.py",
    "rfx/farfield.py",
    "rfx/fidelity.py",
    "rfx/floquet.py",
    "rfx/geometry/__init__.py",
    "rfx/geometry/_mesh_scanline.py",
    "rfx/geometry/_pole_keying.py",
    "rfx/geometry/_traced_conductor.py",
    "rfx/geometry/box_content.py",
    "rfx/geometry/cell_table/__init__.py",
    "rfx/geometry/cell_table/body.py",
    "rfx/geometry/cell_table/box_overlap.py",
    "rfx/geometry/cell_table/nodes.py",
    "rfx/geometry/cell_table/table.py",
    "rfx/geometry/conformal.py",
    "rfx/geometry/conformal_cells.py",
    "rfx/geometry/csg.py",
    "rfx/geometry/curved.py",
    "rfx/geometry/curved_body_size.py",
    "rfx/geometry/mesh_import.py",
    "rfx/geometry/node_tensor.py",
    "rfx/geometry/outside_content.py",
    "rfx/geometry/port_termination.py",
    "rfx/geometry/triplet_tensor.py",
    "rfx/geometry/via.py",
    "rfx/grid.py",
    "rfx/harminv.py",
    "rfx/interop/__init__.py",
    "rfx/interop/_design_codec.py",
    "rfx/interop/_design_comparison.py",
    "rfx/interop/_design_dft_planes.py",
    "rfx/interop/_design_state.py",
    "rfx/interop/_errors.py",
    "rfx/interop/_materials.py",
    "rfx/interop/_shapes.py",
    "rfx/interop/_validate.py",
    "rfx/interop/emitters/__init__.py",
    "rfx/interop/emitters/openems.py",
    "rfx/io.py",
    "rfx/jax_checks.py",
    "rfx/lumped.py",
    "rfx/material_fit.py",
    "rfx/materials/__init__.py",
    "rfx/materials/debye.py",
    "rfx/materials/library.py",
    "rfx/materials/lorentz.py",
    "rfx/materials/nonlinear.py",
    "rfx/materials/pole_map.py",
    "rfx/materials/relative_strength.py",
    "rfx/measurement/__init__.py",
    "rfx/measurement/_block_plan.py",
    "rfx/measurement/_radiated_power.py",
    "rfx/measurement/accumulators.py",
    "rfx/measurement/adjoint_targets.py",
    "rfx/measurement/calculator_drive.py",
    "rfx/measurement/coax_port.py",
    "rfx/measurement/coax_records.py",
    "rfx/measurement/coax_voltage.py",
    "rfx/measurement/coverage.py",
    "rfx/measurement/current_moments.py",
    "rfx/measurement/decay.py",
    "rfx/measurement/declared.py",
    "rfx/measurement/dft.py",
    "rfx/measurement/farfield.py",
    "rfx/measurement/floquet.py",
    "rfx/measurement/flux_weights.py",
    "rfx/measurement/fresnel.py",
    "rfx/measurement/graded_port_meta.py",
    "rfx/measurement/graded_result.py",
    "rfx/measurement/graded_wire.py",
    "rfx/measurement/identification.py",
    "rfx/measurement/incident.py",
    "rfx/measurement/matrix_pencil.py",
    "rfx/measurement/modal.py",
    "rfx/measurement/monitors.py",
    "rfx/measurement/msl_planes.py",
    "rfx/measurement/msl_waves.py",
    "rfx/measurement/ntff_surface.py",
    "rfx/measurement/oblique.py",
    "rfx/measurement/phasors.py",
    "rfx/measurement/plan.py",
    "rfx/measurement/plane_records.py",
    "rfx/measurement/port_records.py",
    "rfx/measurement/port_registration.py",
    "rfx/measurement/port_s.py",
    "rfx/measurement/port_s_contract.py",
    "rfx/measurement/port_sampler.py",
    "rfx/measurement/port_time.py",
    "rfx/measurement/ports.py",
    "rfx/measurement/prepared.py",
    "rfx/measurement/projector.py",
    "rfx/measurement/radiated_flux.py",
    "rfx/measurement/rcs.py",
    "rfx/measurement/record_end.py",
    "rfx/measurement/record_readers.py",
    "rfx/measurement/ringdown_blocked.py",
    "rfx/measurement/ringdown_model.py",
    "rfx/measurement/ringdown_port_s.py",
    "rfx/measurement/ringdown_resolution.py",
    "rfx/measurement/ringdown_tail.py",
    "rfx/measurement/ringdown_tail_jax.py",
    "rfx/measurement/ris.py",
    "rfx/measurement/settling.py",
    "rfx/measurement/setup.py",
    "rfx/measurement/sparam_advisories.py",
    "rfx/measurement/sparam_assembly.py",
    "rfx/measurement/subgrid_lumped_dft.py",
    "rfx/measurement/sweep_settling.py",
    "rfx/measurement/tfsf_reach.py",
    "rfx/measurement/time_records.py",
    "rfx/measurement/waveguide_modal.py",
    "rfx/measurement/waveguide_planes.py",
    "rfx/measurement/waveguide_power.py",
    "rfx/measurement/waveguide_records.py",
    "rfx/measurement/wavelet.py",
    "rfx/mesh_edges.py",
    "rfx/mesh_planner.py",
    "rfx/microstrip.py",
    "rfx/model/__init__.py",
    "rfx/model/coax.py",
    "rfx/model/coax_line.py",
    "rfx/model/coax_stamp.py",
    "rfx/model/conformal_rules.py",
    "rfx/model/design_admission.py",
    "rfx/model/design_box.py",
    "rfx/model/dispersive_poles.py",
    "rfx/model/electric_metrics.py",
    "rfx/model/film_geometry.py",
    "rfx/model/graded_admission.py",
    "rfx/model/graded_grid.py",
    "rfx/model/graded_profile.py",
    "rfx/model/line_streams.py",
    "rfx/model/lumped_stamp.py",
    "rfx/model/node_pec.py",
    "rfx/model/occupancy.py",
    "rfx/model/operands.py",
    "rfx/model/overrides.py",
    "rfx/model/pad_fill.py",
    "rfx/model/point_placement.py",
    "rfx/model/pole_stability.py",
    "rfx/model/realization_disjoint.py",
    "rfx/model/realization_legacy.py",
    "rfx/model/realization_placements.py",
    "rfx/model/realization_reference.py",
    "rfx/model/realization_reference_area.py",
    "rfx/model/realization_report.py",
    "rfx/model/realization_transport.py",
    "rfx/model/realization_unavailable.py",
    "rfx/model/realization_verdict.py",
    "rfx/model/smoothing_admission.py",
    "rfx/model/solver_admission.py",
    "rfx/model/source_coefficients.py",
    "rfx/model/subgrid_options.py",
    "rfx/model/tensor_operands.py",
    "rfx/model/thin_conductors.py",
    "rfx/model/waveguide_aperture.py",
    "rfx/model/waveguide_mesh.py",
    "rfx/model/waveguide_mode_solver.py",
    "rfx/model/waveguide_modes.py",
    "rfx/model/waveguide_planes.py",
    "rfx/model/waveguide_waveform.py",
    "rfx/model/wire_mask.py",
    "rfx/nonuniform.py",
    "rfx/ntff_surface.py",
    "rfx/ntff_sweep.py",
    "rfx/observables.py",
    "rfx/optimize_objectives.py",
    "rfx/pareto.py",
    "rfx/pcb.py",
    "rfx/preflight/",
    "rfx/probes/__init__.py",
    "rfx/probes/flux_region.py",
    "rfx/probes/fresnel.py",
    "rfx/probes/msl_wave_decomp.py",
    "rfx/probes/probes.py",
    "rfx/probes/refplane.py",
    "rfx/probes/settling.py",
    "rfx/profiling.py",
    "rfx/rcs.py",
    "rfx/realized_geometry.py",
    "rfx/realized_placement.py",
    "rfx/ringdown/",
    "rfx/runners/__init__.py",
    "rfx/runners/_distributed_advisory.py",
    "rfx/runners/_distributed_common.py",
    "rfx/runners/_distributed_ntff.py",
    "rfx/runners/_exchange_interval.py",
    "rfx/runners/_rank.py",
    "rfx/runners/_smoothing_admission.py",
    "rfx/runners/_uniform_forward_result.py",
    "rfx/runners/distributed.py",
    "rfx/runners/subgridded.py",
    "rfx/runners/uniform_forward.py",
    "rfx/simulation.py",
    "rfx/smith.py",
    "rfx/snapshots.py",
    "rfx/sources/",
    "rfx/sparams/__init__.py",
    "rfx/sparams/_coax_admission.py",
    "rfx/sparams/_coax_messages.py",
    "rfx/sparams/_coax_msl_stages.py",
    "rfx/sparams/_coax_oneport_record.py",
    "rfx/sparams/_coax_status.py",
    "rfx/sparams/_common.py",
    "rfx/sparams/_grid_metrics.py",
    "rfx/sparams/_mixed_checks.py",
    "rfx/sparams/_mixed_stages.py",
    "rfx/sparams/_msl_stages.py",
    "rfx/sparams/_port_reference.py",
    "rfx/sparams/_record_selection.py",
    "rfx/sparams/_settling_groups.py",
    "rfx/sparams/_settling_warning.py",
    "rfx/sparams/_tail_witness.py",
    "rfx/sparams/_waveguide_planes.py",
    "rfx/sparams/_waveguide_stages.py",
    "rfx/sparams/coax_geometry.py",
    "rfx/sparams/dispatch.py",
    "rfx/sparams/passivity_warning.py",
    "rfx/stepping/__init__.py",
    "rfx/stepping/absorber.py",
    "rfx/stepping/adjoint_monitors.py",
    "rfx/stepping/adjoint_scan.py",
    "rfx/stepping/conductors.py",
    "rfx/stepping/floquet.py",
    "rfx/stepping/graded.py",
    "rfx/stepping/graded_precision.py",
    "rfx/stepping/graded_setup.py",
    "rfx/stepping/incident_line.py",
    "rfx/stepping/kerr.py",
    "rfx/stepping/lumped.py",
    "rfx/stepping/poles.py",
    "rfx/stepping/prepared.py",
    "rfx/stepping/probe_loop.py",
    "rfx/stepping/progress.py",
    "rfx/stepping/rank.py",
    "rfx/stepping/record_join.py",
    "rfx/stepping/remat.py",
    "rfx/stepping/ringdown_emulation.py",
    "rfx/stepping/scan.py",
    "rfx/stepping/sequence.py",
    "rfx/stepping/sharded.py",
    "rfx/stepping/slab.py",
    "rfx/stepping/slab_cpml.py",
    "rfx/stepping/slab_exchange.py",
    "rfx/stepping/slab_lumped.py",
    "rfx/stepping/slab_measurements.py",
    "rfx/stepping/slab_ports.py",
    "rfx/stepping/slab_tfsf.py",
    "rfx/stepping/slab_update.py",
    "rfx/stepping/slab_view.py",
    "rfx/stepping/slab_walls.py",
    "rfx/stepping/step_count.py",
    "rfx/stepping/subgrid_coupling.py",
    "rfx/stepping/subgrid_disjoint.py",
    "rfx/stepping/subgrid_loop.py",
    "rfx/stepping/subgrid_sat.py",
    "rfx/stepping/tfsf.py",
    "rfx/stepping/tfsf_2d.py",
    "rfx/stepping/tfsf_oblique.py",
    "rfx/stepping/uniform.py",
    "rfx/stepping/uniform_setup.py",
    "rfx/stepping/uniform_slab_ports.py",
    "rfx/stepping/vector_pad.py",
    "rfx/stepping/vmap_scan.py",
    "rfx/stepping/waveguide_source.py",
    "rfx/stepping/yee.py",
    "rfx/subgridding/__init__.py",
    "rfx/subgridding/_notice.py",
    "rfx/subgridding/disjoint_3d.py",
    "rfx/subgridding/disjoint_runner_contract.py",
    "rfx/subgridding/jit_runner.py",
    "rfx/subgridding/material_sat.py",
    "rfx/subgridding/modal_trace_filter.py",
    "rfx/subgridding/sbp_sat_1d.py",
    "rfx/subgridding/sbp_sat_2d.py",
    "rfx/subgridding/sbp_sat_3d.py",
    "rfx/subgridding/validation.py",
    "rfx/surrogate.py",
    "rfx/sweep.py",
    "rfx/topology.py",
    "rfx/validation.py",
    "rfx/visualize.py",
    "rfx/visualize3d.py",
    "tests/contracts/fixtures/",
    "tests/contracts/path_equivalence/",
    "tests/crossval/msl_notch_filter/",
    "tests/data/",
    "tests/fixtures/",
    "tests/gpu/",
    "tests/locks/",
    "tests/oracle/",
    "tests/unit/api/",
    "tests/unit/autodiff/",
    "tests/unit/boundaries/",
    "tests/unit/cli/",
    "tests/unit/core/",
    "tests/contracts/test_public_docs_block_runner.py",
    "tests/contracts/test_public_docs_bundle.py",
    "tests/unit/farfield/",
    "tests/unit/geometry/",
    "tests/unit/grid/",
    "tests/unit/measurement/",
    "tests/unit/misc/",
    "tests/unit/model/",
    "tests/unit/nonuniform/",
    "tests/unit/ports/",
    "tests/unit/sources/",
    "tests/unit/sparams/_results_931/",
)
SWEPT_DESTINATION = "a name, an assertion, a test, the docstring, or a dated record"


class Refusal(NamedTuple):
    """One refused comment token, keyed by its path, line and column."""

    path: str
    line: int
    column: int
    text: str


class CheckError(Exception):
    """The check could not inspect the requested changes."""


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="surrogateescape",
    )
    if check and result.returncode:
        raise CheckError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def ensure_commit(root: Path, revision: str) -> str:
    result = git(root, "rev-parse", "--verify", f"{revision}^{{commit}}", check=False)
    if result.returncode:
        git(root, "fetch", "--no-tags", "--depth=1", "origin", revision)
        result = git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    return result.stdout.strip()


def resolve_base(root: Path, explicit: str | None) -> str:
    if explicit:
        revision = ensure_commit(root, explicit)
        return git(root, "merge-base", revision, "HEAD").stdout.strip()
    header = git(root, "cat-file", "-p", "HEAD").stdout.split("\n\n", 1)[0]
    parents = [line.split()[1] for line in header.splitlines() if line.startswith("parent ")]
    if os.environ.get("GITHUB_EVENT_NAME") in {"pull_request", "pull_request_target", "merge_group"}:
        if not parents:
            raise CheckError("HEAD has no parent and no base could be resolved")
        return ensure_commit(root, parents[0])
    origin = git(root, "rev-parse", "--verify", "origin/main^{commit}", check=False)
    if origin.returncode == 0:
        base = git(root, "merge-base", "origin/main", "HEAD").stdout.strip()
        if base != git(root, "rev-parse", "HEAD").stdout.strip():
            return base
    if parents:
        return ensure_commit(root, parents[0])
    raise CheckError("HEAD has no parent and no base could be resolved")


def added_lines(root: Path, base: str) -> dict[str, set[int]]:
    patch = git(
        root, "diff", "--no-color", "--unified=0", "-M", "--diff-filter=AMR",
        base, "--", "*.py",
    ).stdout
    changed: dict[str, set[int]] = {}
    path = None
    line_number = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            path = None
            line_number = None
        elif line.startswith("+++ ") and line_number is None:
            name = line[4:]
            if name.startswith('"'):
                name = ast.literal_eval(name)
            else:
                name = name.removesuffix("\t")
            path = name.removeprefix("b/")
            changed.setdefault(path, set())
        elif match := HUNK.match(line):
            line_number = int(match.group(1))
        elif line_number is not None and path is not None:
            if line.startswith("+"):
                changed.setdefault(path, set()).add(line_number)
                line_number += 1
            elif line.startswith(" "):
                line_number += 1
    return changed


def in_full_ban(path: str) -> bool:
    """Whether a repository-relative path, in exact case, lies where a comment may only be a tool directive."""
    return path.startswith(FULL_BAN_PREFIXES)


def is_directive(token: tokenize.TokenInfo) -> bool:
    """Whether a whole comment token is tool directives and nothing else.

    A shebang counts at column 0 of line 1. A PEP 263 coding declaration counts
    on line 1 or 2 when the comment is the whole physical line and Python's own
    search hits it. Otherwise the token must be ``#`` and one or more
    directives, each later one after whitespace and another ``#``, with only
    whitespace after the last; a quoted mypy value is a comma-separated list of
    names.
    """
    line, column = token.start
    if line == 1 and column == 0 and token.string.startswith("#!"):
        return True
    whole_line = not token.line[:column].strip(" \t\f")
    if line <= 2 and whole_line and CODING_LINE.search(token.string):
        return True
    return DIRECTIVE_TOKEN.fullmatch(token.string) is not None


def narrow_refusal(path: str, line: int, token: str) -> str | None:
    """The diagnostic of the first narrow rule the comment token matches, or None."""
    for rule, pattern, destination in RULES:
        if pattern.search(token):
            return f"{path}:{line}: {rule}: {token}  -> {destination}; {GUIDANCE}"
    return None


def comment_tokens(root: Path, path: str) -> Iterator[tokenize.TokenInfo]:
    """Every COMMENT token of a working-tree file, or CheckError when it cannot be tokenized."""
    try:
        with (root / path).open("rb") as source:
            for token in tokenize.tokenize(source.readline):
                if token.type == tokenize.ERRORTOKEN and not token.string.isspace():
                    raise CheckError(f"{path}: cannot tokenize {token.string!r} at line {token.start[0]}")
                if token.type == tokenize.COMMENT:
                    yield token
    except (OSError, SyntaxError, tokenize.TokenError, UnicodeError) as exc:
        raise CheckError(f"{path}: cannot tokenize: {exc}") from exc


def check_comments(root: Path, changed: dict[str, set[int]]) -> list[Refusal]:
    """Refusals for comment tokens on added lines: the full ban under the four directories, the narrow rules elsewhere."""
    refused = []
    for path, lines in changed.items():
        full_ban = in_full_ban(path)
        for token in comment_tokens(root, path):
            line, column = token.start
            if line not in lines:
                continue
            if full_ban:
                if not is_directive(token):
                    text = (
                        narrow_refusal(path, line, token.string)
                        or f"{path}:{line}: comment: {token.string}  -> {FULL_BAN_DESTINATION}"
                    )
                    refused.append(Refusal(path, line, column, text))
            elif diagnostic := narrow_refusal(path, line, token.string):
                refused.append(Refusal(path, line, column, diagnostic))
    return refused


def malformed_entry(entry: str) -> str | None:
    """Why a swept entry is not a well-formed repository-relative path, or None when it is."""
    if ".." in entry:
        return f"{entry}: swept entry must not contain .."
    if not entry.startswith(FULL_BAN_PREFIXES):
        return f"{entry}: swept entry must lie under rfx/, tests/, examples/ or scripts/"
    if not entry.endswith(("/", ".py")):
        return f"{entry}: swept entry must end with / (a package) or .py (one file)"
    return None


def check_swept(
    root: Path, entries: tuple[str, ...], reported: set[tuple[str, int, int]],
) -> tuple[list[str], list[Refusal], int]:
    """Entry refusals, comment refusals and the number of files read for the swept entries.

    Each tracked ``*.py`` file present in the working tree and matched by a
    well-formed entry (``path.startswith(entry)``) is read whole; a file matched
    by several entries is read once and named by the first. A malformed entry is
    refused and not applied. A token already refused by the added-line check,
    keyed by path, line and column, is not reported again.
    """
    if not entries:
        return [], [], 0
    tracked = [
        path for path in git(root, "ls-files", "-z").stdout.split("\0")
        if path.endswith(".py") and (root / path).is_file()
    ]
    entry_refusals = []
    owner: dict[str, str] = {}
    for entry in entries:
        reason = malformed_entry(entry)
        if reason is None:
            matched = [path for path in tracked if path.startswith(entry)]
            if not matched:
                reason = f"{entry}: swept entry matches no file; name the file's new path or remove the entry"
            for path in matched:
                owner.setdefault(path, entry)
        if reason is not None:
            entry_refusals.append(reason)
    refused = []
    for path in sorted(owner):
        for token in comment_tokens(root, path):
            line, column = token.start
            if (path, line, column) in reported or is_directive(token):
                continue
            refused.append(Refusal(path, line, column, (
                f"{path}:{line}: swept: {token.string}  -> {SWEPT_DESTINATION}; "
                f"{owner[path]} is swept, so every comment in it may only be a tool directive"
            )))
    return entry_refusals, refused, len(owner)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("BASE_SHA"))
    parser.add_argument("--swept", nargs="*", metavar="ENTRY",
                        help="swept entries for this run in place of SWEPT_PREFIXES")
    args = parser.parse_args(argv)
    entries = SWEPT_PREFIXES if args.swept is None else tuple(args.swept)
    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.strip())
        base = resolve_base(root, args.base)
        changed = added_lines(root, base)
        refused = check_comments(root, changed)
        entry_refusals, swept, files = check_swept(
            root, entries, {(refusal.path, refusal.line, refusal.column) for refusal in refused})
        short = git(root, "rev-parse", "--short", base).stdout.strip()
    except (CheckError, OSError) as exc:
        print(f"comment content: {exc}", file=sys.stderr)
        return 2
    for refusal in refused:
        print(refusal.text)
    for diagnostic in entry_refusals:
        print(diagnostic)
    for refusal in swept:
        print(refusal.text)
    print(
        f"comment content: {len(refused)} refused on {sum(map(len, changed.values()))}"
        f" added lines (base {short})"
    )
    if entries:
        print(f"swept: {len(entry_refusals) + len(swept)} refused in {files} files ({len(entries)} entries)")
    return 1 if refused or entry_refusals or swept else 0


if __name__ == "__main__":
    sys.exit(main())
