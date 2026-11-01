"""Pin every remaining broad handler; each row's reason names where the failure is recorded."""
import ast
from collections import Counter
from pathlib import Path

import pytest


# One row per handler, including repeated handlers in the same function.
# Reasons describe the accepted debt, not an exemption for future handlers.
ALLOW_LIST = (
    ('rfx/ringdown/_jax.py', 'host_poles', 'recorded in status (STATUS_FAILED)'),
    ('rfx/api/_ad_carry.py', 'read_carry', 're-raises with context'),
    ('rfx/api/_ad_carry.py', 'linear_accumulators', 'recorded in warnings'),
    ('rfx/api/_ad_memory.py', '_ad_memory_static_accounting', 'recorded in grid_source'),
    ('rfx/api/_ad_memory.py', 'estimate_ad_memory', 'recorded in warning'),
    ('rfx/api/_ad_memory.py', 'estimate_ad_memory', 'recorded in warning'),
    ('rfx/api/_ad_memory_compiled.py', '_resident_bytes', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_read_attr', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_environment_summary', 'recorded in scope_status_reason'),
    ('rfx/api/_compiled_memory.py', '_environment_summary', 'recorded in scope_status_reason'),
    ('rfx/api/_compiled_memory.py', '_environment_summary', 'recorded in scope_status_reason'),
    ('rfx/api/_compiled_memory.py', '_environment_summary', 'recorded in scope_status_reason'),
    ('rfx/api/_compiled_memory.py', '_normalize_compiled_memory_analysis', 'recorded in memory_analysis_status_reason'),
    ('rfx/api/_compiled_memory.py', '_normalize_compiled_memory_analysis', 'recorded in memory_analysis_status_reason'),
    ('rfx/api/_compiled_memory.py', '_json_safe', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_json_safe', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_json_safe', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_json_safe', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_normalize_compiled_memory_analysis', 'recorded in memory_analysis_status_reason'),
    ('rfx/api/_compiled_memory.py', '_compiled_introspection_scope', 'recorded in scope_status_reason'),
    ('rfx/api/_validators.py', '_preflight_json_safe', 're-raises with context'),
    ('rfx/artifacts.py', '_scalar', 'type dispatch: falls through to _array_summary'),
    ('rfx/artifacts.py', '_bbox', 'recorded in bounding_box'),
    ('rfx/artifacts.py', '_mesh_summary', 'recorded in mesh.error'),
    ('rfx/artifacts.py', '_preflight_summary', 'recorded in issues'),
    ('rfx/artifacts.py', '_jsonable', 'recorded in error'),
    ('rfx/artifacts.py', '_jsonable', 'recorded in error'),
    ('rfx/artifacts.py', '_array_summary', 'recorded in peak_error'),
    ('rfx/artifacts.py', '_mesh_summary', 'recorded in mesh.audit.error'),
    ('rfx/batch.py', 'run_batch_with_manifest', 're-raises with context'),
    ('rfx/boundaries/tfsf.py', 'invariant', 'recorded in invariance reason'),
    ('rfx/core/jax_utils.py', 'compile_time_switch', 'recorded in warning message'),
    ('rfx/dashboard/app.py', '<module>', 'recorded in Simulation failed message / Show full traceback'),
    ('rfx/diagnostics.py', '_check_rfx_import', 'recorded in report.lines'),
    ('rfx/diagnostics.py', '_check_jax', 'recorded in report.lines'),
    ('rfx/diagnostics.py', '_check_jax', 'recorded in report.lines'),
    ('rfx/diagnostics.py', '_check_jax', 'recorded in report.lines'),
    ('rfx/diagnostics.py', '_check_smoke', 'recorded in report.lines'),
    ('rfx/diagnostics.py', '_check_core_deps', 'recorded in report.lines'),
    ('rfx/diagnostics.py', '_check_optional_extras', 'recorded in report.lines'),
    ('rfx/fidelity.py', 'fidelity_report', 're-raises with context'),
    ('rfx/fidelity.py', 'fidelity_report', 'recorded in findings.exception / findings.detail'),
    ('rfx/geometry/smoothing.py', 'compute_smoothed_eps_nonuniform', 're-raises with context'),
    ('rfx/measurement/decay.py', 'dc_census', 'open user waveform; recorded in DCNotEvaluated.error; timeout propagates'),
    ('rfx/model/conductors.py', 'coax_kernel_conductors', 'recorded in stamp_check_findings'),
    ('rfx/preflight/line_stub.py', 'read_band', 're-raises with context'),
    ('rfx/preflight/msl.py', 'preflight_msl_probe_clearance', 'recorded in MSLProbeClearance.status/note'),
    ('rfx/preflight/msl.py', '_check_msl_port_geometry', 'recorded in MSLProbeClearance.status/note'),
    ('rfx/preflight/msl.py', 'preflight_msl_probe_clearance', 'recorded in msl.probe_clearance_scan_failed'),
    ('rfx/preflight/msl_reflector.py', 'msl_nearest_downstream_reflector', 'recorded in unevaluated'),
    ('rfx/preflight/msl_reflector.py', 'msl_nearest_downstream_reflector', 'recorded in unevaluated'),
    ('rfx/preflight/waveguide.py', '_waveguide_setup_planes', 'recorded in waveguide_setup_audit_skipped'),
    ('rfx/sparams/waveguide.py', 'compute_waveguide_s_matrix', 'recorded in the UNCHECKED AccuracyWarning'),
    ('rfx/preflight/waveguide.py', '_waveguide_port_cutoff_reading', 'recorded in build_error'),
    ('rfx/profiling.py', 'profile_forward', 'recorded in cells'),
    ('rfx/realized_geometry.py', '_build_record', 'recorded in mask_error'),
    ('rfx/realized_placement.py', '_tfsf_row', 'recorded in SourceGeometry.error'),
    ('rfx/realized_placement.py', '_flux_row', 'recorded in ProbeGeometry.error'),
    ('rfx/realized_placement.py', '_ntff_row', 'recorded in ProbeGeometry.error'),
    ('rfx/realized_placement.py', 'cells_per_wavelength', 'recorded in CellsPerWavelength.error'),
    ('rfx/realized_placement.py', '_sources', 'recorded in SourceGeometry.error'),
    ('rfx/realized_placement.py', '_probes', 'recorded in ProbeGeometry.error'),
    ('rfx/realized_placement.py', '_probes', 'recorded in ProbeGeometry.error'),
    ('rfx/ringdown/__init__.py', '_judge_record', 'recorded in note'),
    ('rfx/ringdown/__init__.py', '_error_witnesses', 'recorded in RingdownWitness.note'),
    ('rfx/surrogate.py', 'export_geometry_sdf', 're-raises with context'),
)


@pytest.mark.source_scan
def test_broad_handlers_match_allow_list():
    root = Path(__file__).resolve().parents[2]
    actual = Counter()
    sites = []
    for path in sorted((root / "rfx").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        owners = {}
        for function in ast.walk(tree):
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(function):
                    owners.setdefault(id(node), function.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            if node.type is not None and not any(
                (isinstance(t, ast.Name) and t.id in {"Exception", "BaseException"})
                or (isinstance(t, ast.Attribute) and t.attr in {"Exception", "BaseException"})
                for t in types
            ):
                continue
            key = (path.relative_to(root).as_posix(), owners.get(id(node), "<module>"))
            actual[key] += 1
            sites.append(f"{key[0]}:{node.lineno} {key[1]}")
    assert all(reason.strip() for _, _, reason in ALLOW_LIST)
    expected = Counter((path, function) for path, function, _ in ALLOW_LIST)
    assert actual == expected, (
        f"Unlisted broad handlers: {actual - expected}\n"
        f"Remove obsolete allowances: {expected - actual}\n"
        + "\n".join(sites)
    )
