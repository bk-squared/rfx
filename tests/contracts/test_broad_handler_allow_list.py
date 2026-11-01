"""Pin every remaining broad handler; B1b/B1c shrink this explicit inventory."""
import ast
from collections import Counter
from pathlib import Path

import pytest


# One row per handler, including repeated handlers in the same function.
# Reasons describe the accepted debt, not an exemption for future handlers.
ALLOW_LIST = (
    ('rfx/ringdown/_jax.py', 'host_poles', 'pending B1c: recorded diagnostic'),
    ('rfx/api/_ad_carry.py', 'read_carry', 're-raises with context'),
    ('rfx/api/_ad_carry.py', 'linear_accumulators', 'pending B1c: recorded diagnostic'),
    ('rfx/api/_ad_memory.py', '_ad_memory_static_accounting', 'pending B1b: grid/check fallback'),
    ('rfx/api/_ad_memory.py', 'estimate_ad_memory', 'pending B1c: environment probe'),
    ('rfx/api/_ad_memory.py', 'estimate_ad_memory', 'pending B1c: environment probe'),
    ('rfx/api/_ad_memory_compiled.py', '_resident_bytes', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_read_attr', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_environment_summary', 'pending B1c: environment probe'),
    ('rfx/api/_compiled_memory.py', '_environment_summary', 'pending B1c: environment probe'),
    ('rfx/api/_compiled_memory.py', '_environment_summary', 'pending B1c: environment probe'),
    ('rfx/api/_compiled_memory.py', '_environment_summary', 'pending B1c: environment probe'),
    ('rfx/api/_compiled_memory.py', '_normalize_compiled_memory_analysis', 'pending B1c: environment probe'),
    ('rfx/api/_compiled_memory.py', '_normalize_compiled_memory_analysis', 'pending B1c: environment probe'),
    ('rfx/api/_compiled_memory.py', '_json_safe', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_json_safe', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_json_safe', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_json_safe', 're-raises with context'),
    ('rfx/api/_compiled_memory.py', '_normalize_compiled_memory_analysis', 'pending B1c: environment probe'),
    ('rfx/api/_compiled_memory.py', '_compiled_introspection_scope', 'pending B1c: environment probe'),
    ('rfx/api/_validators.py', '_preflight_json_safe', 're-raises with context'),
    ('rfx/artifacts.py', '_scalar', 'pending B1c: serialization'),
    ('rfx/artifacts.py', '_safe_float', 'pending B1c: serialization'),
    ('rfx/artifacts.py', '_sequence_summary', 'pending B1c: serialization'),
    ('rfx/artifacts.py', '_profile_summary', 'pending B1c: serialization'),
    ('rfx/artifacts.py', '_bbox', 'pending B1c: recorded diagnostic'),
    ('rfx/artifacts.py', '_mesh_summary', 'pending B1c: recorded diagnostic'),
    ('rfx/artifacts.py', '_preflight_summary', 'pending B1c: recorded diagnostic'),
    ('rfx/artifacts.py', '_freq_summary', 'pending B1c: serialization'),
    ('rfx/artifacts.py', '_git', 'pending B1c: environment probe'),
    ('rfx/artifacts.py', '_jsonable', 'pending B1c: serialization'),
    ('rfx/artifacts.py', '_jsonable', 'pending B1c: serialization'),
    ('rfx/artifacts.py', '_array_summary', 'pending B1c: recorded diagnostic'),
    ('rfx/artifacts.py', '_grid_shape', 'pending B1c: serialization'),
    ('rfx/artifacts.py', '_mesh_summary', 'pending B1c: recorded diagnostic'),
    ('rfx/artifacts.py', '_jsonable', 'pending B1c: serialization'),
    ('rfx/batch.py', 'run_batch_with_manifest', 're-raises with context'),
    ('rfx/boundaries/tfsf.py', 'invariant', 'pending B1c: recorded diagnostic'),
    ('rfx/core/jax_utils.py', 'compile_time_switch', 'pending B1c: environment probe'),
    ('rfx/core/jax_utils.py', 'compile_time_switch', 'pending B1c: environment probe'),
    ('rfx/dashboard/app.py', '<module>', 'pending B1c: recorded diagnostic'),
    ('rfx/diagnostics.py', '_version_of', 'pending B1c: environment probe'),
    ('rfx/diagnostics.py', '_check_rfx_import', 'pending B1c: recorded diagnostic'),
    ('rfx/diagnostics.py', '_check_jax', 'pending B1c: recorded diagnostic'),
    ('rfx/diagnostics.py', '_check_jax', 'pending B1c: recorded diagnostic'),
    ('rfx/diagnostics.py', '_check_jax', 'pending B1c: recorded diagnostic'),
    ('rfx/diagnostics.py', '_check_smoke', 'pending B1c: recorded diagnostic'),
    ('rfx/diagnostics.py', '_check_core_deps', 'pending B1c: recorded diagnostic'),
    ('rfx/diagnostics.py', '_check_optional_extras', 'pending B1c: recorded diagnostic'),
    ('rfx/fidelity.py', 'fidelity_report', 're-raises with context'),
    ('rfx/fidelity.py', 'fidelity_report', 'pending B1c: recorded diagnostic'),
    ('rfx/geometry/smoothing.py', 'compute_smoothed_eps_nonuniform', 're-raises with context'),
    ('rfx/measurement/decay.py', 'dc_census', 'open user waveform; recorded in DCNotEvaluated.error; timeout propagates'),
    ('rfx/measurement/msl_planes.py', 'readable_values', 'pending B1c: serialization'),
    ('rfx/model/conductors.py', 'coax_kernel_conductors', 'pending B1c: recorded diagnostic'),
    ('rfx/preflight/absorber.py', '_validate_cfg_lossless_resonator_in_absorber', 'pending B1c: environment probe'),
    ('rfx/preflight/line_stub.py', 'read_band', 're-raises with context'),
    ('rfx/preflight/msl.py', 'preflight_msl_probe_clearance', 'pending B1b: grid/check fallback'),
    ('rfx/preflight/msl.py', '_check_msl_port_geometry', 'pending B1b: grid/check fallback'),
    ('rfx/preflight/msl.py', 'preflight_msl_probe_clearance', 'pending B1b: per-entry check'),
    ('rfx/preflight/msl_reflector.py', 'msl_nearest_downstream_reflector', 'pending B1b: per-entry check'),
    ('rfx/preflight/msl_reflector.py', 'msl_nearest_downstream_reflector', 'pending B1b: per-entry check'),
    ('rfx/preflight/ntff.py', '_validate_ntff_inverse_design', 'pending B1c: environment probe'),
    ('rfx/preflight/waveguide.py', '_waveguide_setup_planes', 'pending B1b: grid/check fallback'),
    ('rfx/sparams/waveguide.py', 'compute_waveguide_s_matrix', 'recorded in the UNCHECKED AccuracyWarning'),
    ('rfx/preflight/waveguide.py', '_waveguide_port_cutoff_reading', 'pending B1c: recorded diagnostic'),
    ('rfx/profiling.py', 'profile_forward', 'pending B1b: grid/check fallback'),
    ('rfx/realized_geometry.py', '_build_record', 'pending B1c: recorded diagnostic'),
    ('rfx/realized_placement.py', '_tfsf_row', 'pending B1c: recorded diagnostic'),
    ('rfx/realized_placement.py', '_flux_row', 'pending B1c: recorded diagnostic'),
    ('rfx/realized_placement.py', '_ntff_row', 'pending B1c: recorded diagnostic'),
    ('rfx/realized_placement.py', 'cells_per_wavelength', 'pending B1c: recorded diagnostic'),
    ('rfx/realized_placement.py', '_sources', 'pending B1c: recorded diagnostic'),
    ('rfx/realized_placement.py', '_probes', 'pending B1c: recorded diagnostic'),
    ('rfx/realized_placement.py', '_probes', 'pending B1c: recorded diagnostic'),
    ('rfx/ringdown/__init__.py', '_judge_record', 'pending B1c: recorded diagnostic'),
    ('rfx/ringdown/__init__.py', '_error_witnesses', 'pending B1c: recorded diagnostic'),
    ('rfx/sources/_laplace.py', '_solve_laplace_2d', 'pending B1c: environment probe'),
    ('rfx/stepping/rank.py', '_staged_by_enclosing_program', 'pending B1c: environment probe'),
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
