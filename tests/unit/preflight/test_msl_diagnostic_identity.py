"""J-D: same family observations across paths, and no local message bypass."""
import json
import os
from pathlib import Path
import subprocess
import sys
from string import Formatter
import warnings

import pytest

from tests._msl_diagnostic_cases import captured_cases, structure


ROOT = Path(__file__).resolve().parents[3]
EXPECTED = {
    'msl.lateral_clearance', 'msl.normal_resolution',
    'msl.source_absorber_clearance', 'msl.source_near_field',
}


EXPECTED_LEGACY = {
    'msl.' + name: 'msl_port_geometry' for name in (
        'probe_clearance_scan_failed', 'probe_clearance_unavailable',
        'reflector_clearance', 'probe_placement_note', 'probe_placement_failed',
        'lateral_clearance', 'normal_resolution', 'trace_face_alignment',
        'conductor_gap_mismatch', 'source_absorber_clearance', 'probe_ladder_clamped',
        'reflector_scan_unavailable', 'reflector_scan_incomplete',
        'probe_in_absorber', 'probe_near_absorber', 'probe_crosses_feed', 'source_near_field',
    )
}
EXPECTED_LEGACY.update({
    'msl.conductor_assembly_unavailable': 'msl_port_conductor_planes',
    'msl.conductor_attachment': 'msl_port_conductor_planes',
    'msl.line_stub_realization': 'line_stub_realization',
    'msl.line_stub_inspection_unavailable': 'line_stub_inspection_unavailable',
    'msl.line_stub_behind_port': 'line_stub_behind_port',
})
EXPECTED_REFUSALS = {
    'msl.probe_placement_failed', 'msl.conductor_assembly_unavailable',
    'msl.conductor_attachment', 'msl.line_stub_realization',
}


def path_observations():
    import jax
    records = {}
    for path in ('uniform', 'graded', 'distributed', 'distributed_nu'):
        sim = structure(path in ('graded', 'distributed_nu'))
        try:
            result = sim.run(
                n_steps=2,
                compute_s_params=False,
                **({"devices": jax.devices()} if path.startswith("distributed") else {}),
            )
            status = 'returns'
        except (ValueError, NotImplementedError) as error:
            result, status = error, 'refuses'
            assert any(d.severity == 'refusal' for d in error.diagnostics)
        records[path] = {'status': status, 'diagnostics': [d.to_dict() for d in result.diagnostics]}
    return records


def test_jd_path_identity():
    # The second CPU is needed to reach the real multi-device admission fence.
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        PYTHONPATH=str(ROOT),
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    code = ('import json; from tests.unit.preflight.test_msl_diagnostic_identity import path_observations; '
            'print("JD_RESULT=" + json.dumps(path_observations()))')
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    observations = json.loads(next(line.removeprefix('JD_RESULT=')
                                   for line in process.stdout.splitlines() if line.startswith('JD_RESULT=')))
    signatures = {}
    for path, result in observations.items():
        family = [d for d in result['diagnostics'] if d['code'].startswith('msl.')]
        assert {d['code'] for d in family} == EXPECTED
        assert all(d['path'] is None for d in family)  # Common checks precede dispatch.
        signatures[path] = sorted((d['code'], d['subject'], d['message']) for d in family)
        assert result['status'] == 'returns'
    # This aligned, constant-profile structure needs none of the listed wording variants.
    # A path-specific wording mutation must fail here even if its code survives.
    assert all(signature == signatures['uniform'] for signature in signatures.values())


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_jd_emissions_are_silent_and_catalogued(graded, capsys):
    from rfx.preflight.msl_codes import CODES, TEMPLATES
    from tests._msl_diagnostic_golden import GOLDEN

    lane = "graded" if graded else "uniform"
    # These are the only opaque messages: inspection exceptions or forwarded
    # placement advice. Read the catalog, never the implementation's source.
    opaque = {
        'probe_placement_note': 'msl.probe_placement_note',
        'line_stub_realization': 'msl.line_stub_realization',
        'probe_placement_failed': 'msl.probe_placement_failed',
        'probe_clearance_scan_failed': 'msl.probe_clearance_scan_failed',
        'conductor_attachment': 'msl.conductor_attachment',
    }
    opaque_templates = {
        key for key, template in TEMPLATES.items()
        if {field for _, field, _, _ in Formatter().parse(template)} & {'detail', 'exc'}
    }
    assert opaque_templates == set(opaque), "opaque template allowlist changed"
    for code, entry in CODES.items():
        for key in set(entry.templates) & opaque_templates:
            assert code == opaque[key], f"{code}: opaque template {key} belongs to {opaque[key]}"

    expected = {key: texts for key, texts in GOLDEN.items() if key.startswith(lane + "/")}
    golden_texts = {text for texts in expected.values() for text in texts}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            observations = captured_cases(graded)
        except Exception as error:
            diagnostic = getattr(error, "diagnostic", None)
            assert diagnostic is not None, (
                f"{lane}: {type(error).__name__} lacks .diagnostic: {error}"
            )
            assert diagnostic.code in CODES, f"{lane}: unknown exception code {diagnostic.code}"
            assert diagnostic.message == str(error) and str(error) in golden_texts, (
                f"{lane}: exception differs from golden text: {error}"
            )
            raise  # These cases return findings; an unexpected refusal is a finding too.

    stdout = capsys.readouterr().out
    print(f"{lane}: stdout_chars={len(stdout)} bar=0", file=sys.stderr)
    assert stdout == "", f"{lane}: MSL diagnostics wrote {len(stdout)} characters to stdout: {stdout!r}"

    # The helper forwards uncatalogued warnings to this outer capture, keeping
    # WarningMessage's filename metadata (no source reads). Construction/grid
    # warnings also arrive here; unrelated material advisories are outside MSL.
    outer_family = []
    for warning in caught:
        message = warning.message
        diagnostic = getattr(message, "diagnostic", None)
        path = Path(warning.filename)
        family_file = path.parent.name == "preflight" and (
            path.name.startswith("msl") or path.name in {"line_stub.py", "line_port_coverage.py"}
        )
        if family_file or getattr(diagnostic, "code", "").startswith("msl."):
            assert diagnostic is not None, f"{lane}: {warning.filename}: {message} lacks .diagnostic"
            assert diagnostic.code in CODES, f"{lane}: unknown warning code {diagnostic.code}"
            assert diagnostic.message == str(message) and str(message) in golden_texts, (
                f"{lane}: outer warning differs from golden text: {message}"
            )
            outer_family.append(message)

    assert {lane + "/" + case for case in observations} == set(expected), f"{lane}: missing diagnostic cases"
    found = set()
    for case, messages in observations.items():
        row = lane + "/" + case
        diagnostics = [getattr(message, "diagnostic", None) for message in messages]
        missing = sum(diagnostic is None for diagnostic in diagnostics)
        unknown = sum(diagnostic is not None and diagnostic.code not in CODES for diagnostic in diagnostics)
        actual = list(map(str, messages))
        mismatches = sum(a != b for a, b in zip(actual, expected[row])) + abs(len(actual) - len(expected[row]))
        record_mismatches = sum(
            diagnostic is not None and diagnostic.message != str(message)
            for message, diagnostic in zip(messages, diagnostics)
        )
        print(
            f"{row}: messages={len(messages)} missing_diagnostic={missing} unknown_code={unknown} "
            f"golden_mismatches={mismatches} record_mismatches={record_mismatches} bar=0",
            file=sys.stderr,
        )
        assert missing == 0, f"{row}: {missing} emissions lack .diagnostic: {actual}"
        assert unknown == 0, f"{row}: unknown diagnostic codes: {diagnostics}"
        assert actual == expected[row], f"{row}: emitted text differs from golden"
        assert record_mismatches == 0, f"{row}: structured message differs from emitted text"
        found.update(diagnostic.code for diagnostic in diagnostics)
    print(
        f"{lane}: codes_reached={len(found)} expected_codes={len(CODES)} "
        f"outer_family_warnings={len(outer_family)} opaque_templates={len(opaque_templates)}",
        file=sys.stderr,
    )
    assert found == set(CODES), f"{lane}: unreached codes={set(CODES) - found}"


@pytest.mark.parametrize(
    "graded",
    [False, True],
    ids=["uniform", "graded"],
)
def test_byte_identity_and_every_code_reached(graded):
    from rfx.preflight.msl_codes import CODES
    assert {code: entry.legacy_slug for code, entry in CODES.items()} == EXPECTED_LEGACY
    assert {code for code, entry in CODES.items() if entry.severity == 'refusal'} == EXPECTED_REFUSALS
    from tests._msl_diagnostic_golden import GOLDEN as expected
    found = set()
    for case, messages in captured_cases(graded).items():
        key = ('graded' if graded else 'uniform') + '/' + case
        assert list(map(str, messages)) == expected[key], key
        for message in messages:
            found.add(message.diagnostic.code)
            assert message.diagnostic.message == str(message)
            assert EXPECTED_LEGACY[message.diagnostic.code] == message.code
            assert message.diagnostic.severity == (
                "refusal" if message.severity == "error" else "advisory")
    assert found == set(EXPECTED_LEGACY)


def test_two_simulations_alternating_do_not_retain_findings():
    first, second = (
        structure(
            name="first",
        ),
        structure(
            name="second",
        ),
    )
    for sim, name in [(first, 'first'), (second, 'second'), (first, 'first')]:
        result = sim.run(
            n_steps=2,
            compute_s_params=False,
        )
        family = [d for d in result.diagnostics if d.code.startswith('msl.')]
        assert {d.code for d in family} == EXPECTED
        assert {d.subject for d in family} == {name}
        clean = sim.run(
            n_steps=2,
            compute_s_params=False,
            skip_preflight=True,
        )
        assert clean.diagnostics == ()
