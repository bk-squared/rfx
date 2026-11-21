"""Twenty opt-in mutants, each in a fresh CPU process with a named assertion witness."""
from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys


PT = "rfx.geometry.port_termination"
SM = "rfx.geometry.smoothing"
LS = "rfx.preflight.line_stub"
AP = "rfx.preflight.absorber"
TARGET = "tests/contracts/test_msl_continuation_1512.py"
NO_CONTINUATION = (SM, '    if (getattr(sim, "_boundary", None) not in ("cpml", "upml")\n            or int(getattr(sim, "_cpml_layers", 0)) <= 0):\n        return shape',
                   '    if True:\n        return shape')
MSL_HOLD = (PT, '    if collection == "_msl_ports":\n        return ()',
            '    if collection == "_msl_ports":\n        return msl_port_aperture_conductors(sim, grid, port) if port.terminates is None else ()')
NO_RESONANCE = (LS, '            if resonant_odd_orders(finding, band) is not None:',
                '            if False:')
NO_FINDINGS = (LS, '    if not (sim._msl_ports or sim._coaxial_ports):\n        return []',
               '    if True:\n        return []')
ROWS = (
    ("overhang_no_continuation", "test_overhang_continues", NO_CONTINUATION),
    ("overhang_msl_hold", "test_overhang_continues", MSL_HOLD),
    ("gap_no_resonance", "test_interior_gap_refused", NO_RESONANCE),
    ("gap_no_findings", "test_interior_gap_refused", NO_FINDINGS),
    ("reach_no_continuation", "test_realized_reach_boundary", NO_CONTINUATION),
    ("reach_no_resonance", "test_realized_reach_boundary", NO_RESONANCE),
    ("unsupported_no_resonance", "test_unextendable_refused", NO_RESONANCE),
    ("unsupported_no_warning", "test_unextendable_refused",
     (SM, '    conductors = [u for u in unextendable if u.conductor]', '    conductors = []')),
    ("mixed_no_override", "test_mixed_hold_precedence",
     (SM, '        held -= msl_continuation_faces(sim, grid, entry)', '        held -= set()')),
    ("mixed_wrong_face", "test_mixed_hold_precedence",
     (PT, '"lo" if port.direction.startswith("+") else "hi"',
      '"hi" if port.direction.startswith("+") else "lo"')),
    ("registration_accept", "test_registration_refusal",
     (PT, '    if value is None:\n        return None', '    if True:\n        return None')),
    ("registration_wrong_type", "test_registration_refusal",
     (PT, '    raise ValueError(\n        f"{port}: terminates= is not accepted',
      '    raise TypeError(\n        f"{port}: terminates= is not accepted')),
    ("empty_msl_hold", "test_empty_realizations_identical", MSL_HOLD),
    ("empty_no_continuation", "test_empty_realizations_identical", NO_CONTINUATION),
    ("interop_resolves_first", "test_interop_refusal",
     ("rfx.interop._design", '            sim, values["terminates"], port=f"add_msl_port at {values[\'position\']}")',
      '            sim, _termination_inputs(sim, values["terminates"]), port=f"add_msl_port at {values[\'position\']}")')),
    ("interop_wrong_type", "test_interop_refusal",
     (PT, '    raise ValueError(\n        f"{port}: terminates= is not accepted',
      '    raise TypeError(\n        f"{port}: terminates= is not accepted')),
    ("warning_msl_advice", "test_port_continues_warning",
     (AP, 'if collection == "_msl_ports" or port_termination_references(self, collection, port, grid):',
      'if port_termination_references(self, collection, port, grid):')),
    ("warning_lumped_silent", "test_port_continues_warning",
     (AP, 'if collection == "_msl_ports" or port_termination_references(self, collection, port, grid):',
      'if True:')),
    ("fields_no_continuation", "test_run_forward_fields_equal", NO_CONTINUATION),
    ("fields_msl_hold", "test_run_forward_fields_equal", MSL_HOLD),
)


def main():
    """Validate every anchor before execution; errors or skips cannot kill a mutant."""
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for name, witness, (module, before, after) in ROWS:
        path = root / (module.replace(".", "/") + ".py")
        source = sources.setdefault(module, path.read_text())
        if source.count(before) != 1:
            raise RuntimeError(f"{name}: replacement anchor must occur exactly once in {path}")
    if len(sys.argv) == 1:
        env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        for name, _, _ in ROWS:
            completed = subprocess.run(
                [sys.executable, "-m", "tests.contracts.msl_continuation_mutations_1512", name], cwd=root, env=env,
                text=True, capture_output=True, timeout=300)
            if completed.returncode:
                raise AssertionError(f"{name}: no named assertion witness\n{completed.stdout}\n{completed.stderr}")
            print(f"{name}: named assertion failed", flush=True)
        return
    name, witness, (module, before, after) = next(row for row in ROWS if row[0] == sys.argv[1])
    target = import_module(module)
    exec(compile(sources[module].replace(before, after), target.__file__, "exec"), target.__dict__)
    import pytest
    class Witness:
        seen = False
        def pytest_collection_modifyitems(self, items):
            del items[1:]
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            yield
            if (call.when == "call" and call.excinfo is not None
                    and call.excinfo.errisinstance(AssertionError)
                    and item.originalname == witness):
                self.seen = True
    evidence = Witness()
    code = pytest.main(["-q", f"{TARGET}::{witness}", "-o", "addopts="], plugins=[evidence])
    if code != 1 or not evidence.seen:
        raise AssertionError(f"{name}: expected AssertionError in {witness}, pytest={code}")


if __name__ == "__main__":
    main()
