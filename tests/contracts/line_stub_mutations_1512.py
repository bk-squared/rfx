"""Isolated in-memory mutants; never rewrites production files."""
import sys
from pathlib import Path
import pytest
from rfx.preflight import line_stub

def main():
    source = Path('rfx/preflight/line_stub.py').read_text()
    mutations = {
     'drop_refusal': ('                raise ValueError(stub_message(finding, band))', '                return'),
     'device_side': ('length = max(0.0, (plane - end) * sign)', 'length = max(0.0, ((hi if sign > 0 else lo) - plane) * sign)'),
     'declared_length': ('length = max(0.0, (plane - end) * sign)', 'length = declared_length'),
     'inner_default_band': ('return current\n', 'return (0.0, float(sim._freq_max))\n'),
     'substrate_eps': ('port.width, port.height, eps)[1]', 'port.width, port.height, eps)[1]; eps = substrate_eps'),
     'vacuum_eps': ('port.width, port.height, eps)[1]', 'port.width, port.height, eps)[1]; eps = 1.0'),
     'declared_first': ('result = None', 'result = None\n    if explicit is not None:\n        return float(explicit)'),
     'fundamental_only': ('return (2 * first - 1, 2 * last - 1) if first <= last else None', 'return (1, 1) if lo <= 1 <= hi else None'),
     # The pre-fix rule: metal length alone, no open-end extension.
     'no_end_extension': ('extension = open_end_extension(port.width, port.height, eps)', 'extension = 0.0'),
     # An enumeration cap drops every odd order above 3.
     'order_cap': ('    last = math.floor((hi + tol + 1) / 2)\n', '    last = min(math.floor((hi + tol + 1) / 2), 2)\n'),
     # Coax eps read at the metal pin centre again (falls back to 1).
     'coax_centre_eps': ('    if axis is not None:\n', '    if False:\n'),
     # skip_preflight makes a skipped shape silent again.
     'silent_skip': ('        if skipped:\n            _warn_uninspectable(skipped)\n', ''),
     # A fixed stack level cannot serve entries of different depth.
     'fixed_stacklevel': ('warnings.warn(_uninspectable_warning(skipped), stacklevel=level)', 'warnings.warn(_uninspectable_warning(skipped), stacklevel=4)'),
     # One uninspectable shape switches the whole check off again (the pre-isolation behaviour).
     'inspection_raises': ('            cache[key] = []\n', '            raise\n'),
    }
    if sys.argv[1] in ('warning_wrapper', 'early_admission'):
        from functools import wraps
        from rfx import Simulation
        def wrap(function):
            @wraps(function)
            def guarded(self, *args, **kwargs):
                if sys.argv[1] == 'early_admission':
                    line_stub.line_stub_admission(self)
                return function(self, *args, **kwargs)
            return guarded
        for name in ('run', 'forward', '_forward_from_materials', 'compute_msl_s_matrix',
                     'compute_mixed_s_matrix', 'compute_coaxial_line_reflection',
                     'compute_coaxial_two_port', 'compute_coax_msl_transition'):
            setattr(Simulation, name, wrap(getattr(Simulation, name)))
        target = ('tests/contracts/test_line_stub_admission_order.py' if sys.argv[1] == 'early_admission'
                  else 'tests/contracts/test_line_stub_warning_attribution.py')
        sys.exit(pytest.main(['-q', target, '-o', 'addopts=']))
    if sys.argv[1] == 'port_plane':
        import importlib
        for name, y, z in (
            ('test_patch_edgefed_s11_passivity', 'Y_C', 'Z_SUB_LO'),
            ('test_msl_nu_sparam_gate', 'Y_C', 'Z_SUB_LO'),
            ('test_patch_edgefed_resonance_harminv', 'y_c', 'z_sub_lo'),
        ):
            module = importlib.import_module('tests.locks.' + name)
            original = Path(module.__file__).read_text()
            allowance = f'PORT_MARGIN - local_port_cell(sim, (PORT_MARGIN, {y}, {z}))'
            assert original.count(allowance) == 1
            exec(compile(original.replace(allowance, 'PORT_MARGIN'), module.__file__, 'exec'), module.__dict__)
        sys.exit(pytest.main(['-q', 'tests/contracts/test_line_stub_coverage_1512.py', '-o', 'addopts=']))
    a, b = mutations[sys.argv[1]]
    assert source.count(a) == 1
    exec(compile(source.replace(a, b), str(line_stub.__file__), 'exec'), line_stub.__dict__)
    extra = {'no_end_extension': ['tests/unit/preflight/test_line_stub.py', 'tests/contracts/test_line_stub_refusal_1512.py',
                                  '-k', 'independent_closed_form or tail_length_and_frequency or counts_the_open_end'],
             'order_cap': ['tests/contracts/test_line_stub_refusal_1512.py', '-k', 'fifth_odd'],
             'coax_centre_eps': ['tests/unit/preflight/test_line_stub.py', '-k', 'between_pin_and_shield'],
             'silent_skip': ['tests/unit/preflight/test_line_stub.py', '-k', 'named_at_the_solve_entry'],
             'fixed_stacklevel': ['tests/unit/preflight/test_line_stub.py', '-k', 'named_at_the_solve_entry']}
    if sys.argv[1] in extra:
        sys.exit(pytest.main(['-q', *extra[sys.argv[1]], '-o', 'addopts=']))
    if sys.argv[1] == 'inspection_raises':
        sys.exit(pytest.main(['-q', 'tests/unit/preflight/test_line_stub.py', '-k', 'uninspectable', '-o', 'addopts=']))
    if sys.argv[1] == 'declared_first':
        sys.exit(pytest.main(['-q', 'tests/unit/preflight/test_line_stub.py', '-k', 'realized_substrate_owns', '-o', 'addopts=']))
    if sys.argv[1] == 'inner_default_band':
        sys.exit(pytest.main(['-q', 'tests/contracts/test_line_stub_refusal_1512.py', '-k', 'actual_calculator', '-o', 'addopts=']))
    if sys.argv[1] in ('substrate_eps', 'vacuum_eps'):
        sys.exit(pytest.main(['-q', 'tests/unit/preflight/test_line_stub.py', '-k', 'independent_closed_form', '-o', 'addopts=']))
    if sys.argv[1] == 'declared_length':
        sys.exit(pytest.main(['-q', 'tests/contracts/test_line_stub_coverage_1512.py', '-o', 'addopts=']))
    selection = {'drop_refusal': 'every_entry', 'device_side': 'in_band_and_inclusive',
                 'fundamental_only': 'third_odd'}[sys.argv[1]]
    sys.exit(pytest.main(['-q', 'tests/contracts/test_line_stub_refusal_1512.py', '-k', selection, '-o', 'addopts=']))


if __name__ == "__main__":
    main()
