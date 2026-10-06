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
     'fundamental_only': ('return (2 * first - 1, 2 * last - 1) if first <= last else None', 'return (1, 1) if lo <= 1 <= hi else None'),
    }
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
    if sys.argv[1] == 'declared_length':
        sys.exit(pytest.main(['-q', 'tests/contracts/test_line_stub_coverage_1512.py', '-o', 'addopts=']))
    selection = {'drop_refusal': 'every_entry', 'device_side': 'in_band_and_inclusive',
                 'fundamental_only': 'third_odd'}[sys.argv[1]]
    sys.exit(pytest.main(['-q', 'tests/contracts/test_line_stub_refusal_1512.py', '-k', selection, '-o', 'addopts=']))


if __name__ == "__main__":
    main()
