"""Isolated in-memory mutants; never rewrites production files."""
import sys
from pathlib import Path
import pytest
from rfx.preflight import line_stub

def main():
    source = Path('rfx/preflight/line_stub.py').read_text()
    mutations = {
     'drop_refusal': ('                raise ValueError(stub_message(finding, band))', '                return'),
     'device_side': ('end = lo if sign > 0 else hi\n                    length = (plane - end) * sign', 'end = hi if sign > 0 else lo\n                    length = (end - plane) * sign'),
     'fundamental_only': ('return (2 * first - 1, 2 * last - 1) if first <= last else None', 'return (1, 1) if lo <= 1 <= hi else None'),
    }
    a, b = mutations[sys.argv[1]]
    assert source.count(a) == 1
    exec(compile(source.replace(a, b), str(line_stub.__file__), 'exec'), line_stub.__dict__)
    selection = {'drop_refusal': 'every_entry', 'device_side': 'in_band_and_inclusive',
                 'fundamental_only': 'third_odd'}[sys.argv[1]]
    sys.exit(pytest.main(['-q', 'tests/contracts/test_line_stub_refusal_1512.py', '-k', selection, '-o', 'addopts=']))


if __name__ == "__main__":
    main()
