"""D5' mutations, scoped to the distributed runner and restored after pytest."""
import sys
from unittest.mock import patch

import pytest

from rfx.boundaries import tfsf


def main():
    mode = sys.argv[1]
    if mode not in ('disable-validation', 'omit-pec-edges'):
        raise ValueError(mode)
    original = tfsf.validate_realized_vacuum_boundary

    def mutated(*args, **kwargs):
        if sys._getframe(1).f_code.co_filename.endswith('/rfx/runners/distributed_v2.py'):
            if mode == 'disable-validation':
                return None
            kwargs['pec_edge_masks'] = None
        return original(*args, **kwargs)

    with patch.object(tfsf, 'validate_realized_vacuum_boundary', mutated):
        result = pytest.main([
            '-q', '-p', 'no:cacheprovider',
            'tests/contracts/test_tfsf_injection_plane_operands.py',
            '-k', 'test_closed_box_distributed_injection_plane and pec_sheet and not thin_pec_sheet',
        ])
    assert tfsf.validate_realized_vacuum_boundary is original
    return result


if __name__ == '__main__':
    raise SystemExit(main())
