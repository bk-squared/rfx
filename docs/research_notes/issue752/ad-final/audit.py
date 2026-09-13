from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import sys
import jax
import numpy as np
import pytest

root = Path('/root/rfx-codex/.git/issue752/ad-final')
records = []
real_grad = jax.grad


def observe_grad(function, *args, **kwargs):
    derivative = real_grad(function, *args, **kwargs)
    if 'test_compute_msl_s_matrix_ad_smoke_has_finite_gradient' not in function.__qualname__:
        return derivative

    def measured(theta):
        value = derivative(theta)
        g = float(value)
        rows = []
        # Fixed before this audit: both steps must independently agree to1%.
        # This is synthetic S-assembly, not a full-field accuracy claim.
        for step in (0.01, 0.005):
            plus = float(function(theta + step))
            minus = float(function(theta - step))
            fd = (plus - minus) / (2 * step)
            relative = abs(g - fd) / max(abs(g), abs(fd), np.finfo(float).tiny)
            rows.append(dict(step=step, plus=plus, minus=minus, finite_difference=fd,
                             relative_difference=relative))
        records.append(dict(kind='public synthetic S-assembly, same raw objective for AD and FD',
                            theta=float(theta), autodiff=g, finite_differences=rows))
        assert all(row['relative_difference'] < .01 for row in rows), records[-1]
        return value
    return measured


assert sys.flags.optimize == 0
with patch.object(jax, 'grad', observe_grad):
    result = pytest.main([
        'tests/unit/ports/test_msl_source_fixture_static.py::test_auto_eps_msl_gradient_matches_fd_mini_referee',
        'tests/unit/autodiff/test_msl_sparam_ad.py::test_compute_msl_s_matrix_ad_smoke_has_finite_gradient',
        'tests/unit/ports/test_msl_source_work.py',
        '-q', '-s', '-o', 'addopts=',
    ])
receipt = dict(pytest_exit_code=int(result), python=sys.version, jax=jax.__version__,
               backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
               records=records,
               source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
                   'rfx/api/_preflight.py', 'rfx/api/_sparams.py', 'rfx/sources/msl_port.py',
                   'tests/unit/ports/test_msl_source_fixture_static.py',
                   'tests/unit/autodiff/test_msl_sparam_ad.py',
                   'tests/unit/ports/test_msl_source_work.py')})
(root / 'ad-audit.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print('AD audit record:', json.dumps(receipt['records']))
sys.exit(int(result))
