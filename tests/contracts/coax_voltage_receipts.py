"""Optional pytest recorder for before/after coax calculator measurements."""
import functools
import json
import os

import numpy as np


def pytest_configure(config):
    from rfx import Simulation
    for name in ('compute_coaxial_line_reflection', 'compute_coaxial_two_port',
                 'compute_coax_msl_transition'):
        original = getattr(Simulation, name)

        def wrap(function, label):
            @functools.wraps(function)
            def recorded(self, *args, **kwargs):
                result = function(self, *args, **kwargs)
                value = getattr(result, 's_params', getattr(result, 's11', None))
                try:
                    values = np.asarray(value)
                except Exception:
                    return result
                if values.dtype.kind != 'c':
                    return result
                record = dict(test=os.environ.get('PYTEST_CURRENT_TEST', ''),
                              calculator=label, dx=float(self._build_grid().dx),
                              shape=values.shape, real=values.real.tolist(),
                              imag=values.imag.tolist())
                print('S6B_RECEIPT ' + json.dumps(record), flush=True)
                return result
            return recorded

        setattr(Simulation, name, wrap(original, name))
