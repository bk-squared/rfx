"""Pin d(1/Q)/dRs of both x walls to the TE101 wall-loss closed form."""
import json

import numpy as np

from tests._x64_compat import enable_x64
from tests.oracle._sheet_rs_pole_scene import measure


def test_sheet_rs_pole_derivative():
    with enable_x64():
        measured = measure()
    print("sheet Rs pole " + json.dumps(measured, sort_keys=True))
    assert np.isfinite(measured["s"]), f"nonfinite sheet slope: {measured['s']}"
    assert abs(measured["r"]) <= 1e-2, (
        f"P1: abs(r)={abs(measured['r'])} exceeds 0.01; "
        f"s={measured['s']}, s_cf={measured['s_cf']}")
