"""Forward finite-difference accuracy and wrong-gradient falsifiers.
"""
from __future__ import annotations


import numpy as np
import pytest

from tests import _waveguide_chain_battery_gates as G


@pytest.mark.parametrize("theta0", [0.0, 0.7, -0.2])
def test_forward_stencil_differentiates_quadratic(theta0):
    h = 0.05
    def f(x):
        return 2 + 3*x + x*x
    e = G.forward2_ad_fd_entry(g_ad=3+2*theta0, f0=f(theta0), f_plus=f(theta0+h),
                              f_2h=f(theta0+2*h), h=h, loss_dtype=np.float64)
    assert e["g_fd"] == pytest.approx(3+2*theta0, abs=3e-14)
    assert e["verdict"] == "pass"
    assert e["gate"] == 0.05


def test_forward_stencil_has_second_order_error():
    errors = []
    for h in (0.05, 0.025):
        def f(x):
            return 2+x+x**3
        e = G.forward2_ad_fd_entry(g_ad=1.0, f0=f(0), f_plus=f(h), f_2h=f(2*h),
                                  h=h, loss_dtype=np.float64)
        errors.append(abs(e["g_fd"]-1))
    assert errors[0]/errors[1] == pytest.approx(4, rel=1e-9)


def test_resolved_wrong_gradient_still_fails_existing_gate():
    e = G.forward2_ad_fd_entry(g_ad=1.051, f0=2, f_plus=2.05, f_2h=2.1, h=.05, loss_dtype=np.float64)
    assert e["fd_ulp_span"] > 1e4
    assert e["verdict"] == "fail"
