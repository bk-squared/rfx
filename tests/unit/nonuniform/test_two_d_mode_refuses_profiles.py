"""A 2-D ``mode`` with an axis profile is refused.

The non-uniform lane never reads ``mode``. Before the refusal a
``mode="2d_tez"`` model with a ``dy_profile`` ran as a one-cell PEC box, in
which only Ez survives, and returned all-zero fields without a word.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation


@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
@pytest.mark.parametrize("axis", ["dx_profile", "dy_profile", "dz_profile"])
def test_a_two_d_mode_with_a_profile_is_refused_with_the_way_out(mode, axis):
    n = {"dx_profile": 20, "dy_profile": 10, "dz_profile": 1}[axis]
    with pytest.raises(ValueError, match="non-uniform lane solves 3-D only"):
        Simulation(freq_max=30e9, domain=(0.020, 0.010, 1e-3), dx=1e-3,
                   boundary="pec", mode=mode, **{axis: np.full(n, 1e-3)})


def test_the_same_profile_in_three_d_is_still_accepted():
    Simulation(freq_max=30e9, domain=(0.020, 0.010, 1e-3), dx=1e-3,
               boundary="pec", mode="3d", dy_profile=np.full(10, 1e-3))


def test_a_two_d_mode_without_profiles_is_untouched():
    Simulation(freq_max=30e9, domain=(0.020, 0.010, 1e-3), dx=1e-3,
               boundary="pec", mode="2d_tez")
