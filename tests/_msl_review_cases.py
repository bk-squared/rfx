"""Small additions from the reviewer's independent geometry matrix."""
from dataclasses import replace
import numpy as np

from rfx import Box
from tests._msl_diagnostic_cases import structure, U


def additional_cases(graded):
    for runway in (7, 14, 20):
        ramp = U * 1.25 ** np.arange(1, 4)
        profile = np.concatenate([np.full(runway, U), ramp, np.full(6, ramp[-1]),
                                  ramp[::-1], np.full(8, U)])
        for height in (U, 2 * U, 4.45 * U):
            for offset in (None, 3, 12):
                sim = structure(
                    graded,
                    height=height,
                    offset=offset,
                    length=float(profile.sum()),
                )
                sim._dx_profile = profile
                yield sim
    for height in (U, 2 * U):
        sim = structure(
            graded,
            height=height,
            feed=4.3 * U,
        )
        sim._dx_profile = np.full(32, U)
        yield sim
    sim = structure(
        graded,
        feed=16 * U,
        freq_max=100e9,
    )
    sim._msl_ports[0] = replace(
        sim._msl_ports[0],
        eps_r_sub=6.0,
    )
    sim._geometry[-1] = replace(
        sim._geometry[-1],
        shape=Box((0, 4 * U, 2 * U), (32 * U, 8 * U, 2 * U)),
    )
    yield sim
