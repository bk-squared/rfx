"""Named material defaults shared by the public API."""

import numpy as np

from rfx.core.yee import EPS_0
from rfx.materials.debye import DebyePole

_RO4003C_PROCESS_DK = 3.55
_RO4350B_DESIGN_DK = 3.48
_RO4350B_LOSS_TANGENT_AT_10_GHZ = 0.0037
_RT_DUROID_5880_DK = 2.20
_RT_DUROID_5880_LOSS_TANGENT_AT_10_GHZ = 0.0009
_ROGERS_LOSS_REFERENCE_HZ = 10e9


MATERIAL_LIBRARY: dict[str, dict] = {
    "vacuum":      {"eps_r": 1.0, "sigma": 0.0},
    "air":         {"eps_r": 1.0006, "sigma": 0.0},
    "fr4":         {"eps_r": 4.4, "sigma": 0.025},
    "rogers4003c": {"eps_r": _RO4003C_PROCESS_DK, "sigma": 0.0027 * 2 * np.pi * 5e9 * _RO4003C_PROCESS_DK * EPS_0},
    "rogers4350b": {"eps_r": _RO4350B_DESIGN_DK, "sigma": _RO4350B_LOSS_TANGENT_AT_10_GHZ * 2 * np.pi * _ROGERS_LOSS_REFERENCE_HZ * _RO4350B_DESIGN_DK * EPS_0},
    "rt_duroid_5880": {"eps_r": _RT_DUROID_5880_DK, "sigma": _RT_DUROID_5880_LOSS_TANGENT_AT_10_GHZ * 2 * np.pi * _ROGERS_LOSS_REFERENCE_HZ * _RT_DUROID_5880_DK * EPS_0},
    "alumina":     {"eps_r": 9.8, "sigma": 0.0},
    "silicon":     {"eps_r": 11.9, "sigma": 0.01},
    "ptfe":        {"eps_r": 2.1, "sigma": 0.0},
    "copper":      {"eps_r": 1.0, "sigma": 5.8e7},
    "aluminum":    {"eps_r": 1.0, "sigma": 3.5e7},
    "pec":         {"eps_r": 1.0, "sigma": 1e10},
    "water_20c":   {
        "eps_r": 4.9, "sigma": 0.0,
        "debye_poles": [DebyePole(delta_eps=74.1, tau=8.3e-12)],
    },
}
