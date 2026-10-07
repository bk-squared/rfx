"""Named material defaults shared by the public API."""

import numpy as np

from rfx.core.yee import EPS_0
from rfx.materials.debye import DebyePole

MATERIAL_LIBRARY: dict[str, dict] = {
    "vacuum":      {"eps_r": 1.0, "sigma": 0.0},
    "air":         {"eps_r": 1.0006, "sigma": 0.0},
    "fr4":         {"eps_r": 4.4, "sigma": 0.025},
    # RO4003C: this entry uses the *process* Dk 3.55; for 50-ohm impedance
    # synthesis most designs use the *design* Dk 3.38 (Rogers RO4000 datasheet).
    "rogers4003c": {"eps_r": 3.55, "sigma": 0.0027 * 2 * np.pi * 5e9 * 3.55 * EPS_0},
    # RO4350B: Dk(design) 3.48, Df 0.0037 @ 10 GHz (Rogers RO4000 datasheet).
    "rogers4350b": {"eps_r": 3.48, "sigma": 0.0037 * 2 * np.pi * 10e9 * 3.48 * EPS_0},
    # RT/duroid 5880: Dk 2.20, Df 0.0009 @ 10 GHz (Rogers RT/duroid datasheet).
    "rt_duroid_5880": {"eps_r": 2.20, "sigma": 0.0009 * 2 * np.pi * 10e9 * 2.20 * EPS_0},
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
