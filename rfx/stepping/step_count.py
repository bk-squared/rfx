"""The shared period-to-step conversion."""

import numpy as np


def steps_for_periods(dt: float, freq_max: float, num_periods: float) -> int:
    """Return the steps covering num_periods periods of freq_max at dt.

    This is the only place this rule may be written, enforced by
    test_no_inline_step_count_outside_the_rule.
    """
    return int(np.ceil(float(num_periods) * (1.0 / float(freq_max)) / float(dt)))
