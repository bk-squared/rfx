"""Settling warnings name the record realized by an until_decay run."""

import re
import warnings

from rfx.errors import AccuracyWarning
from tests.unit.measurement.test_record_end_names_outputs import SBINS, _build


def test_until_decay_warning_names_realized_record_length():
    sim = _build(5)
    num_periods = 20.0
    planned = sim._build_realized_grid().num_timesteps(num_periods=num_periods)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(
            num_periods=num_periods, until_decay=1e-3,
            decay_min_steps=20, decay_check_interval=20, decay_max_steps=80,
            compute_s_params=True, s_param_freqs=SBINS,
        )
    realized = len(result.time_series)
    assert realized != planned
    messages = [str(w.message) for w in caught
                if issubclass(w.category, AccuracyWarning)
                and str(w.message).startswith("ring-down settling witness NOT PASSED")]
    assert messages, [str(w.message) for w in caught]
    for message in messages:
        count = re.search(r"\bn_steps=(\d+)\b", message)
        assert count is not None, message
        assert int(count.group(1)) == realized, message
