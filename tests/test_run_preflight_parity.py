"""Tier-1: run() must get the same consolidated, skippable preflight that
forward() already gets (issue #66 parity).

Previously run() emitted only scattered raw mesh/config warnings with no
skip_preflight control, so the documented lumped/wire S-parameter path via
run(compute_s_params=True) silently missed part of the proactive error
surface. This locks: (1) run() surfaces a setup footgun, (2) skip_preflight
suppresses it, (3) the consolidated warning routes through _auto_preflight.
"""
import warnings

import pytest

from rfx.api import Simulation


def _sim_with_probe_in_cpml():
    # Probe + source pushed to the very edge => preflight flags CPML overlap.
    sim = Simulation(domain=(0.02, 0.02, 0.02), freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.018), component="ez")
    return sim


def test_run_emits_preflight_warning_by_default():
    sim = _sim_with_probe_in_cpml()
    with pytest.warns(UserWarning, match="(?i)preflight|CPML"):
        sim.run(n_steps=10)


def test_run_skip_preflight_suppresses_it():
    sim = _sim_with_probe_in_cpml()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any preflight UserWarning => failure
        # A genuine numerical warning unrelated to preflight could still fire;
        # this config only trips the preflight surface, so silence == success.
        sim.run(n_steps=10, skip_preflight=True)


def test_run_preflight_warning_is_consolidated_single_warning():
    """_auto_preflight folds all issues into ONE UserWarning (vs the old
    scattered raw warnings)."""
    sim = _sim_with_probe_in_cpml()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim.run(n_steps=10)
    preflight_warnings = [
        w for w in rec
        if issubclass(w.category, UserWarning) and "preflight" in str(w.message).lower()
    ]
    assert len(preflight_warnings) == 1, (
        f"expected one consolidated preflight warning, got {len(preflight_warnings)}"
    )
