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
    # Probe placed past the domain edge, genuinely inside the (exterior-
    # padded) CPML => preflight flags absorber_overlap. Issue #500: CPML
    # pads EXTERIOR to the requested domain (see rfx-known-issues.md
    # #500 / tests/unit/preflight/test_preflight_absorber.py), so a probe merely
    # near an edge but still within [0, 0.02] no longer trips this check
    # (z=0.018 here used to false-fire under the pre-#500 interior-frame
    # bug this file doesn't otherwise care about — it only needs ANY one
    # real preflight warning to exercise the run()-parity/consolidation
    # mechanism this file tests, and absorber_overlap is the original,
    # still-legitimate, and cheapest trigger to construct).
    #
    # Node arithmetic (review finding L7): this domain (0.02m, freq_max=
    # 10e9, default cpml_layers=16) auto-resolves dx~=1.499mm, giving
    # nz=47 with pad_z_lo=pad_z_hi=16 -> interior absolute indices 16..30
    # (last interior node 30 = z~=20.99mm). z=0.021 rounds to node 30
    # (round(0.021/dx)+16 = 30) -- still the LAST INTERIOR node, so it
    # only "fired" via _absorber_boundary_for_axis's up-to-one-cell
    # hi-side conservatism (see that helper's docstring), not because it
    # was genuinely exterior. z=0.0225 rounds to node 31, the first node
    # outside the interior slice -- genuinely in the absorber.
    sim = Simulation(domain=(0.02, 0.02, 0.02), freq_max=10e9, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), component="ez")
    sim.add_probe((0.01, 0.01, 0.0225), component="ez")
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
