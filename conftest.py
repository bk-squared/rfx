"""
Repository-root conftest.py for rfx.

XLA_FLAGS isolation
-------------------
This file sets XLA_FLAGS *before* any ``import jax`` statement is executed.
pytest loads conftest.py at collection time, before test modules are imported,
so this is the correct place to inject the flag.

Without this, the distributed-NU composition tests in
``tests/test_distributed_nu_composition.py`` fail in a subtle way: if jax
was already imported in another module before XLA_FLAGS was set, JAX will
have already enumerated physical devices (typically 1 CPU) and the flag has
no effect.  When that happens, ``len(jax.devices()) < 2`` causes those tests
to skip (by design), but the absence of 2 virtual devices is the sign that
the env-var injection arrived too late.  Setting the var here, at the very top
of the first conftest that is loaded, ensures the flag is in the environment
before JAX's C++ backend initialises.

The ``setdefault`` call is intentional: if the user already has
``XLA_FLAGS`` set (e.g. for a GPU run), this does not overwrite it.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import warnings  # noqa: E402

import jax  # noqa: E402
import pytest  # noqa: E402


def pytest_configure(config):
    """Warn (but do not abort) when fewer than 2 JAX devices are visible.

    The distributed-NU composition tests require 2 virtual CPU devices.
    If the device count is 1 it means either:
      - XLA_FLAGS arrived too late (another conftest or test file imported jax first), or
      - the user is running in an environment that explicitly suppressed virtual devices.
    Individual composition tests guard against this with an in-body
    ``pytest.skip`` so the session continues without them.
    """
    n = len(jax.devices())
    if n < 2:
        warnings.warn(
            f"[conftest] Only {n} JAX device(s) visible. "
            "The distributed-NU composition tests will be skipped. "
            "Expected XLA_FLAGS=--xla_force_host_platform_device_count=2 "
            "to produce 2 virtual CPU devices. "
            "Check that nothing imported jax before this conftest was loaded.",
            stacklevel=1,
        )
    _maybe_register_rss_reporter(config)


# ---------------------------------------------------------------------------
# Weekly-lane RSS high-water reporter (issue #545 step 2).
#
# The 2026-07-27 and 2026-08-03 weekly slow-lane runs both had shards 1+3
# killed by a runner shutdown (exit 143) with zero test failures, i.e. the
# kill is infra-level, not a test assertion. ubuntu-latest gives ~7 GB RAM,
# and the unsharded full run was OOM-killed historically, so exit 143 is
# consistent with (but not proof of) memory pressure. This reporter gives
# the NEXT weekly run visibility into which tests are driving the process's
# memory footprint upward, to discriminate an OOM-adjacent kill from a pure
# infra shutdown.
#
# Opt-in via RFX_WEEKLY_RSS=1 (wired into validation.yml's slow-tests shard
# steps only) so it never touches the PR-gating fast suite. When the env var
# is unset, `_maybe_register_rss_reporter` returns before registering
# anything with pytest's pluginmanager — no hook exists at all, so there is
# no per-test overhead beyond the one `os.environ.get` check done once at
# session configure time. See
# tests/test_weekly_rss_reporter.py::test_disabled_by_default_registers_nothing.
#
# NOTE: RFX_WEEKLY_RSS_THRESHOLD_MB is intentionally NOT parsed at module
# scope. A malformed value would raise ValueError while conftest.py is being
# imported, which happens for EVERY pytest invocation in this repo (not just
# weekly-lane runs) -- that would red the entire test suite over a typo in
# an env var this file's own default gate is supposed to make irrelevant.
# The parse lives inside _maybe_register_rss_reporter, after the enable
# gate, with a fallback to the 500 MB default on a bad value. See
# tests/test_weekly_rss_reporter.py::test_malformed_threshold_with_reporter_off_does_not_raise.


def _rss_env_enabled() -> bool:
    return os.environ.get("RFX_WEEKLY_RSS", "").strip().lower() not in ("", "0", "false")


class _RSSHighWaterReporter:
    """pytest plugin: report per-test RSS high-water delta AND absolute peak.

    Uses ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (stdlib, no extra
    dependency; Linux reports it in KB). ``RUSAGE_SELF`` covers only this
    process, not any subprocesses it spawns.

    ``ru_maxrss`` is a monotonically non-decreasing high-water mark for the
    WHOLE process, so the *delta* measured around one test is "how much did
    this test push the process's peak RSS up" — a test that runs after the
    process has already reached a higher peak shows a 0 MB delta even if it
    is itself memory-heavy (once one test sets the peak, every later test
    reads +0.0 regardless of its own footprint). The *absolute* peak
    (``ru_maxrss`` itself, at that point in the run) does not have this
    blind spot -- it says the process had reached at least that much RSS by
    the time this test finished, which is what the OOM-vs-infra question
    actually needs. Both numbers are reported per test and in the summary.
    """

    def __init__(self, threshold_mb: float):
        self.threshold_kb = threshold_mb * 1024.0
        self.samples: list = []  # list[tuple[str, float, float]] of (nodeid, delta_mb, peak_mb)

    @staticmethod
    def _maxrss_kb() -> float:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        start_kb = self._maxrss_kb()
        yield
        end_kb = self._maxrss_kb()
        delta_kb = end_kb - start_kb
        if delta_kb >= self.threshold_kb:
            delta_mb = delta_kb / 1024.0
            peak_mb = end_kb / 1024.0
            self.samples.append((item.nodeid, delta_mb, peak_mb))
            print(
                f"\n[rss] {item.nodeid}: +{delta_mb:.1f} MB delta, "
                f"peak {peak_mb:.0f} MB"
            )

    def pytest_terminal_summary(self, terminalreporter):
        terminalreporter.write_line("")
        if not self.samples:
            terminalreporter.write_line(
                f"[rss] no test crossed the {self.threshold_kb / 1024.0:.0f} MB RSS high-water threshold"
            )
            return
        top = sorted(self.samples, key=lambda triple: triple[1], reverse=True)[:10]
        terminalreporter.write_line(
            f"[rss] top {len(top)} test(s) by RSS high-water delta "
            f"(threshold {self.threshold_kb / 1024.0:.0f} MB):"
        )
        for nodeid, delta_mb, peak_mb in top:
            terminalreporter.write_line(
                f"  +{delta_mb:8.1f} MB delta, peak {peak_mb:8.0f} MB  {nodeid}"
            )


def _maybe_register_rss_reporter(config: "pytest.Config") -> None:
    """Register the RSS high-water reporter iff RFX_WEEKLY_RSS is set.

    Kept as a standalone function (rather than inlined in pytest_configure)
    so the zero-cost-when-absent behavior is directly unit-testable against
    a plain mock config, without needing a real jax-aware pytest session.
    """
    if not _rss_env_enabled():
        return
    threshold_raw = os.environ.get("RFX_WEEKLY_RSS_THRESHOLD_MB", "500")
    try:
        threshold_mb = float(threshold_raw)
    except ValueError:
        warnings.warn(
            f"[conftest] RFX_WEEKLY_RSS_THRESHOLD_MB={threshold_raw!r} is not a "
            "valid number; falling back to the 500 MB default.",
            stacklevel=1,
        )
        threshold_mb = 500.0
    config.pluginmanager.register(
        _RSSHighWaterReporter(threshold_mb), "rfx-weekly-rss-reporter"
    )


@pytest.fixture(scope="session")
def two_devices():
    """Return the first 2 JAX devices, or skip the test if unavailable."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "Need >=2 JAX devices. "
            "Set XLA_FLAGS=--xla_force_host_platform_device_count=2 "
            "before importing jax (conftest.py does this automatically)."
        )
    return devices[:2]
