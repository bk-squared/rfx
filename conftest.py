"""
Repository-root conftest.py for rfx.

XLA_FLAGS isolation
-------------------
This file sets XLA_FLAGS *before* any ``import jax`` statement is executed.
pytest loads conftest.py at collection time, before test modules are imported,
so this is the correct place to inject the flag.

Without this, the distributed-NU composition tests in
``tests/unit/runners/test_distributed_nu_composition.py`` fail in a subtle way: if jax
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

import threading  # noqa: E402
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
# tests/unit/misc/test_weekly_rss_reporter.py::test_disabled_by_default_registers_nothing.
#
# NOTE: RFX_WEEKLY_RSS_THRESHOLD_MB is intentionally NOT parsed at module
# scope. A malformed value would raise ValueError while conftest.py is being
# imported, which happens for EVERY pytest invocation in this repo (not just
# weekly-lane runs) -- that would red the entire test suite over a typo in
# an env var this file's own default gate is supposed to make irrelevant.
# The parse lives inside _maybe_register_rss_reporter, after the enable
# gate, with a fallback to the 500 MB default on a bad value. See
# tests/unit/misc/test_weekly_rss_reporter.py::test_malformed_threshold_with_reporter_off_does_not_raise.
#
# issue #545 step 3 (2026-08-06 discriminator run 31103127491): this reporter
# alone left one structural gap disclosed in that run's writeup -- ru_maxrss
# only updates in the POST-test hook, so the exact RSS at the SIGKILL instant
# is never captured (the process dies before that hook resumes). _RSSLiveSampler
# below closes it with an independent background-thread poll of
# /proc/self/status VmHWM, on its own ~2s clock, printing (flushed
# immediately) whenever the high-water mark advances past the threshold --
# see its docstring.


def _rss_env_enabled() -> bool:
    return os.environ.get("RFX_WEEKLY_RSS", "").strip().lower() not in ("", "0", "false")


def _read_proc_vmhwm_mb() -> "float | None":
    """Read VmHWM (peak resident set size, all-time high-water) from
    ``/proc/self/status``, in MB.

    Returns ``None`` -- never raises -- when ``/proc/self/status`` does not
    exist (any non-Linux platform) or the ``VmHWM`` line is missing or
    unparsable, so callers treat "no reading" as "skip this sample," not as
    a crash or a real 0 MB reading. ``/proc`` is a Linux-only kernel
    interface, so this is the deliberate non-Linux guard for
    ``_RSSLiveSampler`` below.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    # Typical line: "VmHWM:\t 1234567 kB"
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except (OSError, IndexError, ValueError):
        return None
    return None


class _RSSLiveSampler:
    """Background daemon thread: samples ``/proc/self/status`` VmHWM on its
    own ~2s clock (independent of any pytest hook) and prints a line the
    instant the high-water mark has advanced by more than ``threshold_mb``
    past the last-printed value.

    Closes a gap ``_RSSHighWaterReporter`` below cannot: that reporter's
    ``pytest_runtest_protocol`` hookwrapper only samples at test
    boundaries (before/after each test runs), so if the OS SIGKILLs the
    process mid-test -- the exit-143 kill class issue #545 exists to
    discriminate -- the RSS at the moment of death is never captured; the
    process dies before the hookwrapper resumes. This thread samples on its
    own independent schedule and flushes every print immediately
    (``flush=True``), so a killed process still leaves its last high-water
    line in the streamed CI log even though it died mid-test.

    Zero cost when not started: the object can be constructed without ever
    launching its thread. In this file it is only constructed (via
    ``_RSSHighWaterReporter.__init__``) when ``_maybe_register_rss_reporter``
    has already passed the RFX_WEEKLY_RSS enable gate, and only started
    (via ``_RSSHighWaterReporter.pytest_sessionstart``) once a real pytest
    session begins -- so a unit test that merely constructs the reporter
    (e.g. against a ``MagicMock`` config) never spins up a thread.
    """

    def __init__(self, threshold_mb: float, interval_s: float = 2.0):
        self.threshold_mb = threshold_mb
        self.interval_s = interval_s
        self.current_nodeid = None  # best-effort, updated by a pytest hook
        self._last_reported_mb = 0.0
        self._started = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="rfx-rss-live-sampler", daemon=True
        )

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._started:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _sample_once(self) -> None:
        mb = _read_proc_vmhwm_mb()
        if mb is None:
            return  # non-Linux, or /proc unreadable -- silently skip
        if mb - self._last_reported_mb > self.threshold_mb:
            self._last_reported_mb = mb
            suffix = f" during {self.current_nodeid}" if self.current_nodeid else ""
            # flush=True: a SIGKILL must leave this line in the streamed log.
            print(f"[rss-live] VmHWM {mb:.0f} MB{suffix}", flush=True)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                self._sample_once()
            except (ValueError, OSError):
                # Narrow teardown-race guard ONLY: if the main thread has
                # already moved past pytest_sessionfinish / interpreter
                # shutdown by the time this daemon thread wakes for its next
                # sample, stdout (or the pipe under it) can already be
                # closed -- CPython raises ValueError ("I/O operation on
                # closed file") or OSError/BrokenPipeError in that case.
                # Deliberately NOT a bare `except Exception`: see
                # feedback_broad_except_eats_async_timeout (#555), where a
                # broad except in preflight swallowed a real SIGALRM timeout
                # and hung the process. A genuine bug in _sample_once should
                # still crash this thread loudly, not vanish here.
                return


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

    Also owns a background ``_RSSLiveSampler`` thread (issue #545 step 3)
    that samples VmHWM directly from ``/proc/self/status`` on its own ~2s
    clock, to capture the high-water mark at the instant of a mid-test
    SIGKILL that this class's own post-test hook cannot see (see
    ``_RSSLiveSampler``'s docstring). Started in ``pytest_sessionstart`` and
    stopped in ``pytest_sessionfinish``, both gated by the same
    RFX_WEEKLY_RSS enable check as this whole reporter, transitively, via
    ``_maybe_register_rss_reporter``.
    """

    def __init__(self, threshold_mb: float):
        self.threshold_kb = threshold_mb * 1024.0
        self.samples: list = []  # list[tuple[str, float, float]] of (nodeid, delta_mb, peak_mb)
        self.live_sampler = _RSSLiveSampler(threshold_mb)

    @staticmethod
    def _maxrss_kb() -> float:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def pytest_sessionstart(self, session):
        self.live_sampler.start()

    def pytest_sessionfinish(self, session, exitstatus):
        self.live_sampler.stop()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self.live_sampler.current_nodeid = item.nodeid
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
def _x64_baseline():
    """The ``jax_enable_x64`` value this session STARTED with."""
    return bool(jax.config.jax_enable_x64)


@pytest.fixture(autouse=True)
def _no_x64_leak(_x64_baseline):
    """Fail the test that leaves ``jax_enable_x64`` flipped (issue #646).

    ``jax_enable_x64`` is process-global. A test that flips it without scoping
    (a bare ``jax.config.update("jax_enable_x64", True)`` instead of
    ``tests/_x64_compat.enable_x64()``) silently changes the dtype environment
    of every test scheduled after it in the same pytest-split worker. That has
    already happened once: it reddened 13 unrelated tests in shard 3 on PR
    #645, and the reported failures pointed at innocent tests. This guard
    makes the leak fail in the test that CAUSED it.

    Two deliberate deviations from the sketch in issue #646:

    * **Function-scoped, not session-scoped.** A session-scoped teardown fires
      once, after everything, and so cannot name the culprit — which is the
      entire point. Per-test teardown can.
    * **Compares against the session baseline, not against ``False``.** The
      supported way to reproduce #646 is to run the suite under
      ``JAX_ENABLE_X64=1``; asserting a hard ``False`` would fail every test
      in that configuration. Anchoring to the session's initial value catches
      a leak in either direction and stays correct under both env values.

    Legitimate scoped use (``with enable_x64(): ...``) restores the flag
    before teardown and is unaffected.

    The flag is RESTORED before the assertion fires, so a leak costs exactly
    one red test — the one that caused it — instead of cascading into every
    test after it. Verified: with the restore, a deliberate leaker errors and
    its neighbour still passes; without it, both error.
    """
    yield
    now = bool(jax.config.jax_enable_x64)
    if now != _x64_baseline:
        jax.config.update("jax_enable_x64", _x64_baseline)
    assert now == _x64_baseline, (
        f"this test leaked jax_enable_x64: session started with "
        f"{_x64_baseline}, test left it {now}. jax_enable_x64 is "
        "process-global, so this changes the dtype environment of every test "
        "scheduled after it in this worker. Scope the flip with "
        "tests/_x64_compat.enable_x64() instead of calling "
        'jax.config.update("jax_enable_x64", ...) directly.'
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
