"""Advisory host limits checked before the first time-stepping scan."""

import sys
import warnings
from pathlib import Path

from rfx.errors import ConfigurationWarning

MAP_COUNT_ADVISED = 262144
_checked = False


class MapCountWarning(ConfigurationWarning):
    """The host mapping limit is low for compiling many distinct grids."""


def _read_map_count() -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        return int(Path("/proc/sys/vm/max_map_count").read_text())
    except (OSError, ValueError):
        return None


def check_map_count() -> None:
    """Warn at most once per process, without interrupting a solve."""
    global _checked
    if _checked:
        return
    _checked = True
    try:
        n = _read_map_count()
        if n is not None and n < MAP_COUNT_ADVISED:
            warnings.warn(
                f"vm.max_map_count is {n}, below {MAP_COUNT_ADVISED:,}. "
                "One Python process that compiles many distinct grid shapes can abort inside the JAX "
                "compiler when it runs out of memory mappings; at the Linux default 65,530 this "
                "happened at about 151 grid shapes. Raise the limit with "
                "`sudo sysctl -w vm.max_map_count=262144`, or call `jax.clear_caches()` between "
                "batches of grid shapes. See docs/guides/known_limitations.md (#2421).",
                MapCountWarning, stacklevel=2,
            )
    except Exception:
        # Host inspection and warning filters (including -W error) must not stop a solve.
        pass
