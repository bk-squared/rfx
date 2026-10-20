"""tests/_module_cache_clear.py -- drop JAX's compiled programs at each module end.

With ``RFX_CLEAR_JAX_CACHES_PER_MODULE`` set (validation.yml's weekly shards
only), ``jax.clear_caches()`` and ``gc.collect()`` run after the last test of
every module, once that module's fixtures are torn down.

Why (#1932; record rfx-archive rfx/records/20261021-weekly-shard5-maps/):
weekly shard 5 aborted with ``LLVM ERROR: Unable to allocate section memory``
at 258,831 of the runner's 262,144 memory mappings, with 3.4 GB still free.
The distributed dispersive-CPML files add about 26,000 mappings each and none
are released at module end. The same shard with this clear completes, peaking
at 65,610 mappings and 7.7 GB (11.8 GB without); returning freed heap to the OS
instead (malloc_trim) still aborts.

Unset, the plugin registers nothing.
"""
import gc
import os

import pytest

ENV = "RFX_CLEAR_JAX_CACHES_PER_MODULE"


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() not in ("", "0", "false")


class _ClearAtModuleEnd:
    @pytest.hookimpl(hookwrapper=True, trylast=True)
    def pytest_runtest_protocol(self, item, nextitem):
        # After the protocol the item's teardown has run, and when the next
        # item is in another module (or there is none) so has the module's.
        yield
        if nextitem is None or nextitem.path != item.path:
            import jax

            jax.clear_caches()
            gc.collect()


def pytest_configure(config):
    if enabled():
        config.pluginmanager.register(_ClearAtModuleEnd(), "rfx-module-cache-clear")
