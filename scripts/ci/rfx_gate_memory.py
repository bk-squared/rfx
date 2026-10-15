"""Gate-only pytest plugin: keep a worker's memory from growing with the number of modules it runs."""


def pytest_runtest_teardown(item, nextitem):
    """Drop JAX's compiled programs when a test module ends.

    JAX keeps every compiled program of a module alive after the module's last test, so a
    worker that runs many modules grows without bound (issue #1528). Selected-tests step, a
    454-file selection at eight workers: restarted twice at the 64 GiB container limit (62 GiB
    of process memory, 101,168 mappings in one process) without this hook, passed at 43 GiB and
    31,337 mappings with it. Contract step at 14 workers on the 32 GiB preset: restarted at the
    limit (32.8 GiB) without, 23.6 GiB with. Nothing is cleared inside a module.
    """
    if nextitem is None or getattr(nextitem, "module", None) is not getattr(item, "module", None):
        import gc

        import jax

        jax.clear_caches()
        gc.collect()
