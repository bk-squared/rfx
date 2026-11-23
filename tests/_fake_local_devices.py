import jax


_real_local_devices = jax.local_devices


def fake_local_devices(default):
    """Return a JAX-signature fake that delegates explicit backends to real devices and otherwise calls default, allowing it to raise."""
    def local_devices(process_index=None, backend=None, host_id=None):
        if backend is not None:
            return _real_local_devices(
                process_index=process_index, backend=backend, host_id=host_id
            )
        return default()

    return local_devices
