"""Old path of two moved names, kept until rfx 3.0 as their warning promises."""
_LAZY_EXPORTS = {
    "plain_dft": "rfx.measurement.ringdown_tail_jax",
    "tail_dft": "rfx.measurement.ringdown_tail_jax",
}


def __getattr__(name):
    from rfx._moved_names import resolve_moved
    return resolve_moved(__name__, _LAZY_EXPORTS, name)
