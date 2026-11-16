"""Assemble uniform forward S from the main run's measured port channels."""


def require_concrete_port_records(result):
    """Retain the uniform distributed entry's host-only port-S admission."""
    import jax
    import numpy as np
    for _, accs in (result.wire_port_sparams or ()) + (result.lumped_port_sparams or ()):
        for value in accs:
            if isinstance(value, jax.core.Tracer):
                # Keep the concrete-array boundary without transferring
                # already concrete spectra back to the host.
                np.asarray(value)


def pack_port_result(result):
    """Publish the same spectra and one-record S as one-device forward()."""
    from rfx.measurement.port_s_contract import uniform_record_pairs
    from rfx.measurement.port_s import port_s_from_record
    from rfx.sparams._record_selection import read_ids
    pairs = uniform_record_pairs(result)
    if not pairs:
        return result
    metas, channels = zip(*pairs)
    return result._replace(
        s_params=port_s_from_record(channels, metas, wire_mode=pairs is result.wire_port_sparams),
        freqs=metas[0].freqs, sparam_read_ids=read_ids(result.sparam_time_records))
