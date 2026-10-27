"""Forward's port schema over the uniform distributed S driver's result."""

from rfx.api._spec import ForwardResult


def _unrecorded(field):
    raise NotImplementedError(
        f"{field}: a multi-device forward() on a uniform mesh does not record "
        "this port quantity; run on one device (omit distributed=True)")


class _PortForwardResult(ForwardResult):
    """Keep incomplete raw port spectra out of the public port fields.

    The driver's reference phasors are placeholders and forward's record
    run forms no spectra, so the port spectrum field of the port's kind
    refuses on read. ``sparam_time_records`` is the main forward run's
    port time record, as run(devices=...) returns it. Settling still
    scores the recorded user probes, as it did before packing the port
    diagonal.
    """
    __slots__ = ()

    @property
    def settling_db(self):
        return ForwardResult(*self).settling_db

    @property
    def settling_witness(self):
        return ForwardResult(*self).settling_witness


class _LumpedPortForwardResult(_PortForwardResult):
    __slots__ = ()

    @property
    def lumped_port_sparams(self):
        return _unrecorded("lumped_port_sparams")


class _WirePortForwardResult(_PortForwardResult):
    __slots__ = ()

    @property
    def wire_port_sparams(self):
        return _unrecorded("wire_port_sparams")


def pack_port_result(result, *, wire):
    """Keep the driver's forward port S matrix and refuse its missing records."""
    cls = _WirePortForwardResult if wire else _LumpedPortForwardResult
    return cls(*result)
