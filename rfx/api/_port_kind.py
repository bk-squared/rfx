"""Which ``Simulation._ports`` entries are ports and which are soft sources.

``Simulation._ports`` holds two kinds of entry, told apart only by impedance:
``add_source`` stamps ``impedance == 0.0`` (a soft point source: no
termination, the drive adds to the field), and ``add_port`` requires a
positive, finite impedance for every lumped or wire port.
For entries stored by these constructors, remaining ``impedance > 0`` tests
are equivalent to :func:`is_port`.
Readers share these functions, including through the ``PortKind`` properties;
a traced impedance is not a soft source.
Deliberate exceptions are ``rfx.artifacts`` (duck-typed entries may omit
impedance), ``rfx.measurement.record_end`` (mixed MSL entries), and
``rfx.interop._design`` (validates the number before classifying it).
This module keeps its rfx import local to avoid import cycles.
"""


def is_soft_source(entry) -> bool:
    """True for an untraced zero-impedance soft point source."""
    from rfx.core.jax_utils import is_tracer

    return not is_tracer(entry.impedance) and float(entry.impedance) == 0.0


def is_port(entry) -> bool:
    """True for an entry that is not a soft point source."""
    return not is_soft_source(entry)


class PortKind:
    """The port / soft-source rule, mixed into ``_PortEntry``."""

    __slots__ = ()

    @property
    def is_soft_source(self) -> bool:
        """True for an ``add_source`` soft point source (``impedance == 0.0``)."""
        return is_soft_source(self)

    @property
    def is_port(self) -> bool:
        """True for an ``add_port`` lumped or wire port: ``not is_soft_source``."""
        return is_port(self)
