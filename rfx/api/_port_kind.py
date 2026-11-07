"""Which ``Simulation._ports`` entries are ports and which are soft sources.

``Simulation._ports`` holds two kinds of entry, told apart only by impedance:
``add_source`` stamps ``impedance == 0.0`` (a soft point source: no
termination, the drive adds to the field), and ``add_port`` refuses an
impedance ``<= 0``, so every lumped or wire port has a nonzero impedance.
A reader that asks "is this a port" uses :attr:`PortKind.is_port`, never its
own impedance test. This module is a leaf: it imports nothing from rfx.
"""


class PortKind:
    """The port / soft-source rule, mixed into ``_PortEntry``."""

    __slots__ = ()

    @property
    def is_soft_source(self) -> bool:
        """True for an ``add_source`` soft point source (``impedance == 0.0``)."""
        return self.impedance == 0.0

    @property
    def is_port(self) -> bool:
        """True for an ``add_port`` lumped or wire port: ``not is_soft_source``."""
        return not self.is_soft_source
