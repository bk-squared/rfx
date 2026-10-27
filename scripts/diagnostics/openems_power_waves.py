"""Power-wave S-parameters from openEMS port waves (Kurokawa; rfx#460, rfx#1059).

openEMS's ``Port.CalcPort`` splits each port's voltage and current at that
port's own reference impedance ``Z_ref``: ``uf_inc = (u + Z_ref i)/2`` and
``uf_ref = (u - Z_ref i)/2``. ``Z_ref`` is the feed resistance of a lumped port
and, unless ``ref_impedance`` is passed, the measured line impedance ``ZL`` of an
MSL port. A ratio ``uf_ref_i / uf_inc_j`` between two ports whose ``Z_ref``
differ is therefore not an S-parameter: power-wave S carries
``sqrt(Re Z_ref_j / Re Z_ref_i)``, the convention rfx's own port lanes use
(``rfx/sparams/_port_reference.py``). A diagonal entry (i == j) needs no factor.

Standalone on purpose: the openEMS referees import no rfx code.
"""

from __future__ import annotations

import numpy as np


def power_wave_s(uf_ref_out, uf_inc_in, z_ref_out, z_ref_in):
    """S from the wave leaving port ``out`` per wave driving port ``in``.

    ``z_ref_out`` / ``z_ref_in`` are the ``Z_ref`` openEMS split each port's
    waves with (scalars or per-frequency arrays); only their real parts enter,
    as in rfx's mixed lane. Equal real parts give the bare ratio.
    """
    r_out = np.real(np.asarray(z_ref_out, dtype=np.complex128))
    r_in = np.real(np.asarray(z_ref_in, dtype=np.complex128))
    if np.any(r_out <= 0.0) or np.any(r_in <= 0.0):
        raise ValueError(
            "power_wave_s: a reference impedance has Re <= 0 "
            f"(min Re Z_out={float(np.min(r_out))!r}, min Re Z_in={float(np.min(r_in))!r})")
    ratio = (np.asarray(uf_ref_out, dtype=np.complex128)
             / np.asarray(uf_inc_in, dtype=np.complex128))
    return ratio * np.sqrt(r_in / r_out)
