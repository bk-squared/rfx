"""Shared experimental ADI notice and public absorber refusal."""

ADI_WARNING = (
    'ADI is experimental and outside the 2.0 supported scope; it refuses graded '
    'meshes, interior PEC, material interfaces, and ports. Its accuracy envelope '
    'already includes a -1.4 % error at 2x CFL in a 3-D 12^3 cavity. '
    'Use solver="yee" for supported results.'
)

ADI_SPONGE_REFUSAL = (
    'solver="adi" refuses absorbing boundaries: this boundary is not a CPML '
    'but an unmatched graded-conductivity sponge (electric loss only, no '
    'magnetic loss). Measured normal-incidence reflection (2-D, the same '
    'construction as 3-D) is -10.4 dB at 10 GHz with the default 16 layers (-3 dB near 2 GHz), against -93 dB '
    'for the uniform-path CPML in the same box. Use solver="yee" with '
    'boundary="cpml" for open structures, or solver="adi" with '
    'boundary="pec" for closed cavities.'
)


class ExperimentalADIWarning(UserWarning):
    """Use of the experimental ADI solver outside the supported 2.0 scope."""


def has_absorber(sim):
    spec = getattr(sim, "_boundary_spec", None)
    return sim._boundary in ("cpml", "upml") or (
        spec is not None and spec.absorber_type is not None
    )


def require_closed_boundary(sim):
    if has_absorber(sim):
        raise ValueError(ADI_SPONGE_REFUSAL)


def warn_experimental():
    import sys
    import warnings

    stacklevel = 2
    frame = sys._getframe(1)
    try:
        while frame is not None and (
            frame.f_globals.get("__name__", "") == "rfx"
            or frame.f_globals.get("__name__", "").startswith("rfx.")
        ):
            stacklevel += 1
            frame = frame.f_back
    finally:
        del frame
    warnings.warn(ADI_WARNING, ExperimentalADIWarning, stacklevel=stacklevel)
