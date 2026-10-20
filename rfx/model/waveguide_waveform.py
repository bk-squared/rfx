"""Waveguide incident-table waveform admission (#1942)."""

DG_REFUSAL = (
    "waveform='differentiated_gaussian' is refused on waveguide ports: its "
    "band-limited incident table is built with a zero-phase filter whose "
    "precursor wraps circularly to the end of the record, so the port "
    "re-injects at the record end (an empty matched WR-90 guide read S21 off "
    "by up to 11 dB). Use waveform='modulated_gaussian' (the default). "
    "Tracked in #1942."
)


def waveguide_waveform_error(waveform):
    """Return the declaration and table builder's shared refusal message."""
    if waveform == "differentiated_gaussian":
        return DG_REFUSAL
    return f"waveform must be 'modulated_gaussian', got {waveform!r}"
