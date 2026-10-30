"""Call-local gate for the option-B uniform-lane research experiment."""

import os


def conformal_b_q():
    """Read once; absence preserves the ordinary operands and timestep."""
    value = os.environ.get("RFX_CONFORMAL_B_Q")
    if value is None:
        return None
    try:
        q = float(value)
    except ValueError:
        raise ValueError("RFX_CONFORMAL_B_Q must be a float in (0, 1)") from None
    if not 0 < q < 1:
        raise ValueError("RFX_CONFORMAL_B_Q must be a float in (0, 1)")
    return q


def conformal_b_floor_frac():
    """Read the diagnostic floor fraction; absence keeps the q**2 floor."""
    value = os.environ.get("RFX_CONFORMAL_B_FLOOR_FRAC")
    if value is None:
        return None
    try:
        f = float(value)
    except ValueError:
        raise ValueError("RFX_CONFORMAL_B_FLOOR_FRAC must be a float in (0, 1]") from None
    if not 0 < f <= 1:
        raise ValueError("RFX_CONFORMAL_B_FLOOR_FRAC must be a float in (0, 1]")
    return f
