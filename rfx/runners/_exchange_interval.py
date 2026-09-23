"""Entry validation for distributed ghost exchange intervals."""

from numbers import Integral


def validate_exchange_interval(exchange_interval):
    """Require an integer exchange_interval of 1."""
    if (isinstance(exchange_interval, bool)
            or not isinstance(exchange_interval, Integral)
            or exchange_interval != 1):
        raise ValueError(
            f"exchange_interval={exchange_interval!r} is refused: "
            "each skipped exchange updates seam cells from stale neighbour values "
            "and the field grows exponentially in a lossless box; "
            "use exchange_interval=1."
        )
