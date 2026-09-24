"""tests/_printed_numbers.py

Compare a number that code printed with the value it stands for, as numbers.

A check that formats the expected value the way the code does and then looks
for that string fails whenever the two values straddle a rounding edge in the
last printed digit. That happens when they are computed along different paths
(a hand formula against a mode solve, one SVD against another) or on machines
whose BLAS kernels round differently: ``main`` went red four times on
2026-09-22/24 when a notes table's last digit followed the OpenBLAS kernel
(#1262). These helpers read the printed token back and allow one unit in its
last printed place, which is the resolution the message itself claims.
"""

from __future__ import annotations

import re

NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def last_place(token: str) -> float:
    """One unit in the last printed place of ``token``.

    ``"3.69748"`` -> 1e-5, ``"1.234e+05"`` -> 1e2, ``"12"`` -> 1.
    """
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 10.0 ** (int(exponent or 0) - decimals)


def agrees(token: str, value: float) -> bool:
    """``token`` is ``value`` to within one unit in its last printed place."""
    return abs(float(token) - value) <= 1.000001 * last_place(token)


def printed_after(text: str, label: str) -> list[str]:
    """Every number printed right after an occurrence of ``label``, as printed."""
    tokens = [NUMBER.match(part.lstrip()) for part in text.split(label)[1:]]
    assert tokens and all(tokens), f"no number after {label!r} in {text!r}"
    return [t.group(0) for t in tokens]


def printed_with_decimals(text: str, decimals: int) -> list[str]:
    """Every number in ``text`` printed with exactly ``decimals`` places."""
    return re.findall(rf"(?<![\d.])[-+]?\d+\.\d{{{decimals}}}(?![\d.])", text)
