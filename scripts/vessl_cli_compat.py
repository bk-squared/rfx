#!/usr/bin/env python3
"""Compatibility wrapper for the installed VESSL CLI on modern NumPy.

The currently installed VESSL CLI imports ``numpy.mat``, which is removed in
newer NumPy releases. This wrapper restores a compatible alias before
dispatching into ``vessl.cli._main``.
"""

from __future__ import annotations

import sys

import numpy as np

if not hasattr(np, "mat"):
    np.mat = np.matrix  # type: ignore[attr-defined]

from vessl.cli._main import cli


if __name__ == "__main__":
    sys.exit(cli())
