"""Phasor exponentials with explicit left-to-right factor ordering."""
import functools
import operator

import numpy as np


def exp_j(*factors, unit=1j, xp=np):
    """exp(unit * f1 * f2 * ...), multiplied left to right as written at the call site."""
    return xp.exp(functools.reduce(operator.mul, factors, unit))
