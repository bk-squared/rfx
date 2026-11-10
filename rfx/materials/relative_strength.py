"""A pole strength differentiated in dimensionless units."""
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class RelativeStrength:
    """Declare SI pole strength as ``reference * value``.

    ``reference`` is a positive finite Python/numpy float, static in the
    pytree; ``value`` is its dimensionless number, array, tracer, or map
    callable. Choose a reference near the largest strength so value is of
    order one: the cotangent's margin shrinks in proportion to its distance
    from one. Field coefficients multiply the concrete SI factors first.

    For traced SI declarations the optional float32 underflow warning uses
    ``RFX_POLE_UNDERFLOW_CHECK=1``; set it before the first call; a cached
    compile keeps the old setting. The check is off by default.
    """

    reference: float
    value: object

    def __post_init__(self):
        if (not isinstance(self.reference, (float, np.floating))
                or not np.isfinite(self.reference) or self.reference <= 0):
            raise ValueError("RelativeStrength reference must be a positive finite Python/numpy float")

    @property
    def si(self):
        """SI strength for admission and readers that do not feed fields."""
        if callable(self.value):
            return lambda *args: self.reference * self.value(*args)
        return self.reference * cast(Any, self.value)

    def _key(self):
        if isinstance(self.value, jax.core.Tracer) or callable(self.value):
            return (id(self),)
        value = np.asarray(self.value)
        return (self.reference, value.shape, tuple(value.flat))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if not isinstance(other, RelativeStrength):
            return NotImplemented
        return self is other or self._key() == other._key()

    def tree_flatten(self):
        return (self.value,), self.reference

    @classmethod
    def tree_unflatten(cls, reference, children):
        return cls(reference, children[0])
