"""Named port time records assembled after a solver scan."""
from dataclasses import dataclass

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class PortTimeRecord:
    """One port's scalar channels; only samples is a JAX leaf."""

    id: str
    kind: str
    channels: tuple[str, ...]
    samples: object

    def __post_init__(self):
        if len(self.channels) != self.samples.shape[1]:
            raise ValueError(f'{self.id}: channels {self.channels} do not match '
                             f'samples shape {self.samples.shape}')

    def __getitem__(self, channel):
        if not isinstance(channel, str):
            raise TypeError(f'read a channel by name from {self.channels}')
        if channel not in self.channels:
            raise KeyError(channel)
        return self.samples[..., self.channels.index(channel)]

    def __array__(self, dtype=None, copy=None):
        raise TypeError(f'read a channel by name from {self.channels}, or use .samples')

    def __iter__(self):
        raise TypeError(f'iterate .channels {self.channels}, not sample rows')

    def tree_flatten(self):
        return (self.samples,), (self.id, self.kind, self.channels)

    @classmethod
    def tree_unflatten(cls, metadata, leaves):
        # JAX may unflatten abstract placeholders or add a vmap batch axis.
        obj = object.__new__(cls)
        for name, value in zip(('id', 'kind', 'channels', 'samples'), (*metadata, leaves[0])):
            object.__setattr__(obj, name, value)
        return obj
