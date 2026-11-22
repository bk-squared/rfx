"""Host material declarations transported with a realized material field."""

from dataclasses import dataclass
from types import SimpleNamespace

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EchoMaterials:
    """Static declaration values, or the reason they could not be read."""

    values: tuple = ()
    reason: str | None = None

    @classmethod
    def from_sim(cls, sim):
        """Retain host values before realization changes or shards the arrays."""
        return cls.from_entries(sim._geometry, sim._resolve_material)

    @classmethod
    def from_entries(cls, entries, resolve):
        """Snapshot the materials used by a geometry declaration."""
        from rfx.materials.debye import DebyePole
        from rfx.materials.lorentz import LorentzPole

        try:
            values = []
            for name in dict.fromkeys(entry.material_name for entry in entries):
                material = resolve(name)
                debye = tuple(DebyePole(float(p.delta_eps), float(p.tau))
                              for p in material.debye_poles or ())
                lorentz = tuple(LorentzPole(float(p.omega_0), float(p.delta),
                                           float(getattr(p.kappa, 'si', p.kappa)))
                                for p in material.lorentz_poles or ())
                values.append((float(material.eps_r), float(material.mu_r), debye, lorentz))
            return cls(tuple(values))
        except (TypeError, ValueError, AttributeError) as exc:
            return cls(reason=f"host material declarations unavailable: {exc}")

    def materials(self):
        """Read the retained declarations without consulting device arrays."""
        if self.reason:
            raise ValueError(self.reason)
        return [SimpleNamespace(eps_r=e, mu_r=m, debye_poles=d, lorentz_poles=l)
                for e, m, d, l in self.values]

    def tree_flatten(self):
        return (), (self.values, self.reason)

    @classmethod
    def tree_unflatten(cls, metadata, leaves):
        return cls(*metadata)
