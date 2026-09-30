"""Opt-in observations of material operands at their consumption sites.

The observer is absent in ordinary runs: guards at the sites add no array
operations. Observed eps_e and mu_h are RELATIVE (dimensionless), sigma_e
is in S/m. A sharded record is on the kernel's local grid, including its
ghosts; consumers compare only the owned rows. Distributed E records carry
the global owned-row offset and count for a full-domain reference. Subgrid face records carry
their slice into the corresponding coarse or fine grid.

This diagnostic is serial and not an autodiff API. Entering/leaving it
clears JAX compilation caches so a previously traced program cannot hide a
site, or retain an observer after exit. The optional transform is solely
for the consumption experiment: the returned operands are used by the
existing update. With no transform the observer returns the same objects.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

ACTIVE = None


@dataclass
class Capture:
    records: list = field(default_factory=list)
    lane: str = ""
    sim: object = None
    transform: object = None
    _seen: set = field(default_factory=set)

    def observe(self, site, payload, *, region=None, periodic=(False,) * 3,
                metadata=None):
        import jax
        import numpy as np

        lane = self.lane
        traced = any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(payload))

        def save(values):
            assert not any(isinstance(x, jax.core.Tracer)
                           for x in jax.tree.leaves(values)), (
                               site, "realized records must contain concrete arrays")
            values = jax.tree.map(lambda x: np.array(x, copy=True), values)
            # A scan emits the same static operands each step. Keep each
            # distinct slab once, including different slabs of equal shape.
            key = (lane, site, repr(region), tuple(
                (x.shape, x.dtype.str, x.tobytes())
                for x in jax.tree.leaves(values)))
            if key not in self._seen:
                self._seen.add(key)
                record = dict(lane=lane, site=site, region=region,
                              periodic=periodic, traced=traced, **values, **(metadata or {}))
                if "waveforms" in values:
                    raw = values["raw_waveform"]
                    nonzero = np.flatnonzero(raw)
                    if not len(nonzero):
                        raise ValueError("a zero waveform cannot measure a drive scale")
                    k = nonzero[np.argmax(np.abs(raw[nonzero]))]
                    record["drive_scale"] = np.asarray(
                        [w[k] / raw[k] for w in values["waveforms"]])
                if "runtime_scale" in values:
                    for source in self.records:
                        if source["site"] == "distributed_nu.sources":
                            source["drive_scale"][list(record["columns"])] *= values["runtime_scale"]
                self.records.append(record)

        if traced:
            jax.debug.callback(save, payload)
        else:
            save(payload)

    def apply(self, site, quantity, values):
        return values if self.transform is None else self.transform(site, quantity, values)


@contextmanager
def capture(*, transform=None):
    """Collect actual operands; wait for all callbacks before returning."""
    import jax

    global ACTIVE
    if ACTIVE is not None:
        raise RuntimeError("realized-array captures cannot overlap")
    jax.clear_caches()
    result = Capture(transform=transform)
    ACTIVE = result
    try:
        yield result
    finally:
        try:
            jax.effects_barrier()
        finally:
            ACTIVE = None
            jax.clear_caches()


def enter(sim, lane):
    ACTIVE.sim = sim
    ACTIVE.lane = "fwd_adi" if lane == "fwd_uniform" and sim._solver == "adi" else lane


def electric(materials, eps, sigma, site, *, periodic=(False,) * 3,
             region=None, owned_start=None, owned_count=None):
    eps = ACTIVE.apply(site, "eps_e", eps)
    sigma = ACTIVE.apply(site, "sigma_e", sigma)
    payload = dict(eps_e=eps, sigma_e=sigma, materials=materials)
    if owned_start is not None:
        payload.update(owned_start=owned_start, owned_count=owned_count)
    ACTIVE.observe(site, payload, region=region, periodic=periodic)
    return eps, sigma


def scalar_electric(materials, site):
    eps, sigma = electric(materials, (materials.eps_r,) * 3,
                          (materials.sigma,) * 3, site)
    return materials._replace(eps_r=eps[0], sigma=sigma[0])


def magnetic(materials, mu, site, *, periodic=(False,) * 3):
    """Observe and replay the relative H operands returned by their owner."""
    mu = ACTIVE.apply(site, "mu_h", mu)
    ACTIVE.observe(site, dict(mu_h=mu, materials=materials), periodic=periodic)
    return mu


def sources(grid, materials, specs, site, *, dt=None):
    """Read the injection tables; divide by declared samples only in the dump.

    A record retains every injected cell of a wire source. P0's cell models
    declare one excitation at a time, so its waveform identifies all cells.
    Multiple declarations require separate captures; there is no guessed
    association between overlapping sources.
    """
    import jax
    import jax.numpy as jnp

    if not specs:
        return specs
    if len(ACTIVE.sim._ports) != 1:
        raise ValueError("capture one declared source or port at a time")
    pe = ACTIVE.sim._ports[0]
    tables = tuple(s.waveform if hasattr(s, "waveform") else s[4] for s in specs)
    cells = tuple((s.i, s.j, s.k, s.component) if hasattr(s, "waveform")
                  else tuple(s[:4]) for s in specs)
    n = len(tables[0])
    step = grid.dt if dt is None else dt
    raw = jax.vmap(pe.waveform)(jnp.arange(n, dtype=jnp.float32) * step)
    ACTIVE.observe(site, dict(waveforms=tables, raw_waveform=raw,
                             materials=materials),
                   metadata=dict(grid=grid, cells=cells, declaration=pe))
    return specs


def face(materials, region, eps, mu, site):
    eps = ACTIVE.apply(site, "eps_e", (eps,))[0]
    mu = ACTIVE.apply(site, "mu_h", (mu,))[0]
    ACTIVE.observe(site, dict(eps_e=(eps, eps), mu_h=(mu, mu),
                             materials=materials), region=region)
    return eps, mu


def runtime_drive(scales, columns):
    scales = ACTIVE.apply("distributed_nu.drive", "drive_scale", scales)
    ACTIVE.observe("distributed_nu.drive", dict(runtime_scale=scales),
                   metadata=dict(columns=columns))
    return scales
