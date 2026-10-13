"""What a per-slab setup producer receives (rfx #1402, W1 staging).

A multi-device run realizes the model on each x slab, not on the whole grid and
then cut: a whole-grid setup leaves whole-domain arrays on every device. Each
producer that builds a grid-shaped quantity (realized materials, occupancy
edge factors, PEC edge masks, pole fractions, CPML state) has ONE
implementation, which runs on the whole grid on one device and on a
:class:`SlabView` on several. The contract, pinned by
``tests/contracts/path_equivalence/test_slab_view_producers.py``::

    view.own(producer(slab inputs, view)) == producer(whole inputs)[view.x0:view.x1]

bit for bit, on every slab, for two and three devices.

A view receives:

* the slab's input rows ``[lo, hi)`` of every x-first input array, cut with
  :meth:`SlabView.rows`: the owned rows ``[x0, x1)`` plus ``halo_lo`` rows
  below and ``halo_hi`` rows above;
* the GLOBAL grid. Its one-dimensional metrics are small and every predicate
  that selects an arithmetic path (``is_constant``, "is this axis graded")
  must be decided on the whole axis, never on the slab: a slab of equal cells
  inside a graded axis would otherwise take the equal-cell arithmetic. A
  producer slices its x metric with :meth:`SlabView.rows` after deciding;
* ``first`` / ``last``: whether the slab holds a global x face, so the
  outside-the-lattice rule (edge replication, zero padding, CPML faces) is
  applied there and only there. At an interior seam the halo supplies the
  neighbour instead;
* ``globals``: whole-model scalars a producer would otherwise compute from
  the whole grid (the Courant budget for the pole stability refusal, dt).

Halo: every grid-shaped producer on the graded path reads the cell one step
back along x and none reads forward (``_material_bwd_neighbour`` in
``edge_averaged_materials`` and ``component_h_materials``, the occupancy edge
factors). So ``HALO_LO = 1`` on every slab but the first and ``HALO_HI = 0``.
This is read from the code; the contract test is what verifies it, and a
producer that needs more is a stencil statement to make before changing it.
"""
from typing import Any, Mapping, NamedTuple

from rfx.stepping.slab import Slab

HALO_LO = 1
HALO_HI = 0


class SlabView(NamedTuple):
    rank: int
    n_devices: int
    x0: int                 # first owned global x row
    x1: int                 # one past the last owned global x row
    halo_lo: int            # input rows below x0 (0 on the first slab)
    halo_hi: int            # input rows above x1 (0 on the last slab)
    grid: Any               # the GLOBAL grid (1-D metrics only)
    globals: Mapping        # whole-model scalars

    @property
    def nx(self):
        return int(self.grid.nx)

    @property
    def first(self):
        return self.x0 == 0

    @property
    def last(self):
        return self.x1 == self.nx

    @property
    def lo(self):
        """Global x row of local input row 0."""
        return self.x0 - self.halo_lo

    @property
    def hi(self):
        return self.x1 + self.halo_hi

    def rows(self, a, axis=0):
        """The slab's input rows of a global array (or 1-D x metric)."""
        index = [slice(None)] * a.ndim
        index[axis] = slice(self.lo, self.hi)
        return a[tuple(index)]

    def own(self, a, axis=0):
        """The owned rows of a slab-local result."""
        index = [slice(None)] * a.ndim
        index[axis] = slice(self.halo_lo, self.halo_lo + self.x1 - self.x0)
        return a[tuple(index)]

    def local(self, i):
        """Local input row of a global x row inside ``[lo, hi)``."""
        if not self.lo <= i < self.hi:
            raise ValueError(f"x row {i} outside slab rows [{self.lo}, {self.hi})")
        return int(i - self.lo)


def slab_views(grid, n_devices, *, globals=None, halo_lo=HALO_LO, halo_hi=HALO_HI):
    """One view per device, with the runtime's x ownership (:class:`Slab`)."""
    layout = Slab(int(grid.nx), int(n_devices))
    views = []
    for rank, (x0, x1) in enumerate(layout.owned):
        views.append(SlabView(rank, layout.n_devices, x0, x1,
                              halo_lo if x0 > 0 else 0,
                              halo_hi if x1 < layout.nx else 0,
                              grid, dict(globals or {})))
    return tuple(views)
