"""pytest plugin (reviewer experiment, D2): a periodic axis gets L/dx nodes
(no fence-post duplicate) on the uniform Grid built by Simulation._build_grid.
Crude: it trims the last node after the stock constructor, so anything that
already sized itself from the stock grid elsewhere is not covered."""
import rfx.grid as G
import rfx.api._compile as C

_HINT = []
_orig_init = G.Grid.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    axes = _HINT[-1] if _HINT else ""
    for ax in axes:
        if ax == "z" and self.is_2d:
            continue
        setattr(self, "n" + ax, getattr(self, "n" + ax) - 1)
    self.shape = (self.nx, self.ny, self.nz)
    if self.is_2d:
        self.interior = (slice(self.pad_x_lo, self.nx - self.pad_x_hi),
                         slice(self.pad_y_lo, self.ny - self.pad_y_hi), slice(0, 1))
    else:
        self.interior = (slice(self.pad_x_lo, self.nx - self.pad_x_hi),
                         slice(self.pad_y_lo, self.ny - self.pad_y_hi),
                         slice(self.pad_z_lo, self.nz - self.pad_z_hi))


G.Grid.__init__ = _init
_orig_build = C._CompileMixin._build_grid


def _build(self, **kw):
    axes = "".join(a for a, p in zip("xyz", self._periodic_flags()) if p)
    _HINT.append(axes)
    try:
        return _orig_build(self, **kw)
    finally:
        _HINT.pop()


C._CompileMixin._build_grid = _build
