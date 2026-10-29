"""Exact rational plane waves on the uniform three-dimensional Yee lattice."""
from __future__ import annotations

from rfx.errors import AccuracyWarning
from dataclasses import dataclass
from functools import lru_cache
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import jax.numpy as jnp
from jax import lax, tree_util

from rfx.core.yee import EPS_0, MU_0, FDTDState, e_update_coeffs

if TYPE_CHECKING:
    from rfx.api import Simulation


COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")
# rfx/snapshots.py:65-70; these also make yee.py's forward-H/backward-E
# differences exactly symmetric about each target component's location.
YEE_OFFSETS = np.array(((.5, 0, 0), (0, .5, 0), (0, 0, .5),
                        (0, .5, .5), (.5, 0, .5), (.5, .5, 0)))
# Target x/y/z curl: (positive source, derivative axis, negative source, axis).
CURL_TERMS = ((2, 1, 1, 2), (0, 2, 2, 0), (1, 0, 0, 1))


class CorrectionTable(NamedTuple):
    target_component: int
    target_index: np.ndarray
    source_component: int
    source_index: np.ndarray
    coefficient: np.ndarray


class LaunchWindow(NamedTuple):
    nodes: np.ndarray
    weights: np.ndarray
    delays: np.ndarray


class IncidentStep(NamedTuple):
    step: object


@dataclass(frozen=True)
class PlaneWaveConfig:
    """Static lattice geometry and gathers for one bounded record."""
    m: tuple
    incidence: dict
    shape: tuple
    bounds: object
    origin: int
    n_nodes: int
    n_steps: int
    fractions: tuple
    projections: tuple
    tables_e: tuple
    tables_h: tuple
    source_nodes: tuple
    delays: tuple
    source_p: float
    dx_1d: float
    dt: float
    ca: object
    cb: object
    ch: object
    src_amp: float
    src_tau: float
    src_t0: float
    src_fcen: float
    src_waveform: str
    custom_waveform: object
    source_windows: tuple = ()
    window_sigma: int = 0
    window_q1: float = 0.0
    closed_box: bool = True
    face_record: object = None
    face_gathers: tuple = ()
    record_faces: bool = False
    # Preserve the requested angles for lane admission after realization.
    theta_deg: float = 0.0
    phi_deg: float = 0.0

    def __getattr__(self, name):
        if name in {a + side for a in "xyz" for side in ("_lo", "_hi")}:
            return getattr(self.bounds, name)
        raise AttributeError(name)


def _tables(shape, bounds, m, projections, dt, dx, magnetic):
    """M curl - curl M; the self-read cancels, leaving one neighbour per term."""
    result = []
    base = dt / ((MU_0 if magnetic else EPS_0) * dx)
    for target, (first, axis_a, second, axis_b) in enumerate(CURL_TERMS):
        for source, axis, curl_sign in ((first, axis_a, 1), (second, axis_b, -1)):
            for high in (False, True):
                axes = [np.arange(getattr(bounds, a + "_lo"), getattr(bounds, a + "_hi") + 1)
                        for a in "xyz"]
                plane = getattr(bounds, "xyz"[axis] + ("_hi" if high else "_lo"))
                plane += (0 if high else -1) if magnetic else (1 if high else 0)
                axes[axis] = np.array([plane])
                target_nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
                source_nodes = target_nodes.copy()
                source_nodes[:, axis] += 1 if magnetic else -1
                component = source if magnetic else source + 3
                line_n = np.floor(source_nodes @ m + projections[component]).astype(np.int32)
                difference = (1 if high else -1) if magnetic else (-1 if high else 1)
                result.append(CorrectionTable(
                    target + (3 if magnetic else 0),
                    np.ravel_multi_index(target_nodes.T, shape).astype(np.int32),
                    component, line_n,
                    np.full(len(line_n), -curl_sign * difference * base, dtype=np.float32)))
    return tuple(result)


@lru_cache(maxsize=256)
def launch_window_width(m):
    """First dispersion maximum and ceil(5/q1), computed in float64 NumPy."""
    m = np.abs(np.asarray(m, dtype=np.float64))
    grid = np.linspace(np.pi / max(m), 2 * np.pi / max(m), 32769)
    derivative = .5 * np.sum(m * np.sin(grid[:, None] * m), axis=1)
    negative = np.flatnonzero(derivative < 0)
    if not len(negative) or negative[0] == 0:
        raise ValueError("cannot bracket the first plane-wave dispersion maximum")
    index = int(negative[0])
    lo, hi = float(grid[index - 1]), float(grid[index])
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if .5 * np.sum(m * np.sin(mid * m)) > 0:
            lo = mid
        else:
            hi = mid
    q1 = (lo + hi) / 2
    if .5 * np.sum(m * m * np.cos(q1 * m)) >= 0:
        raise ValueError("plane-wave dispersion stationary point is not a maximum")
    return q1, int(np.ceil(5 / q1))


def launch_windows(m, fractions, source_p, sigma, dx, dt):
    """Per-sublattice Gaussian support, weights and forward-travelling delays."""
    speed = np.linalg.norm(m) * 299792458.0 * dt / dx
    windows = []
    for fraction in fractions[:3]:
        low = int(np.ceil(source_p - 4 * sigma - fraction))
        high = int(np.floor(source_p + 4 * sigma - fraction))
        nodes = np.arange(low, high + 1, dtype=np.int32)
        offset = nodes + fraction - source_p
        weights = np.exp(-.5 * (offset / sigma) ** 2)
        weights /= np.sum(weights)
        windows.append(LaunchWindow(nodes, weights, offset / speed))
    return tuple(windows)


def require_step_bound(n_steps):
    """Refuse an unbounded auxiliary line before any step-count arithmetic."""
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("closed_box requires a finite positive integer bound on its step count "
                         "(n_steps, or decay_max_steps with until_decay)")


def source_scalar(name, value):
    """Incident-source scalars are fixed inputs, including under forward AD."""
    from rfx.core.jax_utils import is_tracer
    if is_tracer(value):
        raise ValueError(f"{name}: differentiating with respect to the incident source is not supported")
    return float(value)


def validate_run(entry, until_decay, decay_max_steps, under_transform):
    """Run admission precedes setup and decay-parameter arithmetic."""
    from rfx.sources.sources import CustomWaveform
    if entry is None:
        return
    if entry.closed_box and until_decay is not None:
        require_step_bound(decay_max_steps)
    if not (entry.closed_box or isinstance(entry.waveform, CustomWaveform)):
        return
    if under_transform():
        raise NotImplementedError(
            "CustomWaveform and closed_box TFSF cannot be passed through run() "
            "under external JAX transformations; differentiate forward() instead.")


def init_plane_wave(nx, dx, dt, *, ny, nz, cpml_layers=0, tfsf_margin=3,
                    f0=5e9, bandwidth=.5, amplitude=1., polarization="ez", direction="+x",
                    theta_deg=0., phi_deg=0., direction_tol_deg=.1, max_component=100,
                    waveform="differentiated_gaussian", n_steps=None):
    """Build the line with N*max(abs(m))+2 untouched nodes beyond every read."""
    from rfx.sources.sources import CustomWaveform
    from rfx.sources.tfsf import AUX_N_MARGIN_1D, AUX_SRC_OFFSET_1D
    require_step_bound(n_steps)
    amplitude = source_scalar("amplitude", amplitude)
    f0 = source_scalar("f0", f0)
    bandwidth = source_scalar("bandwidth", bandwidth)
    info = incidence(theta_deg, phi_deg, direction, polarization, direction_tol_deg, max_component)
    m = np.asarray(info["m"], dtype=np.int64)
    norm = float(np.linalg.norm(m))
    if norm != 1.:
        amplitude *= norm
    shape = (nx, ny, nz)
    bounds = box_bounds(shape, cpml_layers, tfsf_margin)
    projections = YEE_OFFSETS @ m
    fractions = projections % 1
    tables_e = _tables(shape, bounds, m, projections, dt, dx, False)
    tables_h = _tables(shape, bounds, m, projections, dt, dx, True)
    min_p = min(float(np.min(t.source_index) + fractions[t.source_component])
                for t in tables_e + tables_h)
    radius = int(np.max(np.abs(m)))
    source_p = min_p - (3 * radius + 3)
    windows, q1, sigma = (), 0.0, 0
    if info["m"] == (1, 0, 0):
        # Old i0=n_cpml+10, src=n_cpml+3: seven E nodes before x_lo.
        source_p = bounds.x_lo - (AUX_N_MARGIN_1D - AUX_SRC_OFFSET_1D)
    elif info["m"] == (-1, 0, 0):
        # Old n_tfsf includes x_hi+1: the right source is eight nodes past x_hi.
        source_p = -(bounds.x_hi + AUX_N_MARGIN_1D - AUX_SRC_OFFSET_1D + 1)
    else:
        q1, sigma = launch_window_width(info["m"])
        # The entire launch support must be upstream of the source-free box.
        source_p -= 4 * sigma
    source_n = np.rint(source_p - fractions[:3]).astype(np.int64)
    delays = (source_n + fractions[:3] - source_p) * (dx / np.linalg.norm(m)) / (299792458.0 * dt)
    if sigma:
        windows = launch_windows(m, fractions, source_p, sigma, dx, dt)
    lo = min(int(source_n.min()), *(int(t.source_index.min()) for t in tables_e + tables_h))
    hi = max(int(source_n.max()), *(int(t.source_index.max()) for t in tables_e + tables_h))
    if windows:
        lo = min(lo, *(int(w.nodes.min()) for w in windows))
        hi = max(hi, *(int(w.nodes.max()) for w in windows))
    padding = int(n_steps) * radius + 2
    origin, n_nodes = lo - padding, hi - lo + 1 + 2 * padding
    def translate(tables):
        return tuple(t._replace(source_index=(t.source_index - origin).astype(np.int32)) for t in tables)
    custom = waveform if isinstance(waveform, CustomWaveform) else None
    if custom is not None and theta_deg != 0:
        raise NotImplementedError("closed_box CustomWaveform requires theta_deg=0")
    if custom is None and waveform not in ("differentiated_gaussian", "modulated_gaussian", "continuous_wave"):
        raise ValueError("unsupported closed_box waveform")
    from rfx.sources.tfsf import pulse_timing
    tau, t0 = pulse_timing(waveform, f0, bandwidth)
    tau = source_scalar("tau", tau)
    t0 = source_scalar("t0", t0)
    dtype = jnp.result_type(float)
    # CPU Yee SI coefficient factorization: yee.py:1052-1057, 1221-1228;
    # H: yee.py:496-500. Keep the array operations, not Python-folded ratios.
    one, zero = jnp.asarray(1., dtype), jnp.asarray(0., dtype)
    ca, cb = e_update_coeffs(one, zero, dt)
    ch = dt / (one * MU_0)
    cfg = PlaneWaveConfig(tuple(map(int, m)), info, shape, bounds, origin, n_nodes,
                          int(n_steps), tuple(fractions), tuple(projections),
                          translate(tables_e), translate(tables_h),
                          tuple(map(int, source_n - origin)), tuple(map(float, delays)),
                          float(source_p), dx, dt, ca, cb, ch, amplitude, tau, t0, f0,
                          "custom" if custom is not None else waveform, custom,
                          tuple(w._replace(nodes=(w.nodes - origin).astype(np.int32))
                                for w in windows), sigma, q1,
                          theta_deg=theta_deg, phi_deg=phi_deg)
    fields = [jnp.zeros(n_nodes, dtype) for _ in range(6)]
    return cfg, FDTDState(*fields, jnp.asarray(0, jnp.int32))


def _line_shift(array, shift):
    """Read n+shift; zero extension outside the record's domain of dependence."""
    if shift > 0:
        return jnp.concatenate((array[shift:], jnp.zeros(shift, array.dtype)))
    if shift < 0:
        return jnp.concatenate((jnp.zeros(-shift, array.dtype), array[:shift]))
    return array


def line_difference(cfg, state, target, source, axis):
    half = cfg.m[axis] / 2
    offset = cfg.fractions[target] - cfg.fractions[source]
    field = getattr(state, COMPONENTS[source])
    return _line_shift(field, int(offset + half)) - _line_shift(field, int(offset - half))


def update_line_h(cfg, state, dx, dt):
    """yee.py:503-521: (Da/dx - Db/dx), then H - ch*curl, component by component."""
    changes = {}
    for target, (first, a, second, b) in enumerate(CURL_TERMS, 3):
        curl = (line_difference(cfg, state, target, first, a) / dx
                - line_difference(cfg, state, target, second, b) / dx)
        name = COMPONENTS[target]
        changes[name] = getattr(state, name) - cfg.ch * curl
    return tree_util.tree_map(lax.stop_gradient, state._replace(**changes))


def update_line_e(cfg, state, dx, dt, t=None):
    """yee.py:594-608,1224-1228; launch uses uniform.py:587's float32 step argument."""
    changes = {}
    for target, (first, a, second, b) in enumerate(CURL_TERMS):
        curl = (line_difference(cfg, state, target, first + 3, a) / dx
                - line_difference(cfg, state, target, second + 3, b) / dx)
        name = COMPONENTS[target]
        field = cfg.ca * getattr(state, name) + cfg.cb * curl
        weight = cfg.incidence["e_hat"][target]
        if weight != 0:
            window = cfg.source_windows[target] if cfg.source_windows else None
            delay = window.delays if window is not None else cfg.delays[target]
            time = (state.step.astype(jnp.float32) - jnp.float32(delay)) * dt
            from rfx.sources.tfsf import line_source_value
            value = line_source_value(cfg, time, cfg.src_amp * float(weight))
            if window is not None:
                value = value * jnp.asarray(window.weights, jnp.float32)
                field = field.at[jnp.asarray(window.nodes, jnp.int32)].add(value)
            else:
                field = field.at[cfg.source_nodes[target]].add(value)
        changes[name] = field
    return tree_util.tree_map(lax.stop_gradient, state._replace(**changes, step=state.step + 1))


def table_operands(cfg, incident, table, ti, magnetic):
    """What table ``ti`` reads this step: the carried line, or its face record."""
    if cfg.face_record is None:
        return getattr(incident, COMPONENTS[table.source_component])[table.source_index]
    side = 0 if magnetic else 1
    row = jnp.take(cfg.face_record[side], incident.step, axis=0,
                   mode="fill", fill_value=jnp.nan)
    return row[cfg.face_gathers[side][ti]]


def apply_tables(state, cfg, incident, *, magnetic):
    """Gather each commutator term and scatter-add after its grid update."""
    tables = cfg.tables_h if magnetic else cfg.tables_e
    for ti, table in enumerate(tables):
        name = COMPONENTS[table.target_component]
        values = table_operands(cfg, incident, table, ti, magnetic)
        field = getattr(state, name).reshape(-1)
        field = field.at[table.target_index].add(jnp.asarray(table.coefficient) * values)
        state = state._replace(**{name: field.reshape(cfg.shape)})
    return state


def tfsf_box_indices(sim) -> dict[str, tuple[int, int]]:
    """Realized inclusive bounds for the declared closed box or open slab."""
    if sim._tfsf is None:
        raise ValueError("No TFSF source registered")
    from rfx.sources.tfsf import tfsf_from_declaration, tfsf_injection_planes
    grid = sim._build_grid()
    cfg, _ = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, bounds_only=True)
    return tfsf_injection_planes(cfg)


def normal_incident_spectrum(cfg, state, n_steps, freqs, dt, reference_index):
    """Normal-only compatibility replay; oblique phase references belong to O1b."""
    from rfx.measurement.dft import transform
    from rfx.stepping.incident_line import record_incident_line

    if cfg.m not in ((1, 0, 0), (-1, 0, 0)):
        raise NotImplementedError("oblique incident spectrum requires O1b")
    if n_steps > cfg.n_steps:
        raise ValueError("incident replay exceeds the closed-box line's step-count bound")
    x = cfg.x_lo if reference_index is None else reference_index
    if (isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
            or not cfg.x_lo <= x <= cfg.x_hi):
        raise ValueError("reference_index must be an integer E-node index inside the TFSF slab")
    indices = tuple(int(np.floor(cfg.m[0] * x + cfg.projections[c])) - cfg.origin
                    for c in range(3))
    record = record_incident_line(cfg, n_steps, dt, indices, cfg.incidence["e_hat"])
    return np.asarray(transform(record, freqs, dt, kind="E",
                                time_shift=-1 if reference_index is None else 0))

@lru_cache(maxsize=256)
def rational_direction(theta_deg, phi_deg, direction, direction_tol_deg, max_component):
    """Exhaustive primitive triples, angle then length then lexicographic order."""
    if direction not in ("+x", "-x"):
        raise ValueError("direction must be '+x' or '-x'")
    if (isinstance(max_component, (bool, np.bool_))
            or not isinstance(max_component, (int, np.integer)) or max_component < 1):
        raise ValueError("max_component must be a positive integer")
    if not np.all(np.isfinite([theta_deg, phi_deg, direction_tol_deg])) or direction_tol_deg < 0:
        raise ValueError("angles and direction_tol_deg must be finite; tolerance must be nonnegative")
    if not 0 <= theta_deg < 90:
        raise ValueError("theta_deg must be in [0, 90); choose the half-space with direction")
    theta, phi = np.deg2rad([theta_deg, phi_deg])
    requested = np.array([(1 if direction == "+x" else -1) * np.cos(theta),
                          np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)])
    # Exact cardinal requests must not inherit sin(pi)'s roundoff octant.
    requested[np.abs(requested) < 8 * np.finfo(float).eps] = 0
    requested /= np.linalg.norm(requested)
    signs = np.sign(requested).astype(np.int64)
    side = np.arange(max_component + 1, dtype=np.int64)
    yz = np.stack(np.meshgrid(side, side, indexing="ij"), axis=-1).reshape(-1, 2)
    best = None
    for x in side:
        candidates = np.column_stack((np.full(len(yz), x), yz)) * signs
        candidates = candidates[np.gcd.reduce(np.abs(candidates), axis=1) == 1]
        if not len(candidates):
            continue
        norm2 = np.sum(candidates * candidates, axis=1)
        angles = np.arctan2(np.linalg.norm(np.cross(candidates, requested), axis=1),
                            candidates @ requested)
        index = np.lexsort((candidates[:, 2], candidates[:, 1], candidates[:, 0], norm2, angles))[0]
        key = (float(angles[index]), int(norm2[index]), tuple(map(int, candidates[index])))
        if best is None or key < best:
            best = key
    error = float(np.rad2deg(best[0]))
    if error > direction_tol_deg:
        raise ValueError(f"rational direction angle error {error:.15g} deg exceeds direction_tol_deg={direction_tol_deg}")
    m = best[2]
    return m, tuple(np.asarray(m) / np.sqrt(best[1])), tuple(requested), error


def incidence(theta_deg=0.0, phi_deg=0.0, direction="+x", polarization="ez",
              direction_tol_deg=0.1, max_component=100):
    """Float64 declaration vectors, always formed against the realized k."""
    if polarization not in ("ey", "ez", "s", "p"):
        raise ValueError("polarization must be 'ey', 'ez', 's', or 'p'")
    if theta_deg == 0 and polarization in ("ey", "ez") and phi_deg != 0:
        raise ValueError("nonzero phi_deg at theta_deg=0 requires polarization='s' or 'p'")
    if theta_deg != 0 and polarization not in ("s", "p"):
        raise ValueError("nonzero theta_deg requires polarization='s' or 'p'")
    m, k, requested, error = rational_direction(
        theta_deg, phi_deg, direction, direction_tol_deg, max_component)
    k = np.array(k, dtype=np.float64)
    if m[1:] == (0, 0):
        phi = np.deg2rad(phi_deg if polarization in ("s", "p") else 0.0)
        s = np.array([0., -np.sin(phi), np.cos(phi)])
        p = np.array([0., np.cos(phi), np.sin(phi)])
    else:
        s = np.cross([1., 0., 0.], k)
        s /= np.linalg.norm(s)
        p = np.cross(s, k)
    basis = np.stack((k, s, p))
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-12, rtol=0):
        raise ValueError("realized polarization basis is not orthonormal to 1e-12")
    e = s if polarization in ("ez", "s") else p
    return dict(m=m, k_hat=k, e_hat=e, h_hat=np.cross(k, e),
                requested_k_hat=np.array(requested, dtype=np.float64),
                angle_error_deg=error, axis=direction)


def tfsf_incidence(self):
    """Return the closed box's realized direction and unit polarization vectors."""
    entry = self._tfsf
    if entry is None or not entry.closed_box:
        raise ValueError("tfsf_incidence requires a declared closed_box=True source")
    return incidence(entry.theta_deg, entry.phi_deg, entry.direction, entry.polarization,
                     entry.direction_tol_deg, entry.max_component)


def box_bounds(shape, cpml_layers, margin):
    lo = cpml_layers + margin
    bounds = {a: (lo, n - lo - 1) for a, n in zip("xyz", shape)}
    if any(low <= 0 or low >= high or high >= n - 1
           for (low, high), n in zip(bounds.values(), shape)):
        raise ValueError(f"TFSF margin/cpml_layers are too large for the closed box: shape={shape}")
    return SimpleNamespace(closed_box=True, **{
        a + suffix: value for a, pair in bounds.items() for suffix, value in zip(("_lo", "_hi"), pair)})


def validate_declaration(sim, theta, phi, tol, cap, pol, direction, angle, waveform, margin):
    """Closed-box admission; final realized operators are checked again at execution."""
    from rfx.sources.sources import CustomWaveform
    if angle != 0:
        raise NotImplementedError("closed_box requires normal incidence angle_deg=0; use theta_deg/phi_deg")
    if (sim._uses_nonuniform_mesh or sim._refinement is not None or sim._mode != "3d"
            or sim._solver != "yee" or sim._stencil_order != 2):
        raise NotImplementedError("CustomWaveform and closed_box TFSF require a uniform, "
                                  "3D second-order Yee run/forward with no subgrids")
    if any(token != "cpml" for _, _, token in sim._boundary_spec.faces()):
        raise NotImplementedError("closed_box TFSF requires CPML on all six domain faces")
    if theta != 0 and isinstance(waveform, CustomWaveform):
        raise NotImplementedError("closed_box CustomWaveform requires theta_deg=0")
    incidence(theta, phi, direction, pol, tol, cap)
    grid = sim._build_grid()
    cell = float(np.asarray(grid.cells("x"))[0])
    if any(not np.all(np.asarray(grid.cells(a)) == cell) for a in "xyz"):
        raise NotImplementedError("closed_box requires a cubic uniform grid (dx=dy=dz)")
    cfg = box_bounds(grid.shape, grid.cpml_layers, margin)
    # Declaration checks read cells only; sheets and wires are checked by the
    # final realized-operator check at execution.
    materials, debye, lorentz, pec, _, _, kerr = sim._assemble_materials(
        grid, pec_sheets=[], pec_wires=[])
    masks = [v for v in (pec, kerr) if v is not None]
    for spec in (debye, lorentz):
        if spec is not None:
            masks.extend(spec[1])
    sim._validate_tfsf_vacuum_boundary(materials, cfg, nonvacuum_masks=tuple(masks))

@dataclass(frozen=True)
class _TFSFEntry:
    f0: float | None
    bandwidth: float
    amplitude: float
    margin: int
    polarization: str
    direction: str
    angle_deg: float
    waveform: object = "differentiated_gaussian"
    method: str = "bloch"
    closed_box: bool = False
    theta_deg: float = 0.0
    phi_deg: float = 0.0
    direction_tol_deg: float = 0.1
    max_component: int = 100


def add_tfsf_source(
    self,
    *,
    f0: float | None = None,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    margin: int = 3,
    polarization: str = "ez",
    direction: str = "+x",
    angle_deg: float = 0.0,
    waveform: object = None,
    method: str = "bloch",
    closed_box: bool = False,
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    direction_tol_deg: float = 0.1,
    max_component: int = 100,
) -> "Simulation":
    """Add a plane wave; closed boxes accept rational oblique incidence.

    For closed boxes, theta_deg and phi_deg request a direction about x;
    tfsf_incidence() exposes its realized direction and polarization.
    Nonzero theta requires s or p polarization. direction_tol_deg is a
    refusal threshold; max_component bounds the exhaustive integer search.

    The slab (``closed_box=False``) keeps a narrow scope: x-directed
    propagation, ``ez``/``ey`` polarization, 3D mode and CPML along x;
    ``angle_deg`` tilts it only in the transverse plane implied by the
    chosen polarization.

    Parameters
    ----------
    closed_box : bool
        Use a finite six-face total-field box with CPML on all six
        faces, uniform 3D Yee with cubic cells only. Normal or oblique
        incidence (``theta_deg``/``phi_deg``, ``s`` or ``p``): the box
        carries the lattice plane wave along the realized rational
        direction. No incident spectrum is formed for an oblique box
        (``measure_normal_incident_spectrum`` refuses it). The default keeps
        the historical x slab with periodic y/z. Read the realized
        inclusive node bounds with :meth:`tfsf_box_indices`.
    waveform : {"differentiated_gaussian", "modulated_gaussian", "continuous_wave"}
        Pulse shape injected into the 1D auxiliary grid. (``continuous_wave``
        is only supported at normal incidence — ``angle_deg == 0``.)

        The default ``None`` resolves at declaration to DG on normal,
        closed-box and method-B paths, and MG on oblique Bloch. Explicit
        DG on oblique Bloch is refused: its fixed transverse wavenumber
        carries only an MG around f0. Use ``modulated_gaussian``, leave
        the waveform unset, or use ``method="methodB"`` for an open domain.

        ``differentiated_gaussian`` (legacy rfx): baseband
        pulse with no carrier, DC-free, spectrum peaks near
        ``f0·bandwidth``. Good for broadband measurements but
        spectrum extends well below ``f0``.

        ``continuous_wave``: a single-frequency sinusoid at ``f0`` with a
        raised-cosine turn-on (~8 periods), then constant amplitude. Use for
        steady-state / narrowband measurements that need a well-defined
        time-average ``⟨E²⟩ = A²/2`` (e.g. the quantitative Kerr SPM oracle).
        Not for broadband S-parameters — it excites only ``f0``.

        ``modulated_gaussian``: carrier-modulated Gaussian matching
        Meep's ``GaussianSource(frequency=f0, fwidth=f0·bandwidth)``.
        Spectrum is a Gaussian centered at ``f0`` with 1/e
        half-width ``f0·bandwidth``. Use this for matched rfx-vs-Meep
        crossval comparisons.

        A ``CustomWaveform`` may supply a fixed, JAX-compatible real
        scalar function of time, multiplied by ``amplitude``. It is
        added to the auxiliary E node at ``t=n*dt``; it is not the
        launched E amplitude. Normal incidence only; use a fixed
        record length (``until_decay`` has no source-off contract).
        These source inputs support ``forward`` and its internal
        checkpoints, including external JAX transformations through
        grad/value_and_grad/JVP/vmap/jit/checkpoint/scan. ``run`` refuses
        external transforms; differentiate ``forward`` instead.
    method : {"bloch", "methodB"}
        Oblique-incidence engine (ignored for ``angle_deg=0``, which always
        uses the normal 1D-aux path). ``"bloch"`` (default) is the narrowband
        complex-Bloch 2D-aux path (fields go complex64; frequency-domain
        monitors are not transform-aware — see ``tfsf_2d.py``). ``"methodB"``
        is the open-domain oblique path (real ``float32``, 1D-aux-along-k̂ +
        4-edge box): the transverse axis becomes OPEN/CPML, z stays thin
        periodic (2.5-D, z-invariant far field), and NTFF/DFT/flux monitors
        read physical fields. ``"methodB"`` currently requires
        ``polarization='ez'`` and a single-device uniform grid.
    """
    from rfx.api import _bloch_cutoff_level_db
    from rfx.sources.tfsf import resolve_waveform
    if closed_box:
        validate_declaration(self, theta_deg, phi_deg, direction_tol_deg,
                             max_component, polarization, direction, angle_deg, waveform, margin)
    elif (theta_deg, phi_deg, direction_tol_deg, max_component) != (0.0, 0.0, 0.1, 100):
        raise ValueError("theta_deg/phi_deg/direction_tol_deg/max_component require closed_box=True")
    waveform = resolve_waveform(waveform, angle_deg, method)
    from rfx.boundaries.tfsf import validate_source_boundary
    validate_source_boundary(self, closed_box=closed_box)
    if self._cpml_layers <= 0:
        raise ValueError("TFSF plane-wave source requires cpml_layers > 0")
    if self._mode not in ("3d", "2d_tmz", "2d_tez"):
        raise ValueError(
            f"TFSF plane-wave source requires mode='3d', '2d_tmz', or '2d_tez', got {self._mode!r}"
        )
    transverse = "" if closed_box else (
        "z" if method == "methodB" and abs(angle_deg) > .01 else "yz")
    if any(axis not in transverse for axis in self._periodic_axes):
        raise ValueError(
            "TFSF plane-wave source conflicts with periodic-axis overrides "
            f"{self._periodic_axes!r}; this source permits declared periodic "
            f"axes only in {transverse!r}")
    if self._ports:
        raise ValueError(
            "TFSF plane-wave source is not supported together with lumped ports"
        )
    if f0 is not None and f0 <= 0:
        raise ValueError(f"f0 must be positive when provided, got {f0}")
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    if margin < 1:
        raise ValueError(f"margin must be >= 1, got {margin}")
    if polarization not in (("ez", "ey", "s", "p") if closed_box else ("ez", "ey")):
        raise ValueError(f"polarization must be 'ez' or 'ey', got {polarization!r}")
    if direction not in ("+x", "-x"):
        raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
    if abs(angle_deg) >= 90.0:
        raise ValueError(f"abs(angle_deg) must be < 90, got {angle_deg}")
    if waveform == "continuous_wave" and abs(angle_deg) != 0.0:
        raise ValueError(
            "waveform='continuous_wave' is only supported at normal incidence "
            f"(angle_deg=0); got angle_deg={angle_deg}"
        )
    from rfx.sources.sources import CustomWaveform
    custom = isinstance(waveform, CustomWaveform)
    if not custom and waveform not in ("differentiated_gaussian", "modulated_gaussian", "continuous_wave"):
        raise ValueError(
            "waveform must be 'differentiated_gaussian', 'modulated_gaussian', "
            f"'continuous_wave', or CustomWaveform, got {waveform!r}"
        )
    if (custom or closed_box) and angle_deg != 0.0:
        raise NotImplementedError("CustomWaveform and closed_box require normal incidence (angle_deg=0)")
    if (custom or closed_box) and self._mode != "3d":
        raise NotImplementedError("CustomWaveform and closed_box require mode='3d'")
    if method not in ("bloch", "methodB"):
        raise ValueError(f"method must be 'bloch' or 'methodB', got {method!r}")
    if method == "methodB":
        if abs(angle_deg) <= 0.01:
            raise ValueError(
                "method='methodB' is the open-domain OBLIQUE path; use a "
                "non-zero angle_deg (angle_deg=0 uses the normal 1D-aux path)"
            )
        if polarization != "ez":
            raise NotImplementedError(
                "method='methodB' currently supports polarization='ez' "
                "(transverse=y) only; 'ey' is future work"
            )

    if method == "bloch" and abs(angle_deg) > 0.01:  # the 2D-aux dispatch test in tfsf.py
        cutoff_db = _bloch_cutoff_level_db(angle_deg, bandwidth)
        if cutoff_db > -60.0:
            import warnings

            source_f0 = f0 if f0 is not None else self._freq_max / 2
            cutoff_ghz = source_f0 * abs(math.sin(math.radians(angle_deg))) / 1e9
            warnings.warn(
                f"Bloch TF/SF angle={angle_deg:g} deg, bandwidth={bandwidth:g}: "
                f"{cutoff_db:.2f} dB relative to f0 at f_c={cutoff_ghz:.6g} GHz; "
                "a rectangular-window DFT of this path's probe time series picks up "
                "the non-decaying component at f_c = f0·sinθ, so a value at f0 "
                "depends on the record length. Narrow the bandwidth, or taper "
                "the tail of the time series before the DFT.",
                AccuracyWarning,
                stacklevel=2,
            )

    self._tfsf = _TFSFEntry(
        f0=f0,
        bandwidth=bandwidth,
        amplitude=amplitude,
        margin=margin,
        polarization=polarization,
        direction=direction,
        angle_deg=angle_deg,
        waveform=waveform,
        method=method,
        closed_box=closed_box, theta_deg=theta_deg, phi_deg=phi_deg,
        direction_tol_deg=direction_tol_deg, max_component=max_component,
    )
    mesh = self._declared_mesh
    if mesh["_dx"] is not None and all(mesh[name] is None for name in (
            "_dx_profile", "_dy_profile", "_dz_profile")):
        from rfx.boundaries.tfsf import require_tfsf_side_faces
        require_tfsf_side_faces(self)
    return self


def document_fields(_NUM, _INT, _STR, _BOOL, _opt):
    """Serializable declaration, including the realized-direction inputs."""
    return {
        "f0": _opt(_NUM),
        "bandwidth": _NUM,
        "amplitude": _NUM,
        "margin": _INT,
        "polarization": _STR,
        "direction": _STR,
        "angle_deg": _NUM,
        "waveform": _STR,
        # PR #461: transverse boundary-value-problem selector — 'bloch' runs
        # periodic=(F,T,T)/cpml_axes="x" (laterally periodic: infinite array),
        # 'methodB' runs periodic=(F,F,T)/cpml_axes="xy" (open transverse
        # boundary: compact isolated scatterer). Design state, not run control:
        # the two lanes describe different physical structures. The labels are
        # rfx-internal; a future TFSF emitter should translate the semantic
        # (transverse boundary = periodic vs absorbing), not hunt for another
        # solver's "method B".
        "method": _STR,
        "closed_box": _BOOL,
        "theta_deg": _NUM,
        "phi_deg": _NUM,
        "direction_tol_deg": _NUM,
        "max_component": _INT,
    }
