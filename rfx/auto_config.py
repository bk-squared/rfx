"""Automatic simulation configuration from geometry + frequency range.

Derives all simulation parameters (dx, domain, CPML, n_steps, source)
from user-specified geometry and frequency band. Based on validated rules
from Meep/OpenEMS best practices and rfx convergence testing.

Usage
-----
>>> config = auto_configure(
...     geometry=[(Box(...), "FR4"), (Box(...), "pec")],
...     freq_range=(1e9, 4e9),
...     accuracy="standard",
... )
>>> sim = Simulation(**config.to_sim_kwargs())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from rfx.core.yee import EPS_0  # canonical 8.854187817e-12 (Stage 3.5b)

C0 = 299792458.0


# ---------------------------------------------------------------------------
# Auto-config planning heuristic — ONE shape rule, ONE byte rule (#696)
#
# Both halves of the memory budget go through these: the DECISION half
# (``auto_configure``'s ``max_memory_mb`` coarsening loop, which picks dx)
# and the REPORTING half (``SimConfig.grid_shape`` /
# ``SimConfig.estimated_memory_mb``, which is what the user reads and what
# the documented postcondition ``estimated_memory_mb <= max_memory_mb`` is
# written against).
#
# They used to be two hand-written copies that disagreed twice over: the
# loop counted z as ``ceil(domain_z/dx)`` while the property counted the
# ``dz_profile``, and the loop's byte formula omitted the property's NTFF
# and dispersion terms. So the loop stopped coarsening at a number the
# property never reproduced, and the postcondition failed on boards that
# need a non-uniform z. Measured on a 40x30mm FR4 board with a 0.508 mm
# substrate and a 35 um trace, budget 500 MB: the loop accepted
# dx = 4.0 mm, and the config it returned reported 653.1 MB.
# ---------------------------------------------------------------------------

def _auto_grid_shape(domain, dx, cpml_layers, dz_profile=None):
    """Planning grid shape ``(nx, ny, nz)`` including boundary pads.

    ``+ 1 + 2*cpml_layers`` per axis is the uniform :class:`~rfx.grid.Grid`
    rule. When ``dz_profile`` is set the z axis is counted from the PROFILE
    instead — ``len(profile) + 1 + 2*cpml_layers``, which is what
    :class:`~rfx.nonuniform.NonUniformGrid` builds. A graded stackup
    normally refines below ``dx``, so ``ceil(domain_z/dx)`` describes a
    uniform grid this configuration will never build AND under-counts it.

    This stays the coarse pre-``Simulation`` heuristic: it does not model
    per-face pads or 2-D mode. ``Simulation._ad_memory_static_accounting``
    reads the built grid and is the accurate one; see its docstring.
    """
    nx = int(math.ceil(domain[0] / dx)) + 1 + 2 * cpml_layers
    ny = int(math.ceil(domain[1] / dx)) + 1 + 2 * cpml_layers
    if dz_profile is not None:
        nz = int(len(dz_profile)) + 1 + 2 * cpml_layers
    else:
        nz = int(math.ceil(domain[2] / dx)) + 1 + 2 * cpml_layers
    return (nx, ny, nz)


def _auto_memory_mb(cells: int, cpml_layers: int) -> float:
    """Planning memory heuristic in megabytes for ``cells`` cells.

    6 field + 6 material float32 arrays, ~24 CPML psi arrays over ~15 % of
    the domain, an NTFF surface-DFT term, 4 dispersion auxiliaries, and a
    10x reverse-mode-AD factor. See
    :attr:`SimConfig.estimated_memory_mb` for what each term stands for
    and why this is not the production AD-memory contract.
    """
    cells = int(cells)
    base_bytes = cells * 12 * 4
    cpml_bytes = int(cells * 0.15 * 24 * 4) if cpml_layers > 0 else 0
    ntff_bytes = int(6 * (cells ** 0.67) * 10 * 4 * 8)   # ~10 freqs estimate
    disp_bytes = cells * 4 * 4
    return (base_bytes + cpml_bytes + ntff_bytes + disp_bytes) * 10 / 1e6


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------

class FeatureInfo(NamedTuple):
    """Geometry feature summary for auto-configuration."""
    min_thickness: float      # thinnest dimension (metres)
    max_extent: float         # largest dimension (metres)
    bbox: tuple               # ((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))
    max_eps_r: float          # highest relative permittivity
    has_pec: bool             # any PEC geometry present
    estimated_Q: float        # estimated cavity Q from material loss
    z_features: list = []     # list of (z_lo, z_hi, eps_r) for z-grading


def analyze_features(geometry: list, materials: dict, pec_threshold: float = 1e6) -> FeatureInfo:
    """Extract critical dimensions and material properties from geometry.

    Parameters
    ----------
    geometry : list of (Shape, material_name) tuples
    materials : dict of material_name -> MaterialSpec or dict
    pec_threshold : sigma above which material is PEC
    """
    thicknesses = []
    extents = []
    all_corners_lo = []
    all_corners_hi = []
    max_eps_r = 1.0
    has_pec = False
    max_loss_tangent = 0.0
    z_features = []  # (z_lo, z_hi, eps_r) for non-uniform z detection

    for shape, mat_name in geometry:
        mat = materials.get(mat_name, {})
        sigma = mat.get("sigma", 0.0) if isinstance(mat, dict) else getattr(mat, "sigma", 0.0)
        eps_r = mat.get("eps_r", 1.0) if isinstance(mat, dict) else getattr(mat, "eps_r", 1.0)

        if sigma >= pec_threshold:
            has_pec = True
        else:
            max_eps_r = max(max_eps_r, eps_r)
            if sigma > 0 and eps_r > 1:
                max_loss_tangent = max(max_loss_tangent, sigma / eps_r)

        # Use bounding_box() if available (all shapes), fall back to corner_lo/hi
        if hasattr(shape, "bounding_box"):
            try:
                c1, c2 = shape.bounding_box()
            except NotImplementedError:
                continue
        elif hasattr(shape, "corner_lo") and hasattr(shape, "corner_hi"):
            c1, c2 = shape.corner_lo, shape.corner_hi
        else:
            continue

        dims = [abs(c2[i] - c1[i]) for i in range(3)]
        nonzero_dims = [d for d in dims if d > 1e-12]
        if nonzero_dims:
            thicknesses.append(min(nonzero_dims))
            extents.append(max(dims))
        all_corners_lo.append(c1)
        all_corners_hi.append(c2)

        # Track z-extent for non-uniform mesh detection
        z_lo, z_hi = min(c1[2], c2[2]), max(c1[2], c2[2])
        z_thickness = z_hi - z_lo
        if z_thickness > 1e-12 and sigma < pec_threshold:
            z_features.append((z_lo, z_hi, eps_r))

    # Bounding box of all geometry
    if all_corners_lo:
        bbox_lo = tuple(min(c[i] for c in all_corners_lo) for i in range(3))
        bbox_hi = tuple(max(c[i] for c in all_corners_hi) for i in range(3))
    else:
        bbox_lo = (0, 0, 0)
        bbox_hi = (0.01, 0.01, 0.01)

    min_thickness = min(thicknesses) if thicknesses else 0.001
    max_extent = max(extents) if extents else 0.01

    estimated_Q = max_loss_tangent

    return FeatureInfo(
        min_thickness=min_thickness,
        max_extent=max_extent,
        bbox=(bbox_lo, bbox_hi),
        max_eps_r=max_eps_r,
        has_pec=has_pec,
        estimated_Q=estimated_Q,
        z_features=z_features,
    )


# ---------------------------------------------------------------------------
# Auto-configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Auto-derived simulation configuration."""
    dx: float
    domain: tuple[float, float, float]
    cpml_layers: int
    n_steps: int
    freq_range: tuple[float, float]
    margin: float
    dt: float
    accuracy: str
    source_type: str = "j_source"   # "j_source" or "raw"
    source_info: str = ""           # Human-readable explanation
    warnings: list[str] = field(default_factory=list)
    dz_profile: np.ndarray | None = None
    boundary: str = "cpml"

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Estimated grid shape (nx, ny, nz) including CPML padding.

        Issue #696: when ``dz_profile`` is set the z axis is counted from
        the PROFILE, not from ``ceil(domain_z / dx)``. This property
        described the UNIFORM grid even for a configuration whose whole
        point is a graded z stackup, and a graded stackup normally
        refines below ``dx`` — so the cell count, and every memory number
        derived from it, came out LOW for exactly the meshes people size
        GPUs for. The z term now matches ``NonUniformGrid``'s
        ``len(profile) + pads + 1``.
        """
        return _auto_grid_shape(self.domain, self.dx, self.cpml_layers,
                                self.dz_profile)

    @property
    def estimated_memory_mb(self) -> float:
        """Coarse auto-configuration memory heuristic in megabytes.

        This property is used while choosing a draft mesh from geometry
        before a ``Simulation`` object exists. It intentionally remains a
        conservative planning heuristic, not the calibrated AD-memory
        contract for production runs.

        For user-facing memory reports and inverse-design sizing, prefer
        ``Simulation.estimate_ad_memory(...)`` or
        ``Simulation.mesh_intelligence_report(...)``. Those APIs know the
        final grid/profiles, NTFF state, and segmented scan-of-scan
        checkpointing semantics from issue #31.

        Accounts for:
        - 6 field arrays (Ex, Ey, Ez, Hx, Hy, Hz)
        - ~6 material coefficient arrays (eps_r, sigma, mu_r, Ca, Cb, etc.)
        - CPML auxiliary fields (~24 psi arrays in absorbing region)
        - NTFF DFT accumulators (6 faces × n_freqs × face_cells × 4 components)
        - Debye/Lorentz auxiliary polarization fields (2 per pole)
        - ~10x overhead for reverse-mode AD through jax.lax.scan
          (XLA compilation buffers, reverse-mode tape for scan unrolling,
          CPML auxiliary arrays replicated for AD, jax.checkpoint reduces
          but does not eliminate this)

        Returns
        -------
        float
            Estimated memory in megabytes.
        """
        nx, ny, nz = self.grid_shape
        return _auto_memory_mb(nx * ny * nz, self.cpml_layers)

    @property
    def cells_per_wavelength(self) -> float:
        f_max = self.freq_range[1]
        lambda_min = C0 / f_max
        return lambda_min / self.dx

    @property
    def sim_time_ns(self) -> float:
        return self.n_steps * self.dt * 1e9

    @property
    def uses_nonuniform(self) -> bool:
        return self.dz_profile is not None

    def to_sim_kwargs(self) -> dict:
        """Convert to keyword arguments for Simulation constructor."""
        kwargs = {
            "freq_max": self.freq_range[1],
            "domain": self.domain,
            "boundary": self.boundary,
            "cpml_layers": self.cpml_layers,
            "dx": self.dx,
        }
        if self.dz_profile is not None:
            kwargs["dz_profile"] = self.dz_profile
        return kwargs

    def summary(self) -> str:
        lines = [
            f"SimConfig (accuracy={self.accuracy!r}):",
            f"  dx = {self.dx*1e3:.3f} mm ({self.cells_per_wavelength:.0f} cells/λ_min)",
            f"  domain = {self.domain[0]*1e3:.1f} × {self.domain[1]*1e3:.1f} × {self.domain[2]*1e3:.1f} mm",
            f"  cpml = {self.cpml_layers} layers ({self.cpml_layers*self.dx*1e3:.1f} mm)",
            f"  n_steps = {self.n_steps} ({self.sim_time_ns:.1f} ns)",
            f"  freq = {self.freq_range[0]/1e9:.2f} – {self.freq_range[1]/1e9:.2f} GHz",
            f"  source = {self.source_type} ({self.source_info})" if self.source_info else f"  source = {self.source_type}",
        ]
        if self.dz_profile is not None:
            dz_min = np.min(self.dz_profile) * 1e3
            dz_max = np.max(self.dz_profile) * 1e3
            lines.append(f"  dz = {dz_min:.3f} – {dz_max:.3f} mm ({len(self.dz_profile)} cells, non-uniform)")
        if self.warnings:
            lines.append("  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def auto_configure(
    geometry: list,
    freq_range: tuple[float, float],
    materials: dict | None = None,
    accuracy: str = "standard",
    *,
    boundary: str = "cpml",
    dx_override: float | None = None,
    margin_override: float | None = None,
    n_steps_override: int | None = None,
    max_memory_mb: float | None = None,
    waveform=None,
) -> SimConfig:
    """Derive all simulation parameters from geometry + frequency range.

    Parameters
    ----------
    geometry : list of (Shape, material_name) tuples
    freq_range : (f_min, f_max) in Hz
    materials : dict of material definitions (name -> {eps_r, sigma, ...})
    accuracy : "draft", "standard", or "high"
        draft:    10 cells/λ, 0.15λ margin  (fast, ~10% error)
        standard: 20 cells/λ, 0.25λ margin  (~3% error)
        high:     40 cells/λ, 0.50λ margin  (~1% error)
    boundary : "cpml" or "pec"
        Boundary condition type. Determines source type recommendation:
        - "cpml": J-source with Cb/dV normalization (Meep convention)
        - "pec": raw E-field source (no Cb normalization needed)
    dx_override : force specific cell size
    margin_override : force specific margin
    n_steps_override : force specific step count
    max_memory_mb : float or None
        GPU memory budget in megabytes.  When set, ``dx`` is
        automatically coarsened (increased) until
        ``estimated_memory_mb <= max_memory_mb``.  Useful for
        keeping reverse-mode AD within a 24 GB GPU.  Ignored when
        *dx_override* is provided.

        Practical limits (approximate, with gradient checkpointing):

        - 8 GB GPU:  ``max_memory_mb=6000``
        - 24 GB GPU: ``max_memory_mb=18000``
        - 48 GB GPU: ``max_memory_mb=36000``
    waveform : source waveform or None
        Used only for the source-time term of the ``n_steps`` estimate:
        when the waveform exposes ``t0`` (``GaussianPulse`` /
        ``ModulatedGaussian``: ``t0 = cutoff*tau``), the source window is
        ``2*t0`` so a longer-onset pulse (e.g. ``cutoff=4.5`` →
        ``9*tau``) is fully covered. Without a waveform the historical
        cutoff=3 envelope (``6*tau`` at bw=0.8) is used, bitwise-identical
        to the previous hardcode.

    Returns
    -------
    SimConfig with all derived parameters.
    """
    if materials is None:
        materials = {}

    f_min, f_max = freq_range
    if f_min <= 0 or f_max <= f_min:
        raise ValueError(f"freq_range must be (f_min, f_max) with 0 < f_min < f_max, got {freq_range}")

    lambda_min = C0 / f_max
    lambda_max = C0 / f_min
    f_center = (f_min + f_max) / 2

    # Accuracy presets
    presets = {
        "draft":    {"cpw": 10, "cpf": 2, "margin_frac": 0.15, "pml_frac": 0.10},
        "standard": {"cpw": 20, "cpf": 4, "margin_frac": 0.25, "pml_frac": 0.20},
        "high":     {"cpw": 40, "cpf": 8, "margin_frac": 0.50, "pml_frac": 0.15},
    }
    if accuracy not in presets:
        raise ValueError(f"accuracy must be 'draft', 'standard', or 'high', got {accuracy!r}")
    preset = presets[accuracy]

    # Analyze geometry
    features = analyze_features(geometry, materials)
    warnings = []

    # Detect empty geometry — use wavelength-only sizing to avoid oversized grids
    _empty_geometry = len(geometry) == 0

    if _empty_geometry:
        import warnings as _w
        _w.warn(
            "auto_configure called with empty geometry — domain and dx are "
            "derived from frequency range only. Add geometry before running.",
            stacklevel=2,
        )
        warnings.append(
            "Empty geometry: domain uses lambda_min/10 cell size and minimal "
            "margins. Add geometry for accurate auto-configuration."
        )

    # 1. Cell size
    # λ_min in highest-eps medium
    lambda_min_medium = lambda_min / math.sqrt(features.max_eps_r)
    dx_wavelength = lambda_min_medium / preset["cpw"]

    if _empty_geometry:
        # No features to resolve — use coarser wavelength-only sizing
        # (lambda_min/10 regardless of accuracy) to keep the grid small.
        dx = _round_dx(lambda_min / 10)
    else:
        dx_feature = features.min_thickness / preset["cpf"] if features.min_thickness > 0 else dx_wavelength
        dx = min(dx_wavelength, dx_feature)

    if dx_override is not None:
        dx = dx_override
    else:
        # Round to nice value
        dx = _round_dx(dx)

    # Check feature resolution — auto non-uniform z when thin z-features exist
    # Compare against wavelength-based dx (not the feature-refined dx) to detect
    # cases where non-uniform z saves computation vs. globally finer mesh.
    needs_nonuniform_z = False
    if features.z_features:
        dx_wave_rounded = _round_dx(dx_wavelength)
        for z_lo, z_hi, _ in features.z_features:
            z_thick = z_hi - z_lo
            if z_thick > 0 and z_thick / dx_wave_rounded < 4:
                needs_nonuniform_z = True
                # Use coarser wavelength-based dx for xy (non-uniform z handles thin features)
                if dx_override is None:
                    dx = dx_wave_rounded
                break

    if not _empty_geometry and features.min_thickness > 0 and features.min_thickness / dx < 2 and not needs_nonuniform_z:
        warnings.append(
            f"Thinnest feature ({features.min_thickness*1e3:.2f} mm) has only "
            f"{features.min_thickness/dx:.1f} cells — consider finer dx or subgridding"
        )

    # 2. Domain margins
    if _empty_geometry:
        # Minimal margin — just enough for a small placeholder domain.
        margin = lambda_min * 0.1
    else:
        margin = lambda_max * preset["margin_frac"]
    if margin_override is not None:
        margin = margin_override

    bbox_lo, bbox_hi = features.bbox
    domain = tuple(
        (bbox_hi[i] - bbox_lo[i]) + 2 * margin
        for i in range(3)
    )
    # Ensure minimum domain > 4*dx per dimension
    domain = tuple(max(d, 8 * dx) for d in domain)

    # 3. CPML layers — evidence-based from reflectivity sweep (2026-04-07).
    # 56-run sweep (7 layers × 4 freqs × 2 kappa): ALL achieve < -40 dB.
    # Standard CPML (kappa_max=1.0) >= CFS-CPML (kappa_max=5.0).
    # Sweet spot at 8 layers for 1 GHz (-49.7 dB); 4 layers sufficient everywhere.
    # Conservative floor per accuracy tier plus pml_frac * lambda_max scaling.
    min_cpml_thickness = preset["pml_frac"] * lambda_max  # 0.10/0.20/0.40
    min_cpml_cells = int(np.ceil(min_cpml_thickness / dx))
    cpml_cells = {"draft": 6, "standard": 8, "high": 12}[accuracy]
    cpml_layers = max(min_cpml_cells, cpml_cells)

    # 3b. Memory budget — coarsen dx until estimated memory fits.
    # The physical domain extent is fixed (bbox + margins); only the
    # cell count changes as dx grows.  The 8*dx floor prevents
    # degenerate tiny grids but is re-applied each iteration.
    if max_memory_mb is not None and dx_override is None:
        # Store the fixed physical extent so it doesn't inflate with dx.
        _phys_extent = tuple(
            (bbox_hi[i] - bbox_lo[i]) + 2 * margin for i in range(3)
        )
        # #696: the z extent the profile spans. ``margin`` and the geometry
        # are already fixed here, so this is dx-independent and the profile
        # below is a pure function of the candidate dx — the same call
        # step 4 makes with the dx this loop settles on.
        _phys_z = (max(f[1] for f in features.z_features) + margin
                   if needs_nonuniform_z else None)
        for _ in range(20):  # safety limit on iterations
            _domain = tuple(max(d, 8 * dx) for d in _phys_extent)
            _cpml_cells_now = int(np.ceil(min_cpml_thickness / dx))
            _cpml = max(_cpml_cells_now, cpml_cells)
            # #696: measure the candidate against the grid this config
            # will actually build, with the byte rule the returned
            # SimConfig reports. Counting z as ceil(domain_z/dx) and
            # dropping the NTFF/dispersion terms made the loop stop at a
            # number ``estimated_memory_mb`` never reproduced, so the
            # documented postcondition failed on non-uniform-z boards.
            _dz_prof = (_make_dz_profile(features.z_features, _phys_z, dx)
                        if needs_nonuniform_z else None)
            _nx, _ny, _nz = _auto_grid_shape(_domain, dx, _cpml, _dz_prof)
            _cells = _nx * _ny * _nz
            _est_mb = _auto_memory_mb(_cells, _cpml)
            if _est_mb <= max_memory_mb:
                # Accept this dx and update domain/cpml to match
                domain = _domain
                cpml_layers = _cpml
                break
            # Coarsen dx.  _round_dx rounds DOWN, so we must ensure
            # the rounded result is strictly larger than the old dx.
            old_dx = dx
            dx = _round_dx(dx * 1.5)
            if dx <= old_dx:
                dx = _round_dx(dx * 2.0)
            if dx <= old_dx:
                # Force a jump to the next power-of-ten tier
                dx = old_dx * 2.0
        else:
            # Loop exhausted — apply final values and warn
            domain = _domain
            cpml_layers = _cpml
            warnings.append(
                f"Could not fit within {max_memory_mb:.0f} MB budget "
                f"(estimated {_est_mb:.0f} MB at dx={dx*1e3:.3f} mm). "
                f"Consider reducing domain size or using mixed precision."
            )

    # 4. Non-uniform z profile
    dz_profile = None
    if needs_nonuniform_z:
        # Physical z domain = geometry extent + margin above for radiation
        geo_z_max = max(f[1] for f in features.z_features)
        phys_z = geo_z_max + margin
        dz_profile = _make_dz_profile(features.z_features, phys_z, dx)
        warnings.append(
            f"Non-uniform z mesh enabled: {len(dz_profile)} cells, "
            f"dz={np.min(dz_profile)*1e3:.3f}–{np.max(dz_profile)*1e3:.3f} mm"
        )

    # 4b. Timestep
    if dz_profile is not None:
        dz_min = float(np.min(dz_profile))
        dt = 0.99 / (C0 * math.sqrt(1/dx**2 + 1/dx**2 + 1/dz_min**2))
    else:
        dt = 0.99 * dx / (C0 * math.sqrt(3))

    # 5. Number of steps
    # Source time: the pulse completes at t0 + cutoff*tau = 2*t0
    # (= 6*tau for the historical cutoff=3 Gaussian envelope; the
    # waveform's own t0, when provided, makes this cutoff-aware —
    # e.g. 9*tau at cutoff=4.5). Default 2.0*(3.0*tau) is bitwise-
    # identical to the historical 6*tau (doubling is exact).
    bw = 0.8  # default bandwidth
    tau = 1.0 / (f_center * bw * math.pi)
    t_source = 2.0 * float(getattr(waveform, "t0", 3.0 * tau))

    # Ring-down time: Q / (pi * f_min)
    # Compute Q from stored sigma/eps_r ratio + frequency:
    # tan_d = (sigma/eps_r) / (2*pi*f_center*eps_0), Q ≈ 1/tan_d
    if features.estimated_Q > 0:
        tan_d = features.estimated_Q / (2 * math.pi * f_center * EPS_0)
        Q_est = min(1.0 / tan_d if tan_d > 0 else 1000.0, 1000.0)
    else:
        Q_est = 1000.0  # no loss → high Q, cap at 1000
    t_ringdown = Q_est / (math.pi * f_min)

    t_total = t_source + t_ringdown
    n_steps = int(math.ceil(t_total / dt))

    if n_steps_override is not None:
        n_steps = n_steps_override

    # Sanity cap
    n_steps = min(n_steps, 100000)
    if n_steps > 50000:
        warnings.append(f"Estimated {n_steps} steps (high Q={Q_est:.0f}). Consider reducing freq_range or using decay-based stopping.")
        n_steps = min(n_steps, 500000)

    # Source auto-selection based on boundary type
    if boundary == "pec":
        source_type = "raw"
        source_info = "PEC boundary: raw E-field source (no Cb normalization needed)"
    else:
        source_type = "j_source"
        source_info = "CPML boundary: J-source with Cb/dV normalization (Meep convention)"

    return SimConfig(
        dx=dx,
        domain=domain,
        cpml_layers=cpml_layers,
        n_steps=n_steps,
        freq_range=freq_range,
        margin=margin,
        dt=dt,
        accuracy=accuracy,
        source_type=source_type,
        source_info=source_info,
        warnings=warnings,
        dz_profile=dz_profile,
        boundary=boundary,
    )


def _make_dz_profile(
    z_features: list[tuple[float, float, float]],
    domain_z: float,
    dx: float,
    min_cells_per_feature: int = 4,
) -> np.ndarray:
    """Generate non-uniform z cell profile from z-feature boundaries.

    For each thin z-feature (substrate layer), creates fine cells that
    exactly snap to the feature thickness. Air regions use coarse dx cells.

    Parameters
    ----------
    z_features : list of (z_lo, z_hi, eps_r) for dielectric layers
    domain_z : total physical z domain height (excluding CPML)
    dx : uniform x/y cell size (used as coarse z cell size)
    min_cells_per_feature : minimum cells to resolve each feature
    """
    if not z_features:
        n = max(1, int(round(domain_z / dx)))
        return np.ones(n) * dx

    # Sort features by z_lo
    features = sorted(z_features, key=lambda f: f[0])

    # Collect z-boundary points
    z_max = max(f[1] for f in features)
    # Add air region above features up to domain_z
    air_height = max(0, domain_z - z_max)

    cells = []
    boundary_indices = []  # cell indices at material interfaces
    z_cursor = 0.0
    # #763: realized feature bounds in cumulative-cell coordinates. The
    # thirds rule preserves the cell sum on each side of every boundary it
    # splits, so these coordinates remain cell edges after
    # ``apply_thirds_rule`` and can be mapped back to index ranges there.
    feature_bounds = []
    running = 0.0

    for z_lo, z_hi, eps_r in features:
        # Air gap before this feature
        gap = z_lo - z_cursor
        if gap > dx * 0.5:
            n_gap = max(1, int(round(gap / dx)))
            cells.extend([gap / n_gap] * n_gap)
            running += gap

        # Mark the air-to-dielectric boundary
        boundary_indices.append(len(cells))

        # Feature cells: snap exactly
        thickness = z_hi - z_lo
        n_feat = max(min_cells_per_feature, int(np.ceil(thickness / dx)))
        dz_feat = thickness / n_feat
        cells.extend([dz_feat] * n_feat)

        # Mark the dielectric-to-air boundary
        boundary_indices.append(len(cells))
        feature_bounds.append((running, running + thickness))
        running += thickness
        z_cursor = z_hi

    # Air above features
    if air_height > dx * 0.5:
        n_air = max(1, int(round(air_height / dx)))
        cells.extend([air_height / n_air] * n_air)

    # P3: Apply thirds rule at material interfaces
    cells = apply_thirds_rule(cells, boundary_indices)

    # P2 + #763: smooth the transitions while (a) preserving every feature
    # block's cells bit-identically and (b) renormalizing each free (air)
    # run back to its declared physical length, so the realized profile
    # keeps every declared interface on a cell edge and the total column
    # equal to the declared column. Passing the whole array to
    # ``smooth_grading`` without ``preserve_regions`` inflated the column
    # (demo fixture: declared 1.754 mm realized as 2.380 mm, substrate-top
    # interface mid-cell at fraction 0.346) — issue #763.
    return _smooth_preserving_blocks(cells, feature_bounds, max_ratio=1.3)


def _smooth_preserving_blocks(
    cells: np.ndarray,
    block_bounds: list[tuple[float, float]],
    max_ratio: float = 1.3,
) -> np.ndarray:
    """Smooth-grade ``cells`` while keeping protected blocks exact (#763).

    ``block_bounds`` are (lo, hi) cumulative-coordinate pairs that MUST lie
    on cell edges of ``cells``. Each protected block passes through
    bit-identically. Each free run between blocks is smoothed against the
    adjacent block edge cells (identical transitions to
    ``smooth_grading(..., preserve_regions=...)`` on the full array) and
    then renormalized back to its pre-smoothing length: duplicated
    plateau (coarse) cells are removed while the run stays at or above its
    declared length, and the remaining cells are uniformly rescaled by
    ``f = L_declared / L_run`` (``f <= 1`` since smoothing only inserts).
    Renormalization touches only free-run cells, so every declared
    interface coordinate and the total column length are realized exactly
    (to float64 accumulation, << 1e-12 m).
    """
    cells = np.asarray(cells, dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(cells)])

    # Map block coordinate bounds to index ranges on the post-thirds array.
    ranges = []
    for lo, hi in block_bounds:
        i = int(np.argmin(np.abs(edges - lo)))
        j = int(np.argmin(np.abs(edges - hi)))
        if abs(edges[i] - lo) > 1e-12 or abs(edges[j] - hi) > 1e-12:
            raise AssertionError(
                f"feature bound ({lo}, {hi}) not on a cell edge "
                f"(nearest {edges[i]}, {edges[j]}) — thirds rule no longer "
                "preserves interface edges?"
            )
        ranges.append((i, j))

    # Alternating protected / free segments, in order.
    seg_list: list[tuple[np.ndarray, bool]] = []
    prev_end = 0
    for i, j in ranges:
        if i > prev_end:
            seg_list.append((cells[prev_end:i], False))
        seg_list.append((cells[i:j], True))
        prev_end = j
    if prev_end < len(cells):
        seg_list.append((cells[prev_end:], False))

    out: list[np.ndarray] = []
    for k, (seg, protected) in enumerate(seg_list):
        if protected or len(seg) == 0:
            out.append(np.asarray(seg, dtype=float))
            continue

        length = float(np.sum(seg))

        # Sentinel cells from the neighbouring protected blocks so the
        # transitions match a full-array smooth_grading with
        # preserve_regions (the loop there smooths against the block's
        # edge cell; blocks pass through verbatim, so the edge cell is
        # exactly the input edge cell).
        arr = [float(c) for c in seg]
        lo_off = 0
        if k > 0 and len(seg_list[k - 1][0]) > 0:
            arr = [float(seg_list[k - 1][0][-1])] + arr
            lo_off = 1
        hi_off = 0
        if k + 1 < len(seg_list) and len(seg_list[k + 1][0]) > 0:
            arr = arr + [float(seg_list[k + 1][0][0])]
            hi_off = 1

        sm = smooth_grading(arr, max_ratio=max_ratio)
        run = [float(c) for c in
               (sm[lo_off:len(sm) - hi_off] if hi_off else sm[lo_off:])]

        # Renormalize the free run back to its declared length. First
        # drop duplicated plateau cells (a cell equal to the run maximum
        # with an identical neighbour — removing one keeps the grading
        # smooth) while the run remains at or above the declared length.
        run_sum = float(np.sum(run))
        while run_sum > length:
            v = max(run)
            if run_sum - v < length:
                break
            idx = None
            for t in range(len(run) - 1):
                if run[t] == v and run[t + 1] == v:
                    idx = t
                    break
            if idx is None:
                break
            del run[idx]
            run_sum = float(np.sum(run))

        # Uniform rescale (f <= 1). Internal adjacent-cell ratios are
        # unchanged; only the seam ratio against a protected block edge
        # moves (the first-contact step, exempt by the preserve_regions
        # convention). f == 1.0 when smoothing inserted nothing.
        if run_sum != length and run_sum > 0.0:
            f = length / run_sum
            run = [c * f for c in run]

        out.append(np.asarray(run, dtype=float))

    return np.concatenate([o for o in out if len(o)]) if out else cells


def apply_thirds_rule(
    cells: list[float] | np.ndarray,
    boundary_indices: list[int],
) -> np.ndarray:
    """Apply the thirds rule at conductor-dielectric boundaries.

    At each boundary index, splits the cell so that the E-field sampling
    point falls at 1/3 inside the conductor and 2/3 outside.  This avoids
    placing an E-field node exactly on the material interface, which gives
    worst-case interpolation error.

    Inspired by OpenEMS thirds rule and Tidy3D corner_refinement.

    Parameters
    ----------
    cells : array-like
        Cell sizes along the graded axis.
    boundary_indices : list of int
        Indices in the cumulative cell array where material boundaries occur.
        A boundary at index *k* means the interface lies at the edge between
        cell k-1 and cell k.  The cell on the lower side (k-1) is treated
        as conductor, the cell on the upper side (k) as dielectric.

    Returns
    -------
    np.ndarray
        Cell array with boundary cells split into 1/3 + 2/3 pairs.
    """
    cells = list(np.asarray(cells, dtype=float))
    if not boundary_indices:
        return np.array(cells)

    # Sort boundaries in reverse order so index shifts don't affect later splits
    for idx in sorted(set(boundary_indices), reverse=True):
        if idx <= 0 or idx >= len(cells):
            continue
        # Cell below boundary (conductor side): split into [2/3, 1/3]
        dz_below = cells[idx - 1]
        # Cell above boundary (dielectric side): split into [1/3, 2/3]
        dz_above = cells[idx]

        # Only split if cells are large enough (> 2× minimum reasonable size)
        min_cell = min(dz_below, dz_above) / 3
        if min_cell < 1e-7:  # sub-micron is too fine
            continue

        # Replace cell[idx-1] with [2/3 * dz_below, 1/3 * dz_below]
        # Replace cell[idx]   with [1/3 * dz_above, 2/3 * dz_above]
        new_cells = (
            cells[:idx - 1]
            + [dz_below * 2 / 3, dz_below / 3]
            + [dz_above / 3, dz_above * 2 / 3]
            + cells[idx + 1:]
        )
        cells = new_cells

    return np.array(cells)


def smooth_grading(
    cells: list[float] | np.ndarray,
    max_ratio: float = 1.3,
    *,
    preserve_regions: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Insert geometric transition cells where adjacent ratio exceeds max_ratio.

    Prevents numerical reflections at abrupt cell-size transitions in
    non-uniform meshes.  Inspired by OpenEMS SmoothMeshLines.

    Parameters
    ----------
    cells : array-like
        Cell sizes along the graded axis.
    max_ratio : float
        Maximum allowed ratio between adjacent cells (default 1.3).
        Values 1.2-1.4 are typical for FDTD.
    preserve_regions : list of (start, end) index pairs, optional
        Input-index ranges (half-open) whose cell sizes must remain
        unchanged. Transition cells are inserted **outside** each
        preserved block. This is the canonical Meep/OpenEMS convention
        for thin PEC on a NU mesh — metal planes must land on cell
        boundaries with symmetric neighbouring cells, which is only
        guaranteed if the fine substrate block is preserved intact.

    Returns
    -------
    np.ndarray
        Smoothed cell array with transition cells inserted.
    """
    cells = list(np.asarray(cells, dtype=float))
    n = len(cells)
    if n <= 1:
        return np.array(cells)

    # Normalise preserve_regions into a set of protected input indices.
    protected: set[int] = set()
    regions = sorted(preserve_regions or [])
    for lo, hi in regions:
        if lo < 0 or hi > n or lo >= hi:
            raise ValueError(
                f"preserve_regions entry ({lo}, {hi}) is outside [0, {n}] "
                f"or has lo>=hi"
            )
        for k in range(lo, hi):
            protected.add(k)

    smoothed = [cells[0]]
    for i in range(1, n):
        prev = smoothed[-1]
        target = cells[i]
        # Skip smoothing only when BOTH adjacent cells are inside a
        # preserved block (inside the block, cells must stay exact).
        # At the boundary of a block (one side protected, the other
        # free) we DO insert transitions so the neighbouring coarse
        # region steps down to the fine block.
        both_inside = (i - 1) in protected and i in protected
        if (not both_inside) and prev > 0 and target > 0:
            while target / prev > max_ratio + 1e-12:
                prev = prev * max_ratio
                smoothed.append(prev)
            while prev / target > max_ratio + 1e-12:
                prev = prev / max_ratio
                smoothed.append(prev)
        smoothed.append(target)

    return np.array(smoothed)


def _round_dx(dx: float) -> float:
    """Round dx DOWN to a nice value (never coarser than computed)."""
    exp = math.floor(math.log10(dx))
    mantissa = dx / (10 ** exp)
    nice = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    # Pick the largest nice value <= mantissa
    candidates = [n for n in nice if n <= mantissa + 1e-9]
    best = candidates[-1] if candidates else nice[0]
    return best * (10 ** exp)
