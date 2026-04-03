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

C0 = 299792458.0
EPS_0 = 8.854187817e-12


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

        if hasattr(shape, "corner_lo") and hasattr(shape, "corner_hi"):
            c1, c2 = shape.corner_lo, shape.corner_hi
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
    warnings: list[str] = field(default_factory=list)
    dz_profile: np.ndarray | None = None

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
            "boundary": "cpml",
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
    dx_override: float | None = None,
    margin_override: float | None = None,
    n_steps_override: int | None = None,
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
    dx_override : force specific cell size
    margin_override : force specific margin
    n_steps_override : force specific step count

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
        "high":     {"cpw": 40, "cpf": 8, "margin_frac": 0.50, "pml_frac": 0.40},
    }
    if accuracy not in presets:
        raise ValueError(f"accuracy must be 'draft', 'standard', or 'high', got {accuracy!r}")
    preset = presets[accuracy]

    # Analyze geometry
    features = analyze_features(geometry, materials)
    warnings = []

    # 1. Cell size
    # λ_min in highest-eps medium
    lambda_min_medium = lambda_min / math.sqrt(features.max_eps_r)
    dx_wavelength = lambda_min_medium / preset["cpw"]
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

    if features.min_thickness > 0 and features.min_thickness / dx < 2 and not needs_nonuniform_z:
        warnings.append(
            f"Thinnest feature ({features.min_thickness*1e3:.2f} mm) has only "
            f"{features.min_thickness/dx:.1f} cells — consider finer dx or subgridding"
        )

    # 2. Domain margins
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

    # 3. CPML layers — ensure physical thickness >= 0.4*λ_max (Meep/OpenEMS convention)
    # with a floor from accuracy preset for well-tuned CFS-CPML grading.
    min_cpml_thickness = 0.4 * lambda_max
    min_cpml_cells = int(np.ceil(min_cpml_thickness / dx))
    cpml_cells = {"draft": 8, "standard": 12, "high": 16}[accuracy]
    cpml_layers = max(min_cpml_cells, cpml_cells)

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
    # Source time: 6*tau for Gaussian pulse
    bw = 0.8  # default bandwidth
    tau = 1.0 / (f_center * bw * math.pi)
    t_source = 6 * tau

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

    return SimConfig(
        dx=dx,
        domain=domain,
        cpml_layers=cpml_layers,
        n_steps=n_steps,
        freq_range=freq_range,
        margin=margin,
        dt=dt,
        accuracy=accuracy,
        warnings=warnings,
        dz_profile=dz_profile,
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
    z_cursor = 0.0

    for z_lo, z_hi, eps_r in features:
        # Air gap before this feature
        gap = z_lo - z_cursor
        if gap > dx * 0.5:
            n_gap = max(1, int(round(gap / dx)))
            cells.extend([gap / n_gap] * n_gap)

        # Feature cells: snap exactly
        thickness = z_hi - z_lo
        n_feat = max(min_cells_per_feature, int(np.ceil(thickness / dx)))
        dz_feat = thickness / n_feat
        cells.extend([dz_feat] * n_feat)
        z_cursor = z_hi

    # Air above features
    if air_height > dx * 0.5:
        n_air = max(1, int(round(air_height / dx)))
        cells.extend([air_height / n_air] * n_air)

    return np.array(cells)


def _round_dx(dx: float) -> float:
    """Round dx DOWN to a nice value (never coarser than computed)."""
    exp = math.floor(math.log10(dx))
    mantissa = dx / (10 ** exp)
    nice = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    # Pick the largest nice value <= mantissa
    candidates = [n for n in nice if n <= mantissa + 1e-9]
    best = candidates[-1] if candidates else nice[0]
    return best * (10 ** exp)
