"""Total-Field/Scattered-Field (TFSF) plane-wave source.

Normal incidence along ``+x`` or ``-x`` with transverse polarization in
``Ez`` or ``Ey``.
Uses a 1D auxiliary FDTD with CPML to generate the incident field.

Oblique incidence (``angle_deg != 0``) dispatches to a **2D auxiliary grid**
(``rfx/sources/tfsf_2d.py``), which shares the 3D cell size so numerical
dispersion matches at any angle. That grid carries the oblique transverse
wavenumber ``k_y = k0·sinθ`` via a **Bloch field-transformation** (#404); as a
consequence the oblique source is **narrowband** (single-frequency at ``f0``).
Two caveats for oblique runs — see ``tfsf_2d.py`` for the full discussion:
  - keep the fractional ``bandwidth`` small (≲0.15) or off-``f0`` components
    carry the wrong ``k_y`` and raise TFSF leakage;
  - the injected wave tracks the requested angle in **open / CPML-transverse**
    domains; a **periodic** transverse domain additionally needs
    ``k_y·L_y = 2πm`` (commensurate) or a 3D Bloch-periodic BC (not yet
    implemented) — otherwise the plain-periodic 3D seam re-distorts steep angles.

TFSF boundary cleanly separates:
  - Total-field region (x_lo <= x <= x_hi): incident + scattered
  - Scattered-field region (outside): scattered only

Reference: Taflove & Hagness, Ch. 5.

Corrections derived from Yee update stencil mismatch at TFSF boundary:
  E correction (after update_e):
    Ez[x_lo]   -= (dt/(eps0*dx)) * Hy_inc[x_lo - 0.5]
    Ez[x_hi+1] += (dt/(eps0*dx)) * Hy_inc[x_hi + 0.5]
  H correction (after update_h):
    Hy[x_lo-1] -= (dt/(mu0*dx)) * Ez_inc[x_lo]
    Hy[x_hi]   += (dt/(mu0*dx)) * Ez_inc[x_hi + 1]
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.boundaries.cpml import CPMLParams, _cpml_profile
from rfx.core.yee import EPS_0, MU_0, MaterialArrays, component_h_materials
from rfx.sources.sources import CustomWaveform


# ---------------------------------------------------------------------------
# Auxiliary-grid absorber (#888)
# ---------------------------------------------------------------------------
# See rfx/sources/tfsf_2d.py for the finding.  The 1-D grid is the normal-
# incidence path (cv04's); it shipped at 20 cells with
# sigma_max = 0.8 * 4 / (eta dx_1d) and reflected |B/A| ~ 4e-02.  The 1-D CPML
# update applies b and c but no kappa stretching, so kappa_max stays 1
# and an override to anything else is refused rather than silently ignored.
# With H sampled at its Yee half cell (#1234), 20 cells return at most
# -111.8 dB over the measured 2--30 GHz band and both faces at dx = 1 mm.
# Ten cells return -87.4 dB; the auxiliary requirement is -105 dB.
# Keep the normal-incidence target R = 1e-6 and restore the 20-cell depth.
# The dated #888 design note retains its measurements of the older profile.
AUX_N_CPML_1D = 20
AUX_CPML_ORDER_1D = 3       # rfx/boundaries/cpml.py _cpml_profile default
AUX_CPML_KAPPA_MAX_1D = 1.0
AUX_CPML_R_ASYMPTOTIC_1D = 1e-6
AUX_N_MARGIN_1D = 10
AUX_SRC_OFFSET_1D = 3       # source sits this many cells inside the absorber's inner edge


def pulse_timing(waveform, f0, bandwidth):
    """Return (tau, t0) without coercing traced source parameters."""
    if waveform == "continuous_wave":
        t0 = 8.0 / f0
        tau = t0
    else:
        tau = 1.0 / (f0 * bandwidth * np.pi)
        if waveform == "modulated_gaussian":
            t0 = 5.0 * tau
        else:
            t0 = 3.0 * tau
    return tau, t0


def resolve_waveform(waveform, angle_deg, method):
    """Resolve the declared default, preserving CustomWaveform identity."""
    bloch = abs(angle_deg) > 0.01 and method == "bloch"
    if waveform is None:
        return "modulated_gaussian" if bloch else "differentiated_gaussian"
    if bloch and waveform == "differentiated_gaussian":
        raise ValueError(
            "The Bloch oblique plane wave holds one transverse wavenumber "
            "k_y = k0(f0)·sin θ, so it carries only a modulated Gaussian around f0; "
            'declare waveform="modulated_gaussian" or leave it unset, or use '
            'method="methodB" for an open domain.'
        )
    return waveform


def is_tfsf_2d(cfg) -> bool:
    """Check if TFSF config uses 2D auxiliary grid (for oblique incidence)."""
    from rfx.sources.tfsf_2d import TFSF2DConfig
    return isinstance(cfg, TFSF2DConfig)


def is_tfsf_methodB(cfg) -> bool:
    """Check if TFSF config is the open-domain oblique Method-B config.

    Method B (``rfx/sources/tfsf_oblique_open.py``) injects an oblique plane
    wave into an open (CPML) 2.5-D domain (k̂ in the xy-plane, thin-z periodic)
    using a dispersion-matched 1D-aux-along-k̂ + a 4-edge TFSF box.  Its config
    is a NEW NamedTuple (NOT a ``TFSF2DConfig``), so ``is_tfsf_2d`` is ``False``
    and the fields stay real ``float32`` — the complex-Bloch/NTFF-reject lane
    never fires and NTFF/probes read physical fields.
    """
    from rfx.sources.tfsf_oblique_open import MethodBConfig
    return isinstance(cfg, MethodBConfig)


from rfx.boundaries.tfsf import boundary_flags as tfsf_boundary_flags


def tfsf_injection_planes(cfg) -> "dict[str, tuple[int, int]]":
    """The node planes that bound the total-field region, per injected axis.

    Every config uses the same correction structure on every axis it drives:
    E is corrected at the low plane and one node past the high plane, H at
    one plane below the low plane and at the high plane. Reading the signs
    off those four, the E node at index i and the H half-plane at index i
    both carry incident + scattered for ``lo <= i <= hi`` and scattered only
    outside — one membership test for both fields, on every axis.

    * 1-D normal incidence (:class:`TFSFConfig`, ``x_lo``/``x_hi``) and the
      2-D Bloch ``TFSF2DConfig`` inject on x only. Their slab is infinite in
      y and z, so there are no y or z planes to clear.
    * Open-domain oblique ``MethodBConfig``
      (``rfx/sources/tfsf_oblique_open.py``) injects on a RECTANGLE: x and y
      both. ``apply_methodB_e`` corrects ``Ez[x_lo]`` / ``Ez[x_hi+1]`` and
      ``Ez[., y_lo]`` / ``Ez[., y_hi+1]``; ``apply_methodB_h`` corrects
      ``Hy[x_lo-1]`` / ``Hy[x_hi]`` and ``Hx[., y_lo-1]`` / ``Hx[., y_hi]``.
      Measured in an empty domain at theta = 20 deg: max|Ez| is 7.8e-01
      inside ``y_lo <= j <= y_hi`` and 3.5e-03 outside, with max|Hx|
      switching at the same two indices.

    Returns an empty mapping for a config that injects on no axis this
    function knows, so a caller can skip a placement check rather than fail
    on a missing attribute.
    """
    planes: dict = {}
    x_lo = getattr(cfg, "x_lo", None)
    x_hi = getattr(cfg, "x_hi", None)
    if x_lo is not None and x_hi is not None:
        planes["x"] = (int(x_lo), int(x_hi))
    if is_tfsf_methodB(cfg):
        planes["y"] = (int(cfg.y_lo), int(cfg.y_hi))
    elif getattr(cfg, "closed_box", False):
        planes["y"] = (int(cfg.y_lo), int(cfg.y_hi))
        planes["z"] = (int(cfg.z_lo), int(cfg.z_hi))
    return planes


class TFSFState(NamedTuple):
    """1D auxiliary FDTD state for TFSF plane-wave generation."""
    e1d: jnp.ndarray   # Ez incident (n_1d,)
    h1d: jnp.ndarray   # Hy incident (n_1d,)
    psi_e_lo: jnp.ndarray  # 1D CPML psi for E at lo end
    psi_e_hi: jnp.ndarray
    psi_h_lo: jnp.ndarray
    psi_h_hi: jnp.ndarray
    step: jnp.ndarray


class TFSFConfig(NamedTuple):
    """Static TFSF configuration (compile-time constants for JIT)."""
    x_lo: int       # TFSF box x-lo index (inclusive, total-field starts here)
    x_hi: int       # TFSF box x-hi index (inclusive, total-field ends here)
    i0: int         # 1D index that maps to 3D x_lo
    src_idx: int    # 1D source injection index
    n_cpml: int     # Number of 1D CPML layers
    # 1D CPML coefficients
    b_cpml: jnp.ndarray   # (n_cpml,)
    c_cpml: jnp.ndarray
    # Source waveform (inlined for JIT)
    src_amp: float
    src_t0: float
    src_tau: float
    src_fcen: float          # carrier frequency for modulated Gaussian
    src_waveform: str        # "differentiated_gaussian" or "modulated_gaussian"
    # Polarization selection
    electric_component: str  # "ez" or "ey"
    magnetic_component: str  # "hy" or "hz"
    curl_sign: float         # +1 for Ez/Hy, -1 for Ey/Hz
    direction: str           # "+x" or "-x"
    direction_sign: float    # +1 for +x, -1 for -x
    angle_deg: float
    cos_theta: float
    sin_theta: float
    grid_pad: int
    transverse_axis: str     # "y" for Ez, "z" for Ey
    dx_1d: float             # 1D auxiliary grid cell size (dx/cos(theta))
    magnetic_lo: CPMLParams
    magnetic_hi: CPMLParams
    custom_waveform: object = None  # static JAX-compatible scalar callable
    closed_box: bool = False
    y_lo: int = 0
    y_hi: int = 0
    z_lo: int = 0
    z_hi: int = 0


def validate_custom_tfsf_waveform(waveform, dt: float, n_steps: int) -> None:
    """Refuse complex, non-scalar or nonfinite drive samples before stepping.

    The waveform is a fixed input, not a traced material parameter. Evaluate
    its complete requested record at compile time so this check also holds
    when the user wraps ``forward`` in ``jax.jit``.
    """
    if not isinstance(waveform, CustomWaveform):
        return
    import jax
    with jax.ensure_compile_time_eval():
        times = jnp.arange(n_steps, dtype=jnp.result_type(float)) * dt
        values = jax.vmap(waveform)(times)
        if isinstance(values, jax.core.Tracer):
            raise ValueError(
                "TFSF CustomWaveform is a fixed input: it cannot depend on a traced "
                "parameter (differentiate the materials, not the source waveform)")
        if values.shape != (n_steps,) or not jnp.issubdtype(values.dtype, jnp.floating):
            raise ValueError("TFSF CustomWaveform must return real scalar floating-point samples")
        if not np.all(np.isfinite(np.asarray(values))):
            raise ValueError("TFSF CustomWaveform must return finite samples throughout the requested record")


def _refuse_extended_tfsf(entry, lane: str) -> None:
    """Admission fence for runners that cannot carry either new source input."""
    if entry is not None and (getattr(entry, "closed_box", False)
                              or isinstance(entry.waveform, CustomWaveform)):
        raise NotImplementedError(
            f"CustomWaveform and closed_box TFSF are not supported on {lane}; "
            "use the uniform single-device 3D second-order Yee run/forward")


def init_tfsf(
    nx: int,
    dx: float,
    dt: float,
    cpml_layers: int = 0,
    tfsf_margin: int = 3,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    polarization: str = "ez",
    direction: str = "+x",
    angle_deg: float = 0.0,
    ny: int | None = None,
    nz: int | None = None,
    waveform: str | CustomWaveform | None = None,
    method: str = "bloch",
    aux_n_cpml: int | None = None,
    aux_cpml_order: int | None = None,
    aux_cpml_kappa_max: float | None = None,
    aux_cpml_r_asymptotic: float | None = None,
    closed_box: bool = False,
    *, n_steps: int | None = None, theta_deg: float = 0.0, phi_deg: float = 0.0,
    direction_tol_deg: float = 0.1, max_component: int = 100,
) -> tuple:
    """Initialize TFSF source.

    Parameters
    ----------
    nx : int
        Number of cells in x (3D grid).
    dx, dt : float
        Cell size and timestep.
    cpml_layers : int
        Number of 3D CPML layers (TFSF box is inset from these).
    tfsf_margin : int
        Extra cells between CPML edge and TFSF boundary.
    f0, bandwidth, amplitude : float
        Gaussian pulse parameters. For ``differentiated_gaussian`` the
        time envelope has width ``tau = 1/(π·f0·bandwidth)``. For
        ``modulated_gaussian`` (Meep-equivalent) ``bandwidth`` is the
        fractional frequency width: ``fwidth = f0·bandwidth``, and
        ``tau = 1/(π·fwidth)``.
    polarization : "ez" or "ey"
        Electric-field polarization for the incident plane wave.
    direction : "+x" or "-x"
        Propagation direction along the x-axis.
    angle_deg : float
        Signed oblique angle in degrees away from the x-axis. Positive
        angles tilt toward the positive transverse axis. Non-zero angles
        dispatch to the 2D-aux grid and produce a **narrowband** (single-f0)
        oblique wave via a Bloch field-transformation (#404); use a small
        fractional ``bandwidth`` (≲0.15) and an open/commensurate transverse
        domain (see the module docstring and ``tfsf_2d.py``).
    ny : int | None
        Number of cells in y (3D grid).  Required for oblique incidence
        with ez polarization (2D TMz auxiliary grid).  Ignored for
        normal incidence.
    nz : int | None
        Number of cells in z (3D grid).  Required for oblique incidence
        with ey polarization (2D TEz auxiliary grid).  Ignored for
        normal incidence and ez polarization.
    waveform : {"differentiated_gaussian", "modulated_gaussian", "continuous_wave"}
        None selects DG except on oblique Bloch, where it selects MG.
        Explicit DG on oblique Bloch is refused. ``differentiated_gaussian`` is a
        baseband differentiated Gaussian with no carrier (legacy rfx
        behaviour). ``modulated_gaussian`` matches Meep's
        ``GaussianSource``:
        ``cos(2π·f0·(t-t0)) · exp(-((t-t0)/τ)²)`` with τ = 1/(π·fwidth)
        and t0 = 5·τ (Meep default). ``continuous_wave`` is a single-
        frequency sinusoid at ``f0`` with a raised-cosine turn-on
        (~8 periods) then constant amplitude — for steady-state /
        narrowband measurements needing a well-defined ⟨E²⟩=A²/2 (e.g.
        the #446 Kerr SPM oracle). NORMAL INCIDENCE ONLY (the oblique
        2D-auxiliary path does not carry it).
    aux_n_cpml, aux_cpml_order, aux_cpml_kappa_max, aux_cpml_r_asymptotic
        Override the AUXILIARY grid's own absorber (#888). Defaults are the
        derived constants at the top of this module (normal incidence) and of
        ``tfsf_2d.py`` (oblique Bloch); both dispatch paths honour them, and
        overriding is how the reflection gate measures the profile that shipped.
        ``method='methodB'`` REFUSES them: that path
        (``tfsf_oblique_open.py``) builds its own 20-cell auxiliary absorber
        from the pre-#888 heuristic, so accepting the arguments there would say
        the knob works when it does not.

    Returns
    -------
    (TFSFConfig, TFSFState) for normal incidence, or
    (TFSF2DConfig, TFSF2DState) for oblique incidence (|angle_deg| > 0.01).
    """
    custom = isinstance(waveform, CustomWaveform)
    if (custom or closed_box) and angle_deg != 0.0:
        raise NotImplementedError("CustomWaveform and closed_box require normal incidence (angle_deg=0)")
    waveform = resolve_waveform(waveform, angle_deg, method)
    if not custom and waveform not in ("differentiated_gaussian", "modulated_gaussian", "continuous_wave"):
        raise ValueError(
            f"waveform must be 'differentiated_gaussian', "
            f"'modulated_gaussian', 'continuous_wave', or CustomWaveform, got {waveform!r}"
        )
    if custom:
        import jax
        value = jax.eval_shape(waveform, jnp.asarray(0.0))
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError("TFSF CustomWaveform must return a real scalar floating-point value")
    if closed_box:
        from rfx.sources.tfsf_discrete_plane_wave import init_plane_wave
        if ny is None or nz is None:
            raise ValueError("closed_box=True requires ny and nz")
        return init_plane_wave(nx, dx, dt, ny=ny, nz=nz, cpml_layers=cpml_layers,
            tfsf_margin=tfsf_margin, f0=f0, bandwidth=bandwidth, amplitude=amplitude,
            polarization=polarization, direction=direction, theta_deg=theta_deg,
            phi_deg=phi_deg, direction_tol_deg=direction_tol_deg,
            max_component=max_component, waveform=waveform, n_steps=n_steps)
    # continuous_wave lives only in the 1D-auxiliary (normal-incidence) injector; the 2D
    # oblique grid carries only an analytic MG, so reject it here rather than
    # silently drop it (fail-loud, not right-guard-mis-gated).
    if waveform == "continuous_wave" and abs(angle_deg) > 0.01:
        raise ValueError(
            "waveform='continuous_wave' is only supported at normal incidence "
            f"(angle_deg=0); got angle_deg={angle_deg}"
        )
    if method not in ("bloch", "methodB"):
        raise ValueError(f"method must be 'bloch' or 'methodB', got {method!r}")
    # Oblique open-domain Method B (real float32, 1D-aux-along-k̂ + 4-edge box).
    # Angle alone cannot disambiguate this from the Bloch 2D-aux path, so it is
    # selected explicitly via method='methodB'. angle_deg=0 falls through to the
    # normal 1D-aux path below regardless of method (byte-unchanged).
    if abs(angle_deg) > 0.01 and method == "methodB":
        # Method B builds its own 1-D auxiliary absorber (tfsf_oblique_open.py:
        # 20 cells, sigma_max = 0.8 * 4/(eta d_aux), rho**3) -- the pre-#888
        # heuristic, untouched by this lane. Accepting an aux_* override here
        # and dropping it would be the same silent-drop the waveform guard
        # above exists to prevent.
        _aux_overrides = {
            "aux_n_cpml": aux_n_cpml, "aux_cpml_order": aux_cpml_order,
            "aux_cpml_kappa_max": aux_cpml_kappa_max,
            "aux_cpml_r_asymptotic": aux_cpml_r_asymptotic,
        }
        _given = sorted(k for k, v in _aux_overrides.items() if v is not None)
        if _given:
            raise ValueError(
                f"{_given} are not supported with method='methodB': that path's "
                "auxiliary absorber is built inside "
                "rfx/sources/tfsf_oblique_open.py and is not derived from a "
                "reflection target (#888 covers the normal-incidence and Bloch "
                "oblique paths only)"
            )
        from rfx.sources.tfsf_oblique_open import init_tfsf_methodB
        if ny is None:
            ny = nx
        return init_tfsf_methodB(
            nx, ny, dx, dt,
            nz=nz,
            cpml_layers=cpml_layers,
            tfsf_margin=tfsf_margin,
            f0=f0,
            bandwidth=bandwidth,
            amplitude=amplitude,
            polarization=polarization,
            direction=direction,
            theta_deg=angle_deg,
            waveform=waveform,
        )
    # Dispatch to 2D auxiliary grid for oblique angles (Bloch field-transform).
    # The 2D grid naturally matches the 3D numerical dispersion at any angle.
    if abs(angle_deg) > 0.01:
        from rfx.sources.tfsf_2d import init_tfsf_2d
        if ny is None:
            ny = nx  # default: square grid
        return init_tfsf_2d(
            nx, ny, dx, dt,
            nz=nz,
            cpml_layers=cpml_layers,
            tfsf_margin=tfsf_margin,
            f0=f0,
            bandwidth=bandwidth,
            amplitude=amplitude,
            polarization=polarization,
            direction=direction,
            theta_deg=angle_deg,
            waveform=waveform,
            # forwarded, not dropped: the reflection gate measures the oblique
            # absorber through this entry point (#888)
            aux_n_cpml=aux_n_cpml,
            aux_cpml_order=aux_cpml_order,
            aux_cpml_kappa_max=aux_cpml_kappa_max,
            aux_cpml_r_asymptotic=aux_cpml_r_asymptotic,
        )
    if nx <= 0:
        raise ValueError(f"nx must be positive, got {nx}")
    if cpml_layers < 0:
        raise ValueError(f"cpml_layers must be >= 0, got {cpml_layers}")
    if tfsf_margin < 1:
        raise ValueError(f"tfsf_margin must be >= 1, got {tfsf_margin}")
    if polarization not in ("ez", "ey"):
        raise ValueError(f"polarization must be 'ez' or 'ey', got {polarization!r}")
    if direction not in ("+x", "-x"):
        raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
    if abs(angle_deg) >= 90.0:
        raise ValueError(f"abs(angle_deg) must be < 90, got {angle_deg}")

    theta = np.radians(angle_deg)
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))

    # 1D cell size: for normal incidence (the only case reaching this
    # code path), dx_1d = dx.  Oblique angles dispatch to 2D above.
    dx_1d = dx

    # TFSF box boundaries in x
    offset = cpml_layers + tfsf_margin
    x_lo = offset
    x_hi = nx - offset - 1

    if x_lo <= 0 or x_hi >= nx - 1 or x_lo >= x_hi:
        raise ValueError(
            "TFSF margin/cpml_layers are too large for the grid: "
            f"nx={nx}, cpml_layers={cpml_layers}, tfsf_margin={tfsf_margin}"
        )
    # 1D auxiliary grid: spans x_lo..x_hi + margins for source + CPML
    n_cpml_1d = AUX_N_CPML_1D if aux_n_cpml is None else int(aux_n_cpml)
    n_tfsf = x_hi - x_lo + 2   # +2 for the correction at x_hi+1
    # Normal incidence only reaches this code path (oblique dispatches to 2D above).
    n_margin = AUX_N_MARGIN_1D
    n_1d = n_cpml_1d + n_margin + n_tfsf + n_margin + n_cpml_1d

    # i0: 1D index that maps to 3D x_lo
    i0 = n_cpml_1d + n_margin
    if direction == "+x":
        # Launch from the left margin so the right-going wave enters the mapped region.
        src_idx = n_cpml_1d + AUX_SRC_OFFSET_1D
    else:
        # Launch from the right margin so the left-going wave enters the mapped region.
        src_idx = n_1d - n_cpml_1d - AUX_SRC_OFFSET_1D - 1

    # 1D CPML profile.  sigma_max is DERIVED from a target reflection through the
    # same law the 3-D absorber uses (rfx/boundaries/cpml.py::_cpml_profile), on
    # dx_1d so the oblique dispersion match is preserved -- see AUX_*_1D above
    # and #888.
    _kappa_1d = (AUX_CPML_KAPPA_MAX_1D if aux_cpml_kappa_max is None
                 else float(aux_cpml_kappa_max))
    if _kappa_1d != 1.0:
        raise ValueError(
            "the 1-D auxiliary CPML update carries no kappa (the split updates use b and c "
            f"only), so kappa_max must stay 1.0; got {_kappa_1d}"
        )
    profile_options = dict(
        order=AUX_CPML_ORDER_1D if aux_cpml_order is None else int(aux_cpml_order),
        kappa_max=_kappa_1d,
        R_asymptotic=(AUX_CPML_R_ASYMPTOTIC_1D if aux_cpml_r_asymptotic is None
                      else float(aux_cpml_r_asymptotic)),
    )
    _prof = _cpml_profile(n_cpml_1d, dt, dx_1d, **profile_options)
    magnetic_lo = _cpml_profile(
        n_cpml_1d, dt, dx_1d, sample_offset=0.5, **profile_options)
    magnetic_hi = _cpml_profile(
        n_cpml_1d, dt, dx_1d, sample_offset=-0.5, **profile_options)
    b_prof, c_prof = _prof.b, _prof.c

    tau, t0 = pulse_timing(waveform, f0, bandwidth)

    if polarization == "ez":
        electric_component = "ez"
        magnetic_component = "hy"
        curl_sign = 1.0
        transverse_axis = "y"
    else:
        electric_component = "ey"
        magnetic_component = "hz"
        curl_sign = -1.0
        transverse_axis = "z"

    direction_sign = 1.0 if direction == "+x" else -1.0

    fdtype = jnp.result_type(float)  # float32 normally, float64 with x64

    config = TFSFConfig(
        x_lo=x_lo,
        x_hi=x_hi,
        i0=i0,
        src_idx=src_idx,
        n_cpml=n_cpml_1d,
        b_cpml=jnp.array(b_prof, dtype=fdtype),
        c_cpml=jnp.array(c_prof, dtype=fdtype),
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
        src_fcen=float(f0),
        src_waveform="custom" if custom else waveform,
        electric_component=electric_component,
        magnetic_component=magnetic_component,
        curl_sign=float(curl_sign),
        direction=direction,
        direction_sign=float(direction_sign),
        angle_deg=float(angle_deg),
        cos_theta=cos_theta,
        sin_theta=sin_theta,
        grid_pad=cpml_layers,
        transverse_axis=transverse_axis,
        dx_1d=float(dx_1d),
        magnetic_lo=magnetic_lo,
        magnetic_hi=magnetic_hi,
        custom_waveform=waveform if custom else None,
    )

    state = TFSFState(
        e1d=jnp.zeros(n_1d, dtype=fdtype),
        h1d=jnp.zeros(n_1d, dtype=fdtype),
        psi_e_lo=jnp.zeros(n_cpml_1d, dtype=fdtype),
        psi_e_hi=jnp.zeros(n_cpml_1d, dtype=fdtype),
        psi_h_lo=jnp.zeros(n_cpml_1d, dtype=fdtype),
        psi_h_hi=jnp.zeros(n_cpml_1d, dtype=fdtype),
        step=jnp.array(0, dtype=jnp.int32),
    )

    return config, state


from rfx.stepping.incident_line import prepare_tfsf_for_scan


def refuse_graded_propagation_axis(entry, grid):
    """Refuse a plane wave whose propagation axis is graded (#2078).

    The incident line is one cell size, the boundary cell, while the 3-D grid
    carries the wave through the graded cells; the planes then inject a wave
    that does not match the one arriving. Measured with no scatterer on 60 x
    cells graded 1 / 0.5 / 1 mm: 0.72 of the total-field peak in the
    scattered-field region, against 1e-5 on uniform cells. A graded
    transverse axis is the supported case.
    """
    from rfx._grid_metric import is_one_cell_size
    from rfx.core.jax_utils import is_tracer
    axis = entry.direction[-1]
    cells = grid.cells(axis)
    if is_tracer(cells) or is_one_cell_size(cells):
        return
    sizes = np.asarray(cells, dtype=float)
    raise ValueError(
        f"a TF/SF plane wave along {axis} needs uniform {axis} cells; this mesh "
        f"grades {axis} from {sizes.min() * 1e3:.4g} to {sizes.max() * 1e3:.4g} mm. "
        f"The incident line is built on one cell size, so the injected wave "
        f"does not match the arriving one and leaks into the scattered-field "
        f"region (0.72 of the peak measured on a 1 / 0.5 / 1 mm line, rfx#2078). "
        f"Keep the {axis} cells uniform and grade a transverse axis instead.")


def tfsf_from_declaration(entry, grid, freq_max, *, n_steps=None, line_steps=None,
                          bounds_only=False, record_faces=False):
    """The TF/SF configuration a plane-wave declaration realizes on ``grid``.

    One builder for every entry (run on the uniform and graded lanes, forward,
    the injection-plane query), so no two entries build different sources.
    With ``n_steps``, a CustomWaveform is checked over that record first.
    ``line_steps`` independently sizes the closed-box line; ``bounds_only``
    returns closed-box metadata without allocating that line.
    ``record_faces`` marks a closed box selected by forward's static memory
    rule: the field scan reads a face record instead of carrying the line.
    Runs and forward requests that do not save memory keep the line.
    """
    if n_steps is not None:
        validate_custom_tfsf_waveform(entry.waveform, grid.dt, n_steps)
    if bounds_only and entry.closed_box:
        from rfx.sources.tfsf_discrete_plane_wave import box_bounds
        return box_bounds(grid.shape, grid.cpml_layers, entry.margin), None
    if not bounds_only:
        refuse_graded_propagation_axis(entry, grid)
    pair = init_tfsf(
        grid.nx, float(grid.cells("x")[0]), grid.dt, cpml_layers=grid.cpml_layers,
        tfsf_margin=entry.margin,
        f0=entry.f0 if entry.f0 is not None else freq_max / 2,
        bandwidth=entry.bandwidth, amplitude=entry.amplitude,
        polarization=entry.polarization, direction=entry.direction,
        angle_deg=entry.angle_deg, ny=grid.ny, nz=grid.nz,
        waveform=entry.waveform, method=entry.method, closed_box=entry.closed_box,
        theta_deg=entry.theta_deg, phi_deg=entry.phi_deg,
        direction_tol_deg=entry.direction_tol_deg, max_component=entry.max_component,
        n_steps=line_steps if line_steps is not None else n_steps)
    if record_faces and entry.closed_box:
        from dataclasses import replace
        pair = (replace(pair[0], record_faces=True), pair[1])
    return pair




def update_tfsf_1d_h(cfg: TFSFConfig, st: TFSFState, dx: float,
                      dt: float) -> TFSFState:
    """Advance 1D auxiliary H field: h1d^{n-1/2} → h1d^{n+1/2}.

    Uses e1d at the current time level (time n).
    Call AFTER apply_tfsf_h (which reads e1d at time n) and BEFORE
    update_e / apply_tfsf_e (which need h1d at time n+1/2).

    The 1D grid uses cfg.dx_1d (= dx/cos(theta) for oblique, dx for normal)
    so that numerical dispersion matches the 3D grid along the oblique
    propagation direction.
    """
    if getattr(cfg, "closed_box", False):
        if cfg.face_record is not None:
            return st
        from rfx.sources.tfsf_discrete_plane_wave import update_line_h
        return update_line_h(cfg, st, dx, dt)
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0))[0] * MU_0
    n = cfg.n_cpml
    e1d, h1d = st.e1d, st.h1d
    dx_1d = cfg.dx_1d

    # 1D Faraday pair for +x propagation:
    #   Ez/Hy  ->  +sign
    #   Ey/Hz  ->  -sign
    de = (jnp.concatenate([e1d[1:], jnp.zeros(1)]) - e1d) / dx_1d
    h1d = h1d + cfg.curl_sign * (dt / mu_abs) * de

    # H CPML lo end
    psi_h_lo = cfg.magnetic_lo.b * st.psi_h_lo + cfg.magnetic_lo.c * de[:n]
    h1d = h1d.at[:n].add(cfg.curl_sign * (dt / mu_abs) * psi_h_lo)

    # H CPML hi end
    b_hi = jnp.flip(cfg.magnetic_hi.b)
    c_hi = jnp.flip(cfg.magnetic_hi.c)
    psi_h_hi = b_hi * st.psi_h_hi + c_hi * de[-n:]
    h1d = h1d.at[-n:].add(cfg.curl_sign * (dt / mu_abs) * psi_h_hi)

    return st._replace(h1d=h1d, psi_h_lo=psi_h_lo, psi_h_hi=psi_h_hi)


def update_tfsf_1d_e(cfg: TFSFConfig, st: TFSFState, dx: float,
                      dt: float, t: float) -> TFSFState:
    """Advance 1D auxiliary E field: e1d^{n} → e1d^{n+1} + source injection.

    Uses h1d at time n+1/2 (must call update_tfsf_1d_h first).
    Call AFTER apply_tfsf_e (which reads h1d at time n+1/2).

    The 1D grid uses cfg.dx_1d (= dx/cos(theta) for oblique, dx for normal).
    """
    if getattr(cfg, "closed_box", False):
        if cfg.face_record is not None:
            return st._replace(step=st.step + 1)
        from rfx.sources.tfsf_discrete_plane_wave import update_line_e
        return update_line_e(cfg, st, dx, dt, t)
    n = cfg.n_cpml
    e1d, h1d = st.e1d, st.h1d
    dx_1d = cfg.dx_1d

    # 1D Ampere pair for +x propagation:
    #   Ez/Hy  ->  +sign
    #   Ey/Hz  ->  -sign
    dh = (h1d - jnp.concatenate([jnp.zeros(1), h1d[:-1]])) / dx_1d
    e1d = e1d + cfg.curl_sign * (dt / EPS_0) * dh

    # E CPML lo end
    psi_e_lo = cfg.b_cpml * st.psi_e_lo + cfg.c_cpml * dh[:n]
    e1d = e1d.at[:n].add(cfg.curl_sign * (dt / EPS_0) * psi_e_lo)

    # E CPML hi end
    b_hi = jnp.flip(cfg.b_cpml)
    c_hi = jnp.flip(cfg.c_cpml)
    psi_e_hi = b_hi * st.psi_e_hi + c_hi * dh[-n:]
    e1d = e1d.at[-n:].add(cfg.curl_sign * (dt / EPS_0) * psi_e_hi)

    e1d = e1d.at[cfg.src_idx].add(line_source_value(cfg, t, cfg.src_amp))

    return st._replace(e1d=e1d, psi_e_lo=psi_e_lo, psi_e_hi=psi_e_hi, step=st.step + 1)


def line_source_value(cfg, t, amplitude):
    """The soft-source sample an auxiliary line adds at time ``t``.

    One formula for the legacy 1-D line (``update_tfsf_1d_e``) and the closed
    box's discrete-plane-wave line (``tfsf_discrete_plane_wave.update_line_e``),
    which passes its amplitude already scaled by the polarization weight.
    Waveform selection (baked into cfg at init):
      differentiated_gaussian (legacy): s = -2·arg · exp(-arg²)
      modulated_gaussian (Meep-equivalent): s = cos(2π·fcen·(t-t0)) · exp(-arg²)
    MethodBConfig also uses the legacy kernel and keeps the original waveform
    fields; absent opt-in fields retain that legacy behavior.
    """
    arg = (t - cfg.src_t0) / cfg.src_tau
    env = jnp.exp(-(arg ** 2))
    custom_waveform = getattr(cfg, "custom_waveform", None)
    if custom_waveform is not None:
        return amplitude * custom_waveform(t)
    if cfg.src_waveform == "modulated_gaussian":
        carrier = jnp.cos(2.0 * jnp.pi * cfg.src_fcen * (t - cfg.src_t0))
        return amplitude * env * carrier
    if cfg.src_waveform == "continuous_wave":
        # raised-cosine ramp 0->1 over src_t0, then constant amplitude
        ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(t / cfg.src_t0, 0.0, 1.0)))
        return amplitude * ramp * jnp.sin(2.0 * jnp.pi * cfg.src_fcen * t)
    return amplitude * (-2.0 * arg) * env


# Lazy compatibility exports keep record definitions free of import cycles.
_LAZY_EXPORTS = {
    "apply_tfsf_e": "rfx.stepping.tfsf",
    "apply_tfsf_h": "rfx.stepping.tfsf",
    "measure_normal_incident_spectrum": "rfx.measurement.incident",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value
