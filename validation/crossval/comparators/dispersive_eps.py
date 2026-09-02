"""Dispersive permittivity comparator for crossval 22 (dispersive slab).

Pure numpy. No rfx import, no Meep import -- this module is the ONE place
where the three material conventions meet, and it must be importable by the
rfx case script, the Meep leg (a conda Meep environment without rfx) and the
unit tests alike.

Three conventions live here, and every function names which one it speaks:

1. **rfx** (``rfx/materials/debye.py``, ``rfx/materials/lorentz.py``):
   time dependence ``e^{+jωt}``, ``Im ε < 0`` for loss, SI units.

       debye   : ε(ω) = ε∞ + Δε / (1 + jωτ)
       lorentz : ε(ω) = ε∞ + Δε ω0² / (ω0² − ω² + 2jδω)      (κ = Δε ω0²)
       drude   : ε(ω) = ε∞ − ωp² / (ω² − jγω)                 (ω0 = 0, δ = γ/2, κ = ωp²)
       conductive : ε(ω) = ε' − jσ / (ω ε0)                     (cv23; ``materials.sigma``,
                                                                rfx/core/yee.py update_e)

   The Drude line is the form the ADE in ``lorentz.py`` actually realizes
   (``d²P/dt² + 2δ dP/dt + ω0² P = ε0 κ E`` with ``P ∝ e^{+jωt}``); the module
   docstring at ``lorentz.py:5-6`` prints the ``e^{−iωt}`` sign instead. This
   is recorded in the cv22 pre-declaration note, §2.

2. **Meep** (``meep.LorentzianSusceptibility`` / ``meep.DrudeSusceptibility``):
   time dependence ``e^{−iωt}``, ``Im ε > 0`` for loss, frequencies in units
   of ``c/a`` (``a`` = the Meep length unit in metres); the constructor takes
   ``frequency = ω_n / 2π`` and ``gamma = γ_n / 2π`` (both in c/a units) and
   realizes

       Lorentzian : ε(ω) = ε∞ + Σ σ_n ω_n² / (ω_n² − ω² − iωγ_n)
       Drude      : ε(ω) = ε∞ − Σ σ_n ω_n² / (ω² + iωγ_n)

   Conductivity (cv23) is Meep's ``Medium(epsilon=ε', D_conductivity=σ_D)``:
   ``ε(ω) = ε' (1 + i σ_D / ω)`` with ``ω = 2π f`` and BOTH ``f`` and ``σ_D``
   in units of ``c/a`` (python/geom.py ``Medium._get_epsmu``:
   ``epsmu = (1 + 1j/(2*np.pi*freqs) * conductivity) * epsmu``; the C++
   update ``step_generic.cpp`` is ``D ← ((1 − σ_D dt/2) D + dt curl H)/(1 +
   σ_D dt/2)`` with ``condinv = 1/(1 + σ_D dt/2)`` from ``structure.cpp``).
   Matching ``conj(ε_rfx) = ε'(1 + iσ/(ω ε0 ε'))`` gives
   ``σ_D = σ · a / (c ε0 ε')`` -- the ε' division is the D-vs-E trap and the
   missing 2π is the frequency-unit trap; both are unit-level falsifiers.

   Meep has NO first-order (Debye) susceptibility, and Debye is NOT the
   ``ω_n → 0`` limit of a Lorentzian (that limit is Drude). Debye is mapped
   as the OVERDAMPED limit: ``σ = Δε``, ``γ_n = ω_n² τ`` gives
   ``Δε / (1 − ω²/ω_n² − iωτ)`` -- Debye plus a residual of relative size
   ``(ω/ω_n)² / |1 − iωτ|``. ``fn_debye_map_hz`` sets ``ω_n``; its value is
   bounded above by Meep's own stability (``ω_n·dt_meep < 2``) and below by
   the residual one is willing to carry into the window. Both the residual
   and the exact mapped target are exposed so the window can carry it.

3. **discrete-time** (what the FDTD update actually realizes at a finite dt):
   z-transform of the recurrences with ``z = e^{jω dt}``:

       debye (Crank–Nicolson, debye.py:133-134,237):
           χ_num = Δε / (1 + j ω̃ τ),         ω̃ = (2/dt) tan(ω dt/2)
       lorentz/drude (explicit + CN damping, lorentz.py:154-156,249):
           χ_num = κ / (ω0² − ω̃² + 2jδ ω̂), ω̃ = (2/dt) sin(ω dt/2),
                                              ω̂ = sin(ω dt)/dt
       conductive (semi-implicit σ average, yee.py update_e:
           ca = (1 − σdt/2ε)/(1 + σdt/2ε), cb = (dt/ε)/(1 + σdt/2ε), i.e.
           ε(E^{n+1} − E^n)/dt + σ(E^{n+1} + E^n)/2 = curl H^{n+1/2}):
           ε_num = ε' − j σ_eff / (ω ε0),  σ_eff = σ · x / tan x, x = ω dt/2
           (relative to the common Yee factor 2j sin(x)/dt, as for the ADEs).
           Meep's D_conductivity update has the same form (step_generic.cpp),
           so the same factor applies at Meep's dt.

   Meep's Lorentzian/Drude update has the same characteristic polynomial
   with γ_n = 2δ, so the same function evaluated at Meep's dt is Meep's
   numerical ε.

The slab transfer matrix (``tmm_slab_rt``) is the same 2x2 characteristic
matrix as ``tests/test_dispersive_fresnel_validation.py::_tmm_slab_R`` and
``validation/crossval/04_multilayer_fresnel.py::fresnel_slab_RT``, written
for complex ε and returning both R and T.
"""

from __future__ import annotations

import math

import numpy as np

C0 = 299_792_458.0
TWO_PI = 2.0 * math.pi
EPS_0 = 8.8541878128e-12   # rfx/core/yee.py EPS_0 (CODATA 2018)

MODELS = ("debye", "lorentz", "drude", "conductive")

# Parameter names per model (rfx convention, SI).
PARAM_KEYS = {
    "debye": ("eps_inf", "delta_eps", "tau"),
    "lorentz": ("eps_inf", "delta_eps", "f0", "delta"),
    "drude": ("eps_inf", "fp", "gamma"),
    # cv23: eps_inf is the dispersionless eps', sigma in S/m (materials.sigma)
    "conductive": ("eps_inf", "sigma"),
}


def sigma_from_tan_delta(tan_delta: float, f_hz: float, eps_r: float) -> float:
    """σ [S/m] that realizes tan δ = σ/(ω ε0 ε') at f_hz (cv23 arm definition)."""
    return float(tan_delta) * TWO_PI * float(f_hz) * EPS_0 * float(eps_r)


def tan_delta_of(f_hz, params: dict):
    """tan δ(f) = σ/(ω ε0 ε') of a conductive slab."""
    w = TWO_PI * np.asarray(f_hz, dtype=float)
    return params["sigma"] / (w * EPS_0 * params["eps_inf"])


def skin_depth_m(f_hz, params: dict):
    """1/(k0 |Im n|) of the conductive slab (the e^{-1} field-amplitude depth)."""
    f = np.asarray(f_hz, dtype=float)
    n = np.sqrt(eps_analytic(f, "conductive", params))
    k0 = TWO_PI * f / C0
    return 1.0 / (k0 * np.abs(n.imag))


def _check(model: str, params: dict) -> None:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    missing = [k for k in PARAM_KEYS[model] if k not in params]
    if missing:
        raise ValueError(f"{model}: missing parameters {missing}")


# ---------------------------------------------------------------------------
# 1. rfx convention, continuous
# ---------------------------------------------------------------------------

def eps_analytic(f_hz, model: str, params: dict) -> np.ndarray:
    """Complex ε(f) in the rfx convention (e^{+jωt}, Im ε < 0 for loss)."""
    _check(model, params)
    w = TWO_PI * np.asarray(f_hz, dtype=float)
    ei = float(params["eps_inf"])
    if model == "debye":
        return ei + params["delta_eps"] / (1.0 + 1j * w * params["tau"])
    if model == "lorentz":
        w0 = TWO_PI * params["f0"]
        de, dl = params["delta_eps"], params["delta"]
        return ei + de * w0 ** 2 / (w0 ** 2 - w ** 2 + 2j * dl * w)
    if model == "conductive":
        return ei - 1j * params["sigma"] / (w * EPS_0)
    # drude
    wp = TWO_PI * params["fp"]
    g = params["gamma"]
    return ei - wp ** 2 / (w ** 2 - 1j * g * w)


def rfx_pole_args(model: str, params: dict) -> dict:
    """Arguments for the rfx constructors, so the case script cannot drift.

    debye   -> DebyePole(delta_eps, tau)
    lorentz -> lorentz_pole(delta_eps, omega_0, delta)
    drude   -> drude_pole(omega_p, gamma)
    """
    _check(model, params)
    if model == "conductive":
        raise ValueError("conductive: no pole object; the sigma path is materials.sigma "
                         "(Simulation.add_material(..., sigma=) or init_materials + .at[].set)")
    if model == "debye":
        return {"delta_eps": params["delta_eps"], "tau": params["tau"]}
    if model == "lorentz":
        return {"delta_eps": params["delta_eps"],
                "omega_0": TWO_PI * params["f0"], "delta": params["delta"]}
    return {"omega_p": TWO_PI * params["fp"], "gamma": params["gamma"]}


# ---------------------------------------------------------------------------
# 2. Meep convention
# ---------------------------------------------------------------------------

def to_meep(model: str, params: dict, *, a_m: float = 0.01,
            fn_debye_map_hz: float = 100e9) -> dict:
    """Meep susceptibility constructor arguments, in MEEP units and convention.

    Returns a dict with keys ``kind`` ("LorentzianSusceptibility" or
    "DrudeSusceptibility"), ``frequency``, ``gamma``, ``sigma`` (all as Meep
    expects them: ``frequency = ω_n/2π`` and ``gamma = γ_n/2π`` in units of
    ``c/a``), ``eps_inf``, ``a_m`` and, for Debye, ``debye_map`` describing
    the overdamped-Lorentz mapping.

    Derivation (Meep: ε = ε∞ + σ ω_n²/(ω_n² − ω² − iωγ_n), e^{−iωt}):

    lorentz : conj(rfx) = ε∞ + Δε ω0²/(ω0² − ω² − 2jδω)
              -> ω_n = ω0, γ_n = 2δ, σ = Δε.
    drude   : conj(rfx) = ε∞ − ωp²/(ω² + jγω); Meep Drude is
              ε∞ − σ ω_n²/(ω² + iωγ_n) -> ω_n = ωp, σ = 1, γ_n = γ.
    debye   : conj(rfx) = ε∞ + Δε/(1 − jωτ). Take a Lorentzian with
              σ = Δε, γ_n = ω_n² τ: σ ω_n²/(ω_n² − ω² − iω ω_n² τ)
              = Δε/(1 − (ω/ω_n)² − iωτ) -> Debye + residual (ω/ω_n)².
    Unit conversion: f[c/a] = f[Hz]·a/c, likewise for γ/2π.
    """
    _check(model, params)
    scale = a_m / C0  # Hz -> c/a
    ei = float(params["eps_inf"])
    if model == "conductive":
        # conj(rfx) = ε'(1 + iσ/(ω ε0 ε')); Meep: ε'(1 + iσ_D/ω_m) with
        # ω_m = ω a/c  ->  σ_D = σ/(ε0 ε') · a/c   (units c/a, dimensionless)
        sigma_si = float(params["sigma"])
        return {"kind": "D_conductivity",
                "D_conductivity": sigma_si / (EPS_0 * ei) * scale,
                "sigma_si": sigma_si, "eps_inf": ei, "a_m": a_m}
    if model == "lorentz":
        omega_n = TWO_PI * params["f0"]
        gamma_n = 2.0 * params["delta"]
        return {"kind": "LorentzianSusceptibility",
                "frequency": omega_n / TWO_PI * scale,
                "gamma": gamma_n / TWO_PI * scale,
                "sigma": float(params["delta_eps"]),
                "eps_inf": ei, "a_m": a_m}
    if model == "drude":
        omega_n = TWO_PI * params["fp"]
        gamma_n = params["gamma"]
        return {"kind": "DrudeSusceptibility",
                "frequency": omega_n / TWO_PI * scale,
                "gamma": gamma_n / TWO_PI * scale,
                "sigma": 1.0,
                "eps_inf": ei, "a_m": a_m}
    # debye -> overdamped Lorentzian
    omega_n = TWO_PI * fn_debye_map_hz
    gamma_n = omega_n ** 2 * params["tau"]
    return {"kind": "LorentzianSusceptibility",
            "frequency": omega_n / TWO_PI * scale,
            "gamma": gamma_n / TWO_PI * scale,
            "sigma": float(params["delta_eps"]),
            "eps_inf": ei, "a_m": a_m,
            "debye_map": {"fn_hz": fn_debye_map_hz,
                          "omega_n_rad_s": omega_n,
                          "gamma_n_rad_s": gamma_n,
                          "residual_rel_formula": "(omega/omega_n)^2 / |1 - i omega tau|"}}


def eps_meep_convention(f_hz, meep_params: dict) -> np.ndarray:
    """ε(ω) that Meep realizes for ``meep_params`` (from ``to_meep``), in
    MEEP's convention (e^{−iωt}, Im ε > 0 for loss), evaluated at SI ``f_hz``.

    This is the reconstruction the unit test compares against
    ``conj(eps_analytic)``; the Meep leg additionally queries
    ``meep.Medium.epsilon(f)`` where the installed Meep exposes it.
    """
    f_m = np.asarray(f_hz, dtype=float) * meep_params["a_m"] / C0
    w = TWO_PI * f_m
    if meep_params["kind"] == "D_conductivity":
        # python/geom.py Medium._get_epsmu: (1 + 1j/(2π f) σ_D) · ε
        return meep_params["eps_inf"] * (1.0 + 1j * meep_params["D_conductivity"] / w)
    wn = TWO_PI * meep_params["frequency"]
    gn = TWO_PI * meep_params["gamma"]
    s = meep_params["sigma"]
    ei = meep_params["eps_inf"]
    if meep_params["kind"] == "LorentzianSusceptibility":
        return ei + s * wn ** 2 / (wn ** 2 - w ** 2 - 1j * w * gn)
    if meep_params["kind"] == "DrudeSusceptibility":
        return ei - s * wn ** 2 / (w ** 2 + 1j * w * gn)
    raise ValueError(meep_params["kind"])


def eps_debye_mapped_target(f_hz, params: dict, *, fn_debye_map_hz: float) -> np.ndarray:
    """The overdamped-Lorentz ε that the Debye Meep mapping realizes, in the
    rfx convention: ε∞ + Δε / (1 − (ω/ω_n)² + jωτ)."""
    w = TWO_PI * np.asarray(f_hz, dtype=float)
    wn = TWO_PI * fn_debye_map_hz
    return params["eps_inf"] + params["delta_eps"] / (1.0 - (w / wn) ** 2 + 1j * w * params["tau"])


def debye_mapping_residual(f_hz, params: dict, *, fn_debye_map_hz: float) -> np.ndarray:
    """Relative residual |ε_mapped − ε_debye| / |ε_debye| of the Debye mapping."""
    e = eps_analytic(f_hz, "debye", params)
    em = eps_debye_mapped_target(f_hz, params, fn_debye_map_hz=fn_debye_map_hz)
    return np.abs(em - e) / np.abs(e)


# ---------------------------------------------------------------------------
# 3. Discrete-time (ADE) permittivity
# ---------------------------------------------------------------------------

def sigma_warp(w, dt: float):
    """σ_eff/σ = x/tan(x), x = ω dt/2: the semi-implicit conductivity average
    (yee.py update_e; Meep step_generic.cpp) measured against the common Yee
    temporal factor 2j sin(x)/dt. Derivation: E^{n+1}(1+s) = (1−s)E^n +
    (dt/ε) C^{n+1/2}, s = σdt/2ε; with z = e^{jωdt}, E/C = (dt/ε)/[(1+s)z^{1/2}
    − (1−s)z^{−1/2}] = 1/[ε·2j sin x/dt + σ cos x] = 1/(jω̂ ε0 ε_num), ω̂ = 2 sin x/dt,
    so ε0 ε_num = ε0 ε' + σ cos x/(jω̂) = ε0 ε' − jσ (x/tan x)/ω."""
    x = np.asarray(w, dtype=float) * dt / 2.0
    return x / np.tan(x)


def eps_numerical_ade(f_hz, model: str, params: dict, dt: float) -> np.ndarray:
    """ε_num(f) realized by the ADE recurrences at timestep ``dt`` (rfx
    convention). Same polynomial as Meep's Lorentzian/Drude update, so at
    Meep's dt this is Meep's numerical ε too (Lorentz/Drude only; the Debye
    Meep leg is an overdamped Lorentzian -- use ``eps_numerical_meep``)."""
    _check(model, params)
    w = TWO_PI * np.asarray(f_hz, dtype=float)
    ei = float(params["eps_inf"])
    if model == "debye":
        w_bil = (2.0 / dt) * np.tan(w * dt / 2.0)
        return ei + params["delta_eps"] / (1.0 + 1j * w_bil * params["tau"])
    if model == "conductive":
        return ei - 1j * params["sigma"] * sigma_warp(w, dt) / (w * EPS_0)
    w_s = (2.0 / dt) * np.sin(w * dt / 2.0)
    w_h = np.sin(w * dt) / dt
    if model == "lorentz":
        w0 = TWO_PI * params["f0"]
        kappa = params["delta_eps"] * w0 ** 2
        dl = params["delta"]
    else:
        w0 = 0.0
        kappa = (TWO_PI * params["fp"]) ** 2
        dl = params["gamma"] / 2.0
    return ei + kappa / (w0 ** 2 - w_s ** 2 + 2j * dl * w_h)


def eps_numerical_meep(f_hz, meep_params: dict, dt_meep_s: float) -> np.ndarray:
    """Meep's numerical ε for ``meep_params`` at Meep timestep ``dt_meep_s``
    (seconds), returned in the RFX convention (conjugated) so it can be fed
    to ``tmm_slab_rt`` directly. Meep's update:
    P^{n+1} = [(2 − ω_n²dt²) P^n − (1 − γdt/2) P^{n−1} + σ ω_n² dt² E^n] / (1 + γdt/2)
    (Drude: the ω_n²dt² in the first bracket is dropped)."""
    w = TWO_PI * np.asarray(f_hz, dtype=float)
    scale = C0 / meep_params["a_m"]  # c/a -> Hz
    if meep_params["kind"] == "D_conductivity":
        # step_generic.cpp: D <- ((1 - σ_D dt/2) D + dt curl H) / (1 + σ_D dt/2):
        # the same semi-implicit average as rfx, hence the same x/tan x factor.
        sig_hz = meep_params["D_conductivity"] * scale          # 1/s
        eps_meep = meep_params["eps_inf"] * (1.0 + 1j * sig_hz * sigma_warp(w, dt_meep_s) / w)
        return np.conj(eps_meep)
    wn = TWO_PI * meep_params["frequency"] * scale
    gn = TWO_PI * meep_params["gamma"] * scale
    s = meep_params["sigma"]
    ei = meep_params["eps_inf"]
    dt = dt_meep_s
    w_s = (2.0 / dt) * np.sin(w * dt / 2.0)
    w_h = np.sin(w * dt) / dt
    wn2 = wn ** 2 if meep_params["kind"] == "LorentzianSusceptibility" else 0.0
    # Meep convention: ε∞ + σ ω_n²/(ω_n² − ω̃² − iγ ω̂); conjugate to rfx.
    eps_meep = ei + s * wn ** 2 / (wn2 - w_s ** 2 - 1j * gn * w_h)
    return np.conj(eps_meep)


# ---------------------------------------------------------------------------
# Slab transfer matrix
# ---------------------------------------------------------------------------

def tmm_slab_rt(f_hz, eps, d_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Normal-incidence power R, T of a single slab of complex ε in vacuum.

    2x2 characteristic matrix, y0 = ys = 1 (vacuum both sides):
        M = [[cos δ, j sin δ / n], [j n sin δ, cos δ]],  δ = n k0 d,
        r = (M11 + M12 − M21 − M22) / (M11 + M12 + M21 + M22),
        t = 2 / (M11 + M12 + M21 + M22).
    M is even in n, so the sqrt branch does not matter.
    """
    f = np.asarray(f_hz, dtype=float)
    n = np.sqrt(np.asarray(eps, dtype=complex))
    k0 = TWO_PI * f / C0
    dl = n * k0 * d_m
    cd, sd = np.cos(dl), np.sin(dl)
    m11, m12, m21, m22 = cd, 1j * sd / n, 1j * n * sd, cd
    den = m11 + m12 + m21 + m22
    r = (m11 + m12 - m21 - m22) / den
    t = 2.0 / den
    return np.abs(r) ** 2, np.abs(t) ** 2


def tmm_layers_rt(f_hz, layers) -> tuple[np.ndarray, np.ndarray]:
    """Normal-incidence R, T of a stack of homogeneous layers in vacuum:
    ``layers`` = [(eps (scalar or per-bin array), thickness_m), ...] from the
    incidence side. One layer reduces to ``tmm_slab_rt`` exactly."""
    f = np.asarray(f_hz, dtype=float)
    k0 = TWO_PI * f / C0
    M = np.broadcast_to(np.eye(2, dtype=complex), (f.size, 2, 2)).copy()
    for eps, d in layers:
        n = np.sqrt(np.asarray(eps, dtype=complex) * np.ones(f.size))
        ph = n * k0 * d
        m = np.empty((f.size, 2, 2), dtype=complex)
        m[:, 0, 0] = np.cos(ph); m[:, 0, 1] = 1j * np.sin(ph) / n
        m[:, 1, 0] = 1j * n * np.sin(ph); m[:, 1, 1] = np.cos(ph)
        M = M @ m
    den = M[:, 0, 0] + M[:, 0, 1] + M[:, 1, 0] + M[:, 1, 1]
    r = (M[:, 0, 0] + M[:, 0, 1] - M[:, 1, 0] - M[:, 1, 1]) / den
    t = 2.0 / den
    return np.abs(r) ** 2, np.abs(t) ** 2


def yee_lattice_slab_rt(f_hz, eps_r: float, sigma: float, d_m: float, dx: float, dt: float,
                        *, n_vac: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """EXACT time-harmonic R, T of the 1-D Yee lattice for a staircase slab
    (cv23 note section 12): ``round(d/dx)`` E-nodes carry (eps', sigma) with
    the semi-implicit sigma average, vacuum nodes either side; the normal-
    incidence TMz rig with periodic y IS this lattice. With z = e^{j w dt},
    w_hat = 2 sin(w dt/2)/dt, x = w dt/2, the update equations reduce to
        H_{i+1/2} - H_{i-1/2} = dx (j w_hat eps_i + sigma_i cos x) E_i
        E_{i+1}   - E_i       = dx (j w_hat mu0) H_{i+1/2}
    which are marched from a unit transmitted lattice plane wave (vacuum
    lattice wavenumber k = (2/dx) asin(w_hat dx/2c)) back to the incidence
    side, where two nodes are decomposed into incident + reflected. Contains
    the bulk numerical dispersion of the slab, the node interface and the
    sigma warp at once; converges to ``tmm_slab_rt`` as dx -> 0 (second order).
    """
    mu0 = 1.0 / (C0 ** 2 * EPS_0)
    f = np.asarray(f_hz, dtype=float)
    w = TWO_PI * f
    x = w * dt / 2.0
    w_hat = 2.0 * np.sin(x) / dt
    cx = np.cos(x)
    n_slab = int(round(d_m / dx))
    N = 2 * n_vac + n_slab + 1
    eps = np.full(N, EPS_0); sig = np.zeros(N)
    eps[n_vac:n_vac + n_slab] = EPS_0 * eps_r
    sig[n_vac:n_vac + n_slab] = sigma
    k = (2.0 / dx) * np.arcsin(w_hat * dx / (2.0 * C0))
    R = np.empty(f.size); T = np.empty(f.size)
    for m in range(f.size):
        y = 1j * w_hat[m] * eps + sig * cx[m]
        zm = 1j * w_hat[m] * mu0
        E = np.zeros(N, complex); H = np.zeros(N - 1, complex)
        E[N - 1] = 1.0
        E[N - 2] = np.exp(1j * k[m] * dx)
        H[N - 2] = (E[N - 1] - E[N - 2]) / (dx * zm)
        for i in range(N - 2, 0, -1):
            H[i - 1] = H[i] - dx * y[i] * E[i]
            E[i - 1] = E[i] - dx * zm * H[i - 1]
        M = np.array([[1.0, 1.0], [np.exp(-1j * k[m] * dx), np.exp(1j * k[m] * dx)]])
        a, b = np.linalg.solve(M, E[:2])
        R[m] = abs(b / a) ** 2
        T[m] = abs(1.0 / a) ** 2
    return R, T

