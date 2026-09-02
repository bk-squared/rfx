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

MODELS = ("debye", "lorentz", "drude")

# Parameter names per model (rfx convention, SI).
PARAM_KEYS = {
    "debye": ("eps_inf", "delta_eps", "tau"),
    "lorentz": ("eps_inf", "delta_eps", "f0", "delta"),
    "drude": ("eps_inf", "fp", "gamma"),
}


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
