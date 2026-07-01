"""S-parameter extraction methods for :class:`Simulation`.

Import contract (Part B Stage 2 refactor):
  This module is a transitional mixin. It must import ONLY from
  ``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy.
  It must NEVER do ``from rfx.api import ...`` or ``from . import ...``
  the package, to keep ``rfx/api/__init__.py`` the sole composition point.

The methods here were moved verbatim out of ``rfx/api/__init__.py``'s
``class Simulation`` body. They are pure structural relocations — same
indentation, decorators, signatures, docstrings, and logic. ``Simulation``
inherits ``_SparamMixin`` so every method below remains a bound method on
``Simulation`` instances; all existing call-sites are unaffected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import CoaxialPort
from rfx.sources.waveguide_port import (
    extract_waveguide_s_matrix,
    extract_waveguide_s_matrix_flux,
    extract_waveguide_s_params_normalized,
    extract_multimode_s_matrix,
    extract_multimode_s_matrix_flux,
    waveguide_plane_positions,
)

from rfx.nonuniform import NonUniformGrid

from rfx.api._spec import (
    WaveguideSMatrixResult,
    CoaxialSMatrixResult,
    CoaxialLineReflectionResult,
    MSLSMatrixResult,
    _WaveguidePortEntry,
    _MSLPortEntry,
)


def _msl_cell_profile(grid, axis: str, n: int) -> np.ndarray:
    """Per-cell size array (length ``n``, full/padded) along ``axis`` for
    MSL V/I integration. Graded-mesh aware.

    ``NonUniformGrid`` (a NamedTuple) stores per-cell spacings as
    ``dx_arr`` / ``dy_arr`` / ``dz`` and exposes NO ``*_profile``
    attributes — so the legacy ``getattr(grid, "dy_profile", None)``
    fell through to ``np.full(n, grid.dx)``, i.e. the SCALAR boundary-x
    cell for every transverse cell (wrong axis AND scalar-not-per-cell).
    This reads the real per-cell array on a NU grid. On a uniform
    ``Grid`` it is byte-identical to the legacy path (``Grid`` is not a
    ``NonUniformGrid``, so the per-cell branch is never taken): the
    ``*_profile`` attr if present, else ``np.full(n, grid.dx)`` — the
    legacy behaviour of using ``grid.dx`` for every axis is preserved.
    """
    if isinstance(grid, NonUniformGrid):
        per_cell = {"x": grid.dx_arr, "y": grid.dy_arr, "z": grid.dz}[axis]
        a = np.asarray(per_cell, dtype=float)
        if a.shape != (n,):
            # The NU branch is authoritative — never silently fall back to a
            # scalar boundary-dx fill (that is the exact wrong-number bug this
            # helper exists to fix). A shape mismatch is a wiring error.
            raise ValueError(
                f"NonUniformGrid {axis} per-cell profile shape {a.shape} "
                f"!= expected ({n},)."
            )
        return a
    attr = {"x": "dx_profile", "y": "dy_profile", "z": "dz_profile"}[axis]
    prof = getattr(grid, attr, None)
    if prof is not None:
        return np.asarray(prof, dtype=float)
    return np.full(n, float(grid.dx), dtype=float)


def _warn_if_nonpassive_smatrix(
    result,
    *,
    extractor: str,
    strict: bool = False,
    passivity_tol: float = 0.10,
) -> None:
    """Auto-run the passivity/finiteness self-check on a freshly-extracted
    S-matrix and surface a non-physical result as a warning (or raise when
    ``strict``).

    This operationalizes the R5 "no surface-metric verdict" discipline:
    a passive structure cannot scatter more power than it
    receives, so a per-column power > 1 (e.g. ``|S11| > 1`` on a one-port)
    means the *extractor* is wrong — mismeasured current sign/scale or a
    bad reference plane — and the S-parameters are untrustworthy, NOT that
    the device is exotic. Waveguide and coaxial extractors previously had
    no such self-check; only ``compute_msl_s_matrix`` did. Wiring the
    existing :func:`rfx.validation.validate_port_smatrix` in here is the
    guard that would have short-circuited the multi-session WR-90 ``|S11|``
    chase recorded in durable memory.

    Tracer-safe: under ``jax.grad`` / ``jax.jit`` tracing ``result.s_params``
    is an abstract tracer with no concrete value, so the numpy-based check is
    skipped entirely. The diagnostic is for the eager forward call (the
    common research-tool usage); it deliberately does not fire per optimizer
    iteration.
    """
    s = getattr(result, "s_params", None)
    if s is None:
        return
    try:
        if isinstance(s, jax.core.Tracer):
            return
    except Exception:
        pass
    try:
        s_np = np.asarray(s)
        f_np = np.asarray(result.freqs)
    except Exception:
        # Traced / non-materializable — never let a diagnostic break the
        # numeric return path.
        return

    from rfx.validation import validate_port_smatrix

    report = validate_port_smatrix(
        s_params=s_np,
        freqs=f_np,
        port_names=tuple(result.port_names),
        source=extractor,
        check_passivity=True,
        passivity_limit=1.0,
        passivity_tol=float(passivity_tol),
    )
    bad = [
        i for i in report.issues
        if i.code in ("passivity_violation", "nonfinite_sparams")
    ]
    if not bad:
        return
    detail = "; ".join(f"{i.code}: {i.message}" for i in bad)
    msg = (
        f"{extractor}: extracted S-matrix failed a passivity/finiteness "
        f"self-check — {detail}. A passive structure cannot have column "
        f"power > 1; this almost always means the extractor (current "
        f"sign/scale or reference plane) is wrong and the S-parameters are "
        f"UNRELIABLE. Inspect the V/I dump via "
        f"rfx.validation.validate_port_smatrix / replay_smatrix_from_vi_dump "
        f"before trusting or optimizing against these numbers."
    )
    if strict:
        raise ValueError(msg)
    import warnings as _w
    _w.warn(msg, stacklevel=3)


def _finalize_sparam_result(
    result,
    *,
    extractor: str,
    strict: bool,
    passivity_tol: float = 0.10,
):
    """Shared two-run-S-param epilogue: run the passivity/finiteness guard on a
    freshly-assembled S-matrix result, then return it unchanged.

    This is the one genuinely-common piece of the per-family two-run S-param
    flow at the orchestration layer (W6.4): both
    :meth:`_SparamMixin.compute_waveguide_s_matrix` (NU, multi-mode, and
    single-mode return paths) and :meth:`_SparamMixin.compute_coaxial_s_matrix`
    assemble a family-specific ``*SMatrixResult`` and then invoke
    :func:`_warn_if_nonpassive_smatrix` immediately before returning. The
    per-port drive loop, vacuum-reference override, and rectangular-DFT
    windowing live behind the family-specific extractors (waveguide:
    ``rfx.sources.waveguide_port``; coax: the inline single-run plane-source
    path) and are deliberately *not* unified here — they share no code at this
    layer, so a wider scaffold would be a leaky abstraction.

    ``passivity_tol`` defaults to the tight 0.10 bound (matching the coax call
    site). The waveguide path passes a ``normalize``-aware tolerance.
    """
    _warn_if_nonpassive_smatrix(
        result,
        extractor=extractor,
        strict=strict,
        passivity_tol=passivity_tol,
    )
    return result


class _SparamMixin:
    """S-parameter extraction methods mixed into :class:`Simulation`."""

    def compute_waveguide_s_matrix(
        self,
        *,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        normalize: bool | str = False,
        subpixel_smoothing: bool | str = False,
        eps_override: "jnp.ndarray | None" = None,
        sigma_override: "jnp.ndarray | None" = None,
        checkpoint_segments: int | None = None,
        strict_passivity: bool = False,
    ) -> WaveguideSMatrixResult:
        """Compute a theoretically clean axis-normal boundary-aperture waveguide S-matrix.

        Parameters
        ----------
        num_periods : float
            Length of the FDTD run (in source-period multiples) used to
            derive ``n_steps`` when ``n_steps`` is not given. The
            spectra are computed POST-SCAN from the recorded modal V/I
            time series via a rectangular full-record DFT (matching
            OpenEMS's ``utilities.DFT_time2freq``); ``num_periods``
            therefore governs both the CPML drain horizon AND the DFT
            integration window. Phase 2 cleanup (2026-04-25) removed
            the legacy ``num_periods_dft`` early-gate knob — the rect
            full-record DFT is finite-energy on the recorded transient
            so no gating is needed even on strong reflectors.
        normalize : bool or "flux"
            Controls the S-parameter extraction algorithm:

            ``False`` (default) — modal V/I decomposition, no reference
            run.  Magnitude includes Yee impedance mismatch
            (Z_TE_num/Z_TE_exact ≈ 3 % at dx/λ = 0.07).  Use for
            |S11| of strong reflectors (PEC short, high-Q resonators)
            where this error is smaller than the ±10–20 % round-trip
            dispersion error introduced by ``normalize=True``.

            ``True`` — two-run modal normalization.  Cancels one-way
            Yee dispersion for **transmission** (off-diagonal) by
            dividing device outgoing waves by reference outgoing waves
            at the same port.  **Does not** cancel dispersion for
            reflection (round-trip vs one-way path mismatch); use
            ``normalize=False`` or ``normalize="flux"`` for S11 of
            strong reflectors.

            ``"flux"`` — hybrid power-flux extraction.  Magnitude from
            Poynting-vector DFT (|S|² = P_flux / P_inc), phase from
            modal V/I.  Corrects both the Z_TE impedance-mismatch error
            in S11 and the round-trip dispersion error in the
            ``normalize=True`` diagonal formula.  Costs 2 × N_ports
            FDTD runs (same as ``normalize=True``).
        checkpoint_segments : int or None
            Segmented gradient checkpointing for the **uniform** waveguide
            AD path (issue #73 / PR #125).  Splits the ``n_steps`` scan
            into ``K`` segments that are rematerialised via
            ``jax.checkpoint`` during the backward pass, reducing peak
            reverse-mode memory from O(n_steps·|carry|) to
            O((K + n_steps/K)·|carry|) (≈ O(√n_steps·|carry|) at the
            optimal K ≈ √n_steps, at ≈ 2× backward compute cost).
            ``K`` is forwarded to ``rfx.simulation.run`` with
            ``checkpoint=True``; ``K`` MUST exactly divide the
            auto-computed ``n_steps`` (the runner rejects non-divisors —
            choose the nearest divisor of √n_steps; padding is rejected
            because it would shift the V/I DFT windows).  Default
            ``None`` is byte-identical to the pre-checkpoint scan.

            On a NON-uniform mesh (``dx_profile`` / ``dy_profile``,
            issue #73) ``checkpoint_segments=K`` is now supported: ``K`` is
            translated to the NU runner's ``checkpoint_every`` chunk size — the
            divisor of ``n_steps`` nearest to ``n_steps/K`` — and applied to the
            *device* run only (the vacuum reference is constant in the design
            variable). The chunk MUST divide ``n_steps`` (same as the uniform
            path): a non-divisor chunk would let the NU runner's zero-padding add
            spurious ring-down steps to the carry-accumulated flux DFT and shift
            the S-matrix. With an exact divisor the result is forward-IDENTICAL
            and the ≈O(√n_steps) tape reduction is realised under ``jax.grad``
            with ``normalize='flux'`` + an ``eps_override`` / ``sigma_override``
            design variable.
        """
        if not normalize:
            import warnings
            warnings.warn(
                "compute_waveguide_s_matrix(normalize=False): S21 and "
                "S-parameter phase include Yee numerical dispersion. "
                "For S21 accuracy and reciprocity use normalize=True. "
                "For |S11| of strong reflectors (PEC short, resonators) "
                "normalize=False is more accurate — see the normalize "
                "parameter docstring.",
                stacklevel=2,
            )
        if self._ports or self._tfsf:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported together with lumped ports or TFSF"
            )
        if self._periodic_axes:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported with manual periodic-axis overrides"
            )
        if len(self._waveguide_ports) < 2:
            raise ValueError(
                "compute_waveguide_s_matrix() requires at least two waveguide ports"
            )

        entries = list(self._waveguide_ports)
        if any(entry.probe_plane is not None for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() does not use per-port probe_plane; use reference_plane only or leave probe_plane unset"
            )
        if any(entry.calibration_preset not in (None, "measured") for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() currently supports only measured/default reference planes or explicit reference_plane overrides"
            )

        # Non-uniform-mesh dispatch. Earlier the uniform scan ran with
        # the coarse boundary dx and silently ignored ``dx_profile`` /
        # ``dy_profile`` (handover v2 experiment 12). The dedicated NU
        # two-run extractor below is enabled when its supported scope
        # is met (``normalize=True``, single-mode ports); otherwise
        # raise so the user is not given silently-wrong numbers.
        if self._dx_profile is not None or self._dy_profile is not None:
            if checkpoint_segments is not None and checkpoint_segments < 1:
                raise ValueError(
                    f"checkpoint_segments must be >= 1, got {checkpoint_segments}"
                )
            unsupported = []
            if normalize is not True and normalize != "flux":
                unsupported.append("normalize=True or normalize='flux' is required")
            if any(entry.n_modes > 1 for entry in entries):
                unsupported.append("multi-mode ports (n_modes>1) are not supported")
            # The differentiable eps/sigma AD channel is wired on the NU
            # path only for normalize='flux' (mirrors the uniform PR #172
            # flux-AD fix): the flux extractor is now jnp-native end-to-end
            # so a traced eps_override flows into the device Yee update and
            # back through the S-matrix. normalize=True is kept out of scope
            # — its diagonal a_inc_ref denominator carries the #88 band-edge
            # fragility, so accepting eps_override there could yield
            # silently-wrong gradients.
            if eps_override is not None and normalize != "flux":
                unsupported.append(
                    "eps_override (differentiable AD channel) on the NU path "
                    "requires normalize='flux'"
                )
            if sigma_override is not None and normalize != "flux":
                unsupported.append(
                    "sigma_override (differentiable AD channel) on the NU path "
                    "requires normalize='flux'"
                )
            if subpixel_smoothing:
                unsupported.append("subpixel_smoothing is not supported")
            if unsupported:
                raise NotImplementedError(
                    "compute_waveguide_s_matrix() on a non-uniform mesh "
                    "(dx_profile / dy_profile) supports normalize=True or "
                    "normalize='flux' and single-mode ports. "
                    + "; ".join(unsupported)
                    + ". Drop the dx/dy profile to use the uniform lane."
                )
            _res_nu = self._compute_waveguide_s_matrix_nu(
                n_steps=n_steps,
                num_periods=num_periods,
                normalize=normalize,
                eps_override=eps_override,
                sigma_override=sigma_override,
                checkpoint_segments=checkpoint_segments,
            )
            return _finalize_sparam_result(
                _res_nu,
                extractor="compute_waveguide_s_matrix",
                strict=strict_passivity,
                # normalize=False carries documented Yee-dispersion + band-edge
                # |S11| overshoot (validated paths reach ~1.4-1.7), so use a
                # loose bound there that still catches gross extractor bugs
                # (|S11|>>1); normalize=True/"flux" correct dispersion -> tight.
                passivity_tol=2.0 if normalize is False else 0.10,
            )

        grid = self._build_grid()
        base_materials, debye_spec, lorentz_spec, pec_mask_wg, pec_shapes, boundary_pec_shapes, _ = self._assemble_materials(grid)
        # Waveguide S-matrix runner doesn't support pec_mask yet.
        # Fold PEC mask back into high sigma for compatibility.
        # **Stage 2 caveat**: when ``subpixel_smoothing="kottke_pec"`` is
        # active (use_kottke_pec, computed below), the inverse-eps
        # tensor encodes the PEC zero directly (inv = 0 freezes the
        # field). Folding pec_mask to sigma=1e10 then would conflict
        # with the Yee-stagger offsets in inv_xx/yy/zz: pec_mask is
        # per-cell-center, but inv_xx is at Ex(i+0.5, j, k) offsets,
        # so PEC boundary cells can have sigma=1e10 AND a fractional
        # inv > 0 — that combo blows up Ca ≈ -1 and field NaNs.
        # Skipped for Stage 2; the Kottke union (inv=0 inside PEC,
        # fractional at boundary) is the single source of truth.
        _use_kottke_pec_early = (subpixel_smoothing == "kottke_pec")
        if pec_mask_wg is not None and not _use_kottke_pec_early:
            base_materials = base_materials._replace(
                sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma))
        materials = base_materials
        # G-AD-WIRE-WG2: public eps_override / sigma_override channel.
        # Mirror the MSL pattern: replace eps_r / sigma on the assembled
        # materials *after* the PEC fold so PEC boundaries are untouched.
        if eps_override is not None:
            materials = materials._replace(eps_r=eps_override)
        if sigma_override is not None:
            materials = materials._replace(sigma=sigma_override)
        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)
        _, debye, lorentz = self._init_dispersion(materials, grid.dt, debye_spec, lorentz_spec)

        def _resolve_freqs(entry: _WaveguidePortEntry) -> jnp.ndarray:
            if entry.freqs is not None:
                return entry.freqs
            return jnp.linspace(self._freq_max / 10, self._freq_max, entry.n_freqs)

        freqs = _resolve_freqs(entries[0])
        for entry in entries[1:]:
            entry_freqs = _resolve_freqs(entry)
            if entry_freqs.shape != freqs.shape or not np.allclose(np.asarray(entry_freqs), np.asarray(freqs)):
                raise ValueError("waveguide S-matrix requires matching frequency grids on all ports")

        # Build configs — may be a single config or a list of configs per port
        has_multimode = any(entry.n_modes > 1 for entry in entries)
        raw_cfgs = [self._build_waveguide_port_config(entry, grid, freqs, n_steps) for entry in entries]

        # Unify source waveform across all ports so that the S-matrix
        # extraction uses identical excitation.  Different source spectra
        # (from mismatched f0/bandwidth) cause S11 ≠ S22 artifacts in the
        # unnormalized path because V/I decomposition error varies with
        # frequency.  Use port 0's waveform as the canonical source.
        def _flatten_cfgs(cfgs):
            out = []
            for c in cfgs:
                if isinstance(c, list):
                    out.extend(c)
                else:
                    out.append(c)
            return out

        flat0 = _flatten_cfgs(raw_cfgs)
        ref_t0 = flat0[0].src_t0
        ref_tau = flat0[0].src_tau
        need_unify = any(
            c.src_t0 != ref_t0 or c.src_tau != ref_tau for c in flat0[1:]
        )
        if need_unify:
            raw_cfgs = [
                cfg._replace(src_t0=ref_t0, src_tau=ref_tau)
                if not isinstance(cfg, list)
                else [c._replace(src_t0=ref_t0, src_tau=ref_tau) for c in cfg]
                for cfg in raw_cfgs
            ]

        # Compute Kottke per-component smoothed permittivity if requested.
        # Shared by both single-mode and multi-mode paths.
        # Mirrors rfx/runners/uniform.py: shape_eps_pairs from sim geometry,
        # then compute_smoothed_eps. The reference run is vacuum and has no
        # ε interfaces, so it always passes aniso_eps=None inside the
        # extractor.
        # Stage 2 unified path: subpixel_smoothing="kottke_pec" routes
        # through compute_inv_eps_tensor_diag and skips the Stage 1
        # eps_correction + apply_conformal_pec chain entirely. Both
        # device and reference (vacuum) runs see the same boundary-
        # face PEC walls, so the inverse-permittivity tensor is
        # computed twice (once per material context).
        use_kottke_pec = (subpixel_smoothing == "kottke_pec")
        aniso_eps = None
        aniso_inv_eps = None
        ref_aniso_inv_eps = None
        if use_kottke_pec:
            from rfx.geometry.smoothing import compute_inv_eps_tensor_diag
            shape_eps_pairs = [
                (entry.shape, self._resolve_material(entry.material_name).eps_r)
                for entry in self._geometry
            ]
            aniso_inv_eps = compute_inv_eps_tensor_diag(
                grid,
                dielectric_shapes=shape_eps_pairs,
                pec_shapes=pec_shapes or [],
                background_eps=1.0,
            )
            # Reference run is empty guide with same boundary walls only.
            # Must NOT include interior PEC geometry (e.g. PEC short box):
            # if device and reference share the same obstacle, both DFTs
            # are identical and (device - reference) / incident = 0.
            ref_aniso_inv_eps = compute_inv_eps_tensor_diag(
                grid,
                dielectric_shapes=[],
                pec_shapes=boundary_pec_shapes,
                background_eps=1.0,
            )
            # Yee-stagger correction: the Kottke union reaches inv=0
            # on Yee-staggered components only when the cell-center
            # AND the offset position are both inside the PEC shape.
            # For thin PEC obstacles (e.g. a 1-cell-wide PEC short),
            # cell-center is inside but Ey/Ez Yee positions are at
            # cell-corner offsets that fall *outside* the box → inv
            # remains 1 (vacuum). That leaves the H field free to
            # propagate inside the PEC region and seeds late-time
            # exponential growth.
            #
            # Fix: where ``pec_mask`` (cell-center binary) is True,
            # force all three inv components to zero. This is the
            # cell-center analogue of Stage 1's sigma=1e10 fold,
            # without the Ca→-1 instability that the sigma fold has
            # at Yee-staggered cells where inv > 0.
            if pec_mask_wg is not None:
                inv_xx, inv_yy, inv_zz = aniso_inv_eps
                inv_xx = jnp.where(pec_mask_wg, 0.0, inv_xx)
                inv_yy = jnp.where(pec_mask_wg, 0.0, inv_yy)
                inv_zz = jnp.where(pec_mask_wg, 0.0, inv_zz)
                aniso_inv_eps = (inv_xx, inv_yy, inv_zz)
                # pec_mask_wg marks interior PEC geometry (e.g. the PEC
                # short). The reference run has no interior PEC — do NOT
                # apply pec_mask_wg to ref_aniso_inv_eps, or the reference
                # becomes identical to the device and S11 = 0.
        elif subpixel_smoothing:
            from rfx.geometry.smoothing import compute_smoothed_eps
            shape_eps_pairs = [
                (entry.shape, self._resolve_material(entry.material_name).eps_r)
                for entry in self._geometry
            ]
            if shape_eps_pairs:
                aniso_eps = compute_smoothed_eps(
                    grid, shape_eps_pairs, background_eps=1.0,
                )

        # Stage 1 conformal PEC: when BoundarySpec declares conformal
        # faces and pec_shapes was populated (boundary half-space +
        # any user PEC), compute Dey-Mittra weights and apply
        # eps_correction. Mirrors runners/uniform.py:96-124.
        # ``conformal_weights`` flows through extract_waveguide_*
        # into rfx.simulation.run, which already calls
        # ``apply_conformal_pec`` per step in its scan body.
        # Suppressed when use_kottke_pec — Stage 2 owns the PEC
        # tensor encoding and the eps_correction would double-correct.
        conformal_weights = None
        ref_aniso_eps = None
        if (self._boundary_spec.conformal_faces() and pec_shapes
                and not use_kottke_pec):
            from rfx.geometry.conformal import (
                compute_conformal_weights_sdf,
                clamp_conformal_weights,
                conformal_eps_correction,
            )
            w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, pec_shapes)
            w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, 0.1)
            conformal_weights = (w_ex, w_ey, w_ez)
            # Per-component conformal-corrected eps. Merge with the
            # smoothed eps (if any): conformal overrides at boundary
            # cells, smoothed survives in the interior.
            eps_base = materials.eps_r
            eps_ex_c, eps_ey_c, eps_ez_c = conformal_eps_correction(
                eps_base, w_ex, w_ey, w_ez,
            )
            if aniso_eps is not None:
                s_ex, s_ey, s_ez = aniso_eps
                boundary_ex = w_ex < 1.0
                boundary_ey = w_ey < 1.0
                boundary_ez = w_ez < 1.0
                eps_ex_c = jnp.where(boundary_ex, eps_ex_c, s_ex)
                eps_ey_c = jnp.where(boundary_ey, eps_ey_c, s_ey)
                eps_ez_c = jnp.where(boundary_ez, eps_ez_c, s_ez)
            aniso_eps = (eps_ex_c, eps_ey_c, eps_ez_c)
            # The reference run (vacuum) shares the same boundary
            # walls, so the conformal eps correction applies equally.
            # Build it from the ref vacuum eps so the only difference
            # ref-vs-device is the obstacle in ``materials.eps_r``.
            ref_eps_base = jnp.ones_like(eps_base)
            ref_ex, ref_ey, ref_ez = conformal_eps_correction(
                ref_eps_base, w_ex, w_ey, w_ez,
            )
            ref_aniso_eps = (ref_ex, ref_ey, ref_ez)

        if has_multimode:
            # Multi-mode path: each raw_cfg is a list of WaveguidePortConfig
            port_mode_cfgs: list[list] = []
            for entry, raw in zip(entries, raw_cfgs):
                if isinstance(raw, list):
                    port_mode_cfgs.append(raw)
                else:
                    port_mode_cfgs.append([raw])

            ref_shifts_mm = []
            for entry, mode_cfgs in zip(entries, port_mode_cfgs):
                first_cfg = mode_cfgs[0]
                planes = waveguide_plane_positions(first_cfg)
                desired_ref = (
                    entry.reference_plane
                    if entry.reference_plane is not None
                    else planes["source"]
                )
                ref_shifts_mm.append(desired_ref - planes["reference"])

            mm_pec_axes = "".join(axis for axis in "xyz" if axis not in grid.cpml_axes)
            if normalize == "flux":
                from rfx.core.yee import init_materials as _init_vacuum_materials
                ref_materials = _init_vacuum_materials(grid.shape)
                s_params, mode_map = extract_multimode_s_matrix_flux(
                    grid,
                    materials,
                    ref_materials,
                    port_mode_cfgs,
                    n_steps,
                    boundary="cpml",
                    cpml_axes=grid.cpml_axes,
                    pec_axes=mm_pec_axes,
                    debye=debye,
                    lorentz=lorentz,
                    ref_shifts=ref_shifts_mm,
                    aniso_eps=aniso_eps,
                    conformal_weights=conformal_weights,
                    aniso_inv_eps=aniso_inv_eps,
                )
            elif normalize:
                # The two-run normalized extractor divides each receiving
                # channel by its own empty-guide outgoing wave
                # (b_dev/b_ref). For cross-mode channels the empty-guide
                # reference is ~0 (orthogonal modes do not couple in a
                # uniform guide), so the ratio blows up (measured
                # cross-mode |S| ~ 4.7 on an over-moded WR-90 slab).
                # Use normalize="flux" instead — power ratios referenced
                # to the always-nonzero incident modal power avoid the
                # 0/0 and also fix the reflection noise floor.
                raise ValueError(
                    "compute_waveguide_s_matrix(normalize=True) is not "
                    "supported with n_modes > 1 (cross-mode channels hit a "
                    "0/0 in the two-run normalization). Use "
                    "normalize='flux' for multi-mode S-matrices."
                )
            else:
                s_params, mode_map = extract_multimode_s_matrix(
                    grid,
                    materials,
                    port_mode_cfgs,
                    n_steps,
                    boundary="cpml",
                    cpml_axes=grid.cpml_axes,
                    pec_axes=mm_pec_axes,
                    debye=debye,
                    lorentz=lorentz,
                    ref_shifts=ref_shifts_mm,
                    aniso_eps=aniso_eps,
                    conformal_weights=conformal_weights,
                    aniso_inv_eps=aniso_inv_eps,
                )
            reference_planes = np.array(ref_shifts_mm, dtype=float)
            # Build port names including mode indices
            port_names_mm = []
            port_directions_mm = []
            for port_idx, mode_idx, mtype, m_n in mode_map:
                entry = entries[port_idx]
                port_names_mm.append(f"{entry.name}_mode{mode_idx}_{mtype}{m_n[0]}{m_n[1]}")
                port_directions_mm.append(entry.direction)
            _res_mm = WaveguideSMatrixResult(
                s_params=s_params,
                freqs=jnp.asarray(freqs),
                port_names=tuple(port_names_mm),
                port_directions=tuple(port_directions_mm),
                reference_planes=reference_planes,
            )
            return _finalize_sparam_result(
                _res_mm,
                extractor="compute_waveguide_s_matrix",
                strict=strict_passivity,
                passivity_tol=2.0 if normalize is False else 0.10,
            )

        # Single-mode path (original behavior)
        cfgs = raw_cfgs

        def _slices_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
            return max(a[0], b[0]) < min(a[1], b[1])

        by_direction = {}
        for entry, cfg in zip(entries, cfgs):
            by_direction.setdefault(entry.direction, []).append(cfg)

        for direction, side_cfgs in by_direction.items():
            plane_indices = {cfg.x_index for cfg in side_cfgs}
            if len(plane_indices) != 1:
                raise ValueError(
                    f"waveguide ports on boundary {direction} must share one boundary plane"
                )
            for i in range(len(side_cfgs)):
                for j in range(i + 1, len(side_cfgs)):
                    if _slices_overlap((side_cfgs[i].u_lo, side_cfgs[i].u_hi), (side_cfgs[j].u_lo, side_cfgs[j].u_hi)) and _slices_overlap((side_cfgs[i].v_lo, side_cfgs[i].v_hi), (side_cfgs[j].v_lo, side_cfgs[j].v_hi)):
                        raise ValueError(
                            f"waveguide ports on the same {direction} boundary must have disjoint apertures"
                        )

        ref_shifts = []
        for entry, cfg in zip(entries, cfgs):
            # Default reference plane = the user-facing port plane
            # (snapped x_position). Previously defaulted to the internal
            # ``reference_x_m`` (= source + ref_offset·dx) which left the
            # returned S-matrix phase-shifted by `exp(-jβ·ref_offset·dx)`
            # relative to the physical port — a silent convention mismatch
            # vs. Meep, OpenEMS, and any analytic formula the user would
            # compare against. Keep the ``entry.reference_plane`` override
            # for explicit user control.
            planes = waveguide_plane_positions(cfg)
            desired_ref = (
                entry.reference_plane
                if entry.reference_plane is not None
                else planes["source"]
            )
            ref_shifts.append(desired_ref - planes["reference"])

        _pec_axes = "".join(axis for axis in "xyz" if axis not in grid.cpml_axes)
        # G-WI5 guardrail: conformal=True + normalize=True/"flux" is not supported.
        # Root cause (diagnosed 2026-05-24): the two-run normalisation extractor
        # runs the empty-guide reference WITHOUT conformal_weights while the device
        # run uses them (Dey-Mittra eps_correction).  At fine dx (<=2 mm for WR-90)
        # the asymmetric boundary treatment causes the reference outgoing-wave
        # amplitude to diverge / go unstable, producing NaN |S21|.  Use
        # normalize=False (single-run V/I, no reference run) until the reference
        # run is updated to carry conformal_weights.  Known issue
        # (cv11 mesh-conv NaN xfail).
        if conformal_weights is not None and normalize:
            import warnings as _w
            _w.warn(
                "compute_waveguide_s_matrix(conformal=True, normalize=True) is not "
                "supported: the two-run normalisation reference pass omits "
                "conformal_weights (Dey-Mittra eps_correction), causing the "
                "empty-guide reference outgoing wave to diverge at fine mesh "
                "spacings (dx <= 2 mm for WR-90) and produce NaN |S21|.  "
                "Use normalize=False for conformal=True geometries.  "
                "Tracked at https://github.com/bk-squared/rfx/issues.",
                UserWarning,
                stacklevel=3,
            )

        if normalize == "flux":
            from rfx.core.yee import init_materials as _init_vacuum_materials
            ref_materials = _init_vacuum_materials(grid.shape)
            s_params = extract_waveguide_s_matrix_flux(
                grid,
                materials,
                ref_materials,
                cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes=_pec_axes,
                debye=debye,
                lorentz=lorentz,
                ref_debye=None,
                ref_lorentz=None,
                ref_shifts=ref_shifts,
                aniso_eps=aniso_eps,
                ref_aniso_eps=ref_aniso_eps,
                conformal_weights=conformal_weights,
                aniso_inv_eps=aniso_inv_eps,
                ref_aniso_inv_eps=ref_aniso_inv_eps,
                checkpoint_segments=checkpoint_segments,
            )
        elif normalize:
            from rfx.core.yee import init_materials as _init_vacuum_materials
            ref_materials = _init_vacuum_materials(grid.shape)
            s_params = extract_waveguide_s_params_normalized(
                grid,
                materials,
                ref_materials,
                cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes=_pec_axes,
                debye=debye,
                lorentz=lorentz,
                ref_debye=None,
                ref_lorentz=None,
                ref_shifts=ref_shifts,
                aniso_eps=aniso_eps,
                ref_aniso_eps=ref_aniso_eps,
                conformal_weights=conformal_weights,
                aniso_inv_eps=aniso_inv_eps,
                ref_aniso_inv_eps=ref_aniso_inv_eps,
                checkpoint_segments=checkpoint_segments,
            )
        else:
            s_params = extract_waveguide_s_matrix(
                grid,
                materials,
                cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes=_pec_axes,
                debye=debye,
                lorentz=lorentz,
                ref_shifts=ref_shifts,
                aniso_eps=aniso_eps,
                conformal_weights=conformal_weights,
                aniso_inv_eps=aniso_inv_eps,
                checkpoint_segments=checkpoint_segments,
            )
        reference_planes = np.array(
            [
                entry.reference_plane
                if entry.reference_plane is not None
                else waveguide_plane_positions(cfg)["reference"]
                for entry, cfg in zip(entries, cfgs)
            ],
            dtype=float,
        )
        _res_sm = WaveguideSMatrixResult(
            s_params=s_params,
            freqs=jnp.asarray(freqs),
            port_names=tuple(entry.name for entry in entries),
            port_directions=tuple(entry.direction for entry in entries),
            reference_planes=reference_planes,
        )
        return _finalize_sparam_result(
            _res_sm,
            extractor="compute_waveguide_s_matrix",
            strict=strict_passivity,
            passivity_tol=2.0 if normalize is False else 0.10,
        )

    def compute_msl_s_matrix(
        self,
        *,
        n_steps: int | None = None,
        num_periods: float = 40.0,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 100,
        raw_3probe_dump_path: str | None = None,
        strict_extractor: bool = False,
        eps_override: "jnp.ndarray | None" = None,
        checkpoint_every: int | None = None,
        checkpoint_segments: int | None = None,
    ) -> "MSLSMatrixResult":
        """Compute the MSL S-matrix using N-probe numerical de-embedding.

        For each registered MSL port, runs one FDTD simulation with that
        port driven and the others passive (matched termination). At
        each port ``n_probes`` downstream DFT plane probes record Ez and
        the first probe also records Hy; β, Z0 and the wave amplitudes
        are extracted post-scan via the N-probe least-squares
        wave-decomposition extractor (issue #80 Fix C — SVD lstsq fit of
        ``V_n = α e^{-jβx_n} + γ e^{+jβx_n}`` anchored on the analytic
        Hammerstad-Jensen β guess) and assembled into the full S-matrix.
        The N-probe extractor removes the 3-probe quadratic's q→1
        singularity that produced wrong S11 resonances on thin-substrate
        patches.

        Parameters
        ----------
        n_steps : int or None
            Timesteps per FDTD run. ``None`` → auto from ``num_periods``.
        num_periods : float
            Source-period multiples used to derive ``n_steps`` when not
            provided. Default 40 (MSL transients are slow to drain).
        freqs : array, optional
            Frequency grid. Defaults to
            ``linspace(freq_max / 10, freq_max, n_freqs)``.
        n_freqs : int
            Number of frequencies if ``freqs`` is None.
        raw_3probe_dump_path : str or None
            Optional ``.npz`` path. When provided, write the real
            simulation-derived N-probe voltage/current phasors used by the
            extractor, together with the production S-matrix, so the
            de-embedding can be independently replayed without rerunning
            FDTD. The dump schema is ``rfx.msl_nprobe_dump`` v2 (issue #80
            Fix C); ``raw_v`` has shape ``(n_driven, n_ports, n_probes_max,
            n_freqs)``.
        strict_extractor : bool
            Honesty guard for the de-embedding (issue #80 Fix A). After
            extraction, the per-frequency ``|q|`` and extracted ``Z0``
            are validated against physical bounds (``|q| <= 1`` for a passive
            line; extracted ``Z0`` within 10 % of the analytic
            Hammerstad-Jensen value). When ``False`` (default) a violation
            raises a loud :func:`warnings.warn`; when ``True`` it raises
            :class:`ValueError` instead. With the N-probe extractor (Fix C)
            ``|q|`` and ``Z0`` should be healthy so this rarely fires — it
            is the safety net for pathological geometries.
        checkpoint_segments : int or None
            Gradient-checkpointing segment count for the reverse-mode AD tape on
            the **uniform** mesh (the standard MSL path), forwarded to
            :meth:`forward` (only active on the differentiable ``eps_override``
            channel). Must DIVIDE the auto-computed ``n_steps`` exactly — padding
            is rejected because it would shift the DFT accumulator windows.
            Choose the divisor nearest ``sqrt(n_steps)`` so backward memory scales
            ~``sqrt(n_steps)*carry`` instead of ``n_steps*carry`` — required for
            converged ``num_periods>=20`` AD that otherwise OOMs (G-AD-CHECKPOINT).
            Default ``None`` leaves forward-only runs and small-period AD unchanged.
        checkpoint_every : int or None
            Non-uniform-mesh counterpart of ``checkpoint_segments`` (chunk size,
            not segment count; issue #73). Forwarded to :meth:`forward`; raises
            ``NotImplementedError`` on the uniform path — use
            ``checkpoint_segments`` there.

        Returns
        -------
        MSLSMatrixResult
        """
        from rfx.probes.msl_wave_decomp import extract_msl_nprobe
        from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
        from rfx.sources.msl_port import (
            MSLPort,
            _msl_yz_cells,
            msl_loop_current,
            msl_probe_x_coords_n,
        )

        if not self._msl_ports:
            raise ValueError("No MSL ports registered. Call add_msl_port() first.")
        if self._ports or self._waveguide_ports or self._floquet_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() is defined only for add_msl_port(...) "
                "families in the current simulation. Use separate "
                "simulations for add_port(...), add_waveguide_port(...), "
                "or add_floquet_port(...) S-parameter workflows."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported together with TFSF; "
                "TFSF is a plane-wave source, not an MSL port."
            )
        if self._coaxial_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() does not include add_coaxial_port(...); "
                "coaxial-port S-parameters need a separate validated V/I "
                "extraction and calibration contract."
            )
        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if is_nonuniform and any(
            getattr(pe, "mode", "laplace") == "eigenmode"
            for pe in self._msl_ports
        ):
            raise NotImplementedError(
                "compute_msl_s_matrix() on a non-uniform mesh supports "
                "mode='laplace'/'uniform' (Ez static-Laplace feed) only; the "
                "eigenmode J+M launch needs the magnetic-source channel that "
                "the non-uniform runner does not carry. Use mode='laplace' "
                "(the add_msl_port default) on the graded-mesh lane."
            )
        if self._refinement is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with SBP-SAT "
                "subgridding."
            )
        if self._solver == "adi":
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with solver='adi'; "
                "use the uniform Yee solver."
            )

        entries = list(self._msl_ports)
        n_ports = len(entries)

        # Build the grid used for probe placement + the eps anchor. On the
        # non-uniform lane this MUST be the SAME grid run_nonuniform_path
        # builds (so probe_xs, port cells, dy/dz arrays and the eps anchor
        # align with the run's field planes). build_nonuniform_grid needs a
        # concrete dz_profile — synthesise from dx when absent into a LOCAL.
        # self._dz_profile is mutated only INSIDE the run try/finally below (so
        # the restore always runs even if probe placement / the trace-PEC scan
        # raises) — the subsequent self.run() then reads the same dz and builds
        # a byte-matching grid.
        _dz_profile_saved = self._dz_profile
        _dz_for_grid = self._dz_profile
        if is_nonuniform:
            from rfx.runners.nonuniform import build_nonuniform_grid
            if _dz_for_grid is None:
                _nz_syn = int(round(float(self._domain[2]) / float(self._dx)))
                _dz_for_grid = np.full(max(_nz_syn, 1), float(self._dx))
            grid = build_nonuniform_grid(
                self._freq_max, self._domain, self._dx, self._cpml_layers,
                _dz_for_grid,
                dx_profile=self._dx_profile, dy_profile=self._dy_profile,
                pec_faces=self._boundary_spec.pec_faces()
                    if self._boundary_spec is not None else None,
                pmc_faces=self._boundary_spec.pmc_faces()
                    if self._boundary_spec is not None else None,
                cpml_axes="".join(
                    ax for ax in "xyz"
                    if ax not in (self._periodic_axes or "")
                ),
            )
        else:
            grid = self._build_grid()

        if freqs is None:
            freqs_arr = np.asarray(jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs))
        else:
            freqs_arr = np.asarray(freqs)
        n_freqs_used = int(freqs_arr.shape[0])

        # Build MSLPort descriptors and probe x-coords once (geometry shared).
        msl_ports: list[MSLPort] = []
        for pe in entries:
            x_feed, y_centre, z_lo = pe.position
            msl_ports.append(MSLPort(
                feed_x=float(x_feed),
                y_lo=float(y_centre - pe.width / 2),
                y_hi=float(y_centre + pe.width / 2),
                z_lo=float(z_lo),
                z_hi=float(z_lo + pe.height),
                direction=pe.direction,
                impedance=pe.impedance,
                excitation=pe.waveform,
            ))

        # N-probe placement (issue #80 Fix C). Probe n sits at
        # offset + n*spacing cells from the feed plane. N >= 3.
        n_probes_per_port = [int(pe.n_probes) for pe in entries]
        probe_xs = [
            msl_probe_x_coords_n(
                grid, mp,
                n_probes=n_probes,
                n_offset_cells=pe.n_probe_offset,
                n_spacing_cells=pe.n_probe_spacing,
            )
            for mp, pe, n_probes in zip(msl_ports, entries, n_probes_per_port)
        ]
        # ``probe_xs`` are the N physical x-coordinates fed to the
        # N-probe extractor (issue #80 Fix C), which fits
        # V_n = alpha*exp(-j*beta*x_n) + gamma*exp(+j*beta*x_n). The
        # coordinates are increasing for ``+x`` ports and decreasing for
        # ``-x`` ports; the extractor anchors the model at probe 0 and
        # only uses coordinate differences, so feeding raw physical x
        # keeps alpha = the +x-travelling wave for BOTH port directions
        # — matching the legacy 3-probe sign convention the S11 sign
        # was validated against.

        # Per-axis cell-size arrays for V/I integration. Both uniform and
        # non-uniform grids are supported — NonUniformGrid exposes per-cell
        # dx_arr/dy_arr/dz (NOT *_profile); see _msl_cell_profile.
        dy_arr = _msl_cell_profile(grid, "y", grid.ny)
        dz_arr = _msl_cell_profile(grid, "z", grid.nz)

        # Fixed cross-section indices per port (same across all runs).
        port_idx_meta = []
        for mp in msl_ports:
            cells = _msl_yz_cells(grid, mp)
            j_set = sorted({c[1] for c in cells})
            k_set = sorted({c[2] for c in cells})
            j_lo, j_hi = j_set[0], j_set[-1]
            k_lo, k_hi = k_set[0], k_set[-1]
            j_centre = (j_lo + j_hi) // 2
            k_top = k_hi  # trace sits at the top of the substrate
            port_idx_meta.append(dict(
                j_lo=j_lo, j_hi=j_hi,
                k_lo=k_lo, k_hi=k_hi,
                j_centre=j_centre, k_top=k_top,
                height=mp.z_hi - mp.z_lo,
            ))

        # Analytic Hammerstad-Jensen anchor per port (issue #80 Fix C).
        # ``beta0_per_port[p]`` is the (n_freqs,) propagation-constant
        # guess ``omega * sqrt(eps_eff) / c`` used to centre the N-probe
        # extractor's robust beta scan; ``z0_hj_per_port[p]`` is the
        # analytic Z0 used by the honesty guard.
        #
        # Substrate permittivity precedence (mirrors rfx/runners/uniform.py
        # so the beta anchor and the source see the SAME eps_r): explicit
        # add_msl_port(eps_r_sub=...) > the rasterised FDTD eps_r at the
        # trace-centre substrate cell. Reading the material array makes
        # this robust even when the user did not pass eps_r_sub — a plain
        # pe.eps_r_sub-or-1.0 fallback would anchor the scan on vacuum and
        # land beta outside the scan window for a loaded substrate.
        from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
        _C0_MSL = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
        _msl_assembled = (
            self._assemble_materials_nu(grid) if is_nonuniform
            else self._assemble_materials(grid)
        )
        _msl_materials = _msl_assembled[0]
        _msl_pec_mask = (
            None if _msl_assembled[3] is None
            else np.asarray(_msl_assembled[3])
        )
        beta0_per_port: list[np.ndarray] = []
        z0_hj_per_port: list[float] = []
        for p_idx, pe in enumerate(entries):
            meta = port_idx_meta[p_idx]
            if pe.eps_r_sub is not None:
                eps_r_ref = float(pe.eps_r_sub)
            else:
                k_mid = (meta["k_lo"] + meta["k_hi"]) // 2
                i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
                eps_r_ref = float(
                    np.asarray(
                        _msl_materials.eps_r[i_feed_p, meta["j_centre"], k_mid]
                    )
                )
            z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(
                pe.width, pe.height, eps_r_ref
            )
            beta0_per_port.append(
                2.0 * np.pi * freqs_arr * float(np.sqrt(eps_eff_hj)) / _C0_MSL
            )
            z0_hj_per_port.append(float(z0_hj))

        # Trace-conductor z-cell span per port (issue #80 stage S1). The
        # closed Ampere-loop current needs the PEC trace cells; the trace
        # is the PEC run at/above the substrate top in the port's centre
        # column (the ground-plane PEC sits far below near k_lo).
        trace_k_per_port: list[tuple[int, int]] = []
        for p_idx in range(n_ports):
            meta = port_idx_meta[p_idx]
            i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
            col = (
                None if _msl_pec_mask is None
                else _msl_pec_mask[i_feed_p, meta["j_centre"], meta["k_top"]:]
            )
            k_pec = np.array([], dtype=int) if col is None else np.where(col)[0]
            if k_pec.size == 0:
                raise RuntimeError(
                    "compute_msl_s_matrix: no PEC trace conductor found "
                    "above the substrate top for MSL port "
                    f"{entries[p_idx].name!r}; the closed Ampere-loop "
                    "current (issue #80 stage S1) needs the trace PEC. "
                    "Add the microstrip trace as a Box(material='pec')."
                )
            trace_k_per_port.append((
                int(meta["k_top"] + int(k_pec.min())),
                int(meta["k_top"] + int(k_pec.max())),
            ))

        # Stash existing add_dft_plane_probe registrations and restore on exit.
        saved_dft = list(self._dft_planes)
        saved_msl = list(self._msl_ports)
        saved_ports = list(self._ports)
        try:
            # Mutate self._dz_profile to the (possibly synthesised) grid dz only
            # now — inside the try — so the finally always restores it and the
            # subsequent self.run() builds a grid matching the one above.
            if is_nonuniform:
                self._dz_profile = _dz_for_grid
            _complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
            S = jnp.zeros((n_ports, n_ports, n_freqs_used), dtype=_complex_dtype)
            Z0_per_run = jnp.zeros((n_ports, n_freqs_used), dtype=_complex_dtype)
            beta_first = jnp.zeros(n_freqs_used, dtype=_complex_dtype)
            # N-probe extractor (issue #80 Fix C): store all N voltage
            # probe phasors. n_probes may differ per port — store the
            # max width and zero-pad shorter ports.
            n_probes_max = max(n_probes_per_port)
            raw_v = jnp.zeros(
                (n_ports, n_ports, n_probes_max, n_freqs_used), dtype=_complex_dtype
            )
            raw_i1 = jnp.zeros((n_ports, n_ports, n_freqs_used), dtype=_complex_dtype)
            raw_z0 = jnp.zeros((n_ports, n_ports, n_freqs_used), dtype=_complex_dtype)
            raw_q = jnp.zeros((n_ports, n_ports, n_freqs_used), dtype=_complex_dtype)

            for driven in range(n_ports):
                # Re-instantiate a clean simulation by mutating in place:
                # use add_msl_port as the registration path, but here we
                # need finer control over excite=True/False per-run, so
                # rebuild ``self._msl_ports`` for this run.
                run_entries = []
                for idx, pe in enumerate(entries):
                    new_excite = (idx == driven) and pe.excite
                    if new_excite:
                        wf = pe.waveform if pe.waveform is not None else \
                            GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
                    else:
                        wf = None
                    run_entries.append(_MSLPortEntry(
                        name=pe.name, position=pe.position,
                        width=pe.width, height=pe.height,
                        direction=pe.direction, impedance=pe.impedance,
                        waveform=wf, excite=new_excite,
                        n_probe_offset=pe.n_probe_offset,
                        n_probe_spacing=pe.n_probe_spacing,
                        mode=pe.mode,
                        eps_r_sub=pe.eps_r_sub,
                    ))
                self._msl_ports = run_entries

                # Register DFT plane probes for V (Ez) and I (Hy).
                self._dft_planes = list(saved_dft)
                ez_probe_names: list[list[str]] = [[] for _ in range(n_ports)]
                hy_probe_names: list[str] = [None] * n_ports  # type: ignore
                hz_probe_names: list[str] = [None] * n_ports  # type: ignore
                for p_idx, (mp, pxs) in enumerate(zip(msl_ports, probe_xs)):
                    for q_idx, x_coord in enumerate(pxs):
                        nm = f"_msl_run{driven}_p{p_idx}_ez{q_idx}"
                        self.add_dft_plane_probe(
                            axis="x", coordinate=float(x_coord),
                            component="ez", freqs=jnp.asarray(freqs_arr),
                            name=nm,
                        )
                        ez_probe_names[p_idx].append(nm)
                    nm_hy = f"_msl_run{driven}_p{p_idx}_hy"
                    self.add_dft_plane_probe(
                        axis="x", coordinate=float(pxs[0]),
                        component="hy", freqs=jnp.asarray(freqs_arr),
                        name=nm_hy,
                    )
                    hy_probe_names[p_idx] = nm_hy
                    # Hz plane probe at probe 0 — the side legs of the
                    # closed Ampere-loop current (issue #80 stage S1).
                    nm_hz = f"_msl_run{driven}_p{p_idx}_hz"
                    self.add_dft_plane_probe(
                        axis="x", coordinate=float(pxs[0]),
                        component="hz", freqs=jnp.asarray(freqs_arr),
                        name=nm_hz,
                    )
                    hz_probe_names[p_idx] = nm_hz

                # G-AD-WIRE: when eps_override is provided use the
                # differentiable forward() path so jax.grad can flow
                # from eps_override through the DFT plane accumulators
                # into the V/I assembly. Otherwise fall back to run()
                # for imperative (non-AD) workflows.
                if eps_override is not None:
                    fwd_result = self.forward(
                        eps_override=eps_override,
                        n_steps=n_steps,
                        num_periods=num_periods,
                        checkpoint_every=checkpoint_every,
                        checkpoint_segments=checkpoint_segments,
                    )
                    planes = fwd_result.dft_planes or {}
                else:
                    result = self.run(
                        n_steps=n_steps,
                        num_periods=num_periods,
                        compute_s_params=False,
                    )
                    planes = result.dft_planes or {}

                # Helper: integrate V and I per port from the recorded planes.
                v_per_port: list[list[np.ndarray]] = []
                i_first_per_port: list[np.ndarray] = []
                for p_idx, meta in enumerate(port_idx_meta):
                    vs = []
                    for nm in ez_probe_names[p_idx]:
                        ez_plane = jnp.asarray(planes[nm].accumulator)
                        # ez_plane shape: (n_freqs, ny, nz)
                        v_f = jnp.zeros(n_freqs_used, dtype=_complex_dtype)
                        for k in range(meta["k_lo"], meta["k_hi"] + 1):
                            v_f = v_f + ez_plane[:, meta["j_centre"], k] * float(dz_arr[k])
                        vs.append(v_f)
                    v_per_port.append(vs)
                    # G-AD-WIRE: keep on JAX tape when eps_override is
                    # set. np.asarray() would concretise a JAX tracer and
                    # break jax.grad. jnp.asarray() is a no-op on a real
                    # jnp.ndarray and still works for numpy arrays.
                    hy_plane = jnp.asarray(planes[hy_probe_names[p_idx]].accumulator)
                    hz_plane = jnp.asarray(planes[hz_probe_names[p_idx]].accumulator)
                    # Leapfrog E/H half-step time correction. add_dft_plane_probe
                    # timestamps EVERY component at t = step·dt
                    # (rfx/probes/probes.py:457), but H lives half a step behind E
                    # (H at t − dt/2), so the recorded Hy/Hz DFT is missing the
                    # exp(+jω·dt/2) factor the flux monitor already applies
                    # (rfx/simulation.py:1380-1382: phase_h = phase_e·exp(+jω·dt/2)).
                    # Ez (→ V) is correctly at t and needs no correction. Without
                    # this, I = ∮H·dl carries a spurious exp(−jω·dt/2) so the V·I
                    # de-embedding sees Zin = V/I rotated by exp(+jω·dt/2) — a
                    # frequency-dependent phase that can push Re(Zin) < 0 → a
                    # non-physical |S11| > 1 near the passive boundary (the same
                    # half-step class as the 2026-04-28 s11_from_dumps artefact).
                    _hs_phase = jnp.exp(
                        1j * 2.0 * jnp.pi * jnp.asarray(freqs_arr)
                        * (float(grid.dt) * 0.5)
                    ).astype(hy_plane.dtype)
                    hy_plane = hy_plane * _hs_phase[:, None, None]
                    hz_plane = hz_plane * _hs_phase[:, None, None]
                    # Closed Ampere-loop current ∮H·dl around the trace
                    # conductor (issue #80 stage S1). The pre-S1 inline
                    # integral summed the bottom Hy leg only and undercounted
                    # I by ~1.5x, inflating the de-embedded Z0 to ~74 vs the
                    # ~48 ohm analytic value. msl_loop_current closes the
                    # contour (bottom/top Hy legs + left/right Hz legs) and
                    # carries the +x current-sign convention.
                    k_tr_lo, k_tr_hi = trace_k_per_port[p_idx]
                    i_f = msl_loop_current(
                        hy_plane, hz_plane,
                        j_lo=meta["j_lo"], j_hi=meta["j_hi"],
                        k_trace_lo=k_tr_lo, k_trace_hi=k_tr_hi,
                        dy_arr=dy_arr, dz_arr=dz_arr,
                        direction=msl_ports[p_idx].direction,
                    )
                    i_first_per_port.append(i_f)
                    # N-probe least-squares wave decomposition (issue #80
                    # Fix C). Stack the N voltage probes into (n_freqs, N),
                    # anchor the beta scan on the analytic HJ guess, and
                    # solve the over-determined (alpha, gamma) system by
                    # SVD lstsq — this removes the 3-probe q->1 singularity.
                    n_probes_p = n_probes_per_port[p_idx]
                    v_stack = jnp.stack(v_per_port[p_idx], axis=-1)  # (n_freqs, N)
                    raw_v = raw_v.at[driven, p_idx, :n_probes_p, :].set(jnp.asarray(v_stack.T, dtype=_complex_dtype))
                    raw_i1 = raw_i1.at[driven, p_idx, :].set(jnp.asarray(i_f, dtype=_complex_dtype))
                    res_p = extract_msl_nprobe(
                        jnp.asarray(v_stack),
                        jnp.asarray(np.asarray(probe_xs[p_idx], dtype=float)),
                        jnp.asarray(i_f),
                        jnp.asarray(beta0_per_port[p_idx]),
                        z0_hj=z0_hj_per_port[p_idx],
                    )
                    # Normalize the REPORTED characteristic-impedance sign per port
                    # (issue #140). msl_loop_current negates the loop current ONLY for
                    # "+x" ports (rfx/sources/msl_port.py:947-948), so a "-x" port's
                    # fitted z0 = (alpha - gamma)/I inherits a negative sign while a
                    # physical Z0 is positive-real. Mirror that exact binary so BOTH
                    # ports report a positive Z0. This touches ONLY the reported/dumped
                    # Z0 (raw_z0, Z0_per_run) and the |Z0| honesty-guard; it never
                    # enters S11/S21 (which use the static analytic Hammerstad-Jensen
                    # z0_hj). It removes the spurious ~228% guard deviation on -x ports
                    # while leaving the genuine ~20-27% 3-cell Yee-staircase Z0 warning on
                    # both ports. NB: the raw current dump (raw_i1) intentionally keeps
                    # its un-normalized sign; only the DERIVED Z0 is sign-normalized.
                    dir_sign = 1.0 if msl_ports[p_idx].direction == "+x" else -1.0
                    z0_fit = jnp.asarray(res_p["z0"], dtype=_complex_dtype) * dir_sign
                    raw_z0 = raw_z0.at[driven, p_idx, :].set(z0_fit)
                    raw_q = raw_q.at[driven, p_idx, :].set(jnp.asarray(res_p["q"], dtype=_complex_dtype))
                    if p_idx == driven:
                        # V·I single-plane wave split at probe 0 (issue #80
                        # stage S1): a=(V+Z0*I)/2, b=(V-Z0*I)/2, S11=b/a —
                        # the OpenEMS-style telegrapher de-embedding. With a
                        # real positive Z0 and a passive structure this is
                        # bounded |S11|<=1, unlike the Fix-C alpha/gamma
                        # spatial fit that blew up to |S11|>1 on a strong
                        # reflector. Z0 is analytic Hammerstad-Jensen; I is
                        # the closed Ampere loop.
                        v0_d = v_per_port[driven][0]
                        z0hj_d = z0_hj_per_port[driven]
                        a_fwd_d = 0.5 * (v0_d + z0hj_d * i_f)
                        b_ref_d = 0.5 * (v0_d - z0hj_d * i_f)
                        S = S.at[driven, driven, :].set(jnp.asarray(b_ref_d / (a_fwd_d + 1e-30), dtype=_complex_dtype))
                        Z0_per_run = Z0_per_run.at[driven, :].set(z0_fit)
                        alpha_d = a_fwd_d
                        if driven == 0:
                            beta_first = jnp.asarray(res_p["beta"], dtype=_complex_dtype)

                # Off-diagonal S21: S[j,i] = b_j / a_i (issue #80 stage S1).
                # The wave received from the structure at a passive port is
                # its BACKWARD wave b=(V-Z0*I)/2, not the forward wave a it
                # would launch. For a transmitted wave arriving at a port
                # whose forward reference faces the other way, a~0 and b~V;
                # using a gave the non-physical |S21|~0.08, b gives ~1.
                for j in range(n_ports):
                    if j == driven:
                        continue
                    v0_p = v_per_port[j][0]
                    b_out_p = 0.5 * (
                        v0_p - z0_hj_per_port[j] * i_first_per_port[j]
                    )
                    S = S.at[j, driven, :].set(jnp.asarray(b_out_p, dtype=_complex_dtype) / (jnp.asarray(alpha_d, dtype=_complex_dtype) + 1e-30))

            # --- Honesty guard (issue #80 Fix A, retargeted in stage S1) ---
            # S1 moved S11/S21 onto the OpenEMS-style V·I wave split, which
            # is bounded |S11| <= 1 for a passive structure whenever the
            # closed Ampere-loop current is sound. A V·I-split |S11| > 1 is
            # therefore the primary red flag — it means the current was
            # mismeasured (sign/scale), so S11/S21 are untrustworthy; this
            # is what raises under strict_extractor=True. The reported Z0
            # and beta still ride on the retained N-probe fit, which can be
            # noisy per-frequency on coarse meshes, so a Z0 deviation from
            # analytic Hammerstad-Jensen is reported as a SEPARATE, softer
            # caveat — it does not impugn the V·I-split S11/S21.
            import warnings as _w

            _S11_MAX = 1.0 + 0.05
            _Z0_TOL = 0.10
            for driven in range(n_ports):
                pe = entries[driven]
                z0_hj = z0_hj_per_port[driven]
                s11_abs = np.abs(np.asarray(jax.lax.stop_gradient(S[driven, driven, :])))
                k_s = int(np.argmax(s11_abs))
                s11_max = float(s11_abs[k_s])
                z0_dev = np.abs(np.asarray(jax.lax.stop_gradient(raw_z0[driven, driven, :])) - z0_hj) / z0_hj
                k_z = int(np.argmax(z0_dev))
                z0_dev_max = float(z0_dev[k_z])
                # Primary — V·I-split S11 boundedness (extraction soundness).
                if s11_max > _S11_MAX:
                    msg = (
                        f"compute_msl_s_matrix: V·I-split |S11| = "
                        f"{s11_max:.3f} > 1 for MSL port {pe.name!r} at "
                        f"f = {freqs_arr[k_s] / 1e9:.4f} GHz — non-physical "
                        "for a passive structure. The closed Ampere-loop "
                        "current is likely mismeasured (sign/scale); the "
                        "extracted S11/S21 are UNRELIABLE."
                    )
                    if strict_extractor:
                        raise ValueError(msg)
                    _w.warn(msg, stacklevel=2)
                # Secondary — reported-Z0 sanity (retained N-probe fit).
                if z0_dev_max > _Z0_TOL:
                    _w.warn(
                        f"compute_msl_s_matrix: reported Z0 for MSL port "
                        f"{pe.name!r} = "
                        f"{float(np.asarray(jax.lax.stop_gradient(raw_z0[driven, driven, k_z])).real):.2f} ohm deviates "
                        f"{z0_dev_max * 100:.1f}% from analytic Hammerstad-"
                        f"Jensen {z0_hj:.2f} ohm at "
                        f"f = {freqs_arr[k_z] / 1e9:.4f} GHz. Z0 rides on the "
                        "retained N-probe fit (S1 transitional); on coarse "
                        "meshes this includes Yee-staircase bias. The V·I-"
                        "split S11/S21 are unaffected.",
                        stacklevel=2,
                    )

            if raw_3probe_dump_path is not None:
                import json
                from pathlib import Path

                path = Path(raw_3probe_dump_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "schema": "rfx.msl_nprobe_dump",
                    "schema_version": 2,
                    "production_smatrix_schema": "S[receiver_port, driven_port, frequency_index]",
                    "raw_v_shape": "(n_driven, n_ports, n_probes_max, n_freqs)",
                    "raw_i1_shape": "(n_driven, n_ports, n_freqs)",
                    "n_probes_per_port": [int(n) for n in n_probes_per_port],
                    "phase_convention": "DFT accumulator convention from add_dft_plane_probe",
                    "current_convention": (
                        "line current sign normalized so +x and -x MSL ports "
                        "produce positive characteristic impedance on the "
                        "validated thru-line envelope"
                    ),
                    "deembedding": (
                        "N equally spaced voltage probes plus current at "
                        "probe 0; the N-probe least-squares wave-decomposition "
                        "extractor (issue #80 Fix C) fits V_n = alpha*exp(-j "
                        "beta x_n) + gamma*exp(+j beta x_n) by SVD lstsq, "
                        "recovering q, alpha, gamma, S11, Sij from raw phasors"
                    ),
                    "grid": {
                        "dx_m": float(grid.dx),
                        "dt_s": float(grid.dt),
                        "nx": int(grid.nx),
                        "ny": int(grid.ny),
                        "nz": int(grid.nz),
                    },
                    "simulation": {
                        "freq_max_hz": float(self._freq_max),
                        "num_periods": float(num_periods),
                        "n_steps": None if n_steps is None else int(n_steps),
                    },
                    "port_definitions": [
                        {
                            "name": str(pe.name),
                            "position_m": [float(x) for x in pe.position],
                            "width_m": float(pe.width),
                            "height_m": float(pe.height),
                            "direction": pe.direction,
                            "impedance_ohm": float(pe.impedance),
                            "n_probe_offset": int(pe.n_probe_offset),
                            "n_probe_spacing": int(pe.n_probe_spacing),
                            "n_probes": int(pe.n_probes),
                            "mode": pe.mode,
                        }
                        for pe in entries
                    ],
                }
                np.savez(
                    path,
                    metadata_json=np.asarray(json.dumps(metadata)),
                    freqs_hz=np.asarray(freqs_arr, dtype=np.float64),
                    raw_v=raw_v,
                    raw_i1=raw_i1,
                    raw_z0=raw_z0,
                    raw_q=raw_q,
                    production_smatrix=S,
                    production_z0=Z0_per_run,
                    production_beta=beta_first,
                    port_names=np.asarray(tuple(pe.name for pe in entries), dtype=object),
                    driven_port_indices=np.arange(n_ports, dtype=np.int64),
                )

            return MSLSMatrixResult(
                S=S,
                freqs=np.asarray(freqs_arr),
                Z0=Z0_per_run,
                beta=beta_first,
                port_names=tuple(pe.name for pe in entries),
            )
        finally:
            self._dft_planes = saved_dft
            self._msl_ports = saved_msl
            self._ports = saved_ports
            self._dz_profile = _dz_profile_saved

    def compute_coaxial_s_matrix(
        self,
        *,
        n_steps: int = 320,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 21,
        field_scale: float = 1.0e4,
        magnetic_ratio: float = 1.0,
        signal_floor: float = 1.0e-12,
        reference_plane_axial_index_offset: int = 0,
        strict_passivity: bool = False,
    ) -> "CoaxialSMatrixResult":
        """Experimental coaxial S-matrix via distributed TEM plane sources.

        .. deprecated::
            This single-plane V/I path measures inside a closed PEC box around a
            short coaxial stub, which has no transmission line for a clean
            reflection — it reports non-physical ``|S11|>1`` for a lossless
            short (verified). Use :meth:`compute_coaxial_line_reflection`, which
            builds a real coax line with a matched CPML feed and extracts the
            reflection from a multi-plane matrix-pencil decomposition (validated
            short→Γ=-1, open→|Γ|=1, matched→0 across the band). This method is
            retained only for backward compatibility.

        For each registered ``add_coaxial_port(...)`` port, runs one FDTD
        simulation with that port driven and all other coaxial ports passive.
        A distributed transverse E/M plane source is injected on the port's
        cross-section (the M67 prototype scaffold promoted to the public
        API); DFT plane probes capture the resulting Ex/Ey/Hx/Hy on every
        coaxial port's reference plane; the V/I extractor recovers ``V`` and
        ``I`` via the radial line / azimuthal loop integrals; and the
        standard power-wave decomposition assembles the full S-matrix.

        Status: **experimental**. The plane source can produce a residual
        forward wave and the extracted reference-plane V/I has known
        amplitude bias for coarse grids; ``status="degraded"`` is reported
        when any V/I sample falls below ``signal_floor``. Use this API for
        development; do not promote claims beyond E2/E3 without an external
        cross-solver fixture (see ``port_external_reference_requirements``).

        Parameters
        ----------
        n_steps:
            FDTD timesteps per driven-port run. Default 320.
        freqs:
            Frequency grid (Hz). Defaults to a uniform grid covering
            ``[freq_max / 10, freq_max]``.
        n_freqs:
            Number of frequencies if ``freqs`` is None. Default 21.
        field_scale:
            Linear scale on the radial E waveform. Increase to lift the
            plane signal above DFT noise (V/I extraction is amplitude-linear
            so the S-matrix is invariant under this scale).
        magnetic_ratio:
            Multiplier on the ``H`` waveform after the analytic ``1/Z_TEM``
            factor. ``1.0`` injects the lossless-TEM Poynting-balanced
            amplitude; smaller values bias toward an E-only injection.
        signal_floor:
            Absolute V or I phasor magnitude below which the result is
            flagged as ``"degraded"``.
        reference_plane_axial_index_offset:
            Axial-index offset for the source/probe plane relative to the
            port pin centre.

        Returns
        -------
        CoaxialSMatrixResult
        """

        import warnings
        warnings.warn(
            "compute_coaxial_s_matrix() (single-plane V/I in a closed PEC box) is "
            "deprecated and reports non-physical |S11|>1 for a lossless short; use "
            "compute_coaxial_line_reflection() (validated coax-line method).",
            DeprecationWarning,
            stacklevel=2,
        )

        from rfx.probes.probes import init_dft_plane_probe
        from rfx.simulation import run as _run
        from rfx.sources.coaxial_port import (
            build_coaxial_tem_plane_source_specs,
            extract_coaxial_plane_vi_from_dft,
        )

        if not self._coaxial_ports:
            raise ValueError(
                "No coaxial ports registered. Call add_coaxial_port() first."
            )
        if (
            self._ports
            or self._waveguide_ports
            or self._floquet_ports
            or self._msl_ports
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is defined only for "
                "add_coaxial_port(...) families in the current simulation."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported with TFSF; "
                "TFSF is a plane-wave source, not a coaxial port."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() supports the uniform Yee lane only."
            )

        ports: list[CoaxialPort] = list(self._coaxial_ports)
        n_ports = len(ports)

        # Build the working grid + materials with all coaxial geometries
        # stamped (PEC center pin, PTFE dielectric fill, PEC outer shell from
        # M66). ``_build_materials`` only assembles bulk materials and shapes;
        # ``add_coaxial_port`` only registers the port descriptor, so without
        # this loop the FDTD would run with the source dropped into pure
        # vacuum and the wave would radiate bidirectionally with no coax
        # structure to confine it (this is the real source of the
        # calibration-blocked status documented in the handover).
        from rfx.sources.coaxial_port import (
            setup_coaxial_port,
            add_coaxial_matched_termination,
            add_coaxial_open_termination,
            add_coaxial_pec_end_cap,
        )
        grid = self._build_grid()
        materials, _, _ = self._build_materials(grid)
        for p in ports:
            materials = setup_coaxial_port(grid, p, materials)
        for term_port_idx, term_R, term_offset_cells in self._coaxial_terminations:
            materials = add_coaxial_matched_termination(
                grid,
                ports[term_port_idx],
                materials,
                target_impedance=term_R,
                axial_offset_cells=term_offset_cells,
            )
        for term_port_idx, retract_cells in self._coaxial_open_terminations:
            materials = add_coaxial_open_termination(
                grid,
                ports[term_port_idx],
                materials,
                pin_retract_cells=retract_cells,
            )
        for cap_port_idx, cap_offset_cells in self._coaxial_pec_end_caps:
            materials = add_coaxial_pec_end_cap(
                grid,
                ports[cap_port_idx],
                materials,
                axial_offset_cells=cap_offset_cells,
            )

        # Frequency grid.
        if freqs is None:
            freqs = jnp.linspace(
                self._freq_max / 10.0,
                self._freq_max,
                int(n_freqs),
                dtype=jnp.float32,
            )
        else:
            freqs = jnp.asarray(freqs, dtype=jnp.float32)

        # Reference-plane axial indices per port (cross-section z-plane).
        from rfx.sources.coaxial_port import _coaxial_port_geometry
        plane_indices: list[int] = []
        for p in ports:
            _, _, _, pin_center, _, _ = _coaxial_port_geometry(grid, p)
            plane_indices.append(
                int(grid.position_to_index(pin_center)[2])
                + int(reference_plane_axial_index_offset)
            )

        # Output buffers.
        n_freqs_used = int(freqs.shape[0])
        s = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)
        z_tem_arr = np.zeros((n_ports, n_freqs_used), dtype=np.complex128)
        v_dump = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)
        i_dump = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)

        status = "passed"

        for driven in range(n_ports):
            spec = build_coaxial_tem_plane_source_specs(
                grid=grid,
                port=ports[driven],
                n_steps=int(n_steps),
                field_scale=float(field_scale),
                magnetic_ratio=float(magnetic_ratio),
                reference_plane_axial_index_offset=int(
                    reference_plane_axial_index_offset
                ),
            )
            z_tem_arr[driven, :] = complex(spec.z_tem_ohm)

            # DFT plane probes on every port's cross-section.
            dft_planes = []
            for p_idx, p in enumerate(ports):
                for component in ("ex", "ey", "hx", "hy"):
                    dft_planes.append(
                        init_dft_plane_probe(
                            axis=2,
                            index=plane_indices[p_idx],
                            component=component,
                            freqs=freqs,
                            grid_shape=grid.shape,
                            dft_total_steps=int(n_steps),
                        )
                    )

            result = _run(
                grid,
                materials,
                int(n_steps),
                # Honour the simulation boundary (was hardcoded "pec", the closed
                # box that is the documented root cause); self._boundary is always
                # a str ("pec"/"cpml"/"upml"), even for the BoundarySpec path.
                boundary=self._boundary,
                sources=list(spec.electric_sources),
                mag_sources=list(spec.magnetic_sources),
                dft_planes=dft_planes,
                return_state=False,
            )
            if result.dft_planes is None:
                raise RuntimeError(
                    "compute_coaxial_s_matrix(): runner returned no DFT planes"
                )

            # Slice DFT planes back into per-port (ex, ey, hx, hy) groups.
            per_port: list[dict[str, np.ndarray]] = []
            for p_idx in range(n_ports):
                start = p_idx * 4
                group = result.dft_planes[start : start + 4]
                comp_map = {
                    probe.component: np.asarray(probe.accumulator, dtype=np.complex128)
                    for probe in group
                }
                per_port.append(comp_map)

            # Extract V/I at each port's reference plane.
            voltages = []
            currents = []
            for p_idx, p in enumerate(ports):
                vi = extract_coaxial_plane_vi_from_dft(
                    grid=grid,
                    port=p,
                    plane_axial_index=plane_indices[p_idx],
                    ex_dft=per_port[p_idx]["ex"],
                    ey_dft=per_port[p_idx]["ey"],
                    hx_dft=per_port[p_idx]["hx"],
                    hy_dft=per_port[p_idx]["hy"],
                )
                v = np.asarray(vi.vi.voltage, dtype=np.complex128)
                i = np.asarray(vi.vi.current, dtype=np.complex128)
                voltages.append(v)
                currents.append(i)
                v_dump[driven, p_idx, :] = v
                i_dump[driven, p_idx, :] = i
                if (
                    float(np.max(np.abs(v))) <= float(signal_floor)
                    or float(np.max(np.abs(i))) <= float(signal_floor)
                ):
                    status = "degraded"

            # Power-wave decomposition at each receive port (a_j at driven, b_i
            # at receiver) using the analytic Z_TEM as Z0.
            z0 = complex(spec.z_tem_ohm)
            a_j = (voltages[driven] + z0 * currents[driven]) / (2.0 * np.sqrt(z0))
            for receiver in range(n_ports):
                b_i = (voltages[receiver] - z0 * currents[receiver]) / (
                    2.0 * np.sqrt(z0)
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    s[receiver, driven, :] = np.where(
                        np.abs(a_j) > 0.0,
                        b_i / a_j,
                        np.nan + 1j * np.nan,
                    )

        reference_planes = np.asarray(
            [
                float(grid.position_to_index(p.position)[2] + reference_plane_axial_index_offset)
                * float(grid.dx)
                for p in ports
            ],
            dtype=float,
        )

        _res_coax = CoaxialSMatrixResult(
            s_params=s,
            freqs=np.asarray(freqs, dtype=float),
            port_names=tuple(f"coax_{i}" for i in range(n_ports)),
            port_faces=tuple(p.face for p in ports),
            reference_planes=reference_planes,
            z_tem_ohm=z_tem_arr,
            voltages=v_dump,
            currents=i_dump,
            status=status,
        )
        return _finalize_sparam_result(
            _res_coax,
            extractor="compute_coaxial_s_matrix",
            strict=strict_passivity,
        )

    def compute_coaxial_line_reflection(
        self,
        *,
        termination: str = "short",
        n_steps: int = 6000,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 11,
        field_scale: float = 1.0e4,
        cpml_axes: str = "z",
        dut_offset_cells: int = 4,
        probe_count: int = 12,
        probe_start_cells: int = 8,
        probe_spacing_cells: int = 4,
        feed_impedance: float | None = None,
        dut_impedance: float | None = None,
    ) -> "CoaxialLineReflectionResult":
        """One-port coaxial reflection on a real transmission line (broad-E5).

        Builds a coextensive coax line (face='top', pin along -z) terminated in a
        matched resistive feed near the +z boundary, drives a TEM TFSF source one
        cell below the feed, and reflects off a calibration ``termination`` at the
        -z end: ``"short"`` (Γ=-1), ``"open"`` (Γ=+1), or ``"matched"`` (Γ→0).
        With ``termination="matched"`` and ``dut_impedance=R`` the DUT is instead a
        known resistive load (analytic ``Γ=(R-Z0)/(R+Z0)``) — used by the broad-E5
        envelope to test non-trivial reflection magnitudes against exact truth.
        The reflection is read from the modal voltage ``V(z)=∫E_r dr`` sampled at
        ``probe_count`` equally spaced planes and a matrix-pencil estimate of the
        complex propagation constant (β self-measured, Z0-free).

        Unlike ``compute_coaxial_s_matrix`` (single-plane V/I in a hardcoded
        closed PEC box — non-physical |S11|>1), this uses an absorbing CPML feed
        so a real line exists. **Resolution recipe**: keep ≥~4 cells across the
        (outer-inner) annulus (raise ``freq_max`` to shrink ``dx``); the result
        ``status`` reports ``"under_resolved"`` below ~3.5 cells.

        The conductors deliberately stop ~2 cells short of the +z PML — running
        PEC into CPML is numerically unstable.
        """

        from rfx.probes.probes import init_dft_plane_probe
        from rfx.simulation import run as _run
        from rfx.sources.coaxial_port import (
            CoaxialPort as _CoaxPort,
            build_coaxial_tem_plane_source_specs,
            coaxial_line_plane_voltage,
            coaxial_line_reflection_from_plane_voltages,
            coaxial_tem_characteristic_impedance,
            stamp_coaxial_line,
            stamp_coaxial_short_plane,
            stamp_coaxial_annular_resistor,
        )

        if termination not in ("short", "open", "matched"):
            raise ValueError(
                f"termination must be 'short', 'open' or 'matched', got {termination!r}"
            )
        if len(self._coaxial_ports) != 1:
            raise ValueError(
                "compute_coaxial_line_reflection() is a one-port method; register "
                "exactly one add_coaxial_port()."
            )
        if (
            self._ports or self._waveguide_ports or self._floquet_ports or self._msl_ports
        ):
            raise NotImplementedError(
                "compute_coaxial_line_reflection() is defined only for a single "
                "add_coaxial_port(...) family."
            )
        port = self._coaxial_ports[0]
        if port.face != "top":
            raise NotImplementedError(
                "compute_coaxial_line_reflection() currently supports face='top' "
                "(pin along -z, DUT at the -z end); face='bottom' is symmetric and "
                "not yet wired."
            )

        grid = self._build_grid()
        nz = grid.shape[2]
        dz = float(grid.dx)
        center_xy = (float(port.position[0]), float(port.position[1]))
        a, b = float(port.pin_radius), float(port.outer_radius)

        # Axial layout: DUT just above the -z PML; coax runs up to ~2 cells short
        # of the +z PML; matched feed one cell below the coax top; source below it.
        # The +z offset uses pad_z_hi (not pad_z_lo) so an asymmetric BoundarySpec
        # cannot run the conductors into the +z PML (verified unstable).
        z_dut = int(grid.pad_z_lo) + int(dut_offset_cells)
        z_hi_coax = nz - int(grid.pad_z_hi) - 2
        z_feed = z_hi_coax - 1
        z_src = z_hi_coax - 3
        if not (z_dut + probe_start_cells + 2 * probe_spacing_cells < z_src):
            raise ValueError(
                "domain too short for the requested line layout; increase the z "
                "domain or reduce probe_start_cells/probe_count."
            )
        probes_z = [
            z_dut + int(probe_start_cells) + int(probe_spacing_cells) * k
            for k in range(int(probe_count))
        ]
        probes_z = [z for z in probes_z if z < z_src - 4]
        if len(probes_z) < 3:
            raise ValueError(
                "fewer than 3 usable probe planes; reduce probe_spacing_cells or "
                "lengthen the z domain."
            )

        z_tem = coaxial_tem_characteristic_impedance(a, b)
        R_feed = float(feed_impedance) if feed_impedance is not None else float(z_tem)
        # For termination='matched', the DUT load resistance defaults to the feed
        # (Γ→0); override with dut_impedance to place a known mismatch
        # (Γ = (R-Z0)/(R+Z0)) — used by the broad-E5 envelope's non-trivial loads.
        R_dut = float(dut_impedance) if dut_impedance is not None else R_feed

        materials, _, _ = self._build_materials(grid)
        materials, shell_inner = stamp_coaxial_line(
            grid, materials, center_xy=center_xy, z_lo_index=z_dut,
            z_hi_index=z_hi_coax, pin_radius=a, outer_radius=b,
        )
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=z_feed, pin_radius=a,
            outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
        )
        if termination == "short":
            materials = stamp_coaxial_short_plane(
                grid, materials, center_xy=center_xy, z_index=z_dut, outer_radius=b,
            )
        elif termination == "matched":
            materials = stamp_coaxial_annular_resistor(
                grid, materials, center_xy=center_xy, z_index=z_dut, pin_radius=a,
                outer_radius=b, target_impedance=R_dut, shell_inner_radius=shell_inner,
            )
        # "open": conductors simply end at z_dut (open circuit) — no extra stamp.

        if freqs is None:
            freqs = jnp.linspace(
                0.1 * self._freq_max, 0.6 * self._freq_max, int(n_freqs), dtype=jnp.float32
            )
        else:
            freqs = jnp.asarray(freqs, dtype=jnp.float32)

        # TEM TFSF source at z_src (internal descriptor places pin_center there).
        src_port = _CoaxPort(
            position=(center_xy[0], center_xy[1], (z_src - grid.pad_z_lo) * dz),
            face="top", pin_length=dz, pin_radius=a, outer_radius=b,
            impedance=port.impedance, excitation=port.excitation,
        )
        spec = build_coaxial_tem_plane_source_specs(
            grid=grid, port=src_port, n_steps=int(n_steps), field_scale=float(field_scale),
            magnetic_ratio=1.0,
        )

        planes = []
        for z in probes_z:
            for comp in ("ex", "ey"):
                planes.append(
                    init_dft_plane_probe(
                        axis=2, index=int(z), component=comp, freqs=freqs,
                        grid_shape=grid.shape, dft_total_steps=int(n_steps),
                    )
                )
        result = _run(
            grid, materials, int(n_steps), boundary="cpml", cpml_axes=cpml_axes,
            sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
            dft_planes=planes, return_state=False,
        )
        if result.dft_planes is None:
            raise RuntimeError("compute_coaxial_line_reflection(): runner returned no DFT planes")

        # Modal voltage V(z) at every probe plane, per frequency.
        n_f = int(freqs.shape[0])
        v_by_plane = []
        for pi in range(len(probes_z)):
            ex = result.dft_planes[pi * 2 + 0].accumulator
            ey = result.dft_planes[pi * 2 + 1].accumulator
            v_by_plane.append(
                coaxial_line_plane_voltage(
                    grid, ex, ey, center_xy=center_xy, pin_radius=a, outer_radius=b,
                )
            )
        V = np.stack(v_by_plane, axis=0)          # (n_planes, n_freqs)
        z_planes_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_z], dtype=np.float64)
        ref_m = (z_dut - grid.pad_z_lo) * dz

        s11 = np.zeros(n_f, dtype=np.complex128)
        gamma = np.zeros(n_f, dtype=np.complex128)
        rec_resid = np.zeros(n_f, dtype=np.float64)
        fit_resid = np.zeros(n_f, dtype=np.float64)
        z0_num = np.full(n_f, np.nan + 1j * np.nan, dtype=np.complex128)
        for fi in range(n_f):
            out = coaxial_line_reflection_from_plane_voltages(
                z_planes_m, V[:, fi], reference_plane_m=ref_m,
            )
            s11[fi] = out.reflection
            gamma[fi] = out.gamma
            rec_resid[fi] = out.recurrence_residual
            fit_resid[fi] = out.fit_residual
            if termination == "matched":
                G = out.reflection
                z0_num[fi] = R_dut * (1.0 - G) / (1.0 + G)

        annulus_cells = float((b - a) / dz)
        if annulus_cells < 3.5:
            status = "under_resolved"
        elif float(np.max(rec_resid)) > 0.1:
            status = "contaminated"
        else:
            status = "passed"

        return CoaxialLineReflectionResult(
            s11=s11,
            freqs=np.asarray(freqs, dtype=float),
            gamma=gamma,
            recurrence_residual=rec_resid,
            fit_residual=fit_resid,
            annulus_cells=annulus_cells,
            z0_numerical_ohm=z0_num,
            termination=termination,
            status=status,
        )

    def _compute_waveguide_s_matrix_nu(
        self,
        *,
        n_steps: int | None,
        num_periods: float,
        normalize: bool,
        eps_override=None,
        sigma_override=None,
        checkpoint_segments: int | None = None,
    ) -> WaveguideSMatrixResult:
        """Non-uniform-mesh two-run S-matrix extraction.

        Drives each port in turn, running device + vacuum-reference
        scans through ``run_nonuniform_path`` so ``dx_profile`` /
        ``dy_profile`` actually flow into the Yee update. The per-port
        drive is implemented by temporarily zeroing ``amplitude`` on
        non-driven entries; the original port list is restored in a
        ``finally`` block. Reference run uses ``eps_override`` /
        ``sigma_override`` to replace the assembled materials with
        vacuum before the scan launches.

        Current scope (matches the uniform path minus a few niceties):
          - ``normalize=True`` or ``normalize='flux'``.
          - Single-mode ports (``n_modes == 1``) only.
          - ``eps_override`` / ``sigma_override`` (the differentiable AD
            design variable) are wired only for ``normalize='flux'``: they
            are threaded into the *device* run so the traced eps flows
            through the jnp-native flux extraction and back to the
            S-matrix gradient. The *reference* run stays vacuum. They are
            rejected for ``normalize=True`` (its diagonal a_inc_ref
            denominator carries the #88 band-edge fragility).

        Extracts ``a_inc`` / ``b_out`` via the same
        ``extract_waveguide_port_waves`` helper as the uniform path and
        applies the same diagonal-subtraction + off-diagonal-division
        normalisation (see ``extract_waveguide_s_params_normalized``
        in ``rfx/sources/waveguide_port.py``).
        """
        from dataclasses import replace as _dc_replace
        from rfx.runners.nonuniform import (
            run_nonuniform_path,
            assemble_materials_nu,
        )
        from rfx.sources.waveguide_port import (
            extract_waveguide_port_waves,
            waveguide_plane_positions,
        )

        # ``normalize`` may be True (lumped V/I ratio) or "flux" (Poynting
        # power-ratio magnitude + modal phase). normalize=False is not
        # supported on the NU path.
        _flux_mode = (normalize == "flux")
        if not normalize:
            raise NotImplementedError(
                "compute_waveguide_s_matrix(normalize=False) is not yet "
                "supported on the non-uniform mesh path; use normalize=True, "
                "normalize='flux', or drop dx/dy_profile to stay on the "
                "uniform lane."
            )

        entries = list(self._waveguide_ports)
        if any(entry.n_modes > 1 for entry in entries):
            raise NotImplementedError(
                "Multi-mode waveguide ports are not yet supported on the "
                "non-uniform mesh path."
            )

        n_ports = len(entries)

        # ``_build_nonuniform_grid`` requires a concrete dz_profile.
        # Synthesise one from the scalar dx when the user did not supply
        # a dz_profile (same semantics as the uniform lane's implicit
        # z-resolution). Restored in the ``finally`` below.
        _dz_profile_saved = self._dz_profile
        if self._dz_profile is None:
            _nz = int(round(float(self._domain[2]) / float(self._dx)))
            self._dz_profile = np.full(max(_nz, 1), float(self._dx))

        # Build the grid directly so we can restrict ``cpml_axes`` to
        # axes that are not fully PEC/PMC-bounded. The rasteriser (see
        # ``rfx/geometry/rasterize.py::coords_from_nonuniform_grid``)
        # uses a single ``grid.cpml_layers`` offset for every axis;
        # when a fully PEC-bounded axis is shorter than
        # ``cpml_layers + 1`` cells the offset slice hits IndexError.
        # Dropping that axis from ``cpml_axes`` keeps the physical
        # grid identical (PEC faces already have pad=0) but zeroes the
        # offset so the rasteriser snaps cells to 0 cleanly.
        from rfx.runners.nonuniform import build_nonuniform_grid
        pec_set = (self._boundary_spec.pec_faces()
                   if self._boundary_spec is not None else None) or set()
        pmc_set = (self._boundary_spec.pmc_faces()
                   if self._boundary_spec is not None else None) or set()

        def _axis_fully_closed(ax: str) -> bool:
            return {f"{ax}_lo", f"{ax}_hi"}.issubset(pec_set | pmc_set)

        cpml_axes = "".join(
            ax for ax in "xyz"
            if ax not in (self._periodic_axes or "")
            and not _axis_fully_closed(ax)
        )
        try:
            grid = build_nonuniform_grid(
                self._freq_max, self._domain, self._dx, self._cpml_layers,
                self._dz_profile,
                dx_profile=self._dx_profile,
                dy_profile=self._dy_profile,
                pec_faces=pec_set or None,
                pmc_faces=pmc_set or None,
                cpml_axes=cpml_axes,
            )
        except Exception:
            self._dz_profile = _dz_profile_saved
            raise
        if n_steps is None:
            # ``NonUniformGrid`` does not expose ``num_timesteps`` (known
            # asymmetry vs. ``Grid``); inline the same formula here.
            n_steps = int(np.ceil(num_periods / self._freq_max / float(grid.dt)))

        # Assemble device materials once to learn the full array shape;
        # vacuum reference is shape-matched onto that same array.
        dev_materials_concrete, _, _, _ = assemble_materials_nu(self, grid)
        vacuum_eps = jnp.ones_like(dev_materials_concrete.eps_r)
        vacuum_sigma = jnp.zeros_like(dev_materials_concrete.sigma)

        # Frequency grid must match across ports.
        port_freqs = entries[0].freqs
        if port_freqs is None:
            port_freqs = jnp.linspace(
                self._freq_max / 10, self._freq_max, entries[0].n_freqs,
            )
        for entry in entries[1:]:
            other = entry.freqs if entry.freqs is not None else jnp.linspace(
                self._freq_max / 10, self._freq_max, entry.n_freqs,
            )
            if other.shape != port_freqs.shape or not np.allclose(
                np.asarray(other), np.asarray(port_freqs)
            ):
                raise ValueError(
                    "waveguide S-matrix requires matching frequency grids on all ports"
                )

        # jnp-functional: collect per-drive columns; stack after loop
        s_columns: list[list] = []  # s_columns[drive_idx] = list of (n_freqs,) jnp arrays over recv_idx
        ref_shifts: tuple[float, ...] | None = None
        reference_planes_out: np.ndarray | None = None

        original_entries = list(entries)
        try:
            for drive_idx in range(n_ports):
                self._waveguide_ports = [
                    _dc_replace(
                        e,
                        amplitude=(e.amplitude if idx == drive_idx else 0.0),
                    )
                    for idx, e in enumerate(original_entries)
                ]

                # Device run: thread the public eps/sigma override (the
                # traced design variable) into the device Yee update so the
                # gradient flows from eps_override -> device fields -> flux
                # -> S-matrix (mirrors the uniform PR #172 flux-AD wiring).
                # When eps_override is None the assembled device materials
                # are used unchanged (device fields identical); the np->jnp
                # flux extraction below matches the prior np path to
                # rtol<=1e-5 (float reassociation only, per uniform PR #172).
                # issue #73: translate the uniform `checkpoint_segments` (K
                # segments) → the NU runner's `checkpoint_every` (chunk size).
                # The chunk MUST exactly divide n_steps (pad=0): the NU runner
                # zero-pads non-divisor chunks and those extra ring-down steps
                # would corrupt the carry-accumulated flux DFT (the time_series
                # is truncated to n_steps but the flux accumulator is NOT), so a
                # non-divisor chunk is NOT forward-identical for the flux
                # S-matrix — same divisor rule as the uniform V/I-DFT path. Pick
                # the divisor of n_steps nearest to n_steps/K. Checkpoint ONLY the
                # *device* run — the vacuum reference is constant in the design
                # variable so it carries no AD tape. The √N tape win is realised
                # under jax.grad (normalize='flux' + eps_override); plain forward
                # is identical.
                from rfx.simulation import _nearest_divisor
                _ckpt_every = None
                if checkpoint_segments is not None and checkpoint_segments > 1:
                    _ck = _nearest_divisor(n_steps, max(1, n_steps // int(checkpoint_segments)))
                    if 0 < _ck < n_steps:
                        _ckpt_every = _ck
                dev_result = run_nonuniform_path(
                    self, n_steps=n_steps,
                    eps_override=eps_override,
                    sigma_override=sigma_override,
                    attach_waveguide_flux=_flux_mode,
                    checkpoint_every=_ckpt_every,
                )
                # Reference run stays vacuum (incident-power reference) and is
                # independent of the design variable. ``strip_interior_pec``
                # drops the rasterized interior PEC (iris / wall / post) from
                # the reference so it is a clean empty guide: the boundary y/z
                # guide walls survive (they are enforced via pec_faces, not
                # pec_mask). Without this the vacuum override replaces only
                # eps/sigma and the reference retains the device's interior
                # PEC mask → device and reference DFTs are bit-identical →
                # (device - reference) = 0 → S11 = 0 for any PEC reflector.
                # This mirrors the uniform reference run, which builds the
                # reference with dielectric_shapes=[] + boundary-only PEC.
                ref_result = run_nonuniform_path(
                    self,
                    n_steps=n_steps,
                    eps_override=vacuum_eps,
                    sigma_override=vacuum_sigma,
                    attach_waveguide_flux=_flux_mode,
                    strip_interior_pec=True,
                )

                dev_wg = dev_result.waveguide_ports or {}
                ref_wg = ref_result.waveguide_ports or {}
                if len(dev_wg) != n_ports or len(ref_wg) != n_ports:
                    raise RuntimeError(
                        "NU waveguide S-matrix expected one final cfg per "
                        "port on both device and reference runs"
                    )

                # Compute ref_shifts from the first drive's configs (same
                # measured planes for every drive / run).
                if ref_shifts is None:
                    shifts = []
                    planes_out = []
                    for entry in original_entries:
                        cfg = dev_wg[entry.name]
                        planes = waveguide_plane_positions(cfg)
                        desired = (
                            entry.reference_plane
                            if entry.reference_plane is not None
                            else planes["source"]
                        )
                        shifts.append(desired - planes["reference"])
                        planes_out.append(desired)
                    ref_shifts = tuple(shifts)
                    reference_planes_out = np.asarray(planes_out, dtype=float)

                drive_name = original_entries[drive_idx].name
                a_inc_ref, _ = extract_waveguide_port_waves(
                    ref_wg[drive_name], ref_shift=ref_shifts[drive_idx],
                )
                safe_a_inc = jnp.where(
                    jnp.abs(a_inc_ref) > 1e-30,
                    a_inc_ref,
                    jnp.ones_like(a_inc_ref),
                )

                if _flux_mode:
                    # Power-flux magnitude + modal phase (mirrors the
                    # uniform extract_waveguide_s_matrix_flux). Immune to
                    # the band-edge a_inc_ref denominator collapse that
                    # makes the normalize=True diagonal blow up (issue #88):
                    # P_inc = |F_ref[drive]| is large and well-conditioned
                    # across the whole band, not source-spectrum-weighted.
                    F_ref = ref_result.waveguide_port_flux
                    F_dev = dev_result.waveguide_port_flux
                    if F_ref is None or F_dev is None:
                        raise RuntimeError(
                            "normalize='flux' on the NU path requires "
                            "per-port flux spectra; run_nonuniform_path did "
                            "not return waveguide_port_flux."
                        )
                    # jnp-native (mirrors the uniform PR #172 flux-AD fix):
                    # no np.asarray() concretization — keeps the whole flux
                    # extraction on the AD tape so an eps_override-traced
                    # device run yields finite gradients through the
                    # S-matrix. Uses the DOUBLE-WHERE trick at sqrt(0) /
                    # angle(0) (guard the INPUT, not just the output): a
                    # single jnp.where still leaks NaN grad through the dead
                    # branch (#171/#172/#148). Forward values are identical
                    # to the prior np version for P_inc / P > 0.
                    P_inc = jnp.abs(F_ref[drive_idx])
                    safe_P_inc = jnp.where(
                        P_inc > 1e-60, P_inc, jnp.ones_like(P_inc)
                    )
                    recv_col = []
                    for recv_idx in range(n_ports):
                        recv_name = original_entries[recv_idx].name
                        _, b_recv_dev = extract_waveguide_port_waves(
                            dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                        )
                        # AD-safe angle (double-where): angle() has an
                        # undefined gradient at 0; angle(1)=0 matches
                        # np.angle(0)=0 so the primal is unchanged.
                        ratio = b_recv_dev / safe_a_inc
                        ratio_ok = jnp.abs(ratio) > 0.0
                        phase = jnp.angle(
                            jnp.where(ratio_ok, ratio, jnp.ones_like(ratio))
                        )
                        if recv_idx == drive_idx:
                            P_num = jnp.abs(F_ref[drive_idx] - F_dev[drive_idx])
                        else:
                            P_num = jnp.abs(F_dev[recv_idx])
                        # AD-safe sqrt (double-where): a perfect match/null
                        # makes the power ratio exactly 0, where
                        # d(sqrt)/dx = inf would leak 0*inf=nan through the
                        # backward pass; primal stays exactly sqrt(x) for
                        # x>0 and exactly 0 at x=0.
                        p_ratio = P_num / safe_P_inc
                        p_ok = p_ratio > 0.0
                        mag = jnp.where(
                            p_ok,
                            jnp.sqrt(
                                jnp.where(p_ok, p_ratio, jnp.ones_like(p_ratio))
                            ),
                            0.0,
                        )
                        recv_col.append(mag * jnp.exp(1j * phase))
                    s_columns.append(recv_col)
                    continue

                recv_col: list = []
                for recv_idx in range(n_ports):
                    recv_name = original_entries[recv_idx].name
                    _, b_ref = extract_waveguide_port_waves(
                        ref_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                    )
                    _, b_dev = extract_waveguide_port_waves(
                        dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                    )
                    if recv_idx == drive_idx:
                        recv_col.append((b_dev - b_ref) / safe_a_inc)
                    else:
                        # Use a tighter guard than the diagonal safe_a_inc
                        # (1e-30): the NU path operates at lower float32
                        # signal levels (~1e-31) because the TFSF table
                        # injection scales with dt/dx. The reference
                        # outgoing wave b_ref at non-driven ports is
                        # proportional to the driven-port incident wave and
                        # can fall to ~1e-31 in float32. A 1e-30 guard
                        # fires falsely and replaces b_ref with 1.0, giving
                        # S21 = b_dev * 1e-31 instead of b_dev/b_ref ≈ 1.
                        # 1e-60 is safely below float32 underflow (~1e-38)
                        # so it only fires when b_ref is genuinely zero.
                        safe_b = jnp.where(
                            jnp.abs(b_ref) > 1e-60,
                            b_ref,
                            jnp.ones_like(b_ref),
                        )
                        recv_col.append(b_dev / safe_b)
                s_columns.append(recv_col)
        finally:
            self._waveguide_ports = original_entries
            self._dz_profile = _dz_profile_saved

        return WaveguideSMatrixResult(
            s_params=jnp.stack([jnp.stack(col) for col in s_columns], axis=1),
            freqs=jnp.asarray(port_freqs),
            port_names=tuple(e.name for e in original_entries),
            port_directions=tuple(e.direction for e in original_entries),
            reference_planes=reference_planes_out
            if reference_planes_out is not None
            else np.array(
                [
                    e.reference_plane if e.reference_plane is not None
                    else 0.0
                    for e in original_entries
                ],
                dtype=float,
            ),
        )

