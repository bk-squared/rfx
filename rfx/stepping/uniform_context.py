"""Static context shared by the uniform scan drivers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from rfx.core.drives import StepDrives


@dataclass(frozen=True)
class _StepContext:
    """Static + array context for the shared Yee step kernel (W6.1).

    Holds every free variable the per-step kernel needs: physical constants,
    the resolved ``use_*`` subsystem flags, pre-extracted metadata tuples, and
    the boundary/dispersion helper callables that each caller imports during
    setup.  Lazily-imported helpers (waveguide-port, PMC, conformal, pec-mask,
    NTFF accumulate, RLC update) are re-imported inside the kernel body exactly
    as the original closures did and are therefore NOT stored here.
    """

    # ---- physical constants / grid ----
    grid: Any
    materials: Any
    dt: Any
    dx: Any
    periodic: tuple
    pec_axes: str
    # (2,4) fourth-order-in-space stencil order: 2 (default, byte-identical)
    # or 4. Threaded into the H / E kernel calls in core_step.
    stencil_order: int

    # ---- subsystem flags ----
    use_fast_he: bool
    use_upml: bool
    use_cpml: bool
    use_tfsf: bool
    use_debye: bool
    use_lorentz: bool
    use_ntff: bool
    use_dft_planes: bool
    use_flux_monitors: bool
    use_waveguide_ports: bool
    use_pec_faces: bool
    use_pmc_faces: bool
    use_aniso_inv: bool
    aniso_inv_eps_smooth: bool
    use_pec_edges: bool
    use_pec_occupancy: bool
    use_conformal: bool
    use_wire_sparams: bool
    use_lumped_sparams: bool
    use_lumped_rlc: bool
    use_kerr: bool

    # ---- output behaviour ----
    use_snapshot: bool
    use_monitor: bool
    use_flux_window: bool

    # ---- arrays / coeffs / configs ----
    fast_coeffs: Any = None
    cpml_params: Any = None
    cpml_axes: str = ""
    upml_coeffs: Any = None
    tfsf_cfg: Any = None
    tfsf_is_2d: bool = False
    bloch: Any = None  # per-axis Bloch phase for oblique-periodic TFSF (#404); None = real path
    debye_coeffs: Any = None
    lorentz_coeffs: Any = None
    pec_edge_masks: Any = None
    pec_static_edge_masks: Any = None
    kerr_chi3: Any = None
    ntff: Any = None
    # In-loop block current moments (rfx.current_moments). Default OFF so
    # every existing caller's context is unchanged.
    use_current_moments: bool = False
    current_moments: Any = None
    curl_boundary: Any = None
    pec_faces_frozen: Any = frozenset()
    pmc_faces_frozen: Any = frozenset()

    # ---- pre-extracted metadata ----
    # Sole source authority: None suppresses BOTH stages, including when xs
    # still contains primal samples (the reciprocity adjoint pass).
    drives: StepDrives | None = None
    electric_injector: Callable | None = None
    prb_meta: tuple = ()
    dft_meta: tuple = ()
    flux_meta: tuple = ()
    waveguide_meta: tuple = ()
    port_record_specs: tuple | None = None
    wire_sparam_meta: tuple = ()
    lumped_sparam_meta: tuple = ()
    rlc_meta: tuple = ()
    # issue #313 opt-in reference-plane channel (defaults keep every
    # existing caller byte-identical)
    use_wire_refplanes: bool = False
    wire_refplane_meta: tuple = ()
    # issue #677 node-thin surface-impedance sheet operator (defaults keep
    # every existing caller byte-identical)
    use_sheet_impedance: bool = False
    sheet_impedance: Any = None
    # issue #1179 design-box permittivity (defaults keep every existing
    # caller byte-identical)
    use_design_box: bool = False
    design_box: Any = None
    # issue #1183 design-box PEC occupancy (same default discipline)
    use_design_occupancy: bool = False
    design_occupancy: Any = None

    # ---- output extractors ----
    monitor_component: str = "ez"
    mon_idx: Any = None
    snapshot_extractor: Callable | None = None

    # ---- caller-imported helper callables ----
    apply_cpml_h: Callable | None = None
    apply_cpml_e: Callable | None = None
    apply_upml_h: Callable | None = None
    apply_upml_e: Callable | None = None
    apply_tfsf_h: Callable | None = None
    apply_tfsf_e: Callable | None = None
    update_tfsf_1d_h: Callable | None = None
    update_tfsf_1d_e: Callable | None = None
    update_tfsf_2d_h: Callable | None = None
    update_tfsf_2d_e: Callable | None = None
    init_ntff_data: Callable | None = None
    accumulate_ntff: Callable | None = None
    accumulate_current_moments: Callable | None = None
    apply_kerr_ade: Callable | None = None
    update_rlc_element: Callable | None = None

    @property
    def src_meta(self):
        """Read-only aperture view for legacy diagnostics, never a kernel input."""
        if self.drives is None:
            return ()
        return tuple((i, j, k, component)
                     for component, nodes in self.drives.electric.nodes.items()
                     for i, j, k in zip(*nodes))

