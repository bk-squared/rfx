"""Rectangular waveguide port: analytical TE/TM mode profiles.

Excites and extracts waveguide modes using precomputed analytical profiles
on the port cross-section. Supports TE_mn and TM_mn modes for rectangular
waveguides with PEC walls.

The port sits on a y-z plane at a fixed x index. Mode propagation is along +x.

TE_mn transverse E-field profiles (Pozar, mapped to prop-along-x):
    Ey(y,z) = -(nπ/b) cos(mπy/a) sin(nπz/b)
    Ez(y,z) =  (mπ/a) sin(mπy/a) cos(nπz/b)

where k_c² = (mπ/a)² + (nπ/b)², a = waveguide width (y),
b = waveguide height (z).

Key examples:
    TE10: Ey = 0,  Ez = (π/a) sin(πy/a)
    TE01: Ey = -(π/b) sin(πz/b),  Ez = 0

S21 is extracted using V/I forward-wave decomposition at two probe planes:
    a_fwd(f) = (V(f) + Z_TE(f) * I(f)) / 2
    S21(f)   = a_fwd_probe(f) / a_fwd_ref(f)
This removes the worst standing-wave inflation of voltage-only ratios, though
individual points can still deviate under finite-window/CPML error.

S-matrix extractor decision table
---------------------------------
Which extractor runs is chosen by ``Simulation.compute_waveguide_s_matrix``
(see ``rfx/api/_sparams.py``) from the ``normalize=`` argument, the number of
modes per port, and the mesh path.  This module exports the extractors; the
``compute_waveguide_s_matrix`` dispatch wires them as follows:

    n_modes  mesh        normalize=    extractor called
    -------  ----------  ------------  --------------------------------------
    1        uniform     False         extract_waveguide_s_matrix
    1        uniform     True          extract_waveguide_s_params_normalized
    1        uniform     "flux"        extract_waveguide_s_matrix_flux
    >1       uniform     False         extract_multimode_s_matrix
    >1       uniform     True          (unsupported — raises ValueError)
    >1       uniform     "flux"        extract_multimode_s_matrix_flux
    1        non-uniform True/"flux"   inline flux/normalized logic in
                                       compute_waveguide_s_matrix (mirrors the
                                       uniform extractors; normalize=False and
                                       multi-mode are unsupported on NU)

``normalize=`` semantics:
  - ``False`` — single-run V/I wave decomposition, no reference run.  Carries
    documented Yee-dispersion + band-edge error (|S11| can exceed 1); cheapest.
  - ``True``  — two-run reference normalization (``b_dev / b_ref`` off-diagonal,
    ``(b_dev − b_ref) / a_inc`` diagonal).  Cancels one-way Yee dispersion for
    transmission; AD-traceable.  Unsupported for multi-mode ports because the
    empty-guide cross-mode reference ``b_ref ≈ 0`` produces a 0/0.
  - ``"flux"`` — two-run power-flux magnitude (Poynting-flux ratio for the
    single-mode path, ``0.5·Re(V·I*)`` modal-power ratio for the multi-mode
    path) combined with modal-V/I phase.  Most robust at band edges; NOT
    AD-traceable (drops to NumPy internally — issue #148).

Open issue: waveguide flux extraction is not strictly passive at all run
lengths (residual magnitude can drift slightly above unity — issue #150);
this is a known, unfixed limitation, not addressed here.
"""

from __future__ import annotations

from rfx.model.waveguide_modes import (
    WaveguidePort,
    WaveguidePortConfig,
    C0_LOCAL,
)
from rfx.model import (
    cutoff_frequency,
    init_waveguide_port,
    solve_rectangular_modes,
    init_multimode_waveguide_port,
)
from rfx.stepping import (
    inject_waveguide_port,
    apply_waveguide_port_h,
    apply_waveguide_port_e,
)
from rfx.measurement import (
    modal_voltage,
    modal_current,
    mode_self_overlap,
    overlap_modal_amplitude,
    update_waveguide_port_probe,
    init_overlap_dft,
    update_overlap_dft,
    extract_waveguide_sparams_overlap,
    settling_db_from_named_records,
    settling_db_from_port_records,
    waveguide_plane_positions,
    extract_waveguide_sparams,
    extract_waveguide_s21,
    extract_waveguide_s11,
    extract_waveguide_port_waves,
    extract_waveguide_s_matrix,
    extract_waveguide_s_matrix_flux,
    extract_waveguide_s_params_normalized,
    extract_multimode_s_matrix,
    extract_multimode_s_matrix_flux,
)
from rfx.measurement.waveguide_modal import (
    OverlapDFTAccumulators,
)

__all__ = [
    "WaveguidePort",
    "WaveguidePortConfig",
    "cutoff_frequency",
    "init_waveguide_port",
    "solve_rectangular_modes",
    "init_multimode_waveguide_port",
    "C0_LOCAL",
    "inject_waveguide_port",
    "apply_waveguide_port_h",
    "apply_waveguide_port_e",
    "modal_voltage",
    "modal_current",
    "mode_self_overlap",
    "overlap_modal_amplitude",
    "update_waveguide_port_probe",
    "OverlapDFTAccumulators",
    "init_overlap_dft",
    "update_overlap_dft",
    "extract_waveguide_sparams_overlap",
    "settling_db_from_named_records",
    "settling_db_from_port_records",
    "waveguide_plane_positions",
    "extract_waveguide_sparams",
    "extract_waveguide_s21",
    "extract_waveguide_s11",
    "extract_waveguide_port_waves",
    "extract_waveguide_s_matrix",
    "extract_waveguide_s_matrix_flux",
    "extract_waveguide_s_params_normalized",
    "extract_multimode_s_matrix",
    "extract_multimode_s_matrix_flux",
]
