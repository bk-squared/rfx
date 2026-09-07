"""Slab-family leaf module: the shared rig declaration and the calibration
envelope loader (issue #928).

Two jobs. The module is a LEAF: stdlib, numpy and the leaf ``dispersive_eps``
only -- no rfx, no jax, and above all no sibling that imports it back. That is
what lets both the PRODUCER (``04_multilayer_fresnel.py``) and the CONSUMERS
(``cv22_dispersive_gates.py``, ``cv23_lossy_gates.py``, ``slab_rig.py``) import
it with nothing importing a consumer:

1. **The rig.** The cv04 TFSF slab rig's constants used to live in
   ``cv22_dispersive_gates.py`` -- a module named after a CONSUMER -- and ten
   modules, the producer among them, imported them from there. The direction
   was backwards. They live here now; ``cv22_dispersive_gates`` re-exports them
   so its importers are unchanged.

2. **The calibration envelope.** ``load_adopted_envelope`` reads the producer's
   envelope artifact and returns the values of the revision the consumer's
   ADOPTION RECORD names. Three separate things, kept separate on purpose:

   * *evidence* -- the numbers the producer measured, in the producer's own
     artifact, append-only by revision;
   * *calibration* -- which revision a consumer adopted, declared in the
     consumer with the revision's own hash and the gate policy it was adopted
     under, changeable only by editing that declaration;
   * *expectation* -- the window, derived from the two and written down
     nowhere else.

   The consequence that matters: re-running the producer and appending a new
   revision does NOT move any consumer's window. Adoption is an edit to the
   consumer's declaration, reviewed under the repo's no-silent-gate-loosening
   rule; a producer cannot widen the gates that judge its own family.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
from typing import NamedTuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dispersive_eps as de  # noqa: E402

# ---------------------------------------------------------------------------
# 1. The cv04 TFSF slab rig
#
# These constants used to live in ``cv22_dispersive_gates`` -- a module named
# after a CONSUMER -- and ten modules imported them from there, the PRODUCER
# among them. Values are unchanged and bit-identical; the baseline that proves
# it is tests/fixtures/slab_family_windows_baseline.json.
# ---------------------------------------------------------------------------
DX_M = 1.0e-3
D_SLAB_M = 10.0e-3
EPS_SLAB = 4.0                 # the producer's slab: lossless, eps_r = 4
# The speed of light the PRODUCER's script uses. It is the rounded value, not
# `C0` below (dispersive_eps' exact c, which the analytic models use); the two
# differ by 2.5e-5 relative and the difference is real -- cv04's committed
# numbers were measured with the rounded one, so the emitter that describes
# that run must use the same. Declared here so there is ONE home for it.
C0_SCRIPT = 2.998e8
NX_INTERIOR = 600
N_CPML = 20
TFSF_F0_HZ = 10.0e9
TFSF_BW = 0.5
NFFT_OVERSAMPLE = 8
MASK_F_LO_HZ = 3.0e9
MASK_F_HI_HZ = 15.0e9
MASK_AMP_FRAC = 0.02
# cv04's settling-tail witness and per-bin closure ceiling.
TAIL_WINDOW = 50
TAIL_PURITY_LIMIT = 1e-3
TAIL_LIMIT = 0.10
CONS_MAX_LIMIT = 0.06

# The band the family EVALUATES in (cv22 note section 5, cv23's the same, and
# the band cv04's lattice witness marks): rig-level, so the producer can mark
# its own gated bins without importing a consumer.
BAND_GATED_HZ = (4.0e9, 10.0e9)
# The incident ring band: the part of the pulse whose amplitude is at least
# this fraction of the peak.
RING_W_MIN = 0.5
RING_F_MAX_HZ = MASK_F_HI_HZ


def gated_mask(freqs_hz) -> np.ndarray:
    f = np.asarray(freqs_hz, dtype=float)
    return (f >= BAND_GATED_HZ[0]) & (f <= BAND_GATED_HZ[1])


def incident_amplitude_rel(f_hz):
    """Amplitude spectrum of the rig's differentiated-Gaussian incident pulse,
    relative to its peak: |S(f)| ∝ f exp(-(pi f tau)^2), tau = 1/(pi f0 bw)."""
    tau = 1.0 / (math.pi * TFSF_F0_HZ * TFSF_BW)
    f = np.asarray(f_hz, dtype=float)
    s = f * np.exp(-(math.pi * f * tau) ** 2)
    peak = (1.0 / (math.sqrt(2.0) * math.pi * tau)) * math.exp(-0.5)
    return s / peak


def ring_band_hz():
    """[f_lo, RING_F_MAX_HZ]: the incident band with amplitude >= RING_W_MIN of peak."""
    f = np.linspace(1e7, TFSF_F0_HZ, 20000)
    w = incident_amplitude_rel(f)
    f_lo = float(f[np.argmax(w >= RING_W_MIN)])
    return f_lo, RING_F_MAX_HZ


def slab_ringdown_rates(model: str, params: dict):
    """Amplitude decay rates (1/s) of the slab's own ring-down over the incident
    ring band: the material pole (Debye 1/tau, Lorentz delta, Drude gamma/2)
    and the etalon round-trip, rho = |r|^2 exp(-2 k0 Im(n) d) per
    t_rt = 2 Re(n) d / c, each component weighted by its incident amplitude
    w(f): a component starting at w needs ln(100 w)/rate to reach -40 dB. The
    slowest entry is the one with the largest ln(100 w)/rate.

    Rig + material physics, shared by the producer's witness and both
    consumers' record-length recipes -- which is why it lives here and not in
    either consumer.
    """
    f_lo, f_hi = ring_band_hz()
    f = np.linspace(f_lo, f_hi, 1401)
    eps = de.eps_analytic(f, model, params)
    n = np.sqrt(eps)
    # Forward branch: Re n >= 0 (the principal sqrt already has it; for a
    # passive medium, Im eps < 0, that branch is the decaying one). The
    # earlier `where(n.imag > 0, -n, n)` was a no-op for every passive arm
    # but negated Re n for a GAIN medium (cv23's passivity falsifier),
    # giving |r|^2 = 9; found while deriving that arm's record.
    n = np.where(n.real < 0, -n, n)
    k0 = 2.0 * math.pi * f / de.C0
    r = (1 - n) / (1 + n)
    rho = np.abs(r) ** 2 * np.exp(2 * k0 * n.imag * D_SLAB_M)   # |e^{-j k0 n d}|^2 per round trip
    t_rt = 2 * np.abs(n.real) * D_SLAB_M / de.C0
    if np.any(rho >= 1.0):
        raise ValueError(f"{model}: etalon round-trip gain >= 1 at {f[np.argmax(rho)]/1e9:.2f} GHz "
                         f"(rho {rho.max():.3f}); the slab does not ring down")
    rate_et = -np.log(rho) / t_rt
    if model == "debye":
        rate_mat = 1.0 / params["tau"]
    elif model == "lorentz":
        rate_mat = float(params["delta"])
    elif model == "conductive":
        # cv23: J = sigma E is memoryless (no P recurrence, no material
        # ring-down mode); the charge-relaxation pole sigma/(eps0 eps') is a
        # longitudinal mode that normal incidence does not excite. Only the
        # etalon decays, and its absorption per pass is already in rho.
        rate_mat = float("inf")
    else:
        rate_mat = float(params["gamma"]) / 2.0
    w = incident_amplitude_rel(f)
    rate = np.minimum(rate_et, rate_mat)
    t_need = np.log(100.0 * w) / rate       # seconds to -40 dB of the incident peak
    i = int(np.argmax(t_need))
    return {"rate_material_1_s": (float(rate_mat) if math.isfinite(rate_mat) else None),
            "rate_etalon_slowest_1_s": float(rate_et.min()),
            "f_etalon_slowest_hz": float(f[int(np.argmin(rate_et))]),
            "ring_band_hz": [f_lo, f_hi], "ring_w_min": RING_W_MIN,
            "t_ring_s": float(t_need[i]), "f_ring_hz": float(f[i]), "w_ring": float(w[i]),
            "rate_ring_1_s": float(rate[i]), "rho_etalon": float(rho[i]), "t_rt_s": float(t_rt[i])}


class Windows(NamedTuple):
    """The three R/T windows a case ENFORCES, carried explicitly.

    They used to be module-level constants of cv22, which the shared evaluator
    read directly -- so cv23, which calls that evaluator, was judged by cv22's
    windows no matter what cv23's own adoption record said. A round-2 review
    moved cv22's `W_BIN` alone and watched cv23's realized per-bin window move
    4.8x with it. Passing them makes each case's derivation the one that
    reaches its own verdict.
    """

    w_bin: float
    w_mean_R: float
    w_mean_T: float


def aggregate_gates(gates: dict, *, declared=None,
                    require_complete: bool = False) -> dict:
    """Turn a per-gate dict into a verdict, and say whether it is COMPLETE.

    A gate whose evidence is absent is None (no witness was handed in), and the
    default aggregate skips it -- correct for a diagnostic or analytic call
    that has no run behind it. A claims-bearing caller passes
    ``require_complete=True``: then a missing required witness cannot leave a
    PASS standing, because a verdict over two of three gates is not the verdict
    the case declares (issue #928).

    ``declared`` is the case's own gate-name set. Without it, completeness can
    only see a key that is present and None -- an evaluator that never inserts
    the key at all (an early return, a refactor that drops a branch) reads as
    complete, which is the same silence in a different shape. With it, an
    ABSENT declared name is incomplete too.

    Returns ``e2_ok``, ``gates_complete`` and ``incomplete_gates`` for the
    caller to merge into its result dict.
    """
    missing = [] if declared is None else [n for n in declared if n not in gates]
    incomplete = sorted(missing + [name for name, value in gates.items() if value is None])
    if require_complete:
        ok = bool(gates) and not missing and all(
            value is True for value in gates.values())
    else:
        ok = all(value for value in gates.values() if value is not None)
    return {"e2_ok": bool(ok), "gates_complete": not incomplete,
            "incomplete_gates": incomplete}


# The -40 dB settling bar the family's witnesses use, the TFSF margin and the
# probe offsets in cells, and the source's own t0/tau (rfx.sources.tfsf:
# src_t0 = 3 tau for the differentiated Gaussian).
SETTLING_LIMIT = 1e-2
TFSF_MARGIN = 5
PROBE_OFFSET_CELLS = 30
SRC_T0_OVER_TAU = 3.0
C0 = de.C0


def rig_cells(nx_interior: int, dx_div: int = 1):
    """Cell bookkeeping of the cv04 rig (Grid adds 2*n_cpml + 1 cells)."""
    K = int(dx_div)
    n_cpml = N_CPML * K
    nx = int(nx_interior) * K + 2 * n_cpml + 1
    half = int(D_SLAB_M / (2 * DX_M / K))
    slab_lo = nx // 2 - half
    slab_hi = nx // 2 + half
    x_lo = n_cpml + TFSF_MARGIN * K
    probe_refl = slab_lo - PROBE_OFFSET_CELLS * K
    probe_trans = slab_hi + PROBE_OFFSET_CELLS * K
    return {"nx": nx, "n_cpml": n_cpml, "slab_lo": slab_lo, "slab_hi": slab_hi,
            "x_lo": x_lo, "probe_refl": probe_refl, "probe_trans": probe_trans}


# ---------------------------------------------------------------------------
# The auxiliary grid's own absorber echo (issue #888) -- the term the record
# law has always been on the safe side of WITHOUT SAYING SO.
#
# WHAT THIS IS. Every TF/SF injection in this repo reads its incident field
# from an auxiliary grid that carries its own absorber, and that absorber
# reflects. Measured on THIS rig's 1-D auxiliary grid
# (``rfx/sources/tfsf.py``): |B/A| = 4.40e-02 in steady state, from a
# reflector 6.88 cells inside its own 20-cell CPML
# (docs/design_notes/20260903_cv04_envelope_decomposition.md sections 2, 4.1).
# The 2-D Bloch path (``rfx/sources/tfsf_2d.py``) reflects the same 4-6 %
# class from 8 cells inside its 30-cell absorber
# (docs/design_notes/20260903_cv26_oblique_defect_diagnosis.md section 3).
#
# WHY NOTHING SEES IT. The case normalises R = |E_tot - E_inc|^2/|E_inc|^2 and
# T = |E_tot|^2/|E_inc|^2 with E_inc read from that same auxiliary grid, so the
# contamination cancels IDENTICALLY in vacuum and the leakage / purity
# witnesses read a steady standing wave as "settled". It enters the measured R
# and T only once the record is long enough for the echo to reach the probes.
#
# WHY IT IS A RECORD QUESTION, NOT A MARGIN. The record law counts from the
# PROBE (``t_safe`` = 0.95 x 2 dist(probe -> 3-D CPML)/v); the echo's path
# counts from the auxiliary SOURCE, through the auxiliary reflector, back to
# the probe -- roughly twice as long. The rig therefore buys a factor of ~1.8
# against the auxiliary echo for free, and every committed slab-family rung
# inherits it. That is a property of the geometry, not a margin anyone chose,
# and until #888 it was written down nowhere. ``aux_echo_arrival`` computes it
# so that a record law which ever grew past it fails instead of silently
# importing the echo into every number.
#
# WHAT IT DOES NOT DO. It bounds WHEN the echo arrives. It does not bound HOW
# LARGE the echo is: a deeper auxiliary absorber with sigma re-derived from a
# reflection target is the actual fix (#888 fix candidate 1, undecided), and
# this guard would pass a rig whose absorber was ten times worse.
# ---------------------------------------------------------------------------
AUX_N_CPML_1D = 20        # rfx/sources/tfsf.py: n_cpml_1d -- a hard-coded constant of the
                          # auxiliary grid, NOT scaled by dx_div (cv04 note section 6.1)
AUX_N_MARGIN_1D = 10      # rfx/sources/tfsf.py: n_margin
AUX_SRC_OFFSET_1D = 3     # rfx/sources/tfsf.py: src_idx = n_cpml_1d + 3 for direction "+x"
AUX_I0_1D = AUX_N_CPML_1D + AUX_N_MARGIN_1D   # tfsf.py: i0, the aux index mapping to 3-D x_lo
# Where inside the absorber the reflection is generated, in cells from the
# absorber's inner edge. MEASURED, not assumed: the two-mode fit
# B/A = rho e^{-2 j k L} has a phase slope d(arg B/A)/dk = -1.277755 m, i.e.
# a reflector at auxiliary index 638.88 with the hi CPML at 632..651
# (cv04 note section 2), reproduced at 1038.88 on the nx_interior = 1000
# geometry (section 9). The 2-D grid's counterpart is 8.0 cells inside its
# 30-cell layer (#888 note section 3); it is passed explicitly there.
AUX_REFLECTOR_DEPTH_CELLS = 6.88
# The invariant: a record is admissible only while it ENDS BEFORE the echo
# ARRIVES. Equality is already a failure -- the last recorded step would be the
# first contaminated one.
AUX_ECHO_RATIO_LIMIT = 1.0
AUX_ECHO_SCHEMA = "aux-echo-record-invariant/v1"


def aux_echo_arrival(*, n_aux: int, src_idx: int, aux_n_cpml: int,
                     reflector_depth_cells: float, probe_aux_index: int,
                     v_cells: float, lead_steps: float = 0.0) -> dict:
    """The step at which the auxiliary absorber's echo first reaches one probe.

    Pure geometry -- nothing here is measured on the run it guards, which is
    the whole point: a witness derived from the record it bounds cannot bound
    it. The echo is launched at ``src_idx``, reflects at
    ``(n_aux - aux_n_cpml) + reflector_depth_cells`` (the absorber's inner edge
    plus the measured reflecting depth) and travels back to
    ``probe_aux_index``, the auxiliary index whose sample IS the incident
    reference of the 3-D probe. ``v_cells`` is the propagation speed in cells
    per step along the path.

    ``lead_steps`` shifts the answer EARLIER, to the pulse's leading edge: the
    path arithmetic starts the clock at t = 0 while the injected waveform only
    peaks at t0, so the disturbance that arrives at ``path/v`` has a front
    ``t0/dt`` steps ahead of it. Subtracting it is what makes the number a
    bound rather than an estimate.

    Returns ``arrival_steps`` (the bound, floored to an integer step) and
    ``arrival_centre_steps`` (``path/v`` itself -- the quantity both #888 notes
    tabulate, kept so their tables are reproducible from the artifact).
    """
    reflector = float(n_aux - int(aux_n_cpml)) + float(reflector_depth_cells)
    path = (reflector - float(src_idx)) + (reflector - float(probe_aux_index))
    if path <= 0.0:
        raise ValueError(f"non-positive echo path {path} cells: the probe is behind the reflector")
    if not (float(v_cells) > 0.0):
        raise ValueError(f"v_cells must be positive, got {v_cells!r}")
    centre = path / float(v_cells)
    return {"reflector_index": reflector, "path_cells": path,
            "arrival_centre_steps": int(round(centre)),
            "arrival_steps": int(math.floor(centre - float(lead_steps)))}


def slab_aux_echo(nx_interior: int, dt: float, *, dx_div: int = 1,
                  n_steps: int | None = None) -> dict:
    """``aux_echo_arrival`` at the slab family's own rig (cv04, cv22, cv23).

    The auxiliary layout is ``rfx/sources/tfsf.py``'s:
    ``n_1d = 20 + 10 + (x_hi - x_lo + 2) + 10 + 20``, source at 23, ``i0`` at
    30 mapping to the 3-D ``x_lo``; its constants do NOT scale with ``dx_div``.

    The speed is ``v_cells = c dt/dx``, the Courant cell speed
    ``derive_record_length`` already uses. On the 1-D Yee lattice that is the
    SUPREMUM of the group velocity over the band (v_g -> c dt/dx as k -> 0 and
    falls monotonically with frequency), so no spectral component can arrive
    earlier than this says. ``echo_arrival_steps`` is the earlier of the two
    probes: the record is bounded by whichever is contaminated first.
    """
    K = int(dx_div)
    dx = DX_M / K
    cells = rig_cells(nx_interior, K)
    x_lo = cells["x_lo"]
    x_hi = cells["nx"] - x_lo - 1          # rfx/sources/tfsf.py: x_hi = nx - offset - 1
    n_1d = 2 * AUX_N_CPML_1D + 2 * AUX_N_MARGIN_1D + (x_hi - x_lo + 2)
    src_idx = AUX_N_CPML_1D + AUX_SRC_OFFSET_1D
    v_cells = C0 * float(dt) / dx
    tau = 1.0 / (math.pi * TFSF_F0_HZ * TFSF_BW)
    lead = SRC_T0_OVER_TAU * tau / float(dt)
    probes = {}
    for name, px in (("refl", cells["probe_refl"]), ("trans", cells["probe_trans"])):
        probes[name] = aux_echo_arrival(
            n_aux=n_1d, src_idx=src_idx, aux_n_cpml=AUX_N_CPML_1D,
            reflector_depth_cells=AUX_REFLECTOR_DEPTH_CELLS,
            probe_aux_index=AUX_I0_1D + (px - x_lo),
            v_cells=v_cells, lead_steps=lead)
    first = min(probes, key=lambda k: probes[k]["arrival_steps"])
    out = {
        "schema": AUX_ECHO_SCHEMA, "issue": 888,
        "nx_interior": int(nx_interior) * K, "dx_div": K,
        "aux_n_1d": int(n_1d), "aux_n_cpml": AUX_N_CPML_1D, "aux_src_idx": int(src_idx),
        "aux_reflector_depth_cells": AUX_REFLECTOR_DEPTH_CELLS,
        "aux_reflector_index": probes["trans"]["reflector_index"],
        "v_cells": float(v_cells), "pulse_lead_steps": float(lead),
        "echo_arrival_probe": first,
        "echo_arrival_steps": int(probes[first]["arrival_steps"]),
        "echo_arrival_centre_steps": int(probes[first]["arrival_centre_steps"]),
        "limit": AUX_ECHO_RATIO_LIMIT,
    }
    for name, pr in probes.items():
        out[f"path_cells_{name}"] = pr["path_cells"]
        out[f"arrival_steps_{name}"] = int(pr["arrival_steps"])
        out[f"arrival_centre_steps_{name}"] = int(pr["arrival_centre_steps"])
    if n_steps is not None:
        out.update(aux_echo_verdict(out, int(n_steps)))
    return out


def aux_echo_verdict(echo: dict, n_steps: int) -> dict:
    """``record_steps``, the ratio and the boolean, for a computed ``echo``."""
    ratio = float(n_steps) / float(echo["echo_arrival_steps"])
    return {"record_steps": int(n_steps),
            "record_over_echo_arrival": ratio,
            "ok": bool(ratio < AUX_ECHO_RATIO_LIMIT)}


def aux_echo_failure_message(echo: dict) -> str:
    """What a reader of a red gate needs: the mechanism, and where it is written."""
    return (
        f"AUXILIARY-ECHO RECORD INVARIANT (#888): the record is "
        f"{echo['record_steps']} steps against an auxiliary-absorber echo arrival of "
        f"{echo['echo_arrival_steps']} steps -- ratio "
        f"{echo['record_over_echo_arrival']:.3f} >= {AUX_ECHO_RATIO_LIMIT:.1f}. "
        f"The TF/SF auxiliary grid's own absorber reflects 4-6 % in amplitude "
        f"(|B/A| = 4.40e-02 on this 1-D path) from "
        f"{echo['aux_reflector_depth_cells']} cells inside its "
        f"{echo['aux_n_cpml']}-cell layer; that -x wave is injected into the "
        f"total-field region and, because R and T are normalised by the SAME "
        f"auxiliary field, it cancels in vacuum and is invisible to the leakage "
        f"and purity witnesses. It enters the measured R and T only once the "
        f"record reaches the probes, which this record does. cv26 above 34 deg is "
        f"exactly this failure. See issue #888 and "
        f"docs/design_notes/20260904_aux_echo_record_invariant.md.")


def staged_commit(repo_root: str, cwd: str | None = None) -> str:
    """The source commit: ``.staged_commit`` first (a staged copy on a pod has
    no .git; the orchestrator writes it at staging time -- cv22 review
    finding 6), then ``git rev-parse HEAD``, else "unknown"."""
    staged = os.path.join(repo_root, ".staged_commit")
    if os.path.isfile(staged):
        with open(staged) as fh:
            val = fh.read().strip()
        if val:
            return val
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd or repo_root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 2. The calibration envelope
# ---------------------------------------------------------------------------
CV04_ENVELOPE_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "_04_fresnel_results", "envelope.json"))
CV04_ENVELOPE_REL = "validation/crossval/_04_fresnel_results/envelope.json"

ENVELOPE_SCHEMA = "rfx.crossval_envelope/v1"
REVISION_STATUSES = ("active", "superseded", "scope-limited", "withdrawn")
_HASH_KEY = "revision_hash"


class EnvelopeIntegrityError(ValueError):
    """The artifact, or a consumer's adoption of it, does not hold together."""


def canonical_json(obj) -> str:
    """The one canonical form hashes are taken over."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of(obj) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def revision_hash(block: dict) -> str:
    """Hash of a revision block, excluding the recorded hash itself."""
    return sha256_of({k: v for k, v in block.items() if k != _HASH_KEY})


def _revision_order(name: str) -> tuple:
    """r1 < r2 < r10; anything unnumbered sorts last, by name."""
    if name.startswith("r") and name[1:].isdigit():
        return (0, int(name[1:]), name)
    return (1, 0, name)


def load_envelope(path: str | None = None) -> dict:
    """Read a producer envelope artifact and check it against itself."""
    path = path or CV04_ENVELOPE_PATH
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if doc.get("schema") != ENVELOPE_SCHEMA:
        raise EnvelopeIntegrityError(
            f"{path}: schema {doc.get('schema')!r}, expected {ENVELOPE_SCHEMA!r}")
    if not doc.get("revisions"):
        raise EnvelopeIntegrityError(f"{path}: no revisions")
    for name, block in doc["revisions"].items():
        if block.get("status") not in REVISION_STATUSES:
            raise EnvelopeIntegrityError(
                f"{path}: revision {name} has status {block.get('status')!r}, "
                f"not one of {REVISION_STATUSES}")
        recorded = block.get(_HASH_KEY)
        computed = revision_hash(block)
        if recorded != computed:
            raise EnvelopeIntegrityError(
                f"{path}: revision {name} was edited after it was written -- "
                f"recorded {recorded}, recomputed {computed}. A revision is "
                f"append-only evidence; correct it by appending a new one.")
    return doc


def load_adopted_envelope(adoption: dict, path: str | None = None) -> dict:
    """Resolve a consumer's ADOPTION RECORD against the producer's artifact.

    ``adoption`` must carry ``envelope``, ``adopted_revision``,
    ``revision_sha256``, ``rig_hash``, ``gate_policy`` and
    ``adopted_by_reviewer``, and every one of them is CHECKED here. Every
    mismatch is an error EXCEPT a newer revision existing, which is reported in
    the returned dict for the caller (and the contract test) to surface.

    ``rig_hash`` is RECOMPUTED over the revision's own rig block, not compared
    string-to-string: a review changed ``rig.dx_m`` to 0.123, left the recorded
    hash alone, and the first version of this loader accepted it -- a hash that
    is never recomputed says nothing about what it is supposed to cover.
    """
    doc_path = path or CV04_ENVELOPE_PATH
    doc = load_envelope(doc_path)
    name = adoption["adopted_revision"]
    revisions = doc["revisions"]
    if name not in revisions:
        raise EnvelopeIntegrityError(
            f"adopted revision {name!r} is not in {doc.get('producer')}'s "
            f"envelope (has {sorted(revisions)})")
    block = revisions[name]
    if block["status"] != "active":
        raise EnvelopeIntegrityError(
            f"revision {name} is {block['status']!r}; only an active revision "
            f"may be adopted")
    if adoption["revision_sha256"] != block[_HASH_KEY]:
        raise EnvelopeIntegrityError(
            f"the adoption record pins revision {name} at "
            f"{adoption['revision_sha256']}, the artifact holds "
            f"{block[_HASH_KEY]}. The adopted evidence changed under the "
            f"consumer; re-adopt deliberately or restore the revision.")
    recomputed_rig = sha256_of(block["rig"])
    if recomputed_rig != block["rig_hash"]:
        raise EnvelopeIntegrityError(
            f"revision {name}'s rig block hashes to {recomputed_rig}, but the "
            f"artifact records {block['rig_hash']}: the realized configuration "
            f"was edited under its own hash.")
    if adoption["rig_hash"] != recomputed_rig:
        raise EnvelopeIntegrityError(
            f"the adoption record pins the realized rig at "
            f"{adoption['rig_hash']}, revision {name} realized "
            f"{recomputed_rig}")
    policy = adoption.get("gate_policy")
    if not isinstance(policy, dict):
        raise EnvelopeIntegrityError(
            "the adoption record carries no gate_policy; a calibration is the "
            "revision AND the policy it was adopted under")
    for key in ("multiplier", "quantum"):
        value = policy.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise EnvelopeIntegrityError(
                f"gate_policy[{key!r}] is {value!r}; a positive number is required")
    declared_path = str(adoption.get("envelope") or "")
    if not declared_path or not str(doc_path).replace(os.sep, "/").endswith(
            declared_path.lstrip("./")):
        raise EnvelopeIntegrityError(
            f"the adoption record names envelope {declared_path!r}, but the "
            f"artifact read was {doc_path}: a consumer must say which producer "
            f"artifact it adopts, and be right about it")
    reviewer = str(adoption.get("adopted_by_reviewer") or "").strip()
    producer_author = str(block["run_provenance"].get("producer_run_author") or "").strip()
    if not reviewer:
        raise EnvelopeIntegrityError(
            "the adoption record names no reviewer; adoption is a reviewed "
            "declaration, not a consequence of a run")
    if reviewer == producer_author:
        raise EnvelopeIntegrityError(
            f"the adopting reviewer ({reviewer!r}) is the producer run's own "
            f"author; a run cannot adopt itself into the gates that judge it")
    order = sorted(revisions, key=_revision_order)
    newer = order[order.index(name) + 1:]
    return {
        "values": dict(block["values"]),
        "revision": name,
        "status": block["status"],
        "rig_hash": block["rig_hash"],
        "revision_sha256": block[_HASH_KEY],
        "witness_status": block.get("witness_status"),
        "latest_revision": order[-1],
        "newer_revisions": list(newer),
        "producer": doc["producer"],
        "path": doc_path,
    }
