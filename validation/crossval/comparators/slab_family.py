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
import sys

import numpy as np

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

    ``adoption`` must carry ``adopted_revision``, ``revision_sha256``,
    ``rig_hash``, ``gate_policy`` and ``adopted_by_reviewer``. Every mismatch
    is an error EXCEPT a newer revision existing, which is reported in the
    returned dict for the caller (and the contract test) to surface.
    """
    doc = load_envelope(path)
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
    if adoption["rig_hash"] != block["rig_hash"]:
        raise EnvelopeIntegrityError(
            f"the adoption record pins the realized rig at "
            f"{adoption['rig_hash']}, revision {name} realized "
            f"{block['rig_hash']}")
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
        "path": path or CV04_ENVELOPE_PATH,
    }
