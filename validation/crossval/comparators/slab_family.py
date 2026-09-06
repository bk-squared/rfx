"""Slab-family leaf module: the shared rig declaration and the calibration
envelope loader (issue #928).

Two jobs, both deliberately dependency-free (stdlib only, no numpy, no rfx, no
sibling comparator) so that both the PRODUCER (``04_multilayer_fresnel.py``)
and the CONSUMERS (``cv22_dispersive_gates.py``, ``cv23_lossy_gates.py``,
``slab_rig.py``) can import it without anything importing a consumer:

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
import os

# ---------------------------------------------------------------------------
# The calibration envelope (the rig block lands here in the next commit)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
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
