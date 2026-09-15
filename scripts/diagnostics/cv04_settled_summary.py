"""Pull a small, machine-readable summary out of a cv04 VESSL run's own
lattice_witness.json + stdout log, for a later (separate, reviewed) step to
hand-construct envelope.json's r2 revision from and to re-derive the ~23
stale prose citations that cite cv04's old 719-step numbers.

This script does NOT write envelope.json itself -- that revision is a
deliberate, documented, PR-reviewed act per that artifact's own `notes`
field ("It is never a consequence of re-running the producer"). This just
surfaces the numbers so that step doesn't require re-grepping a VESSL log
by hand.

Usage:
    python scripts/diagnostics/cv04_settled_summary.py \
        <path-to-lattice_witness.json> <path-to-cv04-stdout-log>
"""
import json
import re
import sys


def _grab(log: str, pattern: str) -> float | None:
    m = re.search(pattern, log)
    return float(m.group(1)) if m else None


def main() -> int:
    witness_path, log_path = sys.argv[1], sys.argv[2]
    doc = json.load(open(witness_path))
    r = doc["rungs"]["slab_eps4"]
    log = open(log_path).read()

    # nx_interior and the per-window tail levels are not stored in the
    # witness rung's own shape (LW.evaluate()'s return value drops the raw
    # run/tail sub-dicts it was built from) -- read them from this script's
    # own "SETTLED at nx_interior=..." print line instead.
    settled_line = re.search(
        r"SETTLED at nx_interior=(\d+).*?n_steps=(\d+).*?"
        r"tail scat_refl/trans = ([0-9.eE+-]+)/([0-9.eE+-]+)",
        log,
    )
    summary = {
        "gated_here": doc.get("gated_here"),
        "n_steps": r["n_steps"],
        "nx_interior": int(settled_line.group(1)) if settled_line else None,
        "tail_refl_rel": float(settled_line.group(3)) if settled_line else None,
        "tail_trans_rel": float(settled_line.group(4)) if settled_line else None,
        "witness_ok": r.get("witness_ok"),
        "mean_dR_lattice_gated": r.get("mean_dR_lattice_gated"),
        "mean_dT_lattice_gated": r.get("mean_dT_lattice_gated"),
        "mean_W_witness_R_gated": r.get("mean_W_witness_R_gated"),
        "mean_W_witness_T_gated": r.get("mean_W_witness_T_gated"),
        # For envelope.json r2 (hand-constructed separately -- see docstring):
        "T_mean_err": _grab(log, r"T\(f\) mean err(?:or)?:\s*([0-9.eE+-]+)"),
        "R_mean_err": _grab(log, r"R\(f\) mean err(?:or)?:\s*([0-9.eE+-]+)"),
        "RT_mean_dev_energy": _grab(log, r"R\+T mean dev \(energy\)\s+([0-9.eE+-]+)"),
        "max_RT_minus_1_per_bin": _grab(log, r"max\|R\+T-1\| \(per-bin\)\s+([0-9.eE+-]+)"),
    }
    json.dump(summary, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
