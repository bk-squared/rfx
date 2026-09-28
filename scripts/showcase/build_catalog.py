#!/usr/bin/env python3
"""Write ``docs/public/showcase/showcase.json`` (schema ``rfx-showcase-catalog/1``).

The catalog is what an assistant reads to find each showcase case: its page, the
question it answers, every file on the share host with its URL, size and
SHA-256, the rfx commit that produced each record, and the numbers the pages
print, each with its unit, its full-precision value and where the value comes
from.  Nothing in it is typed by hand:

* file names, sizes, hashes and commits are copied from the share host's
  ``manifest.json``;
* titles and questions are the pages' own front matter;
* each printed number is computed here from the share host's files (the
  ``result.json`` records, and for the two forward comparisons the curves,
  through the same estimators the cross-validation tests judge with), then
  formatted.  A value that no longer rounds to what the page prints makes the
  page and the catalog disagree, which ``tests/unit/docs/test_showcase_catalog.py``
  reports.

Every file this script reads from the share host is checked against the
manifest's SHA-256 before it is used.

    python scripts/showcase/build_catalog.py                  # fetch from the share host
    python scripts/showcase/build_catalog.py --share-dir DIR  # a verified local copy
    python scripts/showcase/build_catalog.py --check          # compare, do not write

The forward comparisons import the cross-validation test modules (and so rfx);
run it from a checkout with the package importable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "docs/public/showcase/showcase.json"
SHARE = "https://remilab.cnu.ac.kr/share/7c02ad43c580/"
SCHEMA = "rfx-showcase-catalog/1"
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

HOME, INDEX = "/rfx/", "/rfx/showcase/"
MAP, COST = "/rfx/showcase/sensitivity-map/", "/rfx/showcase/gradient-cost/"
DESCENT, FORWARD = "/rfx/showcase/descent-to-optimum/", "/rfx/showcase/forward-checks/"

#: Showcase pages: case id -> (page source, share folders the page draws on).
CASES = {
    "sensitivity-map": ("docs/public/showcase/sensitivity-map.mdx", ("patch-sensitivity",)),
    "gradient-cost": ("docs/public/showcase/gradient-cost.mdx", ("patch-sensitivity",)),
    "descent-to-optimum": ("docs/public/showcase/descent-to-optimum.mdx", ("ar-coating",)),
    "forward-checks": ("docs/public/showcase/forward-checks.mdx", ("forward-patch", "forward-notch")),
}
BOXES = ("patch", "design", "layer")


class Share:
    """The share host's files, each checked against the manifest before use."""

    def __init__(self, local: Path | None):
        self.local = local
        raw = self._bytes("manifest.json")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        self._cache: dict[str, bytes] = {}

    def _bytes(self, name: str) -> bytes:
        if self.local is not None:
            return (self.local / name).read_bytes()
        with urllib.request.urlopen(SHARE + name, timeout=60) as response:
            return response.read()

    def read(self, folder: str, name: str) -> bytes:
        key = f"{folder}/{name}"
        if key not in self._cache:
            record = self.manifest["cases"][folder]["files"][name]
            data = self._bytes(key)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise SystemExit(f"{SHARE}{key}: size or SHA-256 differs from manifest.json")
            self._cache[key] = data
        return self._cache[key]

    def json(self, folder: str, name: str) -> dict:
        return json.loads(self.read(folder, name))

    def npz(self, folder: str, name: str):
        return np.load(io.BytesIO(self.read(folder, name)), allow_pickle=False)


def _claim(record: dict, quantity: str) -> float:
    found = [c["value"] for c in record["claims"] if c["quantity"] == quantity]
    if len(found) != 1:
        raise SystemExit(f"{record['id']}: expected one claim {quantity!r}, found {len(found)}")
    return found[0]


def _derived(record: dict, quantity: str) -> float:
    found = [d["value"] for d in record["derived"] if d["quantity"] == quantity]
    if len(found) != 1:
        raise SystemExit(f"{record['id']}: expected one derived {quantity!r}, found {len(found)}")
    return found[0]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ formats
def comma(value: float) -> str:
    return f"{round(value):,}"


def fixed(digits: int):
    return lambda value: f"{value:.{digits}f}"


def sci(value: float) -> str:
    """One significant digit, as the pages print it: 2×10⁻⁵."""
    exponent = math.floor(math.log10(abs(value)))
    mantissa = round(value / 10 ** exponent)
    if mantissa == 10:
        mantissa, exponent = 1, exponent + 1
    return f"{mantissa}×10{str(exponent).translate(SUPERSCRIPT)}"


def _printed(value, fmt) -> str:
    if isinstance(value, list):
        texts = {fmt(v) for v in value}
        if len(texts) != 1:
            raise SystemExit(f"a range prints as {sorted(texts)}; the pages print one number")
        return texts.pop()
    return fmt(value)


# ------------------------------------------------------------------ numbers
def headlines(share: Share) -> dict[str, list[dict]]:
    """Every number the pages print in a tile, a card, the proof strip or the
    cost table, by case."""
    patch = share.json("patch-sensitivity", "result.json")
    ar = share.json("ar-coating", "result.json")
    gates = share.json("ar-coating", "gates.json")
    tmm = share.json("ar-coating", "tmm.json")
    iterations = share.npz("ar-coating", "iterations.npz")
    src = "patch-sensitivity/result.json"
    boxes = patch["model"]["boxes"]
    fd64 = [c["value"] for c in patch["claims"]
            if c["quantity"].startswith("|AD - FD| / |FD|") and c["quantity"].endswith("(float64)")]
    grad = [_claim(patch, f"value_and_grad wall time, box {b}") for b in BOXES]
    ratio = [_derived(patch, f"value_and_grad / forward wall-time ratio, box {b}") for b in BOXES]
    fd_days = {b: _derived(patch, f"central finite-difference cost for every cell of box {b}") / 86400
               for b in BOXES}

    def h(quantity, value, unit, fmt, routes, source):
        return {"quantity": quantity, "printed": _printed(value, fmt), "value": value,
                "unit": unit, "routes": list(routes), "from": source}

    out: dict[str, list[dict]] = {}
    out["sensitivity-map"] = [
        h("design-box cells", boxes["design"]["n_cells"], "cells", comma, (MAP, HOME, INDEX),
          f"{src} model.boxes.design.n_cells"),
        h("largest |AD - FD| / |FD| over the six blocks, float64", max(fd64), "1", sci,
          (MAP, HOME), f"{src} claims '|AD - FD| / |FD| ... (float64)', maximum of {len(fd64)}"),
        h("value and gradient wall time, design box", grad[1], "s", fixed(0), (MAP,),
          f"{src} claim 'value_and_grad wall time, box design'"),
        h("value and gradient / forward wall time, design box", ratio[1], "1", fixed(1), (MAP,),
          f"{src} derived 'value_and_grad / forward wall-time ratio, box design'"),
    ]
    cost = [
        h("value and gradient / forward wall time, three boxes", [min(ratio), max(ratio)], "1",
          fixed(1), (HOME,), f"{src} derived 'value_and_grad / forward wall-time ratio', min and max"),
        h("value and gradient wall time, three boxes", [min(grad), max(grad)], "s", fixed(0),
          (HOME, INDEX), f"{src} claims 'value_and_grad wall time', min and max"),
        h("layer box: cells that are not laminate", boxes["layer"]["n_other_cells"], "cells", comma,
          (COST,), f"{src} model.boxes.layer.n_other_cells"),
    ]
    for b in BOXES:
        # The home page's proof strip and the two cards quote the smallest and
        # the largest box: "12,400 to 43,008 design cells", "4.9 to 16.8 days".
        ends = b != "design"
        cost += [
            h(f"{b} box cells", boxes[b]["n_cells"], "cells", comma,
              (COST, HOME) if ends else (COST,), f"{src} model.boxes.{b}.n_cells"),
            h(f"forward wall time, {b} box",
              _claim(patch, f"forward wall time, box {b} ({boxes[b]['n_cells']} cells)"),
              "s", fixed(2), (COST,), f"{src} claim 'forward wall time, box {b}'"),
            h(f"value and gradient wall time, {b} box", grad[BOXES.index(b)], "s", fixed(2), (COST,),
              f"{src} claim 'value_and_grad wall time, box {b}'"),
            h(f"value and gradient peak device memory, {b} box",
              _claim(patch, f"value_and_grad peak device memory, box {b}") / 1e9, "GB", fixed(2),
              (COST,), f"{src} claim 'value_and_grad peak device memory, box {b}' / 1e9"),
            h(f"central differences for every cell, {b} box (derived)", fd_days[b], "days", fixed(1),
              (COST, HOME, INDEX) if ends else (COST,),
              f"{src} derived 'central finite-difference cost for every cell of box {b}' / 86400"),
        ]
    out["gradient-cost"] = cost

    final, optimum = gates["final_eps_r"], tmm["optimum_eps_r"]
    out["descent-to-optimum"] = [
        h("largest layer-permittivity gap to the transfer-matrix optimum",
          100 * max(abs(f / o - 1) for f, o in zip(final, optimum)), "%", fixed(1),
          (DESCENT, HOME, INDEX), "ar-coating/gates.json final_eps_r vs tmm.json optimum_eps_r"),
        h("transfer-matrix band-mean |R|^2 of the final design above the optimum",
          100 * (float(iterations["tmm_cost51"][-1]) / _claim(ar, "TMM optimum X-band mean |R|^2") - 1),
          "%", fixed(1), (DESCENT,),
          "ar-coating/iterations.npz tmm_cost51[-1] vs result.json claim 'TMM optimum X-band mean |R|^2'"),
        h("Adam steps", ar["model"]["adam"]["iters"], "steps", comma, (DESCENT,),
          "ar-coating/result.json model.adam.iters"),
    ]
    out["forward-checks"] = forward_headlines(share, h)
    return out


def forward_headlines(share: Share, h) -> list[dict]:
    """The two judged forward numbers, recomputed from the share host's curves
    with the estimators the cross-validation tests use."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    judging = _load(REPO / "tests/crossval/_v2_judging.py", "_showcase_v2_judging")
    patch_test = _load(REPO / "tests/crossval/rt5880_patch/test_rt5880_patch.py", "_showcase_patch")
    notch_test = _load(REPO / "tests/crossval/msl_notch_filter/test_msl_notch_filter.py",
                       "_showcase_notch")
    ours, ref = share.npz("forward-patch", "rfx_curves.npz"), share.npz("forward-patch", "reference_curves.npz")
    stage = patch_test.OPENEMS_JUDGED_STAGE
    band = patch_test.RESONANCE_BAND_HZ
    f_ours = judging.refined_remax(ours["freqs_hz"], ours["zin_ohm"], *band)["f0_hz"]
    f_ref = judging.refined_remax(ref[f"openems_{stage}_freqs_hz"],
                                  ref[f"openems_{stage}_zin_re_ohm"]
                                  + 1j * ref[f"openems_{stage}_zin_im_ohm"], *band)["f0_hz"]
    ours, ref = share.npz("forward-notch", "rfx_curves.npz"), share.npz("forward-notch", "reference_curves.npz")
    stage = notch_test.OPENEMS_JUDGED_STAGE
    notch = notch_test._compare("judged", {"freqs_hz": ours["freqs_hz"], "s21": ours["s21"]},
                                ref[f"openems_{stage}_freqs_hz"] / 1e9, ref[f"openems_{stage}_s21_mag"])
    return [
        h("|f0 rfx - f0 openEMS| / f0 openEMS, patch Re(Z_in) peak", 100 * abs(f_ours / f_ref - 1),
          "%", fixed(3), (HOME, INDEX),
          "forward-patch rfx_curves.npz and reference_curves.npz (openEMS stage_b_fine), refined_remax "
          "over the test's RESONANCE_BAND_HZ (tests/crossval/rt5880_patch)"),
        h("|f rfx - f openEMS| / f openEMS, notch", notch["notch_pct"], "%", fixed(3), (HOME, INDEX),
          "forward-notch rfx_curves.npz and reference_curves.npz (openEMS stage_b_fine), the test's "
          "_compare (tests/crossval/msl_notch_filter)"),
    ]


# ------------------------------------------------------------------ catalog
def front_matter(path: Path) -> dict:
    text = path.read_text()
    if not text.startswith("---\n"):
        raise SystemExit(f"{path}: no front matter")
    return yaml.safe_load(text[4:].split("\n---", 1)[0])


def build(share: Share) -> dict:
    numbers = headlines(share)
    cases = []
    for case, (page, folders) in CASES.items():
        meta = front_matter(REPO / page)
        sources = []
        for folder in folders:
            entry = share.manifest["cases"][folder]
            base = f"{SHARE}{folder}/"
            record = share.json(folder, "result.json")
            sources.append({
                "folder": folder, "base_url": base, "rfx_commit": entry["rfx_commit"],
                "question": record["question"], "result_json": base + "result.json",
                "files": [{"name": name, "url": base + name, "bytes": rec["bytes"],
                           "sha256": rec["sha256"]} for name, rec in sorted(entry["files"].items())],
            })
        cases.append({
            "id": case, "title": meta["title"], "question": meta["description"],
            "route": f"/rfx/showcase/{case}/", "markdown": f"/rfx/markdown/showcase/{case}.md",
            "rfx_commits": sorted({s["rfx_commit"] for s in sources}),
            "sources": sources, "headlines": numbers[case],
        })
    return {"schema": SCHEMA, "generator": "scripts/showcase/build_catalog.py",
            "share": {"base_url": SHARE, "manifest_url": SHARE + "manifest.json",
                      "manifest_sha256": share.manifest_sha256,
                      "created": share.manifest["created"], "cite": share.manifest["cite"],
                      **({"supersedes": share.manifest["supersedes"]} if "supersedes" in share.manifest else {})},
            "cases": cases}


def _flat(value) -> bool:
    return not isinstance(value, (dict, list)) or (
        isinstance(value, list) and not any(isinstance(v, (dict, list)) for v in value))


def dumps(value, indent: int = 0, in_list: bool = False) -> str:
    """JSON with one line per file record and per number, so the diff of a
    re-run is readable."""
    pad = "  " * indent
    if _flat(value) or (in_list and isinstance(value, dict) and all(map(_flat, value.values()))):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        items = [f"{pad}  {json.dumps(k)}: {dumps(v, indent + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    items = [f"{pad}  {dumps(v, indent + 1, in_list=True)}" for v in value]
    return "[\n" + ",\n".join(items) + f"\n{pad}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--share-dir", type=Path, help="a local copy of the share folder")
    parser.add_argument("--check", action="store_true", help="compare with the committed catalog")
    args = parser.parse_args()
    text = dumps(build(Share(args.share_dir))) + "\n"
    if args.check:
        if not CATALOG.is_file() or CATALOG.read_text() != text:
            print(f"{CATALOG.relative_to(REPO)} differs; regenerate with scripts/showcase/build_catalog.py")
            return 1
        print(f"{CATALOG.relative_to(REPO)} in sync")
        return 0
    CATALOG.write_text(text)
    print(f"wrote {CATALOG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
