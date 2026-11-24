#!/usr/bin/env python3
"""Write ``docs/public/showcase/showcase.json`` (schema ``rfx-showcase-catalog/1``).

The catalog is what an assistant reads to find each showcase case: its page, the
question it answers, every file on the share host with its URL, size and
SHA-256, the rfx commit that produced each record, and the numbers the pages
print, each with its unit, its full-precision value and where the value comes
from.  Nothing in it is typed by hand:

* file names, sizes, hashes and commits are copied from the share host's
  ``manifest.json``;
* titles and case questions are the pages' own front matter; source questions
  come from result records (falling back to their example name);
* each printed number is computed here from the share host's files (the
  ``result.json`` records, ``lens-page-values.json`` for the lens, and for the
  two forward comparisons the curves,
  through the same estimators the cross-validation tests judge with), then
  formatted.  A value that no longer rounds to what the page prints makes the
  page and the catalog disagree; no test compares the two, so compare each
  ``printed`` string with its page after a regeneration.

Every file this script reads from the share host is checked against the
manifest's SHA-256 before it is used.

    python scripts/showcase/build_catalog.py                  # fetch from the share host
    python scripts/showcase/build_catalog.py --share-dir DIR  # a verified local October copy
    python scripts/showcase/build_catalog.py --lens-share-dir DIR  # a verified local lens copy
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
OCTOBER_SHARE = "https://remilab.cnu.ac.kr/share/f1abd0bff18d/"
LENS_SHARE = "https://remilab.cnu.ac.kr/share/cd6e343367e3/"
FOLDER_SHARES = {
    "patch-calibration": OCTOBER_SHARE,
    "waveguide-filter": OCTOBER_SHARE,
    "grin-lens": LENS_SHARE,
    "grin-lens-fan": LENS_SHARE,
}
SCHEMA = "rfx-showcase-catalog/1"
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

HOME, INDEX = "/rfx/", "/rfx/showcase/"

#: Showcase pages in presentation order: case id -> (page source, share folders).
CASES = {
    "patch-calibration": ("docs/public/showcase/patch-calibration.mdx", ("patch-calibration",)),
    "waveguide-filter": ("docs/public/showcase/waveguide-filter.mdx", ("waveguide-filter",)),
    "grin-lens": ("docs/public/showcase/grin-lens.mdx", ("grin-lens", "grin-lens-fan")),
}


class Share:
    """The share host's files, each checked against the manifest before use."""

    def __init__(self, base_url: str, local: Path | None):
        self.base_url = base_url
        self.local = local
        raw = self._bytes("manifest.json")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        self._cache: dict[str, bytes] = {}

    def _bytes(self, name: str) -> bytes:
        if self.local is not None:
            return (self.local / name).read_bytes()
        with urllib.request.urlopen(self.base_url + name, timeout=60) as response:
            return response.read()

    def read(self, folder: str, name: str) -> bytes:
        key = f"{folder}/{name}"
        if key not in self._cache:
            record = self.manifest["cases"][folder]["files"][name]
            data = self._bytes(key)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise SystemExit(f"{self.base_url}{key}: size or SHA-256 differs from manifest.json")
            self._cache[key] = data
        return self._cache[key]

    def json(self, folder: str, name: str) -> dict:
        return json.loads(self.read(folder, name))

    def npz(self, folder: str, name: str):
        return np.load(io.BytesIO(self.read(folder, name)), allow_pickle=False)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ formats
def fixed(digits: int):
    return lambda value: f"{value:.{digits}f}"


def sci(value: float) -> str:
    """One significant digit, as the pages print it: 2×10⁻⁵."""
    exponent = math.floor(math.log10(abs(value)))
    mantissa = round(value / 10 ** exponent)
    if mantissa == 10:
        mantissa, exponent = 1, exponent + 1
    return f"{mantissa}×10{str(exponent).translate(SUPERSCRIPT)}"


def sci_digits(n: int):
    """Scientific notation with n significant digits, including trailing zeros."""
    if n < 1:
        raise ValueError("significant digits must be positive")

    def format_value(value: float) -> str:
        mantissa, exponent = f"{value:.{n - 1}e}".split("e")
        return f"{mantissa}×10{str(int(exponent)).translate(SUPERSCRIPT)}"

    return format_value


def _printed(value, fmt) -> str:
    if isinstance(value, list):
        texts = {fmt(v) for v in value}
        if len(texts) != 1:
            raise SystemExit(f"a range prints as {sorted(texts)}; the pages print one number")
        return texts.pop()
    return fmt(value)


# ------------------------------------------------------------------ numbers
def headlines(share: Share, lens_share: Share) -> dict[str, list[dict]]:
    """Calibration, lens and filter numbers printed on the public pages."""
    def h(quantity, value, unit, fmt, routes, source, range_ok=False, separator="–"):
        if range_ok:
            low, high = map(fmt, value)
            printed = low if low == high else f"{low}{separator}{high}"
        else:
            printed = _printed(value, fmt)
        return {"quantity": quantity, "printed": printed, "value": value,
                "unit": unit, "routes": list(routes), "from": source}

    patch = share.json("patch-calibration", "result.json")
    src = "patch-calibration/result.json "
    route = "/rfx/showcase/patch-calibration/"
    out: dict[str, list[dict]] = {}
    out["patch-calibration"] = [
        h("hidden permittivity", patch["hidden"]["eps_r"], "1", fixed(2), (route,),
          src + "hidden.eps_r"),
        h("hidden loss", patch["hidden"]["sigma_s_per_m"], "S/m", sci_digits(2), (route,),
          src + "hidden.sigma_s_per_m"),
        h("start permittivity (datasheet)", patch["start"]["eps_r"], "1", fixed(2), (route,),
          src + "start.eps_r"),
        h("recovered permittivity", patch["recovered"]["eps_r"], "1", fixed(3), (route,),
          src + "recovered.eps_r"),
        h("recovered loss", patch["recovered"]["sigma_s_per_m"], "S/m", sci_digits(3), (route,),
          src + "recovered.sigma_s_per_m"),
        h("updates", patch["iterates"], "", fixed(0), (route,), src + "iterates"),
        h("frequency bins", patch["bins"], "", fixed(0), (route,), src + "bins"),
        h("band", [v / 1e9 for v in patch["band_hz"]], "GHz", fixed(2), (route,),
          src + "band_hz / 1e9", range_ok=True),
    ]

    lens = lens_share.json("grin-lens", "lens-page-values.json")
    designed = lens["boresight"]["reported"]["0.0005"]["design_bins_dbi"]
    feed = lens["boresight"]["no_lens"]["0.0005"]["design_bins_dbi"]
    if len(designed) != 3 or len(feed) != 3:
        raise SystemExit("grin-lens/lens-page-values.json: expected three design bins at 0.5 mm")
    differences = [d - f for d, f in zip(designed, feed)]
    ends = [lens["lens_minus_feed_db"][key] for key in ("min_9p5_10p5", "max_9p5_10p5")]
    if not all(math.isclose(stored, computed, rel_tol=0, abs_tol=1e-9)
               for stored, computed in zip(ends, (min(differences), max(differences)))):
        raise SystemExit("grin-lens/lens-page-values.json: lens_minus_feed_db ends differ "
                         "from min/max of reported minus no_lens design bins at 0.5 mm")
    route = "/rfx/showcase/grin-lens/"
    src = "grin-lens/lens-page-values.json "
    out["grin-lens"] = [
        h("boresight directivity at 10 GHz with the designed lens, 0.5 mm mesh", designed[1], "dBi",
          fixed(1), (route, INDEX, HOME), src + 'boresight.reported["0.0005"].design_bins_dbi[1]'),
        h("feed boresight directivity at 10 GHz, 0.5 mm mesh", feed[1], "dBi",
          fixed(1), (route, HOME), src + 'boresight.no_lens["0.0005"].design_bins_dbi[1]'),
        h("designed minus feed directivity at 9.5, 10 and 10.5 GHz, 0.5 mm mesh", ends,
          "dB", lambda v: f"{v:+.1f}", (route,),
          src + "[lens_minus_feed_db.min_9p5_10p5, lens_minus_feed_db.max_9p5_10p5]",
          range_ok=True, separator=" to "),
    ]
    trend = share.json("waveguide-filter", "mesh_trend.json")
    route = "/rfx/showcase/waveguide-filter/"
    src = "waveguide-filter/mesh_trend.json "
    def minus(v):
        return f"{v:.1f}".replace("-", "−")

    out["waveguide-filter"] = [
        h("meshes with the response inside the mask", sum(trend["mask_each_mesh"]), "",
          lambda v: f"{v:d} of {len(trend['mask_each_mesh'])}", (route, INDEX),
          src + "sum(mask_each_mesh)"),
        h("worst |S11| over 10.0–10.6 GHz, 0.318 mm mesh", trend["worst_passband_S11_db"][-1],
          "dB", minus, (route,), src + "worst_passband_S11_db[-1]"),
        h("worst |S21| in the stopbands, 0.318 mm mesh", trend["worst_stopband_S21_db"][-1],
          "dB", minus, (route,), src + "worst_stopband_S21_db[-1]"),
    ]
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


def build(shares: dict[str, Share]) -> dict:
    numbers = headlines(shares[OCTOBER_SHARE], shares[LENS_SHARE])
    cases = []
    for case, (page, folders) in CASES.items():
        meta = front_matter(REPO / page)
        sources = []
        for folder in folders:
            share = shares[FOLDER_SHARES[folder]]
            entry = share.manifest["cases"][folder]
            base = f"{share.base_url}{folder}/"
            record = share.json(folder, "result.json")
            if "question" in record:
                question = record["question"]
            elif "example" in record:
                question = record["example"]
            else:
                raise SystemExit(f"{folder}/result.json: expected question or example")
            sources.append({
                "folder": folder, "base_url": base, "rfx_commit": entry["rfx_commit"],
                "manifest_url": share.base_url + "manifest.json",
                "question": question, "result_json": base + "result.json",
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
            "shares": [{"base_url": share.base_url, "manifest_url": share.base_url + "manifest.json",
                      "manifest_sha256": share.manifest_sha256,
                      "created": share.manifest["created"], "cite": share.manifest["cite"],
                      **({"supersedes": share.manifest["supersedes"]} if "supersedes" in share.manifest else {})}
                       for share in shares.values()],
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
    parser.add_argument("--share-dir", type=Path, help="a local copy of the October share folder")
    parser.add_argument("--lens-share-dir", type=Path, help="a local copy of the new lens share folder")
    parser.add_argument("--check", action="store_true", help="compare with the committed catalog")
    args = parser.parse_args()
    shares = {url: Share(url, local) for url, local in (
        (OCTOBER_SHARE, args.share_dir), (LENS_SHARE, args.lens_share_dir))}
    text = dumps(build(shares)) + "\n"
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
