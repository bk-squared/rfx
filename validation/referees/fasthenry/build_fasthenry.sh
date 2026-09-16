#!/bin/sh
# Build the FastHenry magnetoquasistatic referee from a PINNED upstream source.
#
# WHAT THIS IS.  ``validation/fdfd/fasthenry_referee.py`` needs an independent
# magnetoquasistatic solver to judge the FDFD spiral extraction and the
# Greenhouse (Hoer-Love partial-inductance) referee against each other.  The
# solver is FastHenry, the MIT PEEC/multipole inductance extractor, in the
# maintained "wr" branch (Whiteley Research, shipped inside XicTools as a
# self-contained tarball).  NOTHING is vendored into this repository: this
# script fetches the pinned tarball, verifies its SHA-256, patches ONE printf
# (below), builds it with clang/make and leaves the binary in the cache.
#
# SOURCE (pinned)
#   repo    https://github.com/wrcad/xictools
#   commit  d28932afd22acf116c8bcd22e56a57faf3fb6d07   (2024-11-23, master)
#   path    fasthenry/fasthenry-3.0wr-031424.tar.gz
#   sha256  6da40d0e31425bca85be46434b33ecc194205d705b47f4459d91568c9f4301ef
#   version FastHenry 3.0wr (29Sep96, mod 031424), tarball dir fasthenry-3.0wr
#
# LICENCE.  FastHenry is Copyright (C) 2003 by the Board of Trustees of
# Massachusetts Institute of Technology, released under the permissive MIT
# FastHenry licence ("License to use, copy, modify, sell and/or distribute this
# software and its documentation for any purpose is hereby granted without
# royalty, subject to ... 1. the copyright notice must appear in all copies ...
# 2. the names of the Copyright Owners may not be used in advertising ...
# 3. the software is provided AS-IS"), carried verbatim at the top of every
# source file (e.g. src/fasthenry/induct.c).  The "wr" modifications
# (superconductivity, 64-bit clean-ups, KLU/DSS solver interfaces) are by
# Stephen R. Whiteley, Whiteley Research Inc., distributed under the same
# licence text.  Redistribution is permitted; this script nonetheless
# redistributes nothing, it downloads from the pinned commit at build time.
#
# THE ONE PATCH.  src/fasthenry/induct.c prints the impedance matrix with
# "%13.6lg %+13.6lgj" -- SIX significant digits.  Every quantity this referee
# reports is a DIFFERENCE of impedances (L = Im(Z11 - Z12)/omega for the image
# construction; L(strip) - L(bridge) for the de-embedding, a 4.7 % difference),
# so six digits would put a 1e-5..1e-4 relative floor under numbers that are
# gated at 1e-3.  The patch widens that one format to "%25.17le" in
# cx_dumpMat_totextfile() and changes nothing else -- no physics, no algorithm,
# no default.  It is applied with sed and its effect is asserted after the
# build (a single bar must reproduce the Hoer-Love self-inductance to better
# than 1e-9, which six digits cannot).
#
# USAGE
#   sh validation/referees/fasthenry/build_fasthenry.sh [--force]
#   RFX_REFEREE_CACHE=/somewhere sh .../build_fasthenry.sh
# leaves
#   $CACHE/fasthenry/bin/fasthenry          the binary
#   $CACHE/fasthenry/manifest.json          commit, sha256, patch, build record
# and prints the binary path on the last line.  Re-running is a no-op unless
# --force is given, the manifest does not match the pin, or the manifest
# predates the "binary_sha256" field.
#
# THE BINARY'S OWN SHA-256 is recorded in the manifest, and
# ``fasthenry_referee.py`` puts it in the key of every cached FastHenry run:
# a run recorded by a different (or unpatched) build is therefore never
# replayed as if this build had produced it, and the study's "live" gates
# cannot pass from a stale cache.  The install is a rename, so a concurrent
# study run never execs a half-written binary.  Keying on it costs nothing on a
# rebuild: a clean build here (fetch, verify, patch, make, self-test: 3.8 s)
# reproduces the binary BIT FOR BIT, SHA-256
# 9cd3124570c31c511842795d328c4f73d95574e8eb3c1f6a4eeb7da39993e968, so
# rebuilding the referee does not invalidate a single cached run.
set -eu

COMMIT=d28932afd22acf116c8bcd22e56a57faf3fb6d07
TARPATH=fasthenry/fasthenry-3.0wr-031424.tar.gz
SHA256=6da40d0e31425bca85be46434b33ecc194205d705b47f4459d91568c9f4301ef
SRCDIR=fasthenry-3.0wr
URL="https://raw.githubusercontent.com/wrcad/xictools/$COMMIT/$TARPATH"

CACHE="${RFX_REFEREE_CACHE:-$HOME/.cache/rfx-referees}/fasthenry"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

BIN="$CACHE/bin/fasthenry"
MANIFEST="$CACHE/manifest.json"
if [ "$FORCE" = 0 ] && [ -x "$BIN" ] && grep -q "$COMMIT" "$MANIFEST" 2>/dev/null \
   && grep -q '"binary_sha256"' "$MANIFEST" 2>/dev/null; then
  echo "fasthenry already built for commit $COMMIT" >&2
  echo "$BIN"
  exit 0
fi

mkdir -p "$CACHE/bin"
cd "$CACHE"
TGZ="$CACHE/$(basename "$TARPATH")"
if [ ! -f "$TGZ" ]; then
  echo "fetching $URL" >&2
  curl -fsSL --retry 3 -o "$TGZ.part" "$URL"
  mv "$TGZ.part" "$TGZ"
fi
GOT=$(shasum -a 256 "$TGZ" | cut -d' ' -f1)
if [ "$GOT" != "$SHA256" ]; then
  echo "sha256 mismatch for $TGZ: got $GOT, expected $SHA256" >&2
  exit 1
fi

rm -rf "$CACHE/$SRCDIR"
tar xzf "$TGZ"
cd "$CACHE/$SRCDIR"

# the one patch (see the header): impedance-matrix print precision
sed -i.orig 's/%13\.6lg %+13\.6lgj /%25.17le %+25.17lej /' src/fasthenry/induct.c
if ! grep -q '%25.17le' src/fasthenry/induct.c; then
  echo "patch did not apply: the printf format in src/fasthenry/induct.c moved" >&2
  exit 1
fi
diff src/fasthenry/induct.c.orig src/fasthenry/induct.c > "$CACHE/induct.c.patch.diff" || true

make fasthenry CC="${CC:-clang}" >"$CACHE/build.log" 2>&1 || {
  echo "build failed, see $CACHE/build.log" >&2; tail -20 "$CACHE/build.log" >&2; exit 1; }
# atomic install: a rename cannot be seen half-written by a concurrent run
cp -f bin/fasthenry "$BIN.new"
chmod +x "$BIN.new"
mv -f "$BIN.new" "$BIN"

# assert the patch took effect AND that the binary works: a 100 x 10 x 2 um bar
# must reproduce the Hoer-Love partial self-inductance 6.69092583858045e-11 H
# (spiral_greenhouse.self_inductance_bar) to < 1e-9 relative, which is only
# readable with more than six printed digits.
TMP=$(mktemp -d)
cat > "$TMP/bar.inp" <<'EOF'
* build self-test: one 100 x 10 x 2 um bar, sigma = 2e6 S/m
.units um
.default sigma=2
N1 x=0 y=0 z=0
N2 x=100 y=0 z=0
E1 N1 N2 w=10 h=2 wx=0 wy=1 wz=0
.external N1 N2
.freq fmin=1e3 fmax=1e3 ndec=1
.end
EOF
(cd "$TMP" && "$BIN" -s ludecomp -m direct -a off bar.inp >fh.log 2>&1)
IMAG=$(awk '/Impedance matrix/{getline; print $2}' "$TMP/Zc.mat" | tr -d 'j+')
REL=$(awk -v im="$IMAG" 'BEGIN{L=im/(2*3.141592653589793*1e3); r=L/6.69092583858045e-11-1; if(r<0)r=-r; print r}')
OK=$(awk -v r="$REL" 'BEGIN{print (r<1e-9)?1:0}')
echo "self-test: L = $IMAG / omega, relative to Hoer-Love: $REL" >&2
if [ "$OK" != "1" ]; then
  echo "self-test FAILED (relative $REL > 1e-9)" >&2
  exit 1
fi
rm -rf "$TMP"

VERSION=$("$BIN" -h 2>&1 | sed -n 's/^FastHenry Version \(.*\)$/\1/p' | head -1)
cat > "$MANIFEST" <<EOF
{
 "what": "pinned upstream FastHenry built by validation/referees/fasthenry/build_fasthenry.sh",
 "repo": "https://github.com/wrcad/xictools",
 "commit": "$COMMIT",
 "path_in_repo": "$TARPATH",
 "tarball_sha256": "$SHA256",
 "version": "$VERSION",
 "licence": "MIT FastHenry licence, Copyright (C) 2003 Board of Trustees of MIT (permissive, AS-IS, no-advertising clause); wr modifications by Whiteley Research Inc. under the same terms. Carried in every source file header, e.g. src/fasthenry/induct.c.",
 "patches": ["src/fasthenry/induct.c: cx_dumpMat_totextfile() print format %13.6lg -> %25.17le (impedance-matrix output precision only)"],
 "patch_diff_file": "$CACHE/induct.c.patch.diff",
 "compiler": "$(${CC:-clang} --version 2>&1 | head -1)",
 "uname": "$(uname -a)",
 "binary": "$BIN",
 "binary_sha256": "$(shasum -a 256 "$BIN" | cut -d' ' -f1)",
 "built_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "solver_options": "-s ludecomp -m direct -a off (exact dense mesh solve, no multipole approximation, no automatic filament refinement)"
}
EOF
echo "$BIN"
