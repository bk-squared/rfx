#!/bin/sh
# Cases for check-large-files.sh in a scratch repo. Run: sh .githooks/check-large-files-test.sh "$(git rev-parse --show-toplevel)"
# Each line prints the exit code next to the expected one.
W=$1; D=$(mktemp -d)
cd "$D" && git init -q -b main r && cd r
mkdir -p .githooks "Papers/X fig"
cp -p "$W/.githooks/check-large-files.sh" "$W/.githooks/pre-commit" .githooks/
c() { git -c user.name=t -c user.email=t@t commit -q --no-verify -m "$1"; }
chk() { sh .githooks/check-large-files.sh "$@" >/dev/null 2>&1; echo $?; }
echo s > "Papers/X fig/a.tex"; git add -A; c base; B0=$(git rev-parse HEAD)
head -c 6000000 /dev/urandom > "Papers/X fig/run 1.npz"; git add -A; c add
git rm -q "Papers/X fig/run 1.npz"; c del
echo "R1 add-then-delete in range (expect 1): $(chk $B0 HEAD)"
B1=$(git rev-parse HEAD)
head -c 4000000 /dev/urandom > "Papers/X fig/small.npz"; git add -A; c small
B2=$(git rev-parse HEAD)
git mv "Papers/X fig/small.npz" "Papers/X fig/big.npz"; head -c 2000000 /dev/urandom >> "Papers/X fig/big.npz"; git add -A; c rename-grow
echo "R2 rename and grow (expect 1): $(chk $B2 HEAD)"
head -c 30000000 /dev/urandom > Papers/sub.zip; git add -A; c zip
echo "R3 30 MB zip only (expect 0): $(chk $(git rev-parse HEAD~1) HEAD)"
echo "R4 whole history from empty tree (expect 1): $(chk $(git hash-object -t tree /dev/null) HEAD)"
head -c 6000000 /dev/urandom > d.csv; git add d.csv; echo "S1 staged 6 MB csv (expect 1): $(chk --staged)"
git reset -q; rm d.csv; echo "S2 nothing staged (expect 0): $(chk --staged)"
# Messages, with REPO and PUBLIC set the way the PR builder sets them.
fill() { sed -e "s/^REPO=.*/REPO=t-repo/" -e "s/^PUBLIC=.*/PUBLIC=$1/" "$W/.githooks/check-large-files.sh" > .githooks/check-large-files.sh; }
head -c 6000000 /dev/urandom > d.csv; git add d.csv
fill 0; m=$(sh .githooks/check-large-files.sh --staged 2>&1)
case $m in *"<이름>-workspace/t-repo/./nfs-only/"*NFS_FILES.tsv*) r=1 ;; *) r=0 ;; esac
echo "M1 private message names the repo folder (expect 1): $r"
fill 1; m=$(sh .githooks/check-large-files.sh --staged 2>&1)
case $m in *workspace*|*NFS_FILES*|*remilab-cnu*) r=0 ;; *"연구실 NFS"*) r=1 ;; *) r=0 ;; esac
echo "M2 public message has no internal path (expect 1): $r"
git reset -q; rm d.csv
# Moves: a blob already reachable from the base is no new upload; one added inside the range still is.
head -c 6000000 /dev/urandom > old.npz; git add old.npz; c old; B3=$(git rev-parse HEAD)
mkdir -p moved; git mv old.npz moved/old.npz; c move
echo "R5 move of a file already in the base (expect 0): $(chk $B3 HEAD)"
head -c 6000000 /dev/urandom > new.npz; git add new.npz; c new; git mv new.npz moved/new.npz; c move-new
echo "R6 file added then moved inside the range (expect 1): $(chk $B3 HEAD)"
git mv moved/old.npz old.npz; echo "S3 staged move of a committed file (expect 0): $(chk --staged)"
git reset -q --hard
# The CI clone has no blobs (filter blob:none): the same answers must come from trees alone.
git config uploadpack.allowFilter true
git clone -q --filter=blob:none --no-checkout "file://$PWD" ../p 2>/dev/null && cd ../p
pchk() { sh "$D/r/.githooks/check-large-files.sh" "$@" >/dev/null 2>&1; echo $?; }
gone() { git rev-list --objects --missing=print --all | grep -c '^?'; }
g0=$(gone)
echo "P1 partial clone, move (expect 0): $(pchk $(git rev-parse HEAD~3) HEAD~2)"
echo "P3 the move check fetched no blob (expect 1): $([ "$(gone)" -eq "$g0" ] && echo 1 || echo 0)"
echo "P2 partial clone, added then moved (expect 1): $(pchk $(git rev-parse HEAD~2) HEAD)"
cd / && rm -rf "${D:?}"
