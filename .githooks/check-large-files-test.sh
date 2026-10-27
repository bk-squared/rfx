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
cd / && rm -rf "${D:?}"
