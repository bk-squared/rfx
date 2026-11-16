#!/bin/sh
# Cases for check-large-files.sh in a scratch repo. Run: sh .githooks/check-large-files-test.sh "$(git rev-parse --show-toplevel)"
# Each line prints the exit code next to the expected one.
W=$1; D=$(mktemp -d)
trap 'rm -rf "$D"' 0
cd "$D" && git clone -q --no-checkout --local "$W" r && cd r
git switch -q --orphan large-files-tests
mkdir -p .githooks "Papers/X fig"
cp -p "$W/.githooks/check-large-files.sh" "$W/.githooks/pre-commit" .githooks/
c() { git -c user.name=t -c user.email=t@t commit -q --no-verify -m "$1"; }
chk() { sh .githooks/check-large-files.sh "$@" >/dev/null 2>&1; echo $?; }
echo s > "Papers/X fig/a.tex"; git add -A; c base; B0=$(git rev-parse HEAD)
head -c 6000000 /dev/urandom > "Papers/X fig/run 1.npz"; git add -A; c add
git rm -q "Papers/X fig/run 1.npz"; c del
echo "R1 add-then-delete in range (expect 1): $(chk $B0 HEAD)"
echo "N3 --new matches R1 legacy range (expect 1): $(chk --new HEAD $B0)"
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
cd "$D/r"
git branch n1-remote HEAD
n1=$(sh .githooks/check-large-files.sh --new HEAD n1-remote 2>&1); n1_exit=$?
echo "N1 commits already on another ref (expect 0): $n1_exit"
# Exit alone cannot detect a range regression: seen() also skips excluded blobs.
case $n1 in *'large-files: commits=0 paths=0 fetched=0'*) n1_range=0 ;; *) n1_range=1 ;; esac
echo "N1 zero-commit range assertion (expect 0): $n1_range"
other=$(git rev-parse HEAD)
git switch -q --orphan unrelated
printf 'root\n' > root.txt; git add root.txt; c unrelated-root
head -c 6000000 /dev/urandom > fresh.npz; git add fresh.npz; c unrelated-new
git branch already-remote "$other"
git clone -q --filter=blob:none --no-checkout "file://$PWD" "$D/n2" 2>/dev/null
cd "$D/n2"
n2=$(sh "$W/.githooks/check-large-files.sh" --new HEAD origin/already-remote >/dev/null 2>&1; echo $?)
echo "N2 unrelated history and a new 6 MB npz (expect 1): $n2"
# N4: explicit blob wants must not negotiate commits whose blobs are missing.
cd "$D/r"
git switch -q --orphan n4
printf 'base\n' > base.txt; git add base.txt; c n4-base; n4_base=$(git rev-parse HEAD)
printf 'small\n' > small.txt; git add small.txt; c n4-small; n4_small=$(git rev-parse HEAD)
head -c 6000000 /dev/urandom > large.npz; git add large.npz; c n4-large; n4_large=$(git rev-parse HEAD)
# Force only the script's batch fetch to fail; Git's internal lazy fetch still uses
# the real Git. Fresh clones keep each check independent of earlier downloads.
real_git=$(command -v git)
mkdir "$D/bin"
cat > "$D/bin/git" <<'EOF'
#!/bin/sh
if [ "${N4_FAIL_BATCH:-0}" = 1 ]; then
    for arg do
        if [ "$arg" = --stdin ]; then
            echo forced-batch-failure >&2
            exit 1
        fi
    done
fi
exec "$N4_REAL_GIT" "$@"
EOF
chmod +x "$D/bin/git"
for mode in batch fallback; do
    for kind in small large; do
        case $kind in
            small) n4_head=$n4_small; n4_path=small.txt; expected=0 ;;
            large) n4_head=$n4_large; n4_path=large.npz; expected=1 ;;
        esac
        git clone -q --filter=blob:none --no-checkout "file://$D/r" "$D/n4-$mode-$kind" 2>/dev/null
        cd "$D/n4-$mode-$kind"
        oid=$(git rev-parse "$n4_head:$n4_path")
        missing=$(git rev-list --objects --missing=print "$n4_head" | grep -cx "?$oid")
        echo "N4 $mode $kind blob initially missing (expect 1): $missing"
        fail_batch=0; [ "$mode" = fallback ] && fail_batch=1
        PATH="$D/bin:$PATH" N4_REAL_GIT="$real_git" N4_FAIL_BATCH=$fail_batch \
            GIT_TRACE_PACKET="$D/n4-$mode-$kind.packet" \
            sh "$W/.githooks/check-large-files.sh" --new "$n4_head" "$n4_base" > "$D/n4-$mode-$kind.log" 2>&1
        echo "N4 $mode $kind (expect $expected): $?"
        if [ "$mode" = batch ]; then
            # Exit codes alone also pass when a broken batch is rescued lazily.
            haves=$(grep -c 'fetch> have ' "$D/n4-$mode-$kind.packet")
            echo "N4 batch $kind advertises no haves (expect 0): $haves"
        else
            forced=$(grep -c '^forced-batch-failure$' "$D/n4-$mode-$kind.log")
            echo "N4 fallback $kind exercised (expect 1): $forced"
        fi
    done
done
cd / && rm -rf "${D:?}"
