#!/bin/sh
# Refuse files that belong on the lab NFS, not GitHub (lab rule on large files).
#   .githooks/check-large-files.sh --staged        files added or changed in the index (pre-commit hook)
#   .githooks/check-large-files.sh <base> <head>   every commit after <base> up to <head> (CI): a file added and
#                                                  removed again inside one push still stays in history
#   .githooks/check-large-files.sh --new <head> [<exclude-rev> ...]
# Every file over 50 MB is refused. Over 5 MB, only manuscript, figure and submission formats may stay on GitHub.
# A file whose content is already reachable from the base (moved, renamed, added back) is no new upload and passes.
REPO=rfx
PUBLIC=1
SOFT=5000000
HARD=50000000
DOC=' pdf zip docx doc pptx ppt png jpg jpeg eps tif tiff svg vsdx '
EMPTY=$(git hash-object -t tree /dev/null)

digest() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi | cut -d' ' -f1
}

bad=0
nl='
'
work=$(mktemp -d) || exit 2
known=$work/known
: > "$known"
commits_checked=0
paths_checked=0
fetched=0
trap 'echo "large-files: commits=$commits_checked paths=$paths_checked fetched=$fetched" >&2; rm -rf "$work"' 0
trap 'exit 2' HUP INT TERM

# Excluded history is already on the remote, including blobs absent from a partial clone.
seen() {
    grep -qx "$1" "$known"
}

# check <rev> <names>: <rev> empty means the index. Renames count as additions (--no-renames).
check() {
    rev=$1
    IFS=$nl
    for path in $2; do
        paths_checked=$((paths_checked + 1))
        seen "$(git rev-parse "$rev:$path")" && continue  # before the size: no blob download for a move
        size=$(git cat-file -s "$rev:$path") || exit 2
        [ "$size" -gt "$SOFT" ] || continue
        ext=$(printf '%s' "${path##*.}" | tr 'A-Z' 'a-z')
        if [ "$size" -gt "$HARD" ]; then
            why='50 MB 넘는 파일'
        else
            case $DOC in *" $ext "*) continue ;; esac
            why='5 MB 넘는 자료 파일 (원고·그림·제출 형식만 50 MB까지 GitHub에 둡니다)'
        fi
        if [ $bad -eq 0 ]; then
            if [ "$PUBLIC" = 1 ]; then
                echo '이 파일은 GitHub에 올리지 않고 연구실 NFS에 둡니다.'
            else
                echo '이 파일은 GitHub에 올리지 않고 NFS에 둡니다 (규칙: remilab-cnu/paper docs/DATA_MANAGEMENT.md 2절).'
            fi
        fi
        bad=1
        echo ""
        echo "  $path${rev:+ (커밋 $(git rev-parse --short "$rev"))}"
        echo "  크기 $size 바이트: $why"
        if [ "$PUBLIC" = 1 ]; then
            echo "  1. 파일을 연구실 NFS로 옮깁니다."
            echo "  2. 이 커밋에서 뺍니다: git rm --cached \"$path\""
            continue
        fi
        sha=$(git cat-file blob "$rev:$path" | digest)
        folder=$(dirname "$path")
        echo "  1. 파일을 NFS 본인 작업 공간으로 옮깁니다: <이름>-workspace/$REPO/$folder/nfs-only/"
        echo "  2. $folder/NFS_FILES.tsv 에 다음 한 줄을 더합니다 (탭으로 나눔):"
        printf '     <이름>-workspace/%s/%s/nfs-only/%s\t%s\t%s\n' "$REPO" "$folder" "${path##*/}" "$size" "$sha"
        echo "  3. 이 커밋에서 뺍니다: git rm --cached \"$path\""
    done
    unset IFS
}

if [ "$1" = --staged ]; then
    BASE=$(git rev-parse -q --verify HEAD) || BASE=''
    if [ -n "$BASE" ]; then
        git rev-list --objects --missing=print "$BASE" > "$work/objects" || exit 2
        cut -d' ' -f1 "$work/objects" | sed 's/^?//' > "$known"
    fi
    check '' "$(git -c core.quotepath=off diff --cached --name-only --no-renames --diff-filter=d)"
else
    if [ "$1" = --new ]; then
        head=$2
        shift 2
    else
        base=$1
        head=$2
        set -- "$base"
        [ "$base" != "$EMPTY" ] || set --
    fi
    commits=$(git rev-list --reverse "$head" --not "$@") || exit 2
    if [ "$#" -gt 0 ] && [ -n "$commits" ]; then
        git rev-list --objects --missing=print "$@" > "$work/objects" || exit 2
        cut -d' ' -f1 "$work/objects" | sed 's/^?//' > "$known"
    fi
    : > "$work/entries"
    for c in $commits; do
        parent=$(git rev-parse -q --verify "$c^") || parent=$EMPTY
        git -c core.quotepath=off diff --raw --no-abbrev --no-renames --diff-filter=d "$parent" "$c" > "$work/raw" || exit 2
        # Raw records contain tree blob IDs; obtaining these does not fetch content.
        awk -v c="$c" 'BEGIN { FS="\t"; OFS="\t" } { split($1, h, " "); print c, h[4], $2 }' "$work/raw" >> "$work/entries"
    done
    cut -f2 "$work/entries" | sort -u > "$work/ids"
    awk 'FILENAME == ARGV[1] { seen[$1]=1; next } !($1 in seen)' "$known" "$work/ids" > "$work/new"
    git cat-file --batch-all-objects --batch-check='%(objectname)' > "$work/local" || exit 2
    awk 'FILENAME == ARGV[1] { local[$1]=1; next } !($1 in local)' "$work/local" "$work/new" > "$work/missing"
    fetched=$(wc -l < "$work/missing" | tr -d ' ')
    if [ "$fetched" -gt 0 ]; then
        split -l 5000 "$work/missing" "$work/batch-" || exit 2
        for batch in "$work"/batch-*; do
            git fetch --no-tags --no-write-fetch-head --stdin origin < "$batch" || exit 2
        done
    fi
    for c in $commits; do
        commits_checked=$((commits_checked + 1))
        names=$(awk -F '\t' -v c="$c" '$1 == c { print $3 }' "$work/entries")
        check "$c" "$names"
    done
fi
exit $bad
