#!/bin/sh
# Refuse files that belong on the lab NFS, not GitHub (lab rule on large files).
#   .githooks/check-large-files.sh --staged        files added or changed in the index (pre-commit hook)
#   .githooks/check-large-files.sh <base> <head>   every commit after <base> up to <head> (CI): a file added and
#                                                  removed again inside one push still stays in history
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
known=''
trap '[ -n "$known" ] && rm -f "$known"' EXIT

# seen <blob>: the blob is reachable from $BASE (a moved or renamed file, or an old file added back), so it is no new
# upload. Blobs absent from a partial clone still print (as ?<id>) because their trees are present.
seen() {
    [ -n "$BASE" ] || return 1
    if [ -z "$known" ]; then
        known=$(mktemp) || exit 2
        git rev-list --objects --missing=print "$BASE" | cut -d' ' -f1 | sed 's/^?//' > "$known" || exit 2
    fi
    grep -qx "$1" "$known"
}

# check <rev> <names>: <rev> empty means the index. Renames count as additions (--no-renames).
check() {
    rev=$1
    IFS=$nl
    for path in $2; do
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
    BASE=$(git rev-parse -q --verify HEAD)
    check '' "$(git -c core.quotepath=off diff --cached --name-only --no-renames --diff-filter=d)"
else
    BASE=$1
    [ "$1" = "$EMPTY" ] && BASE=''
    if [ "$1" = "$EMPTY" ]; then
        commits=$(git rev-list --reverse "$2") || exit 2
    else
        commits=$(git rev-list --reverse "$1..$2") || exit 2
    fi
    for c in $commits; do
        parent=$(git rev-parse -q --verify "$c^") || parent=$EMPTY
        check "$c" "$(git -c core.quotepath=off diff --name-only --no-renames --diff-filter=d "$parent" "$c")"
    done
fi
exit $bad
