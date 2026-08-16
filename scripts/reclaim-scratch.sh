#!/usr/bin/env bash
# Reclaim scratch space from campaign working directories.
#
# WHY THIS EXISTS
#
# Disk hits 99% roughly weekly, and it is not a leak — it is arithmetic. The unit of work is
# ~100 GB (a Qwen3.6-27B bf16 master is 52 GB, its f32 offload 71 GB, a per-tensor offload 15 GB),
# and campaigns iterate in double digits. Three mechanisms turn that into saturation:
#
#   1. RETRY LOOPS LEAVE FULL-SIZE CORPSES. Measured 2026-08-12: seven directories named
#      qwen36-model-offload-current-v1..v7, 266 identical files and 15 GB each, created in THIRTEEN
#      MINUTES. One failing job, seven attempts, nothing cleaned between them — 105 GB. A second
#      series (qwen36-s2kf-current-*) added 203 GB the same evening.
#   2. "current" IS APPEND-ONLY. A directory named `-current-vN` that coexists with v1..v(N-1) is
#      not current; it is the newest of N live copies.
#   3. DELETION DOES NOT RECLAIM. 98 GB was sitting in .Trash-1000 — machine-generated intermediates
#      routed through a GUI trash can that nothing ever empties.
#
# Only 8 GB of an 851 GB scratch tree was older than seven days, so age-based pruning alone would
# have reclaimed almost nothing. The waste is same-week duplication, which is what this targets.
#
# SAFETY
#
# Dry-run by default; --apply is required to delete. Never removes anything with an open file
# descriptor (checked against /proc), and never removes the newest member of a version series.
# Deletion is by `rm -rf`, not the trash, because routing intermediates through the trash is
# mechanism 3 above.

set -euo pipefail

SCRATCH="${TRITIUM_SCRATCH:-/mnt/4tb/tmp}"
APPLY=0
KEEP_DAYS="${TRITIUM_SCRATCH_KEEP_DAYS:-7}"

usage() {
    cat <<'USAGE'
usage: reclaim-scratch.sh [--apply] [--scratch DIR] [--keep-days N]

  --apply        actually delete (default is a dry run that only reports)
  --scratch DIR  scratch root (default /mnt/4tb/tmp, or $TRITIUM_SCRATCH)
  --keep-days N  age threshold for the stale sweep (default 7)

Reports and optionally reclaims, in increasing order of risk:
  1. superseded members of `<name>-vN` version series  (keeps the newest AND any superset)
  2. entries older than --keep-days
  3. the volume's trash directory
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --scratch) SCRATCH="$2"; shift 2 ;;
        --keep-days) KEEP_DAYS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -d "$SCRATCH" ] || { echo "scratch root $SCRATCH does not exist" >&2; exit 1; }

# Every path currently held open by any process. A campaign that has been running for days is the
# single most expensive thing on the box to delete by accident.
#
# Reads the symlink targets straight out of /proc rather than parsing `ls -l`. The first version of
# this did parse `ls -l` and was SILENTLY DEAD — it reported "0 skipped" across 240 deletion
# candidates, and a probe holding an open fd on a doomed directory was still listed for removal.
# A safety check that cannot be observed working is not a safety check, which is why the self-test
# below is mandatory rather than optional.
open_paths() {
    find /proc -mindepth 3 -maxdepth 3 -path '/proc/[0-9]*/fd/*' -type l -printf '%l\n' 2>/dev/null |
        sort -u
}
OPEN="$(open_paths || true)"

# NOTE THE HERESTRING. The obvious spelling — `printf '%s\n' "$OPEN" | grep -qF -- "$1"` — is broken
# under `set -o pipefail`: `grep -q` exits the moment it matches, `printf` is still writing, takes
# SIGPIPE, and pipefail propagates PRINTF's failure. The function then returns non-zero exactly when
# it FINDS a match — the guard fails open, silently, only for paths that match early in the sorted
# list. That is what made this dead the first time: the self-test probe lived under /tmp, which
# sorts after /mnt, so grep scanned to the end, printf completed, and the test passed while every
# real /mnt/4tb path inverted. A herestring has no pipeline and no such failure mode.
in_use() {
    [ -n "$OPEN" ] || return 1
    grep -qF -- "$1" <<<"$OPEN"
}

# Self-test the guard against a real held descriptor. Aborts rather than proceeding with a guard
# that does not fire — the failure mode this protects against (deleting a multi-day campaign's
# working set) is far worse than refusing to run.
# The probe MUST live under the scratch root, not $TMPDIR. Sort order decides whether the SIGPIPE
# bug above is exercised, so a probe in /tmp can pass while every real target in /mnt/4tb fails.
# Test the guard where it will actually be used.
guard_self_test() {
    local probe
    probe="$(mktemp -d "$SCRATCH/.reclaim-guard-XXXXXX")"
    : >"$probe/held"
    exec {probe_fd}<"$probe/held"
    OPEN="$(open_paths || true)"
    local ok=0
    in_use "$probe" && ok=1
    exec {probe_fd}<&-
    rm -rf -- "$probe"
    OPEN="$(open_paths || true)"
    if [ "$ok" != 1 ]; then
        echo "FATAL: open-fd guard did not detect a held descriptor; refusing to delete." >&2
        echo "       (/proc may be unreadable, or the process table is hidden — hidepid=2?)" >&2
        exit 3
    fi
}
guard_self_test

human() { du -sh "$1" 2>/dev/null | cut -f1; }

removed_total=0
report() {
    local path="$1" reason="$2"
    if in_use "$path"; then
        echo "  SKIP (open fds)  $(human "$path")  $path"
        return
    fi
    echo "  $([ "$APPLY" = 1 ] && echo REMOVE || echo "would remove")  $(human "$path")  $path   [$reason]"
    if [ "$APPLY" = 1 ]; then
        rm -rf -- "$path"
        removed_total=$((removed_total + 1))
    fi
}

echo "scratch: $SCRATCH   free before: $(df -h "$SCRATCH" | tail -1 | awk '{print $4}')"
[ "$APPLY" = 1 ] || echo "DRY RUN — pass --apply to delete"

# ── 1. Superseded members of a `-vN` version series ──────────────────────────────────────────────
# Keep the newest. Also keep any member with MORE entries than the newest: on 2026-08-12 v6 held
# 106 files that v7 lacked, so v7 was a partial restart and deleting v6 would have destroyed the
# only copy of those tensors. "Newest" is not the same as "most complete".
echo
echo "[1] superseded version-series members"
mapfile -t SERIES < <(
    find "$SCRATCH" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null |
        sed -n 's/-v[0-9]\+$//p' | sort -u
)
for base in "${SERIES[@]:-}"; do
    [ -n "$base" ] || continue
    mapfile -t members < <(find "$SCRATCH" -maxdepth 1 -mindepth 1 -type d \
        -name "$base-v[0-9]*" -printf '%T@ %p\n' 2>/dev/null | sort -n | awk '{print $2}')
    [ "${#members[@]}" -gt 1 ] || continue
    newest="${members[-1]}"
    newest_count=$(find "$newest" -maxdepth 1 | wc -l)
    for m in "${members[@]}"; do
        [ "$m" = "$newest" ] && continue
        count=$(find "$m" -maxdepth 1 | wc -l)
        if [ "$count" -gt "$newest_count" ]; then
            echo "  KEEP (superset of newest: $count > $newest_count entries)  $(human "$m")  $m"
            continue
        fi
        report "$m" "superseded by $(basename "$newest")"
    done
done

# ── 2. Stale entries ─────────────────────────────────────────────────────────────────────────────
echo
echo "[2] entries older than $KEEP_DAYS days"
while IFS= read -r path; do
    [ -n "$path" ] || continue
    report "$path" "older than ${KEEP_DAYS}d"
done < <(find "$SCRATCH" -maxdepth 1 -mindepth 1 -mtime "+$KEEP_DAYS" 2>/dev/null)

# ── 3. Trash on the same volume ──────────────────────────────────────────────────────────────────
echo
echo "[3] trash"
mount_point=$(df --output=target "$SCRATCH" | tail -1)
for trash in "$mount_point/.Trash-$(id -u)" "$HOME/.local/share/Trash"; do
    [ -d "$trash" ] || continue
    echo "  $([ "$APPLY" = 1 ] && echo EMPTY || echo "would empty")  $(human "$trash")  $trash"
    [ "$APPLY" = 1 ] && find "$trash" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
done

echo
echo "free after: $(df -h "$SCRATCH" | tail -1 | awk '{print $4}')"
[ "$APPLY" = 1 ] && echo "removed $removed_total path(s)"
exit 0
