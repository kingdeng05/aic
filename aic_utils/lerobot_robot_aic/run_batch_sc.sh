#!/usr/bin/env bash
# run_batch_sc.sh — driver for an N-episode SC-port collection run.
#
# For each episode:
#   - invokes run_iter_sc.sh (sim launch + record + dataset-quality gate)
#   - parses the sim log for the sampled home offset + TCP home + board pose
#     so the episode is replicable
#   - parses the rec log for retry count and final port-local-z (if success)
#   - appends a TSV row to summary.tsv
#   - prints a one-line PASS / FAIL summary
#
# Episodes that fail the honest QC (no "insertion success" log line, plateau,
# missing videos, etc.) are auto-deleted on disk by run_iter_sc.sh (unless
# --keep-bad is passed here, which propagates --keep down). The TSV row is
# always written.
#
# Once the loop finishes, kicks off generate_montage.sh which builds a 4x4
# grid montage of 16 random good episodes.
#
# Usage:
#   run_batch_sc.sh [<slot>] [-n <count>] [--time-s=<sec>]
#                   [--out-dir=<dir>] [--keep-bad]

set -uo pipefail

SLOT="sc_port_0"
N=150
TIME_S=120
OUT_DIR=""
KEEP_BAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)              N="$2";              shift 2 ;;
        -n=*)            N="${1#*=}";         shift ;;
        --count=*)       N="${1#*=}";         shift ;;
        --time-s=*)      TIME_S="${1#*=}";    shift ;;
        --out-dir=*)     OUT_DIR="${1#*=}";   shift ;;
        --keep-bad)      KEEP_BAD=1;          shift ;;
        --*)             echo "unknown flag: $1" >&2; exit 2 ;;
        *)               SLOT="$1";           shift ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="/tmp/sc_batch_$(date +%Y%m%d_%H%M%S)_$$"
fi
mkdir -p "$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.tsv"
RUN_ITER="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/run_iter_sc.sh"
[[ -x "$RUN_ITER" ]] || { echo "run_iter_sc.sh not found or not executable at $RUN_ITER" >&2; exit 1; }
GEN_VIDEO="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/generate_montage.sh"

# Header
printf "ep\tts_utc\toutcome\trepo_id\thome_dx\thome_dy\thome_dz\ttcp_home_x\ttcp_home_y\ttcp_home_z\tboard_translation\tboard_roll\tboard_pitch\tboard_yaw\tretries\tfinal_port_local_z\tquality\n" > "$SUMMARY"

OK_COUNT=0
BAD_COUNT=0
FAIL_COUNT=0
START_EPOCH=$(date +%s)

echo "=== sc_port_0 batch: N=$N episodes, time_s=$TIME_S, out=$OUT_DIR ==="

for i in $(seq 1 "$N"); do
    EP_LOG_DIR="$OUT_DIR/logs/ep$(printf '%03d' "$i")"
    mkdir -p "$EP_LOG_DIR"
    EP_START_TS=$(date -u +%FT%TZ)

    KEEP_FLAG=""
    [[ $KEEP_BAD -eq 1 ]] && KEEP_FLAG="--keep"
    RAW=$("$RUN_ITER" "$SLOT" --time-s="$TIME_S" $KEEP_FLAG --log-dir="$EP_LOG_DIR" 2>&1)
    LAST=$(echo "$RAW" | tail -1)
    OUTCOME=$(echo "$LAST" | sed -nE 's/.*OUTCOME=([A-Z_]+).*/\1/p')
    REPO=$(echo "$LAST" | sed -nE 's/.*REPO=([^ ]+).*/\1/p')
    QUALITY=$(echo "$LAST" | sed -nE 's/.*QUALITY="([^"]*)".*/\1/p')
    OUTCOME="${OUTCOME:-LAUNCH_FAIL}"
    REPO="${REPO:--}"
    QUALITY="${QUALITY:--}"

    SIM_OUT=$(ls "$EP_LOG_DIR"/sim_*.out 2>/dev/null | head -1)
    REC_LOG=$(ls "$EP_LOG_DIR"/rec_*.log 2>/dev/null | head -1)

    HOME_DX="-"; HOME_DY="-"; HOME_DZ="-"
    TCP_X="-";  TCP_Y="-";  TCP_Z="-"
    BOARD_TX="-"; BOARD_ROLL="-"; BOARD_PITCH="-"; BOARD_YAW="-"
    if [[ -n "$SIM_OUT" && -f "$SIM_OUT" ]]; then
        HOME_LINE=$(grep "dx = " "$SIM_OUT" | head -1)
        if [[ -n "$HOME_LINE" ]]; then
            HOME_DX=$(echo "$HOME_LINE" | sed -nE 's/.*dx = ([+-]?[0-9.]+).*/\1/p')
            HOME_DY=$(echo "$HOME_LINE" | sed -nE 's/.*dy = ([+-]?[0-9.]+).*/\1/p')
            HOME_DZ=$(echo "$HOME_LINE" | sed -nE 's/.*dz = ([+-]?[0-9.]+).*/\1/p')
        fi
        TCP_LINE=$(grep "TCP home" "$SIM_OUT" | head -1)
        if [[ -n "$TCP_LINE" ]]; then
            TCP_X=$(echo "$TCP_LINE" | sed -nE 's/.*TCP home = \(([+-]?[0-9.]+), [+-]?[0-9.]+, [+-]?[0-9.]+\).*/\1/p')
            TCP_Y=$(echo "$TCP_LINE" | sed -nE 's/.*TCP home = \([+-]?[0-9.]+, ([+-]?[0-9.]+), [+-]?[0-9.]+\).*/\1/p')
            TCP_Z=$(echo "$TCP_LINE" | sed -nE 's/.*TCP home = \([+-]?[0-9.]+, [+-]?[0-9.]+, ([+-]?[0-9.]+)\).*/\1/p')
        fi
        # Scene block: capture sc_port_0's translation/roll/pitch/yaw.
        BOARD_TX=$(grep -A4 "^${SLOT}:" "$SIM_OUT" | grep "translation" | head -1 | sed -nE 's/.*= +([+-]?[0-9.]+) m.*/\1/p')
        BOARD_ROLL=$(grep -A4 "^${SLOT}:" "$SIM_OUT" | grep "roll" | head -1 | sed -nE 's/.*= +([+-]?[0-9.]+) rad.*/\1/p')
        BOARD_PITCH=$(grep -A4 "^${SLOT}:" "$SIM_OUT" | grep "pitch" | head -1 | sed -nE 's/.*= +([+-]?[0-9.]+) rad.*/\1/p')
        BOARD_YAW=$(grep -A4 "^${SLOT}:" "$SIM_OUT" | grep "yaw" | head -1 | sed -nE 's/.*= +([+-]?[0-9.]+) rad.*/\1/p')
        BOARD_TX="${BOARD_TX:--}"; BOARD_ROLL="${BOARD_ROLL:--}"
        BOARD_PITCH="${BOARD_PITCH:--}"; BOARD_YAW="${BOARD_YAW:--}"
    fi

    RETRIES=0
    FINAL_PL_Z="-"
    if [[ -n "$REC_LOG" && -f "$REC_LOG" ]]; then
        # `grep -c` exits 1 when there are 0 matches; piping to wc -l avoids
        # that and also avoids the "0\n0" output we'd get from `grep -c ... ||
        # echo 0` when both branches fire.
        RETRIES=$(grep "lifting to z_offset" "$REC_LOG" 2>/dev/null | wc -l)
        FINAL_PL_Z=$(grep "insertion success" "$REC_LOG" 2>/dev/null | sed -nE 's/.*port_local_z=([+-]?[0-9.]+).*/\1/p')
        FINAL_PL_Z="${FINAL_PL_Z:--}"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$i" "$EP_START_TS" "$OUTCOME" "$REPO" \
        "$HOME_DX" "$HOME_DY" "$HOME_DZ" \
        "$TCP_X" "$TCP_Y" "$TCP_Z" \
        "$BOARD_TX" "$BOARD_ROLL" "$BOARD_PITCH" "$BOARD_YAW" \
        "$RETRIES" "$FINAL_PL_Z" "$QUALITY" >> "$SUMMARY"

    FLAG="?"
    case "$OUTCOME" in
        OK)          FLAG="[PASS]"; OK_COUNT=$((OK_COUNT + 1)) ;;
        BAD)         FLAG="[FAIL]"; BAD_COUNT=$((BAD_COUNT + 1)) ;;
        LAUNCH_FAIL) FLAG="[LAUNCH]"; FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
        *)           FLAG="[?]" ;;
    esac
    ELAPSED=$(( $(date +%s) - START_EPOCH ))
    printf "[ep%03d] %s outcome=%-11s home=(%s,%s,%s) tcp=(%s,%s,%s) board_tx=%s retries=%s final_pl_z=%s  ok=%d bad=%d fail=%d t=%dm\n" \
        "$i" "$FLAG" "$OUTCOME" "$HOME_DX" "$HOME_DY" "$HOME_DZ" \
        "$TCP_X" "$TCP_Y" "$TCP_Z" "$BOARD_TX" "$RETRIES" "$FINAL_PL_Z" \
        "$OK_COUNT" "$BAD_COUNT" "$FAIL_COUNT" "$((ELAPSED / 60))"
done

TOTAL=$((OK_COUNT + BAD_COUNT + FAIL_COUNT))
ELAPSED=$(( $(date +%s) - START_EPOCH ))
echo ""
echo "=== Batch complete ==="
echo "Duration:     $((ELAPSED / 60)) min ($ELAPSED s)"
echo "Total:        $TOTAL"
echo "Good (PASS):  $OK_COUNT"
echo "Bad (FAIL):   $BAD_COUNT"
echo "Launch fail:  $FAIL_COUNT"
if [[ "$TOTAL" -gt 0 ]]; then
    echo "Success rate: $(awk "BEGIN {printf \"%.1f%%\", $OK_COUNT * 100 / $TOTAL}")"
fi
echo "Summary TSV:  $SUMMARY"
echo ""

if [[ -x "$GEN_VIDEO" && $OK_COUNT -gt 0 ]]; then
    echo "=== Generating montage video ==="
    "$GEN_VIDEO" "$SUMMARY" "$OUT_DIR/montage_4x4.mp4" || echo "(montage failed; see above)"
fi
