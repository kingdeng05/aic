#!/usr/bin/env bash
# generate_montage.sh — builds a 4x4 grid montage of 16 random successful
# episodes' center_camera videos, sped up 4x. Useful as a visual sanity check
# of an N-episode batch.
#
# Usage:
#   generate_montage.sh <summary.tsv> <out.mp4>
# summary.tsv comes from run_batch_sc.sh; column 3 is outcome, column 4 is
# repo_id. Reads ~/.cache/huggingface/lerobot/<repo>/videos/...

set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <summary.tsv> <out.mp4>" >&2
    exit 2
fi
SUMMARY="$1"
OUT="$2"

[[ -f "$SUMMARY" ]] || { echo "summary.tsv not found: $SUMMARY" >&2; exit 1; }

# ffmpeg lives in the pixi env, not on the host PATH.
PIXI_FFMPEG="$HOME/ws_aic/src/aic/.pixi/envs/default/bin/ffmpeg"
if [[ -x "$PIXI_FFMPEG" ]]; then
    FFMPEG="$PIXI_FFMPEG"
elif command -v ffmpeg >/dev/null; then
    FFMPEG=ffmpeg
else
    echo "ffmpeg not found (looked for $PIXI_FFMPEG and on PATH)" >&2
    exit 1
fi

LEROBOT_CACHE="$HOME/.cache/huggingface/lerobot"
# Pick OK episodes' repo IDs in TSV order, then shuffle and take up to 16.
mapfile -t OK_REPOS < <(awk -F'\t' '$3=="OK" {print $4}' "$SUMMARY" | shuf | head -16)
if [[ ${#OK_REPOS[@]} -lt 1 ]]; then
    echo "No OK episodes in $SUMMARY; cannot build montage." >&2
    exit 1
fi

# Resolve each repo to a center_camera mp4. Tolerate missing videos by
# skipping them rather than aborting.
CLIPS=()
for repo in "${OK_REPOS[@]}"; do
    base="$LEROBOT_CACHE/$repo/videos/observation.images.center_camera"
    mp4=$(find "$base" -name "*.mp4" 2>/dev/null | head -1)
    [[ -f "$mp4" ]] && CLIPS+=("$mp4")
done

N=${#CLIPS[@]}
if [[ "$N" -lt 1 ]]; then
    echo "Found 0 center_camera videos under $LEROBOT_CACHE for the OK episodes." >&2
    exit 1
fi

# We want a tidy 4x4 grid; pad with the first clip if we have fewer than 16.
while [[ ${#CLIPS[@]} -lt 16 ]]; do
    CLIPS+=("${CLIPS[0]}")
done

OUT_DIR=$(dirname "$OUT")
mkdir -p "$OUT_DIR"

# Build ffmpeg command: 16 inputs, scale each to 320x240, tile into 4x4 grid,
# speed up 4x, mute audio. Resulting frame size 1280x960.
INPUT_FLAGS=()
for c in "${CLIPS[@]:0:16}"; do
    INPUT_FLAGS+=(-i "$c")
done

# Filter graph: scale each input, then xstack into 4x4, then setpts for 4x.
FILTER=""
for j in $(seq 0 15); do
    FILTER+="[${j}:v]scale=320:240,setpts=0.25*PTS[v${j}];"
done
FILTER+="[v0][v1][v2][v3][v4][v5][v6][v7][v8][v9][v10][v11][v12][v13][v14][v15]"
FILTER+="xstack=inputs=16:layout=0_0|320_0|640_0|960_0|0_240|320_240|640_240|960_240|0_480|320_480|640_480|960_480|0_720|320_720|640_720|960_720[v]"

echo "Building 4x4 montage from $N OK episode(s) -> $OUT"
"$FFMPEG" -y -hide_banner -loglevel error \
    "${INPUT_FLAGS[@]}" \
    -filter_complex "$FILTER" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 23 \
    "$OUT"

echo "Wrote $OUT"
ls -la "$OUT"
