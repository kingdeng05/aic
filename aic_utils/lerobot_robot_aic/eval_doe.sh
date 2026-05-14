#!/usr/bin/env bash
# eval_doe.sh — DoE driver. Batch1: 5 translations, yaw=0. Batch2: 3x3 grid.
set -uo pipefail

ITER_SH="$(dirname "$0")/run_eval_iter.sh"
OUT="/tmp/aic_eval_doe/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
RESULTS="$OUT/results.tsv"
echo -e "iter\tbatch\ttranslation\tyaw\tscore\toutcome" > "$RESULTS"
echo "DoE root: $OUT"

linspace() { python3 -c "lo,hi,n=$1,$2,$3; print(' '.join(f'{lo+(hi-lo)*i/(n-1):.6f}' for i in range(n)))"; }

run() {
    local i=$1 batch=$2 t=$3 y=$4
    local d="$OUT/$(printf 'iter_%02d_b%d_t%s_y%s' $i $batch $t $y)"
    echo ""; echo "=== iter $i batch=$batch t=$t y=$y === $(date)"
    local res
    res=$(bash "$ITER_SH" "$t" "$y" "$d" | tail -1)
    echo "$res"
    local s o
    s=$(echo "$res" | sed -nE 's/.*trial1_tier3=([^ ]+).*/\1/p')
    o=$(echo "$res" | sed -nE 's/.*outcome=([^ ]+).*/\1/p')
    echo -e "${i}\t${batch}\t${t}\t${y}\t${s}\t${o}" >> "$RESULTS"
}

I=0
echo ""; echo "### Batch 1: 5 translations, yaw=0"
for t in $(linspace -0.0215 0.0234 5); do
    I=$((I+1))
    run $I 1 "$t" "0.000000"
done

echo ""; echo "### Batch 2: 3x3 grid"
for t in $(linspace -0.0215 0.0234 3); do
    for y in $(linspace -0.1745 0.1745 3); do
        I=$((I+1))
        run $I 2 "$t" "$y"
    done
done

echo ""; echo "=== DONE $(date) ==="
echo "Results:"
cat "$RESULTS"
