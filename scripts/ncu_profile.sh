#!/usr/bin/env bash
# ncu_profile.sh: perfil de conflictos de banco con Nsight Compute (OPCIONAL).
# Nunca aborta la corrida: si ncu no esta o no hay permisos de contadores,
# informa y sale con 0.
# Uso: bash scripts/ncu_profile.sh [--impls a,b] [--sizes "1024 2048"] [--out DIR]
set -u -o pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/guard.sh"
run_guard

IMPLS="gpu_tiled,gpu_tiled_pad"
SIZES="1024 2048 4096"
OUT="$REPO_ROOT/results/unidad3/ncu"

usage() {
    cat >&2 <<'EOF'
Uso: ncu_profile.sh [--impls gpu_tiled,gpu_tiled_pad] [--sizes "1024 2048 4096"] [--out DIR]
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --impls) IMPLS="${2:-}"; shift 2 || { usage; exit 2; } ;;
        --sizes) SIZES="${2:-}"; shift 2 || { usage; exit 2; } ;;
        --out) OUT="${2:-}"; shift 2 || { usage; exit 2; } ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu no disponible: etapa omitida" >&2
    exit 0
fi

mkdir -p "$OUT"
bin="$REPO_ROOT/build/bench_gpu"
if [ ! -x "$bin" ]; then
    echo "build/bench_gpu no existe: etapa ncu omitida (compila con make gpu)" >&2
    exit 0
fi

impl_ok=0
for impl in $(printf '%s' "$IMPLS" | tr ',' ' '); do
    for n in $SIZES; do
        log="$OUT/ncu_${impl}_${n}.txt"
        rep="$REPO_ROOT/.sandbox/tmp/ncu_${impl}_${n}"
        errfile="$REPO_ROOT/.sandbox/tmp/ncu_${impl}_${n}.err"
        (
            cd "$REPO_ROOT/.sandbox/tmp" &&
                ncu --section MemoryWorkloadAnalysis -f --log-file "$log" -o "$rep" \
                    "$bin" --unit 3 --impls "$impl" --sizes "$n" --warmup 1 --reps 1 --no-validate
        ) >/dev/null 2>"$errfile"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "ncu ${impl} N=${n}: OK"
            impl_ok=$((impl_ok + 1))
        elif grep -qiE 'ERR_NVGPUCTRPERM|permission' "$errfile"; then
            rm -f "$log"
            echo "ncu sin permisos de contadores: etapa omitida" >&2
            exit 0
        else
            rm -f "$log"
            echo "ncu ${impl} N=${n}: omitido (rc=$rc); ultimas lineas:" >&2
            tail -n 10 "$errfile" >&2 || true
        fi
        rm -f "$errfile"
    done
done

echo "ncu: ${impl_ok} perfiles generados en $OUT"
exit 0
