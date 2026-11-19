#!/usr/bin/env bash
# run_u3.sh: corrida de la Unidad 3 (tiling, padding y cuBLAS como referencia).
# Etapas: build | env_before | cpu | bench | summary | tables | figures | ncu |
#         env_after | runs   (o "all" para todas, en ese orden).
# Uso: bash scripts/run_u3.sh [--stage ETAPA] [--run-id ID] [--skip-cpu] [--use-sudo]
set -u -o pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/guard.sh"
run_guard

UNIT=3
OUT="$REPO_ROOT/results/unidad$UNIT"
STAGE="all"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
SKIP_CPU=0
USE_SUDO=0
REPS=5
WARMUP=1
SEED=42
TOLERANCE=1e-4
SIZES="1024 2048 4096"
LOCKED_CLOCKS=""
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

usage() {
    cat >&2 <<'EOF'
Uso: run_u3.sh [--stage all|build|env_before|cpu|bench|summary|tables|figures|ncu|env_after|runs]
               [--run-id ID] [--skip-cpu] [--use-sudo]
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="${2:-}"; shift 2 || { usage; exit 2; } ;;
        --run-id) RUN_ID="${2:-}"; shift 2 || { usage; exit 2; } ;;
        --skip-cpu) SKIP_CPU=1; shift ;;
        --use-sudo) USE_SUDO=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

case "$STAGE" in
    all|build|env_before|cpu|bench|summary|tables|figures|ncu|env_after|runs) : ;;
    *) echo "error: --stage desconocido ($STAGE)" >&2; usage; exit 2 ;;
esac

mkdir -p "$OUT/baseline" "$OUT/env" "$OUT/tablas" "$OUT/figuras" "$OUT/ncu"

# --- Sudo opcional: fija los relojes de la GPU y los revierte al salir -------
lock_clocks() {
    [ "$USE_SUDO" -eq 1 ] || return 0
    [ -n "$LOCKED_CLOCKS" ] && return 0
    if ! sudo -v; then
        echo "error: sudo -v fallo; corre sin --use-sudo" >&2
        exit 1
    fi
    local line sm mem
    line="$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/ MHz//g; s/ //g')"
    sm="${line%%,*}"
    mem="${line##*,}"
    if [ -z "$sm" ] || [ -z "$mem" ]; then
        echo "error: no se pudieron leer los relojes actuales de la GPU" >&2
        exit 1
    fi
    sudo nvidia-smi -lgc "$sm" -lmc "$mem" >/dev/null || {
        echo "error: no se pudieron fijar los relojes (nvidia-smi -lgc/-lmc)" >&2
        exit 1
    }
    LOCKED_CLOCKS="${sm},${mem}"
    trap 'sudo nvidia-smi -rgc >/dev/null 2>&1; sudo nvidia-smi -rmc >/dev/null 2>&1' EXIT
    echo "OK relojes fijados (sm=${sm} MHz, mem=${mem} MHz); se revierten al salir"
}

# --- Etapas ------------------------------------------------------------------
stage_build() {
    echo "== etapa build =="
    (cd "$REPO_ROOT" && make cpu gpu)
}

stage_env() {
    local moment="$1"
    echo "== etapa env_${moment} =="
    if [ -n "$LOCKED_CLOCKS" ]; then
        bash "$REPO_ROOT/scripts/gpu_env_dump.sh" --moment "$moment" --run-id "$RUN_ID" \
            --out "$OUT/env/gpu_${RUN_ID}_${moment}.csv" \
            --extra-key locked_clocks --extra-val "$LOCKED_CLOCKS"
    else
        bash "$REPO_ROOT/scripts/gpu_env_dump.sh" --moment "$moment" --run-id "$RUN_ID" \
            --out "$OUT/env/gpu_${RUN_ID}_${moment}.csv"
    fi
    nvcc --version 2>/dev/null | tail -n 1 >"$OUT/env/build_${RUN_ID}.txt"
    "$REPO_ROOT/build/bench_gpu" --device-info >>"$OUT/env/build_${RUN_ID}.txt" 2>/dev/null || true
}

stage_cpu() {
    echo "== etapa cpu =="
    if ls "$OUT"/baseline/cpu_summary_*.csv >/dev/null 2>&1; then
        echo "OK (ya presente) baseline CPU"
        return 0
    fi
    # El baseline es del mismo servidor: si la Unidad 2 ya lo midio, se reutiliza.
    local prev
    prev="$(ls -1t "$REPO_ROOT"/results/unidad2/baseline/cpu_summary_*.csv 2>/dev/null | head -n 1)"
    if [ -n "$prev" ]; then
        cp "$prev" "$OUT/baseline/"
        cp "${prev%cpu_summary_*}cpu_raw_"*.csv "$OUT/baseline/" 2>/dev/null || true
        echo "OK baseline CPU reutilizado de la Unidad 2 ($(basename "$prev"))"
        return 0
    fi
    if [ "$SKIP_CPU" -eq 1 ]; then
        echo "error: no hay cpu_summary_*.csv en $OUT/baseline ni en results/unidad2 y se pidio --skip-cpu" >&2
        exit 1
    fi
    "$REPO_ROOT/build/cpu_baseline" --warmup 1 --reps 5 --csv "$OUT/baseline/cpu_raw_${RUN_ID}.csv" \
        $SIZES >"$OUT/baseline/cpu_table_${RUN_ID}.txt"
    python3 "$REPO_ROOT/scripts/summarize.py" --input "$OUT/baseline/cpu_raw_${RUN_ID}.csv" \
        --mode cpu --out "$OUT/baseline/cpu_summary_${RUN_ID}.csv"
}

baseline_file() {
    ls -1t "$OUT"/baseline/cpu_summary_*.csv 2>/dev/null | head -n 1
}

stage_bench() {
    echo "== etapa bench =="
    "$REPO_ROOT/build/bench_gpu" --unit "$UNIT" --warmup "$WARMUP" --reps "$REPS" --seed "$SEED" \
        --tolerance "$TOLERANCE" --run-id "$RUN_ID" \
        --validation-file "$OUT/validation_${RUN_ID}.csv" $SIZES >"$OUT/raw_${RUN_ID}.csv"
}

stage_summary() {
    echo "== etapa summary =="
    local base
    base="$(baseline_file)"
    python3 "$REPO_ROOT/scripts/summarize.py" --input "$OUT/raw_${RUN_ID}.csv" --mode gpu \
        --baseline "$base" --run-id "$RUN_ID" --out "$OUT/summary_${RUN_ID}.csv" >/dev/null
    # Consolidado: cabecera una vez y todas las corridas en orden cronologico.
    local first
    first="$(ls -1 "$OUT"/summary_*.csv | sort | head -n 1)"
    head -n 1 "$first" >"$OUT/all_summary.csv"
    local f
    for f in $(ls -1 "$OUT"/summary_*.csv | sort); do
        tail -n +2 "$f" >>"$OUT/all_summary.csv"
    done
    echo "OK $OUT/all_summary.csv"
}

stage_tables() {
    echo "== etapa tables =="
    python3 "$REPO_ROOT/scripts/make_tables.py" --unit "$UNIT" --summary "$OUT/all_summary.csv" \
        --validation "$OUT/validation_${RUN_ID}.csv" --out "$OUT/tablas"
}

stage_ncu() {
    echo "== etapa ncu =="
    bash "$REPO_ROOT/scripts/ncu_profile.sh" --impls gpu_tiled,gpu_tiled_pad --sizes "$SIZES" \
        --out "$OUT/ncu"
}

stage_figures() {
    echo "== etapa figures =="
    local rc=0
    python3 "$REPO_ROOT/scripts/make_figures.py" --unit "$UNIT" --summary "$OUT/all_summary.csv" \
        --out "$OUT/figuras" || rc=$?
    if [ "$rc" -eq 0 ]; then
        return 0
    fi
    if [ "$rc" -ne 4 ]; then
        return "$rc"
    fi
    echo "matplotlib no disponible: se intenta un venv en .sandbox/" >&2
    if ! python3 -m venv "$REPO_ROOT/.sandbox/venv" >/dev/null 2>&1; then
        echo "figuras: pendiente local (sin venv: se generan tras el git pull)" >&2
        return 0
    fi
    if ! "$REPO_ROOT/.sandbox/venv/bin/pip" install -q matplotlib==3.8.4 numpy==1.26.4 >/dev/null 2>&1; then
        echo "figuras: pendiente local (pip sin red: se generan tras el git pull)" >&2
        return 0
    fi
    "$REPO_ROOT/.sandbox/venv/bin/pip" freeze >"$OUT/env/venv_freeze_${RUN_ID}.txt" 2>/dev/null
    "$REPO_ROOT/.sandbox/venv/bin/python" "$REPO_ROOT/scripts/make_figures.py" --unit "$UNIT" \
        --summary "$OUT/all_summary.csv" --out "$OUT/figuras" ||
        echo "figuras: pendiente local (fallo en el venv)" >&2
}

stage_runs() {
    echo "== etapa runs =="
    local file="$OUT/runs.csv"
    local val_file="$OUT/validation_${RUN_ID}.csv"
    if [ ! -f "$val_file" ]; then
        val_file="$(ls -1t "$OUT"/validation_*.csv 2>/dev/null | head -n 1)"
    fi
    if [ -z "$val_file" ] || [ ! -f "$val_file" ]; then
        echo "error: no hay validation_*.csv en $OUT; corre antes la etapa bench" >&2
        exit 1
    fi
    # La corrida registrada es la que produjo los datos, no necesariamente la
    # actual: las etapas pueden reanudarse con otro run_id.
    local data_run started
    data_run="$(basename "$val_file" .csv)"
    data_run="${data_run#validation_}"
    if [ -f "$OUT/raw_${data_run}.csv" ]; then
        started="$(date -u -r "$OUT/raw_${data_run}.csv" +%Y-%m-%dT%H:%M:%SZ)"
    else
        started="$STARTED_AT"
    fi
    if [ -f "$file" ] && awk -F',' -v id="$data_run" 'NR>1 && $1==id {found=1} END {exit !found}' "$file"; then
        echo "OK (ya registrado) corrida $data_run"
        return 0
    fi
    if [ ! -f "$file" ]; then
        echo "run_id,label,status,started_at,finished_at,unit,impls,sizes,reps,warmup,seed,tolerance,all_valid,gpu_name,gpu_cc,gpu_driver,nvcc_version,clocks_locked,notes" >"$file"
    fi
    local info name cc driver nvcc_version all_valid impls
    info="$("$REPO_ROOT/build/bench_gpu" --device-info 2>/dev/null | head -n 1)"
    name="$(printf '%s' "$info" | sed -n 's/.*name=\([^;]*\).*/\1/p')"
    cc="$(printf '%s' "$info" | sed -n 's/.*cc=\([^;]*\).*/\1/p')"
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
    nvcc_version="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -n 1 | awk '{print $2}')"
    all_valid="$(awk -F',' 'NR>1 {gsub(/"/,"",$11); if ($11 != "true") bad=1} END {print bad ? "false" : "true"}' "$val_file")"
    impls="$(awk -F',' 'NR>1 {print $2}' "$val_file" | sort -u | paste -sd';' -)"
    local locked="no"
    [ -n "$LOCKED_CLOCKS" ] && locked="yes"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$data_run" "" "completed" "$started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$UNIT" \
        "$impls" "$(printf '%s' "$SIZES" | tr ' ' ';')" "$REPS" "$WARMUP" "$SEED" "$TOLERANCE" \
        "$all_valid" "$name" "$cc" "$driver" "$nvcc_version" "$locked" "" >>"$file"
    echo "OK $file"
}

run_stage() {
    case "$1" in
        build) stage_build ;;
        env_before) lock_clocks; stage_env before ;;
        cpu) lock_clocks; stage_cpu ;;
        bench) lock_clocks; stage_bench ;;
        summary) stage_summary ;;
        tables) stage_tables ;;
        figures) stage_figures ;;
        ncu) stage_ncu ;;
        env_after) stage_env after ;;
        runs) stage_runs ;;
    esac
}

if [ "$STAGE" = "all" ]; then
    for s in build env_before cpu bench summary tables figures ncu env_after runs; do
        run_stage "$s"
        rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "FALLO en la etapa $s (rc=$rc)" >&2
            exit "$rc"
        fi
    done
else
    run_stage "$STAGE"
    rc=$?
    [ "$rc" -eq 0 ] || exit "$rc"
fi
