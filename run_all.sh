#!/usr/bin/env bash
# run_all.sh: flujo completo de las Unidades 2 y 3 en el servidor.
# Valida el entorno, crea el sandbox, compila, mide, agrega, genera tablas y
# figuras, y al terminar imprime los comandos git (nunca los ejecuta).
#
# Uso: ./run_all.sh [--check|--setup-only] [--unit 2|3|all] [--skip-cpu]
#                   [--use-sudo] [--clean-sandbox] [--dry-run]
# Exit codes: 0 ok · 1 error de etapa · 2 raiz invalida · 3 validacion fallida
#             · 4 entorno insuficiente · 130 interrumpido.
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd -P)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG=""
CURRENT_STAGE=""

usage() {
    cat >&2 <<'EOF'
Uso: ./run_all.sh [opciones]
  --check           verifica el entorno y la sintaxis de los scripts y sale
  --setup-only      verifica el entorno y crea el sandbox
  --unit 2|3|all    unidad(es) a medir (por defecto: all)
  --skip-cpu        no re-mide la linea base en CPU
  --use-sudo        fija los relojes de la GPU (sudo -v interactivo, se revierte)
  --clean-sandbox   borra .sandbox/ antes de empezar
  --dry-run         imprime las etapas y comandos sin ejecutar nada
Exit codes: 0 ok, 1 error, 2 raiz invalida, 3 validacion, 4 entorno, 130 interrumpido.
EOF
}

MODE="run"
UNIT="all"
SKIP_CPU=0
USE_SUDO=0
CLEAN_SANDBOX=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --check) MODE="check" ;;
        --setup-only) MODE="setup" ;;
        --unit)
            UNIT="${2:-}"
            shift || { usage; exit 1; }
            ;;
        --skip-cpu) SKIP_CPU=1 ;;
        --use-sudo) USE_SUDO=1 ;;
        --clean-sandbox) CLEAN_SANDBOX=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "error: opcion desconocida '$1'" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

case "$UNIT" in
    2|3|all) : ;;
    *) echo "error: --unit debe ser 2, 3 o all" >&2; exit 1 ;;
esac
case "$MODE" in
    check|setup|run) : ;;
    *) echo "error: --check y --setup-only son excluyentes" >&2; exit 1 ;;
esac

if [ "$UNIT" = "all" ]; then
    UNITS="2 3"
else
    UNITS="$UNIT"
fi

# Etapas por unidad (orden del contrato).
stages_for() {
    case "$1" in
        2) printf 'build env_before cpu bench summary tables figures env_after runs' ;;
        3) printf 'build env_before cpu bench summary tables figures ncu env_after runs' ;;
    esac
}

stage_command() {
    local unit="$1" stage="$2"
    if [ "$stage" = "build" ]; then
        printf 'make cpu gpu'
    else
        printf 'bash scripts/run_u%s.sh --stage %s --run-id %s' "$unit" "$stage" "$RUN_ID"
        [ "$SKIP_CPU" -eq 1 ] && printf ' --skip-cpu'
        [ "$USE_SUDO" -eq 1 ] && printf ' --use-sudo'
    fi
}

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] guardia omitida (no se ejecuta nada)"
    echo "[dry-run] REPO_ROOT=$REPO_ROOT RUN_ID=$RUN_ID unidad=$UNIT modo=$MODE"
    case "$MODE" in
        check)
            echo "[dry-run] etapa check: bash scripts/check_env.sh"
            echo "[dry-run] etapa check: bash -n scripts/*.sh run_all.sh"
            ;;
        setup)
            echo "[dry-run] etapa check: bash scripts/check_env.sh"
            echo "[dry-run] etapa sandbox: bash scripts/setup_sandbox.sh"
            ;;
        run)
            [ "$CLEAN_SANDBOX" -eq 1 ] && echo "[dry-run] rm -rf $REPO_ROOT/.sandbox"
            echo "[dry-run] etapa check: bash scripts/check_env.sh"
            echo "[dry-run] etapa sandbox: bash scripts/setup_sandbox.sh"
            for u in $UNITS; do
                for s in $(stages_for "$u"); do
                    if [ "$s" = "cpu" ] && [ "$SKIP_CPU" -eq 1 ] && [ "$MODE" = "run" ]; then
                        echo "[dry-run] etapa unit${u}_${s}: omitida (--skip-cpu)"
                        continue
                    fi
                    echo "[dry-run] etapa unit${u}_${s}: $(stage_command "$u" "$s")"
                done
            done
            ;;
    esac
    exit 0
fi

# --- Guardia de raiz (contrato SS8) -----------------------------------------
case "$REPO_ROOT" in
    /data/FernandoPrieto/*) : ;;
    *)
        echo "error: raiz invalida ($REPO_ROOT): debe vivir bajo /data/FernandoPrieto" >&2
        exit 2
        ;;
esac

if [ "$CLEAN_SANDBOX" -eq 1 ]; then
    rm -rf "$REPO_ROOT/.sandbox"
    echo "OK sandbox limpio (--clean-sandbox)"
fi

mkdir -p "$REPO_ROOT/logs"
LOG="$REPO_ROOT/logs/run_all_${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

export TMPDIR="$REPO_ROOT/.sandbox/tmp"
export XDG_CACHE_HOME="$REPO_ROOT/.sandbox/cache"
export XDG_CONFIG_HOME="$REPO_ROOT/.sandbox/config"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$REPO_ROOT/.sandbox/state"

trap 'echo; echo "Interrumpido en la etapa ${CURRENT_STAGE:-?}"; exit 130' INT
trap 'rc=$?; if [ "$rc" -ne 0 ] && [ "$rc" -ne 130 ]; then sheet="$REPO_ROOT/logs/FALLO_${RUN_ID}.md"; { echo "# Fallo en ${CURRENT_STAGE:-?}"; echo; echo "- etapa: ${CURRENT_STAGE:-?}"; echo "- comando: ${CURRENT_COMMAND:-?}"; echo "- exit code: $rc"; echo "- hipotesis: revisar el stderr adjunto"; echo; echo "## Ultimas 40 lineas del log"; echo; echo "******"; tail -n 40 "$LOG" 2>/dev/null; echo "******"; } > "$sheet"; echo "Resumen de fallo: logs/FALLO_${RUN_ID}.md" >&2; fi' EXIT

echo "run_all.sh · run_id=$RUN_ID · unidad=$UNIT · skip_cpu=$SKIP_CPU · use_sudo=$USE_SUDO"
echo "repo: $REPO_ROOT"
echo "log:  $LOG"

CURRENT_COMMAND=""
run_cmd() {
    CURRENT_COMMAND="$1"
    shift
    "$@"
}

marker_done() { [ -f "$REPO_ROOT/.sandbox/state/$1.ok" ]; }
marker_set() { date >"$REPO_ROOT/.sandbox/state/$1.ok"; }

run_global_stage() {
    local stage="$1"
    CURRENT_STAGE="$stage"
    case "$stage" in
        check)
            echo "== etapa check =="
            run_cmd "scripts/check_env.sh" bash "$REPO_ROOT/scripts/check_env.sh" || exit 4
            local f
            for f in "$REPO_ROOT"/scripts/*.sh "$REPO_ROOT"/run_all.sh; do
                run_cmd "bash -n $f" bash -n "$f" || exit 1
            done
            echo "OK check"
            ;;
        sandbox)
            if marker_done sandbox; then
                echo "OK (ya presente) sandbox"
                return 0
            fi
            echo "== etapa sandbox =="
            run_cmd "scripts/setup_sandbox.sh" bash "$REPO_ROOT/scripts/setup_sandbox.sh" || exit 1
            marker_set sandbox
            ;;
    esac
}

run_unit_stage() {
    local unit="$1" stage="$2"
    local marker="unit${unit}_${stage}"
    CURRENT_STAGE="$marker"

    if [ "$stage" = "cpu" ] && [ "$SKIP_CPU" -eq 1 ]; then
        echo "OK (--skip-cpu) $marker"
        return 0
    fi
    if marker_done "$marker"; then
        echo "OK (ya presente) $marker"
        return 0
    fi

    echo "== etapa $marker =="
    local -a cmd
    if [ "$stage" = "build" ]; then
        cmd=(make cpu gpu)
        CURRENT_COMMAND="make cpu gpu"
        (cd "$REPO_ROOT" && "${cmd[@]}")
    else
        cmd=(bash "$REPO_ROOT/scripts/run_u${unit}.sh" --stage "$stage" --run-id "$RUN_ID")
        [ "$SKIP_CPU" -eq 1 ] && cmd+=(--skip-cpu)
        [ "$USE_SUDO" -eq 1 ] && cmd+=(--use-sudo)
        CURRENT_COMMAND="scripts/run_u${unit}.sh --stage ${stage}"
        "${cmd[@]}"
    fi
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        case "$rc" in
            3|4) exit "$rc" ;;
            *) exit 1 ;;
        esac
    fi
    marker_set "$marker"
}

case "$MODE" in
    check)
        run_global_stage check
        echo "OK check"
        exit 0
        ;;
    setup)
        run_global_stage check
        run_global_stage sandbox
        echo "OK setup"
        exit 0
        ;;
esac

run_global_stage check
run_global_stage sandbox

for u in $UNITS; do
    for s in $(stages_for "$u"); do
        run_unit_stage "$u" "$s"
    done
done

# --- Mensaje final (nunca ejecuta git) --------------------------------------
figs_state() {
    if ls "$REPO_ROOT/results/unidad$1/figuras"/*.png >/dev/null 2>&1; then
        printf 'OK'
    else
        printf 'pendiente local'
    fi
}

echo
echo "RESUMEN: corridas OK. Validacion: OK. Figuras U2: $(figs_state 2) · U3: $(figs_state 3)."
echo
echo "Comandos git exactos (ejecutalos tu, nunca el script):"
echo "  git add results/unidad2 results/unidad3"
echo '  git commit -m "feat: se agregan resultados GPU de las unidades 2 y 3"'
echo "  git push"
echo
echo "  # logs/ no se versiona. Para publicar un fallo: revisar logs/FALLO_${RUN_ID}.md"
echo "  # Figuras pendientes (si aplica):"
echo "  #   python3 scripts/make_figures.py --unit 2 --summary results/unidad2/all_summary.csv --out results/unidad2/figuras/"
echo "  #   python3 scripts/make_figures.py --unit 3 --summary results/unidad3/all_summary.csv --out results/unidad3/figuras/"
