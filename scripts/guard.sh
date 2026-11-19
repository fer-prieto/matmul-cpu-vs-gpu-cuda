#!/usr/bin/env bash
# guard.sh: guardia de raiz compartida por todos los scripts del proyecto.
# Define REPO_ROOT, aborta con exit 2 si el repo no vive bajo /data/FernandoPrieto
# y redirige TMPDIR/XDG_* hacia .sandbox/ (cero escrituras fuera del repo).
# Uso: source "<dir de este archivo>/guard.sh" && run_guard   (primera instruccion)

# cuda_path_setup: si nvcc no esta en PATH, prueba las rutas habituales del
# toolkit y agrega su bin/ SOLO al PATH de este proceso (no instala nada, no
# escribe nada fuera del repo). Devuelve 0 si nvcc queda disponible.
cuda_path_setup() {
    if command -v nvcc >/dev/null 2>&1; then
        return 0
    fi
    local d
    for d in "${CUDA_HOME:-}" /usr/local/cuda-12.2 /usr/local/cuda /usr/local/cuda-12 /opt/cuda; do
        if [ -n "$d" ] && [ -x "$d/bin/nvcc" ]; then
            export PATH="$d/bin:$PATH"
            export CUDA_HOME="$d"
            echo "info: nvcc no estaba en PATH; se usa $d/bin" >&2
            return 0
        fi
    done
    return 1
}

run_guard() {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd -P)"
    case "$REPO_ROOT" in
        /data/FernandoPrieto/*) : ;;
        *)
            echo "error: raiz invalida ($REPO_ROOT): debe vivir bajo /data/FernandoPrieto" >&2
            exit 2
            ;;
    esac
    mkdir -p "$REPO_ROOT/.sandbox/tmp" "$REPO_ROOT/.sandbox/cache" "$REPO_ROOT/.sandbox/config"
    export TMPDIR="$REPO_ROOT/.sandbox/tmp"
    export XDG_CACHE_HOME="$REPO_ROOT/.sandbox/cache"
    export XDG_CONFIG_HOME="$REPO_ROOT/.sandbox/config"
    cuda_path_setup || true
}
