#!/usr/bin/env bash
# guard.sh: guardia de raiz compartida por todos los scripts del proyecto.
# Define REPO_ROOT, aborta con exit 2 si el repo no vive bajo /data/FernandoPrieto
# y redirige TMPDIR/XDG_* hacia .sandbox/ (cero escrituras fuera del repo).
# Uso: source "<dir de este archivo>/guard.sh" && run_guard   (primera instruccion)

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
}
