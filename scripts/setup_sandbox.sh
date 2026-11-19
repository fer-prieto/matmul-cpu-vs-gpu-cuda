#!/usr/bin/env bash
# setup_sandbox.sh: crea el sandbox del proyecto de forma idempotente.
# Uso: bash scripts/setup_sandbox.sh
# No borra nada: solo crea lo que falte y el marcador de inicio.
set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/guard.sh"
run_guard

SB="$REPO_ROOT/.sandbox"
created=0
for d in bin state cache tmp config; do
    if [ ! -d "$SB/$d" ]; then
        mkdir -p "$SB/$d"
        created=1
    fi
done
if [ ! -f "$SB/state/inicio" ]; then
    date +%s >"$SB/state/inicio"
    created=1
fi

if [ "$created" -eq 1 ]; then
    echo "OK sandbox creado"
else
    echo "OK sandbox (ya presente)"
fi
