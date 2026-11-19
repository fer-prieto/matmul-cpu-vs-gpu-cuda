#!/usr/bin/env bash
# check_env.sh: verifica el entorno del servidor. Solo lectura: no instala nada.
# Uso: bash scripts/check_env.sh
# Exit codes: 0 = todo OK · 4 = falta algo (mensaje accionable por cada fallo).
set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/guard.sh"
run_guard

oks=0
fails=0

pass() { printf 'OK    %s\n' "$1"; oks=$((oks + 1)); }
fallo() {
    printf 'FALLO %s: %s\n' "$1" "$2" >&2
    fails=$((fails + 1))
}

# 1. bash >= 4
if [ "${BASH_VERSINFO[0]:-0}" -ge 4 ]; then
    pass "bash ${BASH_VERSION%%(*}"
else
    fallo "bash" "se requiere bash >= 4 (instala bash 4.x o superior)"
fi

# 2-4. nvcc y soporte de sm_60
if command -v nvcc >/dev/null 2>&1; then
    pass "nvcc ($(command -v nvcc))"
    nvcc_ver="$(nvcc --version 2>/dev/null | tail -n 1)"
    if printf '%s' "$nvcc_ver" | grep -qE 'release (1[2-9]|[2-9][0-9])\.'; then
        pass "nvcc version ($nvcc_ver)"
    else
        fallo "nvcc version" "se requiere CUDA >= 12.0 (detectado: ${nvcc_ver:-?})"
    fi
    if nvcc --list-gpu-arch 2>/dev/null | grep -qw 'compute_60'; then
        pass "nvcc soporta compute_60"
    else
        fallo "nvcc compute_60" "este toolkit no soporta sm_60 (instala CUDA 11.x/12.x)"
    fi
else
    fallo "nvcc" "no esta en PATH (agrega el CUDA toolkit al PATH)"
    fallo "nvcc version" "sin nvcc no se puede compilar"
    fallo "nvcc compute_60" "sin nvcc no se puede compilar"
fi

# 5. GPU visible y compatible
if command -v nvidia-smi >/dev/null 2>&1 &&
    gpu_line="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | head -n 1)" &&
    [ -n "$gpu_line" ]; then
    gpu_cc="$(printf '%s' "$gpu_line" | cut -d, -f2 | tr -d ' ')"
    gpu_mem="$(printf '%s' "$gpu_line" | cut -d, -f3 | tr -d ' MiB')"
    if [ "$gpu_cc" = "6.0" ]; then
        pass "GPU compute capability $gpu_cc"
    else
        fallo "GPU compute capability" "se esperaba 6.0 y hay ${gpu_cc}; ajusta CUDA_ARCH en el Makefile"
    fi
    if [ "${gpu_mem:-0}" -ge 10000 ] 2>/dev/null; then
        pass "GPU memoria ${gpu_mem} MiB"
    else
        fallo "GPU memoria" "se requieren >= 10000 MiB libres en el dispositivo (hay ${gpu_mem:-?})"
    fi
else
    fallo "nvidia-smi" "no disponible o sin GPU visible (revisa el driver NVIDIA)"
fi

# 6. g++ >= 11
if command -v g++ >/dev/null 2>&1; then
    gxx_major="$(g++ -dumpversion 2>/dev/null | cut -d. -f1)"
    if [ "${gxx_major:-0}" -ge 11 ] 2>/dev/null; then
        pass "g++ ${gxx_major}"
    else
        fallo "g++" "se requiere g++ >= 11 (hay ${gxx_major:-?})"
    fi
else
    fallo "g++" "no esta en PATH (instala g++)"
fi

# 7. utilidades de build y agregacion
for tool in make awk sort date; do
    if command -v "$tool" >/dev/null 2>&1; then
        pass "$tool"
    else
        fallo "$tool" "no esta en PATH"
    fi
done
if command -v python3 >/dev/null 2>&1 &&
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    pass "python3 $(python3 -V 2>&1 | cut -d' ' -f2)"
else
    fallo "python3" "se requiere python3 >= 3.8 (instala python3)"
fi

# 8. espacio libre en el sistema de archivos del repo
free_mib="$(df -P --block-size=1M "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
if [ "${free_mib:-0}" -ge 5000 ] 2>/dev/null; then
    pass "espacio libre ${free_mib} MiB"
else
    fallo "espacio libre" "se requieren >= 5000 MiB en $REPO_ROOT (hay ${free_mib:-?})"
fi

# 9. escritura dentro del repo
if tmpdir="$(mktemp -d "$REPO_ROOT/.sandbox.write.XXXXXX" 2>/dev/null)"; then
    rmdir "$tmpdir"
    pass "escritura en el repo"
else
    fallo "escritura en el repo" "no se puede escribir en $REPO_ROOT"
fi

printf '\n%d OK, %d fallos\n' "$oks" "$fails"
[ "$fails" -eq 0 ] || exit 4
exit 0
