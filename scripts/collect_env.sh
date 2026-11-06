#!/usr/bin/env bash
# =============================================================================
#  collect_env.sh: volcado MANUAL del entorno de prueba.
#
#  Es un respaldo legible de lo que el arnés guarda automáticamente en
#  results/unidad1/env_<run>.json. Sirve para la tabla "Entorno de prueba" del
#  reporte y para dejar constancia del estado de la máquina antes de medir.
#
#  Uso:  bash scripts/collect_env.sh        (solo muestra el volcado)
#        make env                           (además lo guarda en results/unidad1/)
# =============================================================================
set -u
shopt -s nullglob

section() { printf '\n===== %s =====\n' "$1"; }

# Ejecuta un comando si existe; si no, lo indica.
run() {
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1
    else
        echo "($1 no disponible)"
    fi
}

# Muestra "ruta: contenido" de archivos legibles (contenido en una línea).
show() {
    local f
    for f in "$@"; do
        [ -r "$f" ] && printf '%s: %s\n' "$f" "$(tr '\n' ' ' < "$f" | sed 's/ *$//')"
    done
}

echo "Volcado del entorno de prueba: $(date --iso-8601=seconds)"
echo "Directorio: $(pwd)"

section "Sistema"
run uname -a
show /proc/sys/kernel/hostname
[ -r /etc/os-release ] && cat /etc/os-release

section "CPU (lscpu)"
run lscpu

section "Topología: CPU lógica -> núcleo (lscpu -e)"
run lscpu -e

section "Cachés (lscpu -C)"
run lscpu -C

section "Memoria"
run free -h
grep -E 'MemTotal|MemAvailable|SwapTotal|Hugepagesize' /proc/meminfo
show /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag

section "Frecuencia y gobernador por CPU"
for c in /sys/devices/system/cpu/cpu[0-9]*; do
    f="$c/cpufreq"
    [ -d "$f" ] || continue
    printf '%s: gobernador=%s driver=%s actual=%s kHz min=%s max=%s (hw %s-%s)\n' \
        "$(basename "$c")" "$(cat "$f/scaling_governor" 2>/dev/null)" \
        "$(cat "$f/scaling_driver" 2>/dev/null)" "$(cat "$f/scaling_cur_freq" 2>/dev/null)" \
        "$(cat "$f/scaling_min_freq" 2>/dev/null)" "$(cat "$f/scaling_max_freq" 2>/dev/null)" \
        "$(cat "$f/cpuinfo_min_freq" 2>/dev/null)" "$(cat "$f/cpuinfo_max_freq" 2>/dev/null)"
done
show /sys/devices/system/cpu/intel_pstate/status /sys/devices/system/cpu/intel_pstate/no_turbo \
     /sys/devices/system/cpu/intel_pstate/min_perf_pct /sys/devices/system/cpu/intel_pstate/max_perf_pct \
     /sys/devices/system/cpu/cpufreq/boost /sys/devices/system/cpu/smt/active

section "Throttling térmico (contadores acumulados desde el arranque)"
for c in /sys/devices/system/cpu/cpu[0-9]*; do
    t="$c/thermal_throttle"
    [ -d "$t" ] || continue
    printf '%s:' "$(basename "$c")"
    for f in "$t"/*; do printf ' %s=%s' "$(basename "$f")" "$(cat "$f" 2>/dev/null)"; done
    echo
done

section "Temperaturas y ventiladores"
if command -v sensors >/dev/null 2>&1; then
    sensors 2>&1
else
    for h in /sys/class/hwmon/hwmon*; do
        echo "$h ($(cat "$h/name" 2>/dev/null)):"
        for s in "$h"/temp*_input "$h"/fan*_input; do
            printf '  %s %s = %s\n' "$(basename "$s")" "$(cat "${s%_input}_label" 2>/dev/null)" "$(cat "$s" 2>/dev/null)"
        done
    done
fi

section "Alimentación"
for p in /sys/class/power_supply/*; do
    printf '%s: tipo=%s online=%s estado=%s carga=%s%%\n' "$(basename "$p")" \
        "$(cat "$p/type" 2>/dev/null)" "$(cat "$p/online" 2>/dev/null)" \
        "$(cat "$p/status" 2>/dev/null)" "$(cat "$p/capacity" 2>/dev/null)"
done

section "Carga del sistema"
show /proc/loadavg
run uptime

section "Compiladores y herramientas"
for tool in g++ clang++ make rustc cargo git ar; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf '%-8s %s\n' "$tool" "$("$tool" --version 2>&1 | head -n 1)"
    else
        printf '%-8s (no disponible)\n' "$tool"
    fi
done

section "Qué activa -march=native (GCC)"
if command -v g++ >/dev/null 2>&1; then
    g++ -march=native -Q --help=target 2>/dev/null | grep -E '^\s+-m(arch|tune)=' | sed 's/^ *//'
    echo "Macros SIMD: $(g++ -march=native -dM -E - </dev/null 2>/dev/null | grep -oE '__(SSE[0-9_]*|SSSE3|AVX[0-9A-Z_]*|FMA)__' | sort -u | tr '\n' ' ')"
fi

section "OpenBLAS (opcional)"
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openblas 2>/dev/null; then
    echo "pkg-config openblas: $(pkg-config --modversion openblas) | $(pkg-config --cflags --libs openblas)"
else
    echo "pkg-config no encuentra openblas"
fi
ls -la /usr/lib/libopenblas* /usr/lib64/libopenblas* /usr/lib/x86_64-linux-gnu/libopenblas* 2>/dev/null

section "Repositorio"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "commit: $(git rev-parse HEAD)"
    echo "rama:   $(git rev-parse --abbrev-ref HEAD)"
    echo "describe: $(git describe --tags --always --dirty 2>/dev/null)"
    echo "cambios sin commit:"; git status --short
else
    echo "(no es un repositorio git)"
fi

section "Kernel: línea de arranque y mitigaciones"
show /proc/cmdline
for v in /sys/devices/system/cpu/vulnerabilities/*; do
    printf '%s: %s\n' "$(basename "$v")" "$(cat "$v" 2>/dev/null)"
done
