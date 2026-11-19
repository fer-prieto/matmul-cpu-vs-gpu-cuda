#!/usr/bin/env bash
# gpu_env_dump.sh: volcado sanitizado del estado de la GPU antes/despues de una corrida.
# Uso: bash scripts/gpu_env_dump.sh --moment before|after --run-id ID --out RUTA.csv
#                                  [--extra-key k --extra-val v]
# Nunca escribe UUID, serial, hostname, IP, usuario ni PIDs.
set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/guard.sh"
run_guard

HEADER="run_id,moment,index,name,driver_version,memory_total_mib,memory_used_mib,memory_free_mib,compute_cap,clocks_sm_mhz,clocks_mem_mhz,temp_c,power_w,pstate,throttle_reasons,persistence_mode,pcie_gen_max,pcie_gen_current,pcie_width_current,compute_apps_count,compute_apps_used_mib,load1,extra"

usage() {
    cat >&2 <<'EOF'
Uso: gpu_env_dump.sh --moment before|after --run-id ID --out RUTA.csv [--extra-key k --extra-val v]
EOF
}

MOMENT=""
RUN_ID=""
OUT=""
EXTRA_KEY=""
EXTRA_VAL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --moment) MOMENT="${2:-}"; shift 2 || { usage; exit 2; } ;;
        --run-id) RUN_ID="${2:-}"; shift 2 || { usage; exit 2; } ;;
        --out) OUT="${2:-}"; shift 2 || { usage; exit 2; } ;;
        --extra-key) EXTRA_KEY="${2:-}"; shift 2 || { usage; exit 2; } ;;
        --extra-val) EXTRA_VAL="${2:-}"; shift 2 || { usage; exit 2; } ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

case "$MOMENT" in
    before|after) : ;;
    *) echo "error: --moment debe ser before o after" >&2; usage; exit 2 ;;
esac
[ -n "$RUN_ID" ] || { echo "error: falta --run-id" >&2; usage; exit 2; }
[ -n "$OUT" ] || { echo "error: falta --out" >&2; usage; exit 2; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FALLO: nvidia-smi no disponible (revisa el driver NVIDIA)" >&2
    exit 4
fi
if ! gpu="$(nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,compute_cap,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,clocks_throttle_reasons.active,persistence_mode,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader 2>/dev/null | head -n 1)" || [ -z "$gpu" ]; then
    echo "FALLO: nvidia-smi no devolvio datos de la GPU" >&2
    exit 4
fi

# Se normalizan unidades; se quitan comillas para recomponer el CSV.
gpu="$(printf '%s' "$gpu" | sed 's/ MiB//g; s/ MHz//g; s/ W//g; s/"//g')"
IFS=',' read -r idx name drv mem_total mem_used mem_free cc clk_sm clk_mem temp power pstate thr persist pcie_max pcie_cur pcie_w <<<"$gpu"
trim() { printf '%s' "$1" | sed 's/^ *//; s/ *$//'; }
idx="$(trim "$idx")"; name="$(trim "$name")"; drv="$(trim "$drv")"
mem_total="$(trim "$mem_total")"; mem_used="$(trim "$mem_used")"; mem_free="$(trim "$mem_free")"
cc="$(trim "$cc")"; clk_sm="$(trim "$clk_sm")"; clk_mem="$(trim "$clk_mem")"; temp="$(trim "$temp")"
power="$(trim "$power")"; pstate="$(trim "$pstate")"; thr="$(trim "$thr")"; persist="$(trim "$persist")"
pcie_max="$(trim "$pcie_max")"; pcie_cur="$(trim "$pcie_cur")"; pcie_w="$(trim "$pcie_w")"
idx="$(trim "$idx")"; mem_used="$(trim "$mem_used")"

# Procesos ajenos: solo el conteo y la memoria total usada (sin PIDs).
apps_csv="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
    awk -F',' '{c++; m=$2; gsub(/ MiB| /,"",m); u+=m} END {printf "%d,%d", c+0, u+0}')"
case "$apps_csv" in
    *,*) apps_count="${apps_csv%%,*}"; apps_mib="${apps_csv##*,}" ;;
    *) apps_count=0; apps_mib=0 ;;
esac

load1="$(awk '{print $1" "$2" "$3}' /proc/loadavg)"

if [ -n "$EXTRA_KEY" ]; then
    extra="${EXTRA_KEY}=${EXTRA_VAL}"
else
    extra=""
fi
# Ningun campo puede contener comas (el CSV se lee con separador ',').
thr="$(printf '%s' "$thr" | tr ',' ';')"
extra="$(printf '%s' "$extra" | tr ',' ';')"

outdir="$(dirname "$OUT")"
mkdir -p "$outdir"
if [ ! -f "$OUT" ]; then
    printf '%s\n' "$HEADER" >"$OUT"
fi

printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$RUN_ID" "$MOMENT" "$idx" "$name" "$drv" "$mem_total" "$mem_used" "$mem_free" "$cc" \
    "$clk_sm" "$clk_mem" "$temp" "$power" "$pstate" "$thr" "$persist" "$pcie_max" \
    "$pcie_cur" "$pcie_w" "$apps_count" "$apps_mib" "$load1" "$extra" >>"$OUT"

echo "OK gpu_env_dump $MOMENT ($OUT)"
