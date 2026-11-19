# U0-T04 — scripts/gpu_env_dump.sh (volcado sanitizado del estado GPU)

- **ID**: U0-T04
- **Depende de**: U0-T01
- **Objetivo**: registrar una fila CSV con el estado de la GPU antes/después de cada corrida, sin datos sensibles.

## Archivos
- Crear: `scripts/gpu_env_dump.sh` (ejecutable, `chmod +x`).

## Especificación
- Primera instrucción: `run_guard` (U0-T01). `set -u`.
- Args: `--moment before|after --run-id <id> --out <ruta.csv> [--extra-key k --extra-val v]`. Uso inválido → usage a stderr, exit 2.
- Emite UNA fila CSV con este header (lo escribe si el archivo no existe) y esquema (§10f del contrato):
  `run_id,moment,index,name,driver_version,memory_total_mib,memory_used_mib,memory_free_mib,compute_cap,clocks_sm_mhz,clocks_mem_mhz,temp_c,power_w,pstate,throttle_reasons,persistence_mode,pcie_gen_max,pcie_gen_current,pcie_width_current,compute_apps_count,compute_apps_used_mib,load1,extra`
- Fuentes exactas:
  - `nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,compute_cap,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,clocks_throttle_reasons.active,persistence_mode,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader` → una línea, `index=0`.
  - Apps ajenas sin PID: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F',' '{c++; u+=$2} END {print c","u+0}'` → `compute_apps_count,compute_apps_used_mib`.
  - `load1` = `awk '{print $1","$2","$3}' /proc/loadavg` (sin el "n/xxx").
  - `extra` = `--extra-key k --extra-val v` → `k=v` (vacio si no se pasa).
- Campos con coma se entrecomillan con `"` (throttle_reasons puede ser `"SW_THERMAL,IDLE"`). PROHIBIDO: `nvidia-smi -L`, `hostname`, `whoami`, IP, UUID, PIDs.
- `nvidia-smi` falla → stderr `FALLO: nvidia-smi no disponible (...)` y **exit 4**.
- Todo lo escrito vive en `$out` (dentro del repo).

## Casos límite
- GPU sin apps → `0,0`. `pstate` `N/A` → dejar vacío.

## Verificación LOCAL (sí ejecutable)
```bash
bash -n scripts/gpu_env_dump.sh
bash scripts/gpu_env_dump.sh --moment before --run-id x --out /tmp/x.csv; echo "rc=$?"   # local: rc=2 (guardia), NO debe crear /tmp/x.csv
test ! -e /tmp/x.csv && echo "OK: nada fuera del repo"
```

## Verificación EN SERVIDOR (la corre el usuario)
```bash
./scripts/gpu_env_dump.sh --moment before --run-id 20260101-000000 --out results/unidad2/env/gpu_test.csv
wc -l results/unidad2/env/gpu_test.csv   # = 2 (header + fila)
head -2 results/unidad2/env/gpu_test.csv # fila sin UUID, sin hostname, sin /home/
```

## Criterios de aceptación
- [ ] `bash -n` limpio; fila con 23 columnas exactas; `wc -l` = 2.
- [ ] Ninguna cadena prohibida en la salida (§14 del contrato).

## Errores típicos
- Usar `nvidia-smi -L` o `-q` completo (trae UUID/serial).
- Guardar PID de procesos ajenos.
- Dejar `load1` con la parte `n/xxx` (nº de procesos/hilos del sistema).

## Commit
`chore: volcado sanitizado del estado de la GPU`
