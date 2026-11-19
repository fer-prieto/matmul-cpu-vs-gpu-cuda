# U0-T02 — scripts/check_env.sh (verificación de entorno, solo lectura)

- **ID**: U0-T02
- **Depende de**: U0-T01
- **Objetivo**: comprobar el entorno del servidor y fallar rápido con mensaje accionable (exit 4) si falta algo.

## Archivos
- Crear: `scripts/check_env.sh` (ejecutable, `chmod +x`).

## Especificación
- Primera instrucción: `source .../guard.sh; run_guard` (patrón U0-T01). Después `set -u`.
- Sin argumentos. Cada chequeo imprime `OK  <nombre>` (stdout) o `FALLO <nombre>: <acción>` (stderr). Al final: `N OK, M fallos` y exit 0 si M=0, **4** si M>0.
- Chequeos (en este orden):
  1. `bash` >= 4: `[ "${BASH_VERSINFO[0]}" -ge 4 ]`.
  2. `nvcc` en PATH: `command -v nvcc`; si falta: "instala/agrega el CUDA toolkit al PATH".
  3. Versión nvcc >= 12.0: parsear `nvcc --version | tail -1` (release 12.2 → 12.2).
  4. `nvcc --list-gpu-arch | grep -qw compute_60` (falla → "este toolkit no soporta sm_60").
  5. `nvidia-smi` ejecutable y con salida: `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1`; verificar que compute_cap empieza por `6.0` y memory.total >= 10000.
  6. `g++` >= 11: `g++ -dumpversion | awk -F. '{print $1}'` >= 11.
  7. `make`, `python3` >= 3.8 (`python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)'`), `awk`, `sort`, `date`.
  8. Espacio: `df -P --block-size=1M "$REPO_ROOT" | awk 'NR==2 {print $4}'` >= 5000 MiB.
  9. Escritura dentro del repo: `d=$(mktemp -d "$REPO_ROOT/.sandbox.write.XXXXXX") && rmdir "$d"`.
- NO comprueba red (solo la necesita la ruta opcional de figuras). NO llama a `sudo`. NO escribe nada salvo el mktemp anterior (que se borra).

## Casos límite
- nvcc presente pero sin driver (nvidia-smi falla): el chequeo 5 lo detecta.
- Varias GPUs: solo se valida la GPU 0 (es la que usa el código, `cudaSetDevice(0)`).

## Verificación LOCAL (sí ejecutable)
```bash
bash -n scripts/check_env.sh
bash scripts/check_env.sh; echo "rc=$?"   # local: rc=4 esperado, con FALLOs de nvcc/nvidia-smi accionables
```

## Verificación EN SERVIDOR (la corre el usuario)
```bash
./scripts/check_env.sh; echo "rc=$?"   # esperado: todos OK y rc=0
```

## Criterios de aceptación
- [ ] `bash -n` limpio; rc local = 4 con mensajes accionables por cada falta.
- [ ] En servidor rc = 0 y lista completa de OKs.
- [ ] No crea archivos persistentes (verificar `git status --short` sin cambios).

## Errores típicos
- Salir 0 cuando algo falta, o 1 en vez de 4.
- Escribir resultados de chequeo en archivos del repo (solo stdout/stderr).

## Commit
`chore: verificacion de entorno (solo lectura)`
