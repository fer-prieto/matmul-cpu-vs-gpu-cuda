# U0-T05 — run_all.sh (compositor: guardia, flags, etapas, marcadores, logs, fallo)

- **ID**: U0-T05
- **Depende de**: U0-T01..T04
- **Objetivo**: orquestar todo el flujo del servidor de forma reanudable e idempotente.

## Archivos
- Crear: `run_all.sh` en la raíz (ejecutable, `chmod +x`).

## Especificación
- Cabecera de comentario (3 líneas: uso, exit codes 0/1/2/3/4/130). Primera instrucción: guardia inline con `$0` (patrón contrato §8):
```bash
REPO_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
case "$REPO_ROOT" in /data/FernandoPrieto/*) ;; *) echo "error: raiz invalida..." >&2; exit 2 ;; esac
```
  **Excepción única**: `--dry-run` imprime `[dry-run] guardia omitida` y NO sale con 2 (no ejecuta nada; es la única prueba local posible).
- Luego: `mkdir -p "$REPO_ROOT/logs"`; export TMPDIR/XDG_CACHE_HOME/XDG_CONFIG_HOME (mismo patrón §8); `set -u`.
- Flags (parser manual con case; uso inválido → usage y exit 1): `--check`, `--setup-only`, `--unit 2|3|all` (def `all`), `--skip-cpu`, `--use-sudo`, `--clean-sandbox`, `--dry-run`.
- `--clean-sandbox`: `rm -rf "$REPO_ROOT/.sandbox"` (tras la guardia, antes de todo).
- `--check` → `bash scripts/check_env.sh` + `bash -n scripts/*.sh run_all.sh`; exit 0/4. `check` no usa marcador (es barato y debe repetirse).
- `--setup-only` → check + `scripts/setup_sandbox.sh`.
- Orden de ejecución tras parsear flags: (1) manejar `--dry-run` (imprime y sale), (2) guardia, (3) `--clean-sandbox` si aplica, (4) `mkdir -p logs`, (5) instalar `trap ERR` (escribe `logs/FALLO_<ts>.md`), (6) `exec > >(tee -a logs/run_all_<ts>.log) 2>&1`, (7) etapas. Así la guardia nunca escribe en logs/ y el trap cubre solo las etapas.
- Etapas (tabla del contrato §9). Marcadores: `check` sin marcador; `sandbox` usa `.sandbox/state/sandbox.ok` (global); el resto usa `.sandbox/state/unit${U}_${stage}.ok`.
- Enrutamiento: `sandbox` = `scripts/setup_sandbox.sh`. Todas las etapas por unidad se invocan a través del script de la unidad: `scripts/run_u${U}.sh --stage <stage> --run-id <id> [--skip-cpu] [--use-sudo]`. `--stage` en {build, env_before, cpu, bench, summary, tables, figures, env_after, runs} (+ `ncu` en U3). `run_id = $(date +%Y%m%d-%H%M%S)` se calcula UNA vez y se pasa siempre (sin él cada etapa generaría un id distinto).
- `--use-sudo`: solo se PASA a `run_u${U}.sh`, que hace `sudo -v`, fija relojes y revierte en su propio `trap EXIT` (contrato §9). run_all.sh no ejecuta sudo.
- Si el marcador existe y no es `--clean-sandbox`: imprimir `OK (ya presente)` y saltar la etapa. Si la etapa termina rc=0: `date > .sandbox/state/unit${U}_${stage}.ok` (o `sandbox.ok`).
- Log: todo a `logs/run_all_<ts>.log` vía `exec > >(tee -a "$LOG") 2>&1` tras parsear flags. `ts = $(date +%Y%m%d-%H%M%S)`.
- Fallo: `trap` sobre `ERR` escribe `logs/FALLO_<ts>.md` con: `- etapa: <última etapa>` · `- comando: ${BASH_COMMAND}` · `- exit code: $?` · `- hipótesis: (ver stderr)` · últimas 40 líneas de `$LOG` (via `tail -n 40`). Salir con el exit code de la etapa (3 si validación, 4 si entorno, 1 resto). El trap NO borra el log ni llama a git.
- Mensaje final (solo si todo OK): bloque EXACTO del contrato §9 (git add/commit/push + notas de logs y figuras). Nunca ejecuta git.
- `--dry-run`: imprime por stdout, en orden, `[dry-run] etapa unitX_<stage>: <comando exacto>` para cada etapa que se ejecutaría, y termina. No crea logs, no toca .sandbox.

## Casos límite
- `--unit 3` sin baseline CPU: `run_u3.sh` se autoaprovisiona (ver U3-T04).
- Segunda corrida idéntica: todas las etapas "OK (ya presente)".
- `--clean-sandbox --dry-run`: imprime el rm y termina sin borrar.

## Verificación LOCAL (sí ejecutable)
```bash
bash -n run_all.sh
./run_all.sh --dry-run          # lista de etapas/comandos, sin efectos; verificar que no crea logs/ ni .sandbox/
./run_all.sh --unit 9; echo "rc=$?"   # rc=1 con usage
```

## Verificación EN SERVIDOR (la corre el usuario)
```bash
./run_all.sh --check; echo "rc=$?"     # rc=0
./run_all.sh --setup-only; echo "rc=$?"; ./run_all.sh --setup-only   # segunda: "OK (ya presente)"
```

## Criterios de aceptación
- [ ] `bash -n` limpio; `--dry-run` muestra las etapas exactas del contrato §9.
- [ ] Exit 2 con la guardia fuera de /data/FernandoPrieto (probar desde /home en el servidor).
- [ ] Marcadores creados solo tras éxito; fallo → `logs/FALLO_<ts>.md` autosuficiente.
- [ ] Ninguna llamada a git add/commit/push en el script (solo el texto del mensaje final).

## Errores típicos
- Guardia sin excepción para --dry-run (haría imposible la prueba local).
- Escribir logs antes de mkdir logs/ o fuera de `$REPO_ROOT`.
- `sudo` en scripts hijos o contraseña en cualquier archivo.
- Ejecutar git. Olvidar `tee` (log vacío en el resumen de fallo).

## Commit
`feat: run_all.sh con etapas, marcadores y resumen de fallo`
