# U0-T03 — scripts/setup_sandbox.sh (sandbox idempotente)

- **ID**: U0-T03
- **Depende de**: U0-T01
- **Objetivo**: crear `.sandbox/{bin,state,cache,tmp,config}` e `inicio` una sola vez; idempotente.

## Archivos
- Crear: `scripts/setup_sandbox.sh` (ejecutable, `chmod +x`).

## Especificación
- Primera instrucción: `run_guard` (U0-T01). `set -u`.
- Sin argumentos. Pasos:
  1. `mkdir -p "$REPO_ROOT/.sandbox/bin" "$REPO_ROOT/.sandbox/state" "$REPO_ROOT/.sandbox/cache" "$REPO_ROOT/.sandbox/tmp" "$REPO_ROOT/.sandbox/config"`.
  2. Si `$REPO_ROOT/.sandbox/state/inicio` no existe: `date +%s > "$REPO_ROOT/.sandbox/state/inicio"`.
  3. Si TODOS los directorios ya existían y `inicio` ya existía: imprimir `OK sandbox (ya presente)`. Si creó algo: `OK sandbox creado`.
- Nunca borra nada. No toca `state/*.ok` (marcadores de run_all.sh).

## Casos límite
- `.sandbox/` existente pero incompleta (falta `cache/`): crea lo que falta y responde "creado".

## Verificación LOCAL (sí ejecutable)
```bash
bash -n scripts/setup_sandbox.sh
bash scripts/setup_sandbox.sh; echo "rc=$?"   # local: rc=2 (guardia) y NO debe crear .sandbox/
test ! -e .sandbox && echo "OK: nada creado"
```

## Verificación EN SERVIDOR (la corre el usuario)
```bash
./scripts/setup_sandbox.sh        # -> OK sandbox creado, rc=0
./scripts/setup_sandbox.sh        # -> OK sandbox (ya presente), rc=0
ls .sandbox/{bin,state,cache,tmp,config} && cat .sandbox/state/inicio
```

## Criterios de aceptación
- [ ] `bash -n` limpio; segunda corrida en servidor dice "(ya presente)".
- [ ] Crea solo bajo `.sandbox/`; `git status --short` sin cambios (está en .gitignore, pero no debe aparecer nada más).

## Errores típicos
- Borrar `.sandbox/` al inicio (destruiría marcadores de run_all).
- Crear `inicio` cada vez (rompería la puerta de calidad del find).

## Commit
`chore: sandbox idempotente (.sandbox/)`
