# U0-T01 — scripts/guard.sh (guardia de raíz + variables de sandbox)

- **ID**: U0-T01
- **Depende de**: —
- **Objetivo**: un único punto que define REPO_ROOT, aborta con código 2 fuera de /data/FernandoPrieto y exporta las variables de sandbox.

## Archivos
- Crear: `scripts/guard.sh` (solo lectura para el resto; se hace `source`).

## Especificación
`scripts/guard.sh` define UNA función y nada más (no ejecuta nada al ser sourceado):
```bash
run_guard() {
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd -P)"
  case "$REPO_ROOT" in
    /data/FernandoPrieto/*) : ;;
    *) echo "error: raiz invalida ($REPO_ROOT): debe vivir bajo /data/FernandoPrieto" >&2; exit 2 ;;
  esac
  mkdir -p "$REPO_ROOT/.sandbox/tmp" "$REPO_ROOT/.sandbox/cache" "$REPO_ROOT/.sandbox/config"
  export TMPDIR="$REPO_ROOT/.sandbox/tmp" XDG_CACHE_HOME="$REPO_ROOT/.sandbox/cache" XDG_CONFIG_HOME="$REPO_ROOT/.sandbox/config"
}
```
- Los scripts de `scripts/` lo usan con `source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/guard.sh"` y llaman `run_guard` como PRIMERA instrucción.
- `chmod +x scripts/guard.sh` no es necesario, pero déjalo 644.
- Cabecera: 3 líneas de comentario (qué hace, uso, exit code 2).

## Casos límite
- Ruta con espacios: todo entre comillas dobles.
- Llamado desde symlink: `pwd -P` resuelve.
- `BASH_SOURCE[1]` = ruta del script que hizo source (siempre absoluta por el patrón de uso).

## Verificación LOCAL (sí ejecutable)
```bash
bash -n scripts/guard.sh
bash -c 'source scripts/guard.sh; run_guard; echo "NO DEBE IMPRIMIRSE"'; echo "rc=$?"   # esperado: mensaje de error y rc=2 (raíz local en /home)
```

## Verificación EN SERVIDOR (la corre el usuario; probar la guardia en ambas direcciones)
```bash
cd /data/FernandoPrieto/<repo>/scripts && bash -c 'source guard.sh; run_guard'; echo "rc=$?"   # rc=0
cd /home/<usuario> && bash -c 'source /data/FernandoPrieto/<repo>/scripts/guard.sh; run_guard'; echo "rc=$?"   # rc=2 (raiz invalida)
```

## Criterios de aceptación
- [ ] `bash -n scripts/guard.sh` limpio.
- [ ] En local: exit 2 con mensaje "raiz invalida".
- [ ] Cero escrituras: la guardia no toca nada fuera de `$REPO_ROOT/.sandbox/` (mkdir -p solo ahí).

## Errores típicos
- Usar `$0` en vez de `BASH_SOURCE[1]` (fallaría al hacer source).
- Olvidar `..` en el cd (REPO_ROOT = scripts/).
- Poner la guardia después de algún `cd` o escritura.

## Commit
`chore: guardia de raiz y variables de sandbox`
