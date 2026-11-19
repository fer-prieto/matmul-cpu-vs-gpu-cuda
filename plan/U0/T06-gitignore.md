# U0-T06 — .gitignore (añadir sandbox y logs)

- **ID**: U0-T06
- **Depende de**: —
- **Objetivo**: que `.sandbox/` y `logs/` nunca se versionen.

## Archivos
- Modificar: `.gitignore` (SOLO añadir al final, sin reordenar ni borrar nada).

## Especificación
Añadir estas 2 líneas exactas al final:
```
# Sandbox y logs del servidor (nunca se versionan)
/.sandbox/
/logs/
```
- NO tocar: el resto del archivo. La línea local `contexto/` (modificación sin commit del usuario) se deja tal cual; el commit de esta tarea la incluirá tal como está.

## Casos límite
- Si `.sandbox/` o `logs/` ya existieran con archivos: no se borran (solo se ignoran).

## Verificación EN SERVIDOR
No aplica (solo cambia la configuración de git; la verificación es idéntica en cualquier máquina).

## Verificación LOCAL (sí ejecutable)
```bash
git check-ignore -v .sandbox logs && echo OK
git diff -- .gitignore   # revisar que el diff sea SOLO la adición de las 2 líneas (+ la línea contexto/ preexistente si sigue sin commit)
```

## Criterios de aceptación
- [ ] `git check-ignore` responde para `.sandbox` y `logs`.
- [ ] `git status` no muestra `.sandbox/` ni `logs/` tras crear `mkdir -p .sandbox logs` (borrarlos después de comprobar).

## Errores típicos
- Ignorar `results/` (los resultados SÍ se versionan).
- Reordenar o borrar líneas existentes.

## Commit
`chore: ignorar sandbox y logs del servidor`
