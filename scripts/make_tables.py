#!/usr/bin/env python3
"""make_tables.py: genera las tablas del informe en LaTeX y Markdown.

Entradas (results/unidadX/):
  all_summary.csv        resumen por (implementacion, N)
  validation_<run>.csv   validacion numerica contra cpu_ikj

Salidas (--out DIR):
  tabla_tiempos.tex/.md       t_kernel, t_total y GFLOPS
  tabla_speedup.tex/.md       speedup frente a la CPU del servidor
  tabla_validacion.tex/.md    error relativo, error maximo y veredicto
  tabla_padding.tex/.md       (solo --unit 3) tiled vs tiled_pad
  tabla_comparativa.tex/.md   (solo --unit 3) naive / tiled / tiled_pad / cuBLAS

Los .tex son fragmentos `tabular` planos (sin \\begin{table} ni caption): la
plantilla del informe aporta el entorno, el titulo y la etiqueta.

Solo usa la biblioteca estandar. Exit codes: 0 ok, 1 error, 2 uso invalido.
"""

import argparse
import csv
import os
import sys

IMPL_ORDER = ["gpu_naive", "gpu_tiled", "gpu_tiled_pad", "gpu_cublas"]
UNIT2_IMPLS = ["gpu_naive"]

# Orden de columnas de cada tabla.
TABLA_TIEMPOS_HEADER_TEX = (
    r"\textbf{Implementación} & \textbf{Tipo} & \textbf{N} & "
    r"\textbf{$t_{kernel}$ (s)} & \textbf{$t_{total}$ (s)} & \textbf{GFLOPS} \\"
)
TABLA_TIEMPOS_HEADER_MD = "Implementación | Tipo | N | t_kernel (s) | t_total (s) | GFLOPS"

TABLA_SPEEDUP_HEADER_TEX = (
    r"\textbf{Implementación} & \textbf{N} & \textbf{$S_{kernel}$} & \textbf{$S_{total}$} \\"
)
TABLA_SPEEDUP_HEADER_MD = "Implementación | N | speedup_kernel | speedup_total"

TABLA_VALIDACION_HEADER_TEX = (
    r"\textbf{Implementación} & \textbf{N} & \textbf{Error relativo} & "
    r"\textbf{Máx. abs.} & \textbf{¿Válido?} \\"
)
TABLA_VALIDACION_HEADER_MD = "Implementación | N | Error relativo | Máx. abs. | ¿Válido?"

TABLA_PADDING_HEADER_TEX = (
    r"\textbf{N} & \textbf{GFLOPS \texttt{gpu\_tiled}} & "
    r"\textbf{GFLOPS \texttt{gpu\_tiled\_pad}} & \textbf{$\Delta$ (\%)} \\"
)
TABLA_PADDING_HEADER_MD = "N | GFLOPS gpu_tiled | GFLOPS gpu_tiled_pad | Delta (%)"

TABLA_COMPARATIVA_HEADER_TEX = (
    r"\textbf{N} & \textbf{naive} & \textbf{tiled} & \textbf{tiled\_pad} & "
    r"\textbf{cuBLAS} & \textbf{\% de cuBLAS} \\"
)
TABLA_COMPARATIVA_HEADER_MD = "N | naive | tiled | tiled_pad | cuBLAS | % de cuBLAS"


def die(message, code=1):
    sys.stderr.write("error: %s\n" % message)
    sys.exit(code)


def to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def to_int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def tex_escape(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def num(value, decimals=4, empty="---"):
    return empty if value is None else ("%." + str(decimals) + "f") % value


def sci_tex(value, empty="---"):
    if value is None:
        return empty
    if value == 0.0:
        return r"$0$"
    mantisa, exponente = ("%.2e" % value).split("e")
    return r"$%s\times10^{%d}$" % (mantisa, int(exponente))


def sci_md(value, empty="---"):
    return empty if value is None else "%.2e" % value


def leer_csv(path, required):
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            faltan = [c for c in required if c not in reader.fieldnames]
            if faltan:
                die("%s no tiene las columnas %s" % (path, ", ".join(faltan)))
            return list(reader)
    except OSError as exc:
        die("no se puede abrir %s (%s)" % (path, exc))


def escribir(base, filas_tex, filas_md, header_tex, header_md, out_dir):
    tex = ["\\hline", header_tex, "\\hline"]
    tex.extend(filas_tex)
    tex.append("\\hline")
    with open(os.path.join(out_dir, base + ".tex"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(tex) + "\n")

    columnas = [c.strip() for c in header_md.split("|")]
    md = [
        "| " + " | ".join(columnas) + " |",
        "|" + "|".join(["---"] * len(columnas)) + "|",
    ]
    md.extend(filas_md)
    with open(os.path.join(out_dir, base + ".md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(md) + "\n")
    print(os.path.join(out_dir, base + ".tex"))
    print(os.path.join(out_dir, base + ".md"))


def orden(impls_presentes):
    conocidas = [i for i in IMPL_ORDER if i in impls_presentes]
    extra = sorted(i for i in impls_presentes if i not in IMPL_ORDER)
    return conocidas + extra


def cargar_summary(path, unit):
    filas = []
    for row in leer_csv(path, ["impl", "kind", "n", "t_kernel_median_s", "gflops"]):
        impl = row["impl"]
        if unit == 2 and impl not in UNIT2_IMPLS:
            continue
        n = to_int(row["n"])
        if n is None:
            continue
        filas.append(
            {
                "impl": impl,
                "kind": row.get("kind", ""),
                "n": n,
                "t_kernel": to_float(row.get("t_kernel_median_s")),
                "t_total": to_float(row.get("t_total_median_s")),
                "gflops": to_float(row.get("gflops")),
                "speedup_kernel": to_float(row.get("speedup_kernel")),
                "speedup_total": to_float(row.get("speedup_total")),
            }
        )
    return filas


def cargar_validacion(path, unit):
    filas = []
    for row in leer_csv(path, ["impl", "n", "rel_error", "valid"]):
        impl = row["impl"]
        if unit == 2 and impl not in UNIT2_IMPLS:
            continue
        n = to_int(row["n"])
        if n is None:
            continue
        filas.append(
            {
                "impl": impl,
                "n": n,
                "rel": to_float(row.get("rel_error")),
                "max_abs": to_float(row.get("max_abs_error")),
                "valid": (row.get("valid", "").strip().lower() == "true"),
            }
        )
    return filas


def tabla_tiempos(filas, unit):
    presentes = {f["impl"] for f in filas}
    orden_impl = orden(presentes)
    filas_tex, filas_md = [], []
    for impl in orden_impl:
        for f in sorted((x for x in filas if x["impl"] == impl), key=lambda x: x["n"]):
            filas_tex.append(
                "%s & %s & %d & %s & %s & %s \\\\"
                % (
                    tex_escape(impl),
                    f["kind"],
                    f["n"],
                    num(f["t_kernel"], 4),
                    num(f["t_total"], 4),
                    num(f["gflops"], 2),
                )
            )
            filas_md.append(
                "| %s | %s | %d | %s | %s | %s |"
                % (impl, f["kind"], f["n"], num(f["t_kernel"], 4), num(f["t_total"], 4), num(f["gflops"], 2))
            )
    return filas_tex, filas_md


def tabla_speedup(filas, unit):
    presentes = {f["impl"] for f in filas}
    orden_impl = orden(presentes)
    filas_tex, filas_md = [], []
    for impl in orden_impl:
        for f in sorted((x for x in filas if x["impl"] == impl), key=lambda x: x["n"]):
            filas_tex.append(
                "%s & %d & %s & %s \\\\"
                % (tex_escape(impl), f["n"], num(f["speedup_kernel"], 2), num(f["speedup_total"], 2))
            )
            filas_md.append(
                "| %s | %d | %s | %s |"
                % (impl, f["n"], num(f["speedup_kernel"], 2), num(f["speedup_total"], 2))
            )
    return filas_tex, filas_md


def tabla_validacion(filas):
    filas_tex, filas_md = [], []
    for f in sorted(filas, key=lambda x: (IMPL_ORDER.index(x["impl"]) if x["impl"] in IMPL_ORDER else 99, x["n"])):
        veredicto = "sí" if f["valid"] else "no"
        filas_tex.append(
            "%s & %d & %s & %s & %s \\\\"
            % (tex_escape(f["impl"]), f["n"], sci_tex(f["rel"]), sci_tex(f["max_abs"]), veredicto)
        )
        filas_md.append(
            "| %s | %d | %s | %s | %s |" % (f["impl"], f["n"], sci_md(f["rel"]), sci_md(f["max_abs"]), veredicto)
        )
    return filas_tex, filas_md


def por_impl_n(filas):
    tabla = {}
    for f in filas:
        tabla[(f["impl"], f["n"])] = f
    return tabla


def tabla_padding(filas):
    tabla = por_impl_n(filas)
    filas_tex, filas_md = [], []
    for n in sorted({f["n"] for f in filas}):
        base = tabla.get(("gpu_tiled", n), {}).get("gflops")
        pad = tabla.get(("gpu_tiled_pad", n), {}).get("gflops")
        delta = 100.0 * (pad - base) / base if base and pad is not None else None
        filas_tex.append("%d & %s & %s & %s \\\\" % (n, num(base, 2), num(pad, 2), num(delta, 2)))
        filas_md.append("| %d | %s | %s | %s |" % (n, num(base, 2), num(pad, 2), num(delta, 2)))
    return filas_tex, filas_md


def tabla_comparativa(filas):
    tabla = por_impl_n(filas)
    filas_tex, filas_md = [], []
    for n in sorted({f["n"] for f in filas}):
        naive = tabla.get(("gpu_naive", n), {}).get("gflops")
        tiled = tabla.get(("gpu_tiled", n), {}).get("gflops")
        pad = tabla.get(("gpu_tiled_pad", n), {}).get("gflops")
        cublas = tabla.get(("gpu_cublas", n), {}).get("gflops")
        pct = 100.0 * pad / cublas if cublas and pad is not None else None
        filas_tex.append(
            "%d & %s & %s & %s & %s & %s \\\\"
            % (n, num(naive, 2), num(tiled, 2), num(pad, 2), num(cublas, 2), num(pct, 2))
        )
        filas_md.append(
            "| %d | %s | %s | %s | %s | %s |"
            % (n, num(naive, 2), num(tiled, 2), num(pad, 2), num(cublas, 2), num(pct, 2))
        )
    return filas_tex, filas_md


def main():
    parser = argparse.ArgumentParser(description="Genera las tablas LaTeX y Markdown del informe.")
    parser.add_argument("--unit", type=int, required=True, choices=(2, 3))
    parser.add_argument("--summary", required=True, help="all_summary.csv")
    parser.add_argument("--validation", help="validation_<run>.csv")
    parser.add_argument("--out", required=True, help="directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    filas = cargar_summary(args.summary, args.unit)
    if not filas:
        die("no hay filas para la unidad %d en %s" % (args.unit, args.summary))

    escribir(
        "tabla_tiempos",
        *tabla_tiempos(filas, args.unit),
        header_tex=TABLA_TIEMPOS_HEADER_TEX,
        header_md=TABLA_TIEMPOS_HEADER_MD,
        out_dir=args.out
    )
    escribir(
        "tabla_speedup",
        *tabla_speedup(filas, args.unit),
        header_tex=TABLA_SPEEDUP_HEADER_TEX,
        header_md=TABLA_SPEEDUP_HEADER_MD,
        out_dir=args.out
    )

    validacion = cargar_validacion(args.validation, args.unit) if args.validation else []
    if args.validation and not validacion:
        sys.stderr.write("aviso: %s no tiene filas de validacion\n" % args.validation)
    escribir(
        "tabla_validacion",
        *tabla_validacion(validacion),
        header_tex=TABLA_VALIDACION_HEADER_TEX,
        header_md=TABLA_VALIDACION_HEADER_MD,
        out_dir=args.out
    )

    if args.unit == 3:
        escribir(
            "tabla_padding",
            *tabla_padding(filas),
            header_tex=TABLA_PADDING_HEADER_TEX,
            header_md=TABLA_PADDING_HEADER_MD,
            out_dir=args.out
        )
        escribir(
            "tabla_comparativa",
            *tabla_comparativa(filas),
            header_tex=TABLA_COMPARATIVA_HEADER_TEX,
            header_md=TABLA_COMPARATIVA_HEADER_MD,
            out_dir=args.out
        )


if __name__ == "__main__":
    main()
