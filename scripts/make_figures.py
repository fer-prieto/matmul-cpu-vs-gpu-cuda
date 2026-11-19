#!/usr/bin/env python3
"""make_figures.py: genera las figuras PNG del informe a partir de all_summary.csv.

Unidad 2 -> 3 PNG:  tiempo_vs_n.png, gflops_vs_n.png, speedup_vs_n.png
Unidad 3 -> 4 PNG:  gflops_vs_n.png, speedup_vs_n.png, padding_vs_n.png,
                    tiempo_vs_n.png

Requiere matplotlib (backend Agg, sin display). Si no esta instalado, sale con
codigo 4 y un mensaje accionable: quien invoca decide (venv en .sandbox/ o
generacion local tras el git pull).

Exit codes: 0 ok, 1 error, 2 uso invalido, 4 matplotlib no disponible.
"""

import argparse
import csv
import os
import sys

ESTILO = {
    "gpu_naive": {"color": "#1f77b4", "marker": "o", "label": "naive"},
    "gpu_tiled": {"color": "#ff7f0e", "marker": "s", "label": "tiled (TILE=32)"},
    "gpu_tiled_pad": {"color": "#2ca02c", "marker": "^", "label": "tiled + padding"},
    "gpu_cublas": {"color": "#d62728", "marker": "D", "label": "cuBLAS (referencia)"},
}
ORDEN = ["gpu_naive", "gpu_tiled", "gpu_tiled_pad", "gpu_cublas"]


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


def cargar_summary(path, unit):
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                die("%s esta vacio" % path)
            filas = []
            for row in reader:
                impl = row.get("impl", "")
                if unit == 2 and impl != "gpu_naive":
                    continue
                n = to_int(row.get("n"))
                if n is None:
                    continue
                filas.append(
                    {
                        "impl": impl,
                        "n": n,
                        "t_kernel": to_float(row.get("t_kernel_median_s")),
                        "t_total": to_float(row.get("t_total_median_s")),
                        "gflops": to_float(row.get("gflops")),
                        "speedup_kernel": to_float(row.get("speedup_kernel")),
                        "speedup_total": to_float(row.get("speedup_total")),
                    }
                )
            return filas
    except OSError as exc:
        die("no se puede abrir %s (%s)" % (path, exc))


def por_impl(filas):
    tabla = {}
    for f in filas:
        tabla.setdefault(f["impl"], []).append(f)
    for serie in tabla.values():
        serie.sort(key=lambda x: x["n"])
    return tabla


def estilo(impl):
    return ESTILO.get(impl, {"color": "#7f7f7f", "marker": "x", "label": impl})


def dibujar(plt, series, campo, titulo, ylabel, ruta, escala_log=True):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    dibujado = False
    for impl in ORDEN + [i for i in series if i not in ORDEN]:
        datos = [(f["n"], f.get(campo)) for f in series.get(impl, []) if f.get(campo) is not None]
        if not datos:
            continue
        xs = [d[0] for d in datos]
        ys = [d[1] for d in datos]
        conf = estilo(impl)
        ax.plot(xs, ys, marker=conf["marker"], color=conf["color"], label=conf["label"])
        dibujado = True
    if not dibujado:
        plt.close(fig)
        sys.stderr.write("aviso: sin datos para %s, se omite\n" % os.path.basename(ruta))
        return False
    ax.set_title(titulo)
    ax.set_xlabel("N (dimensión de la matriz)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({f["n"] for serie in series.values() for f in serie}))
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    if escala_log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Genera las figuras del informe (PNG).")
    parser.add_argument("--unit", type=int, required=True, choices=(2, 3))
    parser.add_argument("--summary", required=True, help="all_summary.csv")
    parser.add_argument("--out", required=True, help="directorio de salida")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # ImportError y errores de backend
        sys.stderr.write(
            "FALLO: matplotlib no disponible (%s)\n"
            "  opciones: crear .sandbox/venv con matplotlib, o generar las figuras\n"
            "  en local tras el git pull con el mismo comando.\n" % exc
        )
        sys.exit(4)

    filas = cargar_summary(args.summary, args.unit)
    if not filas:
        die("no hay filas para la unidad %d en %s" % (args.unit, args.summary))
    os.makedirs(args.out, exist_ok=True)
    series = por_impl(filas)
    generadas = []

    if args.unit == 2:
        objetivos = [
            ("t_kernel", "Tiempo de cómputo puro (kernel)", "t_kernel (s)", "tiempo_vs_n.png"),
            ("gflops", "Rendimiento del kernel naive", "GFLOPS", "gflops_vs_n.png"),
            ("speedup_kernel", "Speedup frente a la CPU del servidor", "S = T_cpu / t_kernel", "speedup_vs_n.png"),
        ]
    else:
        objetivos = [
            ("gflops", "Rendimiento por implementación", "GFLOPS", "gflops_vs_n.png"),
            ("speedup_kernel", "Speedup frente a la CPU del servidor", "S = T_cpu / t_kernel", "speedup_vs_n.png"),
            ("t_kernel", "Tiempo de cómputo puro (kernel)", "t_kernel (s)", "tiempo_vs_n.png"),
        ]

    for campo, titulo, ylabel, nombre in objetivos:
        ruta = os.path.join(args.out, nombre)
        if dibujar(plt, series, campo, titulo, ylabel, ruta):
            generadas.append(ruta)

    if args.unit == 3:
        # Ablacion de padding: barras agrupadas por N (escala lineal).
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ns = sorted({f["n"] for f in filas})
        tabla = {(f["impl"], f["n"]): f for f in filas}
        base = [tabla.get(("gpu_tiled", n), {}).get("gflops") for n in ns]
        pad = [tabla.get(("gpu_tiled_pad", n), {}).get("gflops") for n in ns]
        if any(v is not None for v in base + pad):
            ancho = 0.35
            pos = list(range(len(ns)))
            izq = [p - ancho / 2 for p in pos]
            der = [p + ancho / 2 for p in pos]
            valores_base = [v if v is not None else 0.0 for v in base]
            valores_pad = [v if v is not None else 0.0 for v in pad]
            ax.bar(izq, valores_base, width=ancho, color=estilo("gpu_tiled")["color"], label="tiled")
            ax.bar(der, valores_pad, width=ancho, color=estilo("gpu_tiled_pad")["color"], label="tiled + padding")
            ax.set_xticks(pos)
            ax.set_xticklabels([str(n) for n in ns])
            ax.set_title("Ablación de padding (bank conflicts)")
            ax.set_xlabel("N (dimensión de la matriz)")
            ax.set_ylabel("GFLOPS")
            ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
            ax.legend(fontsize=8)
            fig.tight_layout()
            ruta = os.path.join(args.out, "padding_vs_n.png")
            fig.savefig(ruta, dpi=300)
            generadas.append(ruta)
        else:
            sys.stderr.write("aviso: sin datos de gpu_tiled/gpu_tiled_pad, se omite padding_vs_n.png\n")
        plt.close(fig)

    if not generadas:
        die("no se genero ninguna figura")
    for ruta in generadas:
        print(ruta)


if __name__ == "__main__":
    main()
