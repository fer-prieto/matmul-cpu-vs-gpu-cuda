#!/usr/bin/env python3
"""summarize.py: agrega los CSV crudos en un resumen por (implementacion, N).

Modos:
  --mode cpu   entrada = salida de `cpu_baseline --csv`
               (impl,n,phase,rep,t_s,gflops,checksum,seed)
  --mode gpu   entrada = stdout de `build/bench_gpu`
               (impl,kind,n,phase,rep,seed,t_kernel_s,t_total_s,gflops,checksum)

Estadisticos: minimo, mediana, media y desviacion estandar muestral (n-1).
GFLOPS se calcula siempre con la mediana del tiempo de kernel:
    GFLOPS = 2*N^3 / (t_mediana * 1e9)

Los speedups se calculan solo con --baseline (resumen CPU del mismo servidor):
    speedup_kernel = T_cpu / t_kernel_mediana
    speedup_total  = T_cpu / t_total_mediana
con T_cpu = t_median_s de `cpu_ikj` para el mismo N.

Solo usa la biblioteca estandar. Exit codes: 0 ok, 1 error, 2 uso invalido.
"""

import argparse
import csv
import statistics
import sys

CPU_FIELDS = [
    "impl",
    "n",
    "reps",
    "t_min_s",
    "t_median_s",
    "t_mean_s",
    "t_std_s",
    "gflops",
    "checksum",
    "checksum_stable",
    "seed",
]

GPU_FIELDS = [
    "run_id",
    "label",
    "impl",
    "kind",
    "n",
    "reps",
    "t_kernel_min_s",
    "t_kernel_median_s",
    "t_kernel_mean_s",
    "t_kernel_std_s",
    "t_kernel_cv_pct",
    "t_total_min_s",
    "t_total_median_s",
    "t_total_mean_s",
    "t_total_std_s",
    "gflops",
    "gflops_best",
    "gflops_mean",
    "checksum",
    "checksum_stable",
    "seed",
    "speedup_kernel",
    "speedup_total",
]


def die(message, code=1):
    sys.stderr.write("error: %s\n" % message)
    sys.exit(code)


def to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def gflops(n, seconds):
    if seconds is None or seconds <= 0:
        return None
    return 2.0 * float(n) ** 3 / (seconds * 1e9)


def fmt(value, decimals):
    if value is None:
        return ""
    return ("%." + str(decimals) + "f") % value


def fmt_checksum(value):
    if value is None:
        return ""
    return "%.17g" % value


def media(values):
    return sum(values) / len(values) if values else None


def sigma(values):
    return statistics.stdev(values) if len(values) > 1 else None


def mediana(values):
    return statistics.median(values) if values else None


def leer_csv(path, required):
    try:
        handle = sys.stdin if path in ("-", "/dev/stdin") else open(path, newline="", encoding="utf-8")
    except OSError as exc:
        die("no se puede abrir %s (%s)" % (path, exc))
    with handle as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            die("%s esta vacio" % path)
        faltan = [c for c in required if c not in reader.fieldnames]
        if faltan:
            die("%s no tiene las columnas %s" % (path, ", ".join(faltan)))
        return list(reader)


def agrupar(rows):
    """Agrupa por (impl, n) conservando el orden de primera aparicion."""
    grupos = {}
    for row in rows:
        if row.get("phase") != "measured":
            continue
        try:
            n = int(row["n"])
        except (TypeError, ValueError):
            continue
        clave = (row["impl"], n)
        grupos.setdefault(clave, []).append(row)
    return grupos


def leer_baseline(path):
    """(impl, n) -> t_median_s del resumen CPU."""
    base = {}
    for row in leer_csv(path, ["impl", "n", "t_median_s"]):
        try:
            base[(row["impl"], int(row["n"]))] = to_float(row["t_median_s"])
        except (TypeError, ValueError):
            continue
    return base


def resumen_cpu(rows):
    filas = []
    for (impl, n), grupo in agrupar(rows).items():
        tiempos = [t for t in (to_float(r.get("t_s")) for r in grupo) if t is not None]
        if not tiempos:
            sys.stderr.write("aviso: %s N=%d sin tiempos medidos, se omite\n" % (impl, n))
            continue
        checksums = [to_float(r.get("checksum")) for r in grupo]
        t_med = mediana(tiempos)
        filas.append(
            {
                "impl": impl,
                "n": str(n),
                "reps": str(len(tiempos)),
                "t_min_s": fmt(min(tiempos), 9),
                "t_median_s": fmt(t_med, 9),
                "t_mean_s": fmt(media(tiempos), 9),
                "t_std_s": fmt(sigma(tiempos), 9),
                "gflops": fmt(gflops(n, t_med), 9),
                "checksum": fmt_checksum(checksums[-1]),
                "checksum_stable": "true" if len(set(checksums)) <= 1 else "false",
                "seed": grupo[-1].get("seed", ""),
            }
        )
    return filas


def resumen_gpu(rows, baseline, run_id, label):
    filas = []
    for (impl, n), grupo in agrupar(rows).items():
        tk = [t for t in (to_float(r.get("t_kernel_s")) for r in grupo) if t is not None]
        tt = [t for t in (to_float(r.get("t_total_s")) for r in grupo) if t is not None]
        if not tk:
            sys.stderr.write("aviso: %s N=%d sin tiempos de kernel, se omite\n" % (impl, n))
            continue
        checksums = [to_float(r.get("checksum")) for r in grupo]
        semillas = {r.get("seed") for r in grupo}
        if len(semillas) > 1:
            sys.stderr.write("aviso: %s N=%d mezcla semillas %s\n" % (impl, n, sorted(semillas)))
        tk_med = mediana(tk)
        tt_med = mediana(tt)
        gf = gflops(n, tk_med)
        gf_reps = [g for g in (gflops(n, t) for t in tk) if g is not None]
        t_cpu = baseline.get(("cpu_ikj", n)) if baseline is not None else None
        filas.append(
            {
                "run_id": run_id,
                "label": label,
                "impl": impl,
                "kind": grupo[-1].get("kind", ""),
                "n": str(n),
                "reps": str(len(tk)),
                "t_kernel_min_s": fmt(min(tk), 9),
                "t_kernel_median_s": fmt(tk_med, 9),
                "t_kernel_mean_s": fmt(media(tk), 9),
                "t_kernel_std_s": fmt(sigma(tk), 9),
                "t_kernel_cv_pct": fmt(
                    100.0 * sigma(tk) / media(tk) if sigma(tk) is not None and media(tk) else None, 4
                ),
                "t_total_min_s": fmt(min(tt), 9) if tt else "",
                "t_total_median_s": fmt(tt_med, 9),
                "t_total_mean_s": fmt(media(tt), 9),
                "t_total_std_s": fmt(sigma(tt), 9),
                "gflops": fmt(gf, 6),
                "gflops_best": fmt(gflops(n, min(tk)), 6),
                "gflops_mean": fmt(media(gf_reps), 6),
                "checksum": fmt_checksum(checksums[-1]),
                "checksum_stable": "true" if len(set(checksums)) <= 1 else "false",
                "seed": grupo[-1].get("seed", ""),
                "speedup_kernel": fmt(t_cpu / tk_med if t_cpu and tk_med else None, 6),
                "speedup_total": fmt(t_cpu / tt_med if t_cpu and tt_med else None, 6),
            }
        )
    return filas


def escribir(filas, fields, out):
    buffer = []
    writer = csv.DictWriter(_ListSink(buffer), fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for fila in filas:
        writer.writerow(fila)
    texto = "".join(buffer)
    if out in ("-", "/dev/stdout"):
        sys.stdout.write(texto)
        return
    with open(out, "w", newline="", encoding="utf-8") as handle:
        handle.write(texto)
    sys.stdout.write(texto)


class _ListSink:
    """Colector mínimo compatible con csv.writer."""

    def __init__(self, buffer):
        self.buffer = buffer

    def write(self, text):
        self.buffer.append(text)
        return len(text)


def main():
    parser = argparse.ArgumentParser(description="Resume los CSV de mediciones (U2/U3).")
    parser.add_argument("--input", required=True, help="CSV crudo (stdout de cpu_baseline o bench_gpu)")
    parser.add_argument("--mode", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--out", required=True, help="CSV de salida ('-' o /dev/stdout para stdout)")
    parser.add_argument("--baseline", help="cpu_summary_<run>.csv del mismo servidor (modo gpu)")
    parser.add_argument("--run-id", default="", help="identificador de la corrida")
    parser.add_argument("--label", default="", help="etiqueta libre")
    args = parser.parse_args()

    if args.mode == "cpu":
        rows = leer_csv(args.input, ["impl", "n", "phase", "t_s", "checksum"])
        filas = resumen_cpu(rows)
        if not filas:
            die("no hay filas 'measured' en %s" % args.input)
        escribir(filas, CPU_FIELDS, args.out)
        return

    rows = leer_csv(args.input, ["impl", "kind", "n", "phase", "t_kernel_s", "t_total_s", "checksum"])
    baseline = leer_baseline(args.baseline) if args.baseline else None
    filas = resumen_gpu(rows, baseline, args.run_id, args.label)
    if not filas:
        die("no hay filas 'measured' en %s" % args.input)
    escribir(filas, GPU_FIELDS, args.out)


if __name__ == "__main__":
    main()
