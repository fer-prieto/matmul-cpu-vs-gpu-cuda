//! Salida de resultados.
//!
//! Por cada corrida `<run>` (= `<fecha-hora>[_<etiqueta>]`) se escriben en la
//! carpeta de salida (por defecto `results/unidad1/`):
//!
//! | Archivo | Contenido |
//! |---|---|
//! | `raw_<run>.csv` | Una fila por **ejecución** (calentamientos incluidos) con tiempo, checksum, frecuencia, temperatura, throttling y uso de CPU. |
//! | `summary_<run>.csv` | Una fila por (implementación, N): estadísticas, GFLOPS, speedup, % de OpenBLAS y validación. |
//! | `validation_<run>.csv` | Detalle de la validación de cada (implementación, N). |
//! | `telemetry_<run>.csv` | Muestras periódicas de frecuencia, temperaturas y ventiladores. |
//! | `env_<run>.json` | Entorno de prueba y configuración completa de la corrida. |
//! | `table_<run>.md` | Tabla resumen en Markdown (para pegar en el reporte). |
//! | `log_<run>.txt` | Copia de todo lo que se mostró en la terminal. |
//!
//! Los CSV se escriben y se vacían a disco fila a fila: si la corrida se
//! interrumpe, lo medido hasta ese momento queda guardado.
//!
//! Además, al final de cada corrida se regeneran los archivos consolidados
//! `runs.csv` (índice de todas las corridas) y `all_raw.csv`,
//! `all_summary.csv` y `all_validation.csv` (todas las corridas juntas, con
//! la columna `run_id` para distinguirlas). Son derivados: se pueden borrar y
//! regenerar con `make consolidate`.

use crate::sysinfo::EnvInfo;
use serde::Serialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Consola + log
// ---------------------------------------------------------------------------

static LOG_FILE: Mutex<Option<File>> = Mutex::new(None);

/// Activa la copia de la salida de consola en `file`.
pub fn set_log_file(file: File) {
    if let Ok(mut guard) = LOG_FILE.lock() {
        *guard = Some(file);
    }
}

/// Escribe una línea solo en el log.
pub fn log_only(line: &str) {
    if let Ok(mut guard) = LOG_FILE.lock() {
        if let Some(f) = guard.as_mut() {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Muestra una línea en la terminal y la copia en el log.
pub fn emit(line: &str) {
    let mut out = io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
    log_only(line);
}

/// Como [`emit`], pero por la salida de errores.
pub fn emit_err(line: &str) {
    let _ = writeln!(io::stderr().lock(), "{line}");
    log_only(line);
}

/// `println!` que además copia la línea en `log_<run>.txt`.
#[macro_export]
macro_rules! say {
    () => {
        $crate::report::emit("")
    };
    ($($arg:tt)*) => {
        $crate::report::emit(&format!($($arg)*))
    };
}

// ---------------------------------------------------------------------------
// Registros
// ---------------------------------------------------------------------------

/// Una ejecución (fila de `raw_<run>.csv`).
#[derive(Debug, Clone, Serialize)]
pub struct RawRecord {
    pub run_id: String,
    pub label: String,
    /// Orden global de la ejecución dentro de la corrida (1, 2, ...).
    pub seq: u64,
    #[serde(rename = "impl")]
    pub impl_name: String,
    pub kind: String,
    pub threads: usize,
    pub n: usize,
    /// `warmup`, `measured` o `ref_only` (ejecución solo para validar).
    pub phase: String,
    /// Número de repetición dentro de su fase (desde 1).
    pub rep: usize,
    /// Tiempo de cómputo medido en C++ con steady_clock (s).
    pub t_s: f64,
    /// 2·N³ / (t_s · 10⁹).
    pub gflops: f64,
    /// Tiempo de la llamada completa medido desde Rust (incluye la puesta a
    /// cero de C en i-k-j y el coste de la llamada).
    pub t_call_s: f64,
    /// Suma de C en double: debe repetirse exactamente entre ejecuciones.
    pub checksum: f64,
    pub start_unix_s: f64,
    pub end_unix_s: f64,
    /// CPU lógica en la que estaba el hilo antes y después de ejecutar.
    pub cpu_before: Option<usize>,
    pub cpu_after: Option<usize>,
    /// CPU a la que se fijó el hilo con --pin-cpu (vacío si no se fijó).
    pub pinned_cpu: Option<usize>,
    pub freq_before_mhz: Option<f64>,
    pub freq_after_mhz: Option<f64>,
    pub temp_before_c: Option<f64>,
    pub temp_after_c: Option<f64>,
    /// Eventos/ms de throttling del núcleo durante la ejecución (vacío si el
    /// hilo cambió de CPU o no hay contadores).
    pub throttle_core_delta: Option<i64>,
    pub throttle_core_ms_delta: Option<i64>,
    /// Eventos/ms de throttling del paquete durante la ejecución.
    pub throttle_pkg_delta: Option<i64>,
    pub throttle_pkg_ms_delta: Option<i64>,
    /// Tiempo de CPU del proceso (todos los hilos) durante la ejecución.
    pub cpu_user_s: f64,
    pub cpu_sys_s: f64,
    /// Cambios de contexto y fallos de página del hilo de cómputo.
    pub ctx_vol: i64,
    pub ctx_invol: i64,
    pub minflt: i64,
    pub majflt: i64,
    pub seed: u64,
}

/// Resumen por (implementación, N) (fila de `summary_<run>.csv`).
///
/// Las 13 primeras columnas son las del plan; las demás son extra.
#[derive(Debug, Clone, Serialize)]
pub struct SummaryRow {
    #[serde(rename = "impl")]
    pub impl_name: String,
    pub kind: String,
    pub n: usize,
    pub reps: usize,
    pub t_min_s: f64,
    pub t_median_s: f64,
    pub t_mean_s: f64,
    /// Desviación estándar muestral (n − 1); vacía con una sola repetición.
    pub t_std_s: Option<f64>,
    /// GFLOPS calculados con la mediana.
    pub gflops: f64,
    /// T_mediana(cpu_ijk) / T_mediana(impl) para el mismo N.
    pub speedup_vs_ijk: Option<f64>,
    /// GFLOPS(impl) / GFLOPS(cpu_openblas_1t) × 100.
    pub pct_of_openblas: Option<f64>,
    /// Referencia usada ("es_referencia" si esta implementación lo es).
    pub validated_against: String,
    /// ‖C − C_ref‖_F / ‖C_ref‖_F.
    pub rel_error: Option<f64>,
    // ---- columnas extra ----
    pub max_abs_error: Option<f64>,
    pub freivalds_rel_error: Option<f64>,
    pub valid: Option<bool>,
    pub tolerance: Option<f64>,
    pub t_max_s: f64,
    /// Coeficiente de variación: 100 · σ / media.
    pub t_cv_pct: Option<f64>,
    /// GFLOPS con el mejor tiempo (t_min).
    pub gflops_best: f64,
    /// Media de los GFLOPS de cada repetición.
    pub gflops_mean: f64,
    pub t_call_median_s: f64,
    pub threads: usize,
    pub warmup: usize,
    /// Checksum de la última repetición.
    pub checksum: f64,
    /// `true` si todas las ejecuciones (warm-up incluido) dieron el mismo checksum.
    pub checksum_stable: bool,
    pub temp_max_c: Option<f64>,
    pub freq_after_mean_mhz: Option<f64>,
    pub throttle_pkg_ms_total: Option<i64>,
    pub ctx_invol_total: i64,
    pub run_id: String,
    pub label: String,
    pub seed: u64,
}

/// Validación de una (implementación, N) (fila de `validation_<run>.csv`).
#[derive(Debug, Clone, Serialize)]
pub struct ValidationRow {
    pub run_id: String,
    pub label: String,
    pub n: usize,
    #[serde(rename = "impl")]
    pub impl_name: String,
    pub kind: String,
    /// Referencia usada, "es_referencia", o vacío si no había referencia.
    pub reference: String,
    pub rel_error: Option<f64>,
    pub max_abs_error: Option<f64>,
    pub norm_fro: f64,
    pub norm_fro_ref: Option<f64>,
    pub freivalds_rel_error: Option<f64>,
    pub freivalds_vectors: usize,
    pub tolerance: f64,
    /// `true` si pasan todas las comprobaciones hechas; vacío si no se hizo ninguna.
    pub valid: Option<bool>,
    pub checksum: f64,
    pub checksum_ref: Option<f64>,
}

// ---------------------------------------------------------------------------
// Archivos de la corrida
// ---------------------------------------------------------------------------

/// Rutas de todos los archivos de una corrida.
#[derive(Debug, Clone)]
pub struct RunPaths {
    pub dir: PathBuf,
    pub raw: PathBuf,
    pub summary: PathBuf,
    pub validation: PathBuf,
    pub telemetry: PathBuf,
    pub env: PathBuf,
    pub table_md: PathBuf,
    pub log: PathBuf,
}

impl RunPaths {
    pub fn new(dir: &Path, run_id: &str) -> Self {
        let f = |prefix: &str, ext: &str| dir.join(format!("{prefix}_{run_id}.{ext}"));
        Self {
            dir: dir.to_path_buf(),
            raw: f("raw", "csv"),
            summary: f("summary", "csv"),
            validation: f("validation", "csv"),
            telemetry: f("telemetry", "csv"),
            env: f("env", "json"),
            table_md: f("table", "md"),
            log: f("log", "txt"),
        }
    }

    /// `true` si ya existe algún archivo con este identificador.
    pub fn any_exists(&self) -> bool {
        self.all().iter().any(|(_, p)| p.exists())
    }

    /// (tipo, ruta) de todos los archivos.
    pub fn all(&self) -> Vec<(&'static str, &PathBuf)> {
        vec![
            ("raw", &self.raw),
            ("summary", &self.summary),
            ("validation", &self.validation),
            ("telemetry", &self.telemetry),
            ("env", &self.env),
            ("table_md", &self.table_md),
            ("log", &self.log),
        ]
    }

    /// Nombres de los archivos que existen (para env.json).
    pub fn existing(&self) -> BTreeMap<String, String> {
        self.all()
            .into_iter()
            .filter(|(_, p)| p.exists())
            .map(|(k, p)| (k.to_string(), file_name(p)))
            .collect()
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Escritores CSV de la corrida. Cada archivo se crea con su primera fila y
/// se vacía a disco después de cada fila.
pub struct RunWriters {
    raw_path: PathBuf,
    summary_path: PathBuf,
    validation_path: PathBuf,
    raw: Option<csv::Writer<File>>,
    summary: Option<csv::Writer<File>>,
    validation: Option<csv::Writer<File>>,
    pub raw_rows: usize,
    pub summary_rows: usize,
    pub validation_rows: usize,
}

impl RunWriters {
    pub fn new(paths: &RunPaths) -> Self {
        Self {
            raw_path: paths.raw.clone(),
            summary_path: paths.summary.clone(),
            validation_path: paths.validation.clone(),
            raw: None,
            summary: None,
            validation: None,
            raw_rows: 0,
            summary_rows: 0,
            validation_rows: 0,
        }
    }

    fn write<T: Serialize>(slot: &mut Option<csv::Writer<File>>, path: &Path, record: &T) -> Result<(), csv::Error> {
        if slot.is_none() {
            *slot = Some(csv::Writer::from_path(path)?);
        }
        if let Some(w) = slot.as_mut() {
            w.serialize(record)?;
            w.flush()?;
        }
        Ok(())
    }

    pub fn raw(&mut self, record: &RawRecord) -> Result<(), csv::Error> {
        Self::write(&mut self.raw, &self.raw_path, record)?;
        self.raw_rows += 1;
        Ok(())
    }

    pub fn summary(&mut self, row: &SummaryRow) -> Result<(), csv::Error> {
        Self::write(&mut self.summary, &self.summary_path, row)?;
        self.summary_rows += 1;
        Ok(())
    }

    pub fn validation(&mut self, row: &ValidationRow) -> Result<(), csv::Error> {
        Self::write(&mut self.validation, &self.validation_path, row)?;
        self.validation_rows += 1;
        Ok(())
    }
}

/// Escribe `contents` en `path` de forma atómica (archivo temporal + rename).
fn write_atomic(path: &Path, contents: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)
}

/// Guarda el entorno en JSON (legible, con sangría).
pub fn write_env(path: &Path, env: &EnvInfo) -> Result<(), Box<dyn Error>> {
    let mut text = serde_json::to_string_pretty(env)?;
    text.push('\n');
    write_atomic(path, text.as_bytes())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tablas
// ---------------------------------------------------------------------------

fn opt_f(v: Option<f64>, decimals: usize) -> String {
    v.map(|x| format!("{x:.decimals$}")).unwrap_or_else(|| "-".to_string())
}

fn opt_e(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1e}")).unwrap_or_else(|| "-".to_string())
}

/// Texto breve de la validación de una fila del resumen.
pub fn validation_text(row: &SummaryRow) -> String {
    let verdict = match row.valid {
        Some(true) => "OK",
        Some(false) => "FALLO",
        None => "sin validar",
    };
    let mut details = Vec::new();
    match row.validated_against.as_str() {
        "" => {}
        "es_referencia" => details.push("referencia".to_string()),
        r => details.push(format!("vs {r} {}", opt_e(row.rel_error))),
    }
    if let Some(f) = row.freivalds_rel_error {
        details.push(format!("Freivalds {f:.1e}"));
    }
    if details.is_empty() {
        verdict.to_string()
    } else {
        format!("{verdict} ({})", details.join("; "))
    }
}

/// Tabla resumen para la terminal.
pub fn summary_table(rows: &[SummaryRow]) -> String {
    let mut out = String::new();
    let header = format!(
        "{:<16} {:<10} {:>5} {:>4} {:>13} {:>13} {:>11} {:>6} {:>9} {:>8} {:>8}  {}",
        "impl",
        "tipo",
        "N",
        "reps",
        "t_mediana(s)",
        "t_min(s)",
        "sigma(s)",
        "CV%",
        "GFLOPS",
        "x_ijk",
        "%OBLAS",
        "validacion"
    );
    out.push_str(&header);
    out.push('\n');
    out.push_str(&"-".repeat(header.len() + 12));
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "{:<16} {:<10} {:>5} {:>4} {:>13.6} {:>13.6} {:>11} {:>6} {:>9.3} {:>8} {:>8}  {}\n",
            r.impl_name,
            r.kind,
            r.n,
            r.reps,
            r.t_median_s,
            r.t_min_s,
            opt_f(r.t_std_s, 6),
            opt_f(r.t_cv_pct, 2),
            r.gflops,
            opt_f(r.speedup_vs_ijk, 2),
            opt_f(r.pct_of_openblas, 1),
            validation_text(r)
        ));
    }
    out
}

/// Tabla resumen en Markdown, con los datos clave del entorno.
pub fn markdown_table(rows: &[SummaryRow], env: &EnvInfo) -> String {
    let mut md = String::new();
    md.push_str(&format!("# Resultados de la corrida `{}`\n\n", env.run_id));
    md.push_str(&format!(
        "- Fecha: {} · duración: {} s · estado: {}\n",
        env.started_at,
        opt_f(env.duration_s, 1),
        env.status
    ));
    md.push_str(&format!(
        "- CPU: {} ({} núcleos / {} hilos) · gobernador `{}` · turbo {} · corriente {}\n",
        env.cpu.model_name.as_deref().unwrap_or("?"),
        env.cpu
            .physical_cores
            .map(|c| c.to_string())
            .unwrap_or_else(|| "?".into()),
        env.cpu.logical_cpus,
        env.frequency.governor.as_deref().unwrap_or("?"),
        match env.frequency.turbo_enabled {
            Some(true) => "activado",
            Some(false) => "desactivado",
            None => "?",
        },
        match env.power.ac_online {
            Some(true) => "conectada",
            Some(false) => "NO (batería)",
            None => "?",
        }
    ));
    md.push_str(&format!(
        "- Compilación: {} · `{}` · `-march=native` = {}\n",
        env.build.cxx_version.as_deref().unwrap_or("?"),
        env.build.cxxflags.as_deref().unwrap_or("?"),
        env.build.march_native.as_deref().unwrap_or("?")
    ));
    md.push_str(&format!(
        "- Metodología: semilla {}, {} warm-up, repeticiones {}, mediana; tolerancia {:e}\n",
        env.config.seed, env.config.warmup, env.config.reps_rule, env.config.tolerance
    ));
    md.push_str(&format!(
        "- Commit: `{}`{}\n\n",
        env.git.commit.as_deref().map(|c| &c[..c.len().min(12)]).unwrap_or("?"),
        if env.git.dirty == Some(true) {
            " (con cambios sin commit)"
        } else {
            ""
        }
    ));
    md.push_str("| impl | tipo | N | reps | t mediana (s) | t mín (s) | σ (s) | CV (%) | GFLOPS | speedup vs ijk | % OpenBLAS 1t | validación |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for r in rows {
        md.push_str(&format!(
            "| `{}` | {} | {} | {} | {:.6} | {:.6} | {} | {} | {:.3} | {} | {} | {} |\n",
            r.impl_name,
            r.kind,
            r.n,
            r.reps,
            r.t_median_s,
            r.t_min_s,
            opt_f(r.t_std_s, 6),
            opt_f(r.t_cv_pct, 2),
            r.gflops,
            opt_f(r.speedup_vs_ijk, 2),
            opt_f(r.pct_of_openblas, 1),
            validation_text(r)
        ));
    }
    md.push_str(&format!(
        "\nGFLOPS = 2·N³ / (t · 10⁹) con t = mediana. Datos completos: `summary_{0}.csv` (todas las columnas), \
         `raw_{0}.csv` (cada ejecución), `validation_{0}.csv`, `telemetry_{0}.csv` y `env_{0}.json`.\n",
        env.run_id
    ));
    md
}

// ---------------------------------------------------------------------------
// Consolidación de todas las corridas
// ---------------------------------------------------------------------------

/// Resultado de [`consolidate`].
#[derive(Debug, Clone, Default)]
pub struct Consolidation {
    pub runs: usize,
    pub raw_rows: usize,
    pub summary_rows: usize,
    pub validation_rows: usize,
    pub skipped_files: Vec<String>,
}

/// Archivos `<prefix>*.<ext>` de `dir`, ordenados por nombre (= por fecha).
fn files_with(dir: &Path, prefix: &str, ext: &str) -> io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension().is_some_and(|e| e == ext)
                && p.file_name().is_some_and(|f| f.to_string_lossy().starts_with(prefix))
        })
        .collect();
    files.sort();
    Ok(files)
}

/// Une varios CSV en uno. Las columnas se unen (en orden de aparición) para
/// tolerar cambios de esquema entre versiones; los huecos quedan vacíos.
fn concat_csv(files: &[PathBuf], out: &Path, skipped: &mut Vec<String>) -> Result<usize, Box<dyn Error>> {
    let mut header: Vec<String> = Vec::new();
    let mut tables: Vec<(Vec<String>, Vec<csv::StringRecord>)> = Vec::new();
    for f in files {
        let mut reader = match csv::Reader::from_path(f) {
            Ok(r) => r,
            Err(e) => {
                skipped.push(format!("{}: {e}", file_name(f)));
                continue;
            }
        };
        let cols: Vec<String> = match reader.headers() {
            Ok(h) => h.iter().map(str::to_string).collect(),
            Err(e) => {
                skipped.push(format!("{}: {e}", file_name(f)));
                continue;
            }
        };
        if cols.is_empty() {
            continue;
        }
        let records: Vec<csv::StringRecord> = reader.records().filter_map(Result::ok).collect();
        for c in &cols {
            if !header.contains(c) {
                header.push(c.clone());
            }
        }
        tables.push((cols, records));
    }
    let mut writer = csv::Writer::from_writer(Vec::new());
    let mut rows = 0;
    if !header.is_empty() {
        writer.write_record(&header)?;
        for (cols, records) in &tables {
            let index: Vec<Option<usize>> = header.iter().map(|h| cols.iter().position(|c| c == h)).collect();
            for rec in records {
                let row: Vec<&str> = index.iter().map(|i| i.and_then(|i| rec.get(i)).unwrap_or("")).collect();
                writer.write_record(&row)?;
                rows += 1;
            }
        }
    }
    let bytes = writer.into_inner().map_err(|e| e.to_string())?;
    write_atomic(out, &bytes)?;
    Ok(rows)
}

/// Valor JSON como texto plano para el índice de corridas.
fn json_text(v: Option<&serde_json::Value>) -> String {
    use serde_json::Value;
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Array(a)) => a.iter().map(|x| json_text(Some(x))).collect::<Vec<_>>().join(";"),
        Some(other) => other.to_string(),
    }
}

/// Columnas de `runs.csv` y la ruta JSON (dentro de env_<run>.json) de cada una.
const RUN_INDEX_COLUMNS: [(&str, &str); 37] = [
    ("run_id", "/run_id"),
    ("label", "/label"),
    ("status", "/status"),
    ("started_at", "/started_at"),
    ("finished_at", "/finished_at"),
    ("duration_s", "/duration_s"),
    ("sizes", "/config/sizes"),
    ("impls", "/config/impls"),
    ("reps", "/config/reps_rule"),
    ("warmup", "/config/warmup"),
    ("seed", "/config/seed"),
    ("include_slow", "/config/include_slow"),
    ("slow_max_n", "/config/slow_max_n"),
    ("validate", "/config/validate"),
    ("tolerance", "/config/tolerance"),
    ("all_valid", "/results/all_valid"),
    ("executions", "/results/executions"),
    ("pin_cpu", "/config/pin_cpu"),
    ("mt_threads", "/config/mt_threads"),
    ("sample_ms", "/config/sample_ms"),
    ("cooldown_s", "/config/cooldown_s"),
    ("affinity", "/process/affinity"),
    ("governor", "/frequency/governor"),
    ("turbo_enabled", "/frequency/turbo_enabled"),
    ("ac_online", "/power/ac_online"),
    ("temp_start_c", "/thermal/primary_temp_start_c"),
    ("temp_end_c", "/thermal/primary_temp_end_c"),
    ("throttle_pkg_ms_during_run", "/thermal/throttle_pkg_ms_during_run"),
    ("cpu_model", "/cpu/model_name"),
    ("march_native", "/build/march_native"),
    ("cxx_version", "/build/cxx_version"),
    ("cxxflags", "/build/cxxflags"),
    ("features", "/build/features"),
    ("openblas_config", "/openblas/config"),
    ("git_commit", "/git/commit"),
    ("git_dirty", "/git/dirty"),
    ("notes", "/notes"),
];

fn write_run_index(env_files: &[PathBuf], out: &Path, skipped: &mut Vec<String>) -> Result<usize, Box<dyn Error>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    let mut header: Vec<&str> = RUN_INDEX_COLUMNS.iter().map(|(c, _)| *c).collect();
    header.push("warnings");
    writer.write_record(&header)?;
    let mut rows = 0;
    for f in env_files {
        let value: serde_json::Value = match fs::read_to_string(f)
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
        {
            Ok(v) => v,
            Err(e) => {
                skipped.push(format!("{}: {e}", file_name(f)));
                continue;
            }
        };
        let mut row: Vec<String> = RUN_INDEX_COLUMNS
            .iter()
            .map(|(_, ptr)| json_text(value.pointer(ptr)))
            .collect();
        let warnings = value
            .pointer("/warnings")
            .and_then(|w| w.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(" | "))
            .unwrap_or_default();
        row.push(warnings);
        writer.write_record(&row)?;
        rows += 1;
    }
    let bytes = writer.into_inner().map_err(|e| e.to_string())?;
    write_atomic(out, &bytes)?;
    Ok(rows)
}

/// Regenera `runs.csv` y `all_{raw,summary,validation}.csv` en `dir` a partir
/// de los archivos de cada corrida.
pub fn consolidate(dir: &Path) -> Result<Consolidation, Box<dyn Error>> {
    let mut c = Consolidation::default();
    let env_files = files_with(dir, "env_", "json")?;
    c.runs = write_run_index(&env_files, &dir.join("runs.csv"), &mut c.skipped_files)?;
    c.raw_rows = concat_csv(
        &files_with(dir, "raw_", "csv")?,
        &dir.join("all_raw.csv"),
        &mut c.skipped_files,
    )?;
    c.summary_rows = concat_csv(
        &files_with(dir, "summary_", "csv")?,
        &dir.join("all_summary.csv"),
        &mut c.skipped_files,
    )?;
    c.validation_rows = concat_csv(
        &files_with(dir, "validation_", "csv")?,
        &dir.join("all_validation.csv"),
        &mut c.skipped_files,
    )?;
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("matbench_test_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn concatena_csv_con_columnas_distintas() {
        let d = tmp_dir("concat");
        fs::write(d.join("raw_1.csv"), "a,b\n1,2\n3,4\n").unwrap();
        fs::write(d.join("raw_2.csv"), "a,c\n5,6\n").unwrap();
        let mut skipped = Vec::new();
        let files = files_with(&d, "raw_", "csv").unwrap();
        let rows = concat_csv(&files, &d.join("all_raw.csv"), &mut skipped).unwrap();
        assert_eq!(rows, 3);
        let text = fs::read_to_string(d.join("all_raw.csv")).unwrap();
        assert_eq!(text, "a,b,c\n1,2,\n3,4,\n5,,6\n");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn texto_de_valores_json() {
        let v: serde_json::Value = serde_json::json!({"x": [1, 2], "y": null, "z": true});
        assert_eq!(json_text(v.pointer("/x")), "1;2");
        assert_eq!(json_text(v.pointer("/y")), "");
        assert_eq!(json_text(v.pointer("/z")), "true");
        assert_eq!(json_text(v.pointer("/nada")), "");
    }
}
