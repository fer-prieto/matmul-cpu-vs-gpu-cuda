//! Bucle de medición y estadísticas.
//!
//! Para cada N (en el orden de `--sizes`):
//!  1. Se generan A y B con la semilla (A: stream 0, B: stream 1).
//!  2. Para cada implementación (en el orden del registro): `warmup`
//!     ejecuciones de calentamiento no medidas y `reps` medidas. Cada una se
//!     guarda al momento en `raw_<run>.csv` (las de calentamiento también, con
//!     `phase = warmup`), junto con lecturas de frecuencia, temperatura,
//!     throttling y uso de CPU tomadas justo antes y justo después.
//!  3. Se valida la salida de cada implementación (ver [`crate::validate`]).
//!  4. Se calculan las estadísticas y se escribe una fila por implementación
//!     en `summary_<run>.csv`.
//!
//! El tiempo que se usa es el que mide la propia biblioteca C++ con
//! `std::chrono::steady_clock` alrededor del triple bucle. GFLOPS se calcula
//! con la mediana: 2·N³ / (t · 10⁹).

use crate::matrix::Matrix;
use crate::registry::{self, Implementation, Threading, CPU_IJK, CPU_OPENBLAS_1T};
use crate::report::{RawRecord, RunWriters, SummaryRow, ValidationRow};
use crate::rng::{STREAM_A, STREAM_B};
use crate::say;
use crate::sysinfo;
use crate::telemetry::{self, Sampler, Sensors, Snapshot};
use crate::validate;
use serde::Serialize;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Repeticiones medidas por defecto.
pub const REPS_DEFAULT: usize = 5;
/// Repeticiones por defecto para N ≥ [`LARGE_N`].
pub const REPS_LARGE: usize = 3;
/// A partir de este N se usan [`REPS_LARGE`] repeticiones por defecto.
pub const LARGE_N: usize = 4096;

pub const PHASE_WARMUP: &str = "warmup";
pub const PHASE_MEASURED: &str = "measured";
pub const PHASE_REF_ONLY: &str = "ref_only";
/// Valor de `validated_against` para la implementación que hace de referencia.
pub const IS_REFERENCE: &str = "es_referencia";

// ---------------------------------------------------------------------------
// Interrupción ordenada con Ctrl+C
// ---------------------------------------------------------------------------

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Error que devuelve la corrida cuando el usuario la interrumpe con Ctrl+C.
#[derive(Debug, Clone, Copy)]
pub struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "corrida interrumpida por el usuario (Ctrl+C)")
    }
}

impl Error for Interrupted {}

extern "C" fn on_sigint(_signal: libc::c_int) {
    // Solo operaciones seguras dentro de un manejador de señales: atómicos,
    // write(2), signal(2) y raise(3).
    if STOP_REQUESTED.swap(true, Ordering::SeqCst) {
        // Segundo Ctrl+C: comportamiento por defecto (terminar ya).
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::raise(libc::SIGINT);
        }
    } else {
        let msg = b"\n[matbench] Interrupcion solicitada: se detendra al terminar la ejecucion en curso \
                    (pulsa Ctrl+C otra vez para salir inmediatamente).\n";
        unsafe {
            libc::write(2, msg.as_ptr().cast(), msg.len());
        }
    }
}

/// Instala el manejador de Ctrl+C: la primera vez la corrida se detiene al
/// terminar la ejecución en curso (y se guarda con estado "interrupted"); la
/// segunda vez el proceso termina inmediatamente. Lo medido hasta ese momento
/// ya está en disco en ambos casos.
pub fn install_interrupt_handler() {
    let handler: extern "C" fn(libc::c_int) = on_sigint;
    // SAFETY: el manejador solo usa funciones seguras en contexto de señal.
    unsafe {
        libc::signal(libc::SIGINT, handler as libc::sighandler_t);
    }
}

/// `true` si se pidió detener la corrida.
pub fn stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::SeqCst)
}

/// Configuración efectiva de una corrida (se guarda completa en env.json).
#[derive(Debug, Clone, Serialize)]
pub struct RunConfig {
    pub sizes: Vec<usize>,
    pub impls: Vec<String>,
    /// Repeticiones fijadas con --reps (vacío = regla por defecto).
    pub reps: Option<usize>,
    /// Descripción legible de la regla de repeticiones.
    pub reps_rule: String,
    pub warmup: usize,
    pub seed: u64,
    pub include_slow: bool,
    pub slow_max_n: usize,
    pub validate: bool,
    pub tolerance: f64,
    pub freivalds_vectors: usize,
    /// Referencia de correctitud preferida.
    pub reference: String,
    pub sample_ms: u64,
    pub cooldown_s: f64,
    pub pin_cpu: Option<usize>,
    pub mt_threads: usize,
    pub label: String,
    pub notes: String,
    pub out_dir: String,
}

impl RunConfig {
    /// Repeticiones medidas para un N.
    pub fn reps_for(&self, n: usize) -> usize {
        self.reps
            .unwrap_or(if n >= LARGE_N { REPS_LARGE } else { REPS_DEFAULT })
    }

    /// Texto de la regla de repeticiones.
    pub fn describe_reps(reps: Option<usize>) -> String {
        match reps {
            Some(r) => format!("{r}"),
            None => format!("{REPS_DEFAULT} ({REPS_LARGE} para N >= {LARGE_N})"),
        }
    }
}

/// Estadísticas de una muestra de tiempos.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Stats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    /// Desviación estándar muestral (divisor n − 1); `None` si count < 2.
    pub std: Option<f64>,
}

impl Stats {
    /// `None` si no hay muestras.
    pub fn from_samples(samples: &[f64]) -> Option<Stats> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let count = sorted.len();
        let median = if count % 2 == 1 {
            sorted[count / 2]
        } else {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        };
        let mean = samples.iter().sum::<f64>() / count as f64;
        let std = (count >= 2).then(|| {
            let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64;
            var.sqrt()
        });
        Some(Stats {
            count,
            min: sorted[0],
            max: sorted[count - 1],
            mean,
            median,
            std,
        })
    }

    /// Coeficiente de variación en % (100 · σ / media).
    pub fn cv_pct(&self) -> Option<f64> {
        self.std
            .map(|s| if self.mean != 0.0 { 100.0 * s / self.mean } else { 0.0 })
    }
}

/// GFLOPS = 2·N³ / (t · 10⁹). NaN si t ≤ 0.
pub fn gflops(n: usize, seconds: f64) -> f64 {
    let nf = n as f64;
    if seconds > 0.0 {
        2.0 * nf * nf * nf / (seconds * 1e9)
    } else {
        f64::NAN
    }
}

/// Resultado global de la corrida.
#[derive(Debug, Clone, Default)]
pub struct RunOutcome {
    pub summaries: Vec<SummaryRow>,
    pub validations: Vec<ValidationRow>,
    /// Ejecuciones registradas en raw (calentamientos incluidos).
    pub executions: usize,
    pub skipped: Vec<String>,
}

impl RunOutcome {
    /// `Some(true)` si todo lo validado pasó; `None` si no se validó nada.
    pub fn all_valid(&self) -> Option<bool> {
        let checked: Vec<bool> = self.validations.iter().filter_map(|v| v.valid).collect();
        (!checked.is_empty()).then(|| checked.iter().all(|&v| v))
    }
}

/// Mediciones de una implementación para un N.
struct ImplRun {
    imp: Implementation,
    threads: usize,
    records: Vec<RawRecord>,
    stats: Stats,
    output: Matrix,
}

impl ImplRun {
    fn measured(&self) -> impl Iterator<Item = &RawRecord> {
        self.records.iter().filter(|r| r.phase == PHASE_MEASURED)
    }
}

/// Ejecuta la corrida completa.
pub struct Runner<'a> {
    config: &'a RunConfig,
    run_id: &'a str,
    sensors: &'a Sensors,
    writers: &'a mut RunWriters,
    sampler: Option<&'a Sampler>,
    original_affinity: Option<Vec<usize>>,
    seq: u64,
}

impl<'a> Runner<'a> {
    pub fn new(
        config: &'a RunConfig,
        run_id: &'a str,
        sensors: &'a Sensors,
        writers: &'a mut RunWriters,
        sampler: Option<&'a Sampler>,
        original_affinity: Option<Vec<usize>>,
    ) -> Self {
        Self {
            config,
            run_id,
            sensors,
            writers,
            sampler,
            original_affinity,
            seq: 0,
        }
    }

    /// Mide todas las implementaciones en todos los tamaños.
    pub fn run(&mut self, impls: &[Implementation]) -> Result<RunOutcome, Box<dyn Error>> {
        let config = self.config;
        let mut outcome = RunOutcome::default();
        let mut result = Ok(());
        for &n in &config.sizes {
            if let Err(e) = self.run_size(n, impls, &mut outcome) {
                result = Err(e);
                break;
            }
        }
        self.restore_affinity();
        self.set_context("-", 0, "fin", 0);
        outcome.executions = self.seq as usize;
        result.map(|()| outcome)
    }

    fn set_context(&self, impl_name: &str, n: usize, phase: &str, rep: usize) {
        if let Some(s) = self.sampler {
            s.set_context(impl_name, n, phase, rep);
        }
    }

    fn restore_affinity(&self) {
        if let (Some(_), Some(orig)) = (self.config.pin_cpu, &self.original_affinity) {
            let _ = sysinfo::set_thread_affinity(orig);
        }
    }

    /// Fija el hilo a `--pin-cpu` para implementaciones de un hilo; para las
    /// multihilo restaura la afinidad original. Devuelve la CPU fijada.
    fn apply_pinning(&self, imp: &Implementation) -> Result<Option<usize>, Box<dyn Error>> {
        let Some(cpu) = self.config.pin_cpu else {
            return Ok(None);
        };
        match imp.threading {
            Threading::Single => {
                sysinfo::set_thread_affinity(&[cpu])
                    .map_err(|e| format!("no se pudo fijar el hilo a la CPU {cpu}: {e}"))?;
                Ok(Some(cpu))
            }
            Threading::Multi => {
                self.restore_affinity();
                Ok(None)
            }
        }
    }

    fn run_size(&mut self, n: usize, impls: &[Implementation], outcome: &mut RunOutcome) -> Result<(), Box<dyn Error>> {
        let bytes = n * n * std::mem::size_of::<f32>();
        say!();
        say!(
            "== N = {n}  ({:.1} MiB por matriz, {:.3} GFLOP por producto) ==",
            bytes as f64 / (1024.0 * 1024.0),
            2.0 * (n as f64).powi(3) / 1e9
        );
        self.set_context("-", n, "setup", 0);
        let a = Matrix::random(n, self.config.seed, STREAM_A);
        let b = Matrix::random(n, self.config.seed, STREAM_B);

        let mut runs: Vec<ImplRun> = Vec::new();
        for imp in impls {
            if !imp.runs_at(n, self.config.include_slow, self.config.slow_max_n) {
                let msg = format!(
                    "{} omitida en N = {n}: es lenta para N > {} (usa --include-slow)",
                    imp.name, self.config.slow_max_n
                );
                say!("  {msg}");
                outcome.skipped.push(msg);
                continue;
            }
            runs.push(self.run_impl(imp, &a, &b, n)?);
        }

        let validations = if self.config.validate {
            self.validate_size(n, &a, &b, &runs)?
        } else {
            Vec::new()
        };
        for v in &validations {
            self.writers.validation(v)?;
        }
        let rows = summarize(n, &runs, &validations, self.config, self.run_id);
        for row in &rows {
            self.writers.summary(row)?;
        }
        outcome.summaries.extend(rows);
        outcome.validations.extend(validations);
        Ok(())
    }

    fn run_impl(&mut self, imp: &Implementation, a: &Matrix, b: &Matrix, n: usize) -> Result<ImplRun, Box<dyn Error>> {
        let threads = imp.threads(self.config.mt_threads);
        let reps = self.config.reps_for(n);
        let warmup = self.config.warmup;
        let pinned = self.apply_pinning(imp)?;
        say!(
            "  {} [{}]: {warmup} calentamiento(s) + {reps} repeticion(es){}{}",
            imp.name,
            imp.kind.as_str(),
            if threads > 1 {
                format!(", {threads} hilos")
            } else {
                String::new()
            },
            pinned.map(|c| format!(", fijada a la CPU {c}")).unwrap_or_default()
        );
        let mut c = Matrix::zeros(n);
        let mut records = Vec::with_capacity(warmup + reps);
        for w in 1..=warmup {
            let rec = self.execute(imp, a, b, &mut c, n, threads, PHASE_WARMUP, w, pinned)?;
            print_progress(&rec, warmup);
            records.push(rec);
        }
        for r in 1..=reps {
            let rec = self.execute(imp, a, b, &mut c, n, threads, PHASE_MEASURED, r, pinned)?;
            print_progress(&rec, reps);
            records.push(rec);
        }
        let times: Vec<f64> = records
            .iter()
            .filter(|r| r.phase == PHASE_MEASURED)
            .map(|r| r.t_s)
            .collect();
        let stats = Stats::from_samples(&times).ok_or("se necesita al menos una repetición medida")?;
        say!(
            "    -> mediana {:.6} s | {:.3} GFLOPS | min {:.6} s | sigma {} | CV {}",
            stats.median,
            gflops(n, stats.median),
            stats.min,
            stats.std.map(|s| format!("{s:.6} s")).unwrap_or_else(|| "-".into()),
            stats
                .cv_pct()
                .map(|c| format!("{c:.2} %"))
                .unwrap_or_else(|| "-".into())
        );
        Ok(ImplRun {
            imp: *imp,
            threads,
            records,
            stats,
            output: c,
        })
    }

    /// Una ejecución: lecturas previas, llamada a C++, lecturas posteriores,
    /// checksum y escritura inmediata en raw_<run>.csv.
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &mut self,
        imp: &Implementation,
        a: &Matrix,
        b: &Matrix,
        c: &mut Matrix,
        n: usize,
        threads: usize,
        phase: &str,
        rep: usize,
        pinned: Option<usize>,
    ) -> Result<RawRecord, Box<dyn Error>> {
        if self.config.cooldown_s > 0.0 && self.seq > 0 {
            self.set_context(imp.name, n, "cooldown", rep);
            std::thread::sleep(Duration::from_secs_f64(self.config.cooldown_s));
        }
        self.set_context(imp.name, n, phase, rep);
        let before = Snapshot::take(self.sensors);
        let t0 = Instant::now();
        let t = (imp.run_timed)(a, b, c, threads)?;
        let t_call = t0.elapsed().as_secs_f64();
        let after = Snapshot::take(self.sensors);
        self.set_context(imp.name, n, "post", rep);
        let checksum = c.checksum();
        self.seq += 1;

        let same_cpu = before.cpu.is_some() && before.cpu == after.cpu;
        let process = after.process.since(&before.process);
        let thread = after.thread.since(&before.thread);
        let record = RawRecord {
            run_id: self.run_id.to_string(),
            label: self.config.label.clone(),
            seq: self.seq,
            impl_name: imp.name.to_string(),
            kind: imp.kind.as_str().to_string(),
            threads,
            n,
            phase: phase.to_string(),
            rep,
            t_s: t,
            gflops: gflops(n, t),
            t_call_s: t_call,
            checksum,
            start_unix_s: before.unix_s,
            end_unix_s: after.unix_s,
            cpu_before: before.cpu,
            cpu_after: after.cpu,
            pinned_cpu: pinned,
            freq_before_mhz: before.freq_mhz,
            freq_after_mhz: after.freq_mhz,
            temp_before_c: before.temp_c,
            temp_after_c: after.temp_c,
            throttle_core_delta: if same_cpu {
                telemetry::delta(before.throttle.core_count, after.throttle.core_count)
            } else {
                None
            },
            throttle_core_ms_delta: if same_cpu {
                telemetry::delta(before.throttle.core_ms, after.throttle.core_ms)
            } else {
                None
            },
            throttle_pkg_delta: telemetry::delta(before.throttle.pkg_count, after.throttle.pkg_count),
            throttle_pkg_ms_delta: telemetry::delta(before.throttle.pkg_ms, after.throttle.pkg_ms),
            cpu_user_s: process.user_s,
            cpu_sys_s: process.sys_s,
            ctx_vol: thread.nvcsw,
            ctx_invol: thread.nivcsw,
            minflt: thread.minflt,
            majflt: thread.majflt,
            seed: self.config.seed,
        };
        self.writers.raw(&record)?;
        if stop_requested() {
            return Err(Box::new(Interrupted));
        }
        Ok(record)
    }

    /// Valida cada implementación del N actual contra la referencia y con
    /// Freivalds. Si la referencia no se midió en esta corrida pero puede
    /// ejecutarse, se ejecuta una vez (fase `ref_only`, también queda en raw).
    fn validate_size(
        &mut self,
        n: usize,
        a: &Matrix,
        b: &Matrix,
        runs: &[ImplRun],
    ) -> Result<Vec<ValidationRow>, Box<dyn Error>> {
        let tol = self.config.tolerance;
        let ref_name = registry::preferred_reference();
        let ref_imp = registry::find(ref_name);
        let ref_allowed = ref_imp.is_some_and(|r| r.runs_at(n, self.config.include_slow, self.config.slow_max_n));
        let measured_ref = runs.iter().find(|r| r.imp.name == ref_name);

        let ref_only = match (ref_imp, measured_ref) {
            (Some(imp), None) if ref_allowed => {
                say!(
                    "  {}: 1 ejecución adicional solo como referencia de validación",
                    imp.name
                );
                let pinned = self.apply_pinning(&imp)?;
                let threads = imp.threads(self.config.mt_threads);
                let mut c = Matrix::zeros(n);
                let rec = self.execute(&imp, a, b, &mut c, n, threads, PHASE_REF_ONLY, 1, pinned)?;
                print_progress(&rec, 1);
                Some(c)
            }
            _ => None,
        };
        let reference: Option<&Matrix> = if ref_allowed {
            measured_ref.map(|r| &r.output).or(ref_only.as_ref())
        } else {
            None
        };

        say!(
            "  Validación (tolerancia {tol:e}): referencia {}{}",
            if reference.is_some() {
                ref_name.to_string()
            } else {
                format!("no disponible ({ref_name} omitida)")
            },
            if self.config.freivalds_vectors > 0 {
                format!(" + Freivalds con {} vector(es)", self.config.freivalds_vectors)
            } else {
                String::new()
            }
        );
        let checksum_ref = reference.map(Matrix::checksum);
        let norm_ref = reference.map(Matrix::frobenius_norm);
        let mut rows = Vec::with_capacity(runs.len());
        for run in runs {
            self.set_context(run.imp.name, n, "validation", 0);
            let is_ref = run.imp.name == ref_name && reference.is_some();
            let cmp = if is_ref {
                None
            } else {
                reference.map(|r| validate::compare(&run.output, r))
            };
            let frei = (self.config.freivalds_vectors > 0).then(|| {
                validate::freivalds(
                    a,
                    b,
                    &run.output,
                    self.config.freivalds_vectors,
                    self.config.seed ^ n as u64,
                )
            });
            let valid = (cmp.is_some() || frei.is_some())
                .then(|| cmp.is_none_or(|c| c.passes(tol)) && frei.is_none_or(|f| f <= tol));
            let reference_label = if is_ref {
                IS_REFERENCE.to_string()
            } else if reference.is_some() {
                ref_name.to_string()
            } else {
                String::new()
            };
            say!(
                "    {:<16} {:<5} {}{}",
                run.imp.name,
                match valid {
                    Some(true) => "OK",
                    Some(false) => "FALLO",
                    None => "-",
                },
                match (is_ref, cmp) {
                    (true, _) => "es la referencia".to_string(),
                    (false, Some(c)) => format!("error rel. {:.2e}, máx. abs. {:.2e}", c.rel_error, c.max_abs_error),
                    (false, None) => "sin referencia exacta".to_string(),
                },
                frei.map(|f| format!(" | Freivalds {f:.2e}")).unwrap_or_default()
            );
            rows.push(ValidationRow {
                run_id: self.run_id.to_string(),
                label: self.config.label.clone(),
                n,
                impl_name: run.imp.name.to_string(),
                kind: run.imp.kind.as_str().to_string(),
                reference: reference_label,
                rel_error: cmp.map(|c| c.rel_error),
                max_abs_error: cmp.map(|c| c.max_abs_error),
                norm_fro: run.output.frobenius_norm(),
                norm_fro_ref: if is_ref { None } else { norm_ref },
                freivalds_rel_error: frei,
                freivalds_vectors: self.config.freivalds_vectors,
                tolerance: tol,
                valid,
                checksum: run.output.checksum(),
                checksum_ref: if is_ref { None } else { checksum_ref },
            });
        }
        Ok(rows)
    }
}

fn print_progress(rec: &RawRecord, total: usize) {
    let phase = match rec.phase.as_str() {
        PHASE_WARMUP => "warm-up",
        PHASE_MEASURED => "rep",
        PHASE_REF_ONLY => "ref",
        other => other,
    };
    say!(
        "    {:<7} {:>2}/{:<2} {:>12.6} s {:>9.3} GFLOPS | {} | {} | cpu {}{}",
        phase,
        rec.rep,
        total,
        rec.t_s,
        rec.gflops,
        rec.temp_after_c
            .map(|t| format!("{t:.0} C"))
            .unwrap_or_else(|| "- C".into()),
        rec.freq_after_mhz
            .map(|f| format!("{f:.0} MHz"))
            .unwrap_or_else(|| "- MHz".into()),
        rec.cpu_after.map(|c| c.to_string()).unwrap_or_else(|| "-".into()),
        match rec.throttle_pkg_ms_delta {
            Some(ms) if ms > 0 => format!(" | throttling {ms} ms"),
            _ => String::new(),
        }
    );
}

fn median(values: &[f64]) -> Option<f64> {
    Stats::from_samples(values).map(|s| s.median)
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Filas de summary_<run>.csv para un N.
fn summarize(
    n: usize,
    runs: &[ImplRun],
    validations: &[ValidationRow],
    config: &RunConfig,
    run_id: &str,
) -> Vec<SummaryRow> {
    let median_of = |name: &str| runs.iter().find(|r| r.imp.name == name).map(|r| r.stats.median);
    let t_ijk = median_of(CPU_IJK);
    let gflops_openblas = median_of(CPU_OPENBLAS_1T).map(|t| gflops(n, t));
    runs.iter()
        .map(|run| {
            let v = validations.iter().find(|v| v.impl_name == run.imp.name);
            let measured: Vec<&RawRecord> = run.measured().collect();
            let gf = gflops(n, run.stats.median);
            let per_rep_gflops: Vec<f64> = measured.iter().map(|r| r.gflops).collect();
            let t_calls: Vec<f64> = measured.iter().map(|r| r.t_call_s).collect();
            let temps: Vec<f64> = measured
                .iter()
                .flat_map(|r| [r.temp_before_c, r.temp_after_c])
                .flatten()
                .collect();
            let freqs: Vec<f64> = measured.iter().filter_map(|r| r.freq_after_mhz).collect();
            let throttle: Vec<i64> = measured.iter().filter_map(|r| r.throttle_pkg_ms_delta).collect();
            let first_checksum = run.records.first().map(|r| r.checksum);
            SummaryRow {
                impl_name: run.imp.name.to_string(),
                kind: run.imp.kind.as_str().to_string(),
                n,
                reps: run.stats.count,
                t_min_s: run.stats.min,
                t_median_s: run.stats.median,
                t_mean_s: run.stats.mean,
                t_std_s: run.stats.std,
                gflops: gf,
                speedup_vs_ijk: t_ijk.map(|t| t / run.stats.median),
                pct_of_openblas: gflops_openblas.map(|g| 100.0 * gf / g),
                validated_against: v.map(|v| v.reference.clone()).unwrap_or_default(),
                rel_error: v.and_then(|v| v.rel_error),
                max_abs_error: v.and_then(|v| v.max_abs_error),
                freivalds_rel_error: v.and_then(|v| v.freivalds_rel_error),
                valid: v.and_then(|v| v.valid),
                tolerance: v.map(|v| v.tolerance),
                t_max_s: run.stats.max,
                t_cv_pct: run.stats.cv_pct(),
                gflops_best: gflops(n, run.stats.min),
                gflops_mean: mean(&per_rep_gflops).unwrap_or(f64::NAN),
                t_call_median_s: median(&t_calls).unwrap_or(f64::NAN),
                threads: run.threads,
                warmup: config.warmup,
                checksum: measured.last().map(|r| r.checksum).unwrap_or(f64::NAN),
                checksum_stable: run.records.iter().all(|r| Some(r.checksum) == first_checksum),
                temp_max_c: temps.iter().copied().reduce(f64::max),
                freq_after_mean_mhz: mean(&freqs),
                throttle_pkg_ms_total: (!throttle.is_empty()).then(|| throttle.iter().sum()),
                ctx_invol_total: measured.iter().map(|r| r.ctx_invol).sum(),
                run_id: run_id.to_string(),
                label: config.label.clone(),
                seed: config.seed,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estadisticas_impar() {
        let s = Stats::from_samples(&[3.0, 1.0, 2.0]).unwrap();
        assert_eq!((s.min, s.median, s.max, s.mean), (1.0, 2.0, 3.0, 2.0));
        assert!((s.std.unwrap() - 1.0).abs() < 1e-12); // muestral: sqrt(2/2)
        assert!((s.cv_pct().unwrap() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn estadisticas_par_y_unitaria() {
        let s = Stats::from_samples(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(s.median, 2.5);
        let one = Stats::from_samples(&[7.0]).unwrap();
        assert_eq!(one.median, 7.0);
        assert_eq!(one.std, None);
        assert!(Stats::from_samples(&[]).is_none());
    }

    #[test]
    fn gflops_con_2n3() {
        // N = 1000 -> 2e9 FLOP; en 1 s son 2 GFLOPS.
        assert!((gflops(1000, 1.0) - 2.0).abs() < 1e-12);
        assert!(gflops(10, 0.0).is_nan());
    }

    #[test]
    fn regla_de_repeticiones() {
        let mut cfg = RunConfig {
            sizes: vec![],
            impls: vec![],
            reps: None,
            reps_rule: String::new(),
            warmup: 1,
            seed: 42,
            include_slow: false,
            slow_max_n: 2048,
            validate: true,
            tolerance: 1e-4,
            freivalds_vectors: 2,
            reference: String::new(),
            sample_ms: 0,
            cooldown_s: 0.0,
            pin_cpu: None,
            mt_threads: 1,
            label: String::new(),
            notes: String::new(),
            out_dir: String::new(),
        };
        assert_eq!(cfg.reps_for(1024), 5);
        assert_eq!(cfg.reps_for(4096), 3);
        cfg.reps = Some(2);
        assert_eq!(cfg.reps_for(4096), 2);
    }
}
