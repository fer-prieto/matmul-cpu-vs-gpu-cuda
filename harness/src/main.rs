//! CLI del arnés `matbench`.
//!
//! Flujo: valida las opciones → recoge el entorno (env_<run>.json, estado
//! "running") → arranca la telemetría → mide (raw/summary/validation CSV,
//! escritos fila a fila) → completa el entorno (estado "completed") → tabla
//! en terminal y Markdown → regenera los consolidados (runs.csv, all_*.csv).
//!
//! Códigos de salida: 0 = todo correcto; 1 = error; 2 = opciones inválidas;
//! 3 = alguna validación falló (los resultados se guardan igualmente);
//! 130 = interrumpida con Ctrl+C (lo medido hasta entonces queda guardado).

use clap::Parser;
use matbench::bench::{self, RunConfig, Runner};
#[cfg(feature = "openblas")]
use matbench::ffi;
use matbench::registry::{self, Implementation, Threading};
use matbench::report::{self, Consolidation, RunPaths, RunWriters};
use matbench::say;
use matbench::sysinfo::{self, EnvInfo, OpenBlasInfo, RunResults};
use matbench::telemetry::{self, Sampler, Sensors};
use matbench::validate::DEFAULT_TOLERANCE;
use std::error::Error;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

const AFTER_HELP: &str = "\
Ejemplos:
  matbench                                   tamaños oficiales 1024, 2048 y 4096
  matbench --sizes 256,512 --label prueba    prueba rápida
  matbench --label oficial --notes \"AC conectado, gobernador performance\"
  matbench --pin-cpu 2                       fija el hilo de cálculo a la CPU 2
  matbench --impls cpu_ijk --sizes 4096 --include-slow --reps 1 --warmup 0
  matbench --list                            implementaciones disponibles
  matbench --consolidate                     regenera runs.csv y all_*.csv

Desde la raíz del repositorio es más cómodo usar make:
  make bench ARGS=\"--sizes 256,512 --label prueba\"";

#[derive(Parser, Debug)]
#[command(
    name = "matbench",
    version,
    about = "Arnés de benchmark de C = A x B (FP32). Unidad 1: línea base en CPU.",
    after_help = AFTER_HELP
)]
struct Cli {
    /// Tamaños N a medir, separados por comas
    #[arg(long, value_delimiter = ',', default_values_t = [1024usize, 2048, 4096])]
    sizes: Vec<usize>,

    /// Implementaciones a medir, separadas por comas [por defecto: todas; ver --list]
    #[arg(long, value_delimiter = ',')]
    impls: Option<Vec<String>>,

    /// Repeticiones medidas por (implementación, N) [por defecto: 5, y 3 para N >= 4096]
    #[arg(long)]
    reps: Option<usize>,

    /// Ejecuciones de calentamiento no medidas (quedan en raw con phase=warmup)
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Semilla del generador de A y B
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Ejecuta también las implementaciones lentas (cpu_ijk) para N > --slow-max-n
    #[arg(long)]
    include_slow: bool,

    /// Mayor N para el que se ejecutan las implementaciones lentas sin --include-slow
    #[arg(long, default_value_t = 2048)]
    slow_max_n: usize,

    /// Valida cada implementación (activado por defecto)
    #[arg(long, overrides_with = "no_validate")]
    validate: bool,

    /// Desactiva la validación
    #[arg(long, overrides_with = "validate")]
    no_validate: bool,

    /// Tolerancia del error relativo en la validación
    #[arg(long, default_value_t = DEFAULT_TOLERANCE)]
    tolerance: f64,

    /// Vectores aleatorios para la verificación de Freivalds (0 = desactivada)
    #[arg(long, default_value_t = 2)]
    freivalds_vectors: usize,

    /// Carpeta de salida [por defecto: <repositorio>/results/unidad1]
    #[arg(long)]
    out: Option<PathBuf>,

    /// Etiqueta de la corrida (va en el nombre de los archivos), p. ej. "oficial"
    #[arg(long)]
    label: Option<String>,

    /// Notas libres que se guardan en env.json (condiciones de la medición, etc.)
    #[arg(long)]
    notes: Option<String>,

    /// Intervalo de la telemetría en milisegundos (0 = desactivada)
    #[arg(long, default_value_t = 500)]
    sample_ms: u64,

    /// Pausa en segundos antes de cada ejecución (para que se enfríe la CPU)
    #[arg(long, default_value_t = 0.0)]
    cooldown_s: f64,

    /// Fija el hilo de cálculo a esta CPU lógica (solo implementaciones de un hilo)
    #[arg(long)]
    pin_cpu: Option<usize>,

    /// Hilos para las implementaciones multihilo [por defecto: todas las CPUs disponibles]
    #[arg(long)]
    mt_threads: Option<usize>,

    /// Lista las implementaciones disponibles y termina
    #[arg(long)]
    list: bool,

    /// Solo regenera los archivos consolidados (runs.csv, all_*.csv) y termina
    #[arg(long)]
    consolidate: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            report::emit_err(&format!("error: {err}"));
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, Box<dyn Error>> {
    if cli.list {
        print_list();
        return Ok(ExitCode::SUCCESS);
    }
    let out_dir = cli
        .out
        .clone()
        .unwrap_or_else(|| sysinfo::repo_root().join("results").join("unidad1"));
    if cli.consolidate {
        fs::create_dir_all(&out_dir)?;
        let c = report::consolidate(&out_dir)?;
        print_consolidation(&out_dir, &c);
        return Ok(ExitCode::SUCCESS);
    }

    // --- Opciones -------------------------------------------------------------
    let impls = resolve_impls(cli.impls.as_deref())?;
    let mut sizes: Vec<usize> = Vec::new();
    for &n in &cli.sizes {
        if n == 0 {
            return Err("los tamaños deben ser enteros mayores que 0".into());
        }
        if !sizes.contains(&n) {
            sizes.push(n);
        }
    }
    if cli.reps == Some(0) {
        return Err("--reps debe ser al menos 1".into());
    }
    if !(cli.tolerance.is_finite() && cli.tolerance > 0.0) {
        return Err("--tolerance debe ser un número positivo".into());
    }
    if !(cli.cooldown_s.is_finite() && cli.cooldown_s >= 0.0) {
        return Err("--cooldown-s debe ser un número no negativo".into());
    }

    let sensors = Sensors::discover();
    let original_affinity = sysinfo::thread_affinity();
    if let Some(cpu) = cli.pin_cpu {
        let allowed = original_affinity.as_deref().unwrap_or(&[]);
        if !allowed.contains(&cpu) {
            return Err(format!(
                "--pin-cpu {cpu}: esa CPU no está entre las permitidas ({})",
                telemetry::format_cpu_list(allowed)
            )
            .into());
        }
    }
    let mt_threads = cli
        .mt_threads
        .unwrap_or_else(|| default_mt_threads(original_affinity.as_deref()));
    if mt_threads == 0 {
        return Err("--mt-threads debe ser al menos 1".into());
    }

    // --- Identificador y archivos de la corrida ------------------------------
    let label = sanitize_label(cli.label.as_deref().unwrap_or(""));
    let started = chrono::Local::now();
    fs::create_dir_all(&out_dir)?;
    let out_dir = fs::canonicalize(&out_dir).unwrap_or(out_dir);
    let run_id = make_run_id(&started, &label, &out_dir);
    let paths = RunPaths::new(&out_dir, &run_id);
    report::set_log_file(File::create(&paths.log)?);

    let config = RunConfig {
        sizes,
        impls: impls.iter().map(|i| i.name.to_string()).collect(),
        reps: cli.reps,
        reps_rule: RunConfig::describe_reps(cli.reps),
        warmup: cli.warmup,
        seed: cli.seed,
        include_slow: cli.include_slow,
        slow_max_n: cli.slow_max_n,
        validate: !cli.no_validate,
        tolerance: cli.tolerance,
        freivalds_vectors: cli.freivalds_vectors,
        reference: registry::preferred_reference().to_string(),
        sample_ms: cli.sample_ms,
        cooldown_s: cli.cooldown_s,
        pin_cpu: cli.pin_cpu,
        mt_threads,
        label,
        notes: cli.notes.clone().unwrap_or_default(),
        out_dir: out_dir.display().to_string(),
    };

    // --- Entorno ---------------------------------------------------------------
    let mut env = sysinfo::collect(&config, &run_id, &started, &sensors);
    env.openblas = openblas_info(mt_threads);
    let has_multithread = impls.iter().any(|i| i.threading == Threading::Multi);
    env.warnings = sysinfo::warnings(&env, &config, has_multithread);
    env.files = paths.existing();
    report::write_env(&paths.env, &env)?;

    print_header(&env, &config, &impls, &paths);
    for w in &env.warnings {
        say!("Aviso: {w}");
    }

    // --- Medición --------------------------------------------------------------
    let sampler = if config.sample_ms > 0 {
        Some(Sampler::start(
            &paths.telemetry,
            &run_id,
            &sensors,
            Duration::from_millis(config.sample_ms),
        )?)
    } else {
        None
    };
    let mut writers = RunWriters::new(&paths);
    bench::install_interrupt_handler();
    let t0 = Instant::now();
    let result = Runner::new(
        &config,
        &run_id,
        &sensors,
        &mut writers,
        sampler.as_ref(),
        original_affinity.clone(),
    )
    .run(&impls);
    let telemetry_samples = match sampler {
        Some(s) => match s.stop() {
            Ok(n) => Some(n),
            Err(e) => {
                say!("Aviso: telemetría: {e}");
                None
            }
        },
        None => None,
    };
    let duration = t0.elapsed().as_secs_f64();

    // --- Cierre: entorno final, tablas y consolidación ------------------------
    let results = RunResults {
        all_valid: result.as_ref().ok().and_then(|o| o.all_valid()),
        executions: writers.raw_rows,
        summary_rows: writers.summary_rows,
        validation_rows: writers.validation_rows,
        skipped: result.as_ref().map(|o| o.skipped.clone()).unwrap_or_default(),
        telemetry_samples,
    };
    let interrupted = result.as_ref().err().is_some_and(|e| e.is::<bench::Interrupted>());
    let status = match &result {
        Ok(_) => "completed",
        Err(_) if interrupted => "interrupted",
        Err(_) => "failed",
    };
    let error = result.as_ref().err().map(|e| e.to_string());
    sysinfo::finalize(
        &mut env,
        &chrono::Local::now(),
        duration,
        &sensors,
        status,
        error,
        Some(results),
    );
    if let Ok(outcome) = &result {
        fs::write(&paths.table_md, report::markdown_table(&outcome.summaries, &env))?;
    }
    env.files = paths.existing();
    report::write_env(&paths.env, &env)?;
    let consolidation = report::consolidate(&out_dir);

    if interrupted {
        say!();
        say!(
            "Corrida interrumpida tras {duration:.1} s. Lo medido ({} ejecuciones) está guardado en {} (estado \"interrupted\").",
            writers.raw_rows,
            out_dir.display()
        );
        if let Ok(c) = &consolidation {
            print_consolidation(&out_dir, c);
        }
        return Ok(ExitCode::from(130));
    }
    let outcome = result?;
    say!();
    say!("== Resumen (GFLOPS = 2*N^3 / (t_mediana * 1e9)) ==");
    for line in report::summary_table(&outcome.summaries).lines() {
        say!("{line}");
    }
    say!(
        "Duración total: {duration:.1} s · ejecuciones registradas: {} · muestras de telemetría: {}",
        writers.raw_rows,
        telemetry_samples.map(|n| n.to_string()).unwrap_or_else(|| "-".into())
    );
    if let Some(ms) = env.thermal.throttle_pkg_ms_during_run {
        say!("Throttling térmico del paquete durante la corrida: {ms} ms");
    }
    say!();
    say!("Archivos de esta corrida ({}):", out_dir.display());
    for (_, path) in paths.all() {
        if path.exists() {
            say!(
                "  {}",
                path.file_name().map(|f| f.to_string_lossy()).unwrap_or_default()
            );
        }
    }
    match consolidation {
        Ok(c) => print_consolidation(&out_dir, &c),
        Err(e) => say!("Aviso: no se pudieron regenerar los consolidados: {e}"),
    }

    Ok(match outcome.all_valid() {
        Some(false) => {
            say!();
            say!("ATENCIÓN: alguna validación FALLÓ. Revisa validation_{run_id}.csv.");
            ExitCode::from(3)
        }
        _ => ExitCode::SUCCESS,
    })
}

/// Implementaciones pedidas con --impls, en el orden del registro.
fn resolve_impls(names: Option<&[String]>) -> Result<Vec<Implementation>, Box<dyn Error>> {
    let available = registry::available();
    let Some(names) = names else {
        return Ok(available);
    };
    let mut chosen: Vec<Implementation> = Vec::new();
    for raw in names {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        match available.iter().find(|i| i.name == name) {
            Some(imp) => {
                if !chosen.iter().any(|c| c.name == imp.name) {
                    chosen.push(*imp);
                }
            }
            None => {
                if let Some((_, how)) = registry::not_compiled().into_iter().find(|(n, _)| *n == name) {
                    return Err(format!("'{name}' no está compilada en este binario; para usarla: {how}").into());
                }
                let list: Vec<&str> = available.iter().map(|i| i.name).collect();
                return Err(format!("implementación desconocida '{name}'. Disponibles: {}", list.join(", ")).into());
            }
        }
    }
    if chosen.is_empty() {
        return Err("--impls no contiene ninguna implementación".into());
    }
    chosen.sort_by_key(|imp| available.iter().position(|a| a.name == imp.name));
    Ok(chosen)
}

/// Solo letras ASCII, dígitos, '-' y '_' (para usarla en nombres de archivo).
fn sanitize_label(text: &str) -> String {
    let mut out = String::new();
    for c in text.trim().chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').chars().take(40).collect()
}

/// `<AAAAMMDD-HHMMSS>[_<etiqueta>]`, con sufijo -2, -3... si ya existe.
fn make_run_id(started: &chrono::DateTime<chrono::Local>, label: &str, dir: &Path) -> String {
    let stamp = started.format("%Y%m%d-%H%M%S").to_string();
    let base = if label.is_empty() {
        stamp
    } else {
        format!("{stamp}_{label}")
    };
    let mut id = base.clone();
    let mut k = 2;
    while RunPaths::new(dir, &id).any_exists() {
        id = format!("{base}-{k}");
        k += 1;
    }
    id
}

fn default_mt_threads(affinity: Option<&[usize]>) -> usize {
    affinity
        .map(|a| a.len())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

/// Información de OpenBLAS para el env_<run>.json (None si no está compilado).
fn openblas_info(mt_threads: usize) -> Option<OpenBlasInfo> {
    #[cfg(feature = "openblas")]
    {
        let (config, corename, parallel_mode, num_procs) = ffi::openblas_info();
        Some(OpenBlasInfo {
            config,
            corename,
            parallel_mode,
            num_procs,
            threads_mt: mt_threads,
        })
    }
    #[cfg(not(feature = "openblas"))]
    {
        let _ = mt_threads;
        None
    }
}

fn print_list() {
    println!("Implementaciones disponibles en este binario (se miden en este orden):");
    for imp in registry::available() {
        println!(
            "  {:<16} {:<10} {:<6} {:<7} {}",
            imp.name,
            imp.kind.as_str(),
            if imp.slow { "lenta" } else { "" },
            match imp.threading {
                Threading::Single => "1 hilo",
                Threading::Multi => "multi",
            },
            imp.description
        );
    }
    let missing = registry::not_compiled();
    if !missing.is_empty() {
        println!("No compiladas en este binario:");
        for (name, how) in missing {
            println!("  {name:<16} -> {how}");
        }
    }
    println!("Referencia de validación: {}", registry::preferred_reference());
}

fn print_consolidation(dir: &Path, c: &Consolidation) {
    say!(
        "Consolidados en {}: runs.csv ({} corridas), all_raw.csv ({} filas), all_summary.csv ({} filas), all_validation.csv ({} filas)",
        dir.display(),
        c.runs,
        c.raw_rows,
        c.summary_rows,
        c.validation_rows
    );
    for s in &c.skipped_files {
        say!("  Aviso: se omitió {s}");
    }
}

fn yes_no(v: Option<bool>, yes: &str, no: &str) -> String {
    match v {
        Some(true) => yes.to_string(),
        Some(false) => no.to_string(),
        None => "?".to_string(),
    }
}

fn print_header(env: &EnvInfo, config: &RunConfig, impls: &[Implementation], paths: &RunPaths) {
    let simd = &env.cpu.simd;
    let simd_list: Vec<&str> = [
        ("SSE2", simd.sse2),
        ("AVX", simd.avx),
        ("AVX2", simd.avx2),
        ("FMA", simd.fma),
        ("AVX-512F", simd.avx512f),
    ]
    .iter()
    .filter(|(_, on)| *on)
    .map(|(n, _)| *n)
    .collect();
    let caches: Vec<String> = env
        .caches
        .iter()
        .map(|c| {
            let kind = match c.kind.as_deref() {
                Some("Data") => "d",
                Some("Instruction") => "i",
                _ => "",
            };
            format!(
                "L{}{} {}",
                c.level.map(|l| l.to_string()).unwrap_or_default(),
                kind,
                c.size.as_deref().unwrap_or("?")
            )
        })
        .collect();
    let f = &env.frequency;
    let power = match env.power.ac_online {
        Some(true) => "corriente conectada".to_string(),
        Some(false) => {
            let bat = env.power.supplies.iter().find(|s| s.kind.as_deref() == Some("Battery"));
            format!(
                "BATERÍA ({}%, {})",
                bat.and_then(|b| b.capacity_pct.clone()).unwrap_or_else(|| "?".into()),
                bat.and_then(|b| b.status.clone()).unwrap_or_else(|| "?".into())
            )
        }
        None => "sin información".to_string(),
    };

    say!("matbench {} · Unidad 1: línea base en CPU", env.build.harness_version);
    say!("Corrida      : {}", env.run_id);
    say!("Salida       : {}", paths.dir.display());
    say!(
        "CPU          : {} · {} núcleos / {} hilos · SIMD: {}",
        env.cpu.model_name.as_deref().unwrap_or("?"),
        env.cpu
            .physical_cores
            .map(|c| c.to_string())
            .unwrap_or_else(|| "?".into()),
        env.cpu.logical_cpus,
        if simd_list.is_empty() {
            "-".to_string()
        } else {
            simd_list.join(" ")
        }
    );
    say!("Cachés       : {}", caches.join(" · "));
    say!(
        "Frecuencia   : gobernador {} · driver {} · turbo {} · rango {}-{} MHz",
        f.governor.as_deref().unwrap_or("?"),
        f.driver.as_deref().unwrap_or("?"),
        yes_no(f.turbo_enabled, "activado", "desactivado"),
        f.hw_min_mhz.map(|v| format!("{v:.0}")).unwrap_or_else(|| "?".into()),
        f.hw_max_mhz.map(|v| format!("{v:.0}")).unwrap_or_else(|| "?".into())
    );
    say!("Alimentación : {power}");
    say!(
        "Temperatura  : {} ({})",
        env.thermal
            .primary_temp_start_c
            .map(|t| format!("{t:.0} °C"))
            .unwrap_or_else(|| "?".into()),
        env.thermal.primary_sensor.as_deref().unwrap_or("sin sensor")
    );
    say!(
        "Afinidad     : CPUs {}{}",
        env.process.affinity,
        config
            .pin_cpu
            .map(|c| format!(" · hilo de cálculo fijado a la CPU {c} (--pin-cpu)"))
            .unwrap_or_default()
    );
    say!(
        "Compilación  : {} · {} · -march=native = {}",
        env.build.cxx_version.as_deref().unwrap_or("?"),
        env.build.cxxflags.as_deref().unwrap_or("?"),
        env.build.march_native.as_deref().unwrap_or("?")
    );
    say!(
        "Rust         : {} · perfil {}{}",
        env.build.rustc_version.as_deref().unwrap_or("?"),
        env.build.cargo_profile.as_deref().unwrap_or("?"),
        if env.build.features.is_empty() {
            String::new()
        } else {
            format!(" · features: {}", env.build.features.join(", "))
        }
    );
    say!(
        "Git          : {}{}",
        env.git
            .commit
            .as_deref()
            .map(|c| &c[..c.len().min(12)])
            .unwrap_or("sin repositorio"),
        if env.git.dirty == Some(true) {
            " (con cambios sin commit)"
        } else {
            ""
        }
    );
    say!(
        "Tamaños      : {}",
        config
            .sizes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    say!(
        "Repeticiones : {} · calentamiento {} · semilla {}{}",
        config.reps_rule,
        config.warmup,
        config.seed,
        if config.cooldown_s > 0.0 {
            format!(" · pausa {} s", config.cooldown_s)
        } else {
            String::new()
        }
    );
    say!(
        "Implement.   : {}",
        impls
            .iter()
            .map(|i| format!(
                "{} ({}{})",
                i.name,
                i.kind.as_str(),
                if i.slow {
                    if config.include_slow {
                        ", lenta: incluida".to_string()
                    } else {
                        format!(", lenta: solo N <= {}", config.slow_max_n)
                    }
                } else {
                    String::new()
                }
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    say!(
        "Validación   : {}",
        if config.validate {
            format!(
                "contra {} + Freivalds ({} vectores) · tolerancia {:e}",
                config.reference, config.freivalds_vectors, config.tolerance
            )
        } else {
            "desactivada".to_string()
        }
    );
    say!(
        "Telemetría   : {}",
        if config.sample_ms > 0 {
            format!(
                "cada {} ms -> {}",
                config.sample_ms,
                paths
                    .telemetry
                    .file_name()
                    .map(|f| f.to_string_lossy())
                    .unwrap_or_default()
            )
        } else {
            "desactivada".to_string()
        }
    );
    if !config.notes.is_empty() {
        say!("Notas        : {}", config.notes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etiquetas_seguras() {
        assert_eq!(sanitize_label(" oficial 1 "), "oficial-1");
        assert_eq!(sanitize_label("a/b\\c"), "a-b-c");
        assert_eq!(sanitize_label("pin_cpu2"), "pin_cpu2");
        assert_eq!(sanitize_label("??"), "");
    }

    #[test]
    fn la_cli_es_valida() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn opciones_por_defecto() {
        let cli = Cli::parse_from(["matbench"]);
        assert_eq!(cli.sizes, vec![1024, 2048, 4096]);
        assert_eq!(cli.warmup, 1);
        assert_eq!(cli.seed, 42);
        assert!(!cli.no_validate);
        let cli = Cli::parse_from(["matbench", "--no-validate", "--validate", "--sizes", "64,128"]);
        assert!(!cli.no_validate);
        assert_eq!(cli.sizes, vec![64, 128]);
    }
}
