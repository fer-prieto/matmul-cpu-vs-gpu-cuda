//! Recolección del entorno de prueba.
//!
//! Todo lo que hace falta para la sección "Entorno de prueba" del reporte (y
//! para poder explicar después cualquier variación entre corridas) queda en
//! `env_<run>.json`: CPU, topología, cachés, memoria, SO y kernel, frecuencia
//! y gobernador por CPU, turbo, sensores, throttling, alimentación, afinidad
//! del proceso, compilador y flags efectivos, versión de rustc, commit de git,
//! la configuración completa de la corrida y avisos.
//!
//! También contiene las funciones de afinidad de CPU (`--pin-cpu`).

use crate::bench::RunConfig;
use crate::telemetry::{self, format_cpu_list, parse_cpu_list, read_i64, read_trimmed, Sensors, Throttle};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;

/// Versión del esquema de los archivos de resultados (sube si cambian columnas).
pub const SCHEMA_VERSION: u32 = 1;

/// Entorno y metadatos completos de una corrida (se guarda como JSON).
#[derive(Debug, Clone, Serialize)]
pub struct EnvInfo {
    pub schema_version: u32,
    pub run_id: String,
    pub label: String,
    /// "running" durante la corrida, "completed" al terminar bien,
    /// "interrupted" si se detuvo con Ctrl+C y "failed" si hubo un error. Un
    /// archivo que quedó en "running" es una corrida que terminó de forma
    /// abrupta (p. ej. dos Ctrl+C o un kill).
    pub status: String,
    pub error: Option<String>,
    pub started_at: String,
    pub started_at_utc: String,
    pub finished_at: Option<String>,
    pub duration_s: Option<f64>,
    pub notes: String,
    pub config: RunConfig,
    pub host: HostInfo,
    pub cpu: CpuInfo,
    pub caches: Vec<CacheInfo>,
    pub memory: MemoryInfo,
    pub frequency: FrequencyInfo,
    pub thermal: ThermalInfo,
    pub power: PowerInfo,
    pub process: ProcessInfo,
    pub build: BuildInfo,
    pub git: GitInfo,
    pub openblas: Option<OpenBlasInfo>,
    pub results: Option<RunResults>,
    pub files: BTreeMap<String, String>,
    pub warnings: Vec<String>,
    pub raw: RawText,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostInfo {
    pub hostname: Option<String>,
    pub os: Option<String>,
    pub kernel_release: Option<String>,
    pub kernel_version: Option<String>,
    pub arch: String,
    pub kernel_cmdline: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuTopology {
    pub cpu: usize,
    pub core_id: Option<i64>,
    pub package_id: Option<i64>,
    pub thread_siblings: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimdSupport {
    pub sse2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuInfo {
    pub model_name: Option<String>,
    pub vendor: Option<String>,
    pub family: Option<String>,
    pub model: Option<String>,
    pub stepping: Option<String>,
    pub microcode: Option<String>,
    pub bogomips: Option<String>,
    pub logical_cpus: usize,
    pub online_cpus: String,
    pub physical_cores: Option<usize>,
    pub sockets: Option<usize>,
    pub threads_per_core: Option<usize>,
    pub smt_active: Option<bool>,
    pub simd: SimdSupport,
    pub flags: String,
    pub topology: Vec<CpuTopology>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheInfo {
    pub level: Option<i64>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub size: Option<String>,
    pub size_bytes: Option<u64>,
    pub ways: Option<i64>,
    pub line_size: Option<i64>,
    pub sets: Option<i64>,
    pub shared_cpu_list: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryInfo {
    pub total_bytes: Option<u64>,
    pub available_bytes_start: Option<u64>,
    pub swap_total_bytes: Option<u64>,
    pub hugepage_size_kb: Option<u64>,
    pub thp_enabled: Option<String>,
    pub thp_defrag: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FrequencyInfo {
    pub driver: Option<String>,
    /// Gobernador común a todas las CPUs, o "mixto".
    pub governor: Option<String>,
    pub governors: BTreeMap<String, String>,
    pub available_governors: Option<String>,
    pub energy_performance_preference: BTreeMap<String, String>,
    pub hw_min_mhz: Option<f64>,
    pub hw_max_mhz: Option<f64>,
    pub scaling_min_mhz: Option<f64>,
    pub scaling_max_mhz: Option<f64>,
    pub cur_mhz_start: BTreeMap<String, f64>,
    pub cur_mhz_end: BTreeMap<String, f64>,
    pub intel_pstate_status: Option<String>,
    pub intel_pstate_no_turbo: Option<String>,
    pub intel_pstate_min_perf_pct: Option<String>,
    pub intel_pstate_max_perf_pct: Option<String>,
    pub cpufreq_boost: Option<String>,
    pub turbo_enabled: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThermalInfo {
    pub primary_sensor: Option<String>,
    pub primary_temp_start_c: Option<f64>,
    pub primary_temp_end_c: Option<f64>,
    /// Todas las temperaturas de hwmon al inicio (columna → °C).
    pub sensors_start: BTreeMap<String, f64>,
    pub sensors_end: BTreeMap<String, f64>,
    /// Ventiladores al inicio (columna → RPM).
    pub fans_start: BTreeMap<String, f64>,
    pub fans_end: BTreeMap<String, f64>,
    /// Contadores de throttling por CPU al inicio y al final (acumulados desde el arranque).
    pub throttle_start: BTreeMap<String, Throttle>,
    pub throttle_end: BTreeMap<String, Throttle>,
    /// Milisegundos de throttling del paquete durante la corrida (fin - inicio).
    pub throttle_pkg_ms_during_run: Option<i64>,
    pub throttle_pkg_events_during_run: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PowerSupply {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub online: Option<String>,
    pub status: Option<String>,
    pub capacity_pct: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PowerInfo {
    /// `true` si hay un adaptador de corriente conectado; `None` si la
    /// máquina no informa (p. ej. un equipo de escritorio).
    pub ac_online: Option<bool>,
    pub supplies: Vec<PowerSupply>,
    pub ac_online_end: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessInfo {
    pub pid: u32,
    /// CPUs permitidas al empezar (refleja `taskset` si se usó).
    pub affinity: String,
    pub affinity_cpus: Vec<usize>,
    pub affinity_restricted: bool,
    pub nice: Option<i64>,
    pub loadavg_start: Option<String>,
    pub loadavg_end: Option<String>,
    pub command_line: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuildInfo {
    pub harness_version: String,
    /// Cadena incrustada en la biblioteca C++ (compilador, flags, ISA).
    pub lib_build_info: String,
    pub cxx: Option<String>,
    pub cxx_version: Option<String>,
    pub cxxflags: Option<String>,
    pub cppflags: Option<String>,
    pub march_native: Option<String>,
    pub lib_openblas: Option<String>,
    pub openblas_cflags: Option<String>,
    pub openblas_libs: Option<String>,
    pub rustc_version: Option<String>,
    pub cargo_profile: Option<String>,
    pub rust_opt_level: Option<String>,
    pub target: Option<String>,
    pub features: Vec<String>,
    pub lib_path: Option<String>,
    pub release_build: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GitInfo {
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub describe: Option<String>,
    pub dirty: Option<bool>,
    pub changed_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenBlasInfo {
    pub config: String,
    pub corename: String,
    /// 0 = secuencial, 1 = pthreads, 2 = OpenMP.
    pub parallel_mode: i32,
    pub num_procs: i32,
    pub threads_mt: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunResults {
    pub all_valid: Option<bool>,
    /// Filas de raw (calentamientos y ejecuciones de referencia incluidos).
    pub executions: usize,
    pub summary_rows: usize,
    pub validation_rows: usize,
    pub skipped: Vec<String>,
    pub telemetry_samples: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawText {
    pub lscpu: Option<String>,
    pub lscpu_extended: Option<String>,
    pub uname_a: Option<String>,
    pub os_release: Option<String>,
}

// ---------------------------------------------------------------------------
// Afinidad
// ---------------------------------------------------------------------------

/// CPUs en las que puede ejecutarse el hilo actual.
pub fn thread_affinity() -> Option<Vec<usize>> {
    // SAFETY: cpu_set_t es una máscara de bits plana; sched_getaffinity solo
    // escribe en ella y las macros CPU_* solo la leen o escriben.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return None;
        }
        let max = libc::CPU_SETSIZE as usize;
        Some((0..max).filter(|&c| libc::CPU_ISSET(c, &set)).collect())
    }
}

/// Restringe el hilo actual a las CPUs indicadas.
pub fn set_thread_affinity(cpus: &[usize]) -> io::Result<()> {
    // SAFETY: ver `thread_affinity`; pid 0 = el hilo que llama.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            libc::CPU_SET(c, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Recolección
// ---------------------------------------------------------------------------

fn run_cmd(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn opt_env(value: Option<&'static str>) -> Option<String> {
    value.map(str::to_string).filter(|s| !s.is_empty())
}

fn khz_to_mhz(v: Option<i64>) -> Option<f64> {
    v.map(|khz| khz as f64 / 1000.0)
}

fn collect_host() -> HostInfo {
    let os = fs::read_to_string("/etc/os-release").ok().and_then(|t| {
        t.lines()
            .find_map(|l| l.strip_prefix("PRETTY_NAME="))
            .map(|v| v.trim_matches('"').to_string())
    });
    HostInfo {
        hostname: read_trimmed("/proc/sys/kernel/hostname"),
        os,
        kernel_release: read_trimmed("/proc/sys/kernel/osrelease"),
        kernel_version: read_trimmed("/proc/sys/kernel/version"),
        arch: std::env::consts::ARCH.to_string(),
        kernel_cmdline: read_trimmed("/proc/cmdline"),
    }
}

fn collect_cpu() -> CpuInfo {
    let text = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let blocks: Vec<BTreeMap<String, String>> = text
        .split("\n\n")
        .filter(|b| !b.trim().is_empty())
        .map(|b| {
            b.lines()
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .collect()
        })
        .collect();
    let first = blocks.first().cloned().unwrap_or_default();
    let get = |k: &str| first.get(k).cloned().filter(|s| !s.is_empty());
    let flags = get("flags").unwrap_or_default();
    let has = |f: &str| flags.split_whitespace().any(|x| x == f);

    let mut cores = std::collections::BTreeSet::new();
    let mut sockets = std::collections::BTreeSet::new();
    for b in &blocks {
        if let (Some(p), Some(c)) = (b.get("physical id"), b.get("core id")) {
            cores.insert((p.clone(), c.clone()));
            sockets.insert(p.clone());
        }
    }
    let online = telemetry::online_cpus();
    let logical = blocks
        .iter()
        .filter(|b| b.contains_key("processor"))
        .count()
        .max(online.len());
    let physical_cores = (!cores.is_empty()).then_some(cores.len());
    let topology = online
        .iter()
        .map(|&cpu| {
            let base = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
            CpuTopology {
                cpu,
                core_id: read_i64(format!("{base}/core_id")),
                package_id: read_i64(format!("{base}/physical_package_id")),
                thread_siblings: read_trimmed(format!("{base}/thread_siblings_list")),
            }
        })
        .collect();

    CpuInfo {
        model_name: get("model name"),
        vendor: get("vendor_id"),
        family: get("cpu family"),
        model: get("model"),
        stepping: get("stepping"),
        microcode: get("microcode"),
        bogomips: get("bogomips"),
        logical_cpus: logical,
        online_cpus: format_cpu_list(&online),
        physical_cores,
        sockets: (!sockets.is_empty()).then_some(sockets.len()),
        threads_per_core: physical_cores.filter(|&c| c > 0).map(|c| logical / c),
        smt_active: read_trimmed("/sys/devices/system/cpu/smt/active").map(|v| v == "1"),
        simd: SimdSupport {
            sse2: has("sse2"),
            avx: has("avx"),
            avx2: has("avx2"),
            fma: has("fma"),
            avx512f: has("avx512f"),
        },
        flags,
        topology,
    }
}

/// "32K" -> 32768, "3072K" -> 3145728, "1M" -> 1048576.
fn parse_size(text: &str) -> Option<u64> {
    let t = text.trim();
    let (num, mult) = match t.chars().last()? {
        'K' | 'k' => (&t[..t.len() - 1], 1024),
        'M' | 'm' => (&t[..t.len() - 1], 1024 * 1024),
        'G' | 'g' => (&t[..t.len() - 1], 1024 * 1024 * 1024),
        _ => (t, 1),
    };
    num.trim().parse::<u64>().ok().map(|v| v * mult)
}

fn collect_caches() -> Vec<CacheInfo> {
    let base = Path::new("/sys/devices/system/cpu/cpu0/cache");
    let Ok(entries) = fs::read_dir(base) else {
        return Vec::new();
    };
    let mut dirs: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().is_some_and(|f| f.to_string_lossy().starts_with("index")))
        .collect();
    dirs.sort();
    dirs.into_iter()
        .map(|d| {
            let size = read_trimmed(d.join("size"));
            CacheInfo {
                level: read_i64(d.join("level")),
                kind: read_trimmed(d.join("type")),
                size_bytes: size.as_deref().and_then(parse_size),
                size,
                ways: read_i64(d.join("ways_of_associativity")),
                line_size: read_i64(d.join("coherency_line_size")),
                sets: read_i64(d.join("number_of_sets")),
                shared_cpu_list: read_trimmed(d.join("shared_cpu_list")),
            }
        })
        .collect()
}

fn meminfo() -> BTreeMap<String, u64> {
    fs::read_to_string("/proc/meminfo")
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(':')?;
            let num = v.split_whitespace().next()?.parse::<u64>().ok()?;
            Some((k.trim().to_string(), num))
        })
        .collect()
}

fn collect_memory() -> MemoryInfo {
    let m = meminfo();
    MemoryInfo {
        total_bytes: m.get("MemTotal").map(|kb| kb * 1024),
        available_bytes_start: m.get("MemAvailable").map(|kb| kb * 1024),
        swap_total_bytes: m.get("SwapTotal").map(|kb| kb * 1024),
        hugepage_size_kb: m.get("Hugepagesize").copied(),
        thp_enabled: read_trimmed("/sys/kernel/mm/transparent_hugepage/enabled"),
        thp_defrag: read_trimmed("/sys/kernel/mm/transparent_hugepage/defrag"),
    }
}

fn current_freqs(cpus: &[usize]) -> BTreeMap<String, f64> {
    cpus.iter()
        .filter_map(|&c| telemetry::cpu_freq_mhz(c).map(|f| (format!("cpu{c}"), f)))
        .collect()
}

fn collect_frequency(cpus: &[usize]) -> FrequencyInfo {
    let cpufreq = |cpu: usize, file: &str| read_trimmed(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{file}"));
    let first = cpus.first().copied().unwrap_or(0);
    let governors: BTreeMap<String, String> = cpus
        .iter()
        .filter_map(|&c| cpufreq(c, "scaling_governor").map(|g| (format!("cpu{c}"), g)))
        .collect();
    let mut distinct: Vec<&String> = governors.values().collect();
    distinct.sort();
    distinct.dedup();
    let governor = match distinct.len() {
        0 => None,
        1 => Some(distinct[0].clone()),
        _ => Some("mixto".to_string()),
    };
    let pstate = |f: &str| read_trimmed(format!("/sys/devices/system/cpu/intel_pstate/{f}"));
    let no_turbo = pstate("no_turbo");
    let boost = read_trimmed("/sys/devices/system/cpu/cpufreq/boost");
    let turbo_enabled = match (&no_turbo, &boost) {
        (Some(nt), _) => Some(nt == "0"),
        (None, Some(b)) => Some(b == "1"),
        _ => None,
    };
    let khz = |cpu: usize, f: &str| read_i64(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{f}"));
    FrequencyInfo {
        driver: cpufreq(first, "scaling_driver"),
        governor,
        governors,
        available_governors: cpufreq(first, "scaling_available_governors"),
        energy_performance_preference: cpus
            .iter()
            .filter_map(|&c| cpufreq(c, "energy_performance_preference").map(|v| (format!("cpu{c}"), v)))
            .collect(),
        hw_min_mhz: khz_to_mhz(khz(first, "cpuinfo_min_freq")),
        hw_max_mhz: khz_to_mhz(khz(first, "cpuinfo_max_freq")),
        scaling_min_mhz: khz_to_mhz(khz(first, "scaling_min_freq")),
        scaling_max_mhz: khz_to_mhz(khz(first, "scaling_max_freq")),
        cur_mhz_start: current_freqs(cpus),
        cur_mhz_end: BTreeMap::new(),
        intel_pstate_status: pstate("status"),
        intel_pstate_no_turbo: no_turbo,
        intel_pstate_min_perf_pct: pstate("min_perf_pct"),
        intel_pstate_max_perf_pct: pstate("max_perf_pct"),
        cpufreq_boost: boost,
        turbo_enabled,
    }
}

fn sensor_readings(sensors: &[telemetry::Sensor], scale_temp: bool) -> BTreeMap<String, f64> {
    sensors
        .iter()
        .filter_map(|s| {
            let v = if scale_temp {
                telemetry::read_temp_c(&s.path)
            } else {
                read_i64(&s.path).map(|v| v as f64)
            }?;
            Some((s.column.clone(), v))
        })
        .collect()
}

fn throttle_all(cpus: &[usize]) -> BTreeMap<String, Throttle> {
    cpus.iter()
        .map(|&c| (format!("cpu{c}"), telemetry::throttle(c)))
        .filter(|(_, t)| t.pkg_count.is_some() || t.core_count.is_some())
        .collect()
}

fn collect_thermal(sensors: &Sensors) -> ThermalInfo {
    ThermalInfo {
        primary_sensor: sensors.primary_temp.as_ref().map(|s| s.label.clone()),
        primary_temp_start_c: sensors.primary_temp_c(),
        primary_temp_end_c: None,
        sensors_start: sensor_readings(&sensors.temps, true),
        sensors_end: BTreeMap::new(),
        fans_start: sensor_readings(&sensors.fans, false),
        fans_end: BTreeMap::new(),
        throttle_start: throttle_all(&sensors.cpus),
        throttle_end: BTreeMap::new(),
        throttle_pkg_ms_during_run: None,
        throttle_pkg_events_during_run: None,
    }
}

fn collect_power() -> (Option<bool>, Vec<PowerSupply>) {
    let mut supplies = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/power_supply") {
        let mut dirs: Vec<_> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        dirs.sort();
        for d in dirs {
            supplies.push(PowerSupply {
                name: d
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                kind: read_trimmed(d.join("type")),
                online: read_trimmed(d.join("online")),
                status: read_trimmed(d.join("status")),
                capacity_pct: read_trimmed(d.join("capacity")),
            });
        }
    }
    let mains: Vec<&PowerSupply> = supplies.iter().filter(|s| s.kind.as_deref() == Some("Mains")).collect();
    let ac_online = (!mains.is_empty()).then(|| mains.iter().any(|s| s.online.as_deref() == Some("1")));
    (ac_online, supplies)
}

fn collect_process() -> ProcessInfo {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let affinity = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
        .map(|v| v.trim().to_string())
        .unwrap_or_default();
    let affinity_cpus = parse_cpu_list(&affinity).unwrap_or_default();
    let online = telemetry::online_cpus();
    let nice = fs::read_to_string("/proc/self/stat").ok().and_then(|s| {
        let rest = &s[s.rfind(')')? + 1..];
        // "nice" es el campo 19 → índice 16 contando desde el campo 3.
        rest.split_whitespace().nth(16)?.parse().ok()
    });
    ProcessInfo {
        pid: std::process::id(),
        affinity_restricted: !affinity_cpus.is_empty() && affinity_cpus.len() < online.len(),
        affinity,
        affinity_cpus,
        nice,
        loadavg_start: read_trimmed("/proc/loadavg"),
        loadavg_end: None,
        command_line: std::env::args().collect(),
    }
}

fn collect_build() -> BuildInfo {
    let mut features = Vec::new();
    if cfg!(feature = "openblas") {
        features.push("openblas".to_string());
    }
    BuildInfo {
        harness_version: env!("CARGO_PKG_VERSION").to_string(),
        lib_build_info: crate::ffi::build_info(),
        cxx: opt_env(option_env!("MATBENCH_CFG_CXX")),
        cxx_version: opt_env(option_env!("MATBENCH_CFG_CXX_VERSION")),
        cxxflags: opt_env(option_env!("MATBENCH_CFG_CXXFLAGS")),
        cppflags: opt_env(option_env!("MATBENCH_CFG_CPPFLAGS")),
        march_native: opt_env(option_env!("MATBENCH_CFG_MARCH_NATIVE")),
        lib_openblas: opt_env(option_env!("MATBENCH_CFG_OPENBLAS")),
        openblas_cflags: opt_env(option_env!("MATBENCH_CFG_OPENBLAS_CFLAGS")),
        openblas_libs: opt_env(option_env!("MATBENCH_CFG_OPENBLAS_LIBS")),
        rustc_version: opt_env(option_env!("MATBENCH_RUSTC_VERSION")),
        cargo_profile: opt_env(option_env!("MATBENCH_PROFILE")),
        rust_opt_level: opt_env(option_env!("MATBENCH_OPT_LEVEL")),
        target: opt_env(option_env!("MATBENCH_TARGET")),
        features,
        lib_path: opt_env(option_env!("MATBENCH_LIB_PATH")),
        release_build: !cfg!(debug_assertions),
    }
}

/// Raíz del repositorio (dos niveles por encima de harness/src).
pub fn repo_root() -> &'static Path {
    Path::new(env!("MATBENCH_REPO_ROOT"))
}

fn collect_git() -> GitInfo {
    let root = repo_root().to_string_lossy().into_owned();
    let git = |args: &[&str]| {
        let mut full = vec!["-C", root.as_str()];
        full.extend_from_slice(args);
        run_cmd("git", &full)
    };
    let porcelain = Command::new("git")
        .args(["-C", root.as_str(), "status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
    let changed_files: Vec<String> = porcelain
        .as_deref()
        .map(|p| {
            p.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        })
        .unwrap_or_default();
    GitInfo {
        commit: git(&["rev-parse", "HEAD"]),
        branch: git(&["rev-parse", "--abbrev-ref", "HEAD"]),
        describe: git(&["describe", "--tags", "--always", "--dirty"]),
        dirty: porcelain.as_ref().map(|_| !changed_files.is_empty()),
        changed_files,
    }
}

fn collect_raw() -> RawText {
    RawText {
        lscpu: run_cmd("lscpu", &[]),
        lscpu_extended: run_cmd("lscpu", &["-e"]),
        uname_a: run_cmd("uname", &["-a"]),
        os_release: fs::read_to_string("/etc/os-release").ok(),
    }
}

/// Recoge todo el entorno al inicio de la corrida (`status = "running"`).
pub fn collect(
    config: &RunConfig,
    run_id: &str,
    started: &chrono::DateTime<chrono::Local>,
    sensors: &Sensors,
) -> EnvInfo {
    let (ac_online, supplies) = collect_power();
    EnvInfo {
        schema_version: SCHEMA_VERSION,
        run_id: run_id.to_string(),
        label: config.label.clone(),
        status: "running".to_string(),
        error: None,
        started_at: started.to_rfc3339_opts(chrono::SecondsFormat::Millis, false),
        started_at_utc: started
            .with_timezone(&chrono::Utc)
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        finished_at: None,
        duration_s: None,
        notes: config.notes.clone(),
        config: config.clone(),
        host: collect_host(),
        cpu: collect_cpu(),
        caches: collect_caches(),
        memory: collect_memory(),
        frequency: collect_frequency(&sensors.cpus),
        thermal: collect_thermal(sensors),
        power: PowerInfo {
            ac_online,
            supplies,
            ac_online_end: None,
        },
        process: collect_process(),
        build: collect_build(),
        git: collect_git(),
        openblas: None,
        results: None,
        files: BTreeMap::new(),
        warnings: Vec::new(),
        raw: collect_raw(),
    }
}

/// Completa el entorno al final de la corrida (lecturas finales y estado).
pub fn finalize(
    env: &mut EnvInfo,
    finished: &chrono::DateTime<chrono::Local>,
    duration_s: f64,
    sensors: &Sensors,
    status: &str,
    error: Option<String>,
    results: Option<RunResults>,
) {
    env.status = status.to_string();
    env.error = error;
    env.finished_at = Some(finished.to_rfc3339_opts(chrono::SecondsFormat::Millis, false));
    env.duration_s = Some(duration_s);
    env.thermal.primary_temp_end_c = sensors.primary_temp_c();
    env.thermal.sensors_end = sensor_readings(&sensors.temps, true);
    env.thermal.fans_end = sensor_readings(&sensors.fans, false);
    env.thermal.throttle_end = throttle_all(&sensors.cpus);
    let pkg = |m: &BTreeMap<String, Throttle>| m.values().next().copied();
    if let (Some(s), Some(e)) = (pkg(&env.thermal.throttle_start), pkg(&env.thermal.throttle_end)) {
        env.thermal.throttle_pkg_ms_during_run = telemetry::delta(s.pkg_ms, e.pkg_ms);
        env.thermal.throttle_pkg_events_during_run = telemetry::delta(s.pkg_count, e.pkg_count);
    }
    env.frequency.cur_mhz_end = current_freqs(&sensors.cpus);
    env.process.loadavg_end = read_trimmed("/proc/loadavg");
    env.power.ac_online_end = collect_power().0;
    env.results = results;
}

/// Avisos sobre condiciones que afectan a la calidad de la medición.
pub fn warnings(env: &EnvInfo, config: &RunConfig, has_multithread: bool) -> Vec<String> {
    let mut w = Vec::new();
    if let Some(g) = &env.frequency.governor {
        if g != "performance" {
            w.push(format!(
                "El gobernador de frecuencia es '{g}'. Para las mediciones oficiales conviene 'performance': \
                 echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
            ));
        }
    }
    if env.power.ac_online == Some(false) {
        w.push("La máquina funciona con batería: conéctala a la corriente para las mediciones oficiales.".to_string());
    }
    if !env.build.release_build {
        w.push("El arnés está compilado en modo debug; usa --release (make bench ya lo hace).".to_string());
    }
    if env.process.affinity_restricted {
        w.push(format!(
            "La afinidad del proceso está restringida a las CPUs {} (¿taskset?). Queda registrado en env.json.",
            env.process.affinity
        ));
        if config.sample_ms > 0 && env.process.affinity_cpus.len() == 1 {
            w.push(
                "La telemetría comparte la única CPU permitida con el cómputo. Para fijar solo el hilo de \
                 cálculo usa --pin-cpu en lugar de taskset (o desactiva la telemetría con --sample-ms 0)."
                    .to_string(),
            );
        }
        if has_multithread {
            w.push(
                "Hay implementaciones multihilo y la afinidad está restringida: no aprovecharán todas las CPUs."
                    .to_string(),
            );
        }
    }
    let logical = env.cpu.logical_cpus.max(1) as f64;
    if let Some(load1) = env
        .process
        .loadavg_start
        .as_deref()
        .and_then(|l| l.split_whitespace().next())
        .and_then(|v| v.parse::<f64>().ok())
    {
        if load1 > 0.5 * logical {
            w.push(format!(
                "Carga del sistema alta al empezar (load1 = {load1:.2} con {logical} CPUs): cierra otras aplicaciones."
            ));
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tamanos_de_cache() {
        assert_eq!(parse_size("32K"), Some(32 * 1024));
        assert_eq!(parse_size("3072K"), Some(3072 * 1024));
        assert_eq!(parse_size("1M"), Some(1024 * 1024));
        assert_eq!(parse_size("64"), Some(64));
        assert_eq!(parse_size("x"), None);
    }

    #[test]
    fn afinidad_legible() {
        let cpus = thread_affinity().expect("sched_getaffinity");
        assert!(!cpus.is_empty());
    }
}
