//! Sensores y telemetría de la máquina durante la corrida.
//!
//! - [`Sensors`]: descubre al inicio qué hay disponible (temperaturas y
//!   ventiladores en `/sys/class/hwmon`, frecuencia por CPU en `cpufreq`).
//! - [`Snapshot`]: lectura puntual que el arnés toma justo antes y justo
//!   después de cada ejecución medida: CPU en la que corre el hilo, frecuencia
//!   de esa CPU, temperatura, contadores de throttling térmico y uso de
//!   recursos (`getrusage`: tiempo de CPU, cambios de contexto, fallos de página).
//! - [`Sampler`]: hilo que muestrea periódicamente (por defecto cada 500 ms) y
//!   escribe `telemetry_<run>.csv`, etiquetando cada muestra con lo que se
//!   estaba ejecutando en ese momento (implementación, N, fase, repetición).
//!
//! Todo es de solo lectura y tolerante a fallos: si un sensor no existe o no
//! se puede leer (otra máquina, VM, kernel restringido), el campo queda vacío.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Segundos desde la época Unix (con fracción).
pub fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Contenido de un archivo sin espacios al principio ni al final.
pub fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

/// Entero leído de un archivo (p. ej. de sysfs).
pub fn read_i64(path: impl AsRef<Path>) -> Option<i64> {
    read_trimmed(path)?.parse().ok()
}

/// Interpreta listas de CPUs del kernel: "0-3", "0,2-5", "7".
pub fn parse_cpu_list(text: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.trim().parse().ok()?;
            let hi: usize = hi.trim().parse().ok()?;
            if hi < lo {
                return None;
            }
            out.extend(lo..=hi);
        } else {
            out.push(part.parse().ok()?);
        }
    }
    Some(out)
}

/// Formatea una lista de CPUs de forma compacta: [0,1,2,3,6] -> "0-3,6".
pub fn format_cpu_list(cpus: &[usize]) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < cpus.len() {
        let start = cpus[i];
        let mut end = start;
        while i + 1 < cpus.len() && cpus[i + 1] == end + 1 {
            i += 1;
            end = cpus[i];
        }
        parts.push(if start == end {
            start.to_string()
        } else {
            format!("{start}-{end}")
        });
        i += 1;
    }
    parts.join(",")
}

/// CPUs lógicas en línea.
pub fn online_cpus() -> Vec<usize> {
    read_trimmed("/sys/devices/system/cpu/online")
        .and_then(|s| parse_cpu_list(&s))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            (0..n).collect()
        })
}

/// Un sensor de hwmon (temperatura o ventilador).
#[derive(Debug, Clone)]
pub struct Sensor {
    /// Nombre apto para una columna CSV, p. ej. `coretemp_package_id_0`.
    pub column: String,
    /// Nombre legible, p. ej. `coretemp/Package id 0`.
    pub label: String,
    /// Archivo `*_input` de sysfs.
    pub path: PathBuf,
}

/// Sensores disponibles en la máquina (se descubren una vez al inicio).
#[derive(Debug, Clone)]
pub struct Sensors {
    /// CPUs lógicas en línea.
    pub cpus: Vec<usize>,
    /// Todas las temperaturas legibles de hwmon.
    pub temps: Vec<Sensor>,
    /// Todos los ventiladores legibles de hwmon.
    pub fans: Vec<Sensor>,
    /// Temperatura que representa a la CPU (p. ej. coretemp "Package id 0").
    pub primary_temp: Option<Sensor>,
}

impl Sensors {
    /// Explora `/sys` y elige el sensor de temperatura principal.
    pub fn discover() -> Self {
        let cpus = online_cpus();
        let (temps, fans) = hwmon_sensors();
        let primary_temp = choose_primary(&temps).or_else(thermal_zone_fallback);
        Self {
            cpus,
            temps,
            fans,
            primary_temp,
        }
    }

    /// Temperatura principal en °C.
    pub fn primary_temp_c(&self) -> Option<f64> {
        self.primary_temp.as_ref().and_then(|s| read_temp_c(&s.path))
    }
}

/// Minúsculas, solo `[a-z0-9_]`, sin guiones bajos repetidos.
fn sanitize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

/// Ordena rutas de forma "natural" (hwmon2 antes que hwmon10).
fn natural_key(name: &str) -> (String, u64) {
    let digits: String = name
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let prefix = name[..name.len() - digits.len()].to_string();
    (prefix, digits.parse().unwrap_or(0))
}

fn hwmon_sensors() -> (Vec<Sensor>, Vec<Sensor>) {
    let mut temps = Vec::new();
    let mut fans = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/hwmon") else {
        return (temps, fans);
    };
    let mut dirs: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    dirs.sort_by_key(|p| {
        natural_key(
            &p.file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
        )
    });
    for dir in dirs {
        let chip = read_trimmed(dir.join("name")).unwrap_or_else(|| "hwmon".to_string());
        let Ok(files) = fs::read_dir(&dir) else {
            continue;
        };
        let mut inputs: Vec<String> = files
            .filter_map(|f| f.ok().and_then(|f| f.file_name().into_string().ok()))
            .filter(|f| f.ends_with("_input") && (f.starts_with("temp") || f.starts_with("fan")))
            .collect();
        inputs.sort_by_key(|f| natural_key(f.trim_end_matches("_input")));
        for input in inputs {
            let path = dir.join(&input);
            if read_i64(&path).is_none() {
                continue; // no legible o sin valor
            }
            let base = input.trim_end_matches("_input").to_string();
            let label = read_trimmed(dir.join(format!("{base}_label"))).unwrap_or_else(|| base.clone());
            let sensor = Sensor {
                column: sanitize(&format!("{chip}_{label}")),
                label: format!("{chip}/{label}"),
                path,
            };
            if base.starts_with("temp") {
                temps.push(sensor);
            } else {
                fans.push(sensor);
            }
        }
    }
    dedupe_columns(&mut temps);
    dedupe_columns(&mut fans);
    (temps, fans)
}

/// Evita columnas repetidas añadiendo _2, _3...
fn dedupe_columns(sensors: &mut [Sensor]) {
    let mut seen: Vec<String> = Vec::new();
    for s in sensors.iter_mut() {
        let mut name = s.column.clone();
        let mut k = 2;
        while seen.contains(&name) {
            name = format!("{}_{k}", s.column);
            k += 1;
        }
        seen.push(name.clone());
        s.column = name;
    }
}

/// Elige la temperatura que mejor representa a la CPU.
fn choose_primary(temps: &[Sensor]) -> Option<Sensor> {
    let score = |s: &Sensor| -> u32 {
        let l = s.label.to_lowercase();
        if l.starts_with("coretemp/package id") {
            100
        } else if l.starts_with("k10temp/tctl") || l.starts_with("zenpower/tdie") {
            95
        } else if l.starts_with("k10temp/tdie") || l.starts_with("zenpower/tctl") {
            94
        } else if l.starts_with("coretemp/") || l.starts_with("k10temp/") || l.starts_with("zenpower/") {
            80
        } else if l.starts_with("cpu_thermal/") || l.starts_with("cpu/") {
            60
        } else if l.starts_with("acpitz/") {
            20
        } else {
            10
        }
    };
    let mut best: Option<(&Sensor, u32)> = None;
    for s in temps {
        let sc = score(s);
        if best.is_none_or(|(_, b)| sc > b) {
            best = Some((s, sc));
        }
    }
    best.map(|(s, _)| s.clone())
}

/// Sin hwmon: usa /sys/class/thermal (x86_pkg_temp si existe).
fn thermal_zone_fallback() -> Option<Sensor> {
    let entries = fs::read_dir("/sys/class/thermal").ok()?;
    let mut zones: Vec<(String, PathBuf)> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .is_some_and(|f| f.to_string_lossy().starts_with("thermal_zone"))
        })
        .filter_map(|p| {
            let kind = read_trimmed(p.join("type"))?;
            read_i64(p.join("temp"))?;
            Some((kind, p.join("temp")))
        })
        .collect();
    zones.sort_by_key(|(kind, _)| if kind == "x86_pkg_temp" { 0 } else { 1 });
    zones.into_iter().next().map(|(kind, path)| Sensor {
        column: sanitize(&format!("zone_{kind}")),
        label: format!("thermal_zone/{kind}"),
        path,
    })
}

/// Temperatura en °C a partir de un archivo en miligrados.
pub fn read_temp_c(path: &Path) -> Option<f64> {
    read_i64(path).map(|milli| milli as f64 / 1000.0)
}

/// Frecuencia actual de una CPU en MHz (`scaling_cur_freq` está en kHz).
pub fn cpu_freq_mhz(cpu: usize) -> Option<f64> {
    read_i64(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")).map(|khz| khz as f64 / 1000.0)
}

/// CPU lógica en la que se está ejecutando el hilo que llama.
pub fn current_cpu() -> Option<usize> {
    // SAFETY: sched_getcpu no recibe argumentos y solo devuelve un entero.
    let cpu = unsafe { libc::sched_getcpu() };
    usize::try_from(cpu).ok()
}

/// CPU en la que corre (o corrió por última vez) el hilo `tid` del proceso.
pub fn task_cpu(tid: u32) -> Option<usize> {
    let stat = fs::read_to_string(format!("/proc/self/task/{tid}/stat")).ok()?;
    // El campo 2 (nombre) va entre paréntesis y puede tener espacios: se
    // cuenta desde el último ')'. "processor" es el campo 39 → índice 36.
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().nth(36)?.parse().ok()
}

/// Contadores de throttling térmico (Intel, `thermal_throttle` en sysfs),
/// acumulados desde el arranque.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Throttle {
    /// Eventos de throttling del núcleo al que pertenece la CPU.
    pub core_count: Option<i64>,
    /// Milisegundos totales de throttling de ese núcleo.
    pub core_ms: Option<i64>,
    /// Eventos de throttling del paquete (todo el chip).
    pub pkg_count: Option<i64>,
    /// Milisegundos totales de throttling del paquete.
    pub pkg_ms: Option<i64>,
}

/// Lee los contadores de throttling vistos desde la CPU `cpu`.
pub fn throttle(cpu: usize) -> Throttle {
    let base = format!("/sys/devices/system/cpu/cpu{cpu}/thermal_throttle");
    Throttle {
        core_count: read_i64(format!("{base}/core_throttle_count")),
        core_ms: read_i64(format!("{base}/core_throttle_total_time_ms")),
        pkg_count: read_i64(format!("{base}/package_throttle_count")),
        pkg_ms: read_i64(format!("{base}/package_throttle_total_time_ms")),
    }
}

/// Diferencia `after - before` si ambos existen.
pub fn delta(before: Option<i64>, after: Option<i64>) -> Option<i64> {
    Some(after? - before?)
}

/// Uso de recursos según `getrusage`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Usage {
    /// Tiempo de CPU en modo usuario (s).
    pub user_s: f64,
    /// Tiempo de CPU en modo núcleo (s).
    pub sys_s: f64,
    /// Cambios de contexto voluntarios (el hilo cedió la CPU, p. ej. al dormir).
    pub nvcsw: i64,
    /// Cambios de contexto involuntarios (el planificador lo expulsó).
    pub nivcsw: i64,
    /// Fallos de página menores.
    pub minflt: i64,
    /// Fallos de página mayores (con E/S).
    pub majflt: i64,
}

impl Usage {
    /// Diferencia `self - earlier`.
    pub fn since(&self, earlier: &Usage) -> Usage {
        Usage {
            user_s: round_us(self.user_s - earlier.user_s),
            sys_s: round_us(self.sys_s - earlier.sys_s),
            nvcsw: self.nvcsw - earlier.nvcsw,
            nivcsw: self.nivcsw - earlier.nivcsw,
            minflt: self.minflt - earlier.minflt,
            majflt: self.majflt - earlier.majflt,
        }
    }
}

/// Redondea a microsegundos (la resolución de getrusage), para no arrastrar
/// ruido de la resta en punto flotante a los CSV.
fn round_us(seconds: f64) -> f64 {
    (seconds * 1e6).round() / 1e6
}

fn timeval_s(t: libc::timeval) -> f64 {
    t.tv_sec as f64 + t.tv_usec as f64 * 1e-6
}

fn usage(who: libc::c_int) -> Usage {
    // SAFETY: `rusage` es una estructura C plana; con ceros es válida, y
    // getrusage solo escribe en ella.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(who, &mut ru) };
    if rc != 0 {
        return Usage::default();
    }
    Usage {
        user_s: timeval_s(ru.ru_utime),
        sys_s: timeval_s(ru.ru_stime),
        nvcsw: ru.ru_nvcsw as i64,
        nivcsw: ru.ru_nivcsw as i64,
        minflt: ru.ru_minflt as i64,
        majflt: ru.ru_majflt as i64,
    }
}

/// Uso de recursos de todo el proceso (todos sus hilos).
pub fn process_usage() -> Usage {
    usage(libc::RUSAGE_SELF)
}

/// Uso de recursos solo del hilo que llama.
pub fn thread_usage() -> Usage {
    usage(libc::RUSAGE_THREAD)
}

/// Lectura puntual del estado de la máquina y del proceso.
#[derive(Debug, Clone, Copy)]
pub struct Snapshot {
    pub unix_s: f64,
    pub cpu: Option<usize>,
    pub freq_mhz: Option<f64>,
    pub temp_c: Option<f64>,
    pub throttle: Throttle,
    pub process: Usage,
    pub thread: Usage,
}

impl Snapshot {
    /// Lee todo (unos pocos archivos de sysfs y dos llamadas a getrusage).
    pub fn take(sensors: &Sensors) -> Self {
        let cpu = current_cpu();
        Self {
            unix_s: unix_now(),
            cpu,
            freq_mhz: cpu.and_then(cpu_freq_mhz),
            temp_c: sensors.primary_temp_c(),
            throttle: cpu.map(throttle).unwrap_or_default(),
            process: process_usage(),
            thread: thread_usage(),
        }
    }
}

/// Qué se está ejecutando (para etiquetar las muestras de telemetría).
#[derive(Debug, Clone)]
struct Context {
    impl_name: String,
    n: usize,
    phase: String,
    rep: usize,
}

/// Columnas fijas de telemetry_<run>.csv (después vienen las dinámicas:
/// frecuencia por CPU, cada temperatura y cada ventilador).
pub const TELEMETRY_FIXED_COLUMNS: [&str; 16] = [
    "run_id",
    "t_unix_s",
    "t_rel_s",
    "impl",
    "n",
    "phase",
    "rep",
    "compute_cpu",
    "freq_compute_mhz",
    "freq_min_mhz",
    "freq_mean_mhz",
    "freq_max_mhz",
    "temp_c",
    "load1",
    "throttle_pkg_count",
    "throttle_pkg_ms",
];

/// Hilo de muestreo periódico. Se detiene con [`Sampler::stop`] (o al soltarlo).
pub struct Sampler {
    context: Arc<Mutex<Context>>,
    stop_tx: Option<mpsc::Sender<()>>,
    handle: Option<JoinHandle<Result<u64, String>>>,
}

impl Sampler {
    /// Crea `path`, escribe la cabecera y lanza el hilo de muestreo.
    pub fn start(path: &Path, run_id: &str, sensors: &Sensors, interval: Duration) -> io::Result<Self> {
        let mut writer = csv::Writer::from_path(path).map_err(io::Error::other)?;
        let mut header: Vec<String> = TELEMETRY_FIXED_COLUMNS.iter().map(|s| s.to_string()).collect();
        header.extend(sensors.cpus.iter().map(|c| format!("freq_cpu{c}_mhz")));
        header.extend(sensors.temps.iter().map(|t| format!("temp_{}_c", t.column)));
        header.extend(sensors.fans.iter().map(|f| format!("fan_{}_rpm", f.column)));
        writer.write_record(&header).map_err(io::Error::other)?;
        writer.flush()?;

        let context = Arc::new(Mutex::new(Context {
            impl_name: "-".to_string(),
            n: 0,
            phase: "setup".to_string(),
            rep: 0,
        }));
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let ctx = Arc::clone(&context);
        let sensors = sensors.clone();
        let run_id = run_id.to_string();
        // El cómputo corre en el hilo principal, cuyo TID coincide con el PID.
        let main_tid = std::process::id();
        let interval = interval.max(Duration::from_millis(10));

        let handle =
            std::thread::Builder::new()
                .name("telemetria".to_string())
                .spawn(move || -> Result<u64, String> {
                    let start = Instant::now();
                    let mut count: u64 = 0;
                    loop {
                        let row = sample_row(&run_id, start, &ctx, &sensors, main_tid);
                        writer.write_record(&row).map_err(|e| e.to_string())?;
                        writer.flush().map_err(|e| e.to_string())?;
                        count += 1;
                        // Plazos fijos (no se acumula deriva aunque leer tarde un poco).
                        let next = start + interval.saturating_mul(u32::try_from(count).unwrap_or(u32::MAX));
                        match stop_rx.recv_timeout(next.saturating_duration_since(Instant::now())) {
                            Err(RecvTimeoutError::Timeout) => continue,
                            _ => break,
                        }
                    }
                    // Muestra final al detenerse.
                    let row = sample_row(&run_id, start, &ctx, &sensors, main_tid);
                    writer.write_record(&row).map_err(|e| e.to_string())?;
                    writer.flush().map_err(|e| e.to_string())?;
                    Ok(count + 1)
                })?;

        Ok(Self {
            context,
            stop_tx: Some(stop_tx),
            handle: Some(handle),
        })
    }

    /// Indica qué se está ejecutando ahora.
    pub fn set_context(&self, impl_name: &str, n: usize, phase: &str, rep: usize) {
        if let Ok(mut c) = self.context.lock() {
            c.impl_name.clear();
            c.impl_name.push_str(impl_name);
            c.n = n;
            c.phase.clear();
            c.phase.push_str(phase);
            c.rep = rep;
        }
    }

    /// Detiene el muestreo y devuelve el número de muestras escritas.
    pub fn stop(mut self) -> Result<u64, String> {
        self.shutdown()
    }

    fn shutdown(&mut self) -> Result<u64, String> {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        match self.handle.take() {
            Some(h) => h
                .join()
                .map_err(|_| "el hilo de telemetría terminó con un pánico".to_string())?,
            None => Ok(0),
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn fmt_opt<T: ToString>(v: Option<T>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

fn sample_row(run_id: &str, start: Instant, ctx: &Mutex<Context>, sensors: &Sensors, main_tid: u32) -> Vec<String> {
    let c = ctx.lock().map(|c| c.clone()).unwrap_or(Context {
        impl_name: "-".to_string(),
        n: 0,
        phase: "?".to_string(),
        rep: 0,
    });
    let compute_cpu = task_cpu(main_tid);
    let freqs: Vec<Option<f64>> = sensors.cpus.iter().map(|&cpu| cpu_freq_mhz(cpu)).collect();
    let valid: Vec<f64> = freqs.iter().flatten().copied().collect();
    let (fmin, fmean, fmax) = if valid.is_empty() {
        (None, None, None)
    } else {
        let min = valid.iter().copied().fold(f64::INFINITY, f64::min);
        let max = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (
            Some(min),
            Some(valid.iter().sum::<f64>() / valid.len() as f64),
            Some(max),
        )
    };
    let freq_compute = compute_cpu
        .and_then(|cpu| sensors.cpus.iter().position(|&x| x == cpu))
        .and_then(|idx| freqs[idx]);
    let load1 =
        read_trimmed("/proc/loadavg").and_then(|s| s.split_whitespace().next().and_then(|v| v.parse::<f64>().ok()));
    let thr = throttle(sensors.cpus.first().copied().unwrap_or(0));

    let mut row = vec![
        run_id.to_string(),
        unix_now().to_string(),
        start.elapsed().as_secs_f64().to_string(),
        c.impl_name,
        c.n.to_string(),
        c.phase,
        c.rep.to_string(),
        fmt_opt(compute_cpu),
        fmt_opt(freq_compute),
        fmt_opt(fmin),
        fmt_opt(fmean),
        fmt_opt(fmax),
        fmt_opt(sensors.primary_temp_c()),
        fmt_opt(load1),
        fmt_opt(thr.pkg_count),
        fmt_opt(thr.pkg_ms),
    ];
    row.extend(freqs.into_iter().map(fmt_opt));
    row.extend(sensors.temps.iter().map(|t| fmt_opt(read_temp_c(&t.path))));
    row.extend(sensors.fans.iter().map(|f| fmt_opt(read_i64(&f.path))));
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listas_de_cpus() {
        assert_eq!(parse_cpu_list("0-3"), Some(vec![0, 1, 2, 3]));
        assert_eq!(parse_cpu_list("0,2-4,7"), Some(vec![0, 2, 3, 4, 7]));
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(format_cpu_list(&[0, 1, 2, 3, 6, 8, 9]), "0-3,6,8-9");
        assert_eq!(format_cpu_list(&[]), "");
    }

    #[test]
    fn nombres_de_columna() {
        assert_eq!(sanitize("coretemp_Package id 0"), "coretemp_package_id_0");
        assert_eq!(sanitize("  a--b  "), "a_b");
    }

    #[test]
    fn orden_natural() {
        let mut v = vec!["hwmon10", "hwmon2", "hwmon1"];
        v.sort_by_key(|s| natural_key(s));
        assert_eq!(v, vec!["hwmon1", "hwmon2", "hwmon10"]);
    }

    #[test]
    fn usage_no_decrece() {
        let a = process_usage();
        let mut x = 0u64;
        for i in 0..1_000_000u64 {
            x = x.wrapping_add(i * i);
        }
        std::hint::black_box(x);
        let b = process_usage();
        let d = b.since(&a);
        assert!(d.user_s >= 0.0 && d.sys_s >= 0.0);
    }
}
