//! Script de compilación del arnés `matbench`.
//!
//! NO compila C++: el Makefile es la única fuente de verdad de los flags. Este
//! script solo:
//!  1. comprueba que existe `build/libmatmul_cpu.a` (si no, pide `make lib`);
//!  2. indica a Cargo cómo enlazarla, junto con libstdc++ y, con la feature
//!     `openblas`, con OpenBLAS;
//!  3. lee `build/build_config.txt` (generado por el Makefile) y expone cada
//!     valor como variable de compilación `MATBENCH_CFG_<CLAVE>`, para que el
//!     arnés registre en los resultados con qué flags se compiló el código;
//!  4. registra la versión de rustc, el perfil de Cargo y el target.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Cargo define CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .expect("harness/ debe estar dentro del repositorio")
        .to_path_buf();
    let build_dir = repo_root.join("build");
    let lib_path = build_dir.join("libmatmul_cpu.a");
    let config_path = build_dir.join("build_config.txt");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", lib_path.display());
    println!("cargo:rerun-if-changed={}", config_path.display());
    for header in ["matmul_cpu.h", "matmul_util.h", "matmul_ref.h"] {
        println!(
            "cargo:rerun-if-changed={}",
            repo_root.join("include").join(header).display()
        );
    }

    if !lib_path.exists() {
        panic!(
            "\n\nNo se encontró {}.\n\
             Compila primero la biblioteca C++ desde la raíz del repositorio:\n\n    \
             make lib\n\n\
             (o usa `make bench` / `make test`, que la compilan antes de llamar a Cargo).\n",
            lib_path.display()
        );
    }

    let config = read_config(&config_path);
    if config.is_empty() {
        println!(
            "cargo:warning=No se pudo leer {}; los flags de compilación no quedarán registrados. \
             Compila la biblioteca con `make lib`.",
            config_path.display()
        );
    }

    // --- Enlace -------------------------------------------------------------
    // Orden: primero la biblioteca estática y después las bibliotecas dinámicas
    // de las que depende (el enlazador resuelve de izquierda a derecha).
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=matmul_cpu");

    let feature_openblas = env::var_os("CARGO_FEATURE_OPENBLAS").is_some();
    let lib_has_openblas = config.get("OPENBLAS").is_some_and(|v| v == "1");
    if feature_openblas {
        if !lib_has_openblas {
            panic!(
                "\n\nLa feature `openblas` necesita la biblioteca C++ compilada con OpenBLAS:\n\n    \
                 make lib OPENBLAS=1\n\n\
                 (o usa `make bench-ref` / `make test-ref`).\n"
            );
        }
        link_openblas(config.get("OPENBLAS_LIBS").map(String::as_str).unwrap_or(""));
    }
    // std::chrono::steady_clock::now() vive en libstdc++.
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // --- Metadatos para los resultados --------------------------------------
    for (key, value) in &config {
        println!("cargo:rustc-env=MATBENCH_CFG_{key}={value}");
    }
    println!("cargo:rustc-env=MATBENCH_RUSTC_VERSION={}", rustc_version());
    for var in ["PROFILE", "OPT_LEVEL", "TARGET", "HOST", "DEBUG"] {
        println!("cargo:rustc-env=MATBENCH_{var}={}", env::var(var).unwrap_or_default());
    }
    println!("cargo:rustc-env=MATBENCH_LIB_PATH={}", lib_path.display());
    println!("cargo:rustc-env=MATBENCH_REPO_ROOT={}", repo_root.display());
}

/// Lee `CLAVE=valor` por línea (ignora vacías y comentarios con `#`).
fn read_config(path: &Path) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let Ok(text) = fs::read_to_string(path) else {
        return map;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key: String = key
                .trim()
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() {
                        c.to_ascii_uppercase()
                    } else {
                        '_'
                    }
                })
                .collect();
            map.insert(key, value.trim().to_string());
        }
    }
    map
}

/// Traduce OPENBLAS_LIBS (p. ej. "-L/usr/lib -lopenblas") a directivas de Cargo.
fn link_openblas(libs: &str) {
    let mut any_lib = false;
    for token in libs.split_whitespace() {
        if let Some(dir) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={dir}");
        } else if let Some(name) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={name}");
            any_lib = true;
        } else {
            println!("cargo:rustc-link-arg={token}");
            any_lib = true;
        }
    }
    if !any_lib {
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}

fn rustc_version() -> String {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "desconocida".to_string())
}
