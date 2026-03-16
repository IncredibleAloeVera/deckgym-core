use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

fn main() {
    #[cfg(target_os = "linux")]
    {
        // Rebuild if environment variables change
        println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        let mut paths = HashSet::new();

        // 1. Try ldconfig to find where the libraries actually live on this system
        // We look for CUDA, cuDNN libraries
        if let Ok(output) = Command::new("ldconfig").arg("-p").output() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.contains("libcudnn.so") || line.contains("libcublas.so") {
                    if let Some(path_part) = line.split("=>").last() {
                        if let Some(parent) = Path::new(path_part.trim()).parent() {
                            paths.insert(parent.to_path_buf());
                        }
                    }
                }
            }
        }

        // 2. Add standard fallback paths
        let fallbacks = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/lib/x86_64-linux-gnu",
        ];
        for fb in fallbacks {
            if Path::new(fb).exists() {
                paths.insert(Path::new(fb).to_path_buf());
            }
        }

        // 3. Emit linker arguments to embed these paths in RPATH
        // $ORIGIN allows the dynamic linker to look in the binary's directory
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        for path in paths {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
            // Also add to link search path just in case
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
}
