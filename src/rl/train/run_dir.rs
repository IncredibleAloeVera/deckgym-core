//! The `runs/<name>/` half of §1.5.5 — where one run's artefacts land.
//!
//! A run is identified by its configuration, so the name lives in the `.toml` rather than on the
//! command line: the clone this module drops in `runs/<name>/config.toml` is then a faithful
//! record of the run, with no override channel able to contradict it. Two runs off one config
//! means two config files.

use std::fs;
use std::path::{Path, PathBuf};

/// The four directories of §1.5.5, plus the config clone.
pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    /// Lays out a new run and clones `source` into it.
    ///
    /// Refuses an existing directory rather than merging into it: a run costs hours, and a second
    /// launch that silently interleaved its checkpoints with the first's would corrupt both.
    /// Resuming is [`RunDir::open`].
    pub fn create(root: &Path, name: &str, source: &Path) -> Result<Self, String> {
        let run = root.join(name);
        if run.exists() {
            return Err(format!(
                "run {} already exists — rename the run, or resume it",
                run.display()
            ));
        }
        let dir = RunDir { root: run };
        for path in [dir.checkpoints(), dir.logs(), dir.eval()] {
            fs::create_dir_all(&path)
                .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
        }
        // Copied byte for byte rather than re-serialized: the comments carry the §1.5 references
        // that make the file readable, and no `Serialize` impl would keep them.
        fs::copy(source, dir.config()).map_err(|err| {
            format!(
                "failed to clone {} into {}: {err}",
                source.display(),
                dir.config().display()
            )
        })?;
        Ok(dir)
    }

    /// Reopens an existing run, for resume.
    pub fn open(root: &Path, name: &str) -> Result<Self, String> {
        let run = root.join(name);
        if !run.join("config.toml").is_file() {
            return Err(format!("no run to resume at {}", run.display()));
        }
        Ok(RunDir { root: run })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The config clone. Resuming reads *this*, never the source, so edits to `config/` mid-run
    /// cannot change what a run means after the fact.
    pub fn config(&self) -> PathBuf {
        self.root.join("config.toml")
    }

    pub fn checkpoints(&self) -> PathBuf {
        self.root.join("checkpoints")
    }

    pub fn logs(&self) -> PathBuf {
        self.root.join("logs")
    }

    pub fn eval(&self) -> PathBuf {
        self.root.join("eval")
    }

    /// §1.5.7's shards. A fourth directory beside §1.5.5's three, created lazily: a run that logs
    /// no labels should not leave an empty one suggesting it did.
    pub fn harvest(&self) -> PathBuf {
        self.root.join("harvest")
    }

    /// Dumps of the games an engine panic cost (§1.5.5). Lazy for the same reason as `harvest()`,
    /// and more strongly: the directory's existence *is* the signal that a run hit one.
    pub fn crashes(&self) -> PathBuf {
        self.root.join("crashes")
    }

    /// §1.5.2's clone archive: every best-response checkpoint the pool ever took, whether or not it
    /// still holds a slot. Lazy — a run with no pool writes none — and never pruned, because the
    /// historical slots draw from it and an evicted member has to still be there to be re-drawn.
    pub fn pool(&self) -> PathBuf {
        self.root.join("pool")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("deckgym-run-dir-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    #[test]
    fn create_lays_out_the_four_artefacts_of_1_5_5() {
        let scratch = scratch("layout");
        let source = scratch.join("source.toml");
        fs::write(&source, "[run]\nseed = 0\n").expect("source");

        let run = RunDir::create(&scratch.join("runs"), "alpha", &source).expect("create");

        assert!(run.checkpoints().is_dir());
        assert!(run.logs().is_dir());
        assert!(run.eval().is_dir());
        assert_eq!(
            fs::read_to_string(run.config()).expect("clone"),
            "[run]\nseed = 0\n"
        );
    }

    #[test]
    fn create_refuses_to_reuse_a_run_directory() {
        let scratch = scratch("clobber");
        let source = scratch.join("source.toml");
        fs::write(&source, "[run]\nseed = 0\n").expect("source");
        let runs = scratch.join("runs");

        RunDir::create(&runs, "alpha", &source).expect("first");
        assert!(RunDir::create(&runs, "alpha", &source).is_err());
    }

    #[test]
    fn open_resumes_only_a_run_that_was_created() {
        let scratch = scratch("open");
        let source = scratch.join("source.toml");
        fs::write(&source, "[run]\nseed = 0\n").expect("source");
        let runs = scratch.join("runs");

        assert!(RunDir::open(&runs, "alpha").is_err());
        RunDir::create(&runs, "alpha", &source).expect("create");
        assert!(RunDir::open(&runs, "alpha").is_ok());
    }
}
