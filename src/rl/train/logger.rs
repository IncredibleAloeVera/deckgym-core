//! §1.5.6's logging bridge — one JSON object per batch, into `runs/<name>/logs/metrics.jsonl`.
//!
//! **Why JSONL and not a TensorBoard event file.** Writing `tfrecord`-framed protobuf from Rust
//! would put a protobuf toolchain in the crate for a convenience the training loop does not need,
//! and it would make the log unreadable without TensorBoard. A run instead writes plain lines that
//! `jq`, pandas or a text editor open, and `auxiliaries/jsonl_to_tensorboard.py` replays them into
//! an event file whenever someone actually wants the curves. The conversion is offline and
//! repeatable, so a lost event file is never lost data.
//!
//! **Append, never truncate.** A resume continues the same file, and the batch index is in every
//! record, so a run interrupted three times is still one series. That is also why a record is a
//! flat object of scalars and not a nested one: every consumer downstream is `key → number`.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use super::diagnostics::Scalar;

pub const METRICS_FILE: &str = "metrics.jsonl";

pub struct MetricLog {
    file: File,
}

impl MetricLog {
    /// Opens `logs/metrics.jsonl` for appending, creating it if this is a fresh run.
    pub fn open(logs: &Path) -> Result<Self, String> {
        let path = logs.join(METRICS_FILE);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
        Ok(MetricLog { file })
    }

    /// Writes one record: `batch` plus every scalar handed in.
    ///
    /// Flushed per record rather than per buffer — a batch takes seconds to minutes, so the write
    /// is free at that cadence, and a crash that loses the tail of the log would cost exactly the
    /// evidence one wants in order to explain the crash.
    pub fn record(&mut self, batch: u64, scalars: &[Scalar]) -> Result<(), String> {
        let mut object = serde_json::Map::with_capacity(scalars.len() + 1);
        object.insert("batch".to_string(), serde_json::json!(batch));
        for (name, value) in scalars {
            // Non-finite scalars are dropped rather than written: `NaN` is not JSON, and a
            // diverged loss is worth seeing as a gap in the curve rather than as a parse error
            // that costs the whole file.
            if value.is_finite() {
                object.insert(name.clone(), serde_json::json!(value));
            }
        }

        let line = serde_json::to_string(&serde_json::Value::Object(object))
            .map_err(|err| format!("failed to encode metrics: {err}"))?;
        writeln!(self.file, "{line}").map_err(|err| format!("failed to write metrics: {err}"))?;
        self.file
            .flush()
            .map_err(|err| format!("failed to flush metrics: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("deckgym-logger-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    fn lines(dir: &Path) -> Vec<serde_json::Value> {
        fs::read_to_string(dir.join(METRICS_FILE))
            .expect("log")
            .lines()
            .map(|line| serde_json::from_str(line).expect("valid json per line"))
            .collect()
    }

    #[test]
    fn each_record_is_one_flat_json_line() {
        let dir = scratch("flat");
        let mut log = MetricLog::open(&dir).expect("open");

        log.record(3, &[("loss/policy".to_string(), -0.25)])
            .expect("record");

        let records = lines(&dir);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["batch"], 3);
        assert_eq!(records[0]["loss/policy"], -0.25);
    }

    /// A resume has to extend the series, not restart it — the run is one curve however many
    /// times it was interrupted.
    #[test]
    fn reopening_appends_instead_of_truncating() {
        let dir = scratch("append");

        let mut first = MetricLog::open(&dir).expect("open");
        first.record(0, &[("a".to_string(), 1.0)]).expect("record");
        drop(first);

        let mut second = MetricLog::open(&dir).expect("reopen");
        second.record(1, &[("a".to_string(), 2.0)]).expect("record");

        let records = lines(&dir);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["batch"], 0);
        assert_eq!(records[1]["batch"], 1);
    }

    /// One diverged scalar must not cost the file: `NaN` is not JSON, so a line carrying it would
    /// fail to parse and take every other metric on that batch with it.
    #[test]
    fn a_non_finite_scalar_is_dropped_not_written() {
        let dir = scratch("nan");
        let mut log = MetricLog::open(&dir).expect("open");

        log.record(
            0,
            &[
                ("good".to_string(), 1.5),
                ("diverged".to_string(), f64::NAN),
                ("infinite".to_string(), f64::INFINITY),
            ],
        )
        .expect("record");

        let records = lines(&dir);
        assert_eq!(records[0]["good"], 1.5);
        assert!(records[0].get("diverged").is_none());
        assert!(records[0].get("infinite").is_none());
    }
}
