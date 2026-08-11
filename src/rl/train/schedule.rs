//! Multi-phase schedules for the §1.5.1 step coefficients — `RL_ARCHITECTURE.md` §1.5.5.
//!
//! §1.5.5's "short warmup + constant" is one shape among many, and the shape that a run wants is
//! not knowable before the run. So a coefficient is a **sequence of phases** rather than a value
//! or a single decay: warm up, then anneal, then hold is three phases, and so is anything else.
//!
//! **Durations are absolute or relative, and which one is read off the TOML type.** An integer is
//! a count of batches, a `"5%"` string is a fraction of the run, `"rest"` is whatever is left.
//! Relative is what survives changing `batches`; absolute is what a warmup usually wants, because
//! "500 batches to get off the initial policy" does not scale with how long the run then goes on.
//! Neither is right in general, so the file says which it meant and the parser does not guess.
//!
//! **A phase starts where the previous one ended.** Only the destination is written, which is what
//! makes a schedule editable: inserting a phase cannot leave a discontinuity, and omitting `to`
//! means "hold", so a constant tail is `{ over = "rest" }`.
//!
//! Past its last phase a schedule **holds its final value** rather than ending. A run that outlives
//! its schedule is a common consequence of raising `batches`, and stopping the coefficient dead
//! would be a worse answer than continuing flat — but [`Schedule::span`] reports the boundary so
//! the loop can print it and the mismatch is visible rather than silent.

use serde::Deserialize;

/// How a phase gets from its entry value to its destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Shape {
    #[default]
    Linear,
    /// Half a cosine — flat at both ends. The usual choice for an anneal whose endpoints should
    /// not be approached at full speed.
    Cosine,
    /// Geometric. What a learning rate crossing orders of magnitude wants: linear would spend
    /// almost the whole phase in the top decade.
    Exponential,
}

/// A phase length, before it is resolved against the run's batch count.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum DurationSpec {
    /// A batch count.
    Batches(u64),
    /// `"5%"` — a fraction of the run — or `"rest"`.
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseSpec {
    pub over: DurationSpec,
    /// Destination. Omitted means "hold what came in", which is how a constant phase is written.
    #[serde(default)]
    pub to: Option<f64>,
    #[serde(default)]
    pub shape: Shape,
}

/// A coefficient's schedule as the `.toml` writes it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleSpec {
    /// Required rather than inherited from a scalar elsewhere in `[step]`: a schedule that reads
    /// its own starting point from another key would make two lines of the file jointly own one
    /// number, and §1.5.5's rule is that the file says what the run is.
    pub start: f64,
    pub phases: Vec<PhaseSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Phase {
    /// First batch of the phase.
    start_batch: u64,
    len: u64,
    from: f64,
    to: f64,
    shape: Shape,
}

/// A resolved schedule: batch index in, coefficient out.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    start: f64,
    phases: Vec<Phase>,
}

impl Schedule {
    /// A coefficient that does not move — what an unscheduled `[step]` scalar becomes, so the
    /// learner has one kind of thing to evaluate rather than two.
    pub fn constant(value: f64) -> Self {
        Schedule {
            start: value,
            phases: Vec::new(),
        }
    }

    /// Resolves durations against a run of `total` batches.
    pub fn resolve(spec: &ScheduleSpec, total: u64) -> Result<Self, String> {
        let mut phases = Vec::with_capacity(spec.phases.len());
        let mut cursor = 0u64;
        let mut value = spec.start;

        for (index, phase) in spec.phases.iter().enumerate() {
            let last = index + 1 == spec.phases.len();
            let len = match &phase.over {
                DurationSpec::Batches(batches) => *batches,
                DurationSpec::Named(name) => {
                    let name = name.trim();
                    if name.eq_ignore_ascii_case("rest") {
                        if !last {
                            return Err(format!(
                                "phase {index}: \"rest\" has to be the last phase"
                            ));
                        }
                        total.saturating_sub(cursor)
                    } else if let Some(percent) = name.strip_suffix('%') {
                        let percent: f64 = percent
                            .trim()
                            .parse()
                            .map_err(|_| format!("phase {index}: {name:?} is not a percentage"))?;
                        if !(0.0..=100.0).contains(&percent) {
                            return Err(format!("phase {index}: {percent}% is out of range"));
                        }
                        (total as f64 * percent / 100.0).round() as u64
                    } else {
                        return Err(format!(
                            "phase {index}: {name:?} is neither a percentage nor \"rest\""
                        ));
                    }
                }
            };

            let to = phase.to.unwrap_or(value);
            if phase.shape == Shape::Exponential && (value <= 0.0 || to <= 0.0) {
                return Err(format!(
                    "phase {index}: an exponential phase cannot cross zero ({value} → {to})"
                ));
            }

            // A zero-length phase is dropped rather than rejected: "2%" of a 20-batch smoke run
            // rounds to nothing, and a config that is legal for the real run should not fail for
            // the rehearsal. Its destination still carries, so the schedule stays continuous.
            if len > 0 {
                phases.push(Phase {
                    start_batch: cursor,
                    len,
                    from: value,
                    to,
                    shape: phase.shape,
                });
                cursor += len;
            }
            value = to;
        }

        Ok(Schedule {
            start: spec.start,
            phases,
        })
    }

    /// Batches the schedule covers. Past this the final value holds.
    pub fn span(&self) -> u64 {
        self.phases
            .last()
            .map_or(0, |phase| phase.start_batch + phase.len)
    }

    /// The coefficient at `batch`.
    pub fn at(&self, batch: u64) -> f64 {
        let Some(phase) = self
            .phases
            .iter()
            .find(|phase| batch < phase.start_batch + phase.len)
        else {
            return self.phases.last().map_or(self.start, |phase| phase.to);
        };

        let t = (batch - phase.start_batch) as f64 / phase.len as f64;
        match phase.shape {
            Shape::Linear => phase.from + (phase.to - phase.from) * t,
            Shape::Cosine => {
                phase.from
                    + (phase.to - phase.from) * (1.0 - (std::f64::consts::PI * t).cos()) / 2.0
            }
            Shape::Exponential => phase.from * (phase.to / phase.from).powf(t),
        }
    }

    /// One line naming the resolved boundaries, for the run's startup banner. The resolution is
    /// where a `%` turns into a batch count, and printing it is what makes a schedule that does
    /// not fit the run visible before the run rather than after.
    pub fn describe(&self) -> String {
        if self.phases.is_empty() {
            return format!("{:.3e} (constant)", self.start);
        }
        let phases: Vec<String> = self
            .phases
            .iter()
            .map(|phase| {
                format!(
                    "{:.3e}→{:.3e} over {} ({:?})",
                    phase.from, phase.to, phase.len, phase.shape
                )
            })
            .collect();
        phases.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(toml: &str) -> ScheduleSpec {
        toml::from_str(toml).expect("spec")
    }

    #[test]
    fn a_constant_schedule_never_moves() {
        let schedule = Schedule::constant(0.01);
        assert_eq!(schedule.at(0), 0.01);
        assert_eq!(schedule.at(1_000_000), 0.01);
        assert_eq!(schedule.span(), 0);
    }

    /// The shape the user asked for: warm up, anneal, hold.
    #[test]
    fn warmup_then_anneal_then_hold() {
        let schedule = Schedule::resolve(
            &spec(
                r#"
                start = 0.0
                phases = [
                  { over = 10, to = 1.0 },
                  { over = 10, to = 0.0 },
                  { over = "rest" },
                ]
                "#,
            ),
            100,
        )
        .expect("resolve");

        assert_eq!(schedule.at(0), 0.0);
        assert_eq!(schedule.at(5), 0.5);
        assert_eq!(schedule.at(10), 1.0);
        assert_eq!(schedule.at(15), 0.5);
        assert_eq!(schedule.at(20), 0.0);
        assert_eq!(schedule.at(99), 0.0);
        assert_eq!(schedule.span(), 100);
    }

    /// Absolute and relative are told apart by the TOML type, not by a flag.
    #[test]
    fn an_integer_is_batches_and_a_percent_string_is_a_fraction() {
        let absolute = Schedule::resolve(
            &spec("start = 0.0\nphases = [{ over = 50, to = 1.0 }]"),
            1000,
        )
        .expect("absolute");
        let relative = Schedule::resolve(
            &spec("start = 0.0\nphases = [{ over = \"5%\", to = 1.0 }]"),
            1000,
        )
        .expect("relative");

        assert_eq!(absolute.span(), 50);
        assert_eq!(relative.span(), 50);

        // The point of the distinction: only the relative one follows the run's length.
        let longer = Schedule::resolve(
            &spec("start = 0.0\nphases = [{ over = \"5%\", to = 1.0 }]"),
            4000,
        )
        .expect("relative");
        assert_eq!(longer.span(), 200);
    }

    #[test]
    fn a_phase_starts_where_the_previous_one_ended() {
        let schedule = Schedule::resolve(
            &spec(
                "start = 3.0\nphases = [{ over = 4, to = 7.0 }, { over = 4 }, { over = 4, to = 1.0 }]",
            ),
            12,
        )
        .expect("resolve");

        assert_eq!(schedule.at(4), 7.0);
        // The middle phase omits `to`, so it holds.
        assert_eq!(schedule.at(6), 7.0);
        assert_eq!(schedule.at(8), 7.0);
        assert_eq!(schedule.at(10), 4.0);
    }

    #[test]
    fn an_exponential_phase_is_geometric_not_linear() {
        let schedule = Schedule::resolve(
            &spec("start = 1.0e-3\nphases = [{ over = 2, to = 1.0e-5, shape = \"exponential\" }]"),
            2,
        )
        .expect("resolve");

        // Halfway across two decades is one decade, not the arithmetic midpoint of 5.05e-4.
        assert!((schedule.at(1) - 1.0e-4).abs() < 1.0e-12);
    }

    #[test]
    fn a_cosine_phase_is_flat_at_both_ends() {
        let schedule = Schedule::resolve(
            &spec("start = 0.0\nphases = [{ over = 100, to = 1.0, shape = \"cosine\" }]"),
            100,
        )
        .expect("resolve");

        assert!((schedule.at(50) - 0.5).abs() < 1.0e-12);
        // Flat ends: the first and last tenth move less than a linear ramp would.
        assert!(schedule.at(10) < 0.1);
        assert!(schedule.at(90) > 0.9);
    }

    /// Past the end the value holds. A run outliving its schedule is what raising `batches` does,
    /// and the alternative — the coefficient stopping — would be worse.
    #[test]
    fn the_final_value_holds_past_the_span() {
        let schedule = Schedule::resolve(
            &spec("start = 1.0\nphases = [{ over = 10, to = 0.25 }]"),
            10,
        )
        .expect("resolve");

        assert_eq!(schedule.at(10), 0.25);
        assert_eq!(schedule.at(10_000), 0.25);
    }

    /// A percentage that rounds to nothing must not break the rehearsal of a config written for
    /// the real run.
    #[test]
    fn a_phase_rounding_to_zero_batches_is_dropped_not_rejected() {
        let schedule = Schedule::resolve(
            &spec("start = 0.0\nphases = [{ over = \"2%\", to = 1.0 }, { over = \"rest\" }]"),
            10,
        )
        .expect("resolve");

        assert_eq!(schedule.at(0), 1.0, "the destination still carries");
        assert_eq!(schedule.span(), 10);
    }

    #[test]
    fn structurally_invalid_specs_are_rejected() {
        let cases = [
            (
                "start = 0.0\nphases = [{ over = \"rest\" }, { over = 5 }]",
                "rest",
            ),
            ("start = 0.0\nphases = [{ over = \"soon\" }]", "percentage"),
            ("start = 0.0\nphases = [{ over = \"150%\" }]", "range"),
            (
                "start = 0.0\nphases = [{ over = 5, to = 1.0, shape = \"exponential\" }]",
                "zero",
            ),
        ];
        for (toml, expected) in cases {
            let err = Schedule::resolve(&spec(toml), 100).expect_err(toml);
            assert!(err.contains(expected), "{toml} gave {err:?}");
        }
    }
}
