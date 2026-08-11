//! The training loop's stdout, for a person rather than for a program.
//!
//! §1.5.6's metrics all go to the JSONL and from there to TensorBoard, which is where a curve is
//! actually read. Scrolling those same numbers past a terminal for ten hours answers a different
//! question badly and the only question a watcher really has — *where am I and when does this
//! finish* — not at all. So stdout carries state, not history: a fixed block redrawn in place,
//! holding the run's position (batch, ETA, curriculum stage), how close the advance screen is to
//! firing, and the few health numbers whose *level* is worth a glance. Everything else is a curve
//! and belongs to TensorBoard.
//!
//! **Events still scroll.** A stage transition, a held-out evaluation, a pool clone, an engine
//! panic — those are the run's history and a person wants them kept, in order, above the block.
//! [`Dashboard::event`] prints one and lets the block redraw beneath it.
//!
//! **A redirected run gets the old line-per-batch format instead.** The redraw is cursor
//! repositioning, which turns a log file into a smear of escape codes; and a file is read by
//! scrolling back, which is exactly what the block throws away. So [`Dashboard`] checks for a
//! terminal once, at construction, and a piped run keeps the pre-dashboard output byte-for-byte.

use std::io::{IsTerminal, Write};

/// Cursor to the start of the line `n` up, then erase everything below it. Redrawing over the old
/// block without the erase leaves the tail of a longer previous frame on screen.
fn rewind(out: &mut impl Write, lines: usize) -> std::io::Result<()> {
    if lines > 0 {
        write!(out, "\x1b[{lines}F\x1b[0J")?;
    }
    Ok(())
}

/// What the loop knows at the end of one batch. Owned strings and plain numbers: the dashboard
/// formats, it does not reach back into the training state to ask follow-up questions.
#[derive(Debug, Clone, Default)]
pub struct Frame {
    pub run: String,
    pub batch: u64,
    pub total_batches: u64,
    pub elapsed_seconds: f64,
    /// Mean over the run, not this batch's: an ETA off a single batch swings with every GC pause.
    pub mean_batch_seconds: f64,
    pub games_per_second: f64,
    pub games: usize,

    /// `(index, count, name)`. `None` in a run without `[[curriculum.stages]]`.
    pub stage: Option<(usize, usize, String)>,
    pub floor: Option<f64>,
    /// What the screen currently reads against `floor` — the worst *anchor*, not the worst label.
    pub screen: Option<f64>,
    pub holding: Option<(usize, usize)>,
    pub cooldown_remaining: u64,

    pub window_mean: f64,
    pub window_std: f64,
    pub window_batches: usize,
    pub window_capacity: usize,

    /// The last held-out evaluation: `(batch, worst, confirmed)`. Held across batches, since it
    /// only refreshes when the gate fires and a blank row would read as "no evaluation ever ran".
    pub last_eval: Option<(u64, f64, Option<bool>)>,

    /// `(active, archived, periods, elo)`. `None` when `[pool]` is off.
    pub pool: Option<(usize, usize, u64, f64)>,

    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub grad_norm: f32,
    pub kl_magnet: Option<f32>,
    pub magnet_loss: Option<f32>,

    /// Redrawn on the way into a pause. The block is the only thing on screen while the loop
    /// waits, and one that reads exactly like a running batch — a frozen ETA, an elapsed that
    /// stopped moving — is indistinguishable from a hung run.
    pub paused: bool,
}

/// `h:mm:ss`. The JSONL keeps raw seconds — a log is read by a program and this block by a person,
/// and they do not want the same thing.
pub fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    format!(
        "{}:{:02}:{:02}",
        total / 3600,
        (total % 3600) / 60,
        total % 60
    )
}

fn percent(value: f64) -> String {
    format!("{:.1}%", 100.0 * value)
}

/// Redraws a fixed block in place on a terminal, or falls back to the pre-dashboard line-per-batch
/// output when stdout is not one.
pub struct Dashboard {
    interactive: bool,
    /// Lines the last frame occupied, so the next one can erase exactly it.
    drawn: usize,
}

impl Dashboard {
    pub fn new() -> Self {
        Dashboard {
            interactive: std::io::stdout().is_terminal(),
            drawn: 0,
        }
    }

    /// Forces the non-interactive path — for tests, and for a run that wants the plain log.
    pub fn plain() -> Self {
        Dashboard {
            interactive: false,
            drawn: 0,
        }
    }

    pub fn is_interactive(&self) -> bool {
        self.interactive
    }

    /// The column header of the fallback format. Nothing on the interactive path, which labels its
    /// own rows.
    pub fn header(&self) {
        if !self.interactive {
            println!(
                "{:>9} {:>5} {:>7} {:>6} {:>5} {:>9} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7}",
                "elapsed",
                "batch",
                "games",
                "win%",
                "±",
                "pol_loss",
                "val_loss",
                "entropy",
                "grad",
                "games/s",
                "kl",
                "sl_loss",
            );
        }
    }

    /// One line of run history, above the block. Use for anything a person would want to scroll
    /// back to: a stage transition, an evaluation, a crash.
    pub fn event(&mut self, message: impl AsRef<str>) {
        let mut out = std::io::stdout().lock();
        if self.interactive {
            let _ = rewind(&mut out, self.drawn);
            self.drawn = 0;
        }
        let _ = writeln!(out, "{}", message.as_ref());
    }

    /// Draws (or redraws) the block for one batch.
    pub fn draw(&mut self, frame: &Frame) {
        let mut out = std::io::stdout().lock();
        if !self.interactive {
            // The fallback is one row per batch, and this batch already has one: the pause is
            // redrawing a frame, not reporting a new measurement.
            if frame.paused {
                return;
            }
            let dash = |value: Option<f32>| match value {
                Some(value) => format!("{value:.4}"),
                None => "-".to_string(),
            };
            let _ = writeln!(
                out,
                "{:>9} {:>5} {:>7} {:>6.1} {:>5.1} {:>9.4} {:>8.4} {:>8.4} {:>8.3} {:>7.2} {:>7} {:>7}",
                clock(frame.elapsed_seconds),
                frame.batch,
                frame.games,
                100.0 * frame.window_mean,
                100.0 * frame.window_std,
                frame.policy_loss,
                frame.value_loss,
                frame.entropy,
                frame.grad_norm,
                frame.games_per_second,
                dash(frame.kl_magnet),
                dash(frame.magnet_loss),
            );
            return;
        }

        let _ = rewind(&mut out, self.drawn);
        let block = self.render(frame);
        let _ = write!(out, "{block}");
        let _ = out.flush();
        self.drawn = block.lines().count();
    }

    /// Erases the block for good — on the way out, so the final messages are not overwritten by a
    /// frame that no longer describes anything.
    pub fn finish(&mut self) {
        if self.interactive {
            let mut out = std::io::stdout().lock();
            let _ = rewind(&mut out, self.drawn);
            let _ = out.flush();
        }
        self.drawn = 0;
    }

    fn render(&self, frame: &Frame) -> String {
        let remaining = frame.total_batches.saturating_sub(frame.batch);
        let eta = remaining as f64 * frame.mean_batch_seconds;
        let progress = if frame.total_batches > 0 {
            frame.batch as f64 / frame.total_batches as f64
        } else {
            0.0
        };

        let mut block = String::new();
        block.push_str(&format!(
            "{}  batch {}/{}  {}  elapsed {}  eta {}{}\n",
            frame.run,
            frame.batch,
            frame.total_batches,
            bar(progress, 24),
            clock(frame.elapsed_seconds),
            clock(eta),
            // On the position row rather than in one of its own: "is it running" is the question
            // the first line already answers, and a state that scrolled past under the numbers
            // would be read once and then never again.
            if frame.paused {
                "  PAUSED (p to resume)"
            } else {
                ""
            },
        ));

        if let Some((index, count, name)) = &frame.stage {
            let floor = frame
                .floor
                .map(|floor| format!("floor {}", percent(floor)))
                .unwrap_or_else(|| "no floor".to_string());
            // `EvalGate::arm` refuses to count a partial window (a floor read off one carries a
            // wider interval than its length claims), but `EvalGate::screen` reads it anyway — so
            // between a resume and the first full window the screen shows a number above the floor
            // while `hold` structurally cannot leave 0. Saying which of the two is stalling the
            // other is the whole job of this line.
            let screen = match frame.screen {
                Some(value) if frame.window_batches < frame.window_capacity => {
                    format!("screen {} (window filling)", percent(value))
                }
                Some(value) => format!("screen {}", percent(value)),
                None => "screen -".to_string(),
            };
            block.push_str(&format!(
                "  stage    {}/{} {name}   {floor}   {screen}\n",
                index + 1,
                count,
            ));

            let hold = match frame.holding {
                Some((held, target)) => format!("hold {held}/{target}"),
                None => "hold -".to_string(),
            };
            let cooldown = match frame.cooldown_remaining {
                0 => "cooldown ready".to_string(),
                left => format!("cooldown {left}"),
            };
            block.push_str(&format!("  gate     {hold}   {cooldown}\n"));
        }

        block.push_str(&format!(
            "  window   {} ± {:.1}  over {}/{} batches\n",
            percent(frame.window_mean),
            100.0 * frame.window_std,
            frame.window_batches,
            frame.window_capacity,
        ));

        block.push_str(&match frame.last_eval {
            Some((batch, worst, confirmed)) => format!(
                "  eval     b{batch}  worst {}  {}\n",
                percent(worst),
                match confirmed {
                    Some(true) => "confirmed",
                    Some(false) => "not confirmed",
                    None => "",
                }
            ),
            None => "  eval     none yet\n".to_string(),
        });

        if let Some((active, archived, periods, elo)) = frame.pool {
            block.push_str(&format!(
                "  pool     {active} active  {archived} archived  {periods} periods   elo {elo:.0}\n",
            ));
        }

        block.push_str(&format!(
            "  speed    {:.1} games/s  {:.1} s/batch\n",
            frame.games_per_second, frame.mean_batch_seconds,
        ));
        block.push_str(&format!(
            "  step     pol {:.4}  val {:.4}  ent {:.3}  grad {:.3}{}\n",
            frame.policy_loss,
            frame.value_loss,
            frame.entropy,
            frame.grad_norm,
            match frame.kl_magnet {
                Some(kl) => format!("  kl {kl:.4}"),
                None => String::new(),
            },
        ));

        block
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Dashboard::new()
    }
}

/// A progress bar in ASCII rather than block-drawing characters: this is the one line a run's
/// output is most likely to be pasted into an issue or a chat, and those mangle box glyphs.
fn bar(fraction: f64, width: usize) -> String {
    let filled = ((fraction.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    format!("[{}{}]", "=".repeat(filled), " ".repeat(width - filled))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Frame {
        Frame {
            run: "long_v1".to_string(),
            batch: 12480,
            total_batches: 25000,
            elapsed_seconds: 18753.0,
            mean_batch_seconds: 4.7,
            games_per_second: 23.4,
            stage: Some((3, 5, "expert".to_string())),
            floor: Some(0.70),
            screen: Some(0.624),
            holding: Some((0, 50)),
            window_mean: 0.682,
            window_std: 0.041,
            window_batches: 50,
            window_capacity: 50,
            last_eval: Some((11200, 0.66, Some(false))),
            pool: Some((8, 62, 31, 1584.3)),
            ..Frame::default()
        }
    }

    /// The block answers "where am I and when does this finish" without the reader having to
    /// reconstruct anything: the stage is 1-based like the config file's stage list reads, and the
    /// ETA is already a duration rather than a batch count to divide.
    #[test]
    fn the_block_names_the_stage_the_position_and_the_eta() {
        let block = Dashboard::plain().render(&frame());

        assert!(block.contains("batch 12480/25000"), "{block}");
        assert!(block.contains("stage    4/5 expert"), "{block}");
        // 12 520 batches left at 4.7 s each.
        assert!(block.contains("eta 16:20:44"), "{block}");
        assert!(block.contains("floor 70.0%"), "{block}");
        assert!(block.contains("screen 62.4%"), "{block}");
        assert!(block.contains("hold 0/50"), "{block}");
        assert!(block.contains("cooldown ready"), "{block}");
    }

    /// The gate cannot count a partial window, so a resumed run shows a screen well above its floor
    /// next to `hold 0/25` and nothing explains the contradiction. `runs/long_v1` read
    /// `screen 84.4% / floor 75.0% / hold 0/25` for 34 batches after a resume, which looks exactly
    /// like a broken counter.
    #[test]
    fn a_screen_read_off_a_partial_window_says_so() {
        let filling = Dashboard::plain().render(&Frame {
            screen: Some(0.844),
            floor: Some(0.75),
            holding: Some((0, 25)),
            window_batches: 17,
            window_capacity: 50,
            ..frame()
        });
        assert!(
            filling.contains("screen 84.4% (window filling)"),
            "{filling}"
        );

        let full = Dashboard::plain().render(&frame());
        assert!(!full.contains("filling"), "{full}");
    }

    /// A run without a curriculum has no stage or gate rows at all, rather than rows reading `-`:
    /// a placeholder for a mechanism the run does not have is a question the watcher has to answer
    /// before dismissing it.
    #[test]
    fn a_run_without_a_curriculum_shows_no_stage_rows() {
        let block = Dashboard::plain().render(&Frame {
            stage: None,
            ..frame()
        });

        assert!(!block.contains("stage"), "{block}");
        assert!(!block.contains("gate"), "{block}");
        assert!(block.contains("window"), "{block}");
    }

    /// The redraw erases exactly what it wrote, so the line count it records has to be the line
    /// count it emitted — an off-by-one leaves a stale row on screen every frame.
    #[test]
    fn the_recorded_height_matches_the_block() {
        let block = Dashboard::plain().render(&frame());

        assert!(block.ends_with('\n'), "every row is terminated");
        assert_eq!(block.lines().count(), 8, "{block}");
    }

    /// The loop stops redrawing while it waits, so the block left on screen is everything the
    /// watcher gets — and it has to distinguish a pause from a hang on its own.
    #[test]
    fn a_paused_block_says_so_without_changing_its_height() {
        let running = Dashboard::plain().render(&frame());
        let paused = Dashboard::plain().render(&Frame {
            paused: true,
            ..frame()
        });

        assert!(paused.contains("PAUSED"), "{paused}");
        assert!(!running.contains("PAUSED"), "{running}");
        assert_eq!(paused.lines().count(), running.lines().count());
    }

    #[test]
    fn the_bar_fills_from_empty_to_full_without_changing_width() {
        assert_eq!(bar(0.0, 4), "[    ]");
        assert_eq!(bar(0.5, 4), "[==  ]");
        assert_eq!(bar(1.0, 4), "[====]");
        // Out-of-range input is a display bug, never a panic or a ragged line.
        assert_eq!(bar(2.0, 4), "[====]");
        assert_eq!(bar(-1.0, 4), "[    ]");
    }
}
