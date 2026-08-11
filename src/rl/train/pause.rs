//! The pause key — §1.5.5's third loop control, beside the Ctrl-C latch and the step budget.
//!
//! The reason to interrupt a ten-hour run is usually "I need the card back for twenty minutes",
//! not "I am done", and Ctrl-C answers that question badly: the resume is a fresh process, and
//! §1.5.5 is explicit that a resumed run is not the uninterrupted one — the games in flight are
//! dropped and the batch after the resume does not match the batch it replaces. A pause holds the
//! process where it is, so nothing is dropped and nothing is re-derived.
//!
//! One key does both, because the state a watcher has to keep in their head is "is it running",
//! not "which key was the other one".
//!
//! **Raw mode is deliberately not enabled.** In raw mode Ctrl-C arrives as a key event instead of
//! a signal, which would take [`super::checkpoint::Interrupt`] out of service — the graceful stop
//! is worth more than a single-keypress pause on Unix, where cooked mode holds the `p` until
//! Enter. On Windows the console queue delivers it immediately either way.

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};

/// How often [`Pause::wait`] rechecks. Long enough not to matter, short enough that the key and a
/// Ctrl-C during a pause both feel immediate.
const POLL: Duration = Duration::from_millis(100);

/// A `p`-toggled latch, read by the loop at the end of a batch.
#[derive(Clone)]
pub struct Pause(Arc<AtomicBool>);

impl Pause {
    /// `None` when stdin is not a terminal: a piped or detached run has no keyboard, and a reader
    /// thread on a closed stdin would either spin or swallow input meant for something else.
    pub fn install() -> Option<Self> {
        if !std::io::stdin().is_terminal() {
            return None;
        }
        let paused = Arc::new(AtomicBool::new(false));
        let flag = paused.clone();
        std::thread::Builder::new()
            .name("pause-key".to_string())
            .spawn(move || loop {
                match event::read() {
                    Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                        if matches!(key.code, KeyCode::Char('p') | KeyCode::Char('P')) {
                            flag.fetch_xor(true, Ordering::SeqCst);
                        }
                    }
                    Ok(_) => {}
                    // A terminal that has stopped delivering events will not start again, and
                    // retrying would spin on the same error for the rest of the run.
                    Err(_) => break,
                }
            })
            .ok()?;
        Some(Pause(paused))
    }

    /// The flag without a reader thread, for tests and for anything driving the toggle itself.
    pub fn detached() -> Self {
        Pause(Arc::new(AtomicBool::new(false)))
    }

    pub fn toggle(&self) {
        self.0.fetch_xor(true, Ordering::SeqCst);
    }

    pub fn is_paused(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Blocks while the flag is up. `true` when the key resumed the run, `false` when `stop` fired
    /// — a Ctrl-C during a pause is still a stop, and the caller has to be able to tell which of
    /// the two ended the wait.
    pub fn wait(&self, stop: impl Fn() -> bool) -> bool {
        while self.is_paused() {
            if stop() {
                return false;
            }
            std::thread::sleep(POLL);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of one key: the toggle that entered the pause is the toggle that leaves it.
    #[test]
    fn the_same_toggle_pauses_and_resumes() {
        let pause = Pause::detached();
        assert!(!pause.is_paused());

        pause.toggle();
        assert!(pause.is_paused());

        let resumer = pause.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            resumer.toggle();
        });

        assert!(pause.wait(|| false), "the key resumed the run");
        assert!(!pause.is_paused());
    }

    /// A pause must never be able to strand a run that the user is trying to stop.
    #[test]
    fn an_interrupt_ends_the_wait() {
        let pause = Pause::detached();
        pause.toggle();

        assert!(!pause.wait(|| true), "the stop signal ended the wait");
        // Left up on purpose: `wait` reports why it returned, it does not decide what happens next.
        assert!(pause.is_paused());
    }

    /// The loop calls this on every batch, paused or not.
    #[test]
    fn waiting_while_running_returns_immediately() {
        assert!(Pause::detached().wait(|| false));
    }
}
