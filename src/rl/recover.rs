//! Catching engine panics — `RL_ARCHITECTURE.md` §1.5.5.
//!
//! The simulator asserts its own invariants with `expect`, and a rollout is the one caller that
//! reaches states nothing else does: hundreds of thousands of games, driven by a policy that
//! starts out uniform over the legal set and therefore plays lines no heuristic player ever
//! produces. A run of 10⁶ games cannot end because one of them found
//! `Active Pokemon should be there`.
//!
//! So a panic raised while advancing one game is caught, that game is thrown away, and the env
//! takes a fresh one. This is **not** an error-handling layer for the engine: a caught panic is
//! still a bug, and the point of catching it is to keep the evidence (see
//! [`crate::rl::train::crash`]) while the run continues.
//!
//! The two pieces std does not give in one call:
//!
//! - **`catch_unwind` sees the payload, not where it came from.** The location and the backtrace
//!   only exist inside the panic *hook*, which runs before the unwind. So the hook stashes them in
//!   a thread-local and [`catch`] picks them up on the way out.
//! - **The hook is global.** Installing one that swallows every panic in the process would hide
//!   the training loop's own bugs, so it only captures while [`catch`] is on the stack *of this
//!   thread*; every other panic goes to the hook that was installed before.

use std::any::Any;
use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

/// A panic that was caught instead of ending the process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnginePanic {
    /// The payload — what `panic!`/`expect` was given.
    pub message: String,
    /// `file:line:column`, as the default handler prints it.
    pub location: Option<String>,
    /// Forced, not `Backtrace::capture()`: the run that hits this is not the run that thought to
    /// set `RUST_BACKTRACE`, and a panic rare enough to survive a million games is one nobody gets
    /// to reproduce on demand.
    pub backtrace: String,
}

impl std::fmt::Display for EnginePanic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.location {
            Some(location) => write!(f, "{} (at {location})", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

thread_local! {
    /// Depth of the [`catch`] nesting on this thread; zero means "not ours, let it print".
    static ARMED: Cell<usize> = const { Cell::new(0) };
    /// What the hook saw, waiting for [`catch`] to unwind far enough to collect it.
    static CAUGHT: RefCell<Option<(Option<String>, String)>> = const { RefCell::new(None) };
}

static HOOK: Once = Once::new();

fn install_hook() {
    HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if ARMED.with(Cell::get) == 0 {
                previous(info);
                return;
            }
            let location = info
                .location()
                .map(|at| format!("{}:{}:{}", at.file(), at.line(), at.column()));
            let backtrace = Backtrace::force_capture().to_string();
            CAUGHT.with(|caught| *caught.borrow_mut() = Some((location, backtrace)));
        }));
    });
}

/// Runs `f`, turning a panic into an [`EnginePanic`].
///
/// `AssertUnwindSafe` is not a formality here: a half-applied action really can leave the state
/// inconsistent. It is sound because of what the caller does next — the game is **discarded**, so
/// the only use made of the broken state is to serialize it into a crash dump, and a dump of a
/// state that has stopped making sense is the entire point.
pub fn catch<T>(f: impl FnOnce() -> T) -> Result<T, EnginePanic> {
    install_hook();

    ARMED.with(|armed| armed.set(armed.get() + 1));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    ARMED.with(|armed| armed.set(armed.get() - 1));

    result.map_err(|payload| {
        let (location, backtrace) = CAUGHT
            .with(|caught| caught.borrow_mut().take())
            .unwrap_or((None, String::new()));
        EnginePanic {
            message: message_of(&*payload),
            location,
            backtrace,
        }
    })
}

/// The payload as text. `panic!`/`expect` produce one of these two; anything else is reported by
/// shape rather than dropped, since a payload nobody can read is still evidence that a panic
/// happened.
fn message_of(payload: &dyn Any) -> String {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        return (*text).to_string();
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return text.clone();
    }
    "panic payload of an unknown type".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_passes_through_untouched() {
        assert_eq!(catch(|| 2 + 2), Ok(4));
    }

    /// The three fields the crash dump is written from. Location and backtrace are the ones
    /// `catch_unwind` alone cannot give, and they are why this module installs a hook at all.
    #[test]
    fn a_panic_comes_back_with_its_message_location_and_backtrace() {
        let caught = catch(|| -> () { panic!("engine invariant broken") }).expect_err("a panic");

        assert_eq!(caught.message, "engine invariant broken");
        let location = caught.location.expect("the hook saw a location");
        assert!(location.contains("recover.rs"), "location was {location}");
        assert!(
            !caught.backtrace.is_empty(),
            "the backtrace was forced, so it cannot be empty"
        );
    }

    /// `expect` on an `Option` — the exact shape of the panic this module exists for
    /// (`State::get_active`). Its payload is a `String`, not the `&'static str` a bare `panic!`
    /// produces, and reading only one of the two would lose precisely the interesting case.
    #[test]
    fn an_expect_payload_is_read_as_well_as_a_literal_one() {
        // Behind a function so the `None` is not a literal one: `expect` on a literal is folded
        // into a plain `panic!`, which produces the `&'static str` payload this test is not about.
        fn no_active() -> Option<u8> {
            None
        }

        let caught =
            catch(|| no_active().expect("Active Pokemon should be there")).expect_err("a panic");
        assert_eq!(caught.message, "Active Pokemon should be there");
    }

    /// The guard is scoped to the call, so a later panic is the process's business again. Asserted
    /// through the counter rather than by panicking outside the guard, which would end the test.
    #[test]
    fn the_hook_disarms_itself_when_the_guard_returns() {
        assert_eq!(ARMED.with(Cell::get), 0);
        let _ = catch(|| {
            assert_eq!(ARMED.with(Cell::get), 1);
            let _ = catch(|| assert_eq!(ARMED.with(Cell::get), 2));
        });
        assert_eq!(ARMED.with(Cell::get), 0);
    }
}
