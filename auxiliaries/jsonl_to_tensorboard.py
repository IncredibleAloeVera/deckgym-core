"""Replay run metrics.jsonl into TensorBoard event files (RL_ARCHITECTURE 1.5.6).

The training loop writes plain JSON lines so the crate needs no protobuf toolchain and the log
stays readable without TensorBoard; this turns them into curves when someone wants to look.

The conversion is idempotent — a run's event file is rebuilt from scratch on every start, so a
deleted or corrupt event file is never lost data, and a run can be converted while it is still
writing.

One shot, one run:

    uv run --no-project --with tensorboardX auxiliaries/jsonl_to_tensorboard.py runs/default

Every run under runs/, kept in sync, with the server attached:

    uv run --no-project --with tensorboardX --with tensorboard \
        auxiliaries/jsonl_to_tensorboard.py runs --serve
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

EVENT_GLOB = "events.out.tfevents.*"


class RunFeed:
    """One metrics.jsonl, the event directory it feeds, and how far it has been read."""

    def __init__(self, metrics: pathlib.Path, out: pathlib.Path, writer_factory) -> None:
        self.metrics = metrics
        self.out = out
        self._writer_factory = writer_factory
        self._writer = None
        self.offset = 0
        self.written = 0
        self.skipped = 0
        self.disabled = False
        self.purged = False

    def purge(self) -> None:
        """Drops the previous conversion's events. Cheap, and has to happen before a server looks.

        Rebuilt rather than appended: a second writer in a directory that already holds events
        interleaves two series, which TensorBoard renders as a sawtooth.
        """
        if self.purged or not self.out.exists():
            self.purged = True
            return
        for stale in self.out.glob(EVENT_GLOB):
            try:
                stale.unlink()
            except OSError as err:
                # Windows refuses to unlink what another process holds open, which here means a
                # second converter is already following this run. Two writers is the one thing the
                # rebuild exists to prevent, so this run is left to whoever has it.
                self.disabled = True
                print(f"skipping {self.metrics}: {stale.name} is held ({err})", file=sys.stderr)
                return
        self.purged = True

    def _open(self) -> None:
        self.purge()
        if self.disabled:
            return
        self._writer = self._writer_factory(logdir=str(self.out))

    def sync(self) -> int:
        """Consumes whatever has been appended since the last call. Returns records written."""
        if self.disabled:
            return 0

        size = self.metrics.stat().st_size
        if size < self.offset:
            # A shrunk file is a new run reusing the name, so the events on disk describe something
            # else and have to go, even though a server already reading them will stall on the
            # deletion until it is restarted. Nothing kept in that directory could be right.
            self.purged = False
        if self._writer is None or size < self.offset:
            self._close()
            self.offset = 0
            self._open()
            if self._writer is None:
                return 0

        if size == self.offset:
            return 0

        with self.metrics.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()

        # A record still being written has no terminating newline yet. Stopping short of it leaves
        # the offset on a record boundary, so the line is picked up whole on the next pass rather
        # than parsed in half and lost.
        complete, newline, _partial = chunk.rpartition(b"\n")
        if not newline:
            return 0
        self.offset += len(complete) + len(newline)

        before = self.written
        for line in complete.decode("utf-8", errors="replace").splitlines():
            self._record(line)
        return self.written - before

    def _record(self, line: str) -> None:
        if not line.strip():
            return
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A run killed mid-write leaves a torn line. Everything around it is intact, so the
            # line is dropped rather than the file rejected.
            self.skipped += 1
            return

        step = record.pop("batch", None)
        if step is None:
            self.skipped += 1
            return
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._writer.add_scalar(key, value, global_step=int(step))
        self.written += 1

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def _close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    close = _close


def discover(root: pathlib.Path) -> list[pathlib.Path]:
    """Every metrics.jsonl at or under `root`, whether it is a run, a logs/ dir, or runs/."""
    if root.name == "metrics.jsonl":
        return [root] if root.is_file() else []
    direct = root / "metrics.jsonl"
    if direct.is_file():
        return [direct]
    nested = root / "logs" / "metrics.jsonl"
    if nested.is_file():
        return [nested]
    return sorted(root.glob("*/logs/metrics.jsonl")) + sorted(root.glob("*/metrics.jsonl"))


def event_dir(metrics: pathlib.Path, out: pathlib.Path | None, roots: int) -> pathlib.Path:
    if out is None:
        return metrics.parent / "tensorboard"
    # An explicit --out names one directory, so it can only mean one run.
    if roots > 1:
        raise SystemExit("--out takes a single run; drop it to sync several")
    return out


def sweep(feeds) -> int:
    """One pass over every feed. Returns how many records this pass added."""
    fresh = 0
    for feed in list(feeds.values()):
        fresh += feed.sync()
        feed.flush()
    return fresh


def serve(logdir: pathlib.Path, port: int) -> subprocess.Popen:
    """Starts the TensorBoard server on `logdir`, reloading on its own from there."""
    binary = shutil.which("tensorboard")
    command = [binary] if binary else [sys.executable, "-m", "tensorboard.main"]
    command += ["--logdir", str(logdir), "--port", str(port)]
    try:
        return subprocess.Popen(command)
    except (OSError, subprocess.SubprocessError) as err:
        raise SystemExit(f"could not start tensorboard ({err}); add --with tensorboard") from err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path, help="runs/ , runs/<name>/ , or a logs/ dir")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        help="event file directory for a single run (default: <logs>/tensorboard)",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="keep converting as the runs append (implied by --serve)",
    )
    parser.add_argument("--serve", action="store_true", help="run tensorboard alongside, and follow")
    parser.add_argument("--port", type=int, default=6006, help="tensorboard port (default: 6006)")
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="seconds between passes when following (default: 10)",
    )
    args = parser.parse_args()

    found = discover(args.run)
    if not found:
        print(f"no metrics.jsonl under {args.run}", file=sys.stderr)
        return 1

    from tensorboardX import SummaryWriter

    feeds = {
        metrics: RunFeed(metrics, event_dir(metrics, args.out, len(found)), SummaryWriter)
        for metrics in found
    }
    follow = args.follow or args.serve

    # TensorBoard pins each run to one event file and keeps serving it after it is deleted, so it
    # must not see the previous conversion's files: a server started over them freezes every curve
    # at the batch that conversion stopped on. Only the deletion has to happen first, though —
    # replaying is minutes on a long run, and holding the port shut that long looks like a crash.
    for feed in feeds.values():
        feed.purge()

    server = None
    if args.serve:
        # From here TensorBoard polls its logdir on its own, so writing the event files underneath
        # it is all the refresh there is to arrange.
        server = serve(args.run if args.out is None else args.out, args.port)
        print(f"tensorboard on http://localhost:{args.port}")

    print(f"replaying {len(feeds)} run(s)", flush=True)
    sweep(feeds)

    try:
        while follow:
            time.sleep(args.interval)

            # A run started after this script was launched should not need it restarted.
            for metrics in discover(args.run):
                if metrics not in feeds:
                    feeds[metrics] = RunFeed(
                        metrics, event_dir(metrics, None, len(feeds) + 1), SummaryWriter
                    )

            fresh = sweep(feeds)
            if fresh:
                total = sum(feed.written for feed in feeds.values())
                print(f"+{fresh} batches ({total} total)", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for feed in feeds.values():
            feed.close()
        if server is not None:
            server.terminate()

    for feed in feeds.values():
        if feed.disabled:
            continue
        note = f" ({feed.skipped} lines skipped)" if feed.skipped else ""
        print(f"{feed.written} batches -> {feed.out}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
