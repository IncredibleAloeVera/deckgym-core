#!/usr/bin/env python3
"""
League Logger for training progress tracking.

Simplified metrics:
- WR Pool: Agent winrate against the PFSP historical pool (pfsp_Xk opponents)
- WR vs e2: Agent winrate specifically against e2 (expectiminimax depth 2)

Baseline Codes:
- v, w: Simple heuristic bots (ValueFunction, WeightedRandom)
- aa, er: Attach-Attack, Evolution Rusher
- e2, e3, e4: Expectiminimax (depth 2/3/4) - omniscient, excluded from global WR
- o[n][device]: ONNX model (n=index from newest, device=c/g/t for cpu/cuda/trt)
  Examples: o1t = newest ONNX on TensorRT, o2c = 2nd newest on CPU
"""

from typing import Dict, List, Optional
import unicodedata
import numpy as np
from stable_baselines3.common.logger import Logger

from deckgym.league.pool import OpponentPool


def _display_width(s: str) -> int:
    """Calculate the display width of a string in terminal columns.

    Accounts for:
    - Wide characters (CJK, emojis) that take 2 columns
    - Zero-width characters (combining marks, variation selectors)
    """
    width = 0
    for char in s:
        cat = unicodedata.category(char)
        # Skip zero-width characters (Mn=Mark Nonspacing, Me=Mark Enclosing, Cf=Format)
        if cat in ("Mn", "Me", "Cf"):
            continue
        east_asian_width = unicodedata.east_asian_width(char)
        # Wide (W) and Fullwidth (F) characters take 2 columns
        if east_asian_width in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _pad_center(s: str, width: int, fillchar: str = " ") -> str:
    """Center a string to a given display width, accounting for unicode."""
    display_w = _display_width(s)
    padding = width - display_w
    if padding <= 0:
        return s
    left = padding // 2
    right = padding - left
    return fillchar * left + s + fillchar * right


def _pad_left(s: str, width: int, fillchar: str = " ") -> str:
    """Left-align a string to a given display width, accounting for unicode."""
    display_w = _display_width(s)
    padding = width - display_w
    if padding <= 0:
        return s
    return s + fillchar * padding


def _pad_right(s: str, width: int, fillchar: str = " ") -> str:
    """Right-align a string to a given display width, accounting for unicode."""
    display_w = _display_width(s)
    padding = width - display_w
    if padding <= 0:
        return s
    return fillchar * padding + s


def _truncate_to_width(s: str, max_width: int) -> str:
    """Truncate a string to fit within max_width display columns."""
    width = 0
    result = []
    for char in s:
        cat = unicodedata.category(char)
        if cat in ("Mn", "Me", "Cf"):
            char_width = 0
        else:
            east_asian_width = unicodedata.east_asian_width(char)
            char_width = 2 if east_asian_width in ("W", "F") else 1
        if width + char_width > max_width:
            break
        result.append(char)
        width += char_width
    return "".join(result)


class LeagueLogger:
    """
    Handles logging of league metrics to TensorBoard and console.

    Simplified to two key metrics:
    - WR Pool (PFSP excluded baselines): Overall performance against the historical self-play pool
    - WR vs e2: Benchmark against optimal play
    """

    def __init__(
        self, pool: OpponentPool, logger: Optional[Logger] = None, verbose: int = 1
    ):
        self.pool = pool
        self.logger = logger
        self.verbose = verbose

    def _is_omniscient(self, baseline_code: str) -> bool:
        """Check if baseline is omniscient (e[n] except er/et)."""
        lower = baseline_code.lower()
        if lower in ["er", "et"]:
            return False
        if lower.startswith("e") and len(lower) >= 2 and lower[1:].isdigit():
            return True
        if lower.startswith("m"):
            return True
        return False

    def _is_onnx_model(self, baseline_code: str) -> bool:
        """Check if baseline is an ONNX model (o[n][device])."""
        return baseline_code.lower().startswith("o")

    def _get_winrate(self, name: str) -> float:
        """Get agent winrate against a specific opponent (cumulative stats)."""
        data = self.pool.get_data(name)
        if not data:
            return 0.5
        total = data["wins"] + data["losses"] + data["draws"]
        if total > 0:
            return data["losses"] / total  # Agent wins = opponent losses
        return 0.5

    def _get_winrate_from_rollout(
        self, name: str, rollout_results: Dict[str, Dict[str, int]]
    ) -> Optional[float]:
        """Get agent winrate from current rollout results only."""
        if name not in rollout_results:
            return None
        r = rollout_results[name]
        total = r["wins"] + r["losses"] + r["draws"]
        if total > 0:
            return r["losses"] / total  # Agent wins = opponent losses
        return None

    def get_pool_winrate(
        self, rollout_results: Optional[Dict[str, Dict[str, int]]] = None
    ) -> float:
        """ Backward compatibility: average agent winrate against non-baseline opponents (the PFSP pool). """
        names = [n for n, d in self.pool.opponents.items() if not d.get("is_baseline")]
        if not names:
            return 0.5

        if rollout_results is not None:
            winrates = []
            for n in names:
                wr = self._get_winrate_from_rollout(n, rollout_results)
                if wr is not None:
                    winrates.append(wr)
            if winrates:
                return float(np.mean(winrates))
            return 0.5
        else:
            winrates = [self._get_winrate(n) for n in names]
            return float(np.mean(winrates))

    def get_e2_winrate(
        self, rollout_results: Optional[Dict[str, Dict[str, int]]] = None
    ) -> Optional[float]:
        """Agent winrate specifically against e2."""
        if "baseline_e2" not in self.pool.opponents:
            return None
        if rollout_results is not None:
            wr = self._get_winrate_from_rollout("baseline_e2", rollout_results)
            if wr is not None:
                return wr
        return self._get_winrate("baseline_e2")


    def log_metrics(
        self,
        rollout_count: int,
        rollout_results: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """Record league metrics to TensorBoard/Console.

        Args:
            rollout_count: Current rollout number.
            rollout_results: Per-rollout stats {name: {wins, losses, draws}}.
                            If provided, metrics use per-rollout data.
                            Otherwise falls back to cumulative stats.
        """
        pool_wr = self.get_pool_winrate(rollout_results)
        e2_wr = self.get_e2_winrate(rollout_results)

        if self.logger:
            self.logger.record("pfsp/winrate_pool", pool_wr)
            if e2_wr is not None:
                self.logger.record("pfsp/winrate_e2", e2_wr)

        if self.verbose > 0:
            self._print_summary(rollout_count, pool_wr, e2_wr)

    def _print_summary(
        self, rollout_count: int, pool_wr: float, e2_wr: Optional[float]
    ):
        """Print beautiful CLI summary."""
        e2_str = f"{e2_wr:.1%}" if e2_wr is not None else "—"
        col_width = 50

        print(f"\n┌{'─'*col_width}┐")
        print(f"│{_pad_center('LEAGUE STATUS', col_width)}│")
        print(f"├{'─'*col_width}┤")

        pool_line = (
            f"  Rollout: {rollout_count:<10} Pool: {self.pool.total_count} agents"
        )
        print(f"│{_pad_left(pool_line, col_width)}│")
        print(f"├{'─'*col_width}┤")

        wr_pool_line = f"  📊 WR Pool (PFSP):        {pool_wr:>6.1%}"
        wr_e2_line = f"  🎯 WR vs e2:              {e2_str:>6}"
        print(f"│{_pad_left(wr_pool_line, col_width)}│")
        print(f"│{_pad_left(wr_e2_line, col_width)}│")
        print(f"└{'─'*col_width}┘")

    def log_detailed_info(self, rollout_results: Dict[str, Dict[str, int]]):
        """Log detailed performance against each opponent."""
        if self.verbose <= 0:
            return

        # Sort: baselines first (omniscient last), then models by age
        def sort_key(name):
            data = self.pool.opponents.get(name, {})
            is_baseline = data.get("is_baseline", False)
            code = data.get("baseline_code", "")
            is_omni = self._is_omniscient(code) if is_baseline else False
            return (not is_baseline, is_omni, data.get("added_at_step", 0))

        sorted_names = sorted(self.pool.opponents.keys(), key=sort_key)

        # Column widths
        col1_width = 25  # Opponent name
        col2_width = 22  # This Rollout
        col3_width = 22  # All-Time
        total_width = col1_width + col2_width + col3_width + 2  # +2 for inner borders

        print(f"\n┌{'─'*total_width}┐")
        print(f"│{_pad_center('DETAILED OPPONENT BREAKDOWN', total_width)}│")
        print(f"├{'─'*col1_width}┬{'─'*col2_width}┬{'─'*col3_width}┤")
        print(
            f"│{_pad_center('Opponent', col1_width)}│{_pad_center('This Rollout', col2_width)}│{_pad_center('All-Time', col3_width)}│"
        )
        print(f"├{'─'*col1_width}┼{'─'*col2_width}┼{'─'*col3_width}┤")

        for name in sorted_names:
            data = self.pool.opponents[name]

            # Display name with type indicator
            if data.get("is_baseline"):
                code = data.get("baseline_code", "?")
                if self._is_omniscient(code):
                    display = f"⚠️  {name}"
                elif self._is_onnx_model(code):
                    display = f"🤖 {name}"
                else:
                    display = f"🎮 {name}"
            else:
                display = f"📸 {name}"

            # Truncate using display width, not character count
            display = _truncate_to_width(display, col1_width - 2)

            # Total stats
            tw = data.get("total_wins", 0)
            tl = data.get("total_losses", 0)
            td = data.get("total_draws", 0)
            total = tw + tl + td
            t_wr = (tl / total * 100) if total > 0 else 50.0

            # Rollout stats
            r = rollout_results.get(name, {"wins": 0, "losses": 0, "draws": 0})
            rw, rl, rd = r["wins"], r["losses"], r["draws"]
            r_total = rw + rl + rd
            r_wr = (rl / r_total * 100) if r_total > 0 else 50.0

            # Format: WR% W-L-D
            r_str = (
                f"{r_wr:5.1f}% {rl:2}-{rw:2}-{rd:2}" if r_total > 0 else "     —      "
            )
            t_str = f"{t_wr:5.1f}% {tl:3}-{tw:3}-{td:3}"

            print(
                f"│{_pad_center(display, col1_width)}│{_pad_center(r_str, col2_width)}│{_pad_center(t_str, col3_width)}│"
            )

        print(f"└{'─'*col1_width}┴{'─'*col2_width}┴{'─'*col3_width}┘")
        print()
