#!/usr/bin/env python3
"""
Scientific Evaluation Script for DeckGym with 2 modes.
Extensive Chaos: Performs a full matrix cross-evaluation (396x396) to eliminate deck bias.
Generalization: Tests the model's ability to generalize to unseen decks.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deckgym.deck_loader import MetaDeckLoader, DeckInfo

console = Console()


class ScientificEvaluator:
    def __init__(self, meta_dir: str, model_code: str = "o1"):
        self.meta_dir = meta_dir
        self.model_code = model_code
        self.num_games = 1
        self.loader = MetaDeckLoader(meta_dir)
        self.stats = {}

    def analyze_data(self):
        """Minimal data analysis for Chaos mode."""
        self.stats = {
            "total_decks": len(self.loader.decks),
            "archetypes_count": len(self.loader.archetypes),
        }

        console.print(
            Panel(
                f"Total Decks in Meta-DB: [bold cyan]{self.stats['total_decks']}[/bold cyan]\n"
                f"Archetypes Identified: [bold magenta]{self.stats['archetypes_count']}[/bold magenta]\n",
                title="Dataset Summary",
                border_style="blue",
            )
        )

    def run_simulation(
        self,
        player_folder: str,
        opponent_folder: str,
        players: str,
        num_total_games: int,
        progress=None,
        task_id=None,
    ) -> List[Dict[str, Any]]:
        """Run the optimized matrix simulation and stream results/progress."""
        binary = "./target/release/deckgym"
        if not os.path.exists(binary):
            cmd_prefix = ["cargo", "run", "--release", "--features", "onnx", "--"]
        else:
            # Check if binary is executable
            if not os.access(binary, os.X_OK):
                os.chmod(binary, 0o755)
            cmd_prefix = [binary]

        cmd = cmd_prefix + [
            "simulate",
            player_folder,
            opponent_folder,
            "--num",
            str(num_total_games),
            "--players",
            players,
            "--parallel",
            "-vv",
        ]

        # Stream stderr in real-time
        process = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, bufsize=1
        )

        matchups = []
        # Progress bar parsing (e.g., [00:00:05] [#####] 2500/5000)
        # Result parsing (e.g., RESULT: ...)

        if process.stderr:
            for line in process.stderr:
                line = line.strip()
                if not line:
                    continue

                # 1. Parse progress for the UI
                if "/" in line and "[" in line and "]" in line:
                    try:
                        progress_part = line.split("]")[-1].strip().split()[0]
                        current, total = map(int, progress_part.split("/"))
                        if progress and task_id:
                            progress.update(task_id, completed=current)
                    except:
                        pass

                # 2. Parse Result lines
                if "RESULT:" in line:
                    try:
                        parts = line.split("|")
                        matchup_part = parts[0].replace("RESULT:", "").strip()
                        da_db = matchup_part.split(" VS ")
                        da_path = da_db[0].strip()
                        db_path = da_db[1].strip()

                        stats_part = parts[1].strip()
                        stats_parts = stats_part.split()
                        wins = int(stats_parts[1])
                        losses = int(stats_parts[3])
                        ties = int(stats_parts[5])

                        matchups.append(
                            {
                                "p1_deck": da_path,
                                "p2_deck": db_path,
                                "p1_wins": wins,
                                "p2_wins": losses,
                                "ties": ties,
                                "total": wins + losses + ties,
                            }
                        )
                    except Exception:
                        pass

                # 3. Echo ONNX/INFO/Error logs
                if (
                    "[ONNX]" in line
                    or "[INFO]" in line
                    or "[PROGRESS]" in line
                    or "Error" in line
                    or "panic" in line
                ):
                    console.print(f"[dim blue]{line}[/dim blue]")

        process.wait()

        if process.returncode != 0:
            console.print(
                f"[bold red]Simulation failed with exit code {process.returncode}[/bold red]"
            )
            return []

        return matchups

    def run_generalization_protocol(self):
        """Execute the Generalization Protocol."""
        self.analyze_data()

        # 0. Setup Seed
        seed = 42
        random.seed(seed)
        np.random.seed(seed)

        # Force 1 game per matchup for speed and fairness in large pools
        self.num_games = 1

        # 1. Sample 10k decks for reference opponent
        console.print(
            "[bold yellow]Sampling 10,000 decks for generalization opponent pool...[/bold yellow]"
        )
        gen_opponent_pool = self.loader.sample_n_deck_info(10000, mode="uniform")

        # 2. Sample 2 control decks
        console.print("[bold yellow]Sampling 2 control decks...[/bold yellow]")
        control_decks = self.loader.sample_n_deck_info(2, mode="uniform")

        # 3. Load 2 never-used decks
        console.print("[bold yellow]Loading never-used decks...[/bold yellow]")
        neverused_dir = Path("example_decks/neverused_decks")
        neverused_files = sorted(list(neverused_dir.glob("*.txt")))
        if len(neverused_files) < 2:
            console.print(
                "[red]Error: Need at least 2 decks in example_decks/neverused_decks[/red]"
            )
            return

        gen_decks = []
        for fpath in neverused_files[:2]:
            with open(fpath, "r") as f:
                content = f.read()
                gen_decks.append(DeckInfo(archetype=fpath.stem, deck_string=content))

        # 4. Trials
        # Control Trial: players (o1, e2) using 2 control decks VS opponent (e2) using 10k meta decks
        # Note o1 is used as an example, other ONNX models can be used instead.
        console.print("\n[bold cyan]--- PHASE 1: Control Trial ---[/bold cyan]")
        console.print(
            f"Goal: Measure performance of {self.model_code} and e2 on known meta decks vs 10k meta pool."
        )

        # Save original model_code
        original_model = self.model_code

        # ONNX Control
        self.model_code = original_model
        results_control_onnx = self.run_paradigm(
            f"Control {original_model}", control_decks, gen_opponent_pool, silent=True
        )
        wr_control_onnx = self._get_wr(results_control_onnx)

        # e2 Control
        self.model_code = "e2"
        results_control_e2 = self.run_paradigm(
            "Control e2", control_decks, gen_opponent_pool, silent=True
        )
        wr_control_e2 = self._get_wr(results_control_e2)

        # Gen Trial: players (o1, e2) using 2 unseen decks VS opponent (e2) using 10k meta decks
        console.print("\n[bold cyan]--- PHASE 2: Generalization Trial ---[/bold cyan]")
        console.print(
            f"Goal: Measure performance of {original_model} and e2 on unseen decks vs 10k meta pool."
        )

        # ONNX Gen
        self.model_code = original_model
        results_gen_onnx = self.run_paradigm(
            f"Generalization {original_model}",
            gen_decks,
            gen_opponent_pool,
            silent=True,
        )
        wr_gen_onnx = self._get_wr(results_gen_onnx)

        # e2 Gen
        self.model_code = "e2"
        results_gen_e2 = self.run_paradigm(
            "Generalization e2", gen_decks, gen_opponent_pool, silent=True
        )
        wr_gen_e2 = self._get_wr(results_gen_e2)

        # Restore original model_code
        self.model_code = original_model

        # 5. Metrics
        drop_onnx = wr_control_onnx - wr_gen_onnx
        drop_e2 = wr_control_e2 - wr_gen_e2
        gen_score = drop_e2 - drop_onnx

        # 6. Report
        self._report_generalization(
            wr_control_onnx,
            wr_gen_onnx,
            drop_onnx,
            wr_control_e2,
            wr_gen_e2,
            drop_e2,
            gen_score,
        )

        # Save results
        all_results = {
            "Control o1": results_control_onnx,
            "Control e2": results_control_e2,
            "Generalization o1": results_gen_onnx,
            "Generalization e2": results_gen_e2,
        }

        # Additional data for save_results to include in markdown
        self.gen_protocol_data = {
            "wr_control_onnx": wr_control_onnx,
            "wr_gen_onnx": wr_gen_onnx,
            "drop_onnx": drop_onnx,
            "wr_control_e2": wr_control_e2,
            "wr_gen_e2": wr_gen_e2,
            "drop_e2": drop_e2,
            "gen_score": gen_score,
        }

        self.save_results(all_results)

    def _get_wr(self, results: List[Dict]) -> float:
        if not results:
            return 0.0
        total_wins = sum(r.get("wins", 0) for r in results)
        total_games = sum(r.get("total", 0) for r in results)
        return total_wins / total_games if total_games > 0 else 0.0

    def _report_generalization(
        self, wr_c_onnx, wr_g_onnx, drop_onnx, wr_c_e2, wr_g_e2, drop_e2, score
    ):
        table = Table(
            title="[bold yellow]Generalization Protocol Results[/bold yellow]",
            show_lines=True,
        )
        table.add_column("Metric", style="cyan")
        table.add_column(f"{self.model_code}", justify="right", style="green")
        table.add_column("e2 (Reference)", justify="right", style="magenta")

        table.add_row("Winrate Control", f"{wr_c_onnx*100:.2f}%", f"{wr_c_e2*100:.2f}%")
        table.add_row("Winrate Gen", f"{wr_g_onnx*100:.2f}%", f"{wr_g_e2*100:.2f}%")
        table.add_row(
            "Performance Drop",
            f"[bold red]{drop_onnx*100:.2f}%[/bold red]",
            f"[bold red]{drop_e2*100:.2f}%[/bold red]",
        )

        console.print("\n")
        console.print(table)

        interpretation = ""
        if score > 0.02:
            interpretation = (
                f"[bold green]{self.model_code} generalizes BETTER than e2[/bold green]"
            )
        elif score < -0.02:
            interpretation = (
                f"[bold red]{self.model_code} generalizes WORSE than e2[/bold red]"
            )
        else:
            interpretation = "[bold yellow]Same generalization capability[/bold yellow]"

        console.print(
            Panel(
                f"Generalization Score: [bold white]{score*100:.2f}%[/bold white]\n\n"
                f"Interpretation: {interpretation}",
                title="Final Verification",
                border_style="blue",
            )
        )

    def run_paradigm(
        self,
        name: str,
        player_pool: List[DeckInfo],
        opponent_pool: List[DeckInfo],
        silent: bool = False,
    ):
        """Execute a full paradigm evaluation using optimized matrix batching."""
        if not silent:
            console.print(
                f"\n[bold green]>>> Starting Paradigm: {name}[/bold green] ({len(player_pool)}x{len(opponent_pool)} matchups)"
            )

        p1_results = []

        with tempfile.TemporaryDirectory(
            prefix="p1_"
        ) as tmp_p1, tempfile.TemporaryDirectory(prefix="p2_") as tmp_p2:
            # Write player decks to p1 folder
            for i, d in enumerate(player_pool):
                fname = f"p1_{i:03d}.json"
                with open(os.path.join(tmp_p1, fname), "w") as f:
                    f.write(d.deck_string)

            # Write opponent decks to p2 folder
            for i, d in enumerate(opponent_pool):
                fname = f"p2_{i:03d}.json"
                with open(os.path.join(tmp_p2, fname), "w") as f:
                    f.write(d.deck_string)

            # Configure players
            opponent_code = "e2" if self.model_code != "e2" else "r"
            players = f"{self.model_code},{opponent_code}"
            total_games = len(player_pool) * len(opponent_pool) * self.num_games

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                disable=silent,
            ) as progress:
                task = progress.add_task(f"Simulating {name}...", total=total_games)

                # Execute single matrix run
                matchups = self.run_simulation(
                    tmp_p1,
                    tmp_p2,
                    players,
                    total_games,
                    progress=progress,
                    task_id=task,
                )

                if not matchups and not silent:
                    console.print("[red]No matchups were processed successfully.[/red]")

                if not silent:
                    progress.update(task, completed=total_games)

            # Re-map results back to p1_results using index-based filenames
            # Regex to extract the index from filenames like 'p1_000.json'
            import re

            for m in matchups:
                p1_fname = os.path.basename(m["p1_deck"])
                p2_fname = os.path.basename(m["p2_deck"])

                p1_match = re.search(r"p1_(\d+)", p1_fname)
                p2_match = re.search(r"p2_(\d+)", p2_fname)

                if p1_match and p2_match:
                    p1_idx = int(p1_match.group(1))
                    p2_idx = int(p2_match.group(1))

                    if p1_idx < len(player_pool) and p2_idx < len(opponent_pool):
                        p1_deck = player_pool[p1_idx]
                        p2_deck = opponent_pool[p2_idx]

                        if m["total"] == 0:
                            continue

                        wr = m["p1_wins"] / m["total"]
                        p1_results.append(
                            {
                                "wr": wr,
                                "p1_arch": p1_deck.archetype,
                                "p2_arch": p2_deck.archetype,
                                "wins": m["p1_wins"],
                                "losses": m["p2_wins"],
                                "ties": m["ties"],
                                "total": m["total"],
                            }
                        )

        if not silent:
            self._report_paradigm(name, p1_results)
        return p1_results

    def _report_paradigm(self, name: str, results: List[Dict]):
        if not results:
            return

        wrs = np.array([r["wr"] for r in results])
        total_wins = sum(r.get("wins", 0) for r in results)
        total_games = sum(r.get("total", 0) for r in results)
        global_wr = total_wins / total_games if total_games > 0 else np.mean(wrs)

        ci = 1.96 * (np.std(wrs) / math.sqrt(len(wrs)))

        panel_content = (
            f"Global Winrate: [bold green]{global_wr*100:.2f}%[/bold green]\n"
        )
        panel_content += f"Confidence Interval (95% CI): ±{ci*100:.3f}%\n"
        panel_content += f"Sample Size: {len(results)} matchups"

        console.print(
            Panel(panel_content, title=f"Paradigm: {name}", border_style="green")
        )

    def run_all(self):
        """Execute the Extensive Chaos Tie-Breaker (Full Matrix Aller-Retour)."""
        self.analyze_data()

        # 1. Sample 1 deck per archetype (randomly)
        chaos_decks = []
        for name, arch in self.loader.archetypes.items():
            if arch.decks:
                chaos_decks.append(random.choice(arch.decks))

        n = len(chaos_decks)
        total_matchups = n * n  # Full matrix

        console.print(
            Panel(
                f"Archetypes Sampled: [bold cyan]{n}[/bold cyan]\n"
                f"Total Matrix Matchups: [bold magenta]{total_matchups}[/bold magenta]\n"
                f"Format: [bold white]Aller-Retour (Fixed 1 game/side)[/bold white]",
                title="[red]Extensive Chaos Mode[/red]",
                border_style="red",
            )
        )

        # 2. Run the matrix
        # Force 1 game per direction for Chaos balance
        self.num_games = 1

        results = self.run_paradigm("Extensive Chaos", chaos_decks, chaos_decks)

        # 3. Report and Save
        all_results = {"Extensive Chaos": results}
        self._print_final_summary(all_results)
        self.save_results(all_results)

    def save_results(self, all_results: Dict):
        """Save results to eval_reports directory."""
        report_dir = Path("eval_reports")
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"scientific_eval_{self.model_code}_{timestamp}"

        # JSON Report
        json_path = report_dir / f"{filename_base}.json"

        # We need to make results JSON serializable (DeckInfo -> dict)
        serializable_results = {}
        for paradigm, matchups in all_results.items():
            serializable_results[paradigm] = matchups

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_code,
            "global_stats": {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in self.stats.items()
            },
            "paradigms": serializable_results,
        }

        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)

        # Markdown summary report
        md_path = report_dir / f"{filename_base}.md"
        with open(md_path, "w") as f:
            f.write(f"# Scientific Evaluation Report: {self.model_code}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Global Dataset Statistics\n")
            f.write("| Metric | Value |\n| :--- | :--- |\n")
            for k, v in self.stats.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                f.write(f"| {k} | {val} |\n")
            f.write("\n")

            f.write("## Paradigm Summary\n")
            f.write("| Paradigm | Global Winrate | 95% CI | Matchups |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")

            for name, results in all_results.items():
                if not results:
                    continue
                wrs = [r["wr"] for r in results]
                total_wins = sum(r.get("wins", 0) for r in results)
                total_games = sum(r.get("total", 0) for r in results)
                global_wr = (
                    total_wins / total_games if total_games > 0 else np.mean(wrs)
                )

                ci = 1.96 * np.std(wrs) / math.sqrt(len(wrs))
                f.write(
                    f"| {name} | {global_wr*100:.2f}% | ±{ci*100:.2f}% | {len(results)} |\n"
                )

            if hasattr(self, "gen_protocol_data"):
                d = self.gen_protocol_data
                f.write("\n## Generalization Analysis\n")
                f.write(f"| Metric | {self.model_code} | e2 (Ref) |\n")
                f.write("| :--- | :--- | :--- |\n")
                f.write(
                    f"| Winrate Control | {d['wr_control_o1']*100:.2f}% | {d['wr_control_e2']*100:.2f}% |\n"
                )
                f.write(
                    f"| Winrate Gen | {d['wr_gen_o1']*100:.2f}% | {d['wr_gen_e2']*100:.2f}% |\n"
                )
                f.write(
                    f"| Performance Drop | {d['drop_o1']*100:.2f}% | {d['drop_e2']*100:.2f}% |\n"
                )
                f.write(f"\n**Generalization Score: {d['gen_score']*100:.2f}%**\n")

                interpretation = (
                    "o1 generalizes BETTER"
                    if d["gen_score"] > 0.02
                    else (
                        "o1 generalizes WORSE"
                        if d["gen_score"] < -0.02
                        else "Same generalization"
                    )
                )
                f.write(f"Interpretation: {interpretation}\n")

        console.print(f"\n[bold green]Report saved to:[/bold green] {json_path}")
        console.print(f"[bold green]Markdown summary saved to:[/bold green] {md_path}")

    def _print_final_summary(self, all_results: Dict):
        summary_table = Table(
            title="[bold yellow]Scientific Evaluation Summary[/bold yellow]",
            show_lines=True,
        )
        summary_table.add_column("Paradigm", style="cyan")
        summary_table.add_column("Global Winrate", justify="right", style="green")
        summary_table.add_column("Precision (95% CI)", justify="right")
        summary_table.add_column("Matchups", justify="right")

        for name, results in all_results.items():
            if not results:
                continue
            wrs = [r["wr"] for r in results]
            ci = 1.96 * np.std(wrs) / math.sqrt(len(wrs))

            total_wins = sum(r.get("wins", 0) for r in results)
            total_games = sum(r.get("total", 0) for r in results)
            global_wr = total_wins / total_games if total_games > 0 else np.mean(wrs)

            summary_table.add_row(
                name, f"{global_wr*100:.2f}%", f"±{ci*100:.3f}%", str(len(results))
            )

        console.print("\n")
        console.print(summary_table)


app = typer.Typer(help="DeckGym Scientific Evaluation")


@app.command("chaos")
def chaos(
    model: str = typer.Argument(..., help="Model code (e.g. o1, o2c, etc.)"),
    meta: str = typer.Option(
        "archetypes_by_era", "--meta", help="Path to meta decks directory"
    ),
):
    """Extensive Chaos mode: full matrix cross-evaluation across all archetypes."""
    evaluator = ScientificEvaluator(meta, model)
    evaluator.run_all()


@app.command("generalization")
def generalization(
    model: str = typer.Argument(..., help="Model code (e.g. o1, o2c, etc.)"),
    meta: str = typer.Option(
        "archetypes_by_era", "--meta", help="Path to meta decks directory"
    ),
):
    """Generalization Protocol: measures how well the model generalises to unseen decks."""
    evaluator = ScientificEvaluator(meta, model)
    evaluator.run_generalization_protocol()


if __name__ == "__main__":
    app()
