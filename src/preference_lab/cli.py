from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import mock_logprob_score, pairwise_accuracy, write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")

ConfigOption = Annotated[Path, typer.Option("--config", help="Path to YAML config")]

@app.command()
def validate(data: Path) -> None:
    try:
        examples = load_jsonl(data)
    except ValueError as exc:
        print(f"[red]Validation failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    print(f"[green]Loaded {len(examples)} preference examples[/green]")

@app.command()
def evaluate(config: ConfigOption) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    chosen_scores = [mock_logprob_score(e.prompt, e.chosen) for e in examples]
    rejected_scores = [mock_logprob_score(e.prompt, e.rejected) for e in examples]
    metrics = {"pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores)}
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")

@app.command()
def train(config: ConfigOption) -> None:
    cfg = load_config(config)
    training_cfg = TrainingConfig(**cfg["training"])
    trainer = PreferenceTrainer(training_cfg)
    try:
        metrics = trainer.train(
            data_path=cfg["paths"]["train_data"],
            output_dir=cfg["paths"]["output_dir"],
            seed=cfg.get("seed", 42),
        )
    except (ImportError, NotImplementedError) as exc:
        print(f"[red]Training failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    print(f"[green]Training complete ({training_cfg.method}): {metrics}[/green]")

if __name__ == "__main__":
    app()
