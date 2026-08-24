from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import load_jsonl, split_by_prompt
from .evaluate import mock_logprob_score, pairwise_accuracy, write_metrics
from .losses import FloatArray, dpo_loss
from .schemas import PreferenceExample


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2

class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, data_path: str | Path, output_dir: str | Path, seed: int = 42) -> dict[str, float]:
        """Train the policy and write metrics/checkpoints to `output_dir`.

        method="mock" runs a deterministic, CPU-only proxy scorer end-to-end
        through the DPO loss so the full pipeline (data -> loss -> metrics)
        can be exercised without GPU/model dependencies. method="dpo"/"orpo"
        require the optional 'train' extras and a real trainer implementation.
        """
        examples = load_jsonl(data_path)
        train_examples, val_examples = split_by_prompt(examples, validation_ratio=0.2, seed=seed)

        if self.config.method == "mock":
            metrics = self._train_mock(train_examples, val_examples)
        elif self.config.method in ("dpo", "orpo"):
            metrics = self._train_real(train_examples, val_examples)
        else:
            raise ValueError(f"Unknown training method: {self.config.method!r}")

        write_metrics(metrics, output_dir, filename="train_metrics.json")
        return metrics

    def _score(self, examples: list[PreferenceExample]) -> tuple[FloatArray, FloatArray]:
        chosen = np.array([mock_logprob_score(e.prompt, e.chosen) for e in examples])
        rejected = np.array([mock_logprob_score(e.prompt, e.rejected) for e in examples])
        return chosen, rejected

    def _train_mock(
        self, train_examples: list[PreferenceExample], val_examples: list[PreferenceExample]
    ) -> dict[str, float]:
        train_chosen, train_rejected = self._score(train_examples)
        # No fine-tuned/reference model split in mock mode: use a flat reference so
        # the DPO loss reduces to how strongly the mock scorer already prefers chosen.
        ref_chosen = np.zeros_like(train_chosen)
        ref_rejected = np.zeros_like(train_rejected)
        train_loss = (
            dpo_loss(train_chosen, train_rejected, ref_chosen, ref_rejected, beta=self.config.beta)
            if len(train_examples)
            else 0.0
        )

        val_chosen, val_rejected = self._score(val_examples)
        val_accuracy = pairwise_accuracy(val_examples, val_chosen.tolist(), val_rejected.tolist())

        return {"train_loss": train_loss, "val_pairwise_accuracy": val_accuracy}

    def _train_real(
        self, train_examples: list[PreferenceExample], val_examples: list[PreferenceExample]
    ) -> dict[str, float]:
        try:
            import torch  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"Training with method={self.config.method!r} requires the optional 'train' "
                "extras (torch/transformers/trl). Install with: pip install -e '.[train]'"
            ) from exc
        raise NotImplementedError(
            f"TODO(student): wire up a TRL-backed trainer for method={self.config.method!r}"
        )
