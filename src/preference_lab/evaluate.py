from __future__ import annotations

import json
import math
from pathlib import Path

from .schemas import PreferenceExample


def mock_logprob_score(prompt: str, response: str) -> float:
    """Deterministic, CPU-only proxy for a model log-probability score.

    Not a real language model: combines response informativeness (length)
    and lexical overlap with the prompt into a bounded pseudo log-prob
    (<= 0), so the pipeline can be exercised end-to-end without GPU/model
    dependencies. See docs/lab_guide.md Task 3.
    """
    response_tokens = response.lower().split()
    if not response_tokens:
        return -10.0
    prompt_tokens = set(prompt.lower().split())
    overlap = len(prompt_tokens & set(response_tokens)) / max(len(prompt_tokens), 1)
    length_term = math.log1p(len(response_tokens))
    strength = 0.5 * length_term + 2.0 * overlap
    return -1.0 / (1.0 + strength)

def pairwise_accuracy(
    examples: list[PreferenceExample], chosen_scores: list[float], rejected_scores: list[float]
) -> float:
    """Return fraction where chosen score is strictly greater than rejected score.

    Ties (chosen_score == rejected_score) count as neither a win nor a loss
    for the "chosen is better" direction, so they contribute 0 to the
    numerator but still count toward the denominator.
    """
    if not (len(examples) == len(chosen_scores) == len(rejected_scores)):
        raise ValueError(
            f"examples ({len(examples)}), chosen_scores ({len(chosen_scores)}), and "
            f"rejected_scores ({len(rejected_scores)}) must have the same length"
        )
    if not examples:
        return 0.0
    wins = sum(1 for c, r in zip(chosen_scores, rejected_scores) if c > r)
    return wins / len(examples)

def write_metrics(metrics: dict[str, float], output_dir: str | Path, filename: str = "metrics.json") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / filename
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
