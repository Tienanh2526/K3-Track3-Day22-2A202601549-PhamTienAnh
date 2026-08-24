from __future__ import annotations

import json
import random
import re
import warnings
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample

_PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "long digit sequence": re.compile(r"\d{9,}"),
}

def _warn_possible_pii(lineno: int, example: PreferenceExample) -> None:
    text = f"{example.prompt} {example.chosen} {example.rejected}"
    for label, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            warnings.warn(f"line {lineno}: possible PII detected ({label})", stacklevel=2)

def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Reports the 1-based line number for malformed JSON, schema violations, and
    exact duplicate rows (same prompt/chosen/rejected repeated, e.g. from
    re-running a data generation script in append mode). Also emits a
    non-fatal warning if a row looks like it may contain PII.
    """
    examples: list[PreferenceExample] = []
    seen_rows: set[tuple[str, str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON ({exc.msg})") from exc
            try:
                example = PreferenceExample.model_validate(data)
            except ValidationError as exc:
                raise ValueError(f"line {lineno}: schema error ({exc})") from exc
            key = (example.prompt, example.chosen, example.rejected)
            if key in seen_rows:
                raise ValueError(f"line {lineno}: duplicate preference pair (identical to an earlier row)")
            seen_rows.add(key)
            _warn_possible_pii(lineno, example)
            examples.append(example)
    return examples

def split_by_prompt(
    examples: list[PreferenceExample], validation_ratio: float = 0.2, seed: int = 42
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples into train/validation sets, grouped by prompt to avoid leakage.

    All rows sharing the same prompt land in the same split. Prompts are
    shuffled deterministically (given `seed`) before being partitioned so
    repeated calls with the same inputs produce the same split.
    """
    if not examples:
        return [], []
    unique_prompts = sorted({example.prompt for example in examples})
    shuffled_prompts = unique_prompts[:]
    random.Random(seed).shuffle(shuffled_prompts)

    val_count = round(len(shuffled_prompts) * validation_ratio)
    if len(shuffled_prompts) > 1:
        val_count = min(max(val_count, 1), len(shuffled_prompts) - 1)
    else:
        val_count = 0
    val_prompts = set(shuffled_prompts[:val_count])

    train = [example for example in examples if example.prompt not in val_prompts]
    val = [example for example in examples if example.prompt in val_prompts]
    return train, val
