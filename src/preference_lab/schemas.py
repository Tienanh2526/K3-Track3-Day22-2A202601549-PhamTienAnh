from __future__ import annotations

import difflib
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_NEAR_DUPLICATE_THRESHOLD = 0.97

def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace so equivalent text compares equal."""
    return re.sub(r"\s+", " ", text.strip().lower())

class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if chosen is None:
            return rejected
        norm_chosen, norm_rejected = _normalize(chosen), _normalize(rejected)
        if norm_chosen == norm_rejected:
            raise ValueError("chosen and rejected must differ (ignoring case/whitespace)")
        similarity = difflib.SequenceMatcher(None, norm_chosen, norm_rejected).ratio()
        if similarity > _NEAR_DUPLICATE_THRESHOLD:
            raise ValueError(
                f"chosen and rejected are near-duplicates (similarity={similarity:.2f}); "
                "responses must be meaningfully different"
            )
        return rejected
