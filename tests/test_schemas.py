import pytest
from pydantic import ValidationError

from preference_lab.schemas import PreferenceExample


def test_identical_chosen_and_rejected_rejected() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(prompt="p", chosen="same text", rejected="same text")

def test_whitespace_and_case_only_difference_rejected() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(prompt="p", chosen="Same   Text", rejected="same text")

def test_near_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(
            prompt="p",
            chosen="The quick brown fox jumps over the lazy dog today",
            rejected="The quick brown fox jumps over the lazy dog today.",
        )

def test_meaningfully_different_responses_accepted() -> None:
    example = PreferenceExample(prompt="p", chosen="a detailed correct answer", rejected="a wrong answer")
    assert example.chosen != example.rejected
