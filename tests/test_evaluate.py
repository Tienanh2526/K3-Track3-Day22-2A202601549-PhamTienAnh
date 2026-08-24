import pytest

from preference_lab.evaluate import mock_logprob_score, pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0

def test_pairwise_accuracy_handles_ties_as_not_wins() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [1.0], [1.0]) == 0.0

def test_pairwise_accuracy_rejects_mismatched_lengths() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError):
        pairwise_accuracy(examples, [1.0, 2.0], [1.0])

def test_mock_logprob_score_is_deterministic_and_bounded() -> None:
    score1 = mock_logprob_score("explain X", "a detailed accurate answer about X")
    score2 = mock_logprob_score("explain X", "a detailed accurate answer about X")
    assert score1 == score2
    assert -1.0 < score1 <= 0.0
