import math

import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_prefers_policy_that_favors_chosen() -> None:
    loss_good = dpo_loss(np.array([-0.2]), np.array([-2.0]), np.array([-0.5]), np.array([-0.5]), beta=0.1)
    loss_bad = dpo_loss(np.array([-2.0]), np.array([-0.2]), np.array([-0.5]), np.array([-0.5]), beta=0.1)
    assert loss_good < loss_bad

def test_dpo_loss_matches_closed_form() -> None:
    result = dpo_loss(np.array([-0.5]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1)
    logits = 0.1 * ((-0.5 - -1.5) - (-0.6 - -1.0))
    expected = -math.log(1.0 / (1.0 + math.exp(-logits)))
    assert result == pytest.approx(expected, rel=1e-6)

def test_dpo_loss_numerically_stable_for_extreme_logprobs() -> None:
    result = dpo_loss(np.array([-1e-8]), np.array([-50.0]), np.array([-25.0]), np.array([-25.0]), beta=1.0)
    assert math.isfinite(result)

def test_dpo_loss_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        dpo_loss(np.array([-0.5, -0.1]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1)

def test_orpo_loss_prefers_higher_chosen_odds() -> None:
    loss_good = orpo_loss(np.array([1.0]), np.array([-0.2]), np.array([-2.0]), lambda_orpo=0.1)
    loss_bad = orpo_loss(np.array([1.0]), np.array([-2.0]), np.array([-0.2]), lambda_orpo=0.1)
    assert loss_good < loss_bad

def test_orpo_loss_includes_sft_term() -> None:
    low_nll = orpo_loss(np.array([0.1]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    high_nll = orpo_loss(np.array([5.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    assert low_nll < high_nll

def test_orpo_loss_numerically_stable_near_certainty() -> None:
    result = orpo_loss(np.array([1.0]), np.array([-1e-9]), np.array([-1e-9]), lambda_orpo=0.1)
    assert math.isfinite(result)
