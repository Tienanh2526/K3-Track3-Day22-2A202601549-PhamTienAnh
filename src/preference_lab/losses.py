from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

def _log_sigmoid(x: FloatArray) -> FloatArray:
    """Numerically stable log(sigmoid(x)) == -softplus(-x), via logaddexp."""
    return -np.logaddexp(0.0, -x)

def _require_same_shape(**arrays: FloatArray) -> None:
    shapes = {name: arr.shape for name, arr in arrays.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"All inputs must have the same shape, got {shapes}")

def dpo_loss(
    policy_chosen_logps: FloatArray,
    policy_rejected_logps: FloatArray,
    ref_chosen_logps: FloatArray,
    ref_rejected_logps: FloatArray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    L = -E[log sigmoid(beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)))]

    Uses log-sigmoid via `np.logaddexp` for numerical stability instead of
    computing sigmoid(x) and taking its log directly (which underflows for
    very negative x).
    """
    policy_chosen_logps = np.asarray(policy_chosen_logps, dtype=np.float64)
    policy_rejected_logps = np.asarray(policy_rejected_logps, dtype=np.float64)
    ref_chosen_logps = np.asarray(ref_chosen_logps, dtype=np.float64)
    ref_rejected_logps = np.asarray(ref_rejected_logps, dtype=np.float64)
    _require_same_shape(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
    )

    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    losses = -_log_sigmoid(logits)
    return float(np.mean(losses))

def _log_odds(logps: FloatArray) -> FloatArray:
    """log(p / (1 - p)) for p = exp(logp), clamped away from 0/1 for stability."""
    probs = np.clip(np.exp(logps), 1e-7, 1 - 1e-7)
    result: FloatArray = np.log(probs) - np.log1p(-probs)
    return result

def orpo_loss(
    sft_nll: FloatArray,
    chosen_logps: FloatArray,
    rejected_logps: FloatArray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective: SFT loss + odds-ratio penalty.

    L = mean(sft_nll) + lambda_orpo * mean(-log sigmoid(log_odds(chosen) - log_odds(rejected)))
    """
    sft_nll = np.asarray(sft_nll, dtype=np.float64)
    chosen_logps = np.asarray(chosen_logps, dtype=np.float64)
    rejected_logps = np.asarray(rejected_logps, dtype=np.float64)
    _require_same_shape(sft_nll=sft_nll, chosen_logps=chosen_logps, rejected_logps=rejected_logps)

    log_odds_chosen = _log_odds(chosen_logps)
    log_odds_rejected = _log_odds(rejected_logps)
    or_losses = -_log_sigmoid(log_odds_chosen - log_odds_rejected)
    return float(np.mean(sft_nll) + lambda_orpo * np.mean(or_losses))
