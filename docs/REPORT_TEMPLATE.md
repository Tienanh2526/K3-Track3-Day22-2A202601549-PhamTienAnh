# Preference Alignment Experiment Report

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: `24` (from `data/sample_preferences.jsonl`)
- **Validation issues found**: Line 1 had malformed JSON — the prompt text contained an unescaped inner pair of double quotes (`"self-attention"`), which broke the JSON string boundary. `load_jsonl` now surfaces this class of error as `line {n}: invalid JSON (...)` instead of a bare traceback, and also detects exact duplicate rows (same prompt/chosen/rejected) with a `line {n}: duplicate preference pair` error — useful because `scripts/generate_data.py` appends to its output file, so re-running it can otherwise silently introduce duplicates.
- **Cleaning steps taken**: Escaped the inner quotes on line 1 (`\"self-attention\"`) so all 24 rows parse and pass schema validation.

### Split Strategy
- **Train/Val Ratio**: 80/20 (`validation_ratio=0.2`, the config default)
- **Leakage Prevention**: `split_by_prompt` collects the **set of unique prompts**, deterministically shuffles it with a seeded `random.Random(seed)`, and assigns whole prompts (not rows) to validation. Every row for a given prompt therefore lands in the same split, so no prompt appears on both sides. With a single unique prompt, everything goes to train (there's nothing safe to hold out). The seed is threaded through the CLI/config (`seed: 42` in `configs/local.yaml`), so splits are reproducible across runs.

## 2. Implementation: DPO (with ORPO also implemented)

### Objective Selection
- **Why this method?**: DPO was implemented as the primary objective because it's the simpler closed-form loss (no explicit reward model, just a log-sigmoid over log-ratios) and is what the mock CPU trainer exercises end-to-end. ORPO was implemented alongside it since the config already exposes `lambda_orpo` and the two objectives share the same numerical-stability concerns (log-sigmoid, log-odds).
- **Key Hyperparameters**:
    - `beta`: `0.1`
    - `lambda_orpo`: `0.1`

### Numerical Stability
- **Challenges**: Both objectives risk over/underflow — DPO's `log(sigmoid(x))` underflows to `-inf` for very negative `x` if computed naively (`log(1/(1+exp(-x)))`); ORPO's log-odds `log(p/(1-p))` blows up as `p -> 0` or `p -> 1` (`log(0)`).
- **Solutions**: `dpo_loss` uses `-np.logaddexp(0.0, -x)` for `log(sigmoid(x))` instead of computing `sigmoid` then taking `log` — stable for arbitrarily large/small logits (verified in `tests/test_losses.py::test_dpo_loss_numerically_stable_for_extreme_logprobs` with logits spanning beta=1.0, logprob differences up to 50). `orpo_loss` clips `exp(logp)` to `[1e-7, 1-1e-7]` before computing log-odds via `log(p) - log1p(-p)`, avoiding `log(0)` at the boundary (verified in `test_orpo_loss_numerically_stable_near_certainty`). Both loss functions also validate that all input arrays share the same shape before computing, to fail fast on batch-size mismatches instead of silently broadcasting.

## 3. Evaluation Results

*Scores come from `mock_logprob_score` (`src/preference_lab/evaluate.py`), a deterministic, CPU-only proxy — not a real model — that combines response length and lexical overlap with the prompt into a bounded pseudo log-probability. It exists so the full data -> score -> loss -> metrics pipeline can be exercised without GPU/model dependencies; see docs/lab_guide.md Task 3.*

**Scoring conventions**: `pairwise_accuracy` first checks that `chosen_scores`/`rejected_scores` are the same length as `examples`, raising `ValueError` on any mismatch instead of silently truncating via `zip`. A pair counts as a win only when `chosen_score > rejected_score` **strictly**; a tie (`chosen_score == rejected_score`) counts toward the denominator but not the numerator (neither a win nor a loss), so ties pull the reported accuracy down rather than being hidden. No ties occurred in this 24-pair run.

### Metrics
| Metric | Value |
|---|---|
| Pairwise Accuracy (`pref-lab evaluate`, all 24 pairs) | `83.3%` (20/24) |
| Mock DPO Train Loss (`pref-lab train`, method=mock) | `0.692` |
| Val Pairwise Accuracy (`pref-lab train`, 20% held-out prompts) | `80.0%` |

### Qualitative Review
- **Prompt**: `Explain the purpose of regularization in deep learning.`
- **Chosen Response**: `Regularization techniques like L1 and L2 regularization help prevent overfitting by adding a penalty term to the loss function, encouraging the model to learn simpler patterns.`
- **Rejected Response**: `Regularization is used to speed up the training process by reducing the number of layers in the network.`
- **Model Preference**: `Incorrect` — scorer gave chosen `-0.318` vs. rejected `-0.288`, i.e. it preferred the wrong answer.

## 4. Discussion & Failure Modes

- **What went well?**: The pipeline runs end-to-end on CPU with zero optional dependencies (`torch`/`transformers` not required) — `make test`, `pref-lab validate`, `pref-lab evaluate`, and `pref-lab train --config configs/local.yaml` (method=mock) all succeed out of the box. Both loss functions pass closed-form and directional correctness tests, and stay finite under extreme logprob inputs.
- **Observed Bias**: The mock scorer got 4/24 pairs wrong (83.3% pairwise accuracy), and in **every single miss the chosen response was the longer one** — so it is not a "prefers shorter" bias. The actual failure mode is the opposite: the scorer over-weights raw lexical overlap with the prompt. All four misses share a pattern — the *rejected* answer restates prompt keywords more densely in fewer words (e.g. "Batch normalization is used to reduce the dimensionality..." directly echoes "batch normalization" and "purpose of"), which inflates its overlap term enough to beat a longer, more substantive but less prompt-echoing chosen answer. This is a known weakness of keyword-overlap heuristics standing in for real semantic/factual judgment, and it's exactly the kind of gap a real logprob-based or learned reward model would need to close — a genuine model conditioned on the text can catch that the rejected answer is factually wrong ("used to reduce dimensionality") rather than just less prompt-similar.
- **Safety**: Not evaluated in this run — `docs/regression_prompts.md` lists 4 prompts (high-risk medical advice, strict-limit summarization, admitting uncertainty, troubleshooting with missing context) intended to be run against an actual trained policy's generations before/after training. Since no real model was fine-tuned here (only the CPU mock scorer/loss pipeline), there is no policy output to check these against yet; this is the natural next step once `pip install -e '.[train]'` and a TRL-backed trainer (see `PreferenceTrainer._train_real`, still `TODO(student)`) are wired up.
