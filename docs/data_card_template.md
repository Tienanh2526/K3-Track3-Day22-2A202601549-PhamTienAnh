# Data Card

- **Dataset name**: `sample_preferences` (Preference Alignment Lab starter set)
- **Source**: Hand-authored for this lab (`data/sample_preferences.jsonl`), covering common ML/DL concepts (self-attention, backprop, GANs, bias-variance, regularization, etc.). Optionally extensible via `scripts/generate_data.py`, which calls an external LLM (OpenAI, requires `OPENAI_API_KEY`) to synthesize additional pairs into `data/synthetic_preferences.jsonl`.
- **License/permission**: Original, lab-authored educational content; not sourced from a third-party licensed dataset. Any synthetic additions generated via `scripts/generate_data.py` inherit the terms of the API provider used to generate them and should be reviewed before reuse outside this lab.
- **Schema**: JSONL, one `PreferenceExample` per line (`src/preference_lab/schemas.py`):
  - `prompt: str` — the instruction/question (non-empty, whitespace-stripped)
  - `chosen: str` — the higher-quality response
  - `rejected: str` — a lower-quality/incorrect response; must differ from `chosen` (case/whitespace-insensitive, and rejected if near-duplicate with similarity > 0.97)
  - `metadata: dict` — free-form; every row in this set uses `{"domain": "education", "rubric": "accuracy"}`
- **Labeling rubric**: `accuracy` — chosen responses are factually correct explanations of the concept asked; rejected responses contain a plausible-sounding but factually wrong or misleading statement about the same concept (not stylistic/formatting differences).
- **Known biases**:
  - Single domain (`education`, ML/DL fundamentals) and single rubric (`accuracy`) — not representative of instruction-following, safety, or multi-turn preference data.
  - All 24 examples come from one author in one pass, so surface style (sentence structure, length) is fairly uniform; models evaluated on this set may not generalize preference-detection accuracy to more diverse phrasing.
  - Chosen responses tend to be longer/more detailed than rejected ones (this is incidental to explaining concepts correctly, not an intentional length signal) — see `docs/REPORT_TEMPLATE.md` for how this interacts with a length/overlap-sensitive scorer.
- **Safety/PII checks**: `load_jsonl` scans each row for an email-like pattern or a 9+ digit sequence and emits a `UserWarning` (non-fatal) if found; no matches were found in this dataset. No manual PII review beyond that automated check was performed, since content is synthetic/educational and contains no personal data by construction.
- **Train/validation/test split method**: `split_by_prompt` (`src/preference_lab/data.py`) — splits by the **set of unique prompts** (not rows) using a seeded deterministic shuffle (`seed=42` by default, configurable via `configs/local.yaml`), then assigns whole prompts to validation so no prompt leaks across splits. No held-out test split is defined for this lab; only train/validation.
