from pathlib import Path

import pytest

from preference_lab.trainers import PreferenceTrainer, TrainingConfig


def test_mock_trainer_writes_metrics(tmp_path: Path) -> None:
    config = TrainingConfig(method="mock", beta=0.1)
    trainer = PreferenceTrainer(config)
    metrics = trainer.train(data_path="data/sample_preferences.jsonl", output_dir=tmp_path, seed=42)
    assert "train_loss" in metrics
    assert "val_pairwise_accuracy" in metrics
    assert (tmp_path / "train_metrics.json").exists()

def test_unknown_method_raises() -> None:
    trainer = PreferenceTrainer(TrainingConfig(method="bogus"))
    with pytest.raises(ValueError, match="Unknown training method"):
        trainer.train(data_path="data/sample_preferences.jsonl", output_dir="outputs")

def test_real_method_without_torch_reports_clear_error(tmp_path: Path) -> None:
    trainer = PreferenceTrainer(TrainingConfig(method="dpo"))
    with pytest.raises((ImportError, NotImplementedError)):
        trainer.train(data_path="data/sample_preferences.jsonl", output_dir=tmp_path)
