import json
import os
import sys
from pathlib import Path

import pytest

from graph_attention_networks.cli import main as cli_main

CI = os.getenv("CI") == "true"
requires_data = pytest.mark.skipif(CI, reason="Skip dataset download/train in CI")


@requires_data
def test_train_one_epoch_creates_artifacts(monkeypatch, tmp_path, capsys):
    # Work in an isolated temp dir
    monkeypatch.chdir(tmp_path)

    # Minimal config in temp dir
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        "seed: 0\n"
        "epochs: 1\n"
        "hidden: 8\n"
        "heads: 8\n"
        "dropout: 0.6\n"
        "lr: 0.005\n"
        "weight_decay: 0.0005\n"
    )

    # Simulate CLI call: train
    sys.argv = [
        "python3",
        "-m",
        "graph_attention_networks.cli",
        "train",
        "--config",
        "configs/default.yaml",
    ]
    cli_main()

    # Artifacts should exist
    assert Path("results/metrics.json").exists()
    assert Path("results/metrics_epoch.csv").exists()
    assert Path("models/gat_cora.pt").exists()

    # metrics.json should be valid JSON
    data = json.loads(Path("results/metrics.json").read_text())
    assert "epoch" in data and "val" in data and "test" in data


def test_eval_raises_if_no_checkpoint(monkeypatch, tmp_path):
    # No training here; just confirm evaluate() errors cleanly without a ckpt
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    from graph_attention_networks.eval import evaluate

    with pytest.raises(FileNotFoundError):
        evaluate(model_path="models/gat_cora.pt")
