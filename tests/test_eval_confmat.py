import os
import sys
from pathlib import Path

import pytest

from graph_attention_networks.cli import main as cli_main

CI = os.getenv("CI") == "true"
requires_data = pytest.mark.skipif(CI, reason="Skip dataset download/train in CI")


@requires_data
def test_train_then_eval_confmat(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    # Minimal config
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

    # Train 1 epoch
    sys.argv = [
        "python3",
        "-m",
        "graph_attention_networks.cli",
        "train",
        "--config",
        "configs/default.yaml",
    ]
    cli_main()
    assert Path("models/gat_cora.pt").exists()

    # Eval should generate confusion_matrix.png
    sys.argv = [
        "python3",
        "-m",
        "graph_attention_networks.cli",
        "eval",
        "--model",
        "models/gat_cora.pt",
    ]
    cli_main()
    assert Path("results/confusion_matrix.png").exists()
