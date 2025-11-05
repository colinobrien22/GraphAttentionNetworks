import json
from pathlib import Path

from graph_attention_networks.cli import main as cli_main


def test_train_then_eval_confmat(monkeypatch, tmp_path, capsys):
    # work in a temp directory
    monkeypatch.chdir(tmp_path)

    # minimal config
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

    # train one epoch (creates checkpoint + metrics)
    import sys

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
    assert Path("results/metrics.json").exists()
    _ = json.loads(Path("results/metrics.json").read_text())

    # eval (should also produce a confusion matrix plot)
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
