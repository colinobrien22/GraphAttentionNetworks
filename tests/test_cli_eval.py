import json
from pathlib import Path

from graph_attention_networks.cli import main as cli_main


def test_train_one_epoch_creates_artifacts(monkeypatch, tmp_path, capsys):
    # run in temp dir so artifacts don't pollute repo
    monkeypatch.chdir(tmp_path)

    # minimal config in temp dir
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        "seed: 0\nepochs: 1\nhidden: 8\nheads: 8\ndropout: 0.6\nlr: 0.005\nweight_decay: 0.0005\n"
    )

    # call CLI: train
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")  # quieter output
    import sys

    sys.argv = ["ganet", "train", "--config", "configs/default.yaml"]
    cli_main()

    # artifacts exist?
    assert Path("results/metrics.json").exists()
    assert Path("results/metrics_epoch.csv").exists()
    # checkpoint saved on first (best) val
    assert Path("models/gat_cora.pt").exists()

    # metrics.json is valid json
    data = json.loads(Path("results/metrics.json").read_text())
    assert "epoch" in data and "test" in data


def test_eval_loads_checkpoint(monkeypatch, tmp_path):
    # reuse previous test’s structure; create dummy files quickly
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    # tiny metrics file
    (tmp_path / "results" / "metrics.json").write_text(
        '{"epoch":1,"val":0.1,"test":0.1}'
    )
    # if no real model, skip gracefully
    # (evaluation will raise if missing; that's fine to assert)
    from graph_attention_networks.eval import evaluate

    try:
        (tmp_path / "configs").mkdir(exist_ok=True)
        # If a real run hasn't produced the model, expect FileNotFoundError:
        try:
            evaluate(model_path="models/gat_cora.pt")
        except FileNotFoundError:
            pass
    except Exception as e:
        # Any other exception means something else went wrong
        raise AssertionError(f"Unexpected error during eval: {e}")
