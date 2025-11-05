import sys

import pytest


def test_cli_help_runs(capsys, monkeypatch):
    # Simulate: ganet train -h  (argv[0] is arbitrary program name)
    monkeypatch.setattr(sys, "argv", ["ganet", "train", "-h"])

    from graph_attention_networks.cli import main

    # argparse should exit 0 for -h
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0

    out = capsys.readouterr().out + capsys.readouterr().err
    assert "Train a GAT model on Cora" in out or "usage:" in out
