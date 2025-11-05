import sys

import pytest


def test_cli_help_runs(capsys, monkeypatch):
    # Simulate: python3 -m graph_attention_networks.cli -h
    monkeypatch.setattr(
        sys, "argv", ["python3", "-m", "graph_attention_networks.cli", "-h"]
    )
    from graph_attention_networks.cli import main

    # argparse exits 0 on -h
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0

    out = capsys.readouterr().out + capsys.readouterr().err
    assert "Graph Attention Networks on Cora" in out
