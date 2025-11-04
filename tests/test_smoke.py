def test_cli_runs(capsys):
    from graph_attention_networks.cli import main

    main()
    out = capsys.readouterr().out
    assert "placeholder" in out.lower()


def test_pkg_imports():
    import importlib

    pkg = importlib.import_module("graph_attention_networks")
    assert pkg is not None
