import domaingraph


def test_version() -> None:
    assert domaingraph.__version__ == "0.1.0"


def test_main_runs(capsys) -> None:
    domaingraph.main()
    assert "domaingraph" in capsys.readouterr().out
