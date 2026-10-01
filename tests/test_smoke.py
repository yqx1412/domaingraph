import domaingraph
from domaingraph.cli import main


def test_version() -> None:
    assert domaingraph.__version__ == "0.1.0"


def test_main_runs(capsys) -> None:
    assert main([]) == 0
    assert "domaingraph" in capsys.readouterr().out
