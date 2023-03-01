import pytest

from pytorch_test.pytorch_test import main


def test_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["pytorch_test.py", "-h"])
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "usage" in captured.out
