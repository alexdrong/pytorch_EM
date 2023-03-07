import pytest

from pytorch_test.pytorch_test import read_TE_list, read_pkl_list, main, EMRunner, TorchCSRStrategy, ScipyCSRStrategy


def test_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["pytorch_test.py", "-h"])
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "usage" in captured.out


def test_read_TE_list():
    read_TE_list("tests/test_data/TE_list.txt")


def test_read_pkl_list():
    read_pkl_list("tests/test_data/G_of_R_list.txt")


def test_l1em_torch_csr():
    emrunner = EMRunner(TorchCSRStrategy(device_name='cpu', G_of_R_list_file="tests/test_data/G_of_R_list.txt"),
                        TE_list="tests/test_data/TE_list.txt", stop_thresh=1e-5, max_nEMsteps=250)
    X, times = emrunner.run_em()


def test_l1em_scipy_csr():
    emrunner = EMRunner(ScipyCSRStrategy(G_of_R_list_file="tests/test_data/G_of_R_list.txt"),
                        TE_list="tests/test_data/TE_list.txt", stop_thresh=1e-5, max_nEMsteps=250)
    X, times = emrunner.run_em()
