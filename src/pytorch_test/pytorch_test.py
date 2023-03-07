import argparse
import logging
import sys
import pickle
from scipy import sparse
import numpy
import datetime
import torch
from abc import ABC, abstractmethod
from typing import Tuple, List
from numpy.typing import NDArray


def read_pkl_list(G_of_R_list_file: str = "G_of_R_list.txt"):
    G_of_R_list: list[sparse.csr_matrix] = []
    with open(G_of_R_list_file) as pkl_list:
        pkl_file_list = [pkl.strip() for pkl in pkl_list]
        for pkl_file in pkl_file_list:
            G_of_R_list.append(pickle.load(open(pkl_file, 'rb'), encoding="latin1"))
    return sparse.vstack(G_of_R_list)


def read_TE_list(TE_list: str = "TE_list.txt"):
    TE_names = list()
    for name in open(TE_list):
        TE_names.append(name.strip().split('\t')[0])
    return TE_names


def scipy_to_torch_sparse(scipy_sparse):
    coo = scipy_sparse.tocoo()
    indices = torch.from_numpy(numpy.vstack((coo.row, coo.col)).astype(numpy.int64))
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, requires_grad=False, dtype=torch.float32)


class EMStrategy(ABC):
    """
    The EMStrategy interface declares operations common to all supported versions
    of some EM algorithm.

    The RUNNER uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def do_algorithm(self, max_nEMsteps: int, stop_thresh: float):
        ...

    @abstractmethod
    def write_X(self, X: NDArray[numpy.float32], out_file: str = "X.pkl"):
        ...


class TorchCSRStrategy(EMStrategy):
    """
    Concrete EM Strategy implementing the algorithm using torch.sparse_csr while following the base EMStrategy
    interface. The interface makes them interchangeable in the Context.
    """
    def __init__(self, device_name: str = 'cpu', G_of_R_list_file: str = "G_of_R_list.txt") -> None:
        self.device = torch.device(device_name)
        logging.info("reading pkl files")
        self.G_of_R = read_pkl_list(G_of_R_list_file)
        logging.info("converting to torch sparse")
        self.G_of_R = scipy_to_torch_sparse(self.G_of_R).coalesce().to_sparse_csr().to(self.device, dtype=torch.float32)
        self.X = (torch.ones(self.G_of_R.shape[1], dtype=torch.float32, requires_grad=False)/self.G_of_R.shape[1]).to(self.device)

    def do_algorithm(self, max_nEMsteps: int, stop_thresh: float) -> Tuple[NDArray[numpy.float32], List[float]]:
        step_times: list[float] = []
        for step in range(max_nEMsteps):
            starttime = datetime.datetime.now()
            L_of_R = self.G_of_R.matmul(self.X)
            L_of_R_inv = torch.pow(L_of_R, -1)
            exp_counts = L_of_R_inv.matmul(self.G_of_R).multiply(self.X)
            X_new = exp_counts/torch.sum(exp_counts)
            loglik = torch.sum(torch.log(L_of_R))
            print(step, torch.max(torch.abs(X_new-self.X)), loglik, datetime.datetime.now()-starttime)
            if torch.allclose(X_new, self.X, atol=stop_thresh):
                break
            del self.X
            self.X = X_new
            step_times.append((datetime.datetime.now()-starttime).total_seconds())
        return (self.X.cpu().numpy(), step_times)

    def write_X(self, X: NDArray[numpy.float32], out_file: str = "X.pkl"):
        pickle.dump(X, open(out_file, 'wb'))


class ScipyCSRStrategy(EMStrategy):
    """
    Concrete EM Strategy implementing the algorithm using scipy.sparse_csr while following the base EMStrategy
    interface. The interface makes them interchangeable in the Context.
    """
    def __init__(self, device_name: str = 'cpu', G_of_R_list_file: str = "G_of_R_list.txt") -> None:
        self.device = torch.device(device_name)
        logging.info("reading pkl files")
        self.G_of_R = read_pkl_list(G_of_R_list_file)
        self.X = sparse.csr_matrix(numpy.ones((1, self.G_of_R.shape[1]), dtype=numpy.float64)/self.G_of_R.shape[1])

    def calculate_expcounts(self, G_of_R, X):
        L_of_R_mat = X.multiply(G_of_R)
        L_of_R = numpy.array(L_of_R_mat.sum(1))
        L_of_R_mat = L_of_R_mat[L_of_R[:, 0] >= 10**-200, :]
        L_of_R = L_of_R[L_of_R >= 10**-200]
        L_of_R_inv = sparse.csr_matrix(1.0/L_of_R).transpose()
        exp_counts = L_of_R_mat.multiply(L_of_R_inv).sum(0)
        loglik = numpy.sum(numpy.log(L_of_R))
        if numpy.isfinite(loglik):
            return exp_counts, loglik
        else:
            return numpy.zeros(G_of_R.shape[1]), 0.0

    def do_algorithm(self, max_nEMsteps: int, stop_thresh: float) -> Tuple[NDArray[numpy.float32], List[float]]:
        step_times: list[float] = []
        for step in range(max_nEMsteps):
            starttime = datetime.datetime.now()
            exp_counts, loglik = self.calculate_expcounts(self.G_of_R, self.X)
            last_X = self.X.copy()
            self.X = sparse.csr_matrix(exp_counts/numpy.sum(exp_counts))
            print(step, numpy.max(numpy.abs(self.X.toarray()-last_X.toarray())), loglik, datetime.datetime.now()-starttime)
            if numpy.max(numpy.abs(self.X.toarray()-last_X.toarray())) < stop_thresh:
                break
            step_times.append((datetime.datetime.now()-starttime).total_seconds())
        return (self.X.todense(), step_times)

    def write_X(self, X: NDArray[numpy.float32], out_file: str = "X.pkl"):
        pickle.dump(X, open(out_file, 'wb'))


class EMRunner():
    """
    Interface for running EM
    """

    def __init__(self, strategy: EMStrategy, TE_list: str = "TE_list.txt", stop_thresh: float = 1e-6, max_nEMsteps: int = 10000) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """
        self.stop_thresh = stop_thresh
        self.max_nEMsteps = max_nEMsteps
        self._strategy = strategy
        self.TE_names = read_TE_list(TE_list)

    @property
    def strategy(self) -> EMStrategy:
        """
        The Context maintains a reference to one of the EMStrategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the EMStrategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: EMStrategy) -> None:
        """
        Usually, the Context allows replacing a EMStrategy object at runtime.
        """

        self._strategy = strategy

    def run_em(self) -> Tuple[NDArray[numpy.float32], List[float]]:
        """
        The Context delegates some work to the EMStrategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        logging.info("starting EM")
        X, times = self._strategy.do_algorithm(self.max_nEMsteps, self.stop_thresh)
        logging.info("finished EM")
        return X, times


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=str, help='device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--X_dense', type=bool, help='X_dense', default=True)
    parser.add_argument('--TE_list', type=str, help='TE_list', default='tests/test_data/TE_list.txt')
    parser.add_argument('--G_of_R_list_file', type=str, help='G_of_R_list_file', default='tests/test_data/G_of_R_list.txt')
    parser.add_argument('--loglevel', type=str, help='log level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--X_out', '-o', type=str, help='X_out', default='tests/test_data/X.pkl')
    parser.add_argument('--stop_thresh', type=float, help='stop_thresh', default=1e-6)
    parser.add_argument('--max_nEMsteps', type=int, help='max_nEMsteps', default=10000)
    parser.add_argument('--strategy', type=str, help='strategy', default='TorchCSRStrategy', choices=['TorchCSRStrategy', 'ScipyCSRStrategy'])
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=args.loglevel, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if args.strategy == 'TorchCSRStrategy':
        emrunner = EMRunner(TorchCSRStrategy(device_name=args.device, G_of_R_list_file=args.G_of_R_list_file),
                            TE_list=args.TE_list, stop_thresh=args.stop_thresh, max_nEMsteps=args.max_nEMsteps)
    if args.strategy == 'ScipyCSRStrategy':
        emrunner = EMRunner(ScipyCSRStrategy(G_of_R_list_file=args.G_of_R_list_file),
                            TE_list=args.TE_list, stop_thresh=args.stop_thresh, max_nEMsteps=args.max_nEMsteps)
    X, times = emrunner.run_em()
    emrunner.strategy.write_X(X, out_file=args.X_out)


if __name__ == "__main__":
    main()
