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
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def do_algorithm(self, max_nEMsteps: int, stop_thresh: float, G_of_R, X):
        pass


class EMRunner():
    """
    Interface for running EM
    """

    def __init__(self, strategy: EMStrategy, TE_list: str = "TE_list.txt", G_of_R_list_file: str = "G_of_R_list.txt", device_name: str = 'cpu',) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy
        self.device = torch.device(device_name)
        self.TE_names = read_TE_list(TE_list)
        logging.info("reading pkl files")
        self.G_of_R = read_pkl_list(G_of_R_list_file)
        logging.info("converting to torch sparse")
        self.G_of_R = scipy_to_torch_sparse(self.G_of_R).coalesce().to_sparse_csr().to(self.device, dtype=torch.float32)
        self.X = (torch.ones(len(self.TE_names), dtype=torch.float32, requires_grad=False)/len(self.TE_names)).to(self.device)
        logging.info("starting EM")
        self.max_nEMsteps = 10000
        self.stop_thresh = 1e-6

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

        X, times = self._strategy.do_algorithm(self.max_nEMsteps, self.stop_thresh, self.G_of_R, self.X)
        return X, times


class TorchCSRStrategy(EMStrategy):
    """
    Concrete EM Strategies implement the algorithm using torch.sparse_csr while following the base EMStrategy
    interface. The interface makes them interchangeable in the Context.
    """
    def do_algorithm(self, max_nEMsteps: int, stop_thresh: float, G_of_R, X) -> Tuple[NDArray[numpy.float32], List[float]]:
        step_times: list[float] = []
        for step in range(max_nEMsteps):
            starttime = datetime.datetime.now()
            L_of_R = G_of_R.matmul(X)
            L_of_R_inv = torch.pow(L_of_R, -1)
            exp_counts = L_of_R_inv.matmul(G_of_R).multiply(X)
            X_new = exp_counts/torch.sum(exp_counts)
            loglik = torch.sum(torch.log(L_of_R))
            print(step, torch.max(torch.abs(X_new-X)), loglik, datetime.datetime.now()-starttime)
            if torch.max(torch.abs(X_new-X)) < stop_thresh:
                break
            del X
            X = X_new
            step_times.append((datetime.datetime.now()-starttime).total_seconds())
        logging.info("finished EM")
        return (X.cpu().numpy(), step_times)


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=str, help='device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--X_dense', type=bool, help='X_dense', default=True)
    parser.add_argument('--TE_list', type=str, help='TE_list', default='tests/test_data/TE_list.txt')
    parser.add_argument('--G_of_R_list_file', type=str, help='G_of_R_list_file', default='tests/test_data/G_of_R_list.txt')
    parser.add_argument('--loglevel', type=str, help='log level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--X_out', '-o', type=str, help='X_out', default='tests/test_data/X.pkl')
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=args.loglevel)
    emrunner = EMRunner(TorchCSRStrategy(), TE_list=args.TE_list, G_of_R_list_file=args.G_of_R_list_file, device_name=args.device)
    X, times = emrunner.run_em()
    with open(args.X_out, 'wb') as fh:
        pickle.dump(X, fh)


if __name__ == "__main__":
    main()
