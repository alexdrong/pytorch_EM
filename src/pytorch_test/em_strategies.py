
import datetime
import logging
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
import numpy
from numpy.typing import NDArray
from scipy import sparse
from multiprocessing import Pool


def read_pkl_list(G_of_R_list_file: str = "G_of_R_list.txt"):
    G_of_R_list: list[sparse.csr_matrix] = []
    with open(G_of_R_list_file) as pkl_list:
        pkl_file_list = [pkl.strip() for pkl in pkl_list]
        for pkl_file in pkl_file_list:
            G_of_R_list.append(pickle.load(open(pkl_file, 'rb'), encoding="latin1"))
    return sparse.vstack(G_of_R_list)


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


class TorchCSRMultiGPUStrategy(EMStrategy):
    """
    Concrete EM Strategy implementing the algorithm using torch.sparse_csr while following the base EMStrategy
    interface. The interface makes them interchangeable in the Context.
    """

    def __init__(self, G_of_R_list_file: str = "G_of_R_list.txt", nGPU: int = 4) -> None:
        self.nGPU = max(nGPU, torch.cuda.device_count())
        logging.info(f"running on {self.nGPU} out of {str(torch.cuda.device_count())} GPUs")
        logging.info("reading pkl files")
        self.G_of_R = read_pkl_list(G_of_R_list_file)
        logging.info("converting to torch sparse")
        self.indices = numpy.array_split(numpy.arange(self.G_of_R.shape[0]), torch.cuda.device_count())
        self.G_of_R_split = []
        for i in range(self.nGPU):
            self.G_of_R_split.append(scipy_to_torch_sparse(self.G_of_R[self.indices[i], :]).coalesce().to_sparse_csr().to(f"cuda:{i}", dtype=torch.float32))
        self.X = (torch.ones(self.G_of_R.shape[1], dtype=torch.float32, requires_grad=False)/self.G_of_R.shape[1])
        self.L_of_R_inv = torch.zeros(self.G_of_R.shape[0], dtype=torch.float32, requires_grad=False)

    def do_algorithm(self, max_nEMsteps: int, stop_thresh: float) -> Tuple[NDArray[numpy.float32], List[float]]:
        step_times: list[float] = []
        for step in range(max_nEMsteps):
            starttime = datetime.datetime.now()
            loglik = torch.zeros(1)
            exp_counts = torch.empty(0)
            for i in range(self.nGPU):
                L_of_R = self.G_of_R_split[i].matmul(self.X.to(f"cuda:{i}"))
                L_of_R_inv = torch.pow(L_of_R, -1)
                exp_counts = torch.cat(exp_counts, L_of_R_inv.matmul(self.G_of_R_split[i]).multiply(self.X).to("cpu"))
                loglik += torch.sum(torch.log(L_of_R)).to("cpu")
            X_new = exp_counts/torch.sum(exp_counts)
            print(step, torch.max(torch.abs(X_new-self.X)), loglik, datetime.datetime.now()-starttime)
            if torch.allclose(X_new, self.X, atol=stop_thresh):
                break
            del self.X
            self.X = X_new
            step_times.append((datetime.datetime.now()-starttime).total_seconds())
        return (self.X.cpu().numpy(), step_times)

    def write_X(self, X: NDArray[numpy.float32], out_file: str = "X.pkl"):
        pickle.dump(X, open(out_file, 'wb'))


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


class ScipyCSRMultiprocessingStrategy(EMStrategy):
    """
    Concrete EM Strategy implementing the algorithm using scipy.sparse_csr and multiprocessing while following the base EMStrategy
    """
    def __init__(self, G_of_R_list_file: str = "G_of_R_list.txt", TE_list: str = "TE_list.txt", nThreads: int = 2) -> None:
        self.nThreads = nThreads
    # All the transcripts names in the same order as the G_of_R matrix columns
    # Intial guess
        self.TE_names = list()
        for name in open(TE_list):
            self.TE_names.append(name.strip().split('\t')[0])
        self.X = sparse.csr_matrix(numpy.ones((1, len(self.TE_names)), dtype=numpy.float64)/len(self.TE_names))
        # Split up the pickle files into a set for each thread.
        G_of_R_pkl_fulllist = list()
        for G_of_R_pkl in open(G_of_R_list_file):
            G_of_R_pkl_fulllist.append(G_of_R_pkl.strip())
        self.G_of_R_pkl_lists = list()
        listsize = len(G_of_R_pkl_fulllist)//nThreads
        nlistsp1 = len(G_of_R_pkl_fulllist) % nThreads
        k = 0
        for i in range(nlistsp1):
            self.G_of_R_pkl_lists.append(G_of_R_pkl_fulllist[k:k+listsize+1])
            k += listsize+1
        for i in range(nlistsp1, nThreads):
            self.G_of_R_pkl_lists.append(G_of_R_pkl_fulllist[k:k+listsize])
            k += listsize
            self.masterPool = Pool(processes=nThreads)

    @staticmethod
    def calculate_expcounts(G_of_R_pkl, X):
        G_of_R_file = open(G_of_R_pkl, 'rb')
        G_of_R = pickle.load(G_of_R_file, encoding='latin1')
        G_of_R_file.close()
        if G_of_R is None:
            return 0.0, 0.0
        L_of_R_mat = G_of_R.multiply(X)
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

    @staticmethod
    def calculate_expcounts_chunk(input):
        G_of_R_pkl_list, X_len = input
        exp_counts = numpy.zeros(X_len.shape, dtype=numpy.float64)
        loglik = 0.0
        for G_of_R_pkl in G_of_R_pkl_list:
            this_exp_counts, this_loglik = ScipyCSRMultiprocessingStrategy.calculate_expcounts(G_of_R_pkl, X_len)
            exp_counts += this_exp_counts
            loglik += this_loglik
        return exp_counts, loglik

    # Run the EM steps
    def do_algorithm(self, max_nEMsteps: int, stop_thresh: float) -> Tuple[NDArray[numpy.float32], List[float]]:
        step_times: list[float] = []
        for step in range(max_nEMsteps):
            starttime = datetime.datetime.now()
            exp_counts = numpy.zeros((1, len(self.TE_names)), dtype=numpy.float64)
            loglik = 0.0

            outputs = self.masterPool.map(ScipyCSRMultiprocessingStrategy.calculate_expcounts_chunk, zip(self.G_of_R_pkl_lists, [self.X]*int(self.nThreads)))
            for output in outputs:
                this_exp_counts, this_loglik = output
                exp_counts += this_exp_counts
                loglik += this_loglik

            last_X = self.X.copy()
            self.X = sparse.csr_matrix(exp_counts/numpy.sum(exp_counts))
            print(step, numpy.max(numpy.abs(self.X.toarray()-last_X.toarray())), loglik, datetime.datetime.now()-starttime)
            if numpy.max(numpy.abs(self.X.toarray()-last_X.toarray())) < stop_thresh:
                break
            step_times.append((datetime.datetime.now()-starttime).total_seconds())
        return (self.X.todense(), step_times)

    def write_X(self, X: NDArray[numpy.float32], out_file: str = "X.pkl"):
        pickle.dump(X, open(out_file, 'wb'))
