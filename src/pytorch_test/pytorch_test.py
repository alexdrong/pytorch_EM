import argparse
import logging
import sys
import pickle
from scipy import sparse
import numpy
import datetime
import torch


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


def l1em_torch_csr(TE_list: str = "TE_list.txt", G_of_R_list_file: str = "G_of_R_list.txt", device_name: str = 'cpu', X_dense=True):
    device = torch.device(device_name)
    TE_names = read_TE_list(TE_list)
    logging.info("reading pkl files")
    G_of_R = read_pkl_list(G_of_R_list_file)
    logging.info("converting to torch sparse")
    G_of_R = scipy_to_torch_sparse(G_of_R).coalesce().to_sparse_csr().to(device, dtype=torch.float32)
    X = (torch.ones(len(TE_names), dtype=torch.float32, requires_grad=False)/len(TE_names)).to(device)
    logging.info("starting EM")
    max_nEMsteps = 10000
    stop_thresh = 1e-6
    step_times = []
    if device_name == 'cuda':
        torch.cuda.empty_cache()
    if not X_dense:
        X_diag = torch.sparse_csr_tensor(crow_indices=torch.arange(X.shape[0]+1, device=device),
                                         col_indices=torch.arange(X.shape[0], device=device),
                                         values=X, requires_grad=False)
# Run the EM steps
    for step in range(max_nEMsteps):
        starttime = datetime.datetime.now()
        L_of_R = G_of_R.matmul(X)
        L_of_R_inv = torch.pow(L_of_R, -1)
        if X_dense:
            exp_counts = L_of_R_inv.matmul(G_of_R).multiply(X)
            X_new = exp_counts/torch.sum(exp_counts)
            loglik = torch.sum(torch.log(L_of_R))
        else:
            if device_name == 'cuda':
                torch.cuda.empty_cache()
            exp_counts = L_of_R_inv.matmul(G_of_R).matmul(X_diag)
            loglik = torch.sum(torch.log(L_of_R.values()))
        print(step, torch.max(torch.abs(X_new-X)), loglik, datetime.datetime.now()-starttime)
        if torch.max(torch.abs(X_new-X)) < stop_thresh:
            break
        del X
        X = X_new
        step_times.append((datetime.datetime.now()-starttime).total_seconds())
    logging.info("finished EM")
    return X, step_times


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
    X, times = l1em_torch_csr(TE_list=args.TE_list, G_of_R_list_file=args.G_of_R_list_file, device_name=args.device, X_dense=args.X_dense)
    with open(args.X_out, 'wb') as fh:
        pickle.dump(X.cpu().numpy(), fh)


if __name__ == "__main__":
    main()
