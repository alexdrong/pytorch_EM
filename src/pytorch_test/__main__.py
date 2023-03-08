import argparse
import logging
import sys

from pytorch_test.em_runner import EMRunner
from pytorch_test.em_strategies import ScipyCSRStrategy, TorchCSRStrategy, ScipyCSRMultiprocessingStrategy, TorchCSRMultiGPUStrategy


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, help='device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--X_dense', type=bool, help='X_dense', default=True)
    parser.add_argument('--TE_list', type=str, help='TE_list', default='tests/test_data/TE_list.txt')
    parser.add_argument('--G_of_R_list_file', type=str, help='G_of_R_list_file', default='tests/test_data/G_of_R_list.txt')
    parser.add_argument('--loglevel', type=str, help='log level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--X_out', '-o', type=str, help='X_out', default='tests/test_data/X.pkl')
    parser.add_argument('--stop_thresh', type=float, help='stop_thresh', default=1e-6)
    parser.add_argument('--max_nEMsteps', type=int, help='max_nEMsteps', default=10000)
    parser.add_argument('--nThreads', type=int, help='nThreads', default=2)
    parser.add_argument('--nGPU', type=int, help='nGPU', default=2)
    parser.add_argument('--strategy', type=str, help='strategy', default='TorchCSRStrategy',
                        choices=['TorchCSRStrategy', 'TorchCSRMultiGPUStrategy', 'ScipyCSRStrategy', 'ScipyCSRMultiprocessingStrategy'])
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=args.loglevel, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if args.strategy == 'TorchCSRStrategy':
        emrunner = EMRunner(TorchCSRStrategy(device_name=args.device, G_of_R_list_file=args.G_of_R_list_file),
                            TE_list=args.TE_list, stop_thresh=args.stop_thresh, max_nEMsteps=args.max_nEMsteps)
    if args.strategy == 'TorchCSRMultiGPUStrategy':
        emrunner = EMRunner(TorchCSRMultiGPUStrategy(G_of_R_list_file=args.G_of_R_list_file, nGPU=args.nGPU),
                            TE_list=args.TE_list, stop_thresh=args.stop_thresh, max_nEMsteps=args.max_nEMsteps)
    if args.strategy == 'ScipyCSRStrategy':
        emrunner = EMRunner(ScipyCSRStrategy(G_of_R_list_file=args.G_of_R_list_file),
                            TE_list=args.TE_list, stop_thresh=args.stop_thresh, max_nEMsteps=args.max_nEMsteps)
    if args.strategy == 'ScipyCSRMultiprocessingStrategy':
        emrunner = EMRunner(ScipyCSRMultiprocessingStrategy(G_of_R_list_file=args.G_of_R_list_file, TE_list=args.TE_list, nThreads=args.nThreads),
                            TE_list=args.TE_list, stop_thresh=args.stop_thresh, max_nEMsteps=args.max_nEMsteps)
    X, times = emrunner.run_em()
    emrunner.strategy.write_X(X, out_file=args.X_out)


if __name__ == "__main__":
    main()
