import logging
from typing import List, Tuple

import numpy
from numpy.typing import NDArray

from pytorch_test.em_strategies import EMStrategy


def read_TE_list(TE_list: str = "TE_list.txt"):
    TE_names = list()
    for name in open(TE_list):
        TE_names.append(name.strip().split('\t')[0])
    return TE_names


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
        logging.info("finished EM, total time: " + str(sum(times)) + " seconds")
        return X, times
