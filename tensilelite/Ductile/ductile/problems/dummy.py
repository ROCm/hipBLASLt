from ..core import SearchSpace, Selection, Crossover, Mutation, Mating, Survival
from ..algorithm import GeneticAlgorithm

from .. import config as config
import numpy as np
import string
from functools import partial


def fn(target, X):
    return np.array([(np.array(list(x.values())) == target).mean() for x in X])


# def fn(X, xt=np.array([4, -2, 3.5, 5, -11, -4.7]), y=44):
#     output = np.sum(np.array([list(x.values()) for x in X]) * xt, axis=1)
#     fitness = 1.0 / (np.abs(output - y) + 1e-6)
#     return fitness


class Dummy:
    def __init__(self):
        all_chars = list(string.printable)
        target = "Hola! My name is Antonio and I am a Genetic Algorithm! What about you?"
        params = {i: all_chars for i in range(len(target))}
        # params = {i: np.linspace(-2, 5, 1000) for i in range(6)}
        cfg = config.load()
        space = SearchSpace(params,
                            valid=None,
                            max_iters=cfg["max_iters"])

        selection = Selection.get(**config.populate(cfg, "selection"))
        crossover = Crossover.get(**config.populate(cfg, "crossover"))
        mutation = Mutation(space, **cfg["mutation"])
        mating = Mating(space=space,
                        selection=selection,
                        crossover=crossover,
                        mutation=mutation,
                        max_iters=cfg["max_iters"])
        survival = Survival.get(**config.populate(cfg, "survival"))
        self.algorithm = GeneticAlgorithm(space,
                                          mating,
                                          evaluate=partial(fn, np.array(list(target))),
                                          survival=survival,
                                          pop_size=cfg["pop_size"],
                                          n_gen=cfg["n_gen"],
                                          n_offspring=cfg["n_offspring"],
                                          tol=cfg["tol"],
                                          period=cfg["period"],
                                          seed=cfg["seed"],
                                          verbose=1,
                                          log_file=cfg["log_file"])

    def solve(self):
        return self.algorithm.optimize()
