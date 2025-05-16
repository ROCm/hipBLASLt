from ..core import Population, SearchSpace, Mating, Survival
from ..utils import Logger
from ..config import DEFAULTS
from typing import Callable

import numpy as np
import random
import logging
import time


# noinspection DuplicatedCode
class GeneticAlgorithm:
    name = "GA"

    def __init__(self,
                 space: SearchSpace,
                 mating: Mating,
                 evaluate: Callable,
                 survival: Survival = Survival.get("fitness"),
                 pop_size: int = 512,
                 n_gen: int = 20,
                 tol: float = 1e-4,
                 period: int = 5,
                 seed: int = None,
                 verbose: int = 1,
                 log_file: str = None):

        self.logger = Logger(self.name, log_file=log_file, verbose=verbose)

        if not isinstance(space, SearchSpace):
            raise ValueError("space must be of type SearchSpace.")
        self.space = space

        if not isinstance(mating, Mating):
            raise ValueError("mating must be of type Mating.")
        self.mating = mating

        if not isinstance(survival, Survival):
            raise ValueError("survival must be of type Survival.")
        self.survival = survival

        if not callable(evaluate):
            raise ValueError("'evaluate' must be a callable function.")
        self.evaluate = evaluate

        if not (isinstance(pop_size, int) and pop_size > 2):
            self.logger.error("pop_size must be an integer larger than 2, changing to default value.")
            pop_size = DEFAULTS["pop_size"]
        self.pop_size = pop_size

        if not (isinstance(n_gen, int) and n_gen > 0):
            self.logger.error("n_gen must be an integer larger than 0, changing to default value.")
            n_gen = DEFAULTS["n_gen"]
        self.n_gen = n_gen

        if not (isinstance(tol, float) and tol >= 0):
            self.logger.error("tol must be a float larger or equal to 0, changing to default value.")
            tol = DEFAULTS["tol"]
        self.tol = tol

        if period and not (isinstance(period, int) and period >= 0):
            self.logger.error("period must be an integer larger or equal to 0, changing to default value.")
            period = DEFAULTS["period"]
        self.period = period

        if self.space.n_perms < self.pop_size:
            raise ValueError("pop_size must be larger than the total amount of variable permutations.")

        max_sp_sz = max(sz for sz in self.space.sizes.values())
        if max_sp_sz > pop_size:
            self.logger.warning(f"Some variables have a larger search space than pop_size. "
                                f"Increasing pop_size for the first generations.")
            self.decay = lambda sz: int(pop_size + (sz - pop_size) / 2)
            self.pop_size = int(max_sp_sz * 1.15)
            
        self.stats = {}

        random.seed(seed)
        np.random.seed(seed)
        self.logger.info(f"Setup completed.")
        self.logger.log_lines(self.__repr__(), logging.INFO)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.stats:
                self.stats[k].append(v)
            else:
                self.stats[k] = [v]
        if self.period and 'f_avg' in self.stats and len(self.stats["f_avg"]) > self.period:
            w = slice(-self.period - 1, -1)
            ma = np.mean(self.stats["f_avg"][w])
            if (ma + self.tol) >= self.stats["f_avg"][-1]:
                raise StopIteration(f"f_avg did not increase for the last {self.period} generations.")
            
        self.pop_size = self.decay(self.pop_size) if hasattr(self, "decay") else self.pop_size

    def optimize(self):
        self.logger.info(f"Starting optimization...")
        start = time.time()
        self.stats = {}
        best = None
        n_evals = 0
        old_pop, pop = Population(), self.space.sample(self.pop_size)
        try:
            for gen in range(1, self.n_gen + 1):
                scores = self.evaluate(self.space.transform(pop))
                n_evals += (scores > 0).sum()
                pop.F = scores
                if best is None or scores.max() > best:
                    best = pop[pop.argmax()].copy()
                    self.logger.debug(f"New best ==> {self.space.transform(best)}")

                f_avg, f_max = scores[scores > 0].mean(), best.F  # Here we ignore non-valid solutions (F<=0)
                self.logger.print_stats(n_gen=gen, n_evals=n_evals, f_avg=f_avg, f_max=f_max)
                self.update(f_avg=f_avg, f_max=f_max)

                old_pop = self.survival(old_pop, pop, self.pop_size)
                pop = self.mating(old_pop, self.pop_size)
        except StopIteration as e:
            self.logger.info(f"Stop criterion reached: {e}")

        X = self.space.transform(best)
        self.logger.info(f"Finished optimization in {(time.time() - start):.3f}s")
        self.logger.info(f"X: {X}")
        self.logger.info(f"F: {best.F:.4f}")
        return X, best.F

    def __repr__(self):
        return f"GeneticAlgorithm(pop_size={self.pop_size}, n_gen={self.n_gen})\n{self.mating}\n{self.space}"
