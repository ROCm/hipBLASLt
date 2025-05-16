from ..core import Population, SearchSpace, Selection, Survival
from ..utils import Logger
from ..config import DEFAULTS
from typing import Callable

import numpy as np
import random
import logging
import time


class PBIL:
    name = "pbil"

    def __init__(self,
                 space: SearchSpace,
                 evaluate: Callable,
                 selection: Selection,
                 survival: Survival = Survival.get("fitness"),
                 lr: float = 0.02,
                 mut_prob: float = 0.02,
                 mut_shift: float = 0.02,
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

        if not isinstance(selection, Selection):
            raise ValueError("mating must be of type Mating.")
        self.selection = selection

        if not isinstance(survival, Survival):
            raise ValueError("survival must be of type Survival.")
        self.survival = survival

        self.probs = {k: np.ones((len(v),)) / len(v) for k, v in self.space.space.items()}
        self.lr = lr
        self.mut_prob = mut_prob
        self.mut_shift = mut_shift

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

        if any(sz > pop_size for sz in self.space.sizes.values()):
            self.logger.warning(f"Some variables have a larger search space than pop_size. "
                                f"Consider increasing pop_size to better cover the search space.")

        self.stats = {}

        random.seed(seed)
        np.random.seed(seed)
        self.logger.info(f"Setup completed.")
        self.logger.log_lines(self.__repr__(), logging.INFO)

    def update_probs(self, pop: Population):
        rel_probs = {k: np.zeros(len(p), ) for k, p in self.probs.items()}
        for ind in pop:
            for k, p in rel_probs.items():
                p[self.space.space[k].index(ind[k])] += 1 / pop.size

        for k in self.probs:
            self.probs[k] = (1 - self.lr) * self.probs[k] + self.lr * rel_probs[k]

            if random.random() < self.mut_prob:
                p = self.probs[k]
                idx = np.random.randint(0, high=len(p))
                p_new = p[idx] * (1 - self.mut_shift) + np.random.randint(0, high=2) * self.mut_shift
                p = (1 - p_new) * (p / (1 - p[idx]))
                p[idx] = p_new
                self.probs[k] = p

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.stats:
                self.stats[k].append(v)
            else:
                self.stats[k] = [v]
        if self.period and 'f_max' in self.stats and len(self.stats["f_max"]) > self.period:
            w = slice(-self.period - 1, -1)
            ma = np.mean(self.stats["f_max"][w])
            if (ma + self.tol) >= self.stats["f_max"][-1]:
                raise StopIteration(f"f_max did not increase for the last {self.period} generations.")

    def optimize(self):
        self.logger.info(f"Starting optimization...")
        start = time.time()
        self.stats = {}
        best = None
        n_evals = 0
        old_pop, pop = Population(), self.space.sample(self.pop_size, p=self.probs)
        try:
            for gen in range(1, self.n_gen + 1):
                scores = self.evaluate(self.space.transform(pop))
                n_evals += (scores >= 0).sum()

                pop.F = scores
                if best is None or scores.max() > best:
                    best = pop[pop.argmax()].copy()
                    self.logger.debug(f"New best ==> {self.space.transform(best)}")

                f_avg, f_max = scores[scores > 0].mean(), best.F  # Here we ignore non-valid solutions (F==0)
                self.logger.print_stats(n_gen=gen, n_evals=n_evals, f_avg=f_avg, f_max=f_max)
                self.update(f_avg=f_avg, f_max=f_max)

                old_pop = self.survival(old_pop, pop, self.pop_size)
                selected = self.selection(old_pop)
                self.update_probs(selected)
                pop = self.space.sample(self.pop_size, p=self.probs)
        except StopIteration as e:
            self.logger.info(f"Stop criterion reached: {e}")

        X = self.space.transform(best)
        self.logger.info(f"Finished optimization in {(time.time() - start):.3f}s")
        self.logger.info(f"X: {X}")
        self.logger.info(f"F: {best.F:.4f}")
        return X, best.F

    def __repr__(self):
        return f"PBIL(pop_size={self.pop_size}, n_gen={self.n_gen})\n{self.selection}\n{self.space}"  # TODO
