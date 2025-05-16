from .population import Individual, Population, IndividualSet, ExceedsCapacity
from typing import Callable, Union

import math
import numpy as np


class SearchSpace:
    def __init__(self,
                 space: dict,
                 max_iters: int = 100,
                 valid: Callable = None):
        if not (isinstance(space, dict) and len(space) > 0):
            raise ValueError("space must be a non-empty dictionary")

        # space = {k: np.unique(v, axis=0).tolist() for k, v in space.items()}

        self.sizes = {k: len(v) for k, v in space.items()}
        if any(v == 0 for v in self.sizes.values()):
            raise ValueError("some variables have a search space of length = 0")

        self.map = {k: v for k, v in space.items()}
        self.space = {k: list(range(len(v))) for k, v in space.items()}
        self.n_perms = math.prod(v for v in self.sizes.values())

        self.max_iters = max_iters
        self.valid = lambda x: valid(self.transform(x)) if valid else lambda x: True

    def __contains__(self, key: str):
        return key in self.space

    def __getitem__(self, key: str):
        return self.space[key]

    def __iter__(self):
        return dict.__iter__(self.space)

    def __len__(self):
        return len(self.space)

    def size(self, key: str = None) -> int:
        if key is None:
            return len(self)
        return self.sizes[key]

    def transform(self, X: Union[Individual, Population]):
        if isinstance(X, Individual):
            return {k: self.map[k][i] for k, i in X.items}
        return [{k: self.map[k][i] for k, i in x.items} for x in X]

    def sample(self, size: int, p: dict = None) -> Population:
        p = p if p else {}
        pop = IndividualSet(capacity=size)
        it, max_iters = 0, self.max_iters * size
        while it < max_iters:
            it += 1
            try:
                ind = Individual({k: np.random.choice(s, p=p.get(k, None)) for k, s in self.sizes.items()})
                if self.valid(ind):
                    pop.add(ind)
            except ExceedsCapacity:
                break 
        if it == max_iters:
            raise ValueError("max iters reached while sampling a population")
        return Population(pop)

    def __repr__(self):
        return f"SearchSpace(num_vars={len(self)}, n_perms={float(self.n_perms)}, counts={self.sizes})"
