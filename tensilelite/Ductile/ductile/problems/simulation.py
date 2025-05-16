import os.path

from ..core import SearchSpace, Selection, Crossover, Mutation, Mating, Survival
from ..algorithm import GeneticAlgorithm

from .. import config as config
import numpy as np
import pickle
import pandas as pd

from functools import partial


def categorize(params, el):
    return dict(sorted({k: el[k] if not isinstance(el[k], list) else params[k].index(el[k]) for k in el}.items()))


def build_df(params, solutions):
    X = []
    y = []
    for sol, score in solutions:
        X.append(categorize(params, sol))
        y.append(max(score, 0))
    y = np.array(y)
    df = pd.DataFrame(X)
    return df, y


def build_X(params, solutions):
    df = pd.DataFrame([categorize(params, sol) for sol in solutions])
    return df


def build_regressor(X, y):
    import lightgbm as lgb

    # Model settings
    MAX_LEAF = 512
    NUM_OF_TREES = 160
    LEARNING_RATE = 0.1
    METRIC = "l1"

    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": METRIC,
        "num_leaves": MAX_LEAF,
        "n_estimators": NUM_OF_TREES,
        "boost_from_average": True,
        "n_jobs": -1,
        "learning_rate": LEARNING_RATE,
        "seed": 42
    }

    lgb_regressor = lgb.LGBMRegressor(**params)

    lgb_regressor.fit(
        X=X,
        y=y,

    )
    return lgb_regressor


def fn(regr, params, X):
    scores = regr.predict(build_X(params, X))
    scores = np.clip(scores, -1, np.inf)
    return scores


class Simulation:
    def __init__(self, cfg):
        data = pickle.load(open("sim/data0.pkl", "rb"))
        params = data["params"]
        params = dict(sorted({k: np.unique(v, axis=0).tolist() for k, v in params.items()}.items()))

        if os.path.isfile("sim/regr.pkl"):
            regr = pickle.load(open("sim/regr.pkl", "rb"))
        else:
            for f in os.listdir("sim"):
                if "data" not in f or f == "data0.pkl": continue
                nd = pickle.load(open(os.path.join("sim", f), "rb"))
                data["solutions"].extend(nd["solutions"])
            X, y = build_df(params, data["solutions"])
            regr = build_regressor(X, y)
            print(f"Regressor R2={regr.score(X, y)}")
            pickle.dump(regr, open("sim/regr.pkl", "wb"))

        # cfg = config.load()
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
                                          evaluate=partial(fn, regr, params),
                                          survival=survival,
                                          pop_size=cfg["pop_size"],
                                          n_gen=cfg["n_gen"],
                                          n_offspring=cfg["n_offspring"],
                                          tol=cfg["tol"],
                                          period=cfg["period"],
                                          seed=cfg["seed"],
                                          verbose=cfg["verbose"],
                                          log_file=cfg["log_file"])

    def solve(self):
        return self.algorithm.optimize()
