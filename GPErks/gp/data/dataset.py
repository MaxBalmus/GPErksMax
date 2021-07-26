from typing import Callable, List, Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import qmc

from GPErks.constants import HEIGHT, WIDTH
from GPErks.plot.options import PlotOptions
from GPErks.plot.plottable import Plottable


class RandomEngine(qmc.QMCEngine):  # TODO: move somewhere else
    def __init__(self, d, seed=None):
        super().__init__(d=d, seed=seed)

    def random(self, n=1):
        self.num_generated += n
        return self.rng.random((n, self.d))

    def reset(self):
        super().__init__(d=self.d, seed=self.rng_seed)
        return self

    def fast_forward(self, n):
        self.random(n)
        return self


class Dataset(Plottable):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        x_labels: Optional[List[str]] = None,
        y_label: Optional[str] = None,
        plot_options: PlotOptions = PlotOptions(),
    ):
        super(Dataset, self).__init__(plot_options)
        self.X_train: np.ndarray = X_train
        self.y_train: np.ndarray = y_train
        self.X_val: Optional[np.ndarray] = X_val
        self.y_val: Optional[np.ndarray] = y_val
        self.X_test: Optional[np.ndarray] = X_test
        self.y_test: Optional[np.ndarray] = y_test
        self.with_val: bool = X_val is not None and y_val is not None

        self.sample_size = self.X_train.shape[0]
        self.input_size = (
            self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1
        )

        self.x_labels: List[str] = (
            x_labels
            if x_labels
            else [f"p{i+1}" for i in range(self.input_size)]
        )
        self.y_label: str = y_label if y_label else "Output"

    def plot(self):
        self.plot_train()

    def plot_pairwise(self):
        self._plot_pairwise(self.X_train)

    def plot_train(self):
        self._plot(self.X_train, self.y_train)

    def plot_test(self):
        self._plot(self.X_test, self.y_test)

    def plot_val(self):
        self._plot(self.X_val, self.y_val)

    def _plot(self, X, y):
        sample_size = X.shape[0]
        self.input_size = X.shape[1]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=self.input_size,
            sharey="row",
            figsize=(2 * WIDTH, 2 * HEIGHT / 5),
        )
        for i, axis in enumerate(axes.flat):
            axis.scatter(X[:, i], y, facecolor="C0", edgecolor="C0")
            axis.set_xlabel(self.x_labels[i], fontsize=12)
        axes[0].set_ylabel(self.y_label, fontsize=12)

        plt.suptitle(f"Sample dimension = {sample_size} points", fontsize=12)
        plt.show()

    def _plot_pairwise(self, X):
        sample_size = X.shape[0]
        self.input_size

        fig, axes = plt.subplots(
            nrows=self.input_size,
            ncols=self.input_size,
            sharex="col",
            sharey="row",
            figsize=(2 * WIDTH, 2 * HEIGHT / 2),
        )
        for t, axis in enumerate(axes.flat):
            i = t % self.input_size
            j = t // self.input_size
            if j >= i:
                axis.scatter(X[:, i], X[:, j], facecolor="C0", edgecolor="C0")
            else:
                axis.set_axis_off()
            if i == 0:
                axis.set_ylabel(self.x_labels[j])
            if j == self.input_size - 1:
                axis.set_xlabel(self.x_labels[i])

        plt.suptitle(f"Sample dimension = {sample_size} points", fontsize=12)
        plt.show()

    @classmethod
    def build_from_function(
        cls,
        f: Callable[[np.ndarray], np.ndarray],
        d: int,
        n_train_samples: int,
        n_val_samples: int,
        n_test_samples: int,
        design: str = "lhs",
        seed: Optional[int] = None,
        l_bounds: Optional[List[float]] = None,
        u_bounds: Optional[List[float]] = None,
    ):

        if design == "srs":
            sampler = RandomEngine(d=d, seed=seed)
        elif design == "lhs":
            sampler = qmc.LatinHypercube(d=d, seed=seed)
        elif design == "sobol":
            sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        else:
            raise ValueError(
                "Not a valid sampling design! Choose between 'srs', 'lhs', 'sobol'"
            )

        X_train = sampler.random(n=n_train_samples)
        X_val = sampler.random(n=n_val_samples)
        X_test = sampler.random(n=n_test_samples)

        if l_bounds is not None and u_bounds is not None:
            X_train = qmc.scale(X_train, l_bounds, u_bounds)
            X_val = qmc.scale(X_val, l_bounds, u_bounds)
            X_test = qmc.scale(X_test, l_bounds, u_bounds)

        y_train = f(X_train)
        y_val = f(X_val)
        y_test = f(X_test)

        return cls(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
