import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import iqr

from GPErks.log.logger import get_logger
from GPErks.utils.array import get_minmax
from GPErks.utils.indices import diff, part_and_select, whereq_whernot
from GPErks.utils.jsonfiles import load_json, save_json, load_pickle, save_pickle

log = get_logger()


class Wave:
    """
    A module to perform Bayesian history matching using trained univariate emulators to
    match target distributions' mean and SD values.
    """

    def __init__(
        self,
        emulator=None,
        Itrain=None,
        cutoff=None,
        maxno=None,
        mean=None,
        var=None,
    ):
        self.emulator = emulator
        self.Itrain = Itrain
        self.cutoff = cutoff
        self.maxno = maxno
        self.mean = mean
        self.var = var
        self.I = None
        self.PV = None
        self.NIMP = None
        self.nimp_idx = None
        self.IMP = None
        self.imp_idx = None

    def compute_impl(self, X, compute_I0 = False):
        n_samples = X.shape[0]
        output_dim = len(self.emulator)

        # Collect all means and variances in one go
        M = np.zeros((n_samples, output_dim), dtype=float)
        V = np.zeros((n_samples, output_dim), dtype=float)
        for j, emul in enumerate(self.emulator):
            mean, std = emul.predict(X)  # Assuming std is std. deviation
            M[:, j] = mean
            V[:, j] = np.square(std)

        # Add small epsilon to prevent divide-by-zero
        eps = 1e-10
        denom = V + self.var + eps
        num = np.square(M - self.mean)

        In = np.sqrt(num / denom)  # shape: (n_samples, output_dim)
        PVn = V / (self.var + eps)
        if compute_I0: I0n = np.sqrt(num / (self.var + eps))

        # Sort across output dimensions
        In_sorted = np.sort(In, axis=1)
        PVn_sorted = np.sort(PVn, axis=1)
        if compute_I0: I0n_sorted = np.sort(I0n, axis=1)

        # Extract the maxno-th largest value (i.e., from the end)
        I = In_sorted[:, -self.maxno]
        PV = PVn_sorted[:, -self.maxno]
        if compute_I0: 
            I0 = I0n_sorted[:, -self.maxno]
            return I, PV, I0

        return I, PV

    def compute_total_normalised_variance(self, X):
        n_samples = X.shape[0]
        output_dim = len(self.emulator)

        # Collect all means and variances in one go
        M = np.zeros((n_samples, output_dim), dtype=float)
        V = np.zeros((n_samples, output_dim), dtype=float)
        for j, emul in enumerate(self.emulator):
            mean, std = emul.predict(X)  # Assuming std is std. deviation
            M[:, j] = mean
            V[:, j] = np.square(std)

        # Add small epsilon to prevent divide-by-zero
        eps = 1e-10
        PVn = V / (self.var + eps)

        # Sort across output dimensions
        PVt = PVn.sum(axis=1)
        return PVt

    def find_regions(self, X):
        n_samples = X.shape[0]
        I, PV = self.compute_impl(X)
        l = list(np.where(I < self.cutoff)[0])
        nl = diff(range(n_samples), l)

        self.I = I
        self.PV = PV
        self.nimp_idx = l
        self.NIMP = X[l]
        self.imp_idx = nl
        self.IMP = X[nl]

    def print_stats(self):
        nimp = len(self.nimp_idx)
        imp = len(self.imp_idx)
        tests = nimp + imp
        perc = 100 * nimp / tests

        stats = pd.DataFrame(
            index=["TESTS", "IMP", "NIMP", "PERC"],
            columns=["#POINTS"],
            data=[tests, imp, nimp, f"{perc:.4f} %"],
        )
        print(stats)

    def reconstruct_tests(self):
        n_samples = self.NIMP.shape[0] + self.IMP.shape[0]
        input_dim = self.NIMP.shape[1]
        X = np.zeros((n_samples, input_dim), dtype=float)
        X[self.nimp_idx] = self.NIMP
        X[self.imp_idx] = self.IMP
        return X

    def save(self, filename:str):
        dct = vars(self)
        excluded_keys = ["emulator"]
        obj_dct = {}
        obj_dct.update({k: dct[k] for k in set(list(dct.keys())) - set(excluded_keys)})
        # save_json(obj_dct, filename)
        if '.json' in filename:
            filename_pkl = filename.replace('.json', '.pkl')
        elif '.pkl' not in filename:
            filename_pkl = filename + '.pkl'
        else:
            filename_pkl = filename
        save_pickle(obj_dct, filename_pkl)

    def load(self, filename):
        if '.json' in filename:
            filename_pkl = filename.replace('.json', '.pkl')
        elif '.pkl' not in filename:
            filename_pkl = filename + '.pkl'
        else:
            filename_pkl = filename
        obj_dict = load_pickle(filename_pkl)
        for k, v in obj_dict.items():
            setattr(self, k, v)

    def get_nimps(self, n_points):
        nimp = len(self.nimp_idx)
        if n_points >= nimp - 1:
            raise ValueError(
                "Not enough NIMP points to choose from! n_points must be strictly "
                + "less than W.NIMP.shape[0] - 1."
            )
        else:
            X = part_and_select(self.NIMP, n_points)
            _, nl = whereq_whernot(self.NIMP, X)
        return X, self.NIMP[nl]

    def get_nimps_adaptive(self, n_points, cutoff=3.0):
        """
        Adaptive version of get_nimps that prioritizes points with high GPE variance, 
        high implausibility, and spatial diversity for training Gaussian process emulators.
        
        Parameters
        ----------
        n_points : int
            Number of points to select for training
        alpha : float, optional
            Weight for implausibility score (default: 0.5)
            Higher values prioritize points closer to the implausibility boundary
        beta : float, optional  
            Weight for variance score (default: 0.5)
            Higher values prioritize points with high emulator uncertainty
        top_fraction : float, optional
            Fraction of NIMP points to consider (default: 0.2, must be <= 1.0)
            Selects top (top_fraction * nimp) candidates before applying part_and_select
            Higher values expand the candidate pool for better spatial diversity
            
        Returns
        -------
        X : np.ndarray
            Selected training points with shape (n_points, input_dim)
        X_remaining : np.ndarray
            Remaining NIMP points not selected for training
            
        Notes
        -----
        Two-stage selection process:
        1. Score all NIMP points: alpha * normalized_implausibility + beta * normalized_variance
        2. Select top (top_fraction * nimp) candidates by score
        3. Apply part_and_select to ensure spatial diversity among candidates
        This balances information value (implausibility + variance) with spatial spread.
        """
        nimp = len(self.nimp_idx)
        if n_points >= nimp - 1:
            raise ValueError(
                "Not enough NIMP points to choose from! n_points must be strictly "
                + "less than W.NIMP.shape[0] - 1."
            )
        
        # Compute implausibility and variance for all NIMP points
        _, _, I0 = self.compute_impl(self.NIMP, compute_I0=True)
        
        # Find the candidate points which have an ideal implausibility higher than cutoff
        candidate_mask = I0 >= cutoff
        
        # Select top candidates based on quality scores
        candidates = self.NIMP[candidate_mask]
        
        # Apply part_and_select for spatial diversity
        X = part_and_select(candidates, n_points)
        
        # Find remaining points
        _, nl = whereq_whernot(self.NIMP, X)
        X_remaining = self.NIMP[nl]
        
        return X, X_remaining

    # Note: the Wave object instance internal structure will be compromised after
    # calling this method: we recommend calling self.copy() and/or self.save()
    # beforehand!
    def augment_nimp(self, n_total_points, scaling=0.1, n_max=2000, max_seed_uses=None):
        lbounds = self.Itrain[:, 0]
        ubounds = self.Itrain[:, 1]

        log.info(
            f"\nRequested points: {n_total_points}\nAvailable points: "
            + f"{self.NIMP.shape[0]}\nStart searching..."
        )

        count = 0
        n_current = self.NIMP.shape[0]
        a, b = (
            n_current if n_current < n_total_points else n_total_points,
            n_total_points - n_current if n_total_points - n_current > 0 else 0,
        )
        log.info(
            f"[Iteration: {count:<2}] Found: {a:<{len(str(n_total_points))}} "
            + f"({'{:.2f}'.format(100*a/n_total_points):>6}%) | "
            + f"Missing: {b:<{len(str(n_total_points))}}"
        )

        # Preallocate a temporary array for perturbations to avoid repeated allocations
        X = np.zeros((n_total_points, self.NIMP.shape[1]), dtype=float)
        X[:n_current] = self.NIMP

        # Tracks how many times each point in X has been used as a perturbation
        # seed, so points that never yield valid (non-implausible) neighbours can
        # be retired instead of being resampled forever.
        seed_uses = np.zeros(n_total_points, dtype=int)

        while n_current < n_total_points:
            count += 1

            bounds = get_minmax(X[:n_current])
            scale = scaling * (bounds[:, 1] - bounds[:, 0])

            if max_seed_uses is not None:
                seed_idx = np.where(seed_uses[:n_current] < max_seed_uses)[0]
                if seed_idx.size == 0:
                    log.info(
                        f"\nAll {n_current} current points have reached the "
                        f"maximum number of seed uses ({max_seed_uses}); "
                        "stopping early."
                    )
                    n_total_points = n_current
                    break
            else:
                seed_idx = np.arange(n_current)

            seeds = X[seed_idx]
            temp = np.random.normal(loc=seeds, scale=scale)
            # Tracks, per row of temp, which seed (index into X) produced it.
            temp_seed_idx = seed_idx.copy()

            # Vectorized boundary checking
            count2 = 0
            while True:
                count2 += 1
                # Direct comparison: True where in bounds
                in_bounds = np.all((temp >= lbounds) & (temp <= ubounds), axis=1)
                out_of_bounds = ~in_bounds

                if count2 > n_max:
                    temp = temp[in_bounds]
                    temp_seed_idx = temp_seed_idx[in_bounds]
                    break
                if np.any(out_of_bounds):
                    temp[out_of_bounds] = np.random.normal(loc=X[temp_seed_idx[out_of_bounds]], scale=scale)
                    continue
                else:
                    break

            I, _ = self.compute_impl(temp)
            nimp_idx = np.where(I < self.cutoff)[0]
            valid_points = temp[nimp_idx]

            # A seed's use only "counts" against its budget when it produced an
            # implausible (I >= cutoff) point.
            failed_mask = I >= self.cutoff
            seed_uses[temp_seed_idx[failed_mask]] += 1

            if len(valid_points) > 0:
                if n_current + len(valid_points) > n_total_points:
                    valid_points = valid_points[: n_total_points - n_current]
                X[n_current:n_current + len(valid_points)] = valid_points
                n_current += len(valid_points)

            a, b = (
                n_current if n_current < n_total_points else n_total_points,
                n_total_points - n_current if n_total_points - n_current > 0 else 0,
            )
            log.info(
                f"[Iteration: {count:<2}] Found: {a:<{len(str(n_total_points))}} "
                + f"({'{:.2f}'.format(100*a/n_total_points):>6}%) | "
                + f"Missing: {b:<{len(str(n_total_points))}}"
            )

        log.info("\nDone.")

        X = X[:n_current]

        nimp = len(self.nimp_idx)
        NIMP_aug = part_and_select(X[nimp:], n_total_points - nimp)
        I, PV = self.compute_impl(NIMP_aug)
        self.NIMP = np.vstack((X[:nimp], NIMP_aug))
        self.I = np.concatenate((self.I, I))
        self.PV = np.concatenate((self.PV, PV))
        imp = len(self.imp_idx)
        self.nimp_idx += list(range(nimp + imp, imp + n_total_points))

    # Note: the Wave object instance internal structure will be compromised after
    # calling this method: we recommend calling self.copy() and/or self.save()
    # beforehand!
    def augment_nimp_efficient(
        self, n_total_points, scaling=0.1, n_max=2000, max_seed_uses=None, max_batch_size=None
    ):
        """
        Faster variant of augment_nimp.

        The original method draws exactly one perturbation per active seed
        per outer iteration, so when the non-implausible acceptance rate is
        low it needs many outer iterations -- each with its own
        self.compute_impl() call over every emulator -- to fill the request.
        This version tracks a running estimate of that acceptance rate and
        sizes each batch to close the remaining gap in as few iterations
        (and compute_impl calls) as possible, cycling through the active
        seeds as needed rather than being limited to one draw per seed. It
        also maintains the perturbation-scale bounds incrementally instead
        of recomputing get_minmax() over the whole, ever-growing point set
        on every iteration.

        max_batch_size caps how many candidates can be drawn in a single
        iteration (defaults to 20 * n_total_points) to bound memory use when
        the acceptance rate is very low.
        """
        lbounds = self.Itrain[:, 0]
        ubounds = self.Itrain[:, 1]

        if max_batch_size is None:
            max_batch_size = 20 * n_total_points

        log.info(
            f"\nRequested points: {n_total_points}\nAvailable points: "
            + f"{self.NIMP.shape[0]}\nStart searching..."
        )

        count = 0
        n_current = self.NIMP.shape[0]
        a, b = (
            n_current if n_current < n_total_points else n_total_points,
            n_total_points - n_current if n_total_points - n_current > 0 else 0,
        )
        log.info(
            f"[Iteration: {count:<2}] Found: {a:<{len(str(n_total_points))}} "
            + f"({'{:.2f}'.format(100*a/n_total_points):>6}%) | "
            + f"Missing: {b:<{len(str(n_total_points))}}"
        )

        X = np.zeros((n_total_points, self.NIMP.shape[1]), dtype=float)
        X[:n_current] = self.NIMP

        seed_uses = np.zeros(n_total_points, dtype=int)

        running_min = X[:n_current].min(axis=0)
        running_max = X[:n_current].max(axis=0)

        # Running estimate of the fraction of perturbation candidates that
        # turn out non-implausible. Starts optimistic (1.0) so the first
        # batch behaves like the original one-per-seed draw; converges to
        # whatever this region's true acceptance rate is.
        acceptance_rate = 1.0

        while n_current < n_total_points:
            count += 1

            scale = scaling * (running_max - running_min)

            if max_seed_uses is not None:
                seed_idx = np.where(seed_uses[:n_current] < max_seed_uses)[0]
                if seed_idx.size == 0:
                    log.info(
                        f"\nAll {n_current} current points have reached the "
                        f"maximum number of seed uses ({max_seed_uses}); "
                        "stopping early."
                    )
                    n_total_points = n_current
                    break
            else:
                seed_idx = np.arange(n_current)

            n_missing = n_total_points - n_current
            n_batch = int(
                np.clip(np.ceil(n_missing / acceptance_rate), n_missing, max_batch_size)
            )

            # Cycle through the active seeds (repeating as needed) instead of
            # being limited to one perturbation attempt per seed, so a single
            # batch can close the whole remaining gap even when few seeds are
            # left active.
            temp_seed_idx = seed_idx[np.arange(n_batch) % seed_idx.size]
            seeds = X[temp_seed_idx]
            temp = np.random.normal(loc=seeds, scale=scale)

            # Vectorized boundary checking
            count2 = 0
            while True:
                count2 += 1
                in_bounds = np.all((temp >= lbounds) & (temp <= ubounds), axis=1)
                out_of_bounds = ~in_bounds

                if count2 > n_max:
                    temp = temp[in_bounds]
                    temp_seed_idx = temp_seed_idx[in_bounds]
                    break
                if np.any(out_of_bounds):
                    temp[out_of_bounds] = np.random.normal(loc=X[temp_seed_idx[out_of_bounds]], scale=scale)
                    continue
                else:
                    break

            I, _ = self.compute_impl(temp)
            nimp_idx = np.where(I < self.cutoff)[0]
            valid_points = temp[nimp_idx]

            # A seed's use only "counts" against its budget when it produced
            # an implausible (I >= cutoff) point. temp_seed_idx may contain
            # repeated seed indices (a seed can appear in several rows of
            # this batch), so np.add.at is required here -- plain fancy-index
            # += silently drops all but one increment per repeated index.
            failed_mask = I >= self.cutoff
            np.add.at(seed_uses, temp_seed_idx[failed_mask], 1)

            # Update the acceptance-rate estimate (exponential moving
            # average) so the next batch size reflects how hard this region
            # is to sample.
            if len(temp) > 0:
                observed_rate = len(valid_points) / len(temp)
                acceptance_rate = 0.5 * acceptance_rate + 0.5 * max(
                    observed_rate, 1.0 / max_batch_size
                )

            if len(valid_points) > 0:
                if n_current + len(valid_points) > n_total_points:
                    valid_points = valid_points[: n_total_points - n_current]
                X[n_current:n_current + len(valid_points)] = valid_points
                running_min = np.minimum(running_min, valid_points.min(axis=0))
                running_max = np.maximum(running_max, valid_points.max(axis=0))
                n_current += len(valid_points)

            a, b = (
                n_current if n_current < n_total_points else n_total_points,
                n_total_points - n_current if n_total_points - n_current > 0 else 0,
            )
            log.info(
                f"[Iteration: {count:<2}] Found: {a:<{len(str(n_total_points))}} "
                + f"({'{:.2f}'.format(100*a/n_total_points):>6}%) | "
                + f"Missing: {b:<{len(str(n_total_points))}}"
            )

        log.info("\nDone.")

        X = X[:n_current]

        nimp = len(self.nimp_idx)
        NIMP_aug = part_and_select(X[nimp:], n_total_points - nimp)
        I, PV = self.compute_impl(NIMP_aug)
        self.NIMP = np.vstack((X[:nimp], NIMP_aug))
        self.I = np.concatenate((self.I, I))
        self.PV = np.concatenate((self.PV, PV))
        imp = len(self.imp_idx)
        self.nimp_idx += list(range(nimp + imp, imp + n_total_points))

    def get_trains(self, X, n_points):
        n_samples = X.shape[0]
        if n_points > n_samples:
            raise ValueError(
                "Cannot return more points than totally available points! "
                + "Choose n_points <= X_train.shape[0]."
            )
        elif n_points == n_samples:
            return X
        else:
            I, PV = self.compute_impl(X)
            l = np.argsort(I)[:n_points]
            return X[l]

    def copy(self):
        W = Wave(
            emulator=self.emulator,
            Itrain=self.Itrain,
            cutoff=self.cutoff,
            maxno=self.maxno,
            mean=self.mean,
            var=self.var,
        )
        W.I = np.copy(self.I)
        W.PV = np.copy(self.PV)
        W.NIMP = np.copy(self.NIMP)
        W.nimp_idx = self.nimp_idx.copy()
        W.IMP = np.copy(self.IMP)
        W.imp_idx = self.imp_idx.copy()
        return W

    def plot_wave(self, xlabels=None, display="impl", filepath=None, figsize=None):
        X = self.reconstruct_tests()
        input_dim = X.shape[1]

        if xlabels is None:
            xlabels = [f"p{i+1}" for i in range(X.shape[1])]

        if display == "impl":
            C = self.I
            cmap = "jet"
            vmin = 1.0
            vmax = self.cutoff
            cbar_label = "Implausibility measure"

        elif display == "var":
            C = self.PV
            cmap = "bone_r"
            vmin = np.max(
                [
                    np.percentile(self.PV, 25) - 1.5 * iqr(self.PV),
                    self.PV.min(),
                ]
            )
            vmax = np.min(
                [
                    np.percentile(self.PV, 75) + 1.5 * iqr(self.PV),
                    self.PV.max(),
                ]
            )
            cbar_label = "GPE variance / EXP. variance"

        else:
            raise ValueError(
                "Not a valid display option! Can only display implausibilty maps "
                + "('impl') or proportion-of-exp.variance maps ('var')."
            )

        height = 9.36111
        width = 5.91667
        if figsize is None:
            fig = plt.figure(figsize=(2 * width, 1.2 * 2 * height / 3))
        else:
            fig = plt.figure(figsize=figsize)
        gs = grsp.GridSpec(
            input_dim - 1,
            input_dim,
            width_ratios=(input_dim - 1) * [1.0] + [0.1],
        )

        for k in range(input_dim * input_dim):
            i = k % input_dim
            j = k // input_dim

            if i > j:
                axis = fig.add_subplot(gs[i - 1, j])
                axis.set_facecolor("xkcd:light grey")

                im = axis.hexbin(
                    X[:, j],
                    X[:, i],
                    C=C,
                    reduce_C_function=np.min,
                    gridsize=20,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                axis.set_xlim([self.Itrain[j, 0], self.Itrain[j, 1]])
                axis.set_ylim([self.Itrain[i, 0], self.Itrain[i, 1]])

                if i == input_dim - 1:
                    axis.set_xlabel(xlabels[j], fontsize=12)
                else:
                    axis.set_xticklabels([])
                if j == 0:
                    axis.set_ylabel(xlabels[i], fontsize=12)
                else:
                    axis.set_yticklabels([])

        cbar_axis = fig.add_subplot(gs[:, input_dim - 1])
        cbar = fig.colorbar(im, cax=cbar_axis)
        cbar.set_label(cbar_label, fontsize=12)
        fig.tight_layout()
        if filepath is not None:
            plt.savefig(filepath, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()
