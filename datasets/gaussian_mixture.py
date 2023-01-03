import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["GaussianMixture"]


class GaussianMixture(Dataset):
    N_CLASSES = 2

    def __init__(self, sdev, n):
        """
        Class to generate data sampled from a Gaussian mixture

        where theta is sampled uniformly at random from the sphere
        # :param n_classes: number of classification categories
        # :param dim: input dimension
        :param sdev: standard deviation of the covariates
        :param n: number of samples
        """
        # generate centers
        center_pos = np.array([2, 0.5])
        center_neg = np.array([-2, -0.5])
        cov_mat = sdev ** 2 * np.identity(2)

        # generate Gaussian blob for each class
        n_samples_per_class = n // 2
        samples_pos = np.random.multivariate_normal(mean=center_pos, cov=cov_mat, size=n_samples_per_class)
        samples_neg = np.random.multivariate_normal(mean=center_neg, cov=cov_mat, size=n_samples_per_class)
        # self._x = np.concatenate([samples_pos, samples_neg], axis=0, dtype=float)
        # self._y = np.concatenate([np.zeros(n_samples_per_class, dtype=int),
        #                           np.ones(n_samples_per_class, dtype=int)])
        self._x = np.concatenate([samples_pos, samples_neg], axis=0)
        self._y = np.concatenate([np.zeros(n_samples_per_class, dtype=int),
                                  np.ones(n_samples_per_class, dtype=int)])

    @property
    def data(self):
        return self._x, self._y

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._x[idx], self._y[idx]


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     ds = GaussianMixture(1.0, 10000)
#     x, y = ds.data
#     plt.scatter(x[y == 0, 0], x[y == 0, 1], s=30)
#     plt.scatter(x[y == 1, 0], x[y == 1, 1], s=30)
#     plt.savefig('tmp')
