from matplotlib import pyplot as plt
import numpy as np


def plot_hist_marginals(data, weights=None, lims=None, gt=None, upper=False, rasterized=False):
    """
    Plots marginal histograms and pairwise scatter plots of a dataset.
    """

    data = np.asarray(data)
    n_bins = int(np.sqrt(data.shape[0]))

    if data.ndim == 1:

        fig, ax = plt.subplots(1, 1)
        ax.hist(data, weights=weights, bins=n_bins, normed=True, rasterized=rasterized)
        ax.set_ylim([0.0, ax.get_ylim()[1]])
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if lims is not None: ax.set_xlim(lims)
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        n_dim = data.shape[1]
        fig = plt.figure()

        if weights is None:
            col = 'k'
            vmin, vmax = None, None
        else:
            col = weights
            vmin, vmax = 0., np.max(weights)

        if lims is not None:
            lims = np.asarray(lims)
            lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

        for i in range(n_dim):
            for j in range(i, n_dim) if upper else range(i + 1):

                ax = fig.add_subplot(n_dim, n_dim, i * n_dim + j + 1)

                if i == j:
                    ax.hist(data[:, i], weights=weights, bins=n_bins, normed=True, rasterized=rasterized)
                    ax.set_ylim([0.0, ax.get_ylim()[1]])
                    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                    if i < n_dim - 1 and not upper: ax.tick_params(axis='x', which='both', labelbottom=False)
                    if lims is not None: ax.set_xlim(lims[i])
                    if gt is not None: ax.vlines(gt[i], 0, ax.get_ylim()[1], color='r')

                else:
                    ax.scatter(data[:, j], data[:, i], c=col, s=3, marker='o', vmin=vmin, vmax=vmax, cmap='binary', edgecolors='none', rasterized=rasterized)
                    if i < n_dim - 1: ax.tick_params(axis='x', which='both', labelbottom=False)
                    if j > 0: ax.tick_params(axis='y', which='both', labelleft=False)
                    if j == n_dim - 1: ax.tick_params(axis='y', which='both', labelright=True)
                    if lims is not None:
                        ax.set_xlim(lims[j])
                        ax.set_ylim(lims[i])
                    if gt is not None: ax.scatter(gt[j], gt[i], c='r', s=12, marker='o', edgecolors='none')

    return fig