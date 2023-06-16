import ot
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def computeWasser(mu, nu):
    n_batch = mu.shape[0]
    M = torch.zeros([n_batch,n_batch])
    for i, pathi in enumerate(mu):
        for j, pathj in enumerate(nu):
            M[i,j] = (pathi - pathj).abs().sum()
    a, b = torch.ones((n_batch,)) / n_batch, torch.ones((n_batch,)) / n_batch
    G0 = ot.emd(a, b, M)
    WD = (G0*M).sum().numpy()
    return WD


def plot_reals_recons_fakes(x_reals, y_reals, x_fakes):
    x_real_dim = x_reals.shape[-1]
    for i in range(x_real_dim):
        fig, ax = plt.subplots(1,3, figsize = [16,4], sharex=True, sharey=True)
        ax[0].plot(x_reals[:100,:,i].T, alpha=0.3)
        ax[1].plot(y_reals[:100,:,i].T, alpha=0.3)
        ax[2].plot(x_fakes[:100,:,i].T, alpha=0.3)
    plt.show()
