#!/usr/bin/env python3
# pylint: disable=missing-module-docstring

from functools import partial
import json
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

import statsmodels.api as sm

from datasets import *

random_state = 0


def _main():
    # Load digits dataset
    #x, y = datasets.load_digits(return_X_y=True)

    #scaler = StandardScaler()
    #x_scaled = scaler.fit_transform(x)
    ## select 0 and 1
    #idx = np.where(np.logical_or(y == 0, y == 1))
    #y = y[idx]
    #x_scaled = x_scaled[idx]

    one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

    ## Load camp datasets
    data_dir = "../data/"
    folds = ["Minawao_feb_2017", "Tza_oct_2016"]#"Minawao_june_2016"]#
    inds = [folds.index(fold) for fold in folds] # index o each folder in the list for later use
    paths = [
        data_dir + f"/{folder}" for folder in folds
    ]  # dataset paths taken directly from the list
    with open("params_file.txt", "r") as params_log:
        params = json.load(params_log)
    train_dataset = [
        TrainDataset(
            root=paths[ind],
            func=params[folds[ind]]["func"],
            equalize=params[folds[ind]]["equalize"],
            nb_channels=5,
            ndvi_treshold=params[folds[ind]]["ndvi_treshold"],
            intensity_treshold=params[folds[ind]]["intensity_treshold"],
            fake_dataset_size=100,
            c_treshold=params[folds[ind]]["contrast_treshold"],
            b_treshold=params[folds[ind]]["brightness_treshold"],
            with_prob=False,
            with_condition=True,
        )
        for ind in inds
    ]
    images_1ch = []
    labels = []
    labels_ = []
    ndvi = []
    for i in range((train_dataset[0].__len__())):
        images_1ch.append(train_dataset[0].__getitem__(i)[0][:3, ::4,::4].flatten().numpy())
        labels.append(-1)#"Minawao_feb_2017")
        labels_.append(np.zeros(images_1ch[0].shape))#"Minawao_june_2016")
        ndvi.append(train_dataset[0].__getitem__(i)[0][4, ::4,::4].flatten().numpy())

    for i in range((train_dataset[1].__len__())):
        images_1ch.append(train_dataset[1].__getitem__(i)[0][:3, ::4,::4].flatten().numpy())
        labels.append(1)#"Minawao_june_2016")
        labels_.append(np.ones(images_1ch[0].shape))#"Minawao_june_2016")
        ndvi.append(train_dataset[1].__getitem__(i)[0][4, ::4,::4].flatten().numpy())

    x = np.stack(images_1ch, axis=0).astype(np.float32)
    rng = np.random.default_rng(12345)
    rng.shuffle(x)
    x = x[:500]
    y = np.stack(labels, axis=0)
    #y_ = np.stack(labels_, axis=0)
    y_ = np.stack(ndvi, axis=0)
    rng = np.random.default_rng(12345)
    rng.shuffle(y)
    y = y[:500]
    rng = np.random.default_rng(12345)
    rng.shuffle(y_)
    y_ = y_[:500]

    def plot_several(list_x, list_y):
        fig, axes = plt.subplots(len(list_x), len(list_x[0]))
        for j, x in enumerate(list_x):
            for i, x_ in enumerate(x):
                stand_img = np.moveaxis(x_.reshape(3, 64, 64), 0, 2)
                stand_img = ((stand_img - np.amin(stand_img)) /
                    (np.amax(stand_img) - np.amin(stand_img)))
                axes[j, i].imshow(
                        stand_img
                    )
                axes[j, i].set_title(list_y[j][i])
        plt.show()
    #plot_several(x[:6], y[:6])

    #plt.imshow(x[0].reshape(64, 64))
    #plt.show()

    x = np.concatenate([x, y_], axis=1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    q = 300

    # First PCA to reduce dimensionality
    pca_preproc = PCA(n_components=400, random_state=random_state)
    pca_preproc.fit(x_scaled)
    x_scaled_pca = pca_preproc.transform(x_scaled)
    #print("Done with first PCA")
    #print(pca.noise_variance_, pca.score(x_scaled))


    # Second PCA 
    pca = PCA(n_components=q, random_state=random_state)
    pca.fit(x_scaled_pca)
    x_embedded_pca = pca.transform(x_scaled_pca)
    #print(pca.noise_variance_, pca.score(x_scaled))

    avg_llkh, sig2, x_embedded_ppca = ppca(x_scaled_pca, q=q)
    #print(sig2, avg_llkh)

    # For PPCCA our covariables are the average pixel value and the percentage
    # of white pixel (standardized value > 0.5)
    # covars = jnp.stack([
    #    jnp.mean(x, axis=1),
    #    jnp.sum((x - jnp.amin(x)) / (jnp.amax(x) - jnp.amin(x)) > 0.5, axis=1) /
    #        x.shape[1]
    #    ],
    #    axis=1
    # )
    # Or try covariables corresponding to the one hot encoding of the label
    covars = one_hot_encoder.fit_transform(y.reshape((-1, 1)))
    #covars = np.where(covars == 0, -10, 10)
    #covars = (np.repeat(y[:, None], repeats=2, axis=1)) * 10.

    alpha, sig2, x_embedded_ppcca, W, muhat = ppcca(x_scaled_pca, covars=covars, q=q)
    #print(alpha, sig2)

    #print(W.shape, muhat.shape, x_embedded_ppcca.shape)
    #print(y[1])
    
    recs = [pca_preproc.inverse_transform(W @ x_embedded_ppcca[i].T + muhat)[:,
        :3 * 4096]
            for i in range(6)]
    print(recs[0].shape)
    #plot_several(recs, y[:6])

    #Modify reconstructions
    for i in range(len(x_embedded_ppcca)):
        if y[i] == 1:
            x_embedded_ppcca = x_embedded_ppcca.at[i, -1].set(10)
        else:
            x_embedded_ppcca = x_embedded_ppcca.at[i, -1].set(10)
    recs_mod = [pca_preproc.inverse_transform(W @ x_embedded_ppcca[i].T + muhat)[:,
        :4096 * 3]
            for i in range(6)]
    print(recs_mod[0].shape)
    plot_several([x[:6, :3 * 4096], recs, recs_mod], [y[:6], y[:6], y[:6]])

    fig, axes = plt.subplots(1, 6)
    axes[0].scatter(x_embedded_pca[:, 0], x_embedded_pca[:, 1], c=y)  # , cmap="Set1")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    scatter = axes[1].scatter(x_embedded_ppca[:, 0], x_embedded_ppca[:, 1], c=y)  # , cmap="Set1")
    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(x_embedded_ppcca[:, :-1])
    x_embedded_ppcca_red = pca.transform(x_embedded_ppcca[:, :-1])
    scatter = axes[2].scatter(
        x_embedded_ppcca_red[:, 0], x_embedded_ppcca_red[:, 1], c=y
    )  # , cmap="Set1")
    scatter = axes[3].scatter(
         np.zeros(len(x_embedded_ppcca[:, 0])),x_embedded_ppcca[:, 0], c=y
    )  # , cmap="Set1")
    scatter = axes[4].scatter(
         np.zeros(len(x_embedded_ppcca[:, 0])),x_embedded_ppcca[:, 1], c=y
    )  # , cmap="Set1")
    scatter = axes[5].scatter(
         np.zeros(len(x_embedded_ppcca[:, 0])),x_embedded_ppcca[:, -1], c=y
    )  # , cmap="Set1")
    fig.show()
    plt.legend(handles=scatter.legend_elements()[0], labels=list(np.arange(10)))
    plt.show()


@partial(jax.jit, static_argnums=1)
def ppca(X, q):
    """
    EM algorithm for Probabilistic PCA

    Parameters
    ----------
    X : array
      The array of data on which to perform PPCA of shape (n_samples,
      n_features)
    q : integer
      The dimension of the feature space to learn

    Returns
    -------
    double
      the value of the average likelihood under the learnt model
    double
      the value of the noise variance in the learnt model
    jnp array
      the score, ie the project of the samples in feature space that has been
      learnt
    """

    max_iter = 100
    N, p = X.shape  # number of observations and variables

    muhat = jnp.mean(X, axis=0, keepdims=1)
    Xc = X - muhat

    # init the model
    S = 1 / N * (Xc.T @ Xc)
    temp_val, temp_vec = jnp.linalg.eig(S)
    idx_sorting = jnp.argsort(temp_val)[::-1]  # because it is sort in decreasing
    temp_val = jnp.real(temp_val[idx_sorting])
    temp_vec = jnp.real(temp_vec[:, idx_sorting])
    sig2 = jnp.abs(1 / (p - q)) * jnp.sum(temp_val[q : p + 1])  # start val for
    # variance, it is a scalar
    W = temp_vec[:, :q]  # start val for loading, proj matrix of dim (p, q)

    def _scan_fun(carry, k):  # pylint: disable=unused-argument
        W, sig2 = carry

        # E-Step
        M_1 = jnp.linalg.inv(W.T @ W + sig2 * jnp.diag(jnp.ones(q)))

        _x_ = M_1 @ (W.T @ Xc.T)  # dim (q, N) or (q, 1) if we strictly follow
        # the formula but we have vectorized the computation over all the samples
        # thanks to the matrix product

        sum_xx_ = N * sig2 * M_1 + (_x_ @ _x_.T)  # dim (q, q), same remark, it has
        # been vectorized. Note the N * because it is **sum**_xx_

        # M-Step
        W = (Xc.T @ _x_.T) @ jnp.linalg.inv(sum_xx_)  # dim (p, q), estimation
        # of the loadings, same remark, it has been vectorized

        sig2 = (
            1
            / (N * p)
            * (
                jnp.sum(jnp.square(Xc), axis=(0, 1))
                - 2 * jnp.trace(_x_.T @ W.T @ Xc.T)
                + jnp.trace((W.T @ W) @ sum_xx_)
            )
        )

        return (W, sig2), _x_

    (W, sig2), list_x_ = jax.lax.scan(_scan_fun, (W, sig2), jnp.arange(max_iter))

    # compute loglikelihood
    C = sig2 * jnp.diag(jnp.ones(p)) + W @ W.T
    avg_llkh = (
        -1
        / 2
        * (
            p * jnp.log(2 * jnp.pi)
            + jnp.log(jnp.linalg.det(C))
            + jnp.trace(jnp.linalg.inv(C) @ S)
        )
    )

    return avg_llkh, sig2, list_x_[-1].T


def ppcca(X, covars, q):
    """
    EM algorithm for Probabilistic Principal Components and Covariates Analysis

    Parameters
    ----------
    X : array
      The array of data on which to perform PPCCA of shape (n_samples,
      n_features)
    covars : array
      The array of covariables to consider of shape (n_samples, n_covariables)
    q : integer
      The dimension of the feature space to learn

    Returns
    -------
    double
      the value of the average likelihood under the learnt model
    double
      the value of the noise variance in the learnt model
    jnp array
      the score, ie the project of the samples in feature space that has been
      learnt
    """

    max_iter = 1000
    N, p = X.shape  # number of observations and variables

    null_alpha = False
    if covars is None:
        L = 1
        covars = jnp.ones((L + 1, N))
        null_alpha = True
    else:
        L = covars.shape[1]
        # standardize the covars for stability
        #covars = (covars - jnp.amin(covars, axis=0, keepdims=1)) / (
        #    jnp.amax(covars, axis=0, keepdims=1) - jnp.amin(covars, axis=0, keepdims=1)
        #)
        covars = jnp.concatenate(
                #NOTE
            #[jnp.ones((2, N)), covars.T], axis=0
            [jnp.zeros((q - 1, N)), covars.T], axis=0
        )  # now dim (L + 1, N)
        #covars = covars.T

    muhat = jnp.mean(X, axis=0, keepdims=1)
    Xc = X - muhat

    # init the model
    S = 1 / N * (Xc.T @ Xc)
    temp_val, temp_vec = jnp.linalg.eig(S)
    idx_sorting = jnp.argsort(temp_val)[::-1]  # because it is sort in decreasing
    temp_val = jnp.real(temp_val[idx_sorting])
    temp_vec = jnp.real(temp_vec[:, idx_sorting])
    sig2 = jnp.abs(1 / (p - q)) * jnp.sum(
        temp_val[q : p + 1]
    )  # init variance, it is a scalar
    W = temp_vec[:, :q]  # init proj matrix of dim (p, q)

    ## initialization of alpha
    #if null_alpha:
    alpha = jnp.zeros((q, L + 1))
    #else:
    #    scores = jnp.transpose(
    #        jnp.linalg.inv((W.T @ W) + (sig2 * jnp.diag(jnp.ones(q)))) @ W.T @ Xc.T
    #    )
    #    alpha = np.empty((q, L + 1))
    #    for i in range(q):
    #        alpha[i] = (
    #            sm.GLM(
    #                np.asarray(scores[:, i]),
    #                np.asarray(
    #                    covars[
    #                        0:,
    #                    ]
    #                ).T,
    #                family=sm.families.Gaussian(),
    #            )
    #            .fit()
    #            .params
    #        )
    #    alpha = jnp.asarray(alpha)

    def _scan_fun(carry, k):  # pylint: disable=unused-argument
        W, sig2, alpha = carry

        # E-Step
        M_1 = jnp.linalg.inv(W.T @ W + sig2 * jnp.diag(jnp.ones(q)))

        _x_ = M_1 @ (
            # W.T @ Xc.T + sig2 * (alpha @ covars)
            # NOTE
            W.T @ Xc.T + sig2 * covars
        )  # dim (q, N) or (q, 1) if we strictly follow
        # the formula but we have vectorized the computation over all the samples
        # thanks to the matrix product

        sum_xx_ = N * sig2 * M_1 + (_x_ @ _x_.T)  # dim (q, q), same remark, it has
        # been vectorized. Note the N * because it is **sum**_xx_

        # M-Step
        #NOTE
        #alpha = (_x_ @ covars.T) @ jnp.linalg.inv(covars @ covars.T)  # estimation
        # of the regression coefficient, vectorized
        W = (Xc.T @ _x_.T) @ jnp.linalg.inv(sum_xx_)  # dim (p, q), estimation
        # of the loadings, same remark, it has been vectorized

        sig2 = (
            1
            / (N * p)
            * (
                jnp.sum(jnp.square(Xc), axis=(0, 1))
                - 2 * jnp.trace(_x_.T @ W.T @ Xc.T)
                + jnp.trace((W.T @ W) @ sum_xx_)
            )
        )

        return (W, sig2, alpha), _x_

    (W, sig2, alpha), list_x_ = jax.lax.scan(
        _scan_fun, (W, sig2, alpha), jnp.arange(max_iter)
    )

    return alpha, sig2, list_x_[-1].T, W, muhat


if __name__ == "__main__":
    _main()
