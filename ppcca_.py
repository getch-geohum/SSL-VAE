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

    ### TRAIN
    ## Load camp datasets
    data_dir = "../data/"
    folds = ["Minawao_feb_2017", "Tza_oct_2016", "Minawao_june_2016",
        "Deghale_Apr_2017", "Kule_tirkidi_marc_2017"]
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

    nb_samples = 500
    nb_datasets = len(train_dataset)
    nb_samples_by_dataset = nb_samples // nb_datasets
    images_1ch = []
    labels = []
    labels_ = []
    ndvi = []
    for i in range(nb_samples_by_dataset):
        for d in range(nb_datasets):
            images_1ch.append(train_dataset[d].__getitem__(i)[0][:3, :, :].flatten().numpy())
            labels.append(d + 1)
        #labels_.append(np.zeros(images_1ch[0].shape))
        #ndvi.append(train_dataset[0].__getitem__(i)[0][4, :,:].flatten().numpy())

    x = np.stack(images_1ch, axis=0).astype(np.float32)
    rng = np.random.default_rng(12345)
    rng.shuffle(x)
    x = x[:nb_samples]
    y = np.stack(labels, axis=0)
    #y_ = np.stack(labels_, axis=0)
    #y_ = np.stack(ndvi, axis=0)
    rng = np.random.default_rng(12345)
    rng.shuffle(y)
    y = y[:nb_samples]
    #rng = np.random.default_rng(12345)
    #rng.shuffle(y_)
    #y_ = y_[:500]
    #####

    ## TEST
    test_dataset = [
        TestDataset(
            root=paths[ind],
            func=params[folds[ind]]["func"],
            equalize=params[folds[ind]]["equalize"],
            nb_channels=5,
            ndvi_treshold=params[folds[ind]]["ndvi_treshold"],
            intensity_treshold=params[folds[ind]]["intensity_treshold"],
            fake_dataset_size=100,
            c_treshold=params[folds[ind]]["contrast_treshold"],
            b_treshold=params[folds[ind]]["brightness_treshold"],
            with_condition=True,
        )
        for ind in inds
    ]
    nb_samples_test = 100
    nb_datasets_test = len(test_dataset)
    nb_samples_by_dataset_test = nb_samples_test // nb_datasets_test
    images_1ch = []
    labels = []
    for i in range(nb_samples_by_dataset_test):
        for d in range(nb_datasets_test):
            images_1ch.append(test_dataset[d].__getitem__(i)[0][:3, :, :].flatten().numpy())
            labels.append((d + 1) * 1)

    x_test = np.stack(images_1ch, axis=0).astype(np.float32)
    rng = np.random.default_rng(12345)
    rng.shuffle(x_test)
    x_test = x_test[:nb_samples_test]
    y_test = np.stack(labels, axis=0)
    rng = np.random.default_rng(12345)
    rng.shuffle(y_test)
    y_test = y_test[:nb_samples_test]
    y_test_legend = y_test + nb_datasets
    #####

    def plot_several(list_x, list_y, normalize=True):
        fig, axes = plt.subplots(len(list_x), len(list_x[0]))
        all_times_max = [-np.inf for i in range(len(list_x))]
        all_times_min = [np.inf for i in range(len(list_x))]
        for j, x in enumerate(list_x):
            for i, x_ in enumerate(x):
                if np.amax(x_) > all_times_max[j]:
                    all_times_max[j] = np.amax(x_)
                if np.amin(x_) < all_times_min[j]:
                    all_times_min[j] = np.amin(x_)
        for j, x in enumerate(list_x):
            for i, x_ in enumerate(x):
                stand_img = np.moveaxis(x_.reshape(3, 256, 256), 0, 2)
                if normalize:
                    #stand_img = ((stand_img - all_times_min[j]) /
                    #    (all_times_max[j] - all_times_min[j]))
                    #stand_img = ((stand_img - np.amin(stand_img)) /
                    #    (np.amax(stand_img) - np.amin(stand_img)))
                    pass
                axes[j, i].imshow(
                        stand_img
                    )
                axes[j, i].set_title(list_y[j][i])
        plt.show()

    plot_several(
        [x[:15], x_test[:15]],
        [y[:15], y_test[:15]],
        normalize=False
    )

    #x = np.concatenate([x, y_], axis=1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_test_scaled = scaler.transform(x_test)

    q = 250

    # First PCA to reduce dimensionality
    pca_preproc = PCA(n_components=498, random_state=random_state)
    pca_preproc.fit(x_scaled)
    x_scaled_pca = pca_preproc.transform(x_scaled)
    x_test_scaled_pca = pca_preproc.transform(x_test_scaled)
    print("Done with first PCA")
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
    dim_constrained = 200
    covars = one_hot_encoder.fit_transform(y.reshape((-1, 1)))

    covars_test = one_hot_encoder.transform(y_test.reshape((-1, 1)))

    alpha, sig2, x_embedded_ppcca, W, muhat, M_1 = ppcca(x_scaled_pca,
            covars=covars, q=q, q_constrained=dim_constrained)
    # PROJECT test data
    N_test = x_test_scaled_pca.shape[0]
    muhat_test = jnp.mean(x_test_scaled_pca, axis=0, keepdims=1)
    x_test_embedded_ppcca = M_1 @ (
            W.T @ (x_test_scaled_pca - muhat_test).T + sig2 * jnp.concatenate([
                jnp.zeros((q - dim_constrained, N_test)),
                alpha @ jnp.concatenate([jnp.ones((1, N_test)), covars_test.T], axis=0)
                ], axis=0)
        )
    x_test_embedded_ppcca = x_test_embedded_ppcca.T


    # REC AND MOD REC ON TRAIN
    nb_sample_plot = 15
    recs = [scaler.inverse_transform(pca_preproc.inverse_transform(W @
        x_embedded_ppcca[i].T + muhat))[:,
        :3 * 65536]
            for i in range(nb_sample_plot)]

    x_embedded_ppcca_ori = copy.deepcopy(x_embedded_ppcca)
    #Modify reconstructions
    for i in range(len(x_embedded_ppcca)):
        for d_c in range(1, dim_constrained + 1):
            if y[i] == 1:
                x_embedded_ppcca = x_embedded_ppcca.at[i, -d_c].set(0)
            else:
                x_embedded_ppcca = x_embedded_ppcca.at[i, -d_c].set(0)
    recs_mod = [scaler.inverse_transform(pca_preproc.inverse_transform(W @
        x_embedded_ppcca[i].T + muhat))[:,
        :65536 * 3]
            for i in range(nb_sample_plot)]

    plot_several([x[:nb_sample_plot, :3 * 65536], recs, recs_mod],
            [y[:nb_sample_plot], y[:nb_sample_plot], y[:nb_sample_plot]])
    ####

    # REC AND MOD REC ON TEST
    recs_test = [scaler.inverse_transform(pca_preproc.inverse_transform(W @ x_test_embedded_ppcca[i].T +
        muhat_test))[:, :3 * 65536]
            for i in range(nb_sample_plot)]

    x_test_embedded_ppcca_ori = copy.deepcopy(x_test_embedded_ppcca)
    #Modify reconstructions
    for i in range(len(x_test_embedded_ppcca)):
        for d_c in range(1, dim_constrained + 1):
            if y[i] == 1:
                x_test_embedded_ppcca = x_test_embedded_ppcca.at[i, -d_c].set(0)
            else:
                x_test_embedded_ppcca = x_test_embedded_ppcca.at[i, -d_c].set(0)
    recs_mod_test = [scaler.inverse_transform(pca_preproc.inverse_transform(W @ x_test_embedded_ppcca[i].T +
        muhat_test))[:, :65536 * 3]
            for i in range(nb_sample_plot)]

    plot_several([x_test[:nb_sample_plot, :3 * 65536], recs_test, recs_mod_test],
            [y_test[:nb_sample_plot], y_test[:nb_sample_plot], y_test[:nb_sample_plot]])
    ####

    fig, axes = plt.subplots(1, 6)
    axes[0].scatter(x_embedded_pca[:, 0], x_embedded_pca[:, 1], c=y)  # , cmap="Set1")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("PCA")
    scatter = axes[1].scatter(x_embedded_ppca[:, 0], x_embedded_ppca[:, 1], c=y)  # , cmap="Set1")
    axes[1].set_title("PPCA")
    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(x_embedded_ppcca_ori[:, :-dim_constrained])
    x_embedded_ppcca_red = pca.transform(x_embedded_ppcca_ori[:, :-dim_constrained])
    x_test_embedded_ppcca_red = pca.transform(x_test_embedded_ppcca_ori[:,
        :-dim_constrained])
    scatter = axes[2].scatter(
        np.concatenate([x_embedded_ppcca_red[:, 0], x_test_embedded_ppcca_red[:, 0]]),
        np.concatenate([x_embedded_ppcca_red[:, 1], x_test_embedded_ppcca_red[:, 1]]),
        c=np.concatenate([y, y_test_legend]),
        vmin=0, vmax=nb_datasets+nb_datasets_test
    )
    axes[2].set_title("PPCCA (unconstrained dims)")
    axes[2].legend(handles=scatter.legend_elements()[0],
            labels=list(np.arange(nb_datasets + nb_datasets_test)))
    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(x_embedded_ppcca_ori[:, -dim_constrained:])
    x_embedded_ppcca_red = pca.transform(x_embedded_ppcca_ori[:, -dim_constrained:])
    x_test_embedded_ppcca_red = pca.transform(x_test_embedded_ppcca_ori[:,
        -dim_constrained:])
    scatter = axes[3].scatter(
        np.concatenate([x_embedded_ppcca_red[:, 0], x_test_embedded_ppcca_red[:, 0]]),
        np.concatenate([x_embedded_ppcca_red[:, 1], x_test_embedded_ppcca_red[:, 1]]),
        c=np.concatenate([y, y_test_legend]),
        vmin=0, vmax=nb_datasets+nb_datasets_test
    )
    axes[3].set_title("PPCCA (constrained dims)")
    axes[3].legend(handles=scatter.legend_elements()[0],
            labels=list(np.arange(nb_datasets + nb_datasets_test)))
    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(x_embedded_ppcca_ori)
    x_embedded_ppcca_red = pca.transform(x_embedded_ppcca_ori)
    x_test_embedded_ppcca_red = pca.transform(x_test_embedded_ppcca_ori)
    scatter = axes[4].scatter(
        np.concatenate([x_embedded_ppcca_red[:, 0], x_test_embedded_ppcca_red[:, 0]]),
        np.concatenate([x_embedded_ppcca_red[:, 1], x_test_embedded_ppcca_red[:, 1]]),
        c=np.concatenate([y, y_test_legend]),
        vmin=0, vmax=nb_datasets+nb_datasets_test
    )
    axes[4].set_title("PPCCA (all dims)")
    axes[4].legend(handles=scatter.legend_elements()[0],
            labels=list(np.arange(nb_datasets + nb_datasets_test)))
    fig.show()
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


def ppcca(X, covars, q, q_constrained):
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
            [jnp.ones((1, N)), covars.T], axis=0
            #[jnp.zeros((q - 100, N)), covars.T], axis=0
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
    if null_alpha:
        alpha = jnp.zeros((q_constrained, L + 1))
    else:
        scores = jnp.transpose(
            jnp.linalg.inv((W.T @ W) + (sig2 * jnp.diag(jnp.ones(q)))) @ W.T @ Xc.T
        )
        alpha = np.empty((q_constrained, L + 1))
        for i in range(1, q_constrained + 1):
            alpha[i - 1] = (
                sm.GLM(
                    np.asarray(scores[:, -i]),
                    np.asarray(
                        covars[
                            0:,
                        ]
                    ).T,
                    family=sm.families.Gaussian(),
                )
                .fit()
                .params
            )
        alpha = jnp.asarray(alpha)

    M_1 = jnp.linalg.inv(W.T @ W + sig2 * jnp.diag(jnp.ones(q)))

    def _scan_fun(carry, k):  # pylint: disable=unused-argument
        W, sig2, alpha, M_1 = carry

        # E-Step
        M_1 = jnp.linalg.inv(W.T @ W + sig2 * jnp.diag(jnp.ones(q)))

        _x_ = M_1 @ (
             #W.T @ Xc.T + sig2 * (alpha @ covars)
            # NOTE
            #W.T @ Xc.T + sig2 * covars
            W.T @ Xc.T + sig2 * jnp.concatenate([
                    jnp.zeros((q - q_constrained, N)),
                    alpha @ covars,
                    ], axis=0)
        )  # dim (q, N) or (q, 1) if we strictly follow
        # the formula but we have vectorized the computation over all the samples
        # thanks to the matrix product

        sum_xx_ = N * sig2 * M_1 + (_x_ @ _x_.T)  # dim (q, q), same remark, it has
        # been vectorized. Note the N * because it is **sum**_xx_

        # M-Step
        #NOTE
        alpha = (_x_[-q_constrained:] @ covars.T) @ jnp.linalg.inv(covars @ covars.T)  # estimation
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

        return (W, sig2, alpha, M_1), _x_

    (W, sig2, alpha, M_1), list_x_ = jax.lax.scan(
        _scan_fun, (W, sig2, alpha, M_1), jnp.arange(max_iter)
    )

    return alpha, sig2, list_x_[-1].T, W, muhat, M_1


if __name__ == "__main__":
    _main()
