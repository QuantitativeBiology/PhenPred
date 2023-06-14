from sklearn.covariance import log_likelihood
import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class CLinesLosses:
    @classmethod
    def loss_function(
        cls,
        hypers,
        views,
        views_hat,
        means,
        log_variances,
        z_joint,
        views_nans=None,
        covariates=None,
        labels=None,
        labels_hat=None,
        z_pre=None,
    ):
        # Compute reconstruction loss across views
        mse_loss, view_mse_loss = 0, {}
        for i, k in enumerate(hypers["datasets"]):
            X, X_ = views[i], views_hat[i]

            if views_nans is not None:
                X, X_ = X[views_nans[i]], X_[views_nans[i]]

            loss_func = cls.reconstruction_loss(hypers["reconstruction_loss"])
            v_mse_loss = loss_func(X_.cpu(), X)

            mse_loss += v_mse_loss
            view_mse_loss[k] = v_mse_loss

        # Compute KL divergence loss
        kl_loss, kl_losses = 0, {}
        for mu, log_var, n in zip(means, log_variances, hypers["datasets"]):
            k = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / len(mu)
            kl_loss += k
            kl_losses[n] = k

        kl_loss /= hypers["batch_size"]
        kl_loss *= hypers["beta"]

        # Compute batch covariate loss
        covariate_loss = 0 if covariates is None else cls.mmd_loss(z_joint, covariates)

        # Compute batch label loss
        label_loss = 0 if labels is None else F.cross_entropy(labels_hat, labels)
        label_loss *= hypers["beta"]

        # Mixture log-likehood loss
        mix_loss = 0
        if hypers["n_components"] > 1:
            prior = cls.gaussian_parameters(z_pre, dim=1)
            log_q_phi = cls.log_normal(z_joint, means, log_variances)
            log_p_theta = cls.log_normal_mixture(z_joint, prior[0], prior[1])
            mix_loss = log_q_phi - log_p_theta

        # Compute total loss
        total_loss = kl_loss + mse_loss + covariate_loss + label_loss + mix_loss

        # Return total loss, total MSE loss, and view specific MSE loss
        return dict(
            total=total_loss,
            mse=mse_loss,
            kl=kl_loss,
            covariate=covariate_loss,
            label=label_loss,
            mix=mix_loss,
            mse_views=view_mse_loss,
            kl_views=kl_losses,
        )

    @classmethod
    def gaussian_parameters(cls, h, dim=-1):
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        return m, v

    @classmethod
    def log_normal(cls, x, m, v):
        const = -0.5 * x.size(-1) * torch.log(2 * torch.tensor(np.pi))
        log_det = -0.5 * torch.sum(torch.log(v), dim=-1)
        log_exp = -0.5 * torch.sum((x - m) ** 2 / v, dim=-1)
        log_prob = const + log_det + log_exp
        return log_prob

    @classmethod
    def log_normal_mixture(cls, z, m, v):
        z = z.unsqueeze(1)
        log_probs = cls.log_normal(z, m, v)
        log_prob = cls.log_mean_exp(log_probs, 1)
        return log_prob

    @classmethod
    def log_normal(cls, x, m, v):
        const = -0.5 * x.size(-1) * torch.log(2 * torch.tensor(np.pi))
        log_det = -0.5 * torch.sum(torch.log(v), dim=-1)
        log_exp = -0.5 * torch.sum((x - m) ** 2 / v, dim=-1)
        log_prob = const + log_det + log_exp
        return log_prob

    @classmethod
    def log_mean_exp(cls, x, dim):
        return cls.log_sum_exp(x, dim) - np.log(x.size(dim))

    @classmethod
    def log_sum_exp(cls, x, dim=0):
        max_x = torch.max(x, dim)[0]
        new_x = x - max_x.unsqueeze(dim).expand_as(x)
        return max_x + (new_x.exp().sum(dim)).log()

    @classmethod
    def duplicate(x, rep):
        return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])

    @classmethod
    def class_mlp(
        cls,
        x,
        y,
        mode="pred",
        param_grid={
            "hidden_layer_sizes": [(50,), (50, 40), (50, 40, 31)],
            "activation": ["sigmoid", "logistic", "tanh", "relu"],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
            "learning_rate": ["constant", "adaptive"],
            "solver": ["sgd", "adam"],
            "max_iter": [500, 1000, 2500, 3000],
        },
        params={
            "activation": "logistic",
            "alpha": 0.0001,
            "hidden_layer_sizes": (50,),
            "learning_rate": "adaptive",
            "max_iter": 500,
            "solver": "adam",
        },
    ):
        if mode == "grid":
            mlp = MLPClassifier()
            clf = GridSearchCV(mlp, param_grid, cv=5, verbose=10, n_jobs=-1)
        else:
            clf = MLPClassifier(**params)

        clf.fit(x, y)

    @classmethod
    def mmd_loss(cls, means, covariates):
        dist = torch.cdist(means, means, p=2)

        losses = []
        for i in range(covariates.shape[1]):
            idx_x = torch.nonzero(covariates[:, i]).flatten()
            idx_y = torch.nonzero(covariates[:, i] == 0).flatten()

            if len(idx_x) == 0 or len(idx_y) == 0:
                continue

            k_xx = cls.gaussian_kernel(dist[idx_x][:, idx_x]).mean()
            k_yy = cls.gaussian_kernel(dist[idx_y][:, idx_y]).mean()
            k_xy = cls.gaussian_kernel(dist[idx_x][:, idx_y]).mean()

            losses.append(k_xx + k_yy - 2 * k_xy)

        return torch.stack(losses).mean() if len(losses) > 0 else 0

    @classmethod
    def gaussian_kernel(cls, D, gamma=None):
        if gamma is None:
            gamma = [
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                1,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                100,
                1e3,
                1e4,
                1e5,
                1e6,
            ]

        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K / len(gamma)

    @classmethod
    def get_optimizer(cls, hyper, model):
        if hyper["optimizer_type"] == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=hyper["learning_rate"],
                weight_decay=hyper["w_decay"],
            )
        else:
            return torch.optim.RAdam(
                model.parameters(),
                lr=hyper["learning_rate"],
                weight_decay=hyper["w_decay"],
            )

    @classmethod
    def activation_function(cls, name):
        name = name.lower()

        if name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "elu":
            return nn.ELU()
        elif name == "selu":
            return nn.SELU()
        elif name == "softplus":
            return nn.Softplus()
        else:
            return nn.Identity()

    @classmethod
    def reconstruction_loss(cls, name):
        if name == "mse":
            return F.mse_loss
        elif name == "bce":
            return F.binary_cross_entropy
        elif name == "gauss":
            return nn.GaussianNLLLoss
        else:
            return F.mse_loss
