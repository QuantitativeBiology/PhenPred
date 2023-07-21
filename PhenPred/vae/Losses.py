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
    def unlabeled_loss(
        cls,
        views,
        out_net,
        views_mask=None,
        rec_type="mse",
        w_rec=1,
        w_gauss=0.01,
        w_cat=0.001,
        num_cat=None,
        view_loss_weights=None,
    ):
        """
        Sourced from: https://github.com/jariasf/GMVAE/tree/master/pytorch

        Method defining the loss functions derived from the variational lower bound
        Args:
            data: (array) corresponding array containing the input data
            out_net: (dict) contains the graph operations or nodes of the network output

        Returns:
            loss_dic: (dict) contains the values of each loss function and predictions
        """
        # obtain network variables
        z, x_hat = out_net["z"], out_net["x_hat"]
        logits, prob_cat = out_net["y_logits"], out_net["y_prob"]
        y_mu, y_var = out_net["y_mu"], out_net["y_var"]
        mu, var = out_net["z_mu"], out_net["z_var"]

        # reconstruction loss
        loss_rec = 0
        recon_loss_views = []
        for i, k in enumerate(views):
            real, predicted = k, x_hat[i]

            if views_mask is not None:
                real, predicted = real[views_mask[i]], predicted[views_mask[i]]

            if type(rec_type) == str:
                recon_xi = cls.reconstruction_loss(real, predicted, rec_type)
            else:
                recon_xi = rec_type(real, predicted)
            loss_rec += recon_xi * view_loss_weights[i]
            recon_loss_views.append(recon_xi)

        # gaussian loss
        loss_gauss = cls.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -cls.entropy(logits, prob_cat) - np.log(1 / num_cat)

        # total loss
        loss_total = w_rec * loss_rec + w_gauss * loss_gauss + w_cat * loss_cat

        # obtain predictions
        _, predicted_labels = torch.max(logits, dim=1)

        loss_dic = dict(
            total=loss_total,
            reconstruction=loss_rec,
            reconstruction_views=recon_loss_views,
            gaussian=loss_gauss,
            categorical=loss_cat,
            predicted_labels=predicted_labels,
        )

        return loss_dic

    @classmethod
    def reconstruction_loss(cls, real, predicted, rec_type="mse"):
        """Reconstruction loss between the true and predicted outputs
           mse = (1/n)*Σ(real - predicted)^2
           bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        if rec_type == "mse":
            loss = F.mse_loss(predicted, real, reduction="mean")
        elif rec_type == "bce":
            loss = F.binary_cross_entropy(predicted, real, reduction="mean")
        else:
            raise "invalid loss function... try bce or mse..."
        return loss.sum(-1).mean()

    @classmethod
    def gaussian_loss(cls, z, z_mu, z_var, z_mu_prior, z_var_prior):
        """Variational loss when using labeled data without considering reconstruction loss
           loss = log q(z|x,y) - log p(z) - log p(y)

        Args:
           z: (array) array containing the gaussian latent variable
           z_mu: (array) array containing the mean of the inference model
           z_var: (array) array containing the variance of the inference model
           z_mu_prior: (array) array containing the prior mean of the generative model
           z_var_prior: (array) array containing the prior variance of the generative mode

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        loss = cls.log_normal(z, z_mu, z_var) - cls.log_normal(
            z, z_mu_prior, z_var_prior
        )
        return loss.mean()

    @staticmethod
    def log_normal(x, mu, var, eps=1e-8):
        """Logarithm of normal distribution with mean=mu and variance=var
           log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
           x: (array) corresponding array containing the input
           mu: (array) corresponding array containing the mean
           var: (array) corresponding array containing the variance

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        if eps > 0.0:
            var = var + eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1
        )

    @classmethod
    def entropy(cls, logits, targets):
        """Entropy loss
            loss = (1/n) * -Σ targets*log(predicted)

        Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

    @classmethod
    def kl_divergence(cls, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / len(mu)

    @classmethod
    def loss_function(
        cls,
        hypers,
        views,
        x_hat,
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
            X, X_ = views[i], x_hat[i]

            if views_nans is not None:
                X, X_ = X[views_nans[i]], X_[views_nans[i]]

            loss_func = cls.reconstruction_loss_method(hypers["reconstruction_loss"])
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
            raise "Use ModelGMVAE.py"

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
    def get_optimizer(cls, model, args):
        if args["optimizer_type"] == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=args["learning_rate"],
                weight_decay=args["w_decay"],
            )
        elif args["optimizer_type"] == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=args["learning_rate"],
                weight_decay=args["w_decay"],
            )
        else:
            return torch.optim.RAdam(
                model.parameters(),
                lr=args["learning_rate"],
                weight_decay=args["w_decay"],
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
        elif name == "prelu":
            return nn.PReLU()
        else:
            return nn.Identity()

    @classmethod
    def reconstruction_loss_method(cls, name):
        if name == "mse":
            return F.mse_loss
        elif name == "bce":
            return F.binary_cross_entropy
        elif name == "ce":
            return F.cross_entropy
        elif name == "gauss":
            return nn.GaussianNLLLoss
        else:
            return F.mse_loss

    @classmethod
    def get_scheduler(cls, optimizer, args):
        name = args["scheduler"].lower()

        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                threshold=args["scheduler_threshold"],
                factor=args["scheduler_factor"],
                patience=args["scheduler_patience"],
                min_lr=args["scheduler_min_lr"],
                verbose=True,
            )
        else:
            return None
