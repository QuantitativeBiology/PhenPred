import torch
import PhenPred
import numpy as np
import torch.nn as nn
from pytorch_metric_learning import losses
from PhenPred.vae.Losses import CLinesLosses
from pytorch_metric_learning.reducers import DoNothingReducer
from pytorch_metric_learning.distances import CosineSimilarity
from PhenPred.vae.Layers import BottleNeck, Gaussian, ViewDropout


class MOVE(nn.Module):
    def __init__(
        self,
        hypers,
        views_sizes,
        conditional_size,
        views_sizes_full=None,
        lazy_init=False,
        only_return_mu=False,
    ):
        super().__init__()

        self.hypers = hypers

        self.views_sizes = views_sizes
        self.views_sizes_full = views_sizes_full

        self.views_latent_sizes = {
            k: int(v * self.hypers["view_latent_dim"])
            for k, v in self.views_sizes.items()
        }

        self.conditional_size = conditional_size

        self.recon_criterion = self.hypers["reconstruction_loss"]
        self.activation_function = self.hypers["activation_function"]

        if not lazy_init:
            self._build()

            print(f"# ---- MOVE")
            self.total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            print(f"Total parameters: {self.total_params:,d}")
            print(self)

    def _build(self):
        self._build_encoders()

        latent_views_sum = sum(self.views_latent_sizes.values())
        self.joint = Gaussian(
            latent_views_sum,
            self.hypers["latent_dim"],
        )

        self._build_decoders()

    def _build_encoders(self, suffix_layers=None):
        self.encoders = nn.ModuleList()
        for n in self.views_sizes:
            layers = nn.ModuleList()

            layer_sizes = [self.views_sizes[n] + self.conditional_size] + [
                int(v * self.views_sizes[n]) for v in self.hypers["hidden_dims"]
            ]

            if suffix_layers is not None:
                layer_sizes += suffix_layers

            layers.append(nn.Dropout(p=self.hypers["feature_dropout"]))
            if self.hypers["view_dropout"] > 0:
                layers.append(ViewDropout(p=self.hypers["view_dropout"]))

            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

                if self.hypers["batch_norm"]:
                    layers.append(nn.BatchNorm1d(layer_sizes[i]))

                layers.append(nn.Dropout(p=self.hypers["probability"]))

                layers.append(self.activation_function)

            layers.append(
                nn.Linear(
                    layer_sizes[-1],
                    self.views_latent_sizes[n],
                )
            )
            layers.append(self.activation_function)

            self.encoders.append(nn.Sequential(*layers))

    def _build_decoders(self):
        input_sizes = (
            self.views_sizes_full if self.views_sizes_full else self.views_sizes
        )

        self.decoders = nn.ModuleList()
        for n in self.views_sizes:
            layer_sizes = [self.hypers["latent_dim"] + self.conditional_size] + [
                int(v * input_sizes[n]) for v in self.hypers["hidden_dims"][::-1]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

                if self.hypers["batch_norm"]:
                    layers.append(nn.BatchNorm1d(layer_sizes[i]))

                layers.append(nn.Dropout(p=self.hypers["probability"]))

                layers.append(self.activation_function)

            layers.append(nn.Linear(layer_sizes[-1], input_sizes[n]))

            self.decoders.append(nn.Sequential(*layers))

    def forward(self, x_all):
        # Encoder
        if self.hypers["use_conditionals"]:
            x = x_all[:-1]
            y = x_all[-1]
            zs = [self.encoders[i](torch.cat([x[i], y], dim=1)) for i in range(len(x))]
        else:
            x = x_all
            zs = [self.encoders[i](x[i]) for i in range(len(x))]

        # Joint
        mu, log_var, z = self.joint(torch.cat(zs, dim=1))

        # Decoder
        if self.hypers["use_conditionals"]:
            x_hat = [self.decoders[i](torch.cat([z, y], dim=1)) for i in range(len(x))]
        else:
            x_hat = [self.decoders[i](z) for i in range(len(x))]
        if self.only_return_mu:
            return mu
        else:
            return dict(
                x_hat=x_hat,
                z=z,
                mu=mu,
                log_var=log_var,
            )

    def loss(self, x, x_nans, out_net, y, x_mask, view_loss_weights=None):
        view_loss_weights = view_loss_weights if view_loss_weights else [1] * len(x)

        # Reconstruction loss
        x_hat, mu, logvar = out_net["x_hat"], out_net["mu"], out_net["log_var"]

        recon_loss, recon_loss_views = 0, []
        for i in range(len(x)):
            mask = x_nans[i].int()
            x_hat_i, x_i = (x_hat[i] * mask), (x[i] * mask)

            # TODO: Imbalance in CNV (0>>n), calculate MSE per class and then mean
            if i == 6:
                recon_xi = torch.stack(
                    [
                        self.recon_criterion(
                            x_hat_i[x_i == v],
                            x_i[x_i == v],
                            reduction="sum",
                        )
                        / c
                        for v, c in zip(*torch.unique(x_i, return_counts=True))
                    ]
                ).mean()

            else:
                recon_xi = (
                    self.recon_criterion(
                        x_hat_i,
                        x_i,
                        reduction="sum",
                    )
                    / mask.sum()
                )

            recon_xi *= view_loss_weights[i]

            recon_loss_views.append(recon_xi)
            recon_loss += recon_xi

        recon_loss *= self.hypers["w_rec"]

        # KL divergence loss
        kl_loss = CLinesLosses.kl_divergence(mu, logvar) * self.hypers["w_kl"]

        # Contrastive loss of joint embeddings
        loss_func = losses.ContrastiveLoss(
            distance=CosineSimilarity(),
            pos_margin=self.hypers["contrastive_pos_margin"],
            neg_margin=self.hypers["contrastive_neg_margin"],
        )

        c_loss = [loss_func(mu, y[:, i]) for i in range(32)]
        c_loss = torch.stack(c_loss).sum() * self.hypers["w_contrastive"]

        # Total loss
        loss = recon_loss + kl_loss + c_loss

        return dict(
            total=loss,
            reconstruction=recon_loss,
            reconstruction_views=recon_loss_views,
            kl=kl_loss,
            contrastive=c_loss,
        )
