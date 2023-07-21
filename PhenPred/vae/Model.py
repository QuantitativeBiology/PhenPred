import torch
import PhenPred
import numpy as np
import torch.nn as nn
from pytorch_metric_learning import losses
from PhenPred.vae.Losses import CLinesLosses
from PhenPred.vae.Layers import BottleNeck, Gaussian
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import DoNothingReducer


class MOVE(nn.Module):
    def __init__(self, hypers, views_sizes, conditional_size, views_sizes_full=None):
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

        self._build()

        print(f"# ---- MOVE")
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
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

    def _build_encoders(self):
        self.encoders = nn.ModuleList()
        for n in self.views_sizes:
            layer_sizes = [self.views_sizes[n] + self.conditional_size] + [
                int(v * self.views_sizes[n]) for v in self.hypers["hidden_dims"]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
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
        if self.views_sizes_full is None:
            input_sizes = self.views_sizes
        else:
            input_sizes = self.views_sizes_full

        latent_views_sum = self.hypers["latent_dim"]

        self.decoders = nn.ModuleList()
        for n in self.views_sizes:
            layer_sizes = [latent_views_sum + self.conditional_size] + [
                int(v * input_sizes[n]) for v in self.hypers["hidden_dims"][::-1]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hypers["probability"]))
                layers.append(self.activation_function)

            layers.append(nn.Linear(layer_sizes[-1], input_sizes[n]))

            self.decoders.append(nn.Sequential(*layers))

    def forward(self, x, y):
        # Encoder
        zs = [self.encoders[i](torch.cat([x[i], y], dim=1)) for i in range(len(x))]

        # Joint
        mu, log_var, z = self.joint(torch.cat(zs, dim=1))

        # Decoder
        x_hat = [self.decoders[i](torch.cat([z, y], dim=1)) for i in range(len(x))]

        return dict(
            x_hat=x_hat,
            z=z,
            mu=mu,
            log_var=log_var,
        )

    def loss(self, x, x_nans, out_net, y, x_mask, view_names):
        # Reconstruction loss
        x_hat = out_net["x_hat"]
        mu = out_net["mu"]
        logvar = out_net["log_var"]

        recon_loss, recon_loss_views = 0, []
        for i in range(len(x)):
            mask = x_nans[i].int()

            # if view_names[i] == "copynumber":
            #     x_hat[i] = torch.round(x_hat[i])

            recon_xi = self.recon_criterion(
                (x_hat[i] * mask)[:, x_mask[i][0]],
                (x[i] * mask)[:, x_mask[i][0]],
                reduction="sum",
            )

            # if view_names[i] == "copynumber":
            #     recon_xi /= ((x[i] * mask)[:, x_mask[i][0]] != 0).sum()

            # else:
            recon_xi /= mask[:, x_mask[i][0]].sum()

            recon_loss_views.append(recon_xi)
            recon_loss += recon_xi

        # KL divergence loss
        kl_loss = self.hypers["w_kl"] * CLinesLosses.kl_divergence(mu, logvar)

        # Contrastive loss of joint embeddings
        loss_func = losses.ContrastiveLoss(
            distance=CosineSimilarity(),
            pos_margin=0.8,
            neg_margin=0.2,
        )

        c_loss = [loss_func(out_net["mu"], y[:, i]) for i in range(32)]
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
