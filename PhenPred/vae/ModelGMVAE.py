import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from PhenPred import vae
from PhenPred.vae.Model import MOVE
from PhenPred.vae.Losses import CLinesLosses
from PhenPred.vae.Layers import JointInference, ViewDropout


class GMVAE(MOVE):
    def __init__(
        self,
        hypers,
        views_sizes,
        conditional_size,
        views_sizes_full=None,
        only_return_mu=False,
    ) -> None:
        super().__init__(
            hypers,
            views_sizes,
            conditional_size,
            views_sizes_full=views_sizes_full,
            lazy_init=True,
        )

        self.k = self.hypers["gmvae_k"]
        self.views_logits = self.hypers["gmvae_views_logits"]
        self.hidden_size = self.hypers["gmvae_hidden_size"]
        self.only_return_mu = only_return_mu

        self.y_mu = nn.Linear(self.k, self.hypers["latent_dim"])
        self.y_var = nn.Linear(self.k, self.hypers["latent_dim"])

        self._build()

        print(f"# ---- GMVAE: k={self.k}")
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {self.total_params:,d}")
        print(self)

    def _build(self):
        self._build_encoders()

        latent_views_sum = sum(self.views_latent_sizes.values())
        self.joint_inference = JointInference(
            x_dim=latent_views_sum,
            z_dim=self.hypers["latent_dim"],
            y_dim=self.k,
            hidden_size=self.hidden_size,
        )

        # Decoders
        self._build_decoders()

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    def forward(self, x, temperature=1.0, hard=0, labels=None):
        # Encoders
        views_logits = [
            self.encoders[i](torch.cat((x[i], labels), dim=1))
            for i, _ in enumerate(self.views_sizes)
        ]

        # Joint Inference
        z_mu, z_var, z, y_logits, y_prob, y = self.joint_inference(
            torch.cat(views_logits, dim=1), temperature, hard
        )

        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # Decoders
        x_hat = [
            self.decoders[i](torch.cat((z, labels), dim=1))
            for i, _ in enumerate(self.views_sizes)
        ]

        if self.only_return_mu:
            return z_mu
        else:
            return dict(
                x_hat=x_hat,
                z=z,
                mu=z_mu,
                log_var=z_var,
                y=y,
                y_logits=y_logits,
                y_prob=y_prob,
                y_mu=y_mu,
                y_log_var=y_var,
            )

    def loss(self, x, x_nans, out_net, y, x_mask, view_loss_weights=None):
        view_loss_weights = view_loss_weights if view_loss_weights else [1] * len(x)

        # MOVE loss
        vae_loss = super().loss(
            x, x_nans, out_net, y, x_mask, self.hypers["view_loss_weights"]
        )

        # GMVAE losss
        loss_gauss = (
            CLinesLosses.gaussian_loss(
                out_net["z"],
                out_net["mu"],
                out_net["log_var"],
                out_net["y_mu"],
                out_net["y_log_var"],
            )
            * self.hypers["w_gauss"]
        )

        loss_cat = -CLinesLosses.entropy(
            out_net["y_logits"], out_net["y_prob"]
        ) - np.log(1 / self.k)
        loss_cat *= self.hypers["w_cat"]

        _, predicted_labels = torch.max(out_net["y_logits"], dim=1)

        loss_total = (
            vae_loss["reconstruction"] + loss_gauss + loss_cat + vae_loss["contrastive"]
        )

        return dict(
            total=loss_total,
            reconstruction=vae_loss["reconstruction"],
            reconstruction_views=vae_loss["reconstruction_views"],
            gaussian=loss_gauss,
            categorical=loss_cat,
            contrastive=vae_loss["contrastive"],
            predicted_labels=predicted_labels,
        )
