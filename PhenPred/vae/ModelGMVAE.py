import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from PhenPred.vae.Layers import BottleNeck, Gaussian, JointInference


class CLinesGMVAE(nn.Module):
    def __init__(
            self, hypers, views_sizes, k=2, views_logits=256, hidden_size=512
    ) -> None:
        super().__init__()

        self.k = k
        self.hypers = hypers
        self.views_sizes = views_sizes
        self.views_logits = views_logits
        self.hidden_size = hidden_size

        self.y_mu = nn.Linear(self.k, self.hypers["latent_dim"])
        self.y_var = nn.Linear(self.k, self.hypers["latent_dim"])

        self._build()

        print(f"# ---- CLinesGMVAE: k={self.k}")
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {self.total_params:,d}")

    def _build(self):
        # Group Bottleneck
        if self.hypers["n_groups"] is not None:
            self.groups = nn.ModuleList()
            for n in self.views_sizes:
                self.groups.append(
                    BottleNeck(
                        self.views_sizes[n],
                        self.hypers["n_groups"],
                        self.hypers["activation_function"],
                    )
                )

        # Encoders
        self.encoders = nn.ModuleList()
        for n in self.views_sizes:
            if self.views_logits > 1:
                layer_sizes = (
                        [self.views_sizes[n]]
                        + [int(v * self.views_sizes[n]) for v in self.hypers["hidden_dims"]]
                        + [self.views_logits]
                )
            else:
                layer_sizes = (
                        [self.views_sizes[n]]
                        + [int(v * self.views_sizes[n]) for v in self.hypers["hidden_dims"]]
                        + [int(self.views_logits * self.views_sizes[n])]
                )
            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                # layers.append(nn.BatchNorm1d(layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hypers["probability"]))
                layers.append(self.hypers["activation_function"])

            self.encoders.append(nn.Sequential(*layers))

        # Joint Inference
        if self.views_logits > 1:
            self.joint_inference = JointInference(
                x_dim=self.views_logits * len(self.views_sizes),
                z_dim=self.hypers["latent_dim"],
                y_dim=self.k,
                hidden_size=self.hidden_size,
            )
        else:
            self.joint_inference = JointInference(
                x_dim=sum([int(self.views_logits * self.views_sizes[n]) for n in self.views_sizes]),
                z_dim=self.hypers["latent_dim"],
                y_dim=self.k,
                hidden_size=self.hidden_size,
            )

        # p(z|y)
        # self.pzy = Gaussian(self.k, self.hypers["latent_dim"])

        # Decoders
        self.decoders = nn.ModuleList()
        for n in self.views_sizes:
            if self.views_logits > 1:
                layer_sizes = [self.hypers["latent_dim"]] + [self.views_logits * len(self.views_sizes)] + [
                    int(v * self.views_sizes[n]) for v in self.hypers["hidden_dims"][::-1]
                ]
            else:
                layer_sizes = [self.hypers["latent_dim"]] + [
                    sum([int(self.views_logits * self.views_sizes[n]) for n in self.views_sizes])] + [
                                  int(v * self.views_sizes[n]) for v in self.hypers["hidden_dims"][::-1]
                              ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                # layers.append(nn.BatchNorm1d(layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hypers["probability"]))
                layers.append(self.hypers["activation_function"])

            layers.append(nn.Linear(layer_sizes[-1], self.views_sizes[n]))
            layers.append(nn.Sigmoid())

            self.decoders.append(nn.Sequential(*layers))

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    def forward(self, views, temperature=1.0, hard=0):
        # Group Bottleneck
        if self.hypers["n_groups"] is not None:
            for i, n in enumerate(self.views_sizes):
                views[n] = self.groups[i](views[n])

        # Encoders
        views_logits = [
            self.encoders[i](views[i]) for i, _ in enumerate(self.views_sizes)
        ]

        # Joint Inference
        z_mu, z_var, z, y_logits, y_prob, y = self.joint_inference(
            torch.cat(views_logits, dim=1), temperature, hard
        )

        # p(z|y)
        # y_mu, y_var, _ = self.pzy(y)
        y_mu, y_var = self.pzy(y)

        # Decoders
        views_hat = []
        for i, n in enumerate(self.views_sizes):
            views_hat.append(self.decoders[i](z))

        return dict(
            views_hat=views_hat,
            z_mu=z_mu,
            z_var=z_var,
            z=z,
            y_logits=y_logits,
            y_prob=y_prob,
            y=y,
            y_mu=y_mu,
            y_var=y_var,
        )
