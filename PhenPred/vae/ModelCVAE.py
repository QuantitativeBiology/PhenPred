import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class CLinesCVAE(nn.Module):
    def __init__(
        self, views_sizes, hyper, labels_size=None, conditional_size=None, device="cpu"
    ):
        super(CLinesCVAE, self).__init__()

        self.hyper = hyper
        self.device = device
        self.views_sizes = views_sizes
        self.labels_size = labels_size
        self.conditional_size = 0 if conditional_size is None else conditional_size

        if self.hyper["n_groups"] is not None:
            self._build_groupbottleneck()

        self._build_encoders()
        self._build_mean_vars()
        self._build_decoders()

        self._build_classifier()

        print("# ---- CLinesCVAE")
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {self.total_params:,d}")

    def _build_classifier(self):
        self.classifier = nn.Sequential(
            nn.Linear(self.hyper["latent_dim"], 50),
            nn.Sigmoid(),
            nn.Linear(50, self.labels_size),
        )

    def _build_groupbottleneck(self):
        self.groups = nn.ModuleList()

        for n in self.views_sizes:
            self.groups.append(
                BottleNeck(
                    self.views_sizes[n] + self.conditional_size,
                    self.hyper["n_groups"],
                    self.hyper["activation_function"],
                )
            )

    def _build_encoders(self):
        self.encoders = nn.ModuleList()

        for n in self.views_sizes:
            layer_sizes = [self.views_sizes[n] + self.conditional_size]

            layer_sizes += [
                int(v * self.views_sizes[n]) for v in self.hyper["hidden_dims"]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                layers.append(nn.BatchNorm1d(layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hyper["probability"]))
                layers.append(self.hyper["activation_function"])

            self.encoders.append(nn.Sequential(*layers))

    def _build_decoders(self):
        self.decoders = nn.ModuleList()
        for n in self.views_sizes:
            layer_sizes = [self.hyper["latent_dim"] + self.conditional_size]

            layer_sizes += [
                int(v * self.views_sizes[n]) for v in self.hyper["hidden_dims"][::-1]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                layers.append(nn.BatchNorm1d(layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hyper["probability"]))
                layers.append(self.hyper["activation_function"])

            layers.append(nn.Linear(layer_sizes[-1], self.views_sizes[n]))

            self.decoders.append(nn.Sequential(*layers))

    def _build_mean_vars(self):
        self.mus, self.log_vars = nn.ModuleList(), nn.ModuleList()

        for n in self.views_sizes:
            s_i = int(self.hyper["hidden_dims"][-1] * self.views_sizes[n])
            s_o = self.hyper["latent_dim"]

            self.mus.append(nn.Sequential(nn.Linear(s_i, s_o)))
            self.log_vars.append(nn.Sequential(nn.Linear(s_i, s_o)))

    def _build_latents(self):
        self.latents = nn.ModuleList()
        for n in self.views_sizes:
            self.latents.append(
                nn.Sequential(
                    nn.Linear(
                        int(self.hyper["hidden_dims"][-1] * self.views_sizes[n]),
                        self.hyper["latent_dim"],
                    ),
                )
            )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def encode(self, views, conditional=None):
        encoders = []
        for i, _ in enumerate(self.views_sizes):
            x = views[i]

            if self.conditional_size > 0:
                x = torch.cat((x, conditional), dim=1)

            if self.hyper["n_groups"] is not None:
                x = self.groups[i](x)

            x = self.encoders[i](x)
            encoders.append(x)
        return encoders

    def decode(self, z, conditional=None):
        decoders = []
        for i, _ in enumerate(self.views_sizes):
            if self.conditional_size > 0:
                z = torch.cat((z, conditional), dim=1)
            decoders.append(self.decoders[i](z))
        return decoders

    def forward(self, views, conditional=None):
        views_ = [v.to(self.device) for v in views]

        encoders = self.encode(views_, conditional)

        means, log_variances = self.mean_variance(encoders)

        mu, logvar = self.product_of_experts(means, log_variances)

        z = self.reparameterize(mu, logvar)

        decoders = self.decode(z, conditional)

        classes = self.classifier(z)

        return decoders, mu, logvar, means, log_variances, classes

    def mean_variance(self, hs):
        means, logs = [], []

        for h_bottleneck, mu, log_var in zip(hs, self.mus, self.log_vars):
            means.append(mu(h_bottleneck))
            logs.append(log_var(h_bottleneck))

        return means, logs

    def product_of_experts(self, means, logvars):
        precision_list = [torch.exp(-logvar) for logvar in logvars]

        combined_precision = torch.sum(torch.stack(precision_list), dim=0)
        combined_mu = torch.sum(
            torch.stack(
                [precision * mu for precision, mu in zip(precision_list, means)]
            ),
            dim=0,
        )

        combined_logvar = -torch.log(combined_precision)

        return combined_mu / combined_precision, combined_logvar


class BottleNeck(nn.Module):
    def __init__(self, in_features, n_groups, activation_function):
        super(BottleNeck, self).__init__()
        self.in_features = in_features
        self.n_groups = n_groups
        self.activation_function = activation_function
        self._build_bottleneck()

    def _build_bottleneck(self):
        self.groups = nn.ModuleList()
        size, rest = divmod(self.in_features, self.n_groups)

        for _ in range(self.n_groups):
            group_layer = nn.ModuleList()
            group_layer.append(nn.Sequential(nn.Linear(self.in_features, size)))
            group_layer.append(nn.Sequential(nn.Linear(size, size)))
            self.groups.append(group_layer)

        if rest != 0:
            self._build_residual_layer(rest)

    def _build_residual_layer(self, rest):
        group_layer = nn.ModuleList()
        group_layer.append(nn.Sequential(nn.Linear(self.in_features, rest)))
        group_layer.append(nn.Sequential(nn.Linear(rest, rest)))
        self.groups.append(group_layer)

    def forward(self, x):
        out = []
        for gl in self.groups:
            group_out = x
            for l in gl:
                group_out = self.activation_function(l(group_out))
            out.append(group_out)
        out = torch.cat(out, dim=1)
        out += torch.narrow(x, 1, 0, self.in_features)
        return out
