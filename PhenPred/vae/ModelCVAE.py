import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class CLinesCVAE(nn.Module):
    def __init__(self, views, hyper, conditional=None):
        super(CLinesCVAE, self).__init__()

        self.views = views
        self.hyper = hyper
        self.conditional = conditional
        self.conditional_size = 0 if conditional is None else conditional.shape[1]

        print("# ---- CLinesCVAE ---- #")
        self.views_sizes = {n: v.shape[1] for n, v in self.views.items()}

        if self.hyper["n_groups"] is not None:
            self._build_groupbottleneck()

        self._build_encoders()
        self._build_mean_vars()

        self._build_decoders()

    def _build_groupbottleneck(self):
        self.groups = nn.ModuleList()

        for n in self.views:
            self.groups.append(
                BottleNeck(
                    self.views_sizes[n],
                    self.hyper["n_groups"],
                    self.hyper["activation_function"],
                )
            )

    def _build_encoders(self):
        self.encoders = nn.ModuleList()

        for n in self.views:
            layer_sizes = [self.views_sizes[n] + self.conditional_size]

            layer_sizes += [
                int(v * self.views_sizes[n]) for v in self.hyper["hidden_dims"]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hyper["probability"]))
                layers.append(self.hyper["activation_function"])

            self.encoders.append(nn.Sequential(*layers))

    def _build_mean_vars(self):
        self.mus, self.log_vars = nn.ModuleList(), nn.ModuleList()

        for n in self.views:
            s_i = int(self.hyper["hidden_dims"][-1] * self.views_sizes[n])
            s_o = self.hyper["latent_dim"]

            self.mus.append(nn.Sequential(nn.Linear(s_i, s_o)))
            self.log_vars.append(nn.Sequential(nn.Linear(s_i, s_o)))

    def _build_latents(self):
        self.latents = nn.ModuleList()
        for n in self.views:
            self.latents.append(
                nn.Sequential(
                    nn.Linear(
                        int(self.hyper["hidden_dims"][-1] * self.views_sizes[n]),
                        self.hyper["latent_dim"],
                    ),
                )
            )

    def _build_decoders(self):
        self.decoders = nn.ModuleList()
        for n in self.views:
            layer_sizes = [self.hyper["latent_dim"]]

            layer_sizes += [
                int(v * self.views_sizes[n]) for v in self.hyper["hidden_dims"][::-1]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hyper["probability"]))
                layers.append(self.hyper["activation_function"])

            layers.append(
                nn.Linear(layer_sizes[-1], self.views_sizes[n] + self.conditional_size)
            )

            self.decoders.append(nn.Sequential(*layers))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def encode(self, views, labels=None):
        encoders = []
        for i, _ in enumerate(self.views):
            x = views[i]

            if self.conditional_size != 0:
                x = torch.cat([x, labels], dim=1)

            if self.hyper["n_groups"] is not None:
                x = self.groups[i](x)

            x = self.encoders[i](x)
            encoders.append(x)
        return encoders

    def decode(self, z, labels=None):
        decoders = []
        for i, _ in enumerate(self.views):
            z_ = z

            if self.conditional_size != 0:
                z_ = torch.cat([z_, labels], dim=1)

            x = self.decoders[i](z_)

            decoders.append(x)
        return decoders

    def forward(self, views, labels=None):
        encoders = self.encode(views, labels)
        means, log_variances = self.mean_variance(encoders)

        if self.conditional is not None:
            zs = [
                self.reparameterize(mu, logvar)
                for mu, logvar in zip(means, log_variances)
            ]
            mu, logvar = self.context_att(zs, labels)
        else:
            mu, logvar = self.product_of_experts(means, log_variances)

        z = self.reparameterize(mu, logvar)
        decoders = self.decode(z)
        return decoders, mu, logvar, means, log_variances

    def mean_variance(self, hs):
        means, logs = [], []

        for h_bottleneck, mu, log_var in zip(hs, self.mus, self.log_vars):
            means.append(mu(h_bottleneck))
            logs.append(log_var(h_bottleneck))

        return means, logs

    def product_of_experts(self, means, logvars):
        # Convert logvar to precision (inverse variance)
        precision_list = [torch.exp(-logvar) for logvar in logvars]

        # Compute the combined precision and mu
        combined_precision = torch.sum(torch.stack(precision_list), dim=0)
        combined_mu = torch.sum(
            torch.stack(
                [precision * mu for precision, mu in zip(precision_list, means)]
            ),
            dim=0,
        )

        # Convert back to logvar
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
