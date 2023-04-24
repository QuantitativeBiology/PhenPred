import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class CLinesCVAE(nn.Module):
    def __init__(self, views, hyper, condi=None):
        super(CLinesCVAE, self).__init__()

        self.views = views
        self.hyper = hyper
        self.condi = condi

        print("# ---- CLinesCVAE ---- #")

        self.views_sizes = {n: v.shape[1] for n, v in self.views.items()}

        if self.condi is not None:
            self.condi_size = self.condi.shape[1]

        self._build_groupbottleneck()
        self._build_encoders()
        self._build_mean_vars()

        if self.condi is not None:
            self._build_contextualized_attention()

        self._build_decoders()

    def _build_groupbottleneck(self):
        self.groups = nn.ModuleList()

        for n in self.views:
            self.groups.append(
                BottleNeck(
                    self.views_sizes[n],
                    self.hyper["group_size"],
                    self.hyper["activation_function"],
                )
            )

    def _build_encoders(self):
        self.encoders = nn.ModuleList()

        for n in self.views:
            layer_sizes = [self.views_sizes[n]]
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

    def _build_contextualized_attention(self):
        self.context_att = ContextualizedAttention(
            context_dim=self.condi_size,
            latent_dim=self.hyper["latent_dim"],
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
            layers.append(nn.Linear(layer_sizes[-1], self.views_sizes[n]))

            self.decoders.append(nn.Sequential(*layers))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, views, labels=None):
        encoders = []
        for i, k in enumerate(self.views):
            # x = torch.cat((views[i], labels), dim=1)
            x = self.groups[i](views[i])
            x = self.encoders[i](x)
            encoders.append(x)
        return encoders

    def decode(self, z, labels=None):
        decoders = []
        for i, k in enumerate(self.views):
            # if self.condi is not None:
            #     x = self.decoders[i](torch.cat((z, labels), dim=1))
            # else:
            x = self.decoders[i](z)

            decoders.append(x)
        return decoders

    def forward(self, views, labels):
        encoders = self.encode(views, labels)
        means, log_variances = self.mean_variance(encoders)

        if self.condi is not None:
            zs = [
                self.reparameterize(mu, logvar)
                for mu, logvar in zip(means, log_variances)
            ]
            mu, logvar = self.context_att(zs, labels)
        else:
            mu, logvar = self.product_of_experts(means, log_variances)

        z = self.reparameterize(mu, logvar)
        decoders = self.decode(z, labels)
        return decoders, mu, logvar

    def mean_variance(self, hs):
        means, logs = [], []

        for h_bottleneck, mu, log_var in zip(hs, self.mus, self.log_vars):
            means.append(mu(h_bottleneck))
            logs.append(log_var(h_bottleneck))

        return means, logs

    def product_of_experts(self, means, logs):
        logvar_joint = torch.sum(
            torch.stack([1.0 / torch.exp(log_var) for log_var in logs]),
            dim=0,
        )
        logvar_joint = torch.log(1.0 / logvar_joint)

        mu_joint = torch.sum(
            torch.stack([mu / torch.exp(log_var) for mu, log_var in zip(means, logs)]),
            dim=0,
        )

        mu_joint *= torch.exp(logvar_joint)

        return mu_joint, logvar_joint


class ContextualizedAttention(nn.Module):
    def __init__(self, context_dim, latent_dim):
        super(ContextualizedAttention, self).__init__()
        self.context_dim = context_dim
        self.latent_dim = latent_dim

        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, views, labels):
        # Stack latents
        latents = torch.stack(views, dim=1)

        # Tile the context vector to match the batch size and number of views
        context = labels.unsqueeze(1).repeat(1, len(views), 1)

        # Concatenate the latents and context along the last dimension
        combined_input = torch.cat((latents, context), dim=-1)

        # Compute attention weights
        attention_weights = F.softmax(combined_input, dim=1)

        # Extract attention weights for latents
        attention_latents = attention_weights[:, :, : self.latent_dim]

        # Element-wise multiplication with broadcasting
        weighted_latents = attention_latents * latents

        # Weighted sum of the latents using attention weights
        combined_latent = torch.sum(weighted_latents, dim=1)

        # Calculate means and log variances from the combined latent representation
        combined_mean = self.fc_mean(combined_latent)
        combined_log_var = self.fc_log_var(combined_latent)

        return combined_mean, combined_log_var


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

        # Build groups
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
