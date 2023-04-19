import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from PhenPred.vae import CLinesVAEPlot as ploter
from PhenPred.vae.CLinesDrugResponseBenchmark import DrugResponseBenchmark
from PhenPred.vae.CLinesProteomicsBenchmark import ProteomicsBenchmark


class OMIC_VAE(nn.Module):
    def __init__(
        self,
        views,
        hyperparameters,
        conditional=None,
    ):
        super(OMIC_VAE, self).__init__()

        self.views = views
        self.hyper = hyperparameters
        self.conditional = conditional

        print("# ---- OMIC_VAE ---- #")

        # -- Bottlenecks
        self.omics_bottlenecks = nn.ModuleList()
        for n, v in self.views.items():
            s = self._get_size_input(v.shape[1])
            self.omics_bottlenecks.append(
                BottleNeck(
                    hidden_dim=s,
                    group=self.hyper["group"],
                    activation_function=self.hyper["activation_function"],
                )
            )
            print(f"BottleNeck {n}: {s} -> {self.hyper['group']} -> {s}")

        # -- Encoders
        self.omics_encoders = nn.ModuleList()
        for n, v in self.views.items():
            s_i = self._get_size_group(self._get_size_input(v.shape[1]))
            s_o = self._get_size_group(self.hyper["hidden_dim_1"] * v.shape[1])
            self.omics_encoders.append(
                nn.Sequential(
                    nn.Linear(s_i, s_o),
                    nn.Dropout(p=self.hyper["probability"]),
                    self.hyper["activation_function"],
                )
            )
            print(f"Encoder {n}: {s_i} -> {s_o}")

        # -- Means and Log-Vars
        self.mus, self.log_vars = nn.ModuleList(), nn.ModuleList()

        for n, v in self.views.items():
            s_i = self._get_size_group(self.hyper["hidden_dim_1"] * v.shape[1])
            s_o = self.hyper["latent_dim"]

            self.mus.append(nn.Sequential(nn.Linear(s_i, s_o)))
            self.log_vars.append(nn.Sequential(nn.Linear(s_i, s_o)))
            print(f"Means & Vars {n}: {s_i} -> {s_o}")

        # -- Decoders
        self.omics_decoders = nn.ModuleList()
        for n, v in self.views.items():
            s_i = self._get_size_input(self.hyper["latent_dim"])
            s_g = self._get_size_group(self.hyper["hidden_dim_1"] * v.shape[1])
            s_o = v.shape[1]

            self.omics_decoders.append(
                nn.Sequential(
                    nn.Linear(s_i, s_g),
                    # nn.Dropout(p=probability),
                    self.hyper["activation_function"],
                    nn.Linear(s_g, s_o),
                )
            )
            print(f"Decoder {n}: {s_i} -> {s_g} -> {s_o}")

    def _get_size_input(self, view_size):
        if self.conditional is None:
            return view_size
        else:
            return view_size + self.conditional.shape[1]

    def _get_size_group(self, view_size):
        return int(view_size // self.hyper["group"] * self.hyper["group"])

    def mean_variance(self, h_bottlenecks):
        means, logs = [], []

        for h_bottleneck, mu, log_var in zip(h_bottlenecks, self.mus, self.log_vars):
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

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        z = mean + eps * std
        return z

    def encode(self, views, c=None):
        hs = []

        for view, encoder, bottleneck in zip(
            views, self.omics_encoders, self.omics_bottlenecks
        ):
            v = view if self.conditional is None else torch.cat([view, c], dim=1)

            h = bottleneck(v)
            h = encoder(h)
            hs.append(h)

        return hs

    def decode(self, z, c=None):
        z_c = z if self.conditional is None else torch.cat([z, c], dim=1)
        return [decoder(z_c) for decoder in self.omics_decoders]

    def forward(self, views, c=None):
        hs = self.encode(views, c)

        means, log_variances = self.mean_variance(hs)

        mu_joint, logvar_joint = self.product_of_experts(means, log_variances)

        z = self.reparameterize(mu_joint, logvar_joint)

        views_hat = self.decode(z, c)

        return views_hat


class BottleNeck(nn.Module):
    def __init__(self, hidden_dim, group, activation_function):
        super(BottleNeck, self).__init__()

        self.activation_function = activation_function
        self.hidden_dim = hidden_dim
        self.group = group

        self.groups = nn.ModuleList()
        for _ in range(group):
            group_layers = nn.ModuleList()
            group_layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim // group))
            )
            group_layers.append(
                nn.Sequential(nn.Linear(hidden_dim // group, hidden_dim // group))
            )
            self.groups.append(group_layers)

    def forward(self, x):
        activation = self.activation_function

        # start with the input, which in this case will be the result of the first
        # fully connected layer
        identity = torch.narrow(x, 1, 0, self.hidden_dim // self.group * self.group)
        out = []

        for group_layers in self.groups:
            group_out = x

            for layer in group_layers:
                group_out = activation(layer(group_out))
            out.append(group_out)

        # concatenate, the size should be equal to the hidden size
        out = torch.cat(out, dim=1)

        # Why do we add here the identity
        out += identity

        return out
