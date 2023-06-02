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

        if self.hyper["n_groups"] is not None:
            self._build_groupbottleneck()

        self._build_encoders()
        self._build_mean_vars()

        if self.condi is not None:
            self._build_contextualized_attention()

        if self.hyper["mlp_join"] or self.hyper["common_unique_join"]:
            self._build_mlp()

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
            layer_sizes = [self.views_sizes[n]]
            layer_sizes += [
                int(v * self.views_sizes[n]) for v in self.hyper["hidden_dims"]
            ]

            layers = nn.ModuleList()
            if self.hyper['view_dropout'] > 0:
                layers.append(ViewDropout(p=self.hyper['view_dropout']))
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hyper["probability"]))
                layers.append(self.hyper["activation_function"])
                if self.hyper['skip_connections']:
                    layers.append(BasicBlock(layer_sizes[i]))

            self.encoders.append(nn.Sequential(*layers))

    def _build_mean_vars(self):
        self.mus, self.log_vars = nn.ModuleList(), nn.ModuleList()

        for n in self.views:
            s_i = int(self.hyper["hidden_dims"][-1] * self.views_sizes[n])
            s_o = self.hyper["latent_dim"] if self.hyper["latent_dim"] > 1 else int(
                self.hyper["latent_dim"] * self.views_sizes[n]
            )

            self.mus.append(nn.Sequential(nn.Linear(s_i, s_o)))
            self.log_vars.append(nn.Sequential(nn.Linear(s_i, s_o)))

    def _build_latents(self):
        self.latents = nn.ModuleList()
        for n in self.views:
            self.latents.append(
                nn.Sequential(
                    nn.Linear(
                        int(self.hyper["hidden_dims"][-1] * self.views_sizes[n]),
                        self.hyper["latent_dim"] if self.hyper["latent_dim"] > 1 else int(
                            self.hyper["latent_dim"] * self.views_sizes[n]),
                    ),
                )
            )

    def _build_contextualized_attention(self):
        self.context_att = ContextualizedAttention(
            context_dim=self.condi_size,
            latent_dim=self.hyper["latent_dim"],
        )

    def _build_mlp(self):
        self.mlp_mu = MLP(self.hyper["latent_dim"] * len(self.views), self.hyper["latent_dim"])
        self.mlp_var = MLP(self.hyper["latent_dim"] * len(self.views), self.hyper["latent_dim"])

    def _build_decoders(self):
        self.decoders = nn.ModuleList()
        for n in self.views:
            if self.hyper["concat_join"]:
                if self.hyper["latent_dim"] > 1:
                    layer_sizes = [self.hyper["latent_dim"] * len(self.views)]
                else:
                    layer_sizes = [sum([int(self.hyper["latent_dim"] * x) for x in self.views_sizes.values()])]
            elif self.hyper["common_unique_join"]:
                layer_sizes = [self.hyper["latent_dim"] * 2]
            else:
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

    def reparameterize_2(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def encode(self, views):
        encoders = []
        for i, _ in enumerate(self.views):
            x = views[i]

            if self.hyper["n_groups"] is not None:
                x = self.groups[i](x)

            x = self.encoders[i](x)
            encoders.append(x)
        return encoders

    def decode(self, z):
        decoders = []
        for i, _ in enumerate(self.views):
            x = self.decoders[i](z)
            decoders.append(x)
        return decoders

    def forward(self, views, labels=None):
        encoders = self.encode(views)
        means, log_variances = self.mean_variance(encoders)  # shape: (n_views, batch_size, latent_dim)

        if self.condi is not None:
            zs = [
                self.reparameterize(mu, logvar)
                for mu, logvar in zip(means, log_variances)
            ]
            mu, logvar = self.context_att(zs, labels)
        elif self.hyper["mlp_join"]:
            mu, logvar = self.mlp_combine(means, log_variances)
        elif self.hyper["concat_join"]:
            mu = torch.cat(means, dim=1)
            logvar = torch.cat(log_variances, dim=1)
        elif self.hyper["common_unique_join"]:
            mu, logvar = self.product_of_experts_2(means, log_variances)
            z_common = self.reparameterize_2(mu, logvar)
            for mu_unique, logvar_unique in zip(means, log_variances):
                z_unique = self.reparameterize_2(mu_unique, logvar_unique)
                z = torch.cat([z_unique, z_common], dim=1)
        else:
            mu, logvar = self.product_of_experts_2(means, log_variances)

        if not self.hyper["common_unique_join"]:
            z = self.reparameterize_2(mu, logvar)

        decoders = self.decode(z)
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

    def mlp_combine(self, means, logs):
        mu_joint = torch.cat(means, dim=1)
        logvar_joint = torch.cat(logs, dim=1)
        mu = self.mlp_mu(mu_joint)
        log_var = self.mlp_var(logvar_joint)
        return mu, log_var

    def product_of_experts_2(self, mu_list, logvar_list):
        # Convert logvar to precision (inverse variance)
        precision_list = [torch.exp(-logvar) for logvar in logvar_list]

        # Compute the combined precision and mu
        combined_precision = torch.sum(torch.stack(precision_list), dim=0)
        combined_mu = torch.sum(torch.stack([precision * mu for precision, mu in zip(precision_list, mu_list)]), dim=0)

        # Convert back to logvar
        combined_logvar = -torch.log(combined_precision)

        return combined_mu / combined_precision, combined_logvar


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


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

        # # Tile the context vector to match the batch size and number of views
        # context = labels.unsqueeze(1).repeat(1, len(views), 1)

        # # Concatenate the latents and context along the last dimension
        # combined_input = torch.cat((latents, context), dim=-1)

        # # Compute attention weights
        # attention_weights = F.softmax(combined_input, dim=1)

        # # Extract attention weights for latents
        # attention_latents = attention_weights[:, :, : self.latent_dim]

        # # Element-wise multiplication with broadcasting
        # weighted_latents = attention_latents * latents

        # weighted_latents = attention_weights * combined_input

        attention_weights = F.softmax(latents, dim=1)

        weighted_latents = attention_weights * latents

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


class ViewDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            if np.random.binomial(1, self.p):
                x.zero_()
        return x


class BasicBlock(nn.Module):
    def __init__(self, hidden_width):
        super(BasicBlock, self).__init__()
        self.hidden = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.hidden1 = nn.Sequential(nn.Linear(hidden_width, hidden_width))
        self.hidden2 = nn.Sequential(nn.Linear(hidden_width, hidden_width))

    def forward(self, x):
        identity = x
        out = self.hidden1(x)
        out = self.relu(out)
        out = self.hidden2(out)
        out += identity
        out = self.relu(out)

        return out
