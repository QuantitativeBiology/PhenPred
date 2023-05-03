import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class CLinesCVAE(nn.Module):
    def __init__(self, views, hyper, condi):
        super(CLinesCVAE, self).__init__()

        self.views = views
        self.hyper = hyper
        self.condi = condi

        print("# ---- CLinesCVAE ---- #")

        # Sizes
        self.views_sizes = {n: v.shape[1] for n, v in self.views.items()}
        self.condi_size = self.condi.shape[1]

        # Build models
        self._build_encoders()
        self._build_latents()
        self._build_contextualized_attention()
        self._build_decoders()

    def _build_encoders(self):
        self.encoders = nn.ModuleList()
        for n in self.views:
            self.encoders.append(
                nn.Sequential(
                    nn.Linear(
                        self.views_sizes[n] + self.condi_size,
                        int(self.hyper["hidden_dim_1"] * self.views_sizes[n]),
                    ),
                    nn.Dropout(p=self.hyper["probability"]),
                    self.hyper["activation_function"],
                )
            )

    def _build_latents(self):
        self.latents = nn.ModuleList()
        for n in self.views:
            self.latents.append(
                nn.Sequential(
                    nn.Linear(
                        int(self.hyper["hidden_dim_1"] * self.views_sizes[n]),
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
        for n, v in self.views.items():
            self.decoders.append(
                nn.Sequential(
                    nn.Linear(
                        self.hyper["latent_dim"] + self.condi_size,
                        int(self.hyper["hidden_dim_1"] * self.views_sizes[n]),
                    ),
                    nn.Dropout(p=self.hyper["probability"]),
                    self.hyper["activation_function"],
                    nn.Linear(
                        int(self.hyper["hidden_dim_1"] * self.views_sizes[n]),
                        self.views_sizes[n],
                    ),
                )
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, views, labels):
        encoders = []
        for i, k in enumerate(self.views):
            x = torch.cat((views[i], labels), dim=1)
            x = self.encoders[i](x)
            x = self.latents[i](x)
            encoders.append(x)
        return encoders

    def decode(self, z, labels):
        decoders = []
        for i, k in enumerate(self.views):
            x = torch.cat((z, labels), dim=1)
            x = self.decoders[i](x)
            decoders.append(x)
        return decoders

    def forward(self, views, labels):
        encoders = self.encode(views, labels)
        mu, logvar = self.context_att(encoders, labels)
        z = self.reparameterize(mu, logvar)
        decoders = self.decode(z, labels)
        return decoders, mu, logvar


class ContextualizedAttention(nn.Module):
    def __init__(self, context_dim, latent_dim):
        super(ContextualizedAttention, self).__init__()
        self.context_dim = context_dim
        self.latent_dim = latent_dim

        self.fc_mean = nn.Linear(latent_dim + context_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim + context_dim, latent_dim)

    def forward(self, views, labels):
        # Stack latents
        latents = torch.stack(views, dim=1)

        # Tile the context vector to match the batch size and number of views
        context = labels.unsqueeze(1).repeat(1, len(views), 1)

        # Concatenate the latents and context along the last dimension
        combined_input = torch.cat((latents, context), dim=-1)

        # Compute attention weights
        attention_weights = F.softmax(combined_input, dim=1)

        # # Extract attention weights for latents
        # attention_latents = attention_weights[:, :, : self.latent_dim]

        # # Element-wise multiplication with broadcasting
        # weighted_latents = attention_latents * latents

        weighted_latents = attention_weights * combined_input

        # Weighted sum of the latents using attention weights
        combined_latent = torch.sum(weighted_latents, dim=1)

        # Calculate means and log variances from the combined latent representation
        combined_mean = self.fc_mean(combined_latent)
        combined_log_var = self.fc_log_var(combined_latent)

        return combined_mean, combined_log_var
