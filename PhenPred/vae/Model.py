import torch
import PhenPred
import torch.nn as nn
from PhenPred.vae.Losses import CLinesLosses
from PhenPred.vae.Layers import BottleNeck, Gaussian


class MOVE(nn.Module):
    def __init__(
        self,
        hypers,
        views_sizes,
        conditional_size,
    ):
        super().__init__()

        self.hypers = hypers
        self.views_sizes = views_sizes
        self.conditional_size = conditional_size

        self.recon_criterion = CLinesLosses.reconstruction_loss_method(
            self.hypers["reconstruction_loss"]
        )

        self._build()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print(f"# ---- MOVE")
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {self.total_params:,d}")

    def _build(self):
        self._build_encoders()

        self.joint = Gaussian(
            self.hypers["latent_dim"] * len(self.views_sizes),
            self.hypers["latent_dim"],
        )

        self._build_decoders()

    def _build_encoders(self):
        self.encoders = nn.ModuleList()
        for n in self.views_sizes:
            layer_sizes = (
                [self.views_sizes[n] + self.conditional_size]
                + [int(v * self.views_sizes[n]) for v in self.hypers["hidden_dims"]]
                + [self.hypers["latent_dim"]]
            )

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                layers.append(nn.BatchNorm1d(layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hypers["probability"]))
                layers.append(self.hypers["activation_function"])

            self.encoders.append(nn.Sequential(*layers))

    def _build_decoders(self):
        self.decoders = nn.ModuleList()
        for n in self.views_sizes:
            layer_sizes = [self.hypers["latent_dim"] + self.conditional_size] + [
                int(v * self.views_sizes[n]) for v in self.hypers["hidden_dims"][::-1]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
                layers.append(nn.BatchNorm1d(layer_sizes[i]))
                layers.append(nn.Dropout(p=self.hypers["probability"]))
                layers.append(self.hypers["activation_function"])

            layers.append(nn.Linear(layer_sizes[-1], self.views_sizes[n]))
            layers.append(nn.Sigmoid())

            self.decoders.append(nn.Sequential(*layers))

    def forward(self, x, y):
        # Encoder
        zs = [self.encoders[i](torch.cat([x[i], y], dim=1)) for i in range(len(x))]

        # Concatenate
        z = self.view_concat(torch.cat(zs, dim=1))

        # Joint
        z, mu, log_var = self.joint(z)

        # Decoder
        x_hat = [self.decoders[i](torch.cat([z, y], dim=1)) for i in range(len(x))]

        return x_hat, z, mu, log_var

    def loss(self, x, x_hat, x_nans, mu, logvar):
        # Reconstruction loss
        recon_loss, recon_loss_views = 0, []
        for i in range(len(x)):
            recon_xi = self.recon_criterion(x_hat[i][x_nans[i]], x[i][x_nans[i]])
            recon_loss_views.append(recon_xi)
            recon_loss += recon_xi

        # KL divergence loss
        kl_loss = self.hypers["w_kl"] * CLinesLosses.kl_divergence(mu, logvar)

        # Total loss
        loss = recon_loss + kl_loss

        return dict(
            total=loss,
            reconstruction=recon_loss,
            reconstruction_views=recon_loss_views,
            kl=kl_loss,
        )
