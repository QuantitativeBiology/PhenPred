import math
import torch
import PhenPred
import numpy as np
import torch.nn as nn
from pytorch_metric_learning import losses
from PhenPred.vae.Losses import CLinesLosses
from pytorch_metric_learning.reducers import DoNothingReducer
from pytorch_metric_learning.distances import CosineSimilarity
from PhenPred.vae.Layers import BottleNeck, Gaussian, ViewDropout
import torch.nn.functional as F


class MOSA(nn.Module):
    def __init__(
        self,
        hypers,
        views_sizes,
        conditional_size,
        views_sizes_full=None,
        lazy_init=False,
        return_for_shap=None,
    ):
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
        self.return_for_shap = return_for_shap
        self.layer_idx_map = {
            key: index for index, key in enumerate(list(self.hypers["datasets"].keys()))
        }

        if not lazy_init:
            self._build()

            print(f"# ---- MOSA")
            self.total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
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

    def _build_encoders(self, suffix_layers=None):
        self.encoders = nn.ModuleList()
        for n in self.views_sizes:
            layers = nn.ModuleList()

            layer_sizes = [self.views_sizes[n] + self.conditional_size] + [
                int(v * self.views_sizes[n]) for v in self.hypers["hidden_dims"]
            ]

            if suffix_layers is not None:
                layer_sizes += suffix_layers

            if self.hypers["feature_dropout"] > 0:
                layers.append(nn.Dropout(p=self.hypers["feature_dropout"]))

            if self.hypers["view_dropout"] > 0:
                layers.append(ViewDropout(p=self.hypers["view_dropout"]))

            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

                if self.hypers["batch_norm"]:
                    layers.append(nn.BatchNorm1d(layer_sizes[i]))

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
        input_sizes = (
            self.views_sizes_full if self.views_sizes_full else self.views_sizes
        )

        self.decoders = nn.ModuleList()
        for n in self.views_sizes:
            layer_sizes = [self.hypers["latent_dim"] + self.conditional_size] + [
                int(v * input_sizes[n]) for v in self.hypers["hidden_dims"][::-1]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

                if self.hypers["batch_norm"]:
                    layers.append(nn.BatchNorm1d(layer_sizes[i]))

                layers.append(nn.Dropout(p=self.hypers["probability"]))

                layers.append(self.activation_function)

            layers.append(nn.Linear(layer_sizes[-1], input_sizes[n]))

            self.decoders.append(nn.Sequential(*layers))

    def forward(self, x_all):
        # Encoder
        if self.hypers["use_conditionals"]:
            x = x_all[:-1]
            y = x_all[-1]
            zs = [self.encoders[i](torch.cat([x[i], y], dim=1)) for i in range(len(x))]

        else:
            x = x_all
            zs = [self.encoders[i](x[i]) for i in range(len(x))]

        # Joint
        mu, log_var, z = self.joint(torch.cat(zs, dim=1))

        # Decoder
        if self.hypers["use_conditionals"]:
            x_hat = [self.decoders[i](torch.cat([z, y], dim=1)) for i in range(len(x))]

        else:
            x_hat = [self.decoders[i](z) for i in range(len(x))]

        if self.return_for_shap == "latent":
            return mu
        elif self.return_for_shap is not None:
            return x_hat[self.layer_idx_map[self.return_for_shap]]
        else:
            return dict(
                x_hat=x_hat,
                z=z,
                mu=mu,
                log_var=log_var,
            )

    def loss(self, x, x_nans, out_net, y, x_mask, view_loss_weights=None):
        view_loss_weights = view_loss_weights if view_loss_weights else [1] * len(x)

        # Reconstruction loss
        x_hat, mu, logvar = out_net["x_hat"], out_net["mu"], out_net["log_var"]

        recon_loss, recon_loss_views = 0, []
        for i in range(len(x)):
            mask = x_nans[i].int()
            x_hat_i, x_i = (x_hat[i] * mask), (x[i] * mask)

            if self.hypers["view_loss_recon_type"][i].lower() == "macro":
                recon_xi = torch.stack(
                    [
                        self.recon_criterion(
                            x_hat_i[x_i == v],
                            x_i[x_i == v],
                            reduction="sum",
                        )
                        / c
                        for v, c in zip(*torch.unique(x_i, return_counts=True))
                    ]
                ).mean()

            elif self.hypers["view_loss_recon_type"][i].lower() == "autofocus":
                gamma = 2
                mask_sum = mask.sum()

                if mask_sum == 0:
                    recon_xi = 0
                else:
                    recon_xi = torch.sum((x_hat_i - x_i) ** (2 + gamma))
                    recon_xi /= mask_sum
            else:
                mask_sum = mask.sum()

                if mask_sum == 0:
                    recon_xi = 0

                else:
                    recon_xi = self.recon_criterion(
                        x_hat_i,
                        x_i,
                        reduction="sum",
                    )
                    recon_xi /= mask_sum

            recon_xi *= view_loss_weights[i]

            recon_loss_views.append(recon_xi)
            recon_loss += recon_xi

        recon_loss *= self.hypers["w_rec"]

        # KL divergence loss
        kl_loss = CLinesLosses.kl_divergence(mu, logvar) * self.hypers["w_kl"]

        # Contrastive loss of joint embeddings
        # TODO: Make this not position dependent; needs to use the labels_name from DatasetDepMap23Q2
        if len(self.hypers["labels"]) == 0:
            c_loss = 0

        else:
            loss_func = losses.ContrastiveLoss(
                distance=CosineSimilarity(),
                pos_margin=self.hypers["contrastive_pos_margin"],
                neg_margin=self.hypers["contrastive_neg_margin"],
            )

            c_loss = [loss_func(mu, y[:, i]) for i in range(32)]
            c_loss = torch.stack(c_loss).sum() * self.hypers["w_contrastive"]

        if self.hypers.get("dip_vae_type") != None:
            dipvae_loss = CLinesLosses.dip_vae(
                mu,
                logvar,
                self.hypers["lambda_od"],
                self.hypers["lambda_d"],
                self.hypers["dip_vae_type"],
            )

            # Total loss
            loss = recon_loss + kl_loss + c_loss + dipvae_loss

            return dict(
                total=loss,
                reconstruction=recon_loss,
                reconstruction_views=recon_loss_views,
                kl=kl_loss,
                contrastive=c_loss,
                dipvae=dipvae_loss,
            )

        else:
            # Total loss
            loss = recon_loss + kl_loss + c_loss

            return dict(
                total=loss,
                reconstruction=recon_loss,
                reconstruction_views=recon_loss_views,
                kl=kl_loss,
                contrastive=c_loss,
            )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerMOSA(nn.Module):
    def __init__(
        self,
        hypers,
        views_sizes,
        conditional_size,
        views_sizes_full=None,
        lazy_init=False,
        return_for_shap=None,
    ):
        super().__init__()

        self.hypers = hypers
        self.views_sizes = views_sizes
        self.views_sizes_full = views_sizes_full
        self.conditional_size = conditional_size
        self.return_for_shap = return_for_shap

        # Store activation function like MOSA does
        self.activation_function = self.hypers["activation_function"]

        # Transformer specific parameters
        self.d_model = hypers.get(
            "transformer_d_model", 256
        )  # Hidden dimension of transformer embeddings
        self.nhead = hypers.get(
            "transformer_nhead", 8
        )  # Number of attention heads in multi-head attention
        self.num_encoder_layers = hypers.get(
            "transformer_encoder_layers", 2
        )  # Number of transformer encoder layers for cross-modal learning
        self.dim_feedforward = hypers.get(
            "transformer_dim_feedforward", 1024
        )  # Hidden dimension of feedforward network in transformer
        self.dropout = hypers.get(
            "probability", 0.1
        )  # Dropout probability for regularization

        # Number of learnable query vectors for pooling
        self.num_global_queries = hypers.get("transformer_num_global_queries", 4)

        # Store reconstruction loss criterion
        self.recon_criterion = self.hypers["reconstruction_loss"]

        self.layer_idx_map = {
            key: index for index, key in enumerate(list(self.hypers["datasets"].keys()))
        }

        if not lazy_init:
            self._build()

            print(f"# ---- Enhanced TransformerMOSA")
            self.total_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            print(f"Total parameters: {self.total_params:,d}")
            print(self)

    def _build(self):
        self._build_view_embedders()
        self._build_transformer()
        self._build_latent_head()
        self._build_decoders()

    def _build_view_embedders(self):
        """Convert each view's features into transformer embeddings using learned projections"""
        self.view_embedders = nn.ModuleDict()
        self.view_tokens_per_view = {}  # Number of tokens per view

        # Calculate view-specific parameters based on view sizes
        for view_name, view_size in self.views_sizes.items():
            input_size = view_size + self.conditional_size

            # Adaptive token allocation strategy based on view size and importance
            if input_size <= 500:  # Small views (metabolomics, proteomics)
                tokens_per_view = 4  # Few tokens for small views
            elif input_size <= 2000:  # Medium views (drug response)
                tokens_per_view = 8
            elif input_size <= 6000:  # Large views
                tokens_per_view = 16
            else:  # Very large views (transcriptomics, CRISPR, methylation)
                tokens_per_view = 32

            self.view_tokens_per_view[view_name] = tokens_per_view

            print(
                f"View {view_name}: {input_size} features → {tokens_per_view} learned tokens"
            )

            # Build view embedder layers like MOSA encoder
            layers = []

            # Add feature dropout if specified
            if self.hypers["feature_dropout"] > 0:
                layers.append(nn.Dropout(p=self.hypers["feature_dropout"]))

            # Add ViewDropout like MOSA encoder
            if self.hypers["view_dropout"] > 0:
                layers.append(ViewDropout(p=self.hypers["view_dropout"]))

            # Add the main projection layer
            layers.extend(
                [
                    nn.Linear(input_size, self.d_model * tokens_per_view),
                    nn.LayerNorm(self.d_model * tokens_per_view),
                    nn.Dropout(self.dropout),
                ]
            )

            self.view_embedders[view_name] = nn.Sequential(*layers)

        # View type embeddings to distinguish between different modalities
        self.view_type_embeddings = nn.Embedding(len(self.views_sizes), self.d_model)

        # Positional encoding with maximum possible sequence length
        max_total_tokens = sum(self.view_tokens_per_view.values())
        self.pos_encoder = PositionalEncoding(self.d_model, max_total_tokens)

    def _build_transformer(self):
        """Build the main transformer architecture"""
        # Cross-modal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_encoder_layers
        )

    def _build_latent_head(self):
        """Build the latent space mapping with multiple learned queries"""
        # Multiple learned queries for global pooling
        self.global_queries = nn.Parameter(
            torch.randn(self.num_global_queries, self.d_model) * 0.02
        )

        # Multi-head attention for pooling with learned queries
        self.global_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True,
        )

        # Aggregation method for multiple query outputs
        self.query_aggregator = nn.Sequential(
            nn.Linear(self.num_global_queries * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        # Latent space projection
        self.latent_dim = self.hypers["latent_dim"]
        self.mu_head = nn.Linear(self.d_model, self.latent_dim)
        self.logvar_head = nn.Linear(self.d_model, self.latent_dim)

    def _build_decoders(self):
        """Build decoders for each view using the same structure as MOSA"""
        input_sizes = (
            self.views_sizes_full if self.views_sizes_full else self.views_sizes
        )

        self.decoders = nn.ModuleList()
        for n in self.views_sizes:
            layer_sizes = [self.hypers["latent_dim"] + self.conditional_size] + [
                int(v * input_sizes[n]) for v in self.hypers["hidden_dims"][::-1]
            ]

            layers = nn.ModuleList()
            for i in range(1, len(layer_sizes)):
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

                if self.hypers["batch_norm"]:
                    layers.append(nn.BatchNorm1d(layer_sizes[i]))

                layers.append(nn.Dropout(p=self.hypers["probability"]))

                layers.append(self.activation_function)

            layers.append(nn.Linear(layer_sizes[-1], input_sizes[n]))

            self.decoders.append(nn.Sequential(*layers))

    def _create_view_tokens(self, x, view_name):
        """Convert feature vector into learned token embeddings"""
        batch_size = x.shape[0]
        tokens_per_view = self.view_tokens_per_view[view_name]

        # Project features to token embeddings
        embedded = self.view_embedders[view_name](
            x
        )  # [batch, d_model * tokens_per_view]

        # Reshape to tokens
        embedded = embedded.view(batch_size, tokens_per_view, self.d_model)
        return embedded

    def forward(self, x_all):
        if self.hypers["use_conditionals"]:
            x = x_all[:-1]
            y = x_all[-1]
        else:
            x = x_all
            y = None

        batch_size = x[0].shape[0]
        device = x[0].device

        # Process each view with learned embeddings
        view_embeddings = []

        for i, (view_name, view_data) in enumerate(zip(self.views_sizes.keys(), x)):
            # Concatenate with conditionals if needed
            if y is not None:
                view_input = torch.cat([view_data, y], dim=1)
            else:
                view_input = view_data

            # Create learned token embeddings
            embedded = self._create_view_tokens(view_input, view_name)

            # Add view type embedding
            view_type_emb = self.view_type_embeddings(torch.tensor(i, device=device))
            embedded = embedded + view_type_emb.unsqueeze(0).unsqueeze(0)

            # Add positional encoding
            embedded = embedded.transpose(0, 1)
            embedded = self.pos_encoder(embedded)
            embedded = embedded.transpose(0, 1)

            view_embeddings.append(embedded)

        # Concatenate all views
        all_embeddings = torch.cat(
            view_embeddings, dim=1
        )  # [batch, total_tokens, d_model]

        # Apply cross-modal transformer (no masking needed since we use learned embeddings)
        encoded = self.transformer_encoder(all_embeddings)

        # Global pooling with multiple learned queries
        # Expand queries for batch
        queries = self.global_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_queries, d_model]

        # Apply attention with multiple queries
        global_representations, _ = self.global_attention(
            queries, encoded, encoded
        )  # [batch, num_queries, d_model]

        # Aggregate multiple query outputs
        global_repr = global_representations.reshape(batch_size, -1)
        global_repr = self.query_aggregator(global_repr)  # [batch, d_model]

        # Get latent representation
        mu = self.mu_head(global_repr)
        log_var = self.logvar_head(global_repr)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # Decode each view using the same structure as MOSA
        x_hat = []
        for i, view_name in enumerate(self.views_sizes.keys()):
            # Prepare decoder input
            if self.hypers["use_conditionals"]:
                decoder_input = torch.cat([z, y], dim=1)
            else:
                decoder_input = z

            # Apply decoder (now using ModuleList like MOSA)
            reconstructed = self.decoders[i](decoder_input)
            x_hat.append(reconstructed)

        if self.return_for_shap == "latent":
            return mu
        elif self.return_for_shap is not None:
            return x_hat[self.layer_idx_map[self.return_for_shap]]
        else:
            return dict(
                x_hat=x_hat,
                z=z,
                mu=mu,
                log_var=log_var,
            )

    def loss(self, x, x_nans, out_net, y, x_mask, view_loss_weights=None):
        """Use the same loss function as MOSA"""
        # This can reuse the exact same loss function from MOSA
        # since the output format is identical
        view_loss_weights = view_loss_weights if view_loss_weights else [1] * len(x)

        # Reconstruction loss
        x_hat, mu, logvar = out_net["x_hat"], out_net["mu"], out_net["log_var"]

        recon_loss, recon_loss_views = 0, []
        for i in range(len(x)):
            mask = x_nans[i].int()
            x_hat_i, x_i = (x_hat[i] * mask), (x[i] * mask)

            if self.hypers["view_loss_recon_type"][i].lower() == "macro":
                recon_xi = torch.stack(
                    [
                        self.recon_criterion(
                            x_hat_i[x_i == v],
                            x_i[x_i == v],
                            reduction="sum",
                        )
                        / c
                        for v, c in zip(*torch.unique(x_i, return_counts=True))
                    ]
                ).mean()

            elif self.hypers["view_loss_recon_type"][i].lower() == "autofocus":
                gamma = 2
                mask_sum = mask.sum()

                if mask_sum == 0:
                    recon_xi = 0
                else:
                    recon_xi = torch.sum((x_hat_i - x_i) ** (2 + gamma))
                    recon_xi /= mask_sum
            else:
                mask_sum = mask.sum()

                if mask_sum == 0:
                    recon_xi = 0
                else:
                    recon_xi = self.recon_criterion(
                        x_hat_i,
                        x_i,
                        reduction="sum",
                    )
                    recon_xi /= mask_sum

            recon_xi *= view_loss_weights[i]
            recon_loss_views.append(recon_xi)
            recon_loss += recon_xi

        recon_loss *= self.hypers["w_rec"]

        # KL divergence loss
        kl_loss = CLinesLosses.kl_divergence(mu, logvar) * self.hypers["w_kl"]

        # Contrastive loss
        if len(self.hypers["labels"]) == 0:
            c_loss = 0
        else:
            loss_func = losses.ContrastiveLoss(
                distance=CosineSimilarity(),
                pos_margin=self.hypers["contrastive_pos_margin"],
                neg_margin=self.hypers["contrastive_neg_margin"],
            )
            c_loss = [loss_func(mu, y[:, i]) for i in range(32)]
            c_loss = torch.stack(c_loss).sum() * self.hypers["w_contrastive"]

        # Optional DIP-VAE loss
        if self.hypers.get("dip_vae_type") != None:
            dipvae_loss = CLinesLosses.dip_vae(
                mu,
                logvar,
                self.hypers["lambda_od"],
                self.hypers["lambda_d"],
                self.hypers["dip_vae_type"],
            )

            loss = recon_loss + kl_loss + c_loss + dipvae_loss

            return dict(
                total=loss,
                reconstruction=recon_loss,
                reconstruction_views=recon_loss_views,
                kl=kl_loss,
                contrastive=c_loss,
                dipvae=dipvae_loss,
            )
        else:
            loss = recon_loss + kl_loss + c_loss

            return dict(
                total=loss,
                reconstruction=recon_loss,
                reconstruction_views=recon_loss_views,
                kl=kl_loss,
                contrastive=c_loss,
            )


class DiffusionScheduler:
    def __init__(
        self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"
    ):
        """
        Initialize the diffusion scheduler with noise schedule parameters.

        Args:
            num_timesteps: Number of diffusion steps
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            device: Device to place the schedule tensors on
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # Create noise schedule and move to specified device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Pre-compute values used in diffusion process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)

        # Calculate posterior variance
        alphas_cumprod_prev = torch.cat(
            [self.alphas_cumprod[0:1], self.alphas_cumprod[:-1]]
        )
        self.posterior_variance = (
            self.betas * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - the forward diffusion process.

        Args:
            x_0: Initial data [batch_size, latent_dim]
            t: Timesteps tensor [batch_size]
            noise: Optional pre-generated noise [batch_size, latent_dim]

        Returns:
            Noised version of the data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        t = t.to(self.device)

        # Get appropriate alphas and reshape for broadcasting
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(
            -1, 1
        )  # [batch_size, 1]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1
        )  # [batch_size, 1]

        # Now the broadcasting will work correctly across the latent dimension
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean_variance(self, model_output, x_t, t):
        """
        Calculate mean and variance for the reverse diffusion process.

        Args:
            model_output: Predicted noise by the model
            x_t: Input at timestep t
            t: Current timestep tensor (should be on the same device as x_t)

        Returns:
            Dictionary containing mean and variance
        """
        # Ensure t is on the correct device
        t = t.to(self.device)

        # Get appropriate values for timesteps
        posterior_variance = self.posterior_variance[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]

        # Calculate predicted x_0
        pred_x0 = sqrt_recip_alphas_t * (
            x_t - model_output * self.sqrt_one_minus_alphas_cumprod[t]
        )

        # Calculate mean for reverse process
        model_mean = pred_x0

        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": torch.log(posterior_variance),
        }


class DiffusionMOSA(nn.Module):
    def __init__(self, base_mosa, diffusion_scheduler):
        super().__init__()
        self.base_mosa = base_mosa
        self.scheduler = diffusion_scheduler

        # Get latent dimension from base MOSA
        latent_dim = self.base_mosa.hypers["latent_dim"]

        # Changed: Make timestep embedding dimension match latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Changed: Noise predictor now takes concatenated input correctly
        self.noise_predictor = nn.Sequential(
            nn.Linear(
                latent_dim * 2, latent_dim * 2
            ),  # *2 because we concatenate z and t_emb
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def get_timestep_embedding(self, timesteps):
        """
        Create timestep embeddings.

        Args:
            timesteps: [batch_size] tensor of timesteps

        Returns:
            [batch_size, latent_dim] tensor of timestep embeddings
        """
        # Reshape timesteps to [batch_size, 1]
        t_emb = timesteps.unsqueeze(-1).float()

        # Pass through embedding network
        return self.time_embed(t_emb)

    def forward(self, x_all, noise_pred_only=False):
        # Get MOSA outputs
        mosa_out = self.base_mosa(x_all)

        if noise_pred_only:
            return mosa_out

        # Get latent representation
        z = mosa_out["z"]  # [batch_size, latent_dim]

        # Sample timesteps
        batch_size = z.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,), device=z.device
        )

        # Add noise to latent representation
        noise = torch.randn_like(z)
        z_noisy = self.scheduler.q_sample(z, t, noise)  # [batch_size, latent_dim]

        # Get timestep embeddings - now returns [batch_size, latent_dim]
        t_emb = self.get_timestep_embedding(t)  # [batch_size, latent_dim]

        # Concatenate noisy latent and time embedding
        model_input = torch.cat([z_noisy, t_emb], dim=1)  # [batch_size, latent_dim*2]

        # Predict noise
        noise_pred = self.noise_predictor(model_input)

        mosa_out.update(
            {"noise": noise, "noise_pred": noise_pred, "z_noisy": z_noisy, "t": t}
        )

        return mosa_out

    def sample(self, num_samples, device="cuda"):
        """
        Generate samples using the reverse diffusion process.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated samples
        """
        # Start from random noise
        z = torch.randn(num_samples, self.base_mosa.hypers["latent_dim"]).to(device)

        # Iteratively denoise
        for t in reversed(range(self.scheduler.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=device)
            t_emb = self.get_timestep_embedding(t_batch)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.noise_predictor(torch.cat([z, t_emb], dim=-1))

            # Get mean and variance for reverse process
            out = self.scheduler.p_mean_variance(noise_pred, z, t_batch)

            # Sample
            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = 0

            z = out["mean"] + torch.exp(0.5 * out["log_variance"]) * noise

        return z

    # Modify the existing loss function to incorporate diffusion loss
    def combined_loss(self, model_output, original_loss_output, diffusion_weight=1.0):
        """
        Combine original MOSA loss with diffusion loss.

        Args:
            model_output: Output from DiffusionMOSA
            original_loss_output: Output from original MOSA loss function
            diffusion_weight: Weight for diffusion loss component

        Returns:
            Combined loss
        """
        diff_loss = diffusion_weight * F.mse_loss(
            model_output["noise_pred"], model_output["noise"]
        )
        total_loss = original_loss_output["total"] + diff_loss

        loss_dict = original_loss_output.copy()
        loss_dict.update({"total": total_loss, "diffusion": diff_loss})

        return loss_dict


class TransformerDiffusionMOSA(nn.Module):
    """
    Enhanced model combining TransformerMOSA with diffusion in latent space.
    This leverages the improved cross-modal learning of transformers with
    the generative benefits of diffusion models.
    """

    def __init__(self, base_transformer, diffusion_scheduler):
        super().__init__()
        self.base_transformer = base_transformer
        self.scheduler = diffusion_scheduler

        # Get latent dimension from base TransformerMOSA
        latent_dim = self.base_transformer.hypers["latent_dim"]

        # Timestep embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Enhanced noise predictor with more capacity for transformer features
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def get_timestep_embedding(self, timesteps):
        """
        Create timestep embeddings.

        Args:
            timesteps: [batch_size] tensor of timesteps

        Returns:
            [batch_size, latent_dim] tensor of timestep embeddings
        """
        t_emb = timesteps.unsqueeze(-1).float()
        return self.time_embed(t_emb)

    def forward(self, x_all, noise_pred_only=False):
        # Get TransformerMOSA outputs (better cross-modal representations)
        transformer_out = self.base_transformer(x_all)

        if noise_pred_only:
            return transformer_out

        # Get latent representation from transformer
        z = transformer_out["z"]  # [batch_size, latent_dim]

        # Sample timesteps
        batch_size = z.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,), device=z.device
        )

        # Add noise to latent representation
        noise = torch.randn_like(z)
        z_noisy = self.scheduler.q_sample(z, t, noise)

        # Get timestep embeddings
        t_emb = self.get_timestep_embedding(t)

        # Concatenate noisy latent and time embedding
        model_input = torch.cat([z_noisy, t_emb], dim=1)

        # Predict noise
        noise_pred = self.noise_predictor(model_input)

        transformer_out.update(
            {"noise": noise, "noise_pred": noise_pred, "z_noisy": z_noisy, "t": t}
        )

        return transformer_out

    def sample(self, num_samples, device="cuda"):
        """
        Generate samples using the reverse diffusion process.
        """
        # Start from random noise
        z = torch.randn(num_samples, self.base_transformer.hypers["latent_dim"]).to(
            device
        )

        # Iteratively denoise
        for t in reversed(range(self.scheduler.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=device)
            t_emb = self.get_timestep_embedding(t_batch)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.noise_predictor(torch.cat([z, t_emb], dim=-1))

            # Get mean and variance for reverse process
            out = self.scheduler.p_mean_variance(noise_pred, z, t_batch)

            # Sample
            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = 0

            z = out["mean"] + torch.exp(0.5 * out["log_variance"]) * noise

        return z

    def combined_loss(self, model_output, original_loss_output, diffusion_weight=1.0):
        """
        Combine TransformerMOSA loss with diffusion loss.
        """
        diff_loss = diffusion_weight * F.mse_loss(
            model_output["noise_pred"], model_output["noise"]
        )
        total_loss = original_loss_output["total"] + diff_loss

        loss_dict = original_loss_output.copy()
        loss_dict.update({"total": total_loss, "diffusion": diff_loss})

        return loss_dict
