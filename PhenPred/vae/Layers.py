import torch
import torch.nn as nn
from torch.nn import functional as F


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


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, var):
        if self.training:
            std = torch.sqrt(var + 1e-10)
            noise = torch.randn_like(std)
            return mu + noise * std
        else:
            return mu

    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


class GumbelSoftmax(nn.Module):
    """
    Sourced from: https://github.com/jariasf/GMVAE/tree/master/pytorch
    """

    def __init__(self, f_dim, c_dim):
        super().__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        # categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


class JointInference(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, hidden_size=128):
        super().__init__()

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList(
            [
                nn.Linear(x_dim, hidden_size),
                nn.ReLU(),
                GumbelSoftmax(hidden_size, y_dim),
            ]
        )

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList(
            [
                nn.Linear(x_dim + y_dim, hidden_size),
                nn.ReLU(),
                Gaussian(hidden_size, z_dim),
            ]
        )

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, temperature=1.0, hard=0):
        # q(y|x)
        logits, prob, y = self.qyx(x, temperature, hard)

        # q(z|x,y)
        mu, var, z = self.qzxy(x, y)

        return mu, var, z, logits, prob, y
