import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

class FlattenMLP(nn.Module):
    def __init__(self, d, N, hidden=128):
        super().__init__()
        in_dim = N * d
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ELU(inplace=True),
            nn.Linear(hidden, in_dim)
        )
        self.d = d
        self.N = N

    def forward(self, x):
        # x: B x N x d x d
        B = x.size(0)
        # Flatten
        x = x.view(B, -1) # B x N*d
        
        out = self.net(x) # B x N*d -> B x N*d
        return out.view(B, self.N, self.d) # Output: B x N x d

class SoftplusParameterization(nn.Module):
    # Make weights positive
    def forward(self, X):
        return F.softplus(X)

class PESymmetry(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, pool='sum', use_tanh=False, pos=False):
        super().__init__()
        assert pool in ('sum', 'mean', 'max')

        self.pool = pool
        self.use_tanh = use_tanh
        # Per item transform
        self.individual = nn.Linear(in_dim, out_dim)
        # Pool transform
        self.pooling = nn.Linear(in_dim, out_dim, bias=False)

        if pos:
            P.register_parametrization(self.individual, "weight", SoftplusParameterization())
            P.register_parametrization(self.pooling, "weight", SoftplusParameterization())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x N x obs_dim
        
        # --------- Pooling --------- #
        # B x N x obs_dim  -> B x 1 x obs_dim
        pooled = {
            'sum': x.sum(dim=-2),
            'mean': x.mean(dim=-2),
            'max': x.max(dim=-2).values
        }[self.pool]

        x_mean = self.pooling(pooled) # B x obs_dim
        x_mean = x_mean.unsqueeze(1) # B x 1 x obs_dim

        # Use tanh if called for
        if self.use_tanh:
            x_mean = F.tanh(x_mean) # B x 1 x obs_dim

        # --------- Individual --------- #
        x = self.individual(x) # B x N x obs_dim

        # --------- Output --------- #
        output = x + x_mean # B x N x obs_dim

        return output

class DeepSets(nn.Module):
    def __init__(self, d, hidden=64, pool='sum', use_tanh=False, pe_layers=2, pos=False):
        super().__init__()
        assert pool in ('sum', 'mean', 'max')
        assert pe_layers >= 1

        self.d = d
        # Matrix flattened
        input_dim = d

        layers = []

        if pe_layers == 1:
            layers.append(PESymmetry(input_dim, input_dim, pool, use_tanh, pos))

        else:
            # First layer
            layers.append(PESymmetry(input_dim, hidden, pool, use_tanh, pos))
            layers.append(nn.ELU(inplace=True))
            # Intermediate layers
            for i in range(pe_layers-2):
                layers.append(PESymmetry(hidden, hidden, pool, use_tanh, pos))
                layers.append(nn.ELU(inplace=True))
            # Last layer
            layers.append(PESymmetry(hidden, input_dim, pool, use_tanh, pos))

        self.pe_net = nn.Sequential(*layers)

    def forward(self, x):
        # B x N x d  ->  B x N x d
        y_pred = self.pe_net(x)

        return y_pred  # B x N x d